"""Differentiable NVIDIA Warp boundary-condition engine for incompressible and Boussinesq FVM solvers (``picoext``, ``picocmpt``, ``buoyantBoussinesqpicoext``, ``buoyantBoussinesqpicocmpt``)."""
from typing import Any, Optional, Tuple

import torch
import warp as wp

from FVdomain.SetBC.onthefly.Parafunction import Functionbase

wp.config.quiet = True


def _wp_from_torch(t: torch.Tensor, dtype=None):
    return wp.from_torch(t, dtype=dtype, requires_grad=False)

U_FIXED = 0
U_NOSLIP = 1
U_ZEROGRAD = 2
U_CYCLIC = 3
U_SYMMETRY = 4
U_SLIP = 5
U_RACK_PRESERVE = 6
U_INLETOUTLET = 7
U_ADVECTIVE = 8
U_FLOW_OUTLET = 9
U_CALCULATED = 10

P_FIXED = 0
P_ZEROGRAD = 2
P_CYCLIC = 3
P_FIXEDFLUX = 4
P_CALCULATED = 10

T_FIXED = 0
T_ZEROGRAD = 2
T_CYCLIC = 3
T_IBBM = 4
T_CBBM = 5
T_CALCULATED = 10

G_NONE = 0
G_ZERO = 1
G_SYMMETRY_U = 2
G_RACK_INLET_U = 3
G_INLETOUTLET_U = 4
G_FIXEDFLUX_P = 5


def _make_bc_kernels(scalar_t, vec3_t):
    @wp.kernel
    def copy_state_wzero_kernel(
        phi_old_in: wp.array3d(dtype=scalar_t),
        phi_new_in: wp.array3d(dtype=scalar_t),
        wzero: int,
        n_ch: int,
        phi_old_out: wp.array3d(dtype=scalar_t),
        phi_new_out: wp.array3d(dtype=scalar_t),
    ):
        t, cpd = wp.tid()
        for c in range(n_ch):
            if wzero == 1 and c == 2:
                phi_old_out[t, cpd, c] = scalar_t(0.0)
                phi_new_out[t, cpd, c] = scalar_t(0.0)
            else:
                phi_old_out[t, cpd, c] = phi_old_in[t, cpd, c]
                phi_new_out[t, cpd, c] = phi_new_in[t, cpd, c]

    @wp.kernel
    def flowout_reduce_kernel(
        phi_new: wp.array3d(dtype=scalar_t),
        fo_off: wp.array(dtype=wp.int32),
        fo_owner: wp.array(dtype=wp.int32),
        fo_nrm: wp.array(dtype=vec3_t),
        fo_rho_area: wp.array(dtype=wp.float64),
        est_flow: wp.array2d(dtype=wp.float64),
    ):
        t, p = wp.tid()
        acc = wp.float64(0.0)
        for k in range(fo_off[p], fo_off[p + 1]):
            oi = fo_owner[k]
            n = fo_nrm[k]
            un = (
                phi_new[t, oi, 0] * n[0]
                + phi_new[t, oi, 1] * n[1]
                + phi_new[t, oi, 2] * n[2]
            )
            if un > scalar_t(0.0):
                acc += fo_rho_area[k] * wp.float64(un)
        est_flow[t, p] = acc

    @wp.kernel
    def cbbm_reduce_kernel(
        phi_new: wp.array3d(dtype=scalar_t),
        cb_off: wp.array(dtype=wp.int32),
        cb_owner: wp.array(dtype=wp.int32),
        cb_area: wp.array(dtype=wp.float64),
        cb_area_sum: wp.array(dtype=wp.float64),
        cbbm_t_avg: wp.array2d(dtype=wp.float64),
    ):
        t, p = wp.tid()
        acc = wp.float64(0.0)
        for k in range(cb_off[p], cb_off[p + 1]):
            oi = cb_owner[k]
            acc += wp.float64(phi_new[t, oi, 4]) * cb_area[k]
        cbbm_t_avg[t, p] = acc / cb_area_sum[p]

    @wp.kernel
    def ghost_fill_kernel(
        phi_old_in: wp.array3d(dtype=scalar_t),
        phi_new_in: wp.array3d(dtype=scalar_t),
        ghost_cpd: wp.array(dtype=wp.int32),
        owner_cpd: wp.array(dtype=wp.int32),
        nrm: wp.array(dtype=vec3_t),
        u_code: wp.array(dtype=wp.int32),
        u_partner: wp.array(dtype=wp.int32),
        u_val: wp.array(dtype=vec3_t),
        u_rack_old: wp.array(dtype=wp.int32),
        u_patch_slot: wp.array(dtype=wp.int32),
        u_flow_tgt: wp.array(dtype=wp.float64),
        u_area_sum: wp.array(dtype=wp.float64),
        adv_dt_dx: wp.array(dtype=scalar_t),
        adv_floor: wp.array(dtype=scalar_t),
        est_flow: wp.array2d(dtype=wp.float64),
        p_code: wp.array(dtype=wp.int32),
        p_partner: wp.array(dtype=wp.int32),
        p_val: wp.array(dtype=scalar_t),
        ghb_ghost: wp.array(dtype=scalar_t),
        t_code: wp.array(dtype=wp.int32),
        t_partner: wp.array(dtype=wp.int32),
        t_val: wp.array(dtype=scalar_t),
        t_patch_slot: wp.array(dtype=wp.int32),
        cbbm_t_avg: wp.array2d(dtype=wp.float64),
        energy: int,
        include_old_u: int,
        wzero: int,
        phi_old_out: wp.array3d(dtype=scalar_t),
        phi_new_out: wp.array3d(dtype=scalar_t),
    ):
        t, g = wp.tid()
        gi = ghost_cpd[g]
        oi = owner_cpd[g]
        n = nrm[g]

        uc = u_code[g]
        uo0 = phi_new_in[t, oi, 0]
        uo1 = phi_new_in[t, oi, 1]
        uo2 = scalar_t(0.0) if wzero == 1 else phi_new_in[t, oi, 2]

        u_old_o0 = phi_old_in[t, oi, 0]
        u_old_o1 = phi_old_in[t, oi, 1]
        u_old_o2 = scalar_t(0.0) if wzero == 1 else phi_old_in[t, oi, 2]

        ug0 = phi_new_in[t, gi, 0]
        ug1 = phi_new_in[t, gi, 1]
        ug2 = scalar_t(0.0) if wzero == 1 else phi_new_in[t, gi, 2]

        ug_old0 = phi_old_in[t, gi, 0]
        ug_old1 = phi_old_in[t, gi, 1]
        ug_old2 = scalar_t(0.0) if wzero == 1 else phi_old_in[t, gi, 2]

        uv = u_val[g]
        if uc == U_FIXED:
            ug0 = uv[0]
            ug1 = uv[1]
            ug2 = uv[2]
        elif uc == U_NOSLIP:
            ug0 = scalar_t(0.0)
            ug1 = scalar_t(0.0)
            ug2 = scalar_t(0.0)
        elif uc == U_ZEROGRAD:
            ug0 = uo0
            ug1 = uo1
            ug2 = uo2
        elif uc == U_CYCLIC:
            pi = u_partner[g]
            ug0 = phi_new_in[t, pi, 0]
            ug1 = phi_new_in[t, pi, 1]
            ug2 = scalar_t(0.0) if wzero == 1 else phi_new_in[t, pi, 2]
        elif uc == U_SYMMETRY:
            un = uo0 * n[0] + uo1 * n[1] + uo2 * n[2]
            ug0 = uo0 - un * n[0]
            ug1 = uo1 - un * n[1]
            ug2 = uo2 - un * n[2]
        elif uc == U_SLIP:
            two_un = scalar_t(2.0) * (uo0 * n[0] + uo1 * n[1] + uo2 * n[2])
            ug0 = uo0 - two_un * n[0]
            ug1 = uo1 - two_un * n[1]
            ug2 = uo2 - two_un * n[2]
        elif uc == U_RACK_PRESERVE:
            un = uo0 * n[0] + uo1 * n[1] + uo2 * n[2]
            ug0 = (uo0 - un * n[0]) + uv[0]
            ug1 = (uo1 - un * n[1]) + uv[1]
            ug2 = (uo2 - un * n[2]) + uv[2]
        elif uc == U_INLETOUTLET:
            un = uo0 * n[0] + uo1 * n[1] + uo2 * n[2]
            if un >= scalar_t(0.0):
                ug0 = uo0
                ug1 = uo1
                ug2 = uo2
            else:
                ug0 = uv[0]
                ug1 = uv[1]
                ug2 = uv[2]
        elif uc == U_ADVECTIVE:
            if include_old_u == 1:
                un_old = u_old_o0 * n[0] + u_old_o1 * n[1] + u_old_o2 * n[2]
                un_pos = wp.max(un_old, scalar_t(0.0))
                alpha = wp.clamp(un_pos * adv_dt_dx[g], adv_floor[g], scalar_t(1.0))
                ug0 = ug_old0 - alpha * (ug_old0 - u_old_o0)
                ug1 = ug_old1 - alpha * (ug_old1 - u_old_o1)
                ug2 = ug_old2 - alpha * (ug_old2 - u_old_o2)
            else:
                ug0 = uo0
                ug1 = uo1
                ug2 = uo2
        elif uc == U_FLOW_OUTLET:
            un = uo0 * n[0] + uo1 * n[1] + uo2 * n[2]
            ut0 = uo0 - un * n[0]
            ut1 = uo1 - un * n[1]
            ut2 = uo2 - un * n[2]
            nup = wp.max(un, scalar_t(0.0))
            pslot = u_patch_slot[g]
            est = est_flow[t, pslot]
            tgt = u_flow_tgt[g]
            asum = u_area_sum[g]
            nup_c = scalar_t(0.0)
            if est > wp.float64(0.5) * tgt:
                sc = scalar_t(wp.abs(tgt / (est + wp.float64(1.0e-12))))
                nup_c = nup * sc
            else:
                adj = scalar_t((tgt - est) / (asum + wp.float64(1.0e-12)))
                nup_c = nup + adj
            ug0 = ut0 + nup_c * n[0]
            ug1 = ut1 + nup_c * n[1]
            ug2 = ut2 + nup_c * n[2]

        if wzero == 1:
            ug2 = scalar_t(0.0)

        phi_new_out[t, gi, 0] = ug0
        phi_new_out[t, gi, 1] = ug1
        phi_new_out[t, gi, 2] = ug2

        tg_new = scalar_t(0.0)
        if energy == 1:
            tc = t_code[g]
            to_new = phi_new_in[t, oi, 4]
            tg_new = phi_new_in[t, gi, 4]
            if tc == T_FIXED:
                tg_new = t_val[g]
            elif tc == T_ZEROGRAD:
                tg_new = to_new
            elif tc == T_CYCLIC:
                pi = t_partner[g]
                tg_new = phi_new_in[t, pi, 4]
            elif tc == T_IBBM:
                si = t_partner[g]
                tg_new = phi_new_in[t, si, 4] + t_val[g]
            elif tc == T_CBBM:
                pslot = t_patch_slot[g]
                tg_new = scalar_t(cbbm_t_avg[t, pslot]) + t_val[g]
            phi_new_out[t, gi, 4] = tg_new

        pc = p_code[g]
        po = phi_new_in[t, oi, 3]
        pg = phi_new_in[t, gi, 3]
        if pc == P_FIXED:
            pg = p_val[g]
        elif pc == P_ZEROGRAD:
            pg = po
        elif pc == P_CYCLIC:
            pi = p_partner[g]
            pg = phi_new_in[t, pi, 3]
        elif pc == P_FIXEDFLUX:
            pg = po
            if energy == 1:
                pg = po + ghb_ghost[g] * (tg_new - phi_new_in[t, oi, 4])
        phi_new_out[t, gi, 3] = pg

    @wp.kernel
    def flowout_bwd_reduce_kernel(
        phi_new_in: wp.array3d(dtype=scalar_t),
        adj_new_out: wp.array3d(dtype=scalar_t),
        fo_off: wp.array(dtype=wp.int32),
        fo_ghost: wp.array(dtype=wp.int32),
        fo_owner: wp.array(dtype=wp.int32),
        fo_nrm: wp.array(dtype=vec3_t),
        fo_tgt: wp.array(dtype=wp.float64),
        fo_asum: wp.array(dtype=wp.float64),
        est_flow: wp.array2d(dtype=wp.float64),
        adj_est_flow: wp.array2d(dtype=wp.float64),
    ):
        t, p = wp.tid()
        est = est_flow[t, p]
        tgt = fo_tgt[p]
        asum = fo_asum[p]
        denom = est + wp.float64(1.0e-12)
        s_raw = tgt / denom
        sgn = wp.float64(1.0) if s_raw >= wp.float64(0.0) else wp.float64(-1.0)
        dscale_dest = -sgn * tgt / (denom * denom)
        dadj_dest = wp.float64(-1.0) / (asum + wp.float64(1.0e-12))
        use_scale = est > wp.float64(0.5) * tgt

        acc = wp.float64(0.0)
        for k in range(fo_off[p], fo_off[p + 1]):
            gi = fo_ghost[k]
            oi = fo_owner[k]
            n = fo_nrm[k]
            eta = wp.float64(
                adj_new_out[t, gi, 0] * n[0]
                + adj_new_out[t, gi, 1] * n[1]
                + adj_new_out[t, gi, 2] * n[2]
            )
            if use_scale:
                un = (
                    phi_new_in[t, oi, 0] * n[0]
                    + phi_new_in[t, oi, 1] * n[1]
                    + phi_new_in[t, oi, 2] * n[2]
                )
                nup = wp.float64(wp.max(un, scalar_t(0.0)))
                acc += eta * nup * dscale_dest
            else:
                acc += eta * dadj_dest
        adj_est_flow[t, p] = acc

    @wp.kernel
    def cbbm_bwd_out_reduce_kernel(
        adj_new_out: wp.array3d(dtype=scalar_t),
        cb_out_off: wp.array(dtype=wp.int32),
        cb_out_ghost: wp.array(dtype=wp.int32),
        cb_out_ghb: wp.array(dtype=scalar_t),
        cb_out_has_ffp: wp.array(dtype=wp.int32),
        cb_lag_mean: wp.array(dtype=wp.int32),
        adj_cbbm_avg: wp.array2d(dtype=wp.float64),
    ):
        t, p = wp.tid()
        if cb_lag_mean[p] == 1:
            adj_cbbm_avg[t, p] = wp.float64(0.0)
            return
        acc = wp.float64(0.0)
        for k in range(cb_out_off[p], cb_out_off[p + 1]):
            gi = cb_out_ghost[k]
            at = adj_new_out[t, gi, 4]
            if cb_out_has_ffp[k] == 1:
                at = at + cb_out_ghb[k] * adj_new_out[t, gi, 3]
            acc += wp.float64(at)
        adj_cbbm_avg[t, p] = acc

    @wp.kernel
    def ghost_bwd_emit_kernel(
        phi_old_in: wp.array3d(dtype=scalar_t),
        phi_new_in: wp.array3d(dtype=scalar_t),
        adj_old_out: wp.array3d(dtype=scalar_t),
        adj_new_out: wp.array3d(dtype=scalar_t),
        ghost_cpd: wp.array(dtype=wp.int32),
        owner_cpd: wp.array(dtype=wp.int32),
        nrm: wp.array(dtype=vec3_t),
        u_code: wp.array(dtype=wp.int32),
        u_rack_old: wp.array(dtype=wp.int32),
        u_patch_slot: wp.array(dtype=wp.int32),
        u_flow_tgt: wp.array(dtype=wp.float64),
        u_rho_area: wp.array(dtype=wp.float64),
        est_flow: wp.array2d(dtype=wp.float64),
        adj_est_flow: wp.array2d(dtype=wp.float64),
        p_code: wp.array(dtype=wp.int32),
        ghb_ghost: wp.array(dtype=scalar_t),
        t_code: wp.array(dtype=wp.int32),
        energy: int,
        include_old_u: int,
        wzero: int,
        emit_owner_new: wp.array3d(dtype=scalar_t),
        emit_owner_old: wp.array3d(dtype=scalar_t),
        emit_u_part_new: wp.array3d(dtype=scalar_t),
        emit_u_part_old: wp.array3d(dtype=scalar_t),
        emit_p_part_new: wp.array2d(dtype=scalar_t),
        emit_t_part_new: wp.array2d(dtype=scalar_t),
        emit_t_part_old: wp.array2d(dtype=scalar_t),
        adj_old_in: wp.array3d(dtype=scalar_t),
        adj_new_in: wp.array3d(dtype=scalar_t),
    ):
        t, g = wp.tid()
        gi = ghost_cpd[g]
        oi = owner_cpd[g]
        n = nrm[g]

        on0 = scalar_t(0.0)
        on1 = scalar_t(0.0)
        on2 = scalar_t(0.0)
        on3 = scalar_t(0.0)
        on4 = scalar_t(0.0)

        oo0 = scalar_t(0.0)
        oo1 = scalar_t(0.0)
        oo2 = scalar_t(0.0)
        oo4 = scalar_t(0.0)

        upn0 = scalar_t(0.0)
        upn1 = scalar_t(0.0)
        upn2 = scalar_t(0.0)
        upo0 = scalar_t(0.0)
        upo1 = scalar_t(0.0)
        upo2 = scalar_t(0.0)

        ppn = scalar_t(0.0)
        tpn = scalar_t(0.0)
        tpo = scalar_t(0.0)

        pc = p_code[g]
        ap = adj_new_out[t, gi, 3]
        adj_old_in[t, gi, 3] = adj_old_out[t, gi, 3]
        adj_t_from_ffp = scalar_t(0.0)
        if pc == P_CALCULATED:
            adj_new_in[t, gi, 3] = ap
        else:
            adj_new_in[t, gi, 3] = scalar_t(0.0)
            if pc == P_ZEROGRAD:
                on3 += ap
            elif pc == P_CYCLIC:
                ppn = ap
            elif pc == P_FIXEDFLUX:
                on3 += ap
                if energy == 1:
                    ghb = ghb_ghost[g]
                    on4 -= ghb * ap
                    adj_t_from_ffp = ghb * ap

        if energy == 1:
            tc = t_code[g]
            at_new = adj_new_out[t, gi, 4] + adj_t_from_ffp
            adj_old_in[t, gi, 4] = adj_old_out[t, gi, 4]
            if tc == T_CALCULATED:
                adj_new_in[t, gi, 4] = at_new
            elif tc == T_FIXED or tc == T_CBBM:
                adj_new_in[t, gi, 4] = scalar_t(0.0)
            elif tc == T_IBBM:
                adj_new_in[t, gi, 4] = scalar_t(0.0)
                tpn = at_new
            elif tc == T_ZEROGRAD:
                adj_new_in[t, gi, 4] = scalar_t(0.0)
                on4 += at_new
            elif tc == T_CYCLIC:
                adj_new_in[t, gi, 4] = scalar_t(0.0)
                tpn = at_new

        uc = u_code[g]
        au0 = adj_new_out[t, gi, 0]
        au1 = adj_new_out[t, gi, 1]
        au2 = scalar_t(0.0) if wzero == 1 else adj_new_out[t, gi, 2]
        adj_old_in[t, gi, 0] = adj_old_out[t, gi, 0]
        adj_old_in[t, gi, 1] = adj_old_out[t, gi, 1]
        adj_old_in[t, gi, 2] = scalar_t(0.0) if wzero == 1 else adj_old_out[t, gi, 2]

        if uc == U_CALCULATED:
            adj_new_in[t, gi, 0] = au0
            adj_new_in[t, gi, 1] = au1
            adj_new_in[t, gi, 2] = au2
        elif uc == U_FIXED or uc == U_NOSLIP:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
        elif uc == U_ZEROGRAD:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            on0 += au0
            on1 += au1
            on2 += au2
        elif uc == U_CYCLIC:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            upn0 = au0
            upn1 = au1
            upn2 = au2
        elif uc == U_SYMMETRY:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            an = au0 * n[0] + au1 * n[1] + au2 * n[2]
            on0 += au0 - an * n[0]
            on1 += au1 - an * n[1]
            on2 += au2 - an * n[2]
        elif uc == U_SLIP:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            two_an = scalar_t(2.0) * (au0 * n[0] + au1 * n[1] + au2 * n[2])
            on0 += au0 - two_an * n[0]
            on1 += au1 - two_an * n[1]
            on2 += au2 - two_an * n[2]
        elif uc == U_RACK_PRESERVE:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            an = au0 * n[0] + au1 * n[1] + au2 * n[2]
            on0 += au0 - an * n[0]
            on1 += au1 - an * n[1]
            on2 += au2 - an * n[2]
        elif uc == U_INLETOUTLET:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            uo0 = phi_new_in[t, oi, 0]
            uo1 = phi_new_in[t, oi, 1]
            uo2 = scalar_t(0.0) if wzero == 1 else phi_new_in[t, oi, 2]
            if uo0 * n[0] + uo1 * n[1] + uo2 * n[2] >= scalar_t(0.0):
                on0 += au0
                on1 += au1
                on2 += au2
        elif uc == U_ADVECTIVE:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            if include_old_u == 0:
                on0 += au0
                on1 += au1
                on2 += au2
        elif uc == U_FLOW_OUTLET:
            adj_new_in[t, gi, 0] = scalar_t(0.0)
            adj_new_in[t, gi, 1] = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)

            eta = au0 * n[0] + au1 * n[1] + au2 * n[2]
            at0 = au0 - eta * n[0]
            at1 = au1 - eta * n[1]
            at2 = au2 - eta * n[2]

            pslot = u_patch_slot[g]
            est = est_flow[t, pslot]
            tgt = u_flow_tgt[g]
            dnup = eta
            if est > wp.float64(0.5) * tgt:
                sc = scalar_t(wp.abs(tgt / (est + wp.float64(1.0e-12))))
                dnup = eta * sc
            dnup = dnup + scalar_t(adj_est_flow[t, pslot] * u_rho_area[g])

            uo0 = phi_new_in[t, oi, 0]
            uo1 = phi_new_in[t, oi, 1]
            uo2 = scalar_t(0.0) if wzero == 1 else phi_new_in[t, oi, 2]
            un = uo0 * n[0] + uo1 * n[1] + uo2 * n[2]
            if un >= scalar_t(0.0):
                on0 += at0 + dnup * n[0]
                on1 += at1 + dnup * n[1]
                on2 += at2 + dnup * n[2]
            else:
                on0 += at0
                on1 += at1
                on2 += at2

        if wzero == 1:
            on2 = scalar_t(0.0)
            oo2 = scalar_t(0.0)
            upn2 = scalar_t(0.0)
            upo2 = scalar_t(0.0)
            adj_new_in[t, gi, 2] = scalar_t(0.0)
            adj_old_in[t, gi, 2] = scalar_t(0.0)

        emit_owner_new[t, g, 0] = on0
        emit_owner_new[t, g, 1] = on1
        emit_owner_new[t, g, 2] = on2
        emit_owner_new[t, g, 3] = on3
        if energy == 1:
            emit_owner_new[t, g, 4] = on4

        emit_owner_old[t, g, 0] = oo0
        emit_owner_old[t, g, 1] = oo1
        emit_owner_old[t, g, 2] = oo2
        if energy == 1:
            emit_owner_old[t, g, 3] = oo4

        emit_u_part_new[t, g, 0] = upn0
        emit_u_part_new[t, g, 1] = upn1
        emit_u_part_new[t, g, 2] = upn2
        emit_u_part_old[t, g, 0] = upo0
        emit_u_part_old[t, g, 1] = upo1
        emit_u_part_old[t, g, 2] = upo2
        emit_p_part_new[t, g] = ppn
        if energy == 1:
            emit_t_part_new[t, g] = tpn
            emit_t_part_old[t, g] = tpo

    @wp.kernel
    def interior_bwd_gather_kernel(
        adj_old_out: wp.array3d(dtype=scalar_t),
        adj_new_out: wp.array3d(dtype=scalar_t),
        cpd_of: wp.array(dtype=wp.int32),
        p_detach_owner: wp.array(dtype=wp.int32),
        own_off: wp.array(dtype=wp.int32),
        own_g: wp.array(dtype=wp.int32),
        emit_owner_new: wp.array3d(dtype=scalar_t),
        emit_owner_old: wp.array3d(dtype=scalar_t),
        up_off: wp.array(dtype=wp.int32),
        up_g: wp.array(dtype=wp.int32),
        emit_u_part_new: wp.array3d(dtype=scalar_t),
        emit_u_part_old: wp.array3d(dtype=scalar_t),
        pp_off: wp.array(dtype=wp.int32),
        pp_g: wp.array(dtype=wp.int32),
        emit_p_part_new: wp.array2d(dtype=scalar_t),
        tp_off: wp.array(dtype=wp.int32),
        tp_g: wp.array(dtype=wp.int32),
        emit_t_part_new: wp.array2d(dtype=scalar_t),
        emit_t_part_old: wp.array2d(dtype=scalar_t),
        cb_in_off: wp.array(dtype=wp.int32),
        cb_in_slot: wp.array(dtype=wp.int32),
        cb_in_weight: wp.array(dtype=wp.float64),
        adj_cbbm_avg: wp.array2d(dtype=wp.float64),
        energy: int,
        wzero: int,
        adj_old_in: wp.array3d(dtype=scalar_t),
        adj_new_in: wp.array3d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ci = cpd_of[i]

        n0 = adj_new_out[t, ci, 0]
        n1 = adj_new_out[t, ci, 1]
        n2 = scalar_t(0.0) if wzero == 1 else adj_new_out[t, ci, 2]
        n3 = scalar_t(0.0) if p_detach_owner[i] == 1 else adj_new_out[t, ci, 3]
        n4 = adj_new_out[t, ci, 4] if energy == 1 else scalar_t(0.0)

        o0 = adj_old_out[t, ci, 0]
        o1 = adj_old_out[t, ci, 1]
        o2 = scalar_t(0.0) if wzero == 1 else adj_old_out[t, ci, 2]
        o3 = adj_old_out[t, ci, 3]
        o4 = adj_old_out[t, ci, 4] if energy == 1 else scalar_t(0.0)

        for k in range(own_off[i], own_off[i + 1]):
            g = own_g[k]
            n0 += emit_owner_new[t, g, 0]
            n1 += emit_owner_new[t, g, 1]
            n2 += emit_owner_new[t, g, 2]
            if p_detach_owner[i] == 0:
                n3 += emit_owner_new[t, g, 3]
            if energy == 1:
                n4 += emit_owner_new[t, g, 4]
            o0 += emit_owner_old[t, g, 0]
            o1 += emit_owner_old[t, g, 1]
            o2 += emit_owner_old[t, g, 2]
            if energy == 1:
                o4 += emit_owner_old[t, g, 3]

        for k in range(up_off[i], up_off[i + 1]):
            g = up_g[k]
            n0 += emit_u_part_new[t, g, 0]
            n1 += emit_u_part_new[t, g, 1]
            n2 += emit_u_part_new[t, g, 2]
            o0 += emit_u_part_old[t, g, 0]
            o1 += emit_u_part_old[t, g, 1]
            o2 += emit_u_part_old[t, g, 2]

        if p_detach_owner[i] == 0:
            for k in range(pp_off[i], pp_off[i + 1]):
                g = pp_g[k]
                n3 += emit_p_part_new[t, g]

        if energy == 1:
            for k in range(tp_off[i], tp_off[i + 1]):
                g = tp_g[k]
                n4 += emit_t_part_new[t, g]
                o4 += emit_t_part_old[t, g]
            for k in range(cb_in_off[i], cb_in_off[i + 1]):
                pslot = cb_in_slot[k]
                n4 += scalar_t(adj_cbbm_avg[t, pslot] * cb_in_weight[k])

        if wzero == 1:
            n2 = scalar_t(0.0)
            o2 = scalar_t(0.0)

        adj_new_in[t, ci, 0] = n0
        adj_new_in[t, ci, 1] = n1
        adj_new_in[t, ci, 2] = n2
        adj_new_in[t, ci, 3] = n3
        if energy == 1:
            adj_new_in[t, ci, 4] = n4

        adj_old_in[t, ci, 0] = o0
        adj_old_in[t, ci, 1] = o1
        adj_old_in[t, ci, 2] = o2
        adj_old_in[t, ci, 3] = o3
        if energy == 1:
            adj_old_in[t, ci, 4] = o4

    return (
        copy_state_wzero_kernel,
        flowout_reduce_kernel,
        cbbm_reduce_kernel,
        ghost_fill_kernel,
        flowout_bwd_reduce_kernel,
        cbbm_bwd_out_reduce_kernel,
        ghost_bwd_emit_kernel,
        interior_bwd_gather_kernel,
    )


_KERNELS_F32 = _make_bc_kernels(wp.float32, wp.vec3f)
_KERNELS_F64 = _make_bc_kernels(wp.float64, wp.vec3d)


def _csr_from_rows(rows: torch.Tensor, n_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if rows.numel() == 0:
        return (
            torch.zeros(n_rows + 1, dtype=torch.int32, device=rows.device),
            torch.empty(0, dtype=torch.int32, device=rows.device),
        )
    order = torch.argsort(rows, stable=True)
    counts = torch.bincount(rows, minlength=n_rows)
    off = torch.zeros(n_rows + 1, dtype=torch.int32, device=rows.device)
    off[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return off, order.to(torch.int32)


def _eval_fn_scalar(spec: Any, device: torch.device) -> torch.Tensor:
    if isinstance(spec, (int, float)):
        return torch.tensor(float(spec), dtype=torch.float64, device=device)
    if torch.is_tensor(spec):
        return spec.to(device=device).squeeze()
    return Functionbase(
        value_type=spec["value_type"],
        value=spec["val"],
        num_element=1,
        channel_dim=1,
    ).eval(device=device).squeeze()


def _eval_fn_field(spec: Any, n_elem: int, ch: int, device: torch.device, bc_obj: Any = None) -> torch.Tensor:
    vt = getattr(bc_obj, "value_tensor", None)
    if vt is not None and hasattr(vt, "eval"):
        out = vt.eval(device=device)
        if out.shape[0] == n_elem:
            return out
    if isinstance(spec, (int, float)):
        return torch.full((n_elem, ch), float(spec), dtype=torch.float64, device=device)
    return Functionbase(
        value_type=spec["value_type"],
        value=spec["val"],
        num_element=n_elem,
        channel_dim=ch,
    ).eval(device=device)


def _eval_fn_python_float(spec: Any, device: torch.device) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict) and spec.get("value_type") in ("constant", "uniform"):
        v = spec.get("val")
        if isinstance(v, (int, float)):
            dt_fn = torch.float32 if spec.get("value_type") == "uniform" else torch.get_default_dtype()
            return float(torch.tensor(v, dtype=dt_fn).item())
        if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (int, float)):
            dt_fn = torch.float32 if spec.get("value_type") == "uniform" else torch.get_default_dtype()
            return float(torch.tensor(v[0], dtype=dt_fn).item())
    return float(_eval_fn_scalar(spec, device).item())


def bc_structure_signature(app: Any) -> tuple:
    """Content-based, ``id()``-free structural signature of ``app.bcpatch_batch``."""
    fvp = app.fvpatch_batch
    bcb = app.bcpatch_batch
    groups_spec = (
        (0, bcb.first, bcb.first_patch_idx_offset),
        (1, bcb.empty, bcb.empty_patch_idx_offset),
        (2, bcb.cyclic, bcb.cyclic_patch_idx_offset),
        (3, bcb.mixed, bcb.mixed_patch_idx_offset),
        (4, bcb.coupled, bcb.coupled_patch_idx_offset),
    )
    entries = []
    for grp_idx, groups, offsets in groups_spec:
        for gi_case, group in enumerate(groups):
            p_off = int(offsets[gi_case])
            for bc in group:
                cls_name = type(bc).__name__
                vt = getattr(bc, "varType", getattr(bc, "var_type", None))
                vt_key = int(vt.value) if hasattr(vt, "value") else (int(vt) if vt is not None else -1)
                gp = int(bc.patch_idx) + p_off
                sf = int(fvp.start_idx_face[gp])
                ef = int(fvp.end_idx_face[gp])
                sc = int(fvp.start_idx_cpd_cell[gp])
                ec = int(fvp.end_idx_cpd_cell[gp])
                if ef - sf <= 0:
                    continue
                in_gp = int(bc.inlet_patch_idx) + p_off if hasattr(bc, "inlet_patch_idx") else -1
                part_gp = int(bc.partner_patch_idx) + p_off if hasattr(bc, "partner_patch_idx") else -1
                use_vol = getattr(bc, "volumetricFlowRate", None) is not None
                rho_raw = getattr(bc, "rho", 1.0)
                rho_v = float(rho_raw) if isinstance(rho_raw, (int, float)) else 1.0
                entries.append(
                    (
                        grp_idx,
                        gp,
                        sf,
                        ef,
                        sc,
                        ec,
                        vt_key,
                        cls_name,
                        in_gp,
                        part_gp,
                        use_vol,
                        rho_v,
                        bool(getattr(bc, "detach_owner", False)),
                        str(getattr(bc, "tangential", "")),
                        str(getattr(bc, "model", "")),
                        bool(getattr(bc, "lag_mean", False)),
                        bool(getattr(bc, "lag_mean", False)),
                    )
                )
    return (int(app.num_cpd_cells), int(app.unv_face.shape[0]), tuple(entries))


class WarpIncompressibleBCs:
    """Compiled boundary-condition descriptor table + Warp kernel executor."""

    def __init__(self, solver: Any, dtype: torch.dtype):
        wp.init()
        self.solver = solver
        self.dtype = dtype
        self.device = solver.cpd_cell_pos.device
        self.wp_dev = wp.device_from_torch(self.device)
        self.is_f64 = dtype == torch.float64
        self.kernels = _KERNELS_F64 if self.is_f64 else _KERNELS_F32

        solver_name = type(solver).__module__.split(".")[-1].lower()
        self.energy = 1 if ("buoyant" in solver_name or hasattr(solver, "alpha_coeff")) else 0
        self.n_ch = 5 if self.energy else 4
        self.dim = int(getattr(solver, "dim", 3))
        self.wzero = 1 if (self.dim == 2 and "buoyant" in solver_name) else 0

        self._compile_tables(solver)

    def _compile_tables(self, app: Any) -> None:
        dev = self.device
        dt = self.dtype
        n_cpd = int(app.num_cpd_cells)
        n_face = int(app.unv_face.shape[0])
        cpd_of = app._interior_cell_indices.to(device=dev, dtype=torch.int64).reshape(-1)
        n_int = int(cpd_of.numel())
        self.n_cpd = n_cpd
        self.n_int = n_int
        self.n_face = n_face

        cpd_to_int = torch.full((n_cpd,), -1, dtype=torch.int64, device=dev)
        cpd_to_int[cpd_of] = torch.arange(n_int, device=dev, dtype=torch.int64)

        is_int = torch.zeros(n_cpd, dtype=torch.bool, device=dev)
        is_int[cpd_of] = True
        ghost_cpd = torch.nonzero(~is_int, as_tuple=False).reshape(-1)
        n_ghost = int(ghost_cpd.numel())
        self.n_ghost = n_ghost

        cpd_to_ghost = torch.full((n_cpd,), -1, dtype=torch.int64, device=dev)
        cpd_to_ghost[ghost_cpd] = torch.arange(n_ghost, device=dev, dtype=torch.int64)

        C = app.C_senders.to(device=dev, dtype=torch.int64).reshape(-1)
        F = app.F_receivers.to(device=dev, dtype=torch.int64).reshape(-1)
        bnd_face_mask = ~is_int[F]
        bnd_faces = torch.nonzero(bnd_face_mask, as_tuple=False).reshape(-1)
        ghost_face = torch.full((n_ghost,), -1, dtype=torch.int64, device=dev)
        ghost_face[cpd_to_ghost[F[bnd_faces]]] = bnd_faces
        if (ghost_face < 0).any():
            raise NotImplementedError("WarpIncompressibleBCs: ghost cell without boundary face")

        owner_cpd = C[ghost_face]
        nrm = app.unv_face[ghost_face].to(device=dev, dtype=dt).contiguous()
        face_area = app.face_area.reshape(-1).to(device=dev, dtype=torch.float64)

        cyc_partner_all = torch.full((n_cpd,), -1, dtype=torch.int64, device=dev)
        gh_cyc, pa_cyc = app._cyclic_ghost_partner_ids()
        if gh_cyc.numel() > 0:
            cyc_partner_all[gh_cyc.to(device=dev, dtype=torch.int64)] = pa_cyc.to(device=dev, dtype=torch.int64)

        u_code = torch.full((n_ghost,), U_CALCULATED, dtype=torch.int32, device=dev)
        u_partner = torch.full((n_ghost,), -1, dtype=torch.int32, device=dev)
        u_val = torch.zeros((n_ghost, 3), dtype=dt, device=dev)
        u_rack_old = torch.zeros((n_ghost,), dtype=torch.int32, device=dev)
        u_patch_slot = torch.full((n_ghost,), -1, dtype=torch.int32, device=dev)
        u_flow_tgt = torch.zeros((n_ghost,), dtype=torch.float64, device=dev)
        u_area_sum = torch.zeros((n_ghost,), dtype=torch.float64, device=dev)
        u_rho_area = torch.zeros((n_ghost,), dtype=torch.float64, device=dev)
        adv_dt_dx = torch.zeros((n_ghost,), dtype=dt, device=dev)
        adv_floor = torch.full((n_ghost,), 0.02, dtype=dt, device=dev)

        p_code = torch.full((n_ghost,), P_CALCULATED, dtype=torch.int32, device=dev)
        p_partner = torch.full((n_ghost,), -1, dtype=torch.int32, device=dev)
        p_val = torch.zeros((n_ghost,), dtype=dt, device=dev)
        p_detach_owner = torch.zeros((n_int,), dtype=torch.int32, device=dev)

        t_code = torch.full((n_ghost,), T_CALCULATED, dtype=torch.int32, device=dev)
        t_partner = torch.full((n_ghost,), -1, dtype=torch.int32, device=dev)
        t_val = torch.zeros((n_ghost,), dtype=dt, device=dev)
        t_patch_slot = torch.full((n_ghost,), -1, dtype=torch.int32, device=dev)

        self.zero_mass_flux_face = torch.zeros((n_face,), dtype=torch.int32, device=dev)
        self.grad_u_bc_code = torch.full((n_face,), G_NONE, dtype=torch.int32, device=dev)
        self.grad_p_bc_code = torch.full((n_face,), G_NONE, dtype=torch.int32, device=dev)
        self.grad_t_bc_code = torch.full((n_face,), G_NONE, dtype=torch.int32, device=dev)
        self.sym_delta_coeff = torch.zeros((n_face,), dtype=dt, device=dev)

        if self.energy:
            if hasattr(app, "gh_face"):
                ghf_all = app.gh_face().reshape(-1).to(device=dev, dtype=dt)
            else:
                bf_idx = app.batch_face.to(device=dev, dtype=torch.int64)
                g_f = app.gravity.to(device=dev, dtype=dt)[bf_idx]
                href_f = app.hRef.to(device=dev, dtype=dt)[bf_idx].reshape(-1)
                fpos = app.face_pos.to(device=dev, dtype=dt)
                ghf_all = (fpos * g_f).sum(dim=1) - torch.norm(g_f, dim=1) * href_f
            beta_face = app.beta[app.batch_face].reshape(-1).to(device=dev, dtype=dt)
            ghb_face_all = ghf_all * beta_face
            ghb_ghost = ghb_face_all[ghost_face].contiguous()
        else:
            ghb_face_all = torch.zeros((n_face,), dtype=dt, device=dev)
            ghb_ghost = torch.zeros((n_ghost,), dtype=dt, device=dev)

        fvp = app.fvpatch_batch
        bcb = app.bcpatch_batch

        flowout_patches = []
        cbbm_patches = []
        self._rack_lag_bcs = []
        self._val_updaters = []
        self._ghost_face = ghost_face

        from FVdomain.SetBC.onthefly.BCbase import VARType

        groups_spec = [
            ("first", bcb.first, bcb.first_patch_idx_offset),
            ("empty", bcb.empty, bcb.empty_patch_idx_offset),
            ("cyclic", bcb.cyclic, bcb.cyclic_patch_idx_offset),
            ("mixed", bcb.mixed, bcb.mixed_patch_idx_offset),
            ("coupled", bcb.coupled, bcb.coupled_patch_idx_offset),
        ]

        for grp_idx, (grp_name, groups, offsets) in enumerate(groups_spec):
            for gi_case, group in enumerate(groups):
                p_off = int(offsets[gi_case])
                for bc_idx, bc in enumerate(group):
                    cls_name = type(bc).__name__
                    vt = getattr(bc, "varType", getattr(bc, "var_type", None))
                    gp = int(bc.patch_idx) + p_off
                    sf = int(fvp.start_idx_face[gp])
                    ef = int(fvp.end_idx_face[gp])
                    sc = int(fvp.start_idx_cpd_cell[gp])
                    ec = int(fvp.end_idx_cpd_cell[gp])
                    nf_p = ef - sf
                    if nf_p <= 0:
                        continue
                    g_slice = cpd_to_ghost[torch.arange(sc, ec, device=dev, dtype=torch.int64)]
                    if (g_slice < 0).any():
                        raise RuntimeError(f"Patch {bc.patch_name} cpd range [{sc},{ec}) is not ghost")

                    if vt == VARType.VELOCITY:
                        if cls_name == "Calculated":
                            u_code[g_slice] = U_CALCULATED
                        elif cls_name in ("Fixedvalue", "Uniformfixedvalue"):
                            spec_u = bc.uniformValue if cls_name == "Uniformfixedvalue" else bc.value
                            bc_ref = None if cls_name == "Uniformfixedvalue" else bc
                            val_t = _eval_fn_field(spec_u, nf_p, 3, dev, bc_ref).to(dtype=dt)
                            u_code[g_slice] = U_FIXED
                            u_val[g_slice] = val_t
                            self._val_updaters.append(("u_fixed", grp_idx, gi_case, bc_idx, g_slice, nf_p, cls_name))
                        elif cls_name == "Noslip":
                            u_code[g_slice] = U_NOSLIP
                        elif cls_name == "Flowrateinletvelocity":
                            tot_a = face_area[sf:ef].to(dtype=dt).sum()
                            nrm_p = app.unv_face[sf:ef].to(device=dev, dtype=dt).contiguous()
                            if bc.volumetricFlowRate is not None:
                                qv = _eval_fn_python_float(bc.volumetricFlowRate, dev)
                                vmag = qv / tot_a
                            else:
                                mf = _eval_fn_python_float(bc.massFlowRate, dev)
                                vmag = mf / (float(bc.rho) * tot_a)
                            u_code[g_slice] = U_FIXED
                            u_val[g_slice] = -vmag.to(dtype=dt) * nrm_p
                            self._val_updaters.append(("u_flow_in", grp_idx, gi_case, bc_idx, g_slice, tot_a, nrm_p))
                        elif cls_name == "Flowrateoutletvelocity":
                            if bc.volumetricFlowRate is not None:
                                tgt = _eval_fn_python_float(bc.volumetricFlowRate, dev)
                                rho_fl = 1.0
                            else:
                                tgt = _eval_fn_python_float(bc.massFlowRate, dev)
                                rho_fl = float(bc.rho)
                            rho_areas = rho_fl * face_area[sf:ef]
                            asum = float(rho_areas.sum().item())
                            pslot = len(flowout_patches)
                            flowout_patches.append({
                                "ghost_cpd": ghost_cpd[g_slice],
                                "owner_cpd": owner_cpd[g_slice],
                                "nrm": app.unv_face[sf:ef].to(device=dev, dtype=dt),
                                "rho_area": rho_areas,
                                "tgt": tgt,
                                "asum": asum,
                            })
                            u_code[g_slice] = U_FLOW_OUTLET
                            u_patch_slot[g_slice] = pslot
                            u_flow_tgt[g_slice] = tgt
                            u_area_sum[g_slice] = asum
                            u_rho_area[g_slice] = rho_areas
                            self._val_updaters.append(("u_flow_out", grp_idx, gi_case, bc_idx, g_slice, pslot))
                        elif cls_name in ("Zerogradient", "Empty"):
                            u_code[g_slice] = U_ZEROGRAD
                            if cls_name == "Zerogradient":
                                self.grad_u_bc_code[sf:ef] = G_ZERO
                        elif cls_name == "Cyclic":
                            u_code[g_slice] = U_CYCLIC
                            u_partner[g_slice] = cyc_partner_all[ghost_cpd[g_slice]].to(torch.int32)
                        elif cls_name == "Symmetry":
                            u_code[g_slice] = U_SYMMETRY
                            self.zero_mass_flux_face[sf:ef] = 1
                            self.grad_u_bc_code[sf:ef] = G_SYMMETRY_U
                            d_vec = app.cpd_cell_pos[ghost_cpd[g_slice]] - app.cpd_cell_pos[owner_cpd[g_slice]]
                            dist = torch.norm(d_vec, dim=-1).to(dtype=dt)
                            self.sym_delta_coeff[sf:ef] = 1.0 / (dist + 1.0e-12)
                        elif cls_name == "Slip":
                            u_code[g_slice] = U_SLIP
                            self.zero_mass_flux_face[sf:ef] = 1
                            self.grad_u_bc_code[sf:ef] = G_ZERO
                        elif cls_name == "Rackinletvelocity":
                            qv = (
                                float(bc.flow_rate)
                                if hasattr(bc, "flow_rate")
                                else _eval_fn_python_float(bc.volumetricFlowRate, dev)
                            )
                            tot_a = app.face_area[sf:ef].to(device=dev, dtype=dt).sum()
                            nrm_p = app.unv_face[sf:ef].to(device=dev, dtype=dt).contiguous()
                            vn = (qv / tot_a).to(dtype=dt)
                            u_code[g_slice] = U_RACK_PRESERVE
                            u_val[g_slice] = vn * nrm_p
                            self.grad_u_bc_code[sf:ef] = G_RACK_INLET_U
                            self._val_updaters.append(("u_rack_in", grp_idx, gi_case, bc_idx, g_slice, tot_a, nrm_p))
                        elif cls_name == "Rackoutletvelocity":
                            qv = (
                                float(bc.flow_rate)
                                if hasattr(bc, "flow_rate")
                                else _eval_fn_python_float(bc.volumetricFlowRate, dev)
                            )
                            tot_a = app.face_area[sf:ef].to(device=dev, dtype=dt).sum()
                            nrm_p = app.unv_face[sf:ef].to(device=dev, dtype=dt).contiguous()
                            vn = (-qv / tot_a).to(dtype=dt)
                            if getattr(bc, "tangential", "zero") == "preserve":
                                u_code[g_slice] = U_RACK_PRESERVE
                            else:
                                u_code[g_slice] = U_FIXED
                            u_val[g_slice] = vn * nrm_p
                            self._val_updaters.append(("u_rack_out", grp_idx, gi_case, bc_idx, g_slice, tot_a, nrm_p))
                        elif cls_name == "Inletoutlet":
                            in_spec = (
                                getattr(bc, "inletValue", None)
                                or getattr(bc, "inlet_value", None)
                                or bc.value
                            )
                            val_t = _eval_fn_field(in_spec, nf_p, 3, dev).to(dtype=dt)
                            u_code[g_slice] = U_INLETOUTLET
                            u_val[g_slice] = val_t
                            self.grad_u_bc_code[sf:ef] = G_INLETOUTLET_U
                            self._val_updaters.append(("u_inlet_out", grp_idx, gi_case, bc_idx, g_slice, nf_p))
                        elif cls_name == "Advectiveoutlet":
                            d_vec = app.cpd_cell_pos[ghost_cpd[g_slice]] - app.cpd_cell_pos[owner_cpd[g_slice]]
                            dist = torch.norm(d_vec, dim=-1).to(dtype=dt).clamp_min(1.0e-12)
                            own_int = cpd_to_int[owner_cpd[g_slice]]
                            dt_vals = app.dt_cell[own_int].reshape(-1).to(dtype=dt)
                            u_code[g_slice] = U_ADVECTIVE
                            adv_dt_dx[g_slice] = dt_vals / dist
                            adv_floor[g_slice] = float(getattr(bc, "alpha_floor", 0.02))
                            self.grad_u_bc_code[sf:ef] = G_ZERO
                            self._val_updaters.append(("u_advective", grp_idx, gi_case, bc_idx, g_slice, own_int, (1.0 / dist).contiguous()))
                        else:
                            raise NotImplementedError(f"WarpIncompressibleBCs: unsupported U BC {cls_name!r}")

                    elif vt in (VARType.PRESSURE, getattr(VARType, "PRESSURE_RGH", VARType.PRESSURE)):
                        is_prgh = bool(self.energy) or ("setP_rgh" in type(bc).__module__)
                        if cls_name == "Calculated":
                            p_code[g_slice] = P_CALCULATED
                        elif cls_name == "Fixedvalue":
                            val_t = _eval_fn_field(bc.value, nf_p, 1, dev, bc).reshape(-1).to(dtype=dt)
                            p_code[g_slice] = P_FIXED
                            p_val[g_slice] = val_t
                            if getattr(bc, "detach_owner", False):
                                p_detach_owner[cpd_to_int[owner_cpd[g_slice]]] = 1
                            self._val_updaters.append(("p_fixed", grp_idx, gi_case, bc_idx, g_slice, nf_p))
                        elif cls_name in ("Zerogradient", "Symmetry", "Empty"):
                            p_code[g_slice] = P_ZEROGRAD
                            if is_prgh and cls_name in ("Zerogradient", "Symmetry"):
                                self.grad_p_bc_code[sf:ef] = G_ZERO
                        elif cls_name == "Cyclic":
                            p_code[g_slice] = P_CYCLIC
                            p_partner[g_slice] = cyc_partner_all[ghost_cpd[g_slice]].to(torch.int32)
                        elif cls_name == "Fixedfluxpressure":
                            p_code[g_slice] = P_FIXEDFLUX
                            self.grad_p_bc_code[sf:ef] = G_FIXEDFLUX_P
                        else:
                            raise NotImplementedError(f"WarpIncompressibleBCs: unsupported P BC {cls_name!r}")

                    elif vt == VARType.TEMPERATURE and self.energy:
                        if cls_name == "Calculated":
                            t_code[g_slice] = T_CALCULATED
                        elif cls_name == "Fixedvalue":
                            val_t = _eval_fn_field(bc.value, nf_p, 1, dev, bc).reshape(-1).to(dtype=dt)
                            t_code[g_slice] = T_FIXED
                            t_val[g_slice] = val_t
                            self._val_updaters.append(("t_fixed", grp_idx, gi_case, bc_idx, g_slice, nf_p))
                        elif cls_name in ("Zerogradient", "Symmetry", "Empty"):
                            t_code[g_slice] = T_ZEROGRAD
                            if cls_name in ("Zerogradient", "Symmetry"):
                                self.grad_t_bc_code[sf:ef] = G_ZERO
                        elif cls_name == "Cyclic":
                            t_code[g_slice] = T_CYCLIC
                            t_partner[g_slice] = cyc_partner_all[ghost_cpd[g_slice]].to(torch.int32)
                        elif cls_name == "Exprfixedvalue":
                            in_gp = int(bc.inlet_patch_idx) + p_off
                            isf = int(fvp.start_idx_face[in_gp])
                            ief = int(fvp.end_idx_face[in_gp])
                            in_owners = C[isf:ief]
                            in_areas = face_area[isf:ief]
                            dt_rise = _eval_fn_python_float(bc.deltaT, dev)
                            pslot = len(cbbm_patches)
                            cbbm_patches.append({
                                "in_owners": in_owners,
                                "in_areas": in_areas,
                                "in_denom": float(in_areas.sum().item()) + 1.0e-12,
                                "out_g_slice": g_slice,
                                "lag_mean": 1 if getattr(bc, "lag_mean", False) else 0,
                            })
                            t_code[g_slice] = T_CBBM
                            t_patch_slot[g_slice] = pslot
                            t_val[g_slice] = dt_rise
                            self._val_updaters.append(("t_expr", grp_idx, gi_case, bc_idx, g_slice))
                        elif cls_name == "Rackoutlettemperature":
                            dt_rise = float(bc.deltaT)
                            _src_own, src_ghost, tgt_own, _tgt_ghost, _perm = bc._pairing(app, p_off)
                            g_pair = cpd_to_ghost[src_ghost.to(device=dev, dtype=torch.int64)]
                            if getattr(bc, "model", "ibbm") == "ibbm":
                                t_code[g_pair] = T_IBBM
                                t_partner[g_pair] = tgt_own.to(device=dev, dtype=torch.int32)
                                t_val[g_pair] = dt_rise
                                self._val_updaters.append(("t_rack_dt", grp_idx, gi_case, bc_idx, g_pair))
                            elif getattr(bc, "lag_mean", False):
                                in_gp = int(bc.partner_patch_idx) + p_off
                                isf = int(fvp.start_idx_face[in_gp])
                                ief = int(fvp.end_idx_face[in_gp])
                                in_owners = tgt_own.to(device=dev, dtype=torch.int64)
                                in_areas = face_area[isf:ief]
                                in_denom = max(float(in_areas.sum().item()), 1.0e-30)
                                lag_val = getattr(bc, "_lagged_mean", None)
                                t_code[g_pair] = T_FIXED
                                t_val[g_pair] = (float(lag_val) if lag_val is not None else 0.0) + dt_rise
                                lag_idx = len(self._rack_lag_bcs)
                                self._rack_lag_bcs.append({
                                    "bc": bc,
                                    "g_pair": g_pair,
                                    "in_owners": in_owners,
                                    "in_areas": in_areas,
                                    "in_denom": in_denom,
                                    "dt_rise": dt_rise,
                                    "cached": lag_val,
                                })
                                self._val_updaters.append(("t_rack_lag", grp_idx, gi_case, bc_idx, g_pair, lag_idx))
                            else:
                                in_gp = int(bc.partner_patch_idx) + p_off
                                isf = int(fvp.start_idx_face[in_gp])
                                ief = int(fvp.end_idx_face[in_gp])
                                in_owners = tgt_own.to(device=dev, dtype=torch.int64)
                                in_areas = face_area[isf:ief]
                                pslot = len(cbbm_patches)
                                cbbm_patches.append({
                                    "in_owners": in_owners,
                                    "in_areas": in_areas,
                                    "in_denom": max(float(in_areas.sum().item()), 1.0e-30),
                                    "out_g_slice": g_pair,
                                    "lag_mean": 0,
                                })
                                t_code[g_pair] = T_CBBM
                                t_patch_slot[g_pair] = pslot
                                t_val[g_pair] = dt_rise
                                self._val_updaters.append(("t_rack_dt", grp_idx, gi_case, bc_idx, g_pair))
                        else:
                            raise NotImplementedError(f"WarpIncompressibleBCs: unsupported T BC {cls_name!r}")

        owner_int = cpd_to_int[owner_cpd]
        own_off, own_g = _csr_from_rows(owner_int, n_int)

        def _partner_csr(part_cpd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            valid = part_cpd >= 0
            if not valid.any():
                return (
                    torch.zeros(n_int + 1, dtype=torch.int32, device=dev),
                    torch.empty(0, dtype=torch.int32, device=dev),
                )
            g_ids = torch.nonzero(valid, as_tuple=False).reshape(-1).to(torch.int32)
            rows = cpd_to_int[part_cpd[valid].to(torch.int64)]
            off, perm = _csr_from_rows(rows, n_int)
            return off, g_ids[perm]

        up_off, up_g = _partner_csr(u_partner)
        pp_off, pp_g = _partner_csr(p_partner)
        tp_off, tp_g = _partner_csr(t_partner)

        self.n_flowout = len(flowout_patches)
        if self.n_flowout > 0:
            fo_counts = [int(p["owner_cpd"].numel()) for p in flowout_patches]
            fo_off_t = torch.zeros(self.n_flowout + 1, dtype=torch.int32, device=dev)
            fo_off_t[1:] = torch.cumsum(torch.tensor(fo_counts, device=dev, dtype=torch.int32), dim=0)
            fo_owner_t = torch.cat([p["owner_cpd"] for p in flowout_patches]).to(torch.int32)
            fo_ghost_t = torch.cat([p["ghost_cpd"] for p in flowout_patches]).to(torch.int32)
            fo_nrm_t = torch.cat([p["nrm"] for p in flowout_patches]).contiguous()
            fo_rho_area_t = torch.cat([p["rho_area"] for p in flowout_patches]).contiguous()
            fo_tgt_t = torch.tensor([p["tgt"] for p in flowout_patches], dtype=torch.float64, device=dev)
            fo_asum_t = torch.tensor([p["asum"] for p in flowout_patches], dtype=torch.float64, device=dev)
        else:
            fo_off_t = torch.zeros(1, dtype=torch.int32, device=dev)
            fo_owner_t = torch.empty(0, dtype=torch.int32, device=dev)
            fo_ghost_t = torch.empty(0, dtype=torch.int32, device=dev)
            fo_nrm_t = torch.zeros((0, 3), dtype=dt, device=dev)
            fo_rho_area_t = torch.empty(0, dtype=torch.float64, device=dev)
            fo_tgt_t = torch.empty(0, dtype=torch.float64, device=dev)
            fo_asum_t = torch.empty(0, dtype=torch.float64, device=dev)

        self.n_cbbm = len(cbbm_patches)
        if self.n_cbbm > 0:
            cb_counts = [int(p["in_owners"].numel()) for p in cbbm_patches]
            cb_off_t = torch.zeros(self.n_cbbm + 1, dtype=torch.int32, device=dev)
            cb_off_t[1:] = torch.cumsum(torch.tensor(cb_counts, device=dev, dtype=torch.int32), dim=0)
            cb_owner_t = torch.cat([p["in_owners"] for p in cbbm_patches]).to(torch.int32)
            cb_area_t = torch.cat([p["in_areas"] for p in cbbm_patches]).contiguous()
            cb_asum_t = torch.tensor([p["in_denom"] for p in cbbm_patches], dtype=torch.float64, device=dev)
            cb_lag_t = torch.tensor([p["lag_mean"] for p in cbbm_patches], dtype=torch.int32, device=dev)

            out_counts = [int(p["out_g_slice"].numel()) for p in cbbm_patches]
            cb_out_off_t = torch.zeros(self.n_cbbm + 1, dtype=torch.int32, device=dev)
            cb_out_off_t[1:] = torch.cumsum(torch.tensor(out_counts, device=dev, dtype=torch.int32), dim=0)
            out_g_all = torch.cat([p["out_g_slice"] for p in cbbm_patches])
            self._cb_out_g_all = out_g_all
            cb_out_ghost_t = ghost_cpd[out_g_all].to(torch.int32)
            cb_out_ghb_t = ghb_ghost[out_g_all].contiguous()
            cb_out_ffp_t = (p_code[out_g_all] == P_FIXEDFLUX).to(torch.int32)

            in_int_list, in_slot_list, in_w_list = [], [], []
            for p_idx, p in enumerate(cbbm_patches):
                if p["lag_mean"] == 1:
                    continue
                in_int_list.append(cpd_to_int[p["in_owners"]])
                in_slot_list.append(torch.full((p["in_owners"].numel(),), p_idx, dtype=torch.int32, device=dev))
                in_w_list.append(p["in_areas"] / p["in_denom"])
            if in_int_list:
                in_rows = torch.cat(in_int_list)
                in_slots = torch.cat(in_slot_list)
                in_ws = torch.cat(in_w_list)
                cb_in_off_t, perm = _csr_from_rows(in_rows, n_int)
                cb_in_slot_t = in_slots[perm].contiguous()
                cb_in_weight_t = in_ws[perm].contiguous()
            else:
                cb_in_off_t = torch.zeros(n_int + 1, dtype=torch.int32, device=dev)
                cb_in_slot_t = torch.empty(0, dtype=torch.int32, device=dev)
                cb_in_weight_t = torch.empty(0, dtype=torch.float64, device=dev)
        else:
            self._cb_out_g_all = None
            cb_off_t = torch.zeros(1, dtype=torch.int32, device=dev)
            cb_owner_t = torch.empty(0, dtype=torch.int32, device=dev)
            cb_area_t = torch.empty(0, dtype=torch.float64, device=dev)
            cb_asum_t = torch.empty(0, dtype=torch.float64, device=dev)
            cb_lag_t = torch.empty(0, dtype=torch.int32, device=dev)
            cb_out_off_t = torch.zeros(1, dtype=torch.int32, device=dev)
            cb_out_ghost_t = torch.empty(0, dtype=torch.int32, device=dev)
            cb_out_ghb_t = torch.empty(0, dtype=dt, device=dev)
            cb_out_ffp_t = torch.empty(0, dtype=torch.int32, device=dev)
            cb_in_off_t = torch.zeros(n_int + 1, dtype=torch.int32, device=dev)
            cb_in_slot_t = torch.empty(0, dtype=torch.int32, device=dev)
            cb_in_weight_t = torch.empty(0, dtype=torch.float64, device=dev)

        v3_wp = wp.vec3d if self.is_f64 else wp.vec3f

        self._torch_refs = {
            "cpd_of": cpd_of.to(torch.int32).contiguous(),
            "ghost_cpd": ghost_cpd.to(torch.int32).contiguous(),
            "owner_cpd": owner_cpd.to(torch.int32).contiguous(),
            "nrm": nrm.contiguous(),
            "u_code": u_code.contiguous(),
            "u_partner": u_partner.contiguous(),
            "u_val": u_val.contiguous(),
            "u_rack_old": u_rack_old.contiguous(),
            "u_patch_slot": u_patch_slot.contiguous(),
            "u_flow_tgt": u_flow_tgt.contiguous(),
            "u_area_sum": u_area_sum.contiguous(),
            "u_rho_area": u_rho_area.contiguous(),
            "adv_dt_dx": adv_dt_dx.contiguous(),
            "adv_floor": adv_floor.contiguous(),
            "p_code": p_code.contiguous(),
            "p_partner": p_partner.contiguous(),
            "p_val": p_val.contiguous(),
            "p_detach_owner": p_detach_owner.contiguous(),
            "ghb_ghost": ghb_ghost.contiguous(),
            "ghb_face_all": ghb_face_all.contiguous(),
            "t_code": t_code.contiguous(),
            "t_partner": t_partner.contiguous(),
            "t_val": t_val.contiguous(),
            "t_patch_slot": t_patch_slot.contiguous(),
            "own_off": own_off.contiguous(),
            "own_g": own_g.contiguous(),
            "up_off": up_off.contiguous(),
            "up_g": up_g.contiguous(),
            "pp_off": pp_off.contiguous(),
            "pp_g": pp_g.contiguous(),
            "tp_off": tp_off.contiguous(),
            "tp_g": tp_g.contiguous(),
            "fo_off": fo_off_t.contiguous(),
            "fo_owner": fo_owner_t.contiguous(),
            "fo_ghost": fo_ghost_t.contiguous(),
            "fo_nrm": fo_nrm_t.contiguous(),
            "fo_rho_area": fo_rho_area_t.contiguous(),
            "fo_tgt": fo_tgt_t.contiguous(),
            "fo_asum": fo_asum_t.contiguous(),
            "cb_off": cb_off_t.contiguous(),
            "cb_owner": cb_owner_t.contiguous(),
            "cb_area": cb_area_t.contiguous(),
            "cb_asum": cb_asum_t.contiguous(),
            "cb_lag": cb_lag_t.contiguous(),
            "cb_out_off": cb_out_off_t.contiguous(),
            "cb_out_ghost": cb_out_ghost_t.contiguous(),
            "cb_out_ghb": cb_out_ghb_t.contiguous(),
            "cb_out_ffp": cb_out_ffp_t.contiguous(),
            "cb_in_off": cb_in_off_t.contiguous(),
            "cb_in_slot": cb_in_slot_t.contiguous(),
            "cb_in_weight": cb_in_weight_t.contiguous(),
        }

        r = self._torch_refs
        self.w = {
            k: _wp_from_torch(v, dtype=v3_wp) if k in ("nrm", "u_val", "fo_nrm") else _wp_from_torch(v)
            for k, v in r.items()
        }

    def refresh_bc_values(self, app: Any) -> None:
        """Fast per-batch refresh of numeric BC values when mesh + BC structure match."""
        dev = self.device
        dt = self.dtype
        v3_wp = wp.vec3d if self.is_f64 else wp.vec3f
        n_ghost = self.n_ghost
        bcb = app.bcpatch_batch
        groups_by_idx = (bcb.first, bcb.empty, bcb.cyclic, bcb.mixed, bcb.coupled)

        u_val = torch.zeros((n_ghost, 3), dtype=dt, device=dev)
        p_val = torch.zeros((n_ghost,), dtype=dt, device=dev)
        t_val = torch.zeros((n_ghost,), dtype=dt, device=dev)
        u_flow_tgt = (
            torch.zeros((n_ghost,), dtype=torch.float64, device=dev)
            if self.n_flowout > 0
            else self._torch_refs["u_flow_tgt"]
        )
        fo_tgts = [0.0] * self.n_flowout
        adv_dt_dx = None
        adv_floor = None

        for entry in self._val_updaters:
            kind = entry[0]
            grp_idx, gi_case, bc_idx = entry[1], entry[2], entry[3]
            bc = groups_by_idx[grp_idx][gi_case][bc_idx]
            if kind == "u_fixed":
                g_slice, nf_p, cls_name = entry[4], entry[5], entry[6]
                spec_u = bc.uniformValue if cls_name == "Uniformfixedvalue" else bc.value
                bc_ref = None if cls_name == "Uniformfixedvalue" else bc
                u_val[g_slice] = _eval_fn_field(spec_u, nf_p, 3, dev, bc_ref).to(dtype=dt)
            elif kind == "u_flow_in":
                g_slice, tot_a, nrm_p = entry[4], entry[5], entry[6]
                if bc.volumetricFlowRate is not None:
                    qv = _eval_fn_python_float(bc.volumetricFlowRate, dev)
                    vmag = qv / tot_a
                else:
                    mf = _eval_fn_python_float(bc.massFlowRate, dev)
                    vmag = mf / (float(bc.rho) * tot_a)
                u_val[g_slice] = -vmag.to(dtype=dt) * nrm_p
            elif kind == "u_flow_out":
                g_slice, pslot = entry[4], entry[5]
                if bc.volumetricFlowRate is not None:
                    tgt = _eval_fn_python_float(bc.volumetricFlowRate, dev)
                else:
                    tgt = _eval_fn_python_float(bc.massFlowRate, dev)
                u_flow_tgt[g_slice] = tgt
                fo_tgts[pslot] = tgt
            elif kind == "u_rack_in":
                g_slice, tot_a, nrm_p = entry[4], entry[5], entry[6]
                qv = (
                    float(bc.flow_rate)
                    if hasattr(bc, "flow_rate")
                    else _eval_fn_python_float(bc.volumetricFlowRate, dev)
                )
                u_val[g_slice] = (qv / tot_a).to(dtype=dt) * nrm_p
            elif kind == "u_rack_out":
                g_slice, tot_a, nrm_p = entry[4], entry[5], entry[6]
                qv = (
                    float(bc.flow_rate)
                    if hasattr(bc, "flow_rate")
                    else _eval_fn_python_float(bc.volumetricFlowRate, dev)
                )
                u_val[g_slice] = (-qv / tot_a).to(dtype=dt) * nrm_p
            elif kind == "u_inlet_out":
                g_slice, nf_p = entry[4], entry[5]
                in_spec = (
                    getattr(bc, "inletValue", None)
                    or getattr(bc, "inlet_value", None)
                    or bc.value
                )
                u_val[g_slice] = _eval_fn_field(in_spec, nf_p, 3, dev).to(dtype=dt)
            elif kind == "u_advective":
                g_slice, own_int, inv_dist = entry[4], entry[5], entry[6]
                if adv_dt_dx is None:
                    adv_dt_dx = torch.zeros((n_ghost,), dtype=dt, device=dev)
                    adv_floor = torch.full((n_ghost,), 0.02, dtype=dt, device=dev)
                dt_vals = app.dt_cell[own_int].reshape(-1).to(dtype=dt)
                adv_dt_dx[g_slice] = dt_vals * inv_dist
                adv_floor[g_slice] = float(getattr(bc, "alpha_floor", 0.02))
            elif kind == "p_fixed":
                g_slice, nf_p = entry[4], entry[5]
                p_val[g_slice] = _eval_fn_field(bc.value, nf_p, 1, dev, bc).reshape(-1).to(dtype=dt)
            elif kind == "t_fixed":
                g_slice, nf_p = entry[4], entry[5]
                t_val[g_slice] = _eval_fn_field(bc.value, nf_p, 1, dev, bc).reshape(-1).to(dtype=dt)
            elif kind == "t_expr":
                g_slice = entry[4]
                t_val[g_slice] = _eval_fn_python_float(bc.deltaT, dev)
            elif kind == "t_rack_dt":
                g_pair = entry[4]
                t_val[g_pair] = float(bc.deltaT)
            elif kind == "t_rack_lag":
                g_pair, lag_idx = entry[4], entry[5]
                dt_rise = float(bc.deltaT)
                lag_val = getattr(bc, "_lagged_mean", None)
                t_val[g_pair] = (float(lag_val) if lag_val is not None else 0.0) + dt_rise
                self._rack_lag_bcs[lag_idx]["bc"] = bc
                self._rack_lag_bcs[lag_idx]["dt_rise"] = dt_rise
                self._rack_lag_bcs[lag_idx]["cached"] = lag_val

        updates = {
            "u_val": u_val.contiguous(),
            "p_val": p_val.contiguous(),
            "t_val": t_val.contiguous(),
        }
        if self.n_flowout > 0:
            updates["u_flow_tgt"] = u_flow_tgt.contiguous()
            updates["fo_tgt"] = torch.tensor(fo_tgts, dtype=torch.float64, device=dev)
        if adv_dt_dx is not None:
            updates["adv_dt_dx"] = adv_dt_dx.contiguous()
            updates["adv_floor"] = adv_floor.contiguous()
        if self.energy:
            if hasattr(app, "gh_face"):
                ghf_all = app.gh_face().reshape(-1).to(device=dev, dtype=dt)
            else:
                bf_idx = app.batch_face.to(device=dev, dtype=torch.int64)
                g_f = app.gravity.to(device=dev, dtype=dt)[bf_idx]
                href_f = app.hRef.to(device=dev, dtype=dt)[bf_idx].reshape(-1)
                fpos = app.face_pos.to(device=dev, dtype=dt)
                ghf_all = (fpos * g_f).sum(dim=1) - torch.norm(g_f, dim=1) * href_f
            beta_face = app.beta[app.batch_face].reshape(-1).to(device=dev, dtype=dt)
            ghb_face_all = (ghf_all * beta_face).contiguous()
            ghb_ghost = ghb_face_all[self._ghost_face].contiguous()
            updates["ghb_face_all"] = ghb_face_all
            updates["ghb_ghost"] = ghb_ghost
            if self._cb_out_g_all is not None:
                updates["cb_out_ghb"] = ghb_ghost[self._cb_out_g_all].contiguous()

        for k, v in updates.items():
            self._torch_refs[k] = v
            self.w[k] = _wp_from_torch(v, dtype=v3_wp) if k == "u_val" else _wp_from_torch(v)

    def forward_pass(
        self,
        phi_old_3d: torch.Tensor,
        phi_new_3d: torch.Tensor,
        include_old_u: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute Pass-1 ghost enforcement in Warp on ``[T, N_cpd, C]`` tensors."""
        T = int(phi_new_3d.shape[0])
        dev = self.device
        dt = self.dtype
        if self._rack_lag_bcs:
            t_val_ref = self._torch_refs["t_val"]
            for entry in self._rack_lag_bcs:
                bc_obj = entry["bc"]
                lag_val = getattr(bc_obj, "_lagged_mean", None)
                if lag_val is None:
                    intake0 = phi_new_3d[0, entry["in_owners"], 4].detach().to(torch.float64)
                    lag_val = float(((intake0 * entry["in_areas"]).sum() / entry["in_denom"]).item())
                    bc_obj._lagged_mean = lag_val
                if entry["cached"] != lag_val:
                    t_val_ref[entry["g_pair"]] = float(lag_val) + entry["dt_rise"]
                    entry["cached"] = lag_val
        (
            copy_k,
            fo_red_k,
            cb_red_k,
            ghost_k,
            _,
            _,
            _,
            _,
        ) = self.kernels

        phi_old_c = phi_old_3d.contiguous()
        phi_new_c = phi_new_3d.contiguous()
        phi_old_out = torch.empty_like(phi_old_c)
        phi_new_out = torch.empty_like(phi_new_c)

        est_flow = torch.zeros((T, max(self.n_flowout, 1)), dtype=torch.float64, device=dev)
        cbbm_t_avg = torch.zeros((T, max(self.n_cbbm, 1)), dtype=torch.float64, device=dev)

        stream = wp.stream_from_torch(dev)
        w = self.w
        w_old_in = _wp_from_torch(phi_old_c)
        w_new_in = _wp_from_torch(phi_new_c)
        w_old_out = _wp_from_torch(phi_old_out)
        w_new_out = _wp_from_torch(phi_new_out)
        w_est = _wp_from_torch(est_flow)
        w_cbbm = _wp_from_torch(cbbm_t_avg)

        wp.launch(
            copy_k,
            dim=(T, self.n_cpd),
            inputs=[w_old_in, w_new_in, self.wzero, self.n_ch],
            outputs=[w_old_out, w_new_out],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        if self.n_flowout > 0:
            wp.launch(
                fo_red_k,
                dim=(T, self.n_flowout),
                inputs=[w_new_out, w["fo_off"], w["fo_owner"], w["fo_nrm"], w["fo_rho_area"]],
                outputs=[w_est],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        if self.n_cbbm > 0:
            wp.launch(
                cb_red_k,
                dim=(T, self.n_cbbm),
                inputs=[w_new_out, w["cb_off"], w["cb_owner"], w["cb_area"], w["cb_asum"]],
                outputs=[w_cbbm],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        if self.n_ghost > 0:
            wp.launch(
                ghost_k,
                dim=(T, self.n_ghost),
                inputs=[
                    w_old_out,
                    w_new_out,
                    w["ghost_cpd"],
                    w["owner_cpd"],
                    w["nrm"],
                    w["u_code"],
                    w["u_partner"],
                    w["u_val"],
                    w["u_rack_old"],
                    w["u_patch_slot"],
                    w["u_flow_tgt"],
                    w["u_area_sum"],
                    w["adv_dt_dx"],
                    w["adv_floor"],
                    w_est,
                    w["p_code"],
                    w["p_partner"],
                    w["p_val"],
                    w["ghb_ghost"],
                    w["t_code"],
                    w["t_partner"],
                    w["t_val"],
                    w["t_patch_slot"],
                    w_cbbm,
                    self.energy,
                    include_old_u,
                    self.wzero,
                ],
                outputs=[w_old_out, w_new_out],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        return phi_old_out, phi_new_out, est_flow, cbbm_t_avg

    def backward_pass(
        self,
        phi_old_in_3d: torch.Tensor,
        phi_new_in_3d: torch.Tensor,
        est_flow: torch.Tensor,
        adj_old_out_3d: torch.Tensor,
        adj_new_out_3d: torch.Tensor,
        include_old_u: int = 1,
        w_override: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact zero-atomic reverse-mode VJP of ``forward_pass``."""
        T = int(phi_new_in_3d.shape[0])
        dev = self.device
        dt = self.dtype
        (
            _,
            _,
            _,
            _,
            fo_bwd_k,
            cb_bwd_k,
            emit_k,
            gather_k,
        ) = self.kernels

        phi_old_c = phi_old_in_3d.contiguous()
        phi_new_c = phi_new_in_3d.contiguous()
        adj_old_c = adj_old_out_3d.contiguous()
        adj_new_c = adj_new_out_3d.contiguous()

        adj_old_in = torch.zeros_like(phi_old_c)
        adj_new_in = torch.zeros_like(phi_new_c)

        ng = max(self.n_ghost, 1)
        emit_owner_new = torch.zeros((T, ng, self.n_ch), dtype=dt, device=dev)
        emit_owner_old = torch.zeros((T, ng, 4 if self.energy else 3), dtype=dt, device=dev)
        emit_u_part_new = torch.zeros((T, ng, 3), dtype=dt, device=dev)
        emit_u_part_old = torch.zeros((T, ng, 3), dtype=dt, device=dev)
        emit_p_part_new = torch.zeros((T, ng), dtype=dt, device=dev)
        emit_t_part_new = torch.zeros((T, ng), dtype=dt, device=dev)
        emit_t_part_old = torch.zeros((T, ng), dtype=dt, device=dev)

        adj_est_flow = torch.zeros((T, max(self.n_flowout, 1)), dtype=torch.float64, device=dev)
        adj_cbbm_avg = torch.zeros((T, max(self.n_cbbm, 1)), dtype=torch.float64, device=dev)

        stream = wp.stream_from_torch(dev)
        w = self.w if w_override is None else w_override
        w_old_in = _wp_from_torch(phi_old_c)
        w_new_in = _wp_from_torch(phi_new_c)
        w_adj_old_out = _wp_from_torch(adj_old_c)
        w_adj_new_out = _wp_from_torch(adj_new_c)
        w_adj_old_in = _wp_from_torch(adj_old_in)
        w_adj_new_in = _wp_from_torch(adj_new_in)
        w_est = _wp_from_torch(est_flow.contiguous())
        w_adj_est = _wp_from_torch(adj_est_flow)
        w_adj_cbbm = _wp_from_torch(adj_cbbm_avg)

        w_eon = _wp_from_torch(emit_owner_new)
        w_eoo = _wp_from_torch(emit_owner_old)
        w_eupn = _wp_from_torch(emit_u_part_new)
        w_eupo = _wp_from_torch(emit_u_part_old)
        w_eppn = _wp_from_torch(emit_p_part_new)
        w_etpn = _wp_from_torch(emit_t_part_new)
        w_etpo = _wp_from_torch(emit_t_part_old)

        if self.n_flowout > 0:
            wp.launch(
                fo_bwd_k,
                dim=(T, self.n_flowout),
                inputs=[
                    w_new_in,
                    w_adj_new_out,
                    w["fo_off"],
                    w["fo_ghost"],
                    w["fo_owner"],
                    w["fo_nrm"],
                    w["fo_tgt"],
                    w["fo_asum"],
                    w_est,
                ],
                outputs=[w_adj_est],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        if self.n_cbbm > 0:
            wp.launch(
                cb_bwd_k,
                dim=(T, self.n_cbbm),
                inputs=[
                    w_adj_new_out,
                    w["cb_out_off"],
                    w["cb_out_ghost"],
                    w["cb_out_ghb"],
                    w["cb_out_ffp"],
                    w["cb_lag"],
                ],
                outputs=[w_adj_cbbm],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        if self.n_ghost > 0:
            wp.launch(
                emit_k,
                dim=(T, self.n_ghost),
                inputs=[
                    w_old_in,
                    w_new_in,
                    w_adj_old_out,
                    w_adj_new_out,
                    w["ghost_cpd"],
                    w["owner_cpd"],
                    w["nrm"],
                    w["u_code"],
                    w["u_rack_old"],
                    w["u_patch_slot"],
                    w["u_flow_tgt"],
                    w["u_rho_area"],
                    w_est,
                    w_adj_est,
                    w["p_code"],
                    w["ghb_ghost"],
                    w["t_code"],
                    self.energy,
                    include_old_u,
                    self.wzero,
                ],
                outputs=[
                    w_eon,
                    w_eoo,
                    w_eupn,
                    w_eupo,
                    w_eppn,
                    w_etpn,
                    w_etpo,
                    w_adj_old_in,
                    w_adj_new_in,
                ],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        wp.launch(
            gather_k,
            dim=(T, self.n_int),
            inputs=[
                w_adj_old_out,
                w_adj_new_out,
                w["cpd_of"],
                w["p_detach_owner"],
                w["own_off"],
                w["own_g"],
                w_eon,
                w_eoo,
                w["up_off"],
                w["up_g"],
                w_eupn,
                w_eupo,
                w["pp_off"],
                w["pp_g"],
                w_eppn,
                w["tp_off"],
                w["tp_g"],
                w_etpn,
                w_etpo,
                w["cb_in_off"],
                w["cb_in_slot"],
                w["cb_in_weight"],
                w_adj_cbbm,
                self.energy,
                self.wzero,
            ],
            outputs=[w_adj_old_in, w_adj_new_in],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        return adj_old_in, adj_new_in


class _WarpEnforceBCsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        phi_old_3d: torch.Tensor,
        phi_new_3d: torch.Tensor,
        bc_engine: WarpIncompressibleBCs,
        include_old_u: int,
    ):
        phi_old_out, phi_new_out, est_flow, _ = bc_engine.forward_pass(
            phi_old_3d, phi_new_3d, include_old_u=include_old_u
        )
        ctx.bc_engine = bc_engine
        ctx.include_old_u = include_old_u
        ctx.bc_refs_snap = dict(bc_engine._torch_refs)
        ctx.bc_w_snap = dict(bc_engine.w)
        ctx.save_for_backward(phi_old_3d, phi_new_3d, est_flow)
        return phi_old_out, phi_new_out

    @staticmethod
    def backward(ctx, grad_old_out: Optional[torch.Tensor], grad_new_out: Optional[torch.Tensor]):
        phi_old_3d, phi_new_3d, est_flow = ctx.saved_tensors
        bc_engine: WarpIncompressibleBCs = ctx.bc_engine
        if grad_old_out is None:
            grad_old_out = torch.zeros_like(phi_old_3d)
        if grad_new_out is None:
            grad_new_out = torch.zeros_like(phi_new_3d)
        grad_old_in, grad_new_in = bc_engine.backward_pass(
            phi_old_3d,
            phi_new_3d,
            est_flow,
            grad_old_out,
            grad_new_out,
            include_old_u=ctx.include_old_u,
            w_override=ctx.bc_w_snap,
        )
        return (
            grad_old_in if ctx.needs_input_grad[0] else None,
            grad_new_in if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


def warp_enforce_bcs(
    bc_engine: WarpIncompressibleBCs,
    phi_old_cpd_cell: torch.Tensor,
    phi_new_cpd_cell: torch.Tensor,
    include_old_u: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Warp BC enforcement preserving 2D ``[N_cpd, C]`` or 3D ``[T, N_cpd, C]`` rank."""
    is_2d = phi_new_cpd_cell.ndim == 2
    old_3d = phi_old_cpd_cell.unsqueeze(0) if is_2d else phi_old_cpd_cell
    new_3d = phi_new_cpd_cell.unsqueeze(0) if is_2d else phi_new_cpd_cell
    old_out, new_out = _WarpEnforceBCsFn.apply(old_3d, new_3d, bc_engine, int(include_old_u))
    if is_2d:
        return old_out.squeeze(0), new_out.squeeze(0)
    return old_out, new_out
