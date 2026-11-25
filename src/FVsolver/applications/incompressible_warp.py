"""Unified differentiable NVIDIA Warp FVM engine for incompressible and Boussinesq solvers (``picoext``, ``picocmpt``, ``buoyantBoussinesqpicoext``, ``buoyantBoussinesqpicocmpt``)."""
import os
from typing import Any, Optional, Tuple

import torch
from torch_geometric.nn import global_add_pool
import warp as wp


def _wp_from_torch(t: torch.Tensor, dtype=None):
    return wp.from_torch(t, dtype=dtype, requires_grad=False)

from FVdomain.SetBC.incompressible_bc_warp import (
    G_INLETOUTLET_U,
    G_NONE,
    G_RACK_INLET_U,
    G_SYMMETRY_U,
    G_ZERO,
    U_ADVECTIVE,
    U_INLETOUTLET,
    WarpIncompressibleBCs,
    bc_structure_signature,
    warp_enforce_bcs,
)
from FVdomain.SetBC.onthefly.BCbase import VARType

wp.config.quiet = True
wp.set_module_options({"enable_backward": False})

INTERP_LINEAR = 0
INTERP_LINEAR_CORR = 1
INTERP_LINEAR_GRAD = 2

CONV_LINEAR = 0
CONV_LINEAR_CORR = 1
CONV_LINEAR_GRAD = 2
CONV_UPWIND = 3
CONV_LINEAR_UPWIND = 4

SNGRAD_CORRECTED = 0
SNGRAD_UNCORRECTED = 1
SNGRAD_ORTHOGONAL = 2


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


def _make_fvm_kernels(scalar_t, vec3_t):
    @wp.func
    def dot3(a: vec3_t, b: vec3_t):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @wp.func
    def grad_at_vec(grad: wp.array4d(dtype=scalar_t), t: int, cpd: int, ch: int):
        return vec3_t(grad[t, cpd, ch, 0], grad[t, cpd, ch, 1], grad[t, cpd, ch, 2])

    @wp.func
    def eval_sngrad_f(
        q: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        t: int,
        o: int,
        n: int,
        ch: int,
        sngrad_mode: int,
        delta_c: scalar_t,
        nonorth: scalar_t,
        ow: scalar_t,
        nw: scalar_t,
        cv: vec3_t,
    ):
        dq = q[t, n, ch] - q[t, o, ch]
        if sngrad_mode == SNGRAD_ORTHOGONAL:
            return dq * delta_c
        base = dq * nonorth
        if sngrad_mode == SNGRAD_UNCORRECTED:
            return base
        go = grad_at_vec(grad, t, o, ch)
        gn = grad_at_vec(grad, t, n, ch)
        corr = (ow * go[0] + nw * gn[0]) * cv[0] + (ow * go[1] + nw * gn[1]) * cv[1] + (ow * go[2] + nw * gn[2]) * cv[2]
        return base + corr

    @wp.func
    def eval_interp_val(
        q: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        t: int,
        o: int,
        n: int,
        ch: int,
        mode: int,
        gc: scalar_t,
        gf: scalar_t,
        rcorr: vec3_t,
        cf: vec3_t,
        ff: vec3_t,
        is_int_f: int,
    ):
        v = gc * q[t, o, ch] + gf * q[t, n, ch]
        if mode == INTERP_LINEAR_CORR:
            go = grad_at_vec(grad, t, o, ch)
            gn = grad_at_vec(grad, t, n, ch)
            v = v + (gc * go[0] + gf * gn[0]) * rcorr[0] + (gc * go[1] + gf * gn[1]) * rcorr[1] + (gc * go[2] + gf * gn[2]) * rcorr[2]
        elif mode == INTERP_LINEAR_GRAD and is_int_f == 1:
            go = grad_at_vec(grad, t, o, ch)
            gn = grad_at_vec(grad, t, n, ch)
            v = v + gc * dot3(go, cf) + gf * dot3(gn, ff)
        return v

    @wp.func
    def eval_conv_val(
        q: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        t: int,
        o: int,
        n: int,
        ch: int,
        flux: scalar_t,
        mode: int,
        gc: scalar_t,
        gf: scalar_t,
        rcorr: vec3_t,
        cf: vec3_t,
        ff: vec3_t,
        is_int_f: int,
        bnc: int,
    ):
        if mode <= CONV_LINEAR_GRAD:
            return eval_interp_val(q, grad, t, o, n, ch, mode, gc, gf, rcorr, cf, ff, is_int_f)
        if bnc == 1:
            return q[t, n, ch]
        if mode == CONV_UPWIND:
            if flux > scalar_t(0.0):
                return q[t, o, ch]
            else:
                return q[t, n, ch]
        if flux > scalar_t(0.0):
            go = grad_at_vec(grad, t, o, ch)
            return q[t, o, ch] + dot3(go, cf)
        else:
            gn = grad_at_vec(grad, t, n, ch)
            return q[t, n, ch] + dot3(gn, ff)

    @wp.func
    def eval_face_grad_flux(
        q: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        t: int,
        o: int,
        n: int,
        ch: int,
        gc: scalar_t,
        gf: scalar_t,
        ecf: vec3_t,
        inv_dcf: scalar_t,
        apply_corr: int,
        nv: vec3_t,
    ):
        go = grad_at_vec(grad, t, o, ch)
        gn = grad_at_vec(grad, t, n, ch)
        gx = gc * go[0] + gf * gn[0]
        gy = gc * go[1] + gf * gn[1]
        gz = gc * go[2] + gf * gn[2]
        if apply_corr == 1:
            corr = (q[t, n, ch] - q[t, o, ch]) * inv_dcf - (gx * ecf[0] + gy * ecf[1] + gz * ecf[2])
            gx = gx + corr * ecf[0]
            gy = gy + corr * ecf[1]
            gz = gz + corr * ecf[2]
        return gx * nv[0] + gy * nv[1] + gz * nv[2]


    @wp.kernel
    def pack_q_kernel(
        phi_old: wp.array3d(dtype=scalar_t),
        phi_new: wp.array3d(dtype=scalar_t),
        energy: int,
        q_p1: wp.array3d(dtype=scalar_t),
        q_p2: wp.array3d(dtype=scalar_t),
        phi_new_p2: wp.array3d(dtype=scalar_t),
    ):
        t, cpd = wp.tid()
        u0 = phi_new[t, cpd, 0]
        u1 = phi_new[t, cpd, 1]
        u2 = phi_new[t, cpd, 2]
        uo0 = phi_old[t, cpd, 0]
        uo1 = phi_old[t, cpd, 1]
        uo2 = phi_old[t, cpd, 2]
        p = phi_new[t, cpd, 3]

        q_p1[t, cpd, 0] = u0
        q_p1[t, cpd, 1] = u1
        q_p1[t, cpd, 2] = u2
        q_p1[t, cpd, 3] = uo0
        q_p1[t, cpd, 4] = uo1
        q_p1[t, cpd, 5] = uo2
        q_p1[t, cpd, 6] = p

        q_p2[t, cpd, 0] = u0
        q_p2[t, cpd, 1] = u1
        q_p2[t, cpd, 2] = u2
        q_p2[t, cpd, 3] = uo0
        q_p2[t, cpd, 4] = uo1
        q_p2[t, cpd, 5] = uo2
        q_p2[t, cpd, 6] = p

        phi_new_p2[t, cpd, 0] = u0
        phi_new_p2[t, cpd, 1] = u1
        phi_new_p2[t, cpd, 2] = u2
        phi_new_p2[t, cpd, 3] = p

        if energy == 1:
            tn = phi_new[t, cpd, 4]
            to = phi_old[t, cpd, 4]
            q_p1[t, cpd, 7] = tn
            q_p1[t, cpd, 8] = to
            q_p2[t, cpd, 7] = tn
            q_p2[t, cpd, 8] = to
            phi_new_p2[t, cpd, 4] = tn

    @wp.kernel
    def wlsq_fwd_kernel(
        q: wp.array3d(dtype=scalar_t),
        cpd_of: wp.array(dtype=wp.int32),
        w_off: wp.array(dtype=wp.int32),
        w_other: wp.array(dtype=wp.int32),
        w_vec: wp.array(dtype=vec3_t),
        grad: wp.array4d(dtype=scalar_t),
    ):
        t, i, ch = wp.tid()
        ci = cpd_of[i]
        qi = q[t, ci, ch]
        gx = scalar_t(0.0)
        gy = scalar_t(0.0)
        gz = scalar_t(0.0)
        for k in range(w_off[i], w_off[i + 1]):
            d = q[t, w_other[k], ch] - qi
            wk = w_vec[k]
            gx += wk[0] * d
            gy += wk[1] * d
            gz += wk[2] * d
        grad[t, ci, ch, 0] = gx
        grad[t, ci, ch, 1] = gy
        grad[t, ci, ch, 2] = gz

    @wp.kernel
    def ghost_grad_fwd_kernel(
        ghost_cpd: wp.array(dtype=wp.int32),
        cyc_partner: wp.array(dtype=wp.int32),
        grad: wp.array4d(dtype=scalar_t),
    ):
        t, g, ch = wp.tid()
        gi = ghost_cpd[g]
        pi = cyc_partner[g]
        if pi >= 0:
            grad[t, gi, ch, 0] = grad[t, pi, ch, 0]
            grad[t, gi, ch, 1] = grad[t, pi, ch, 1]
            grad[t, gi, ch, 2] = grad[t, pi, ch, 2]
        else:
            grad[t, gi, ch, 0] = scalar_t(0.0)
            grad[t, gi, ch, 1] = scalar_t(0.0)
            grad[t, gi, ch, 2] = scalar_t(0.0)

    @wp.kernel
    def face_pre_kernel(
        q_p1: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        fo: wp.array(dtype=wp.int32),
        fn: wp.array(dtype=wp.int32),
        gC: wp.array(dtype=scalar_t),
        gF: wp.array(dtype=scalar_t),
        rcorr: wp.array(dtype=vec3_t),
        Cf: wp.array(dtype=vec3_t),
        Ff: wp.array(dtype=vec3_t),
        nrm: wp.array(dtype=vec3_t),
        is_int_face: wp.array(dtype=wp.int32),
        interp_mode: int,
        f0_new: wp.array2d(dtype=scalar_t),
        f0_old: wp.array2d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        o = fo[f]
        n = fn[f]
        gc = gC[f]
        gf = gF[f]
        rc = rcorr[f]
        cf = Cf[f]
        ff = Ff[f]
        nv = nrm[f]
        iif = is_int_face[f]

        u0 = eval_interp_val(q_p1, grad, t, o, n, 0, interp_mode, gc, gf, rc, cf, ff, iif)
        u1 = eval_interp_val(q_p1, grad, t, o, n, 1, interp_mode, gc, gf, rc, cf, ff, iif)
        u2 = eval_interp_val(q_p1, grad, t, o, n, 2, interp_mode, gc, gf, rc, cf, ff, iif)

        uo0 = eval_interp_val(q_p1, grad, t, o, n, 3, interp_mode, gc, gf, rc, cf, ff, iif)
        uo1 = eval_interp_val(q_p1, grad, t, o, n, 4, interp_mode, gc, gf, rc, cf, ff, iif)
        uo2 = eval_interp_val(q_p1, grad, t, o, n, 5, interp_mode, gc, gf, rc, cf, ff, iif)

        f0_new[t, f] = u0 * nv[0] + u1 * nv[1] + u2 * nv[2]
        f0_old[t, f] = uo0 * nv[0] + uo1 * nv[1] + uo2 * nv[2]

    @wp.kernel
    def mwi_ap_kernel(
        f0_new: wp.array2d(dtype=scalar_t),
        cpd_of: wp.array(dtype=wp.int32),
        c_off: wp.array(dtype=wp.int32),
        c_face: wp.array(dtype=wp.int32),
        c_sarea: wp.array(dtype=scalar_t),
        c_visc_ap: wp.array(dtype=scalar_t),
        vol: wp.array(dtype=scalar_t),
        ap_int: wp.array2d(dtype=scalar_t),
        d_cpd: wp.array2d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ap = scalar_t(0.0)
        for k in range(c_off[i], c_off[i + 1]):
            outw = f0_new[t, c_face[k]] * c_sarea[k]
            if outw > scalar_t(0.0):
                ap += outw
            ap += c_visc_ap[k]
        ap_int[t, i] = ap
        d_cpd[t, cpd_of[i]] = vol[i] / ap

    @wp.kernel
    def mwi_dghost_kernel(
        ghost_cpd: wp.array(dtype=wp.int32),
        cyc_partner: wp.array(dtype=wp.int32),
        d_cpd: wp.array2d(dtype=scalar_t),
    ):
        t, g = wp.tid()
        gi = ghost_cpd[g]
        pi = cyc_partner[g]
        if pi >= 0:
            d_cpd[t, gi] = d_cpd[t, pi]
        else:
            d_cpd[t, gi] = scalar_t(0.0)

    @wp.kernel
    def pass2_ghost_kernel(
        q_p1: wp.array3d(dtype=scalar_t),
        f0_new: wp.array2d(dtype=scalar_t),
        ghost_cpd: wp.array(dtype=wp.int32),
        owner_cpd: wp.array(dtype=wp.int32),
        ghost_face: wp.array(dtype=wp.int32),
        nrm_ghost: wp.array(dtype=vec3_t),
        u_code: wp.array(dtype=wp.int32),
        u_val: wp.array(dtype=vec3_t),
        adv_dt_dx: wp.array(dtype=scalar_t),
        adv_floor: wp.array(dtype=scalar_t),
        include_old_p1: int,
        wzero: int,
        q_p2: wp.array3d(dtype=scalar_t),
        phi_new_p2: wp.array3d(dtype=scalar_t),
    ):
        t, g = wp.tid()
        uc = u_code[g]
        if uc != U_ADVECTIVE and uc != U_INLETOUTLET:
            return
        gi = ghost_cpd[g]
        oi = owner_cpd[g]
        if uc == U_ADVECTIVE and include_old_p1 == 0:
            n = nrm_ghost[g]
            uoo0 = q_p1[t, oi, 3]
            uoo1 = q_p1[t, oi, 4]
            uoo2 = scalar_t(0.0) if wzero == 1 else q_p1[t, oi, 5]
            ugo0 = q_p1[t, gi, 3]
            ugo1 = q_p1[t, gi, 4]
            ugo2 = scalar_t(0.0) if wzero == 1 else q_p1[t, gi, 5]
            un_pos = wp.max(uoo0 * n[0] + uoo1 * n[1] + uoo2 * n[2], scalar_t(0.0))
            alpha = wp.clamp(un_pos * adv_dt_dx[g], adv_floor[g], scalar_t(1.0))
            ug0 = ugo0 - alpha * (ugo0 - uoo0)
            ug1 = ugo1 - alpha * (ugo1 - uoo1)
            ug2 = scalar_t(0.0) if wzero == 1 else (ugo2 - alpha * (ugo2 - uoo2))
            q_p2[t, gi, 0] = ug0
            q_p2[t, gi, 1] = ug1
            q_p2[t, gi, 2] = ug2
            phi_new_p2[t, gi, 0] = ug0
            phi_new_p2[t, gi, 1] = ug1
            phi_new_p2[t, gi, 2] = ug2
        elif uc == U_INLETOUTLET:
            bf = ghost_face[g]
            fl = f0_new[t, bf]
            uv = u_val[g]
            ug0 = uv[0]
            ug1 = uv[1]
            ug2 = scalar_t(0.0) if wzero == 1 else uv[2]
            if fl >= scalar_t(0.0):
                ug0 = q_p1[t, oi, 0]
                ug1 = q_p1[t, oi, 1]
                ug2 = scalar_t(0.0) if wzero == 1 else q_p1[t, oi, 2]
            q_p2[t, gi, 0] = ug0
            q_p2[t, gi, 1] = ug1
            q_p2[t, gi, 2] = ug2
            phi_new_p2[t, gi, 0] = ug0
            phi_new_p2[t, gi, 1] = ug1
            phi_new_p2[t, gi, 2] = ug2

    @wp.kernel
    def face_main_kernel(
        q_p1: wp.array3d(dtype=scalar_t),
        q_p2: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        f0_new: wp.array2d(dtype=scalar_t),
        f0_old: wp.array2d(dtype=scalar_t),
        d_cpd: wp.array2d(dtype=scalar_t),
        fo: wp.array(dtype=wp.int32),
        fn: wp.array(dtype=wp.int32),
        gC: wp.array(dtype=scalar_t),
        gF: wp.array(dtype=scalar_t),
        nrm: wp.array(dtype=vec3_t),
        rcorr: wp.array(dtype=vec3_t),
        Cf: wp.array(dtype=vec3_t),
        Ff: wp.array(dtype=vec3_t),
        eCF: wp.array(dtype=vec3_t),
        inv_dCF: wp.array(dtype=scalar_t),
        delta_c: wp.array(dtype=scalar_t),
        nonorth: wp.array(dtype=scalar_t),
        ow_u: wp.array(dtype=scalar_t),
        nw_u: wp.array(dtype=scalar_t),
        cv_u: wp.array(dtype=vec3_t),
        ow_p: wp.array(dtype=scalar_t),
        nw_p: wp.array(dtype=scalar_t),
        cv_p: wp.array(dtype=vec3_t),
        ow_t: wp.array(dtype=scalar_t),
        nw_t: wp.array(dtype=scalar_t),
        cv_t: wp.array(dtype=vec3_t),
        is_int_face: wp.array(dtype=wp.int32),
        bnc: wp.array(dtype=wp.int32),
        grad_interp_apply: wp.array(dtype=wp.int32),
        mwi_mask: wp.array(dtype=wp.int32),
        bc_diff_mask: wp.array(dtype=wp.int32),
        zero_flux: wp.array(dtype=wp.int32),
        grad_u_bc_code: wp.array(dtype=wp.int32),
        grad_t_bc_code: wp.array(dtype=wp.int32),
        sym_delta_coeff: wp.array(dtype=scalar_t),
        ghb_face: wp.array(dtype=scalar_t),
        gpc_face: wp.array(dtype=scalar_t),
        cc_face: wp.array(dtype=scalar_t),
        ccont_face: wp.array(dtype=scalar_t),
        nu_face: wp.array(dtype=scalar_t),
        alpha_face: wp.array(dtype=scalar_t),
        energy: int,
        is_cmpt: int,
        use_mwi: int,
        sngrad_mode: int,
        conv_u_mode: int,
        conv_t_mode: int,
        theta_conv: scalar_t,
        facecv_face_buoy: int,
        fvol_pre: wp.array2d(dtype=scalar_t),
        flux_new_bc: wp.array2d(dtype=scalar_t),
        flux_old_bc: wp.array2d(dtype=scalar_t),
        mwi_br: wp.array2d(dtype=scalar_t),
        fmass: wp.array2d(dtype=scalar_t),
        fmom: wp.array3d(dtype=scalar_t),
        fen: wp.array2d(dtype=scalar_t),
        sngrad_p: wp.array2d(dtype=scalar_t),
        sngrad_t: wp.array2d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        o = fo[f]
        n = fn[f]
        gc = gC[f]
        gf = gF[f]
        nv = nrm[f]
        rc = rcorr[f]
        cf = Cf[f]
        ff = Ff[f]
        iif = is_int_face[f]
        bf = bnc[f]
        dc = delta_c[f]
        no = nonorth[f]

        f_pre = f0_new[t, f]
        br = scalar_t(0.0)
        if use_mwi == 1 and mwi_mask[f] == 1:
            snp = eval_sngrad_f(q_p1, grad, t, o, n, 6, sngrad_mode, dc, no, ow_p[f], nw_p[f], cv_p[f])
            go_p = grad_at_vec(grad, t, o, 6)
            gn_p = grad_at_vec(grad, t, n, 6)
            interp_p = (gc * go_p[0] + gf * gn_p[0]) * nv[0] + (gc * go_p[1] + gf * gn_p[1]) * nv[1] + (gc * go_p[2] + gf * gn_p[2]) * nv[2]
            br = interp_p - snp
            if energy == 1:
                snt = eval_sngrad_f(q_p1, grad, t, o, n, 7, sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f])
                go_t = grad_at_vec(grad, t, o, 7)
                gn_t = grad_at_vec(grad, t, n, 7)
                interp_t = (gc * go_t[0] + gf * gn_t[0]) * nv[0] + (gc * go_t[1] + gf * gn_t[1]) * nv[1] + (gc * go_t[2] + gf * gn_t[2]) * nv[2]
                br = br + ghb_face[f] * (snt - interp_t)
            df = scalar_t(0.5) * (d_cpd[t, o] + d_cpd[t, n])
            f_pre = f_pre + gpc_face[f] * df * br

        mwi_br[t, f] = br
        fvol_pre[t, f] = f_pre

        fl_new = scalar_t(0.0) if zero_flux[f] == 1 else f_pre
        fl_old = scalar_t(0.0) if zero_flux[f] == 1 else f0_old[t, f]
        flux_new_bc[t, f] = fl_new
        flux_old_bc[t, f] = fl_old
        fmass[t, f] = ccont_face[f] * fl_new

        cc = cc_face[f]
        one_m_th = scalar_t(1.0) - theta_conv
        bcd = bc_diff_mask[f]
        g_uc = grad_u_bc_code[f]

        gu0 = scalar_t(0.0)
        gu1 = scalar_t(0.0)
        gu2 = scalar_t(0.0)
        if is_cmpt == 0 and bcd == 0:
            ecf = eCF[f]
            idcf = inv_dCF[f]
            gia = grad_interp_apply[f]
            gu0 = eval_face_grad_flux(q_p1, grad, t, o, n, 0, gc, gf, ecf, idcf, gia, nv)
            gu1 = eval_face_grad_flux(q_p1, grad, t, o, n, 1, gc, gf, ecf, idcf, gia, nv)
            gu2 = eval_face_grad_flux(q_p1, grad, t, o, n, 2, gc, gf, ecf, idcf, gia, nv)
            if g_uc == G_ZERO:
                gu0 = scalar_t(0.0)
                gu1 = scalar_t(0.0)
                gu2 = scalar_t(0.0)
            elif g_uc == G_SYMMETRY_U:
                un_o = q_p2[t, o, 0] * nv[0] + q_p2[t, o, 1] * nv[1] + q_p2[t, o, 2] * nv[2]
                sdc = -sym_delta_coeff[f] * un_o
                gu0 = sdc * nv[0]
                gu1 = sdc * nv[1]
                gu2 = sdc * nv[2]
            elif g_uc == G_RACK_INLET_U:
                gn_proj = gu0 * nv[0] + gu1 * nv[1] + gu2 * nv[2]
                gu0 = gn_proj * nv[0]
                gu1 = gn_proj * nv[1]
                gu2 = gn_proj * nv[2]
            elif g_uc == G_INLETOUTLET_U:
                if fl_new >= scalar_t(0.0):
                    gu0 = scalar_t(0.0)
                    gu1 = scalar_t(0.0)
                    gu2 = scalar_t(0.0)

        for c in range(3):
            uc_new = eval_conv_val(q_p2, grad, t, o, n, c, fl_new, conv_u_mode, gc, gf, rc, cf, ff, iif, bf)
            conv_u = theta_conv * (cc * fl_new * uc_new)
            if theta_conv != scalar_t(1.0):
                uc_old = eval_conv_val(q_p2, grad, t, o, n, 3 + c, fl_old, conv_u_mode, gc, gf, rc, cf, ff, iif, bf)
                conv_u = conv_u + one_m_th * (cc * fl_old * uc_old)

            gfl_u = scalar_t(0.0)
            if is_cmpt == 1 or bcd == 1:
                gfl_u = eval_sngrad_f(q_p2, grad, t, o, n, c, sngrad_mode, dc, no, ow_u[f], nw_u[f], cv_u[f])
            else:
                if c == 0:
                    gfl_u = gu0
                elif c == 1:
                    gfl_u = gu1
                else:
                    gfl_u = gu2
            fmom[t, f, c] = conv_u - nu_face[f] * gfl_u

        sngrad_p[t, f] = eval_sngrad_f(q_p2, grad, t, o, n, 6, sngrad_mode, dc, no, ow_p[f], nw_p[f], cv_p[f])
        if energy == 1 and facecv_face_buoy == 1:
            sngrad_t[t, f] = eval_sngrad_f(q_p2, grad, t, o, n, 7, sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f])

        if energy == 1:
            tc_new = eval_conv_val(q_p2, grad, t, o, n, 7, fl_new, conv_t_mode, gc, gf, rc, cf, ff, iif, bf)
            conv_t = theta_conv * (cc * fl_new * tc_new)
            if theta_conv != scalar_t(1.0):
                tc_old = eval_conv_val(q_p2, grad, t, o, n, 8, fl_old, conv_t_mode, gc, gf, rc, cf, ff, iif, bf)
                conv_t = conv_t + one_m_th * (cc * fl_old * tc_old)

            gfl_t = scalar_t(0.0)
            if is_cmpt == 1 or bcd == 1:
                gfl_t = eval_sngrad_f(q_p2, grad, t, o, n, 7, sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f])
            else:
                gfl_t = eval_face_grad_flux(q_p1, grad, t, o, n, 7, gc, gf, eCF[f], inv_dCF[f], grad_interp_apply[f], nv)
                if grad_t_bc_code[f] == G_ZERO:
                    gfl_t = scalar_t(0.0)
            fen[t, f] = conv_t - alpha_face[f] * gfl_t

    @wp.kernel
    def cell_balance_kernel(
        q_p2: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        flux_new_bc: wp.array2d(dtype=scalar_t),
        flux_old_bc: wp.array2d(dtype=scalar_t),
        fmass: wp.array2d(dtype=scalar_t),
        fmom: wp.array3d(dtype=scalar_t),
        fen: wp.array2d(dtype=scalar_t),
        cpd_of: wp.array(dtype=wp.int32),
        c_off: wp.array(dtype=wp.int32),
        c_face: wp.array(dtype=wp.int32),
        c_sarea: wp.array(dtype=scalar_t),
        c_all_off: wp.array(dtype=wp.int32),
        c_all_face: wp.array(dtype=wp.int32),
        c_all_sarea: wp.array(dtype=scalar_t),
        vol: wp.array(dtype=scalar_t),
        tcoef: wp.array(dtype=scalar_t),
        pcoef_vol: wp.array(dtype=scalar_t),
        ghb_vol: wp.array(dtype=scalar_t),
        src_u_base: wp.array2d(dtype=scalar_t),
        src_t_base: wp.array(dtype=scalar_t),
        energy: int,
        mom_cell_active: int,
        bounded_u: int,
        bounded_t: int,
        theta_conv: scalar_t,
        facecv_face_buoy: int,
        div_phi_new_out: wp.array2d(dtype=scalar_t),
        div_phi_old_out: wp.array2d(dtype=scalar_t),
        res_cont: wp.array2d(dtype=scalar_t),
        res_mom_pfree: wp.array3d(dtype=scalar_t),
        res_mom: wp.array3d(dtype=scalar_t),
        res_en: wp.array2d(dtype=scalar_t),
        rhat_cpd: wp.array3d(dtype=scalar_t),
        scale_cont: wp.array2d(dtype=scalar_t),
        scale_mom: wp.array3d(dtype=scalar_t),
        scale_en: wp.array2d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ci = cpd_of[i]
        rc = scalar_t(0.0)
        rm0 = scalar_t(0.0)
        rm1 = scalar_t(0.0)
        rm2 = scalar_t(0.0)
        re = scalar_t(0.0)

        sc_c = scalar_t(0.0)
        sc_m0 = scalar_t(0.0)
        sc_m1 = scalar_t(0.0)
        sc_m2 = scalar_t(0.0)
        sc_e = scalar_t(0.0)

        for k in range(c_off[i], c_off[i + 1]):
            f = c_face[k]
            s = c_sarea[k]
            mc = s * fmass[t, f]
            mm0 = s * fmom[t, f, 0]
            mm1 = s * fmom[t, f, 1]
            mm2 = s * fmom[t, f, 2]
            rc += mc
            rm0 += mm0
            rm1 += mm1
            rm2 += mm2
            sc_c += wp.abs(mc)
            sc_m0 += wp.abs(mm0)
            sc_m1 += wp.abs(mm1)
            sc_m2 += wp.abs(mm2)
            if energy == 1:
                me = s * fen[t, f]
                re += me
                sc_e += wp.abs(me)

        div_new = scalar_t(0.0)
        div_old = scalar_t(0.0)
        if bounded_u == 1 or bounded_t == 1:
            for k in range(c_all_off[i], c_all_off[i + 1]):
                fa = c_all_face[k]
                sa = c_all_sarea[k]
                div_new += sa * flux_new_bc[t, fa]
                if theta_conv != scalar_t(1.0):
                    div_old += sa * flux_old_bc[t, fa]
        div_phi_new_out[t, i] = div_new
        div_phi_old_out[t, i] = div_old

        one_m_th = scalar_t(1.0) - theta_conv
        bnd_u0 = scalar_t(0.0)
        bnd_u1 = scalar_t(0.0)
        bnd_u2 = scalar_t(0.0)
        if bounded_u == 1:
            bnd_u0 = -theta_conv * div_new * q_p2[t, ci, 0]
            bnd_u1 = -theta_conv * div_new * q_p2[t, ci, 1]
            bnd_u2 = -theta_conv * div_new * q_p2[t, ci, 2]
            if theta_conv != scalar_t(1.0):
                bnd_u0 -= one_m_th * div_old * q_p2[t, ci, 3]
                bnd_u1 -= one_m_th * div_old * q_p2[t, ci, 4]
                bnd_u2 -= one_m_th * div_old * q_p2[t, ci, 5]

        tc = tcoef[i]
        uns0 = tc * q_p2[t, ci, 0]
        uns1 = tc * q_p2[t, ci, 1]
        uns2 = tc * q_p2[t, ci, 2]
        src0 = src_u_base[i, 0] + tc * q_p2[t, ci, 3]
        src1 = src_u_base[i, 1] + tc * q_p2[t, ci, 4]
        src2 = src_u_base[i, 2] + tc * q_p2[t, ci, 5]

        buoy0 = scalar_t(0.0)
        buoy1 = scalar_t(0.0)
        buoy2 = scalar_t(0.0)
        if energy == 1:
            gbv = ghb_vol[i]
            buoy0 = gbv * grad[t, ci, 7, 0]
            buoy1 = gbv * grad[t, ci, 7, 1]
            buoy2 = gbv * grad[t, ci, 7, 2]

        kin0 = uns0 + rm0 - src0 + bnd_u0
        kin1 = uns1 + rm1 - src1 + bnd_u1
        kin2 = uns2 + rm2 - src2 + bnd_u2
        pf0 = kin0 - buoy0
        pf1 = kin1 - buoy1
        pf2 = kin2 - buoy2

        res_mom_pfree[t, i, 0] = pf0
        res_mom_pfree[t, i, 1] = pf1
        res_mom_pfree[t, i, 2] = pf2

        inv_v = scalar_t(1.0) / vol[i]
        if facecv_face_buoy == 1:
            rhat_cpd[t, ci, 0] = kin0 * inv_v
            rhat_cpd[t, ci, 1] = kin1 * inv_v
            rhat_cpd[t, ci, 2] = kin2 * inv_v
        else:
            rhat_cpd[t, ci, 0] = pf0 * inv_v
            rhat_cpd[t, ci, 1] = pf1 * inv_v
            rhat_cpd[t, ci, 2] = pf2 * inv_v

        if mom_cell_active == 1:
            pcv = pcoef_vol[i]
            res_mom[t, i, 0] = pf0 + pcv * grad[t, ci, 6, 0]
            res_mom[t, i, 1] = pf1 + pcv * grad[t, ci, 6, 1]
            res_mom[t, i, 2] = pf2 + pcv * grad[t, ci, 6, 2]
        else:
            res_mom[t, i, 0] = pf0
            res_mom[t, i, 1] = pf1
            res_mom[t, i, 2] = pf2

        res_cont[t, i] = rc
        scale_cont[t, i] = sc_c
        scale_mom[t, i, 0] = sc_m0 + wp.abs(uns0) + wp.abs(src0) + wp.abs(buoy0)
        scale_mom[t, i, 1] = sc_m1 + wp.abs(uns1) + wp.abs(src1) + wp.abs(buoy1)
        scale_mom[t, i, 2] = sc_m2 + wp.abs(uns2) + wp.abs(src2) + wp.abs(buoy2)

        if energy == 1:
            uns_t = tc * q_p2[t, ci, 7]
            src_t = src_t_base[i] + tc * q_p2[t, ci, 8]
            bnd_t = scalar_t(0.0)
            if bounded_t == 1:
                bnd_t = -theta_conv * div_new * q_p2[t, ci, 7]
                if theta_conv != scalar_t(1.0):
                    bnd_t -= one_m_th * div_old * q_p2[t, ci, 8]
            res_en[t, i] = uns_t + re - src_t + bnd_t
            scale_en[t, i] = sc_e + wp.abs(uns_t) + wp.abs(src_t)

    @wp.kernel
    def rhat_ghost_kernel(
        ghost_cpd: wp.array(dtype=wp.int32),
        cyc_partner: wp.array(dtype=wp.int32),
        rhat_cpd: wp.array3d(dtype=scalar_t),
    ):
        t, g = wp.tid()
        gi = ghost_cpd[g]
        pi = cyc_partner[g]
        if pi >= 0:
            rhat_cpd[t, gi, 0] = rhat_cpd[t, pi, 0]
            rhat_cpd[t, gi, 1] = rhat_cpd[t, pi, 1]
            rhat_cpd[t, gi, 2] = rhat_cpd[t, pi, 2]
        else:
            rhat_cpd[t, gi, 0] = scalar_t(0.0)
            rhat_cpd[t, gi, 1] = scalar_t(0.0)
            rhat_cpd[t, gi, 2] = scalar_t(0.0)

    @wp.kernel
    def face_cv_fwd_kernel(
        rhat_cpd: wp.array3d(dtype=scalar_t),
        sngrad_p: wp.array2d(dtype=scalar_t),
        sngrad_t: wp.array2d(dtype=scalar_t),
        f0_new: wp.array2d(dtype=scalar_t),
        flux_new_bc: wp.array2d(dtype=scalar_t),
        fo: wp.array(dtype=wp.int32),
        fn: wp.array(dtype=wp.int32),
        nrm: wp.array(dtype=vec3_t),
        gpc_face: wp.array(dtype=scalar_t),
        ghb_face: wp.array(dtype=scalar_t),
        vface: wp.array(dtype=scalar_t),
        twosided: wp.array(dtype=wp.int32),
        active_face: wp.array(dtype=wp.int32),
        facecv_upwind: int,
        upwind_uses_flux_bc: int,
        facecv_face_buoy: int,
        res_mom_face: wp.array2d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        if active_face[f] == 0:
            res_mom_face[t, f] = scalar_t(0.0)
            return
        o = fo[f]
        n = fn[f]
        nv = nrm[f]
        rc0 = rhat_cpd[t, o, 0]
        rc1 = rhat_cpd[t, o, 1]
        rc2 = rhat_cpd[t, o, 2]
        rf0 = rc0
        rf1 = rc1
        rf2 = rc2
        if twosided[f] == 1:
            rn0 = rhat_cpd[t, n, 0]
            rn1 = rhat_cpd[t, n, 1]
            rn2 = rhat_cpd[t, n, 2]
            if facecv_upwind == 1:
                sel_fl = flux_new_bc[t, f] if upwind_uses_flux_bc == 1 else f0_new[t, f]
                if sel_fl < scalar_t(0.0):
                    rf0 = rn0
                    rf1 = rn1
                    rf2 = rn2
            else:
                rf0 = scalar_t(0.5) * (rc0 + rn0)
                rf1 = scalar_t(0.5) * (rc1 + rn1)
                rf2 = scalar_t(0.5) * (rc2 + rn2)
        dot_r = nv[0] * rf0 + nv[1] * rf1 + nv[2] * rf2
        val_f = dot_r + gpc_face[f] * sngrad_p[t, f]
        if facecv_face_buoy == 1:
            val_f = val_f - ghb_face[f] * sngrad_t[t, f]
        res_mom_face[t, f] = val_f * vface[f]


    @wp.kernel
    def face_cv_bwd_emit_kernel(
        adj_res_mom_face: wp.array2d(dtype=scalar_t),
        f0_new: wp.array2d(dtype=scalar_t),
        flux_new_bc: wp.array2d(dtype=scalar_t),
        nrm: wp.array(dtype=vec3_t),
        gpc_face: wp.array(dtype=scalar_t),
        ghb_face: wp.array(dtype=scalar_t),
        vface: wp.array(dtype=scalar_t),
        twosided: wp.array(dtype=wp.int32),
        active_face: wp.array(dtype=wp.int32),
        facecv_upwind: int,
        upwind_uses_flux_bc: int,
        facecv_face_buoy: int,
        adj_sngrad_p: wp.array2d(dtype=scalar_t),
        adj_sngrad_t: wp.array2d(dtype=scalar_t),
        emit_rhat_o: wp.array3d(dtype=scalar_t),
        emit_rhat_n: wp.array3d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        if active_face[f] == 0:
            adj_sngrad_p[t, f] = scalar_t(0.0)
            adj_sngrad_t[t, f] = scalar_t(0.0)
            emit_rhat_o[t, f, 0] = scalar_t(0.0)
            emit_rhat_o[t, f, 1] = scalar_t(0.0)
            emit_rhat_o[t, f, 2] = scalar_t(0.0)
            emit_rhat_n[t, f, 0] = scalar_t(0.0)
            emit_rhat_n[t, f, 1] = scalar_t(0.0)
            emit_rhat_n[t, f, 2] = scalar_t(0.0)
            return

        af = adj_res_mom_face[t, f] * vface[f]
        adj_sngrad_p[t, f] = af * gpc_face[f]
        adj_sngrad_t[t, f] = -af * ghb_face[f] if facecv_face_buoy == 1 else scalar_t(0.0)
        nv = nrm[f]
        wo = scalar_t(1.0)
        wn = scalar_t(0.0)
        if twosided[f] == 1:
            if facecv_upwind == 1:
                sel_fl = flux_new_bc[t, f] if upwind_uses_flux_bc == 1 else f0_new[t, f]
                if sel_fl >= scalar_t(0.0):
                    wo = scalar_t(1.0)
                    wn = scalar_t(0.0)
                else:
                    wo = scalar_t(0.0)
                    wn = scalar_t(1.0)
            else:
                wo = scalar_t(0.5)
                wn = scalar_t(0.5)
        emit_rhat_o[t, f, 0] = af * wo * nv[0]
        emit_rhat_o[t, f, 1] = af * wo * nv[1]
        emit_rhat_o[t, f, 2] = af * wo * nv[2]
        emit_rhat_n[t, f, 0] = af * wn * nv[0]
        emit_rhat_n[t, f, 1] = af * wn * nv[1]
        emit_rhat_n[t, f, 2] = af * wn * nv[2]

    @wp.kernel
    def cell_bwd_kernel(
        q_p2: wp.array3d(dtype=scalar_t),
        div_phi_new: wp.array2d(dtype=scalar_t),
        div_phi_old: wp.array2d(dtype=scalar_t),
        adj_res_cont: wp.array2d(dtype=scalar_t),
        adj_res_mom: wp.array3d(dtype=scalar_t),
        adj_res_en: wp.array2d(dtype=scalar_t),
        emit_rhat_o: wp.array3d(dtype=scalar_t),
        emit_rhat_n: wp.array3d(dtype=scalar_t),
        rhat_off: wp.array(dtype=wp.int32),
        rhat_face: wp.array(dtype=wp.int32),
        rhat_side: wp.array(dtype=wp.int32),
        cpd_of: wp.array(dtype=wp.int32),
        vol: wp.array(dtype=scalar_t),
        tcoef: wp.array(dtype=scalar_t),
        pcoef_vol: wp.array(dtype=scalar_t),
        ghb_vol: wp.array(dtype=scalar_t),
        energy: int,
        mom_cell_active: int,
        mom_face_active: int,
        bounded_u: int,
        bounded_t: int,
        grad_attached: int,
        cell_row_pt_attached: int,
        theta_conv: scalar_t,
        facecv_face_buoy: int,
        adj_pfree_out: wp.array3d(dtype=scalar_t),
        adj_div_new_out: wp.array2d(dtype=scalar_t),
        adj_div_old_out: wp.array2d(dtype=scalar_t),
        adj_q_p2_cell: wp.array3d(dtype=scalar_t),
        adj_grad_cell: wp.array4d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ci = cpd_of[i]
        am0 = adj_res_mom[t, i, 0]
        am1 = adj_res_mom[t, i, 1]
        am2 = adj_res_mom[t, i, 2]

        rh0 = scalar_t(0.0)
        rh1 = scalar_t(0.0)
        rh2 = scalar_t(0.0)
        if mom_face_active == 1:
            for k in range(rhat_off[i], rhat_off[i + 1]):
                f = rhat_face[k]
                if rhat_side[k] == 0:
                    rh0 += emit_rhat_o[t, f, 0]
                    rh1 += emit_rhat_o[t, f, 1]
                    rh2 += emit_rhat_o[t, f, 2]
                else:
                    rh0 += emit_rhat_n[t, f, 0]
                    rh1 += emit_rhat_n[t, f, 1]
                    rh2 += emit_rhat_n[t, f, 2]
            inv_v = scalar_t(1.0) / vol[i]
            rh0 *= inv_v
            rh1 *= inv_v
            rh2 *= inv_v

        pf0 = am0 + rh0
        pf1 = am1 + rh1
        pf2 = am2 + rh2
        adj_pfree_out[t, i, 0] = pf0
        adj_pfree_out[t, i, 1] = pf1
        adj_pfree_out[t, i, 2] = pf2

        tc = tcoef[i]
        dn = div_phi_new[t, i]
        do = div_phi_old[t, i]
        one_m_th = scalar_t(1.0) - theta_conv

        aq0 = tc * pf0
        aq1 = tc * pf1
        aq2 = tc * pf2
        aq3 = -tc * pf0
        aq4 = -tc * pf1
        aq5 = -tc * pf2

        adiv_n = scalar_t(0.0)
        adiv_o = scalar_t(0.0)
        if bounded_u == 1:
            aq0 -= theta_conv * dn * pf0
            aq1 -= theta_conv * dn * pf1
            aq2 -= theta_conv * dn * pf2
            adiv_n -= theta_conv * (q_p2[t, ci, 0] * pf0 + q_p2[t, ci, 1] * pf1 + q_p2[t, ci, 2] * pf2)
            if theta_conv != scalar_t(1.0):
                aq3 -= one_m_th * do * pf0
                aq4 -= one_m_th * do * pf1
                aq5 -= one_m_th * do * pf2
                adiv_o -= one_m_th * (q_p2[t, ci, 3] * pf0 + q_p2[t, ci, 4] * pf1 + q_p2[t, ci, 5] * pf2)

        adj_q_p2_cell[t, ci, 0] = aq0
        adj_q_p2_cell[t, ci, 1] = aq1
        adj_q_p2_cell[t, ci, 2] = aq2
        adj_q_p2_cell[t, ci, 3] = aq3
        adj_q_p2_cell[t, ci, 4] = aq4
        adj_q_p2_cell[t, ci, 5] = aq5

        if mom_cell_active == 1 and (grad_attached == 1 or cell_row_pt_attached == 1):
            pcv = pcoef_vol[i]
            adj_grad_cell[t, ci, 6, 0] = pcv * am0
            adj_grad_cell[t, ci, 6, 1] = pcv * am1
            adj_grad_cell[t, ci, 6, 2] = pcv * am2

        if energy == 1:
            ae = adj_res_en[t, i]
            aq7 = tc * ae
            aq8 = -tc * ae
            if bounded_t == 1:
                aq7 -= theta_conv * dn * ae
                adiv_n -= theta_conv * q_p2[t, ci, 7] * ae
                if theta_conv != scalar_t(1.0):
                    aq8 -= one_m_th * do * ae
                    adiv_o -= one_m_th * q_p2[t, ci, 8] * ae
            adj_q_p2_cell[t, ci, 7] = aq7
            adj_q_p2_cell[t, ci, 8] = aq8

            if grad_attached == 1 or cell_row_pt_attached == 1:
                gbv = ghb_vol[i]
                if facecv_face_buoy == 1:
                    adj_grad_cell[t, ci, 7, 0] = -gbv * am0
                    adj_grad_cell[t, ci, 7, 1] = -gbv * am1
                    adj_grad_cell[t, ci, 7, 2] = -gbv * am2
                else:
                    adj_grad_cell[t, ci, 7, 0] = -gbv * pf0
                    adj_grad_cell[t, ci, 7, 1] = -gbv * pf1
                    adj_grad_cell[t, ci, 7, 2] = -gbv * pf2

        adj_div_new_out[t, i] = adiv_n
        adj_div_old_out[t, i] = adiv_o

    @wp.func
    def accum_sngrad_bwd(
        adj_val: scalar_t,
        sngrad_mode: int,
        delta_c: scalar_t,
        nonorth: scalar_t,
        ow: scalar_t,
        nw: scalar_t,
        cv: vec3_t,
        grad_attached: int,
        eo_q: scalar_t,
        en_q: scalar_t,
        eo_g: vec3_t,
        en_g: vec3_t,
    ):
        c_q = delta_c if sngrad_mode == SNGRAD_ORTHOGONAL else nonorth
        eo_q_out = eo_q - adj_val * c_q
        en_q_out = en_q + adj_val * c_q
        eo_g_out = eo_g
        en_g_out = en_g
        if grad_attached == 1 and sngrad_mode == SNGRAD_CORRECTED:
            eo_g_out = eo_g + (adj_val * ow) * cv
            en_g_out = en_g + (adj_val * nw) * cv
        return eo_q_out, en_q_out, eo_g_out, en_g_out

    @wp.func
    def accum_interp_bwd(
        adj_val: scalar_t,
        mode: int,
        gc: scalar_t,
        gf: scalar_t,
        rcorr: vec3_t,
        cf: vec3_t,
        ff: vec3_t,
        is_int_f: int,
        grad_attached: int,
        eo_q: scalar_t,
        en_q: scalar_t,
        eo_g: vec3_t,
        en_g: vec3_t,
    ):
        eo_q_out = eo_q + adj_val * gc
        en_q_out = en_q + adj_val * gf
        eo_g_out = eo_g
        en_g_out = en_g
        if grad_attached == 1:
            if mode == INTERP_LINEAR_CORR:
                eo_g_out = eo_g + (adj_val * gc) * rcorr
                en_g_out = en_g + (adj_val * gf) * rcorr
            elif mode == INTERP_LINEAR_GRAD and is_int_f == 1:
                eo_g_out = eo_g + (adj_val * gc) * cf
                en_g_out = en_g + (adj_val * gf) * ff
        return eo_q_out, en_q_out, eo_g_out, en_g_out

    @wp.func
    def accum_conv_bwd(
        adj_val: scalar_t,
        flux: scalar_t,
        mode: int,
        gc: scalar_t,
        gf: scalar_t,
        rcorr: vec3_t,
        cf: vec3_t,
        ff: vec3_t,
        is_int_f: int,
        bnc: int,
        grad_attached: int,
        eo_q: scalar_t,
        en_q: scalar_t,
        eo_g: vec3_t,
        en_g: vec3_t,
    ):
        if mode <= CONV_LINEAR_GRAD:
            return accum_interp_bwd(adj_val, mode, gc, gf, rcorr, cf, ff, is_int_f, grad_attached, eo_q, en_q, eo_g, en_g)
        if bnc == 1:
            return eo_q, en_q + adj_val, eo_g, en_g
        if mode == CONV_UPWIND:
            if flux > scalar_t(0.0):
                return eo_q + adj_val, en_q, eo_g, en_g
            else:
                return eo_q, en_q + adj_val, eo_g, en_g
        if flux > scalar_t(0.0):
            eo_g_out = eo_g + adj_val * cf if grad_attached == 1 else eo_g
            return eo_q + adj_val, en_q, eo_g_out, en_g
        else:
            en_g_out = en_g + adj_val * ff if grad_attached == 1 else en_g
            return eo_q, en_q + adj_val, eo_g, en_g_out

    @wp.func
    def accum_face_grad_flux_bwd(
        adj_val: scalar_t,
        gc: scalar_t,
        gf: scalar_t,
        ecf: vec3_t,
        inv_dcf: scalar_t,
        apply_corr: int,
        nv: vec3_t,
        eo_q: scalar_t,
        en_q: scalar_t,
        eo_g: vec3_t,
        en_g: vec3_t,
    ):
        ag = adj_val * nv
        eo_q_out = eo_q
        en_q_out = en_q
        if apply_corr == 1:
            proj = dot3(ag, ecf)
            dq_term = proj * inv_dcf
            eo_q_out -= dq_term
            en_q_out += dq_term
            ag = ag - proj * ecf
        eo_g_out = eo_g + gc * ag
        en_g_out = en_g + gf * ag
        return eo_q_out, en_q_out, eo_g_out, en_g_out

    @wp.kernel
    def face_bwd_FluxAndDiff_kernel(
        q_p1: wp.array3d(dtype=scalar_t),
        q_p2: wp.array3d(dtype=scalar_t),
        grad: wp.array4d(dtype=scalar_t),
        flux_new_bc: wp.array2d(dtype=scalar_t),
        flux_old_bc: wp.array2d(dtype=scalar_t),
        d_cpd: wp.array2d(dtype=scalar_t),
        mwi_br: wp.array2d(dtype=scalar_t),
        adj_res_cont: wp.array2d(dtype=scalar_t),
        adj_pfree: wp.array3d(dtype=scalar_t),
        adj_res_en: wp.array2d(dtype=scalar_t),
        adj_div_new: wp.array2d(dtype=scalar_t),
        adj_div_old: wp.array2d(dtype=scalar_t),
        adj_sngrad_p: wp.array2d(dtype=scalar_t),
        adj_sngrad_t: wp.array2d(dtype=scalar_t),
        fo: wp.array(dtype=wp.int32),
        fn: wp.array(dtype=wp.int32),
        f_int_o: wp.array(dtype=wp.int32),
        f_int_n: wp.array(dtype=wp.int32),
        f_sarea_o: wp.array(dtype=scalar_t),
        f_sarea_n: wp.array(dtype=scalar_t),
        f_all_sarea_o: wp.array(dtype=scalar_t),
        f_all_sarea_n: wp.array(dtype=scalar_t),
        gC: wp.array(dtype=scalar_t),
        gF: wp.array(dtype=scalar_t),
        nrm: wp.array(dtype=vec3_t),
        rcorr: wp.array(dtype=vec3_t),
        Cf: wp.array(dtype=vec3_t),
        Ff: wp.array(dtype=vec3_t),
        eCF: wp.array(dtype=vec3_t),
        inv_dCF: wp.array(dtype=scalar_t),
        delta_c: wp.array(dtype=scalar_t),
        nonorth: wp.array(dtype=scalar_t),
        ow_u: wp.array(dtype=scalar_t),
        nw_u: wp.array(dtype=scalar_t),
        cv_u: wp.array(dtype=vec3_t),
        ow_p: wp.array(dtype=scalar_t),
        nw_p: wp.array(dtype=scalar_t),
        cv_p: wp.array(dtype=vec3_t),
        ow_t: wp.array(dtype=scalar_t),
        nw_t: wp.array(dtype=scalar_t),
        cv_t: wp.array(dtype=vec3_t),
        is_int_face: wp.array(dtype=wp.int32),
        bnc: wp.array(dtype=wp.int32),
        grad_interp_apply: wp.array(dtype=wp.int32),
        mwi_mask: wp.array(dtype=wp.int32),
        bc_diff_mask: wp.array(dtype=wp.int32),
        zero_flux: wp.array(dtype=wp.int32),
        grad_u_bc_code: wp.array(dtype=wp.int32),
        grad_t_bc_code: wp.array(dtype=wp.int32),
        sym_delta_coeff: wp.array(dtype=scalar_t),
        ghb_face: wp.array(dtype=scalar_t),
        gpc_face: wp.array(dtype=scalar_t),
        cc_face: wp.array(dtype=scalar_t),
        ccont_face: wp.array(dtype=scalar_t),
        nu_face: wp.array(dtype=scalar_t),
        alpha_face: wp.array(dtype=scalar_t),
        energy: int,
        is_cmpt: int,
        grad_attached: int,
        use_mwi: int,
        mwi_frozen: int,
        mom_face_active: int,
        bounded_any: int,
        sngrad_mode: int,
        conv_u_mode: int,
        conv_t_mode: int,
        theta_conv: scalar_t,
        facecv_face_buoy: int,
        adj_f0_new_out: wp.array2d(dtype=scalar_t),
        adj_f0_old_out: wp.array2d(dtype=scalar_t),
        emit_d_o: wp.array2d(dtype=scalar_t),
        emit_d_n: wp.array2d(dtype=scalar_t),
        emit_q1_o: wp.array3d(dtype=scalar_t),
        emit_q1_n: wp.array3d(dtype=scalar_t),
        emit_q2_o: wp.array3d(dtype=scalar_t),
        emit_q2_n: wp.array3d(dtype=scalar_t),
        emit_grad_o: wp.array4d(dtype=scalar_t),
        emit_grad_n: wp.array4d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        o = fo[f]
        n = fn[f]
        io = f_int_o[f]
        inb = f_int_n[f]
        so = f_sarea_o[f]
        sn = f_sarea_n[f]

        afmass = so * adj_res_cont[t, io]
        afm0 = so * adj_pfree[t, io, 0]
        afm1 = so * adj_pfree[t, io, 1]
        afm2 = so * adj_pfree[t, io, 2]
        afen = so * adj_res_en[t, io] if energy == 1 else scalar_t(0.0)
        if inb >= 0:
            afmass += sn * adj_res_cont[t, inb]
            afm0 += sn * adj_pfree[t, inb, 0]
            afm1 += sn * adj_pfree[t, inb, 1]
            afm2 += sn * adj_pfree[t, inb, 2]
            if energy == 1:
                afen += sn * adj_res_en[t, inb]

        afl_new = ccont_face[f] * afmass
        afl_old = scalar_t(0.0)
        if bounded_any == 1:
            sao = f_all_sarea_o[f]
            san = f_all_sarea_n[f]
            afl_new += sao * adj_div_new[t, io]
            if theta_conv != scalar_t(1.0):
                afl_old += sao * adj_div_old[t, io]
            if inb >= 0:
                afl_new += san * adj_div_new[t, inb]
                if theta_conv != scalar_t(1.0):
                    afl_old += san * adj_div_old[t, inb]

        gc = gC[f]
        gf = gF[f]
        nv = nrm[f]
        rc = rcorr[f]
        cf = Cf[f]
        ff = Ff[f]
        iif = is_int_face[f]
        bf = bnc[f]
        dc = delta_c[f]
        no = nonorth[f]
        cc = cc_face[f]
        one_m_th = scalar_t(1.0) - theta_conv
        fl_new = flux_new_bc[t, f]
        fl_old = flux_old_bc[t, f]
        bcd = bc_diff_mask[f]

        z0 = scalar_t(0.0)
        vz = vec3_t(z0, z0, z0)

        for c in range(3):
            afm_c = afm0 if c == 0 else (afm1 if c == 1 else afm2)
            eo_q2_new = z0
            en_q2_new = z0
            eo_q2_old = z0
            en_q2_old = z0
            eo_g_new = vz
            en_g_new = vz
            eo_g_old = vz
            en_g_old = vz

            uc_new = eval_conv_val(q_p2, grad, t, o, n, c, fl_new, conv_u_mode, gc, gf, rc, cf, ff, iif, bf)
            afl_new += afm_c * theta_conv * cc * uc_new
            aconv_new = afm_c * theta_conv * cc * fl_new
            eo_q2_new, en_q2_new, eo_g_new, en_g_new = accum_conv_bwd(
                aconv_new, fl_new, conv_u_mode, gc, gf, rc, cf, ff, iif, bf, grad_attached,
                eo_q2_new, en_q2_new, eo_g_new, en_g_new,
            )

            if theta_conv != scalar_t(1.0):
                uc_old = eval_conv_val(q_p2, grad, t, o, n, 3 + c, fl_old, conv_u_mode, gc, gf, rc, cf, ff, iif, bf)
                afl_old += afm_c * one_m_th * cc * uc_old
                aconv_old = afm_c * one_m_th * cc * fl_old
                eo_q2_old, en_q2_old, eo_g_old, en_g_old = accum_conv_bwd(
                    aconv_old, fl_old, conv_u_mode, gc, gf, rc, cf, ff, iif, bf, grad_attached,
                    eo_q2_old, en_q2_old, eo_g_old, en_g_old,
                )

            if is_cmpt == 1 or bcd == 1:
                agfl_u = -nu_face[f] * afm_c
                eo_q2_new, en_q2_new, eo_g_new, en_g_new = accum_sngrad_bwd(
                    agfl_u, sngrad_mode, dc, no, ow_u[f], nw_u[f], cv_u[f], grad_attached,
                    eo_q2_new, en_q2_new, eo_g_new, en_g_new,
                )

            emit_q2_o[t, f, c] = eo_q2_new
            emit_q2_n[t, f, c] = en_q2_new
            emit_q2_o[t, f, 3 + c] = eo_q2_old
            emit_q2_n[t, f, 3 + c] = en_q2_old
            if grad_attached == 1:
                emit_grad_o[t, f, c, 0] = eo_g_new[0]
                emit_grad_o[t, f, c, 1] = eo_g_new[1]
                emit_grad_o[t, f, c, 2] = eo_g_new[2]
                emit_grad_n[t, f, c, 0] = en_g_new[0]
                emit_grad_n[t, f, c, 1] = en_g_new[1]
                emit_grad_n[t, f, c, 2] = en_g_new[2]
                emit_grad_o[t, f, 3 + c, 0] = eo_g_old[0]
                emit_grad_o[t, f, 3 + c, 1] = eo_g_old[1]
                emit_grad_o[t, f, 3 + c, 2] = eo_g_old[2]
                emit_grad_n[t, f, 3 + c, 0] = en_g_old[0]
                emit_grad_n[t, f, 3 + c, 1] = en_g_old[1]
                emit_grad_n[t, f, 3 + c, 2] = en_g_old[2]

        eo_q1_u0 = z0
        en_q1_u0 = z0
        eo_q1_u1 = z0
        en_q1_u1 = z0
        eo_q1_u2 = z0
        en_q1_u2 = z0
        if is_cmpt == 0 and bcd == 0:
            agu0 = -nu_face[f] * afm0
            agu1 = -nu_face[f] * afm1
            agu2 = -nu_face[f] * afm2
            g_uc = grad_u_bc_code[f]
            use_wlsq_u = 1
            if g_uc == G_ZERO:
                use_wlsq_u = 0
            elif g_uc == G_SYMMETRY_U:
                use_wlsq_u = 0
                adot = -sym_delta_coeff[f] * (agu0 * nv[0] + agu1 * nv[1] + agu2 * nv[2])
                emit_q2_o[t, f, 0] += adot * nv[0]
                emit_q2_o[t, f, 1] += adot * nv[1]
                emit_q2_o[t, f, 2] += adot * nv[2]
            elif g_uc == G_RACK_INLET_U:
                apj = agu0 * nv[0] + agu1 * nv[1] + agu2 * nv[2]
                agu0 = apj * nv[0]
                agu1 = apj * nv[1]
                agu2 = apj * nv[2]
            elif g_uc == G_INLETOUTLET_U:
                if fl_new >= scalar_t(0.0):
                    use_wlsq_u = 0

            if use_wlsq_u == 1:
                ecf = eCF[f]
                idcf = inv_dCF[f]
                gia = grad_interp_apply[f]
                for c in range(3):
                    agu_c = agu0 if c == 0 else (agu1 if c == 1 else agu2)
                    go_prev = vec3_t(emit_grad_o[t, f, c, 0], emit_grad_o[t, f, c, 1], emit_grad_o[t, f, c, 2])
                    gn_prev = vec3_t(emit_grad_n[t, f, c, 0], emit_grad_n[t, f, c, 1], emit_grad_n[t, f, c, 2])
                    q1o, q1n, go_upd, gn_upd = accum_face_grad_flux_bwd(
                        agu_c, gc, gf, ecf, idcf, gia, nv, z0, z0, go_prev, gn_prev,
                    )
                    if c == 0:
                        eo_q1_u0 = q1o
                        en_q1_u0 = q1n
                    elif c == 1:
                        eo_q1_u1 = q1o
                        en_q1_u1 = q1n
                    else:
                        eo_q1_u2 = q1o
                        en_q1_u2 = q1n
                    emit_grad_o[t, f, c, 0] = go_upd[0]
                    emit_grad_o[t, f, c, 1] = go_upd[1]
                    emit_grad_o[t, f, c, 2] = go_upd[2]
                    emit_grad_n[t, f, c, 0] = gn_upd[0]
                    emit_grad_n[t, f, c, 1] = gn_upd[1]
                    emit_grad_n[t, f, c, 2] = gn_upd[2]

        emit_q1_o[t, f, 0] = eo_q1_u0
        emit_q1_n[t, f, 0] = en_q1_u0
        emit_q1_o[t, f, 1] = eo_q1_u1
        emit_q1_n[t, f, 1] = en_q1_u1
        emit_q1_o[t, f, 2] = eo_q1_u2
        emit_q1_n[t, f, 2] = en_q1_u2

        eo_q2_p = z0
        en_q2_p = z0
        eo_g_p = vz
        en_g_p = vz
        if mom_face_active == 1:
            eo_q2_p, en_q2_p, eo_g_p, en_g_p = accum_sngrad_bwd(
                adj_sngrad_p[t, f], sngrad_mode, dc, no, ow_p[f], nw_p[f], cv_p[f], grad_attached,
                eo_q2_p, en_q2_p, eo_g_p, en_g_p,
            )

        eo_q2_tnew = z0
        en_q2_tnew = z0
        eo_q2_told = z0
        en_q2_told = z0
        eo_q1_t = z0
        en_q1_t = z0
        eo_g_tnew = vz
        en_g_tnew = vz
        eo_g_told = vz
        en_g_told = vz

        if energy == 1:
            tc_new = eval_conv_val(q_p2, grad, t, o, n, 7, fl_new, conv_t_mode, gc, gf, rc, cf, ff, iif, bf)
            afl_new += afen * theta_conv * cc * tc_new
            aconv_tnew = afen * theta_conv * cc * fl_new
            eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew = accum_conv_bwd(
                aconv_tnew, fl_new, conv_t_mode, gc, gf, rc, cf, ff, iif, bf, grad_attached,
                eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew,
            )
            if theta_conv != scalar_t(1.0):
                tc_old = eval_conv_val(q_p2, grad, t, o, n, 8, fl_old, conv_t_mode, gc, gf, rc, cf, ff, iif, bf)
                afl_old += afen * one_m_th * cc * tc_old
                aconv_told = afen * one_m_th * cc * fl_old
                eo_q2_told, en_q2_told, eo_g_told, en_g_told = accum_conv_bwd(
                    aconv_told, fl_old, conv_t_mode, gc, gf, rc, cf, ff, iif, bf, grad_attached,
                    eo_q2_told, en_q2_told, eo_g_told, en_g_told,
                )

            agfl_t = -alpha_face[f] * afen
            if is_cmpt == 1 or bcd == 1:
                eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew = accum_sngrad_bwd(
                    agfl_t, sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f], grad_attached,
                    eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew,
                )
            else:
                if grad_t_bc_code[f] != G_ZERO:
                    eo_q1_t, en_q1_t, eo_g_tnew, en_g_tnew = accum_face_grad_flux_bwd(
                        agfl_t, gc, gf, eCF[f], inv_dCF[f], grad_interp_apply[f], nv,
                        eo_q1_t, en_q1_t, eo_g_tnew, en_g_tnew,
                    )
            if mom_face_active == 1 and facecv_face_buoy == 1:
                eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew = accum_sngrad_bwd(
                    adj_sngrad_t[t, f], sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f], grad_attached,
                    eo_q2_tnew, en_q2_tnew, eo_g_tnew, en_g_tnew,
                )

        af_pre = z0 if zero_flux[f] == 1 else afl_new
        af0_old = z0 if zero_flux[f] == 1 else afl_old
        af0_new = af_pre

        eo_q1_p = z0
        en_q1_p = z0
        ed_o = z0
        ed_n = z0

        if use_mwi == 1 and mwi_mask[f] == 1:
            df = scalar_t(0.5) * (d_cpd[t, o] + d_cpd[t, n])
            abr = af_pre * gpc_face[f] * df
            if mwi_frozen == 0:
                adf = af_pre * gpc_face[f] * mwi_br[t, f]
                ed_o = scalar_t(0.5) * adf
                ed_n = scalar_t(0.5) * adf

            eo_q1_p, en_q1_p, eo_g_p, en_g_p = accum_sngrad_bwd(
                -abr, sngrad_mode, dc, no, ow_p[f], nw_p[f], cv_p[f], grad_attached,
                eo_q1_p, en_q1_p, eo_g_p, en_g_p,
            )
            if grad_attached == 1:
                eo_g_p = eo_g_p + (abr * gc) * nv
                en_g_p = en_g_p + (abr * gf) * nv
            if energy == 1:
                abrt = abr * ghb_face[f]
                eo_q1_t, en_q1_t, eo_g_tnew, en_g_tnew = accum_sngrad_bwd(
                    abrt, sngrad_mode, dc, no, ow_t[f], nw_t[f], cv_t[f], grad_attached,
                    eo_q1_t, en_q1_t, eo_g_tnew, en_g_tnew,
                )
                if grad_attached == 1:
                    eo_g_tnew = eo_g_tnew - (abrt * gc) * nv
                    en_g_tnew = en_g_tnew - (abrt * gf) * nv

        adj_f0_new_out[t, f] = af0_new
        adj_f0_old_out[t, f] = af0_old
        emit_d_o[t, f] = ed_o
        emit_d_n[t, f] = ed_n

        emit_q2_o[t, f, 6] = eo_q2_p
        emit_q2_n[t, f, 6] = en_q2_p
        emit_q1_o[t, f, 3] = eo_q1_p
        emit_q1_n[t, f, 3] = en_q1_p
        if grad_attached == 1:
            emit_grad_o[t, f, 6, 0] = eo_g_p[0]
            emit_grad_o[t, f, 6, 1] = eo_g_p[1]
            emit_grad_o[t, f, 6, 2] = eo_g_p[2]
            emit_grad_n[t, f, 6, 0] = en_g_p[0]
            emit_grad_n[t, f, 6, 1] = en_g_p[1]
            emit_grad_n[t, f, 6, 2] = en_g_p[2]

        if energy == 1:
            emit_q2_o[t, f, 7] = eo_q2_tnew
            emit_q2_n[t, f, 7] = en_q2_tnew
            emit_q2_o[t, f, 8] = eo_q2_told
            emit_q2_n[t, f, 8] = en_q2_told
            emit_q1_o[t, f, 4] = eo_q1_t
            emit_q1_n[t, f, 4] = en_q1_t
            if grad_attached == 1:
                emit_grad_o[t, f, 7, 0] = eo_g_tnew[0]
                emit_grad_o[t, f, 7, 1] = eo_g_tnew[1]
                emit_grad_o[t, f, 7, 2] = eo_g_tnew[2]
                emit_grad_n[t, f, 7, 0] = en_g_tnew[0]
                emit_grad_n[t, f, 7, 1] = en_g_tnew[1]
                emit_grad_n[t, f, 7, 2] = en_g_tnew[2]
                emit_grad_o[t, f, 8, 0] = eo_g_told[0]
                emit_grad_o[t, f, 8, 1] = eo_g_told[1]
                emit_grad_o[t, f, 8, 2] = eo_g_told[2]
                emit_grad_n[t, f, 8, 0] = en_g_told[0]
                emit_grad_n[t, f, 8, 1] = en_g_told[1]
                emit_grad_n[t, f, 8, 2] = en_g_told[2]

    @wp.kernel
    def mwi_ap_bwd_cell_kernel(
        ap_int: wp.array2d(dtype=scalar_t),
        vol: wp.array(dtype=scalar_t),
        emit_d_o: wp.array2d(dtype=scalar_t),
        emit_d_n: wp.array2d(dtype=scalar_t),
        rhat_off: wp.array(dtype=wp.int32),
        rhat_face: wp.array(dtype=wp.int32),
        rhat_side: wp.array(dtype=wp.int32),
        adj_ap_int: wp.array2d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ad = scalar_t(0.0)
        for k in range(rhat_off[i], rhat_off[i + 1]):
            f = rhat_face[k]
            if rhat_side[k] == 0:
                ad += emit_d_o[t, f]
            else:
                ad += emit_d_n[t, f]
        ap = ap_int[t, i]
        adj_ap_int[t, i] = -ad * vol[i] / (ap * ap)

    @wp.kernel
    def face_pre_bwd_kernel(
        f0_new: wp.array2d(dtype=scalar_t),
        adj_f0_new: wp.array2d(dtype=scalar_t),
        adj_f0_old: wp.array2d(dtype=scalar_t),
        adj_ap_int: wp.array2d(dtype=scalar_t),
        f_int_o: wp.array(dtype=wp.int32),
        f_int_n: wp.array(dtype=wp.int32),
        f_sarea_o: wp.array(dtype=scalar_t),
        f_sarea_n: wp.array(dtype=scalar_t),
        gC: wp.array(dtype=scalar_t),
        gF: wp.array(dtype=scalar_t),
        rcorr: wp.array(dtype=vec3_t),
        Cf: wp.array(dtype=vec3_t),
        Ff: wp.array(dtype=vec3_t),
        nrm: wp.array(dtype=vec3_t),
        is_int_face: wp.array(dtype=wp.int32),
        interp_mode: int,
        grad_attached: int,
        mwi_implicit: int,
        emit_q1_o: wp.array3d(dtype=scalar_t),
        emit_q1_n: wp.array3d(dtype=scalar_t),
        emit_q2_o: wp.array3d(dtype=scalar_t),
        emit_q2_n: wp.array3d(dtype=scalar_t),
        emit_grad_o: wp.array4d(dtype=scalar_t),
        emit_grad_n: wp.array4d(dtype=scalar_t),
    ):
        t, f = wp.tid()
        af0_n = adj_f0_new[t, f]
        if mwi_implicit == 1:
            fl = f0_new[t, f]
            io = f_int_o[f]
            so = f_sarea_o[f]
            if fl * so > scalar_t(0.0):
                af0_n += so * adj_ap_int[t, io]
            inb = f_int_n[f]
            if inb >= 0:
                sn = f_sarea_n[f]
                if fl * sn > scalar_t(0.0):
                    af0_n += sn * adj_ap_int[t, inb]
        af0_o = adj_f0_old[t, f]

        gc = gC[f]
        gf = gF[f]
        rc = rcorr[f]
        cf = Cf[f]
        ff = Ff[f]
        nv = nrm[f]
        iif = is_int_face[f]

        for c in range(3):
            auf_n = af0_n * nv[c]
            q1o = emit_q1_o[t, f, c]
            q1n = emit_q1_n[t, f, c]
            go_n = vec3_t(emit_grad_o[t, f, c, 0], emit_grad_o[t, f, c, 1], emit_grad_o[t, f, c, 2])
            gn_n = vec3_t(emit_grad_n[t, f, c, 0], emit_grad_n[t, f, c, 1], emit_grad_n[t, f, c, 2])
            q1o, q1n, go_n, gn_n = accum_interp_bwd(
                auf_n, interp_mode, gc, gf, rc, cf, ff, iif, grad_attached,
                q1o, q1n, go_n, gn_n,
            )
            emit_q1_o[t, f, c] = q1o
            emit_q1_n[t, f, c] = q1n
            if grad_attached == 1:
                emit_grad_o[t, f, c, 0] = go_n[0]
                emit_grad_o[t, f, c, 1] = go_n[1]
                emit_grad_o[t, f, c, 2] = go_n[2]
                emit_grad_n[t, f, c, 0] = gn_n[0]
                emit_grad_n[t, f, c, 1] = gn_n[1]
                emit_grad_n[t, f, c, 2] = gn_n[2]

            auf_o = af0_o * nv[c]
            q2o_old = emit_q2_o[t, f, 3 + c]
            q2n_old = emit_q2_n[t, f, 3 + c]
            go_o = vec3_t(emit_grad_o[t, f, 3 + c, 0], emit_grad_o[t, f, 3 + c, 1], emit_grad_o[t, f, 3 + c, 2])
            gn_o = vec3_t(emit_grad_n[t, f, 3 + c, 0], emit_grad_n[t, f, 3 + c, 1], emit_grad_n[t, f, 3 + c, 2])
            q2o_old, q2n_old, go_o, gn_o = accum_interp_bwd(
                auf_o, interp_mode, gc, gf, rc, cf, ff, iif, grad_attached,
                q2o_old, q2n_old, go_o, gn_o,
            )
            emit_q2_o[t, f, 3 + c] = q2o_old
            emit_q2_n[t, f, 3 + c] = q2n_old
            if grad_attached == 1:
                emit_grad_o[t, f, 3 + c, 0] = go_o[0]
                emit_grad_o[t, f, 3 + c, 1] = go_o[1]
                emit_grad_o[t, f, 3 + c, 2] = go_o[2]
                emit_grad_n[t, f, 3 + c, 0] = gn_o[0]
                emit_grad_n[t, f, 3 + c, 1] = gn_o[1]
                emit_grad_n[t, f, 3 + c, 2] = gn_o[2]

    @wp.kernel
    def cpd_gather_face_bwd_kernel(
        adj_phi_new_p2: wp.array3d(dtype=scalar_t),
        adj_q_p2_cell: wp.array3d(dtype=scalar_t),
        adj_grad_cell: wp.array4d(dtype=scalar_t),
        emit_q1_o: wp.array3d(dtype=scalar_t),
        emit_q1_n: wp.array3d(dtype=scalar_t),
        emit_q2_o: wp.array3d(dtype=scalar_t),
        emit_q2_n: wp.array3d(dtype=scalar_t),
        emit_grad_o: wp.array4d(dtype=scalar_t),
        emit_grad_n: wp.array4d(dtype=scalar_t),
        cpd_f_off: wp.array(dtype=wp.int32),
        cpd_f_idx: wp.array(dtype=wp.int32),
        cpd_f_side: wp.array(dtype=wp.int32),
        energy: int,
        k_ch: int,
        grad_attached: int,
        adj_q_p1: wp.array3d(dtype=scalar_t),
        adj_q_p2_total: wp.array3d(dtype=scalar_t),
        adj_grad_cpd: wp.array4d(dtype=scalar_t),
    ):
        t, cpd = wp.tid()
        q2_0 = adj_q_p2_cell[t, cpd, 0] + adj_phi_new_p2[t, cpd, 0]
        q2_1 = adj_q_p2_cell[t, cpd, 1] + adj_phi_new_p2[t, cpd, 1]
        q2_2 = adj_q_p2_cell[t, cpd, 2] + adj_phi_new_p2[t, cpd, 2]
        q2_3 = adj_q_p2_cell[t, cpd, 3]
        q2_4 = adj_q_p2_cell[t, cpd, 4]
        q2_5 = adj_q_p2_cell[t, cpd, 5]
        q2_6 = adj_q_p2_cell[t, cpd, 6] + adj_phi_new_p2[t, cpd, 3]
        q2_7 = (adj_q_p2_cell[t, cpd, 7] + adj_phi_new_p2[t, cpd, 4]) if energy == 1 else scalar_t(0.0)
        q2_8 = adj_q_p2_cell[t, cpd, 8] if energy == 1 else scalar_t(0.0)

        q1_0 = scalar_t(0.0)
        q1_1 = scalar_t(0.0)
        q1_2 = scalar_t(0.0)
        q1_p = scalar_t(0.0)
        q1_t = scalar_t(0.0)

        for k in range(cpd_f_off[cpd], cpd_f_off[cpd + 1]):
            f = cpd_f_idx[k]
            if cpd_f_side[k] == 0:
                q2_0 += emit_q2_o[t, f, 0]
                q2_1 += emit_q2_o[t, f, 1]
                q2_2 += emit_q2_o[t, f, 2]
                q2_3 += emit_q2_o[t, f, 3]
                q2_4 += emit_q2_o[t, f, 4]
                q2_5 += emit_q2_o[t, f, 5]
                q2_6 += emit_q2_o[t, f, 6]
                q1_0 += emit_q1_o[t, f, 0]
                q1_1 += emit_q1_o[t, f, 1]
                q1_2 += emit_q1_o[t, f, 2]
                q1_p += emit_q1_o[t, f, 3]
                if energy == 1:
                    q2_7 += emit_q2_o[t, f, 7]
                    q2_8 += emit_q2_o[t, f, 8]
                    q1_t += emit_q1_o[t, f, 4]
            else:
                q2_0 += emit_q2_n[t, f, 0]
                q2_1 += emit_q2_n[t, f, 1]
                q2_2 += emit_q2_n[t, f, 2]
                q2_3 += emit_q2_n[t, f, 3]
                q2_4 += emit_q2_n[t, f, 4]
                q2_5 += emit_q2_n[t, f, 5]
                q2_6 += emit_q2_n[t, f, 6]
                q1_0 += emit_q1_n[t, f, 0]
                q1_1 += emit_q1_n[t, f, 1]
                q1_2 += emit_q1_n[t, f, 2]
                q1_p += emit_q1_n[t, f, 3]
                if energy == 1:
                    q2_7 += emit_q2_n[t, f, 7]
                    q2_8 += emit_q2_n[t, f, 8]
                    q1_t += emit_q1_n[t, f, 4]

        adj_q_p2_total[t, cpd, 0] = q2_0
        adj_q_p2_total[t, cpd, 1] = q2_1
        adj_q_p2_total[t, cpd, 2] = q2_2

        adj_q_p1[t, cpd, 0] = q1_0 + q2_0
        adj_q_p1[t, cpd, 1] = q1_1 + q2_1
        adj_q_p1[t, cpd, 2] = q1_2 + q2_2
        adj_q_p1[t, cpd, 3] = q2_3
        adj_q_p1[t, cpd, 4] = q2_4
        adj_q_p1[t, cpd, 5] = q2_5
        adj_q_p1[t, cpd, 6] = q1_p + q2_6
        if energy == 1:
            adj_q_p1[t, cpd, 7] = q1_t + q2_7
            adj_q_p1[t, cpd, 8] = q2_8

        if grad_attached == 1:
            for ch in range(k_ch):
                gx = adj_grad_cell[t, cpd, ch, 0]
                gy = adj_grad_cell[t, cpd, ch, 1]
                gz = adj_grad_cell[t, cpd, ch, 2]
                for k in range(cpd_f_off[cpd], cpd_f_off[cpd + 1]):
                    f = cpd_f_idx[k]
                    if cpd_f_side[k] == 0:
                        gx += emit_grad_o[t, f, ch, 0]
                        gy += emit_grad_o[t, f, ch, 1]
                        gz += emit_grad_o[t, f, ch, 2]
                    else:
                        gx += emit_grad_n[t, f, ch, 0]
                        gy += emit_grad_n[t, f, ch, 1]
                        gz += emit_grad_n[t, f, ch, 2]
                adj_grad_cpd[t, cpd, ch, 0] = gx
                adj_grad_cpd[t, cpd, ch, 1] = gy
                adj_grad_cpd[t, cpd, ch, 2] = gz

    @wp.kernel
    def pass2_ghost_bwd_fix_kernel(
        f0_new: wp.array2d(dtype=scalar_t),
        adj_q_p2_total: wp.array3d(dtype=scalar_t),
        ghost_cpd: wp.array(dtype=wp.int32),
        owner_cpd: wp.array(dtype=wp.int32),
        ghost_face: wp.array(dtype=wp.int32),
        u_code: wp.array(dtype=wp.int32),
        own_off: wp.array(dtype=wp.int32),
        own_g: wp.array(dtype=wp.int32),
        cpd_of: wp.array(dtype=wp.int32),
        include_old_p1: int,
        wzero: int,
        adj_q_p1: wp.array3d(dtype=scalar_t),
    ):
        t, i = wp.tid()
        ci = cpd_of[i]
        add0 = scalar_t(0.0)
        add1 = scalar_t(0.0)
        add2 = scalar_t(0.0)
        for k in range(own_off[i], own_off[i + 1]):
            g = own_g[k]
            uc = u_code[g]
            if uc == U_ADVECTIVE and include_old_p1 == 0:
                gi = ghost_cpd[g]
                adj_q_p1[t, gi, 0] -= adj_q_p2_total[t, gi, 0]
                adj_q_p1[t, gi, 1] -= adj_q_p2_total[t, gi, 1]
                adj_q_p1[t, gi, 2] -= adj_q_p2_total[t, gi, 2]
            elif uc == U_INLETOUTLET:
                gi = ghost_cpd[g]
                bf = ghost_face[g]
                aq0 = adj_q_p2_total[t, gi, 0]
                aq1 = adj_q_p2_total[t, gi, 1]
                aq2 = adj_q_p2_total[t, gi, 2]
                adj_q_p1[t, gi, 0] -= aq0
                adj_q_p1[t, gi, 1] -= aq1
                adj_q_p1[t, gi, 2] -= aq2
                if f0_new[t, bf] >= scalar_t(0.0):
                    add0 += aq0
                    add1 += aq1
                    if wzero == 0:
                        add2 += aq2
        adj_q_p1[t, ci, 0] += add0
        adj_q_p1[t, ci, 1] += add1
        adj_q_p1[t, ci, 2] += add2

    @wp.kernel
    def wlsq_fold_cyc_bwd_kernel(
        adj_grad_cpd: wp.array4d(dtype=scalar_t),
        cpd_of: wp.array(dtype=wp.int32),
        cyc_off: wp.array(dtype=wp.int32),
        cyc_ghost_of_int: wp.array(dtype=wp.int32),
        ch_offset: int,
        adj_grad_int: wp.array4d(dtype=scalar_t),
    ):
        t, i, c_rel = wp.tid()
        ch = ch_offset + c_rel
        ci = cpd_of[i]
        gx = adj_grad_cpd[t, ci, ch, 0]
        gy = adj_grad_cpd[t, ci, ch, 1]
        gz = adj_grad_cpd[t, ci, ch, 2]
        for k in range(cyc_off[i], cyc_off[i + 1]):
            gi = cyc_ghost_of_int[k]
            gx += adj_grad_cpd[t, gi, ch, 0]
            gy += adj_grad_cpd[t, gi, ch, 1]
            gz += adj_grad_cpd[t, gi, ch, 2]
        adj_grad_int[t, i, ch, 0] = gx
        adj_grad_int[t, i, ch, 1] = gy
        adj_grad_int[t, i, ch, 2] = gz

    @wp.kernel
    def wlsq_transpose_bwd_kernel(
        adj_grad_int: wp.array4d(dtype=scalar_t),
        cpd_to_int: wp.array(dtype=wp.int32),
        wt_off: wp.array(dtype=wp.int32),
        wt_cell: wp.array(dtype=wp.int32),
        wt_vec: wp.array(dtype=vec3_t),
        w_self: wp.array(dtype=vec3_t),
        ch_offset: int,
        adj_q_p1: wp.array3d(dtype=scalar_t),
    ):
        t, cpd, c_rel = wp.tid()
        ch = ch_offset + c_rel
        acc = scalar_t(0.0)
        for m in range(wt_off[cpd], wt_off[cpd + 1]):
            ic = wt_cell[m]
            wv = wt_vec[m]
            acc += (
                wv[0] * adj_grad_int[t, ic, ch, 0]
                + wv[1] * adj_grad_int[t, ic, ch, 1]
                + wv[2] * adj_grad_int[t, ic, ch, 2]
            )
        ii = cpd_to_int[cpd]
        if ii >= 0:
            ws = w_self[ii]
            acc -= (
                ws[0] * adj_grad_int[t, ii, ch, 0]
                + ws[1] * adj_grad_int[t, ii, ch, 1]
                + ws[2] * adj_grad_int[t, ii, ch, 2]
            )
        adj_q_p1[t, cpd, ch] += acc

    @wp.kernel
    def unpack_adj_q_kernel(
        adj_q_p1: wp.array3d(dtype=scalar_t),
        energy: int,
        adj_phi_old: wp.array3d(dtype=scalar_t),
        adj_phi_new: wp.array3d(dtype=scalar_t),
    ):
        t, cpd = wp.tid()
        adj_phi_new[t, cpd, 0] = adj_q_p1[t, cpd, 0]
        adj_phi_new[t, cpd, 1] = adj_q_p1[t, cpd, 1]
        adj_phi_new[t, cpd, 2] = adj_q_p1[t, cpd, 2]
        adj_phi_new[t, cpd, 3] = adj_q_p1[t, cpd, 6]

        adj_phi_old[t, cpd, 0] = adj_q_p1[t, cpd, 3]
        adj_phi_old[t, cpd, 1] = adj_q_p1[t, cpd, 4]
        adj_phi_old[t, cpd, 2] = adj_q_p1[t, cpd, 5]
        adj_phi_old[t, cpd, 3] = scalar_t(0.0)

        if energy == 1:
            adj_phi_new[t, cpd, 4] = adj_q_p1[t, cpd, 7]
            adj_phi_old[t, cpd, 4] = adj_q_p1[t, cpd, 8]

    return {
        "pack_q": pack_q_kernel,
        "wlsq_fwd": wlsq_fwd_kernel,
        "ghost_grad_fwd": ghost_grad_fwd_kernel,
        "face_pre": face_pre_kernel,
        "mwi_ap": mwi_ap_kernel,
        "mwi_dghost": mwi_dghost_kernel,
        "pass2_ghost": pass2_ghost_kernel,
        "face_main": face_main_kernel,
        "cell_balance": cell_balance_kernel,
        "rhat_ghost": rhat_ghost_kernel,
        "face_cv_fwd": face_cv_fwd_kernel,
        "face_cv_bwd_emit": face_cv_bwd_emit_kernel,
        "cell_bwd": cell_bwd_kernel,
        "face_bwd": face_bwd_FluxAndDiff_kernel,
        "mwi_ap_bwd": mwi_ap_bwd_cell_kernel,
        "face_pre_bwd": face_pre_bwd_kernel,
        "cpd_gather_face_bwd": cpd_gather_face_bwd_kernel,
        "pass2_ghost_bwd_fix": pass2_ghost_bwd_fix_kernel,
        "wlsq_fold_cyc_bwd": wlsq_fold_cyc_bwd_kernel,
        "wlsq_transpose_bwd": wlsq_transpose_bwd_kernel,
        "unpack_adj_q": unpack_adj_q_kernel,
    }


_FVM_KERNELS_F32 = _make_fvm_kernels(wp.float32, wp.vec3f)
_FVM_KERNELS_F64 = _make_fvm_kernels(wp.float64, wp.vec3d)


def _parse_interp_mode(scheme: str) -> Optional[int]:
    s = str(scheme).strip()
    if s == "Linear":
        return INTERP_LINEAR
    if s == "Linear corrected":
        return INTERP_LINEAR_CORR
    if s == "Linear grad":
        return INTERP_LINEAR_GRAD
    return None


def _parse_conv_mode(scheme: str) -> Tuple[Optional[int], int]:
    s = str(scheme).strip()
    is_bounded = 1 if s.lower().startswith("bounded") else 0
    if is_bounded:
        parts = s.split(None, 1)
        s = parts[1].strip() if len(parts) > 1 else "Linear"
    if s.endswith(" leastSquares"):
        s = s[: -len(" leastSquares")].strip()
    if s == "Linear":
        return CONV_LINEAR, is_bounded
    if s == "Linear corrected":
        return CONV_LINEAR_CORR, is_bounded
    if s == "Linear grad":
        return CONV_LINEAR_GRAD, is_bounded
    if s == "Upwind":
        return CONV_UPWIND, is_bounded
    if s == "LinearUpwind":
        return CONV_LINEAR_UPWIND, is_bounded
    return None, is_bounded


def _parse_sngrad_mode(scheme: str) -> Optional[int]:
    s = str(scheme).strip()
    if s.endswith(" leastSquares"):
        s = s[: -len(" leastSquares")].strip()
    if s == "corrected":
        return SNGRAD_CORRECTED
    if s == "uncorrected":
        return SNGRAD_UNCORRECTED
    if s == "orthogonal":
        return SNGRAD_ORTHOGONAL
    return None


def can_use_warp(solver: Any) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` indicating whether ``solver`` configuration is supported by the Warp engine."""
    dev = getattr(solver, "cpd_cell_pos", None)
    if dev is None or not dev.is_cuda:
        return False, "Warp FVM solver requires a CUDA device"
    if float(getattr(solver, "loss_phi_jump_weight", 0.0)) != 0.0:
        return False, "loss_phi_jump_weight != 0"
    fvs = solver.params.fvschemes
    if str(fvs.get("gradSchemes", "1st")).strip() != "1st":
        return False, f"gradSchemes={fvs.get('gradSchemes')!r} != '1st'"
    if bool(fvs.get("gradLimiter", {}).get("enabled", False)):
        return False, "gradLimiter enabled"
    if _parse_interp_mode(fvs.get("interpolationSchemes", "Linear")) is None:
        return False, f"unsupported interpolationSchemes={fvs.get('interpolationSchemes')!r}"
    grad_interp = fvs.get("gradInterpolationSchemes", fvs.get("interpolationSchemes", "Linear corrected"))
    if str(grad_interp).strip() not in ("Linear", "Linear corrected", "Linear overRelaxed"):
        return False, f"unsupported gradInterpolationSchemes={grad_interp!r}"
    u_mode, _ = _parse_conv_mode(getattr(solver, "U_scheme", "Linear"))
    if u_mode is None:
        return False, f"unsupported U divScheme={getattr(solver, 'U_scheme', None)!r}"
    if hasattr(solver, "T_scheme"):
        t_mode, _ = _parse_conv_mode(solver.T_scheme)
        if t_mode is None:
            return False, f"unsupported T divScheme={solver.T_scheme!r}"
    if _parse_sngrad_mode(getattr(solver, "_snGrad_scheme", "corrected")) is None:
        return False, f"unsupported snGradScheme={getattr(solver, '_snGrad_scheme', None)!r}"
    return True, "ok"


class WarpIncompressibleEngine:
    """Cached per-(mesh, dtype, BC-batch) Warp discretization & BC engine."""

    def __init__(self, solver: Any, dtype: torch.dtype):
        wp.init()
        self.solver = solver
        self.dtype = dtype
        self.device = solver.cpd_cell_pos.device
        self.wp_dev = wp.device_from_torch(self.device)
        self.is_f64 = dtype == torch.float64
        self.kernels = _FVM_KERNELS_F64 if self.is_f64 else _FVM_KERNELS_F32

        mod_name = type(solver).__module__.split(".")[-1].lower()
        self.mod_name = mod_name
        self.energy = 1 if "buoyant" in mod_name else 0
        self.is_cmpt = 1 if "cmpt" in mod_name else 0
        self.grad_attached = 0 if self.is_cmpt else 1
        self.include_old_p1 = 0 if mod_name == "picocmpt" else 1
        self.upwind_uses_flux_bc = 1 if mod_name in ("picoext", "buoyantboussinesqpicocmpt") else 0
        self.k_ch = 9 if self.energy else 7
        self.n_vars = 5 if self.energy else 4
        self.dim = int(getattr(solver, "dim", 3))

        self.bc_engine = WarpIncompressibleBCs(solver, dtype)
        self._build_static_mesh_tables(solver)
        self.refresh_dynamic_coeffs(solver)

    def _build_static_mesh_tables(self, app: Any) -> None:
        dev = self.device
        dt = self.dtype
        v3_wp = wp.vec3d if self.is_f64 else wp.vec3f

        n_cpd = int(app.num_cpd_cells)
        n_face = int(app.unv_face.shape[0])
        cpd_of = app._interior_cell_indices.to(device=dev, dtype=torch.int64).reshape(-1)
        n_int = int(cpd_of.numel())
        self.n_cpd = n_cpd
        self.n_int = n_int
        self.n_face = n_face
        self.n_ghost = self.bc_engine.n_ghost

        cpd_to_int = torch.full((n_cpd,), -1, dtype=torch.int32, device=dev)
        cpd_to_int[cpd_of] = torch.arange(n_int, device=dev, dtype=torch.int32)

        edge_x = app.neighbor_cpd_cell_x.to(device=dev, dtype=torch.int64)
        twoway = torch.cat((edge_x, edge_x.flip(0)), dim=1)
        B1 = app.single_B_cell_to_cell.to(device=dev, dtype=dt).squeeze(-1)
        B2 = torch.cat((B1, -B1), dim=0)

        out_n = twoway[0]
        in_n = twoway[1]
        in_int = cpd_to_int[in_n]
        keep_w = in_int >= 0
        rows_int = in_int[keep_w]
        others_cpd = out_n[keep_w].to(torch.int32)
        B_keep = B2[keep_w]

        A_int = app.A_cell_to_cell[cpd_of].to(device=dev, dtype=dt)
        if self.dim == 2:
            A_2d = A_int[:, :2, :2]
            if self.is_f64:
                A_inv2 = torch.linalg.inv_ex(A_2d)[0]
            else:
                eye2 = torch.eye(2, device=dev, dtype=dt).expand_as(A_2d)
                A_inv2 = torch.linalg.solve_ex(A_2d, eye2)[0]
            w_xy = torch.bmm(A_inv2[rows_int.to(torch.int64)], B_keep[:, :2].unsqueeze(-1)).squeeze(-1)
            w_raw = torch.cat((w_xy, torch.zeros_like(w_xy[:, :1])), dim=-1)
        else:
            if self.is_f64:
                A_inv3 = torch.linalg.inv_ex(A_int)[0]
            else:
                eye3 = torch.eye(3, device=dev, dtype=dt).expand_as(A_int)
                A_inv3 = torch.linalg.solve_ex(A_int, eye3)[0]
            w_raw = torch.bmm(A_inv3[rows_int.to(torch.int64)], B_keep.unsqueeze(-1)).squeeze(-1)

        w_off, w_perm = _csr_from_rows(rows_int.to(torch.int64), n_int)
        w_other = others_cpd[w_perm].contiguous()
        w_vec = w_raw[w_perm].contiguous()

        w_self = torch.zeros((n_int, 3), dtype=dt, device=dev)
        w_self.index_add_(0, rows_int.to(torch.int64), w_raw)

        wt_off, wt_perm = _csr_from_rows(w_other.to(torch.int64), n_cpd)
        sorted_rows_int = rows_int.to(torch.int32)[w_perm]
        wt_cell = sorted_rows_int[wt_perm].contiguous()
        wt_vec = w_vec[wt_perm].contiguous()

        is_int = torch.zeros(n_cpd, dtype=torch.bool, device=dev)
        is_int[cpd_of] = True
        ghost_cpd = torch.nonzero(~is_int, as_tuple=False).reshape(-1)
        cpd_to_ghost = torch.full((n_cpd,), -1, dtype=torch.int64, device=dev)
        cpd_to_ghost[ghost_cpd] = torch.arange(self.n_ghost, device=dev, dtype=torch.int64)

        C = app.C_senders.to(device=dev, dtype=torch.int64).reshape(-1)
        F = app.F_receivers.to(device=dev, dtype=torch.int64).reshape(-1)
        bnd_faces = torch.nonzero(~is_int[F], as_tuple=False).reshape(-1)
        ghost_face = torch.full((self.n_ghost,), -1, dtype=torch.int64, device=dev)
        ghost_face[cpd_to_ghost[F[bnd_faces]]] = bnd_faces

        cyc_partner_ghost = torch.full((self.n_ghost,), -1, dtype=torch.int32, device=dev)
        gh_cyc, pa_cyc = app._cyclic_ghost_partner_ids()
        if gh_cyc.numel() > 0:
            g_slots = cpd_to_ghost[gh_cyc.to(device=dev, dtype=torch.int64)]
            cyc_partner_ghost[g_slots] = pa_cyc.to(device=dev, dtype=torch.int32)
            pa_int = cpd_to_int[pa_cyc.to(device=dev, dtype=torch.int64)].to(torch.int64)
            cyc_off, cyc_perm = _csr_from_rows(pa_int, n_int)
            cyc_ghost_of_int = gh_cyc.to(device=dev, dtype=torch.int32)[cyc_perm].contiguous()
        else:
            cyc_off = torch.zeros(n_int + 1, dtype=torch.int32, device=dev)
            cyc_ghost_of_int = torch.empty(0, dtype=torch.int32, device=dev)

        fo = C.to(torch.int32).contiguous()
        fn = F.to(torch.int32).contiguous()
        gC = app.gC.to(device=dev, dtype=dt).reshape(n_face).contiguous()
        gF = app.gF.to(device=dev, dtype=dt).reshape(n_face).contiguous()
        nrm = app.unv_face.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous()
        pos = app.cpd_cell_pos.to(device=dev, dtype=dt).reshape(n_cpd, 3)
        fpos = app.face_pos.to(device=dev, dtype=dt).reshape(n_face, 3)
        rcorr = (fpos - (gC.unsqueeze(1) * pos[C] + gF.unsqueeze(1) * pos[F])).contiguous()
        Cf = app.Cf.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous()
        Ff = app.Ff.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous()
        eCF = app.eCF.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous()
        inv_dCF = (1.0 / app.dCF.to(device=dev, dtype=dt).reshape(n_face)).contiguous()
        delta_c = app.deltaCoeffs.to(device=dev, dtype=dt).reshape(n_face).contiguous()
        nonorth = app.nonOrthDeltaCoeffs.to(device=dev, dtype=dt).reshape(n_face).contiguous()

        ow_u, nw_u, cv_u = app._snGrad_operands(VARType.VELOCITY)
        p_vt = VARType.PRESSURE_RGH if self.energy else VARType.PRESSURE
        ow_p, nw_p, cv_p = app._snGrad_operands(p_vt)
        if self.energy:
            ow_t, nw_t, cv_t = app._snGrad_operands(VARType.TEMPERATURE)
        else:
            ow_t, nw_t, cv_t = ow_p, nw_p, cv_p

        is_int_face = app.mask_interior_face.to(device=dev, dtype=torch.int32).reshape(n_face).contiguous()
        bnc = app.mask_boundary_non_cyclic_face.to(device=dev, dtype=torch.int32).reshape(n_face).contiguous()

        fvs = app.params.fvschemes
        grad_interp_scheme = str(fvs.get("gradInterpolationSchemes", fvs.get("interpolationSchemes", "Linear corrected"))).strip()
        if grad_interp_scheme == "Linear":
            grad_interp_apply = torch.zeros(n_face, dtype=torch.int32, device=dev)
        elif grad_interp_scheme == "Linear corrected":
            cos_th = (nrm * eCF).sum(dim=1).abs().clamp(0.0, 1.0)
            sin_th = torch.sqrt((1.0 - cos_th * cos_th).clamp_min(0.0))
            grad_interp_apply = (sin_th > 1.0e-5).to(torch.int32)
        else:
            from Utils.utilities import FaceType
            ft = app.face_type.to(device=dev).reshape(n_face)
            grad_interp_apply = ((ft != FaceType.CYCLIC) & (ft != FaceType.EMPTY)).to(torch.int32)

        if not self.is_cmpt and getattr(app, "_diffusion_sngrad_face", False):
            bc_diff_mask = app._bc_diffusion_face_mask().to(device=dev, dtype=torch.int32).reshape(n_face)
        else:
            bc_diff_mask = torch.zeros(n_face, dtype=torch.int32, device=dev)

        if getattr(app, "_momentum_interpolation_mwi", False):
            mwi_mask = app._mwi_face_mask().to(device=dev, dtype=torch.int32).reshape(n_face)
        else:
            mwi_mask = torch.zeros(n_face, dtype=torch.int32, device=dev)

        cf_faces_all = app.cells_face.to(device=dev, dtype=torch.int64).reshape(-1)
        cf_cells_all = app.cells_face_ptr.to(device=dev, dtype=torch.int64).reshape(-1)
        cf_sarea_all = app.directed_area_cells_face.to(device=dev, dtype=dt).reshape(-1)
        keep_ne = app.mask_non_empty_cells_face.to(device=dev, dtype=torch.bool).reshape(-1)

        c_off, c_perm = _csr_from_rows(cf_cells_all[keep_ne], n_int)
        c_face = cf_faces_all[keep_ne][c_perm].to(torch.int32).contiguous()
        c_sarea = cf_sarea_all[keep_ne][c_perm].contiguous()

        c_all_off, c_all_perm = _csr_from_rows(cf_cells_all, n_int)
        c_all_face = cf_faces_all[c_all_perm].to(torch.int32).contiguous()
        c_all_sarea = cf_sarea_all[c_all_perm].contiguous()

        f_int_o = cpd_to_int[C].contiguous()
        f_int_n = cpd_to_int[F].contiguous()
        f_sarea_o = torch.zeros(n_face, dtype=dt, device=dev)
        f_sarea_n = torch.zeros(n_face, dtype=dt, device=dev)

        ne_f = cf_faces_all[keep_ne]
        ne_c = cf_cells_all[keep_ne].to(torch.int32)
        ne_s = cf_sarea_all[keep_ne]
        match_o = ne_c == f_int_o[ne_f]
        f_sarea_o.index_add_(0, ne_f[match_o], ne_s[match_o])
        match_n = (~match_o) & (ne_c == f_int_n[ne_f])
        f_sarea_n.index_add_(0, ne_f[match_n], ne_s[match_n])

        f_all_sarea_o = torch.zeros(n_face, dtype=dt, device=dev)
        f_all_sarea_n = torch.zeros(n_face, dtype=dt, device=dev)
        all_c32 = cf_cells_all.to(torch.int32)
        match_all_o = all_c32 == f_int_o[cf_faces_all]
        f_all_sarea_o.index_add_(0, cf_faces_all[match_all_o], cf_sarea_all[match_all_o])
        match_all_n = (~match_all_o) & (all_c32 == f_int_n[cf_faces_all])
        f_all_sarea_n.index_add_(0, cf_faces_all[match_all_n], cf_sarea_all[match_all_n])

        cpd_rows = torch.cat((C, F), dim=0)
        f_ids2 = torch.cat(
            (torch.arange(n_face, device=dev, dtype=torch.int32), torch.arange(n_face, device=dev, dtype=torch.int32)),
            dim=0,
        )
        f_sides2 = torch.cat(
            (torch.zeros(n_face, device=dev, dtype=torch.int32), torch.ones(n_face, device=dev, dtype=torch.int32)),
            dim=0,
        )
        cpd_f_off, cpd_f_perm = _csr_from_rows(cpd_rows, n_cpd)
        cpd_f_idx = f_ids2[cpd_f_perm].contiguous()
        cpd_f_side = f_sides2[cpd_f_perm].contiguous()

        cyc_partner_all = torch.full((n_cpd,), -1, dtype=torch.int64, device=dev)
        if gh_cyc.numel() > 0:
            cyc_partner_all[gh_cyc.to(device=dev, dtype=torch.int64)] = pa_cyc.to(device=dev, dtype=torch.int64)
        n_src_cpd = torch.where(is_int[F], F, cyc_partner_all[F])
        valid_n_src = n_src_cpd >= 0
        r_rows = torch.cat((f_int_o.to(torch.int64), cpd_to_int[n_src_cpd[valid_n_src]].to(torch.int64)), dim=0)
        r_faces = torch.cat(
            (
                torch.arange(n_face, device=dev, dtype=torch.int32),
                torch.nonzero(valid_n_src, as_tuple=False).reshape(-1).to(torch.int32),
            ),
            dim=0,
        )
        r_sides = torch.cat(
            (
                torch.zeros(n_face, device=dev, dtype=torch.int32),
                torch.ones(int(valid_n_src.sum().item()), device=dev, dtype=torch.int32),
            ),
            dim=0,
        )
        rhat_off, rhat_perm = _csr_from_rows(r_rows, n_int)
        rhat_face = r_faces[rhat_perm].contiguous()
        rhat_side = r_sides[rhat_perm].contiguous()

        vface = app._facecv_Vface.to(device=dev, dtype=dt).reshape(n_face).contiguous()
        twosided = app._facecv_face_twosided.to(device=dev, dtype=torch.int32).reshape(n_face).contiguous()
        active_face = app._facecv_active_face.to(device=dev, dtype=torch.int32).reshape(n_face).contiguous()
        vol = app.cell_volume.to(device=dev, dtype=dt).reshape(n_int).contiguous()

        self._static_refs = {
            "cpd_of": cpd_of.to(torch.int32).contiguous(),
            "cpd_to_int": cpd_to_int.contiguous(),
            "w_off": w_off.contiguous(),
            "w_other": w_other.contiguous(),
            "w_vec": w_vec.contiguous(),
            "w_self": w_self.contiguous(),
            "wt_off": wt_off.contiguous(),
            "wt_cell": wt_cell.contiguous(),
            "wt_vec": wt_vec.contiguous(),
            "ghost_cpd": ghost_cpd.to(torch.int32).contiguous(),
            "ghost_face": ghost_face.to(torch.int32).contiguous(),
            "cyc_partner_ghost": cyc_partner_ghost.contiguous(),
            "cyc_off": cyc_off.contiguous(),
            "cyc_ghost_of_int": cyc_ghost_of_int.contiguous(),
            "fo": fo,
            "fn": fn,
            "gC": gC,
            "gF": gF,
            "nrm": nrm,
            "rcorr": rcorr,
            "Cf": Cf,
            "Ff": Ff,
            "eCF": eCF,
            "inv_dCF": inv_dCF,
            "delta_c": delta_c,
            "nonorth": nonorth,
            "ow_u": ow_u.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "nw_u": nw_u.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "cv_u": cv_u.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous(),
            "ow_p": ow_p.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "nw_p": nw_p.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "cv_p": cv_p.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous(),
            "ow_t": ow_t.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "nw_t": nw_t.to(device=dev, dtype=dt).reshape(n_face).contiguous(),
            "cv_t": cv_t.to(device=dev, dtype=dt).reshape(n_face, 3).contiguous(),
            "is_int_face": is_int_face,
            "bnc": bnc,
            "grad_interp_apply": grad_interp_apply.contiguous(),
            "mwi_mask": mwi_mask.contiguous(),
            "bc_diff_mask": bc_diff_mask.contiguous(),
            "zero_flux": self.bc_engine.zero_mass_flux_face.contiguous(),
            "grad_u_bc_code": self.bc_engine.grad_u_bc_code.contiguous(),
            "grad_t_bc_code": self.bc_engine.grad_t_bc_code.contiguous(),
            "sym_delta_coeff": self.bc_engine.sym_delta_coeff.contiguous(),
            "c_off": c_off.contiguous(),
            "c_face": c_face.contiguous(),
            "c_sarea": c_sarea.contiguous(),
            "c_all_off": c_all_off.contiguous(),
            "c_all_face": c_all_face.contiguous(),
            "c_all_sarea": c_all_sarea.contiguous(),
            "f_int_o": f_int_o,
            "f_int_n": f_int_n,
            "f_sarea_o": f_sarea_o.contiguous(),
            "f_sarea_n": f_sarea_n.contiguous(),
            "f_all_sarea_o": f_all_sarea_o.contiguous(),
            "f_all_sarea_n": f_all_sarea_n.contiguous(),
            "cpd_f_off": cpd_f_off.contiguous(),
            "cpd_f_idx": cpd_f_idx.contiguous(),
            "cpd_f_side": cpd_f_side.contiguous(),
            "rhat_off": rhat_off.contiguous(),
            "rhat_face": rhat_face.contiguous(),
            "rhat_side": rhat_side.contiguous(),
            "vface": vface,
            "twosided": twosided,
            "active_face": active_face,
            "vol": vol,
        }

        vec3_keys = {"w_vec", "w_self", "wt_vec", "nrm", "rcorr", "Cf", "Ff", "eCF", "cv_u", "cv_p", "cv_t"}
        self.ws = {
            k: _wp_from_torch(v, dtype=v3_wp) if k in vec3_keys else _wp_from_torch(v)
            for k, v in self._static_refs.items()
        }

    def refresh_dynamic_coeffs(self, app: Any) -> None:
        """Refresh PDE / scheme coefficients from ``app`` (cheap vector ops)."""
        dev = self.device
        dt = self.dtype
        n_int = self.n_int
        n_face = self.n_face

        fvs = app.params.fvschemes
        self.interp_mode = _parse_interp_mode(fvs.get("interpolationSchemes", "Linear"))
        self.conv_u_mode, self.bounded_u = _parse_conv_mode(getattr(app, "U_scheme", "Linear"))
        if self.energy:
            self.conv_t_mode, self.bounded_t = _parse_conv_mode(app.T_scheme)
        else:
            self.conv_t_mode, self.bounded_t = CONV_LINEAR, 0
        self.bounded_any = 1 if (self.bounded_u or self.bounded_t) else 0
        self.sngrad_mode = _parse_sngrad_mode(getattr(app, "_snGrad_scheme", "corrected"))
        self.theta_conv = float(getattr(app, "_conv_theta", 1.0))

        self.use_mwi = 1 if bool(getattr(app, "_momentum_interpolation_mwi", False)) else 0
        self.mwi_frozen = 1 if bool(getattr(app, "_mwi_frozen_coefficient", False)) else 0
        self.mom_cell_active = 1 if bool(getattr(app, "_momentum_residual_cell", True)) else 0
        self.mom_face_active = 1 if bool(getattr(app, "_momentum_residual_face", False)) else 0
        self.facecv_upwind = 1 if str(getattr(app, "_facecv_avg", "half")).lower() == "upwind" else 0
        self.facecv_face_buoy = (
            1 if (self.energy and str(getattr(app, "_facecv_buoyancy", "cell")).lower() == "face_sngrad") else 0
        )
        self.cell_row_pt_attached = (
            1 if (self.is_cmpt and self.energy and str(fvs.get("cellRowGradient", "detached")) == "attached") else 0
        )

        vol = self._static_refs["vol"]
        tcoef = (
            app.unsteady_coeff.to(device=dev, dtype=dt).reshape(n_int)
            * vol
            / app.dt_cell.to(device=dev, dtype=dt).reshape(n_int)
        ).contiguous()
        pcoef_vol = (app.theta_PDE[app.batch_cell, 3].to(device=dev, dtype=dt).reshape(n_int) * vol).contiguous()

        def _to_face(coeff: torch.Tensor) -> torch.Tensor:
            c_d = coeff.to(device=dev, dtype=dt)
            if c_d.numel() == n_face:
                return c_d.reshape(n_face).contiguous()
            if c_d.shape[0] == int(app.num_graphs):
                return c_d[app.batch_face].reshape(n_face).contiguous()
            return c_d.reshape(1).expand(n_face).contiguous()

        ccont_face = _to_face(app.continuity_eq_coeff)
        cc_face = _to_face(app.convection_coeff)
        gpc_face = _to_face(app.grad_p_coeff)
        nu_face = _to_face(app.nu_coeff)

        if self.energy:
            alpha_face = _to_face(app.alpha_coeff)
            ghb_face = self.bc_engine._torch_refs["ghb_face_all"]
            beta_c = app.beta[app.batch_cell].to(device=dev, dtype=dt).reshape(n_int)
            g_c = app.gravity[app.batch_cell].to(device=dev, dtype=dt).reshape(n_int, 3)
            ghref_c = torch.norm(g_c, dim=1) * app.hRef[app.batch_cell].to(device=dev, dtype=dt).reshape(n_int)
            pos_int = app.cpd_cell_pos.index_select(0, app._interior_cell_indices).to(device=dev, dtype=dt)
            gh_c = (pos_int * g_c).sum(dim=1) - ghref_c
            ghb_vol = (gh_c * beta_c * vol).contiguous()
            src_u_base = app.source_term.to(device=dev, dtype=dt).expand(n_int, 3).contiguous()
            src_t_base = app.heat_source_term.to(device=dev, dtype=dt).reshape(n_int).contiguous()
        else:
            alpha_face = torch.zeros(n_face, dtype=dt, device=dev)
            ghb_face = torch.zeros(n_face, dtype=dt, device=dev)
            ghb_vol = torch.zeros(n_int, dtype=dt, device=dev)
            src_u_base = app.source_term_base.to(device=dev, dtype=dt).expand(n_int, 3).contiguous()
            src_t_base = torch.zeros(n_int, dtype=dt, device=dev)

        c_face = self._static_refs["c_face"]
        c_sarea = self._static_refs["c_sarea"]
        nonorth = self._static_refs["nonorth"]
        c_visc_ap = (nu_face[c_face] * c_sarea.abs() * nonorth[c_face]).contiguous()

        self._dyn_refs = {
            "tcoef": tcoef,
            "pcoef_vol": pcoef_vol,
            "ccont_face": ccont_face,
            "cc_face": cc_face,
            "gpc_face": gpc_face,
            "nu_face": nu_face,
            "alpha_face": alpha_face,
            "ghb_face": ghb_face,
            "ghb_vol": ghb_vol,
            "src_u_base": src_u_base,
            "src_t_base": src_t_base,
            "c_visc_ap": c_visc_ap,
        }
        self.wd = {k: _wp_from_torch(v) for k, v in self._dyn_refs.items()}

    def run_core_forward(self, phi_old_bc: torch.Tensor, phi_new_p1: torch.Tensor):
        T = int(phi_new_p1.shape[0])
        dev = self.device
        dt = self.dtype
        K = self.kernels
        ws = self.ws
        wd = self.wd
        bc_w = self.bc_engine.w
        wp_scalar = wp.float64 if self.is_f64 else wp.float32
        stream = wp.stream_from_torch(dev)

        q_p1 = torch.empty((T, self.n_cpd, self.k_ch), dtype=dt, device=dev)
        q_p2 = torch.empty((T, self.n_cpd, self.k_ch), dtype=dt, device=dev)
        phi_new_p2 = torch.empty_like(phi_new_p1)
        grad = torch.zeros((T, self.n_cpd, self.k_ch, 3), dtype=dt, device=dev)

        f0_new = torch.empty((T, self.n_face), dtype=dt, device=dev)
        f0_old = torch.empty((T, self.n_face), dtype=dt, device=dev)
        ap_int = torch.ones((T, self.n_int), dtype=dt, device=dev)
        d_cpd = torch.zeros((T, self.n_cpd), dtype=dt, device=dev)

        fvol_pre = torch.empty((T, self.n_face), dtype=dt, device=dev)
        flux_new_bc = torch.empty((T, self.n_face), dtype=dt, device=dev)
        flux_old_bc = torch.empty((T, self.n_face), dtype=dt, device=dev)
        mwi_br = torch.empty((T, self.n_face), dtype=dt, device=dev)
        fmass = torch.empty((T, self.n_face), dtype=dt, device=dev)
        fmom = torch.empty((T, self.n_face, 3), dtype=dt, device=dev)
        fen = torch.zeros((T, self.n_face), dtype=dt, device=dev)
        sngrad_p = torch.empty((T, self.n_face), dtype=dt, device=dev)
        sngrad_t = torch.zeros((T, self.n_face), dtype=dt, device=dev)

        div_phi_new = torch.zeros((T, self.n_int), dtype=dt, device=dev)
        div_phi_old = torch.zeros((T, self.n_int), dtype=dt, device=dev)
        res_cont = torch.empty((T, self.n_int), dtype=dt, device=dev)
        res_mom_pfree = torch.empty((T, self.n_int, 3), dtype=dt, device=dev)
        res_mom = torch.empty((T, self.n_int, 3), dtype=dt, device=dev)
        res_en = torch.zeros((T, self.n_int), dtype=dt, device=dev)
        rhat_cpd = torch.zeros((T, self.n_cpd, 3), dtype=dt, device=dev)
        res_mom_face = torch.zeros((T, self.n_face), dtype=dt, device=dev)

        scale_cont = torch.empty((T, self.n_int), dtype=dt, device=dev)
        scale_mom = torch.empty((T, self.n_int, 3), dtype=dt, device=dev)
        scale_en = torch.zeros((T, self.n_int), dtype=dt, device=dev)

        w_q1 = _wp_from_torch(q_p1)
        w_q2 = _wp_from_torch(q_p2)
        w_p2_out = _wp_from_torch(phi_new_p2)
        w_grad = _wp_from_torch(grad)
        w_f0n = _wp_from_torch(f0_new)
        w_f0o = _wp_from_torch(f0_old)
        w_ap = _wp_from_torch(ap_int)
        w_dcpd = _wp_from_torch(d_cpd)

        wp.launch(
            K["pack_q"],
            dim=(T, self.n_cpd),
            inputs=[_wp_from_torch(phi_old_bc.contiguous()), _wp_from_torch(phi_new_p1.contiguous()), self.energy],
            outputs=[w_q1, w_q2, w_p2_out],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        wp.launch(
            K["wlsq_fwd"],
            dim=(T, self.n_int, self.k_ch),
            inputs=[w_q1, ws["cpd_of"], ws["w_off"], ws["w_other"], ws["w_vec"]],
            outputs=[w_grad],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        if self.n_ghost > 0:
            wp.launch(
                K["ghost_grad_fwd"],
                dim=(T, self.n_ghost, self.k_ch),
                inputs=[ws["ghost_cpd"], ws["cyc_partner_ghost"]],
                outputs=[w_grad],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
        wp.launch(
            K["face_pre"],
            dim=(T, self.n_face),
            inputs=[
                w_q1, w_grad, ws["fo"], ws["fn"], ws["gC"], ws["gF"],
                ws["rcorr"], ws["Cf"], ws["Ff"], ws["nrm"], ws["is_int_face"],
                self.interp_mode,
            ],
            outputs=[w_f0n, w_f0o],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        if self.use_mwi == 1:
            wp.launch(
                K["mwi_ap"],
                dim=(T, self.n_int),
                inputs=[
                    w_f0n, ws["cpd_of"], ws["c_off"], ws["c_face"],
                    ws["c_sarea"], wd["c_visc_ap"], ws["vol"],
                ],
                outputs=[w_ap, w_dcpd],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
            if self.n_ghost > 0:
                wp.launch(
                    K["mwi_dghost"],
                    dim=(T, self.n_ghost),
                    inputs=[ws["ghost_cpd"], ws["cyc_partner_ghost"]],
                    outputs=[w_dcpd],
                    device=self.wp_dev,
                    stream=stream,
                    record_tape=False,
                )
        if self.n_ghost > 0:
            wp.launch(
                K["pass2_ghost"],
                dim=(T, self.n_ghost),
                inputs=[
                    w_q1, w_f0n, bc_w["ghost_cpd"], bc_w["owner_cpd"], ws["ghost_face"],
                    bc_w["nrm"], bc_w["u_code"], bc_w["u_val"], bc_w["adv_dt_dx"],
                    bc_w["adv_floor"], self.include_old_p1, self.bc_engine.wzero,
                ],
                outputs=[w_q2, w_p2_out],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        w_fvol = _wp_from_torch(fvol_pre)
        w_fln = _wp_from_torch(flux_new_bc)
        w_flo = _wp_from_torch(flux_old_bc)
        w_br = _wp_from_torch(mwi_br)
        w_fmass = _wp_from_torch(fmass)
        w_fmom = _wp_from_torch(fmom)
        w_fen = _wp_from_torch(fen)
        w_snp = _wp_from_torch(sngrad_p)
        w_snt = _wp_from_torch(sngrad_t)

        wp.launch(
            K["face_main"],
            dim=(T, self.n_face),
            inputs=[
                w_q1, w_q2, w_grad, w_f0n, w_f0o, w_dcpd,
                ws["fo"], ws["fn"], ws["gC"], ws["gF"], ws["nrm"],
                ws["rcorr"], ws["Cf"], ws["Ff"], ws["eCF"], ws["inv_dCF"],
                ws["delta_c"], ws["nonorth"],
                ws["ow_u"], ws["nw_u"], ws["cv_u"],
                ws["ow_p"], ws["nw_p"], ws["cv_p"],
                ws["ow_t"], ws["nw_t"], ws["cv_t"],
                ws["is_int_face"], ws["bnc"], ws["grad_interp_apply"],
                ws["mwi_mask"], ws["bc_diff_mask"], ws["zero_flux"],
                ws["grad_u_bc_code"], ws["grad_t_bc_code"], ws["sym_delta_coeff"],
                wd["ghb_face"], wd["gpc_face"], wd["cc_face"], wd["ccont_face"],
                wd["nu_face"], wd["alpha_face"],
                self.energy, self.is_cmpt, self.use_mwi,
                self.sngrad_mode, self.conv_u_mode, self.conv_t_mode,
                wp_scalar(self.theta_conv), self.facecv_face_buoy,
            ],
            outputs=[w_fvol, w_fln, w_flo, w_br, w_fmass, w_fmom, w_fen, w_snp, w_snt],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        w_rhat = _wp_from_torch(rhat_cpd)
        wp.launch(
            K["cell_balance"],
            dim=(T, self.n_int),
            inputs=[
                w_q2, w_grad, w_fln, w_flo, w_fmass, w_fmom, w_fen,
                ws["cpd_of"], ws["c_off"], ws["c_face"], ws["c_sarea"],
                ws["c_all_off"], ws["c_all_face"], ws["c_all_sarea"],
                ws["vol"], wd["tcoef"], wd["pcoef_vol"], wd["ghb_vol"],
                wd["src_u_base"], wd["src_t_base"],
                self.energy, self.mom_cell_active, self.bounded_u, self.bounded_t,
                wp_scalar(self.theta_conv), self.facecv_face_buoy,
            ],
            outputs=[
                _wp_from_torch(div_phi_new),
                _wp_from_torch(div_phi_old),
                _wp_from_torch(res_cont),
                _wp_from_torch(res_mom_pfree),
                _wp_from_torch(res_mom),
                _wp_from_torch(res_en),
                w_rhat,
                _wp_from_torch(scale_cont),
                _wp_from_torch(scale_mom),
                _wp_from_torch(scale_en),
            ],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        if self.mom_face_active == 1:
            if self.n_ghost > 0:
                wp.launch(
                    K["rhat_ghost"],
                    dim=(T, self.n_ghost),
                    inputs=[ws["ghost_cpd"], ws["cyc_partner_ghost"]],
                    outputs=[w_rhat],
                    device=self.wp_dev,
                    stream=stream,
                    record_tape=False,
                )
            wp.launch(
                K["face_cv_fwd"],
                dim=(T, self.n_face),
                inputs=[
                    w_rhat, w_snp, w_snt, w_f0n, w_fln, ws["fo"], ws["fn"], ws["nrm"],
                    wd["gpc_face"], wd["ghb_face"], ws["vface"], ws["twosided"], ws["active_face"],
                    self.facecv_upwind, self.upwind_uses_flux_bc, self.facecv_face_buoy,
                ],
                outputs=[_wp_from_torch(res_mom_face)],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        saved = (
            q_p1,
            q_p2,
            grad,
            f0_new,
            flux_new_bc,
            flux_old_bc,
            ap_int,
            d_cpd,
            mwi_br,
            div_phi_new,
            div_phi_old,
        )
        aux = (scale_cont, scale_mom, scale_en, fvol_pre, flux_new_bc, grad)
        return res_cont, res_mom, res_mom_face, res_en, phi_new_p2, saved, aux

    def run_core_backward(
        self,
        saved,
        adj_res_cont: torch.Tensor,
        adj_res_mom: torch.Tensor,
        adj_res_mom_face: torch.Tensor,
        adj_res_en: torch.Tensor,
        adj_phi_new_p2: torch.Tensor,
        wd_override: Optional[dict] = None,
        bc_w_override: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            q_p1,
            q_p2,
            grad,
            f0_new,
            flux_new_bc,
            flux_old_bc,
            ap_int,
            d_cpd,
            mwi_br,
            div_phi_new,
            div_phi_old,
        ) = saved

        T = int(q_p1.shape[0])
        dev = self.device
        dt = self.dtype
        K = self.kernels
        ws = self.ws
        wd = self.wd if wd_override is None else wd_override
        bc_w = self.bc_engine.w if bc_w_override is None else bc_w_override
        wp_scalar = wp.float64 if self.is_f64 else wp.float32
        stream = wp.stream_from_torch(dev)

        adj_sngrad_p = torch.zeros((T, self.n_face), dtype=dt, device=dev)
        adj_sngrad_t = torch.zeros((T, self.n_face), dtype=dt, device=dev)
        emit_rhat_o = torch.zeros((T, self.n_face, 3), dtype=dt, device=dev)
        emit_rhat_n = torch.zeros((T, self.n_face, 3), dtype=dt, device=dev)

        if self.mom_face_active == 1:
            wp.launch(
                K["face_cv_bwd_emit"],
                dim=(T, self.n_face),
                inputs=[
                    _wp_from_torch(adj_res_mom_face.contiguous()),
                    _wp_from_torch(f0_new),
                    _wp_from_torch(flux_new_bc),
                    ws["nrm"],
                    wd["gpc_face"],
                    wd["ghb_face"],
                    ws["vface"],
                    ws["twosided"],
                    ws["active_face"],
                    self.facecv_upwind,
                    self.upwind_uses_flux_bc,
                    self.facecv_face_buoy,
                ],
                outputs=[
                    _wp_from_torch(adj_sngrad_p),
                    _wp_from_torch(adj_sngrad_t),
                    _wp_from_torch(emit_rhat_o),
                    _wp_from_torch(emit_rhat_n),
                ],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        adj_pfree = torch.empty((T, self.n_int, 3), dtype=dt, device=dev)
        adj_div_new = torch.empty((T, self.n_int), dtype=dt, device=dev)
        adj_div_old = torch.empty((T, self.n_int), dtype=dt, device=dev)
        adj_q_p2_cell = torch.zeros((T, self.n_cpd, self.k_ch), dtype=dt, device=dev)
        adj_grad_cell = torch.zeros((T, self.n_cpd, self.k_ch, 3), dtype=dt, device=dev)

        wp.launch(
            K["cell_bwd"],
            dim=(T, self.n_int),
            inputs=[
                _wp_from_torch(q_p2),
                _wp_from_torch(div_phi_new),
                _wp_from_torch(div_phi_old),
                _wp_from_torch(adj_res_cont.contiguous()),
                _wp_from_torch(adj_res_mom.contiguous()),
                _wp_from_torch(adj_res_en.contiguous()),
                _wp_from_torch(emit_rhat_o),
                _wp_from_torch(emit_rhat_n),
                ws["rhat_off"],
                ws["rhat_face"],
                ws["rhat_side"],
                ws["cpd_of"],
                ws["vol"],
                wd["tcoef"],
                wd["pcoef_vol"],
                wd["ghb_vol"],
                self.energy,
                self.mom_cell_active,
                self.mom_face_active,
                self.bounded_u,
                self.bounded_t,
                self.grad_attached,
                self.cell_row_pt_attached,
                wp_scalar(self.theta_conv),
                self.facecv_face_buoy,
            ],
            outputs=[
                _wp_from_torch(adj_pfree),
                _wp_from_torch(adj_div_new),
                _wp_from_torch(adj_div_old),
                _wp_from_torch(adj_q_p2_cell),
                _wp_from_torch(adj_grad_cell),
            ],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        adj_f0_new = torch.empty((T, self.n_face), dtype=dt, device=dev)
        adj_f0_old = torch.empty((T, self.n_face), dtype=dt, device=dev)
        emit_d_o = torch.zeros((T, self.n_face), dtype=dt, device=dev)
        emit_d_n = torch.zeros((T, self.n_face), dtype=dt, device=dev)
        emit_q1_o = torch.zeros((T, self.n_face, 5), dtype=dt, device=dev)
        emit_q1_n = torch.zeros((T, self.n_face, 5), dtype=dt, device=dev)
        emit_q2_o = torch.zeros((T, self.n_face, self.k_ch), dtype=dt, device=dev)
        emit_q2_n = torch.zeros((T, self.n_face, self.k_ch), dtype=dt, device=dev)
        emit_grad_o = torch.zeros((T, self.n_face, self.k_ch, 3), dtype=dt, device=dev)
        emit_grad_n = torch.zeros((T, self.n_face, self.k_ch, 3), dtype=dt, device=dev)

        wp.launch(
            K["face_bwd"],
            dim=(T, self.n_face),
            inputs=[
                _wp_from_torch(q_p1),
                _wp_from_torch(q_p2),
                _wp_from_torch(grad),
                _wp_from_torch(flux_new_bc),
                _wp_from_torch(flux_old_bc),
                _wp_from_torch(d_cpd),
                _wp_from_torch(mwi_br),
                _wp_from_torch(adj_res_cont.contiguous()),
                _wp_from_torch(adj_pfree),
                _wp_from_torch(adj_res_en.contiguous()),
                _wp_from_torch(adj_div_new),
                _wp_from_torch(adj_div_old),
                _wp_from_torch(adj_sngrad_p),
                _wp_from_torch(adj_sngrad_t),
                ws["fo"], ws["fn"], ws["f_int_o"], ws["f_int_n"],
                ws["f_sarea_o"], ws["f_sarea_n"], ws["f_all_sarea_o"], ws["f_all_sarea_n"],
                ws["gC"], ws["gF"], ws["nrm"], ws["rcorr"], ws["Cf"], ws["Ff"],
                ws["eCF"], ws["inv_dCF"], ws["delta_c"], ws["nonorth"],
                ws["ow_u"], ws["nw_u"], ws["cv_u"],
                ws["ow_p"], ws["nw_p"], ws["cv_p"],
                ws["ow_t"], ws["nw_t"], ws["cv_t"],
                ws["is_int_face"], ws["bnc"], ws["grad_interp_apply"],
                ws["mwi_mask"], ws["bc_diff_mask"], ws["zero_flux"],
                ws["grad_u_bc_code"], ws["grad_t_bc_code"], ws["sym_delta_coeff"],
                wd["ghb_face"], wd["gpc_face"], wd["cc_face"], wd["ccont_face"],
                wd["nu_face"], wd["alpha_face"],
                self.energy, self.is_cmpt, self.grad_attached,
                self.use_mwi, self.mwi_frozen, self.mom_face_active, self.bounded_any,
                self.sngrad_mode, self.conv_u_mode, self.conv_t_mode,
                wp_scalar(self.theta_conv), self.facecv_face_buoy,
            ],
            outputs=[
                _wp_from_torch(adj_f0_new),
                _wp_from_torch(adj_f0_old),
                _wp_from_torch(emit_d_o),
                _wp_from_torch(emit_d_n),
                _wp_from_torch(emit_q1_o),
                _wp_from_torch(emit_q1_n),
                _wp_from_torch(emit_q2_o),
                _wp_from_torch(emit_q2_n),
                _wp_from_torch(emit_grad_o),
                _wp_from_torch(emit_grad_n),
            ],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        mwi_implicit = 1 if (self.use_mwi == 1 and self.mwi_frozen == 0) else 0
        adj_ap_int = torch.zeros((T, self.n_int), dtype=dt, device=dev)
        if mwi_implicit == 1:
            wp.launch(
                K["mwi_ap_bwd"],
                dim=(T, self.n_int),
                inputs=[
                    _wp_from_torch(ap_int),
                    ws["vol"],
                    _wp_from_torch(emit_d_o),
                    _wp_from_torch(emit_d_n),
                    ws["rhat_off"],
                    ws["rhat_face"],
                    ws["rhat_side"],
                ],
                outputs=[_wp_from_torch(adj_ap_int)],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        wp.launch(
            K["face_pre_bwd"],
            dim=(T, self.n_face),
            inputs=[
                _wp_from_torch(f0_new),
                _wp_from_torch(adj_f0_new),
                _wp_from_torch(adj_f0_old),
                _wp_from_torch(adj_ap_int),
                ws["f_int_o"], ws["f_int_n"], ws["f_sarea_o"], ws["f_sarea_n"],
                ws["gC"], ws["gF"], ws["rcorr"], ws["Cf"], ws["Ff"], ws["nrm"],
                ws["is_int_face"], self.interp_mode, self.grad_attached, mwi_implicit,
            ],
            outputs=[
                _wp_from_torch(emit_q1_o),
                _wp_from_torch(emit_q1_n),
                _wp_from_torch(emit_q2_o),
                _wp_from_torch(emit_q2_n),
                _wp_from_torch(emit_grad_o),
                _wp_from_torch(emit_grad_n),
            ],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        adj_q_p1 = torch.empty((T, self.n_cpd, self.k_ch), dtype=dt, device=dev)
        adj_q_p2_total = torch.empty((T, self.n_cpd, 3), dtype=dt, device=dev)
        adj_grad_cpd = adj_grad_cell if self.grad_attached == 0 else torch.empty_like(adj_grad_cell)

        wp.launch(
            K["cpd_gather_face_bwd"],
            dim=(T, self.n_cpd),
            inputs=[
                _wp_from_torch(adj_phi_new_p2.contiguous()),
                _wp_from_torch(adj_q_p2_cell),
                _wp_from_torch(adj_grad_cell),
                _wp_from_torch(emit_q1_o),
                _wp_from_torch(emit_q1_n),
                _wp_from_torch(emit_q2_o),
                _wp_from_torch(emit_q2_n),
                _wp_from_torch(emit_grad_o),
                _wp_from_torch(emit_grad_n),
                ws["cpd_f_off"],
                ws["cpd_f_idx"],
                ws["cpd_f_side"],
                self.energy,
                self.k_ch,
                self.grad_attached,
            ],
            outputs=[
                _wp_from_torch(adj_q_p1),
                _wp_from_torch(adj_q_p2_total),
                _wp_from_torch(adj_grad_cpd),
            ],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )

        if self.n_ghost > 0:
            wp.launch(
                K["pass2_ghost_bwd_fix"],
                dim=(T, self.n_int),
                inputs=[
                    _wp_from_torch(f0_new),
                    _wp_from_torch(adj_q_p2_total),
                    bc_w["ghost_cpd"],
                    bc_w["owner_cpd"],
                    ws["ghost_face"],
                    bc_w["u_code"],
                    bc_w["own_off"],
                    bc_w["own_g"],
                    ws["cpd_of"],
                    self.include_old_p1,
                    self.bc_engine.wzero,
                ],
                outputs=[_wp_from_torch(adj_q_p1)],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        if self.grad_attached == 1 or self.cell_row_pt_attached == 1:
            ch_off = 0 if self.grad_attached == 1 else 6
            n_ch_wlsq = self.k_ch if self.grad_attached == 1 else 2
            adj_grad_int = torch.zeros((T, self.n_int, self.k_ch, 3), dtype=dt, device=dev)
            wp.launch(
                K["wlsq_fold_cyc_bwd"],
                dim=(T, self.n_int, n_ch_wlsq),
                inputs=[
                    _wp_from_torch(adj_grad_cpd),
                    ws["cpd_of"],
                    ws["cyc_off"],
                    ws["cyc_ghost_of_int"],
                    ch_off,
                ],
                outputs=[_wp_from_torch(adj_grad_int)],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )
            wp.launch(
                K["wlsq_transpose_bwd"],
                dim=(T, self.n_cpd, n_ch_wlsq),
                inputs=[
                    _wp_from_torch(adj_grad_int),
                    ws["cpd_to_int"],
                    ws["wt_off"],
                    ws["wt_cell"],
                    ws["wt_vec"],
                    ws["w_self"],
                    ch_off,
                ],
                outputs=[_wp_from_torch(adj_q_p1)],
                device=self.wp_dev,
                stream=stream,
                record_tape=False,
            )

        adj_phi_old = torch.empty((T, self.n_cpd, self.n_vars), dtype=dt, device=dev)
        adj_phi_new = torch.empty((T, self.n_cpd, self.n_vars), dtype=dt, device=dev)
        wp.launch(
            K["unpack_adj_q"],
            dim=(T, self.n_cpd),
            inputs=[_wp_from_torch(adj_q_p1), self.energy],
            outputs=[_wp_from_torch(adj_phi_old), _wp_from_torch(adj_phi_new)],
            device=self.wp_dev,
            stream=stream,
            record_tape=False,
        )
        return adj_phi_old, adj_phi_new


class _WarpFVMCoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi_old_bc: torch.Tensor, phi_new_p1: torch.Tensor, engine: WarpIncompressibleEngine):
        res_cont, res_mom, res_mom_face, res_en, phi_new_p2, saved, aux = engine.run_core_forward(
            phi_old_bc, phi_new_p1
        )
        ctx.engine = engine
        ctx.saved_warp = saved
        ctx.dyn_refs_snap = engine._dyn_refs
        ctx.wd_snap = engine.wd
        ctx.bc_refs_snap = dict(engine.bc_engine._torch_refs)
        ctx.bc_w_snap = dict(engine.bc_engine.w)
        ctx.mark_non_differentiable(*aux)
        return (
            res_cont.unsqueeze(-1),
            res_mom,
            res_mom_face.unsqueeze(-1),
            res_en.unsqueeze(-1),
            phi_new_p2,
            *aux,
        )

    @staticmethod
    def backward(ctx, g_cont, g_mom, g_mom_face, g_en, g_phi_p2, *_aux_grads):
        engine: WarpIncompressibleEngine = ctx.engine
        saved = ctx.saved_warp
        q_p1 = saved[0]
        T = int(q_p1.shape[0])
        dev = engine.device
        dt = engine.dtype

        ac = g_cont.squeeze(-1) if g_cont is not None else torch.zeros((T, engine.n_int), dtype=dt, device=dev)
        am = g_mom if g_mom is not None else torch.zeros((T, engine.n_int, 3), dtype=dt, device=dev)
        amf = g_mom_face.squeeze(-1) if g_mom_face is not None else torch.zeros((T, engine.n_face), dtype=dt, device=dev)
        ae = g_en.squeeze(-1) if g_en is not None else torch.zeros((T, engine.n_int), dtype=dt, device=dev)
        ap2 = g_phi_p2 if g_phi_p2 is not None else torch.zeros((T, engine.n_cpd, engine.n_vars), dtype=dt, device=dev)

        adj_old, adj_new = engine.run_core_backward(
            saved,
            ac,
            am,
            amf,
            ae,
            ap2,
            wd_override=ctx.wd_snap,
            bc_w_override=ctx.bc_w_snap,
        )
        return (
            adj_old if ctx.needs_input_grad[0] else None,
            adj_new if ctx.needs_input_grad[1] else None,
            None,
        )


def get_or_build_warp_engine(solver: Any, dtype: torch.dtype) -> WarpIncompressibleEngine:
    """Return cached ``WarpIncompressibleEngine`` on ``solver``, rebuilding only when mesh/BC-structure/dtype changes."""
    from collections import OrderedDict

    mesh_sig = getattr(solver, "_geom_mesh_sig", None)
    bc_sig = bc_structure_signature(solver)
    key = (mesh_sig, bc_sig, dtype, solver.cpd_cell_pos.device)
    cache = getattr(solver, "_warp_fvm_engine_lru", None)
    if cache is None:
        cache = OrderedDict()
        solver._warp_fvm_engine_lru = cache
    eng = cache.get(key)
    if eng is not None:
        cache.move_to_end(key)
        eng.bc_engine.refresh_bc_values(solver)
        eng.refresh_dynamic_coeffs(solver)
        solver._warp_fvm_engine_cache = (key, eng)
        return eng
    eng = WarpIncompressibleEngine(solver, dtype)
    if len(cache) >= 16:
        cache.popitem(last=False)
    cache[key] = eng
    solver._warp_fvm_engine_cache = (key, eng)
    return eng


def _require_warp_supported(solver: Any) -> None:
    ok, reason = can_use_warp(solver)
    if not ok:
        raise NotImplementedError(f"[WarpIncompressibleEngine] Unsupported solver configuration: {reason}")


def warp_solver_enforce_bcs_only(
    solver: Any,
    phi_old_cpd_cell: torch.Tensor,
    phi_new_cpd_cell: torch.Tensor,
) -> torch.Tensor:
    _require_warp_supported(solver)
    eng = get_or_build_warp_engine(solver, phi_new_cpd_cell.dtype)
    _, phi_new_bced = warp_enforce_bcs(eng.bc_engine, phi_old_cpd_cell, phi_new_cpd_cell, include_old_u=1)
    return phi_new_bced


def warp_solver_forward(
    solver: Any,
    phi_old_cpd_cell: torch.Tensor,
    phi_new_cpd_cell: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute full BC + FVM residual + per-graph loss assembly in Warp + PyTorch autograd."""
    _require_warp_supported(solver)
    is_2d = phi_new_cpd_cell.ndim == 2
    old_3d = phi_old_cpd_cell.unsqueeze(0) if is_2d else phi_old_cpd_cell
    new_3d = phi_new_cpd_cell.unsqueeze(0) if is_2d else phi_new_cpd_cell

    eng = get_or_build_warp_engine(solver, new_3d.dtype)
    old_bc, new_p1 = warp_enforce_bcs(
        eng.bc_engine, old_3d, new_3d, include_old_u=eng.include_old_p1
    )

    (
        res_cont,
        res_mom,
        res_mom_face,
        res_en,
        new_p2,
        scale_cont,
        scale_mom,
        scale_en,
        fvol_pre,
        fvol_bc,
        grad_cpd,
    ) = _WarpFVMCoreFn.apply(old_bc, new_p1, eng)

    def _sq0(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if is_2d else x

    phi_bced = _sq0(new_p2)
    res_cont_u = _sq0(res_cont)
    res_mom_u = _sq0(res_mom)
    res_mom_face_u = _sq0(res_mom_face)
    res_en_u = _sq0(res_en)

    solver.bc_uvw_cpd_cell = phi_bced[..., 0:3]
    if eng.energy:
        solver.bc_p_rgh_cpd_cell = phi_bced[..., 3:4]
        solver.bc_t_cpd_cell = phi_bced[..., 4:5]
        if eng.is_cmpt:
            solver._grad_uvwpt_hat_cpd_cell = _sq0(grad_cpd).detach()
        else:
            solver.face_volumetric_flux = _sq0(fvol_pre.unsqueeze(-1)).detach()
            solver.face_volumetric_flux_bc = _sq0(fvol_bc.unsqueeze(-1)).detach()
    else:
        solver.bc_p_cpd_cell = phi_bced[..., 3:4]
        if not eng.is_cmpt:
            solver.face_volumetric_flux = _sq0(fvol_pre.unsqueeze(-1)).detach()
            solver._raw_res_mom_cell = res_mom_u
            solver._raw_res_cont_cell = res_cont_u

    solver.residual_continuity = res_cont_u
    solver.residual_momentum = res_mom_u
    if eng.mom_face_active:
        solver.residual_momentum_face = res_mom_face_u
    if eng.energy:
        solver.residual_energy = res_en_u
        solver.residual_phi_jump = phi_bced.new_zeros(
            (*phi_bced.shape[:-2], solver.num_cells, 4 if eng.dim == 2 else 5)
        )
        solver._checker_loss = None

    eps_sqrt = 1.0e-16
    num_g = int(solver.num_graphs)
    batch_c = solver.batch_cell
    batch_f = solver.batch_face

    with torch.no_grad():
        sc_c = _sq0(scale_cont.unsqueeze(-1))
        sc_m = _sq0(scale_mom)
        solver._norm_scale_cont = sc_c
        solver._norm_scale_mom = sc_m
        solver.norm_factor_cont = torch.sqrt(global_add_pool(sc_c ** 2, batch=batch_c, size=num_g) + eps_sqrt)
        solver.norm_factor_mom = torch.sqrt(global_add_pool(sc_m ** 2, batch=batch_c, size=num_g) + eps_sqrt)
        if eng.energy:
            sc_e = _sq0(scale_en.unsqueeze(-1))
            solver._norm_scale_energy = sc_e
            solver.norm_factor_energy = torch.sqrt(global_add_pool(sc_e ** 2, batch=batch_c, size=num_g) + eps_sqrt)

    loss_cont = torch.sqrt(global_add_pool(res_cont_u ** 2, batch=batch_c, size=num_g) + eps_sqrt)
    if not eng.energy or eng.is_cmpt:
        loss_cont = loss_cont.sum(dim=-1, keepdim=True)

    loss_mom_parts = []
    if eng.mom_cell_active:
        loss_mom_parts.append(
            torch.sqrt(global_add_pool(res_mom_u ** 2, batch=batch_c, size=num_g) + eps_sqrt)[..., 0 : eng.dim]
        )
    if eng.mom_face_active:
        loss_mom_parts.append(
            torch.sqrt(global_add_pool(res_mom_face_u ** 2, batch=batch_f, size=num_g) + eps_sqrt)
        )
    loss_momentum_raw = loss_mom_parts[0] if len(loss_mom_parts) == 1 else torch.cat(loss_mom_parts, dim=-1)

    loss_phi_jump = phi_bced.new_zeros((*phi_bced.shape[:-2], num_g, 1))
    solver.loss_cont_component = loss_cont
    solver.loss_momentum_component = loss_momentum_raw
    solver.loss_phi_jump_component = loss_phi_jump

    if solver.loss_mom_adaptive_balance:
        mom_det = loss_momentum_raw.detach()
        mom_max = torch.amax(mom_det, dim=-1, keepdim=True)
        w_bal = mom_max / (mom_det + 1.0e-12)
        if not eng.energy:
            w_bal = torch.clamp(w_bal, max=float(getattr(solver, "_max_balance_weight", 10.0)))
        loss_momentum = loss_momentum_raw * w_bal
    else:
        loss_momentum = loss_momentum_raw

    if eng.energy:
        loss_energy = torch.sqrt(global_add_pool(res_en_u ** 2, batch=batch_c, size=num_g) + eps_sqrt)
        solver.loss_energy_component = loss_energy
        if eng.is_cmpt:
            loss_p_rgh_ref = solver.loss_setp_rghRef(
                p_rgh_cpd_cell=phi_bced[..., 3:4],
                t_cpd_cell=phi_bced[..., 4:5],
            )
            solver.loss_p_rghRef_component = loss_p_rgh_ref
            loss_inner = (
                solver.loss_cont_weight * loss_cont.sum(dim=-1, keepdim=True)
                + solver.loss_mom_weight * loss_momentum.sum(dim=-1, keepdim=True)
                + solver.loss_energy_weight * loss_energy.sum(dim=-1, keepdim=True)
                + solver.loss_phi_jump_weight * loss_phi_jump.sum(dim=-1, keepdim=True)
                + solver.loss_p_Ref_weight * loss_p_rgh_ref.sum(dim=-1, keepdim=True)
            )
        else:
            loss_p_rgh_ref = solver.loss_setp_rghRef(p_rgh_cpd_cell=phi_bced[..., 3:4])
            solver.loss_p_rghRef_component = loss_p_rgh_ref
            loss_mass_balance = torch.abs(global_add_pool(res_cont_u, batch=batch_c, size=num_g))
            solver.loss_mass_balance_component = loss_mass_balance
            loss_inner = (
                solver.loss_cont_weight * loss_cont.sum(dim=-1, keepdim=True)
                + solver.loss_mom_weight * loss_momentum.sum(dim=-1, keepdim=True)
                + solver.loss_energy_weight * loss_energy.sum(dim=-1, keepdim=True)
                + solver.loss_phi_jump_weight * loss_phi_jump.sum(dim=-1, keepdim=True)
                + solver.loss_p_Ref_weight * loss_p_rgh_ref.sum(dim=-1, keepdim=True)
                + solver.loss_mass_balance_weight * loss_mass_balance.sum(dim=-1, keepdim=True)
            )
        loss_total = torch.log(loss_inner).squeeze(-1)
    else:
        loss_p_ref = solver.loss_setpRef(p_cpd_cell=phi_bced[..., 3:4])
        solver.loss_pRef_component = loss_p_ref
        loss_inner = (
            solver.loss_cont_weight * loss_cont.sum(dim=-1, keepdim=True)
            + solver.loss_mom_weight * loss_momentum.sum(dim=-1, keepdim=True)
            + solver.loss_phi_jump_weight * loss_phi_jump.sum(dim=-1, keepdim=True)
            + solver.loss_p_Ref_weight * loss_p_ref.sum(dim=-1, keepdim=True)
            + 1.0e-8
        )
        loss_total = torch.log(loss_inner).squeeze(-1)

    return loss_total, phi_bced
