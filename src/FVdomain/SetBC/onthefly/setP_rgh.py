"""Boundary conditions of p_rgh = p - rho_k g.h."""

from typing import Dict

import torch

from FVdomain.SetBC.onthefly.BCbase import BoundaryCondition, BCType, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.scalar import ScalarCyclic, ScalarFixedValue, ScalarZeroGradient


class Cyclic(ScalarCyclic):
    FIELD, VAR = "p_rgh_cpd_cell", VARType.PRESSURE_RGH


class Fixedvalue(ScalarFixedValue):
    FIELD, VAR = "p_rgh_cpd_cell", VARType.PRESSURE_RGH


class Zerogradient(ScalarZeroGradient):
    FIELD, VAR = "p_rgh_cpd_cell", VARType.PRESSURE_RGH


class Symmetry(ScalarZeroGradient):
    """Symmetry plane: zero normal gradient."""
    FIELD, VAR = "p_rgh_cpd_cell", VARType.PRESSURE_RGH


class Fixedfluxpressure(BoundaryCondition):
    """OpenFOAM ``fixedFluxPressure`` on walls: the normal gradient of p_rgh balances the
    buoyancy term, ``snGrad(p_rgh) = gh_f beta snGrad(T)`` (``constrainPressure``)."""
    # its ghost value reads T's ghost on the same patch, so it runs after the other
    # Neumann-type conditions of its case
    APPLY_LAST = True

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        self.value = parametric_manager.resolve_value(boundary_info.get("value", {"value_type": "uniform", "val": 0.0}))
        super().__init__(patch_name, patch_idx, VARType.PRESSURE_RGH, BCType.MIXED)

    def init_field(self, p_rgh_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        p_rgh_cpd_cell[s:e, 0:1] = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"], num_element=e - s, channel_dim=1,
        ).eval(device=p_rgh_cpd_cell.device)
        return p_rgh_cpd_cell

    @staticmethod
    def _buoyancy_snGrad_p_rgh(integrator, s_f, e_f, snGrad_T):
        batch_face_bdy = integrator.batch_face[s_f:e_f]
        g_bdy = integrator.gravity[batch_face_bdy]
        ghf = ((integrator.face_pos[s_f:e_f] * g_bdy).sum(dim=1, keepdim=True)
               - torch.norm(g_bdy, dim=1, keepdim=True) * integrator.hRef[batch_face_bdy])
        return ghf * integrator.beta[batch_face_bdy] * snGrad_T

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        neighbor_cpd_cell = integrator.neighbor_cpd_cell
        s_f, e_f = integrator.fvpatch_batch.start_idx_face[g], integrator.fvpatch_batch.end_idx_face[g]

        if "p_rgh_cpd_cell" in fvfield:
            p_rgh_cpd_cell = fvfield["p_rgh_cpd_cell"]
            owner_cells = neighbor_cpd_cell[0, s_f:e_f]
            neighbor_cells = neighbor_cpd_cell[1, s_f:e_f]
            ghost = p_rgh_cpd_cell[..., owner_cells, :]
            if "t_cpd_cell" in fvfield:
                t_cpd_cell = fvfield["t_cpd_cell"]
                jump = t_cpd_cell[..., neighbor_cells, 0:1] - t_cpd_cell[..., owner_cells, 0:1]
                ghost = ghost + self._buoyancy_snGrad_p_rgh(integrator, s_f, e_f, jump)
            p_rgh_cpd_cell[..., neighbor_cells, :] = ghost
            fvfield["p_rgh_cpd_cell"] = p_rgh_cpd_cell

        if "grad_p_rgh_flux_face" in fvfield:
            snGrad_T_bdy = fvfield["grad_t_flux_face"][..., s_f:e_f, 0:1].clone()
            fvfield["grad_p_rgh_flux_face"][..., s_f:e_f, :] = self._buoyancy_snGrad_p_rgh(
                integrator, s_f, e_f, snGrad_T_bdy)
        return fvfield
