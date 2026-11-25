"""Shared scalar boundary condition descriptors (p, p_rgh, T)."""

from typing import Dict

from FVdomain.SetBC.onthefly.BCbase import BCType, BoundaryCondition
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import apply_cyclic_face_values


class _ScalarBC(BoundaryCondition):
    VAR = None
    BC_TYPE = None

    def __init__(
        self,
        patch_name: str,
        patch_idx: int,
        boundary_info: Dict,
        parametric_manager: ParametricManager = None,
    ):
        super().__init__(patch_name, patch_idx, self.VAR, self.BC_TYPE)


class ScalarEmpty(_ScalarBC):
    BC_TYPE = BCType.EMPTY

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
        neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
        if neighbor_cpd_cell is not None:
            phi_cpd_cell[s_c:e_c, :] = phi_cpd_cell[neighbor_cpd_cell[0, s_f:e_f], :]
        return phi_cpd_cell


class ScalarCyclic(_ScalarBC):
    BC_TYPE = BCType.CYCLIC

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
        if neighbor_cpd_cell is None:
            raise ValueError("No neighbor_cpd_cell provided")
        return apply_cyclic_face_values(phi_cpd_cell, fvpatch, neighbor_cpd_cell, self.patch_idx)


class ScalarFixedValue(_ScalarBC):
    BC_TYPE = BCType.FIRST

    def __init__(
        self,
        patch_name: str,
        patch_idx: int,
        boundary_info: Dict,
        parametric_manager: ParametricManager = None,
    ):
        raw_val = boundary_info["value"]
        self.value = parametric_manager.resolve_value(raw_val) if parametric_manager is not None else raw_val
        self.value_tensor = None
        self.detach_owner = bool(boundary_info.get("detach_owner", False))
        super().__init__(patch_name, patch_idx, boundary_info, parametric_manager)

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"],
            value=self.value["val"],
            num_element=e - s,
            channel_dim=1,
        )
        phi_cpd_cell[..., s:e, 0:1] = self.value_tensor.eval(device=phi_cpd_cell.device)
        return phi_cpd_cell


class ScalarZeroGradient(_ScalarBC):
    BC_TYPE = BCType.MIXED

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
        neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
        if neighbor_cpd_cell is not None:
            phi_cpd_cell[s_c:e_c, 0:1] = phi_cpd_cell[neighbor_cpd_cell[0, s_f:e_f], 0:1]
        else:
            phi_cpd_cell[s_c:e_c, 0:1] = 0.0
        return phi_cpd_cell


class ScalarCalculated(ScalarZeroGradient):
    """No-op runtime BC; initializes ghost cells from owner cells."""
