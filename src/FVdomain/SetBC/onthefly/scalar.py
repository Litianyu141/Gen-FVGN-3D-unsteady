"""Conditions shared by the scalar fields (p, p_rgh, T); a subclass names the field."""

from typing import Dict

from FVdomain.SetBC.onthefly.BCbase import BoundaryCondition, BCType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import apply_cyclic_face_values


class _ScalarBC(BoundaryCondition):
    FIELD = None      # key of the compound-cell field in ``fvfield``, e.g. "t_cpd_cell"
    VAR = None        # VARType of the field
    BC_TYPE = None

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, self.VAR, self.BC_TYPE)

    @property
    def grad_flux_key(self):
        return "grad_" + self.FIELD.replace("_cpd_cell", "_flux_face")


class ScalarCyclic(_ScalarBC):
    """Periodic: ghost values and gradients from the partner patch's owner cells."""
    BC_TYPE = BCType.CYCLIC

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        return apply_cyclic_face_values(phi_cpd_cell, fvpatch, kwargs["neighbor_cpd_cell"], self.patch_idx)

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        fvfield[self.FIELD] = apply_cyclic_face_values(
            fvfield.get(self.FIELD, None), integrator.fvpatch_batch, integrator.neighbor_cpd_cell, g)
        fvfield["grad_" + self.FIELD] = apply_cyclic_face_values(
            fvfield.get("grad_" + self.FIELD, None), integrator.fvpatch_batch, integrator.neighbor_cpd_cell, g,
            gradient=True)
        return fvfield


class ScalarFixedValue(_ScalarBC):
    """Prescribed value on the boundary faces."""
    BC_TYPE = BCType.FIRST

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        self.value = parametric_manager.resolve_value(boundary_info["value"])
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, boundary_info, parametric_manager)

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"], num_element=e - s, channel_dim=1)
        phi_cpd_cell[..., s:e, 0:1] = self.value_tensor.eval(device=phi_cpd_cell.device)
        return phi_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        phi_cpd_cell = fvfield[self.FIELD]
        s, e = integrator.fvpatch_batch.start_idx_cpd_cell[g], integrator.fvpatch_batch.end_idx_cpd_cell[g]
        phi_cpd_cell[..., s:e, 0:1] = self.value_tensor.eval(device=phi_cpd_cell.device)
        fvfield[self.FIELD] = phi_cpd_cell
        return fvfield


class ScalarZeroGradient(_ScalarBC):
    """Ghost value equal to the owner cell's; zero diffusive flux."""
    BC_TYPE = BCType.MIXED

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
        phi_cpd_cell[s_c:e_c, 0:1] = phi_cpd_cell[kwargs["neighbor_cpd_cell"][0, s_f:e_f], 0:1]
        return phi_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        neighbor_cpd_cell = integrator.neighbor_cpd_cell
        s_f, e_f = integrator.fvpatch_batch.start_idx_face[g], integrator.fvpatch_batch.end_idx_face[g]
        if self.FIELD in fvfield:
            phi_cpd_cell = fvfield[self.FIELD]
            phi_cpd_cell[..., neighbor_cpd_cell[1, s_f:e_f], :] = phi_cpd_cell[..., neighbor_cpd_cell[0, s_f:e_f], :]
            fvfield[self.FIELD] = phi_cpd_cell
        if self.grad_flux_key in fvfield:
            fvfield[self.grad_flux_key][..., s_f:e_f, :] = 0.0
        return fvfield
