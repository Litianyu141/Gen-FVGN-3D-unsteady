"""Velocity boundary conditions."""

from typing import Dict

import torch

from FVdomain.SetBC.onthefly.BCbase import BoundaryCondition, BCType, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import apply_cyclic_face_values


class Cyclic(BoundaryCondition):
    """Periodic pair of patches: ghost values and gradients from the partner cells."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.CYCLIC)

    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        return apply_cyclic_face_values(phi_cpd_cell, fvpatch, kwargs["neighbor_cpd_cell"], self.patch_idx)

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        global_patch_idx = self.patch_idx + patch_idx_offset
        fvpatch_batch = integrator.fvpatch_batch
        neighbor_cpd_cell = integrator.neighbor_cpd_cell
        fvfield["uvw_cpd_cell"] = apply_cyclic_face_values(
            fvfield.get("uvw_cpd_cell", None), fvpatch_batch, neighbor_cpd_cell, global_patch_idx)
        fvfield["grad_uvw_cpd_cell"] = apply_cyclic_face_values(
            fvfield.get("grad_uvw_cpd_cell", None), fvpatch_batch, neighbor_cpd_cell, global_patch_idx,
            gradient=True)
        return fvfield


class Fixedvalue(BoundaryCondition):
    """Prescribed velocity on the boundary faces."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        self.value = parametric_manager.resolve_value(boundary_info["value"])
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell: torch.Tensor, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"], num_element=e - s, channel_dim=3)
        uvw_cpd_cell[..., s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        uvw_cpd_cell = fvfield["uvw_cpd_cell"]
        s, e = integrator.fvpatch_batch.start_idx_cpd_cell[g], integrator.fvpatch_batch.end_idx_cpd_cell[g]
        uvw_cpd_cell[..., s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        fvfield["uvw_cpd_cell"] = uvw_cpd_cell
        return fvfield


class Noslip(BoundaryCondition):
    """Zero velocity on the boundary faces."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        uvw_cpd_cell[s:e, 0:3] = 0.
        return uvw_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        uvw_cpd_cell = fvfield["uvw_cpd_cell"]
        s, e = integrator.fvpatch_batch.start_idx_cpd_cell[g], integrator.fvpatch_batch.end_idx_cpd_cell[g]
        uvw_cpd_cell[..., s:e, 0:3] = 0.0
        fvfield["uvw_cpd_cell"] = uvw_cpd_cell
        return fvfield


class _FlowRate(BoundaryCondition):
    """Shared set-up of the volumetric flow-rate conditions."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        self.volumetricFlowRate = parametric_manager.resolve_value(boundary_info.get("volumetricFlowRate", None))
        self.value = parametric_manager.resolve_value(boundary_info["value"])
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"], num_element=e - s, channel_dim=3)
        uvw_cpd_cell[s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell

    def _flow_rate(self, device):
        return Functionbase(
            value_type=self.volumetricFlowRate["value_type"], value=self.volumetricFlowRate["val"],
            num_element=1, channel_dim=1).eval(device=device)


class Flowrateinletvelocity(_FlowRate):
    """Uniform inflow normal to the patch carrying ``volumetricFlowRate``."""

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        fvpatch_batch = integrator.fvpatch_batch
        uvw_cpd_cell = fvfield["uvw_cpd_cell"]
        s_f, e_f = fvpatch_batch.start_idx_face[g], fvpatch_batch.end_idx_face[g]
        s_c, e_c = fvpatch_batch.start_idx_cpd_cell[g], fvpatch_batch.end_idx_cpd_cell[g]

        total_area = torch.sum(integrator.face_area[s_f:e_f].to(uvw_cpd_cell.device))
        velocity_magnitude = self._flow_rate(uvw_cpd_cell.device) / total_area
        # face normals point out of the domain
        normals = integrator.unv_face[s_f:e_f].to(uvw_cpd_cell.device)
        uvw_cpd_cell[..., s_c:e_c, 0:3] = -velocity_magnitude * normals
        fvfield["uvw_cpd_cell"] = uvw_cpd_cell
        return fvfield


class Flowrateoutletvelocity(_FlowRate):
    """Outflow extrapolated from the owner cells, its normal component rescaled to carry
    ``volumetricFlowRate`` (OpenFOAM ``flowRateOutletVelocity``); backflow is clipped."""

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        fvpatch_batch = integrator.fvpatch_batch
        uvw_cpd_cell = fvfield["uvw_cpd_cell"]
        s_f, e_f = fvpatch_batch.start_idx_face[g], fvpatch_batch.end_idx_face[g]
        s_c, e_c = fvpatch_batch.start_idx_cpd_cell[g], fvpatch_batch.end_idx_cpd_cell[g]

        owner_cells = integrator.neighbor_cpd_cell[0, s_f:e_f]
        normals = integrator.unv_face[s_f:e_f].to(uvw_cpd_cell.device)
        areas = integrator.face_area[s_f:e_f].to(uvw_cpd_cell.device)
        uvw_owner = uvw_cpd_cell[..., owner_cells, :]
        nUp = torch.sum(uvw_owner * normals, dim=-1, keepdim=True)
        uvw_tan = uvw_owner - nUp * normals
        nUp = torch.clamp(nUp, min=0.0)

        target_flow = self._flow_rate(uvw_cpd_cell.device).squeeze()
        rho_for_flux = 1.0
        estimated_flow = torch.sum(rho_for_flux * areas * nUp, dim=-2, keepdim=True)
        area_sum = torch.sum(rho_for_flux * areas, dim=-2, keepdim=True)
        use_scale = estimated_flow > (0.5 * target_flow)
        scale = torch.abs(target_flow / (estimated_flow + 1e-12))
        adjustment = (target_flow - estimated_flow) / (area_sum + 1e-12)
        nUp = torch.where(use_scale, nUp * scale, nUp + adjustment)

        uvw_cpd_cell[..., s_c:e_c, 0:3] = uvw_tan + nUp * normals
        fvfield["uvw_cpd_cell"] = uvw_cpd_cell
        return fvfield


class Zerogradient(BoundaryCondition):
    """Ghost velocity equal to the owner cell's; zero diffusive flux."""

    def __init__(self, patch_name, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
        owner_cells = kwargs["neighbor_cpd_cell"][0, s_f:e_f]
        uvw_cpd_cell[s_c:e_c, 0:3] = uvw_cpd_cell[owner_cells, 0:3]
        return uvw_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        neighbor_cpd_cell = integrator.neighbor_cpd_cell
        s_f, e_f = integrator.fvpatch_batch.start_idx_face[g], integrator.fvpatch_batch.end_idx_face[g]
        if "uvw_cpd_cell" in fvfield:
            uvw_cpd_cell = fvfield["uvw_cpd_cell"]
            uvw_cpd_cell[..., neighbor_cpd_cell[1, s_f:e_f], :] = uvw_cpd_cell[..., neighbor_cpd_cell[0, s_f:e_f], :]
            fvfield["uvw_cpd_cell"] = uvw_cpd_cell
        if "grad_uvw_flux_face" in fvfield:
            fvfield["grad_uvw_flux_face"][..., s_f:e_f, :] = 0.0
        return fvfield


class Symmetry(BoundaryCondition):
    """Symmetry plane (OpenFOAM ``symmetry``): the face value is the owner velocity minus its
    normal component, the normal face flux is zero, and the normal gradient is
    ``-(u.n) n / d`` for the new and old velocities."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict, parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
        owner_cells = kwargs["neighbor_cpd_cell"][0, s_f:e_f]
        uvw_cpd_cell[s_c:e_c, 0:3] = uvw_cpd_cell[owner_cells, 0:3]
        return uvw_cpd_cell

    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        g = self.patch_idx + patch_idx_offset
        uvw_cpd_cell = fvfield["uvw_cpd_cell"]
        s_f, e_f = integrator.fvpatch_batch.start_idx_face[g], integrator.fvpatch_batch.end_idx_face[g]
        owner_cells = integrator.neighbor_cpd_cell[0, s_f:e_f]
        boundary_face_cells = integrator.neighbor_cpd_cell[1, s_f:e_f]
        normals = integrator.unv_face[s_f:e_f]

        uvw_owner = uvw_cpd_cell[..., owner_cells, :]
        un_owner = torch.sum(uvw_owner * normals, dim=-1, keepdim=True)
        uvw_cpd_cell[..., boundary_face_cells, :] = uvw_owner - un_owner * normals
        fvfield["uvw_cpd_cell"] = uvw_cpd_cell

        if "uvw_flux_face" in fvfield:
            fvfield["uvw_flux_face"][..., s_f:e_f, :] = 0.0
        if "grad_uvw_flux_face" in fvfield:
            grad_uvw_flux_face = fvfield["grad_uvw_flux_face"]
            dCF = integrator.dCF[s_f:e_f].reshape(e_f - s_f, 1)
            delta_coeffs = 1.0 / (dCF + 1e-12)
            uvw_old_owner = fvfield["uvw_old_cpd_cell"][..., owner_cells, :]
            un_new = torch.sum(uvw_owner * normals, dim=-1, keepdim=True)
            un_old = torch.sum(uvw_old_owner * normals, dim=-1, keepdim=True)
            grad_uvw_flux_face[..., s_f:e_f, 0:3] = -delta_coeffs * un_new * normals
            grad_uvw_flux_face[..., s_f:e_f, 3:6] = -delta_coeffs * un_old * normals
            fvfield["grad_uvw_flux_face"] = grad_uvw_flux_face
        return fvfield
