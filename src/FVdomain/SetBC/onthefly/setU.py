"""Velocity (U) boundary condition descriptors."""

from typing import Dict

import torch

from FVdomain.SetBC.onthefly.BCbase import BCType, BoundaryCondition, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import apply_cyclic_face_values
from FVdomain.SetBC.onthefly.setRack import Rackinletvelocity, Rackoutletvelocity


def _copy_owner_u(uvw_cpd_cell, fvpatch, patch_idx, kwargs):
    s_c, e_c = fvpatch.start_idx_cpd_cell[patch_idx], fvpatch.end_idx_cpd_cell[patch_idx]
    s_f, e_f = fvpatch.start_idx_face[patch_idx], fvpatch.end_idx_face[patch_idx]
    neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
    if neighbor_cpd_cell is not None:
        uvw_cpd_cell[s_c:e_c, 0:3] = uvw_cpd_cell[neighbor_cpd_cell[0, s_f:e_f], 0:3]
    else:
        uvw_cpd_cell[s_c:e_c, 0:3] = 0.0
    return uvw_cpd_cell


class Empty(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.EMPTY)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)


class Cyclic(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.CYCLIC)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
        if neighbor_cpd_cell is None:
            raise ValueError("No neighbor_cpd_cell provided")
        return apply_cyclic_face_values(uvw_cpd_cell, fvpatch, neighbor_cpd_cell, self.patch_idx)


class Fixedvalue(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        raw_val = boundary_info["value"]
        self.value = parametric_manager.resolve_value(raw_val) if parametric_manager is not None else raw_val
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"],
            num_element=e - s, channel_dim=3,
        )
        uvw_cpd_cell[..., s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Uniformfixedvalue(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        if parametric_manager is not None:
            self.uniformValue = parametric_manager.resolve_value(boundary_info["uniformValue"])
            self.value = parametric_manager.resolve_value(boundary_info["value"])
        else:
            self.uniformValue = boundary_info["uniformValue"]
            self.value = boundary_info["value"]
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"],
            num_element=e - s, channel_dim=3,
        )
        uvw_cpd_cell[s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Noslip(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        uvw_cpd_cell[s:e, 0:3] = 0.0
        return uvw_cpd_cell


class Flowrateinletvelocity(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        if parametric_manager is not None:
            self.volumetricFlowRate = parametric_manager.resolve_value(boundary_info.get("volumetricFlowRate"))
            self.massFlowRate = parametric_manager.resolve_value(boundary_info.get("massFlowRate"))
            self.value = parametric_manager.resolve_value(boundary_info["value"])
        else:
            self.volumetricFlowRate = boundary_info.get("volumetricFlowRate")
            self.massFlowRate = boundary_info.get("massFlowRate")
            self.value = boundary_info["value"]
        rho_val = boundary_info.get("rho", 1.0)
        self.rho = boundary_info.get("rhoInlet", 1.0) if isinstance(rho_val, str) else rho_val
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"],
            num_element=e_c - s_c, channel_dim=3,
        )
        face_area = kwargs.get("face_area")
        unv_face = kwargs.get("unv_face")
        if face_area is not None and unv_face is not None and (
            self.volumetricFlowRate is not None or self.massFlowRate is not None
        ):
            s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
            total_area = torch.sum(face_area[s_f:e_f].to(device=uvw_cpd_cell.device, dtype=uvw_cpd_cell.dtype))
            if self.volumetricFlowRate is not None:
                q_val = Functionbase(
                    value_type=self.volumetricFlowRate["value_type"],
                    value=self.volumetricFlowRate["val"], num_element=1, channel_dim=1,
                ).eval(device=uvw_cpd_cell.device).to(uvw_cpd_cell.dtype)
                vmag = q_val / total_area
            else:
                m_val = Functionbase(
                    value_type=self.massFlowRate["value_type"],
                    value=self.massFlowRate["val"], num_element=1, channel_dim=1,
                ).eval(device=uvw_cpd_cell.device).to(uvw_cpd_cell.dtype)
                vmag = m_val / (self.rho * total_area)
            normals = unv_face[s_f:e_f].to(device=uvw_cpd_cell.device, dtype=uvw_cpd_cell.dtype)
            uvw_cpd_cell[..., s_c:e_c, 0:3] = -vmag * normals
            return uvw_cpd_cell
        uvw_cpd_cell[..., s_c:e_c, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Flowrateoutletvelocity(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        if parametric_manager is not None:
            self.volumetricFlowRate = parametric_manager.resolve_value(boundary_info.get("volumetricFlowRate"))
            self.massFlowRate = parametric_manager.resolve_value(boundary_info.get("massFlowRate"))
            self.value = parametric_manager.resolve_value(boundary_info["value"])
        else:
            self.volumetricFlowRate = boundary_info.get("volumetricFlowRate")
            self.massFlowRate = boundary_info.get("massFlowRate")
            self.value = boundary_info["value"]
        rho_val = boundary_info.get("rho", 1.0)
        self.rho = boundary_info.get("rhoOutlet", boundary_info.get("rhoInlet", 1.0)) if isinstance(rho_val, str) else rho_val
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.FIRST)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s_c, e_c = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"],
            num_element=e_c - s_c, channel_dim=3,
        )
        face_area = kwargs.get("face_area")
        unv_face = kwargs.get("unv_face")
        if face_area is not None and unv_face is not None and (
            self.volumetricFlowRate is not None or self.massFlowRate is not None
        ):
            s_f, e_f = fvpatch.start_idx_face[self.patch_idx], fvpatch.end_idx_face[self.patch_idx]
            normals = unv_face[s_f:e_f].to(device=uvw_cpd_cell.device, dtype=uvw_cpd_cell.dtype)
            areas = face_area[s_f:e_f].view(-1, 1).to(device=uvw_cpd_cell.device, dtype=uvw_cpd_cell.dtype)
            neighbor_cpd_cell = kwargs.get("neighbor_cpd_cell")
            uvw_owner = (
                uvw_cpd_cell[..., neighbor_cpd_cell[0, s_f:e_f], 0:3]
                if neighbor_cpd_cell is not None
                else torch.zeros_like(normals)
            )
            nUp = torch.sum(uvw_owner * normals, dim=-1, keepdim=True)
            uvw_tan = uvw_owner - nUp * normals
            nUp = torch.clamp(nUp, min=0.0)
            spec = self.volumetricFlowRate if self.volumetricFlowRate is not None else self.massFlowRate
            rho_for_flux = 1.0 if self.volumetricFlowRate is not None else self.rho
            target_flow = Functionbase(
                value_type=spec["value_type"], value=spec["val"], num_element=1, channel_dim=1,
            ).eval(device=uvw_cpd_cell.device).to(uvw_cpd_cell.dtype).squeeze()
            est = torch.sum(rho_for_flux * areas * nUp, dim=-2, keepdim=True)
            asum = torch.sum(rho_for_flux * areas, dim=-2, keepdim=True)
            nUp = torch.where(
                est > 0.5 * target_flow,
                nUp * torch.abs(target_flow / (est + 1e-12)),
                nUp + (target_flow - est) / (asum + 1e-12),
            )
            uvw_cpd_cell[..., s_c:e_c, 0:3] = uvw_tan + nUp * normals
            return uvw_cpd_cell
        uvw_cpd_cell[..., s_c:e_c, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Zerogradient(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)


class Advectiveoutlet(BoundaryCondition):
    depends_on_old_state = True

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        self.alpha_floor = float(boundary_info.get("alphaFloor", 0.02))
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)


class Slip(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)


class Symmetry(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)


class Inletoutlet(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        if parametric_manager is not None:
            self.inletValue = parametric_manager.resolve_value(boundary_info["inletValue"])
            self.value = parametric_manager.resolve_value(boundary_info["value"])
        else:
            self.inletValue = boundary_info["inletValue"]
            self.value = boundary_info["value"]
        self.value_tensor = None
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s, e = fvpatch.start_idx_cpd_cell[self.patch_idx], fvpatch.end_idx_cpd_cell[self.patch_idx]
        self.value_tensor = Functionbase(
            value_type=self.value["value_type"], value=self.value["val"],
            num_element=e - s, channel_dim=3,
        )
        uvw_cpd_cell[s:e, 0:3] = self.value_tensor.eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Calculated(BoundaryCondition):
    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, BCType.MIXED)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        return _copy_owner_u(uvw_cpd_cell, fvpatch, self.patch_idx, kwargs)
