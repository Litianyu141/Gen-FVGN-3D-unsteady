"""Cross-patch rack black-box boundary condition descriptors (IBBM / CBBM)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from FVdomain.SetBC.onthefly.BCbase import BCType, BoundaryCondition, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import _as_int


def rack_through_direction(
    source_face_pos: torch.Tensor,
    target_face_pos: torch.Tensor,
    source_face_normal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if source_face_normal is not None:
        mean_normal = source_face_normal.mean(dim=0)
        norm = torch.linalg.norm(mean_normal)
        if norm > 1e-12:
            return mean_normal / norm
    offset = target_face_pos.mean(dim=0) - source_face_pos.mean(dim=0)
    norm = torch.linalg.norm(offset)
    if norm <= 1e-12:
        raise ValueError("Rack patches share a centroid and have no usable mean normal.")
    return offset / norm


def rack_face_permutation(
    source_face_pos: torch.Tensor,
    target_face_pos: torch.Tensor,
    *,
    source_face_normal: Optional[torch.Tensor] = None,
    tol_rel: float = 1e-3,
) -> torch.Tensor:
    """Return bijection ``perm`` matching ``target[perm[i]]`` opposite ``source[i]``."""
    if source_face_pos.shape[0] != target_face_pos.shape[0]:
        raise ValueError(
            f"Rack patch face count mismatch: {source_face_pos.shape[0]} vs {target_face_pos.shape[0]}"
        )
    axis = rack_through_direction(
        source_face_pos.double(),
        target_face_pos.double(),
        None if source_face_normal is None else source_face_normal.double(),
    )

    def _transverse(pts: torch.Tensor) -> torch.Tensor:
        return pts - (pts @ axis).unsqueeze(-1) * axis

    source_t = _transverse(source_face_pos.double())
    target_t = _transverse(target_face_pos.double())
    residual, perm = torch.cdist(source_t, target_t).min(dim=1)

    extent = source_t.max(dim=0).values - source_t.min(dim=0).values
    scale = float(torch.linalg.norm(extent)) or 1.0
    worst = float(residual.max())
    if worst > tol_rel * scale:
        raise ValueError(
            f"Rack face pairing residual {worst:.3e} exceeds {tol_rel:.1e} x extent {scale:.3e}"
        )
    if torch.unique(perm).numel() != perm.numel():
        raise ValueError("Rack face pairing is not a bijection.")
    return perm.to(device=source_face_pos.device, dtype=torch.long)


def get_rack_patch_pair_indices(
    fvpatch_like,
    neighbor_cpd_cell: torch.Tensor,
    face_pos: torch.Tensor,
    source_patch_idx: int,
    target_patch_idx: int,
    *,
    face_normal: Optional[torch.Tensor] = None,
    tol_rel: float = 1e-3,
):
    s_idx = _as_int(source_patch_idx)
    t_idx = _as_int(target_patch_idx)
    s_start, s_end = _as_int(fvpatch_like.start_idx_face[s_idx]), _as_int(fvpatch_like.end_idx_face[s_idx])
    t_start, t_end = _as_int(fvpatch_like.start_idx_face[t_idx]), _as_int(fvpatch_like.end_idx_face[t_idx])
    pairs = neighbor_cpd_cell.long()
    source_owner, source_ghost = pairs[:, s_start:s_end]
    target_owner, target_ghost = pairs[:, t_start:t_end]
    perm = rack_face_permutation(
        face_pos[s_start:s_end],
        face_pos[t_start:t_end],
        source_face_normal=None if face_normal is None else face_normal[s_start:s_end],
        tol_rel=tol_rel,
    )
    return (
        source_owner,
        source_ghost,
        target_owner.index_select(0, perm),
        target_ghost.index_select(0, perm),
        perm,
    )


def _resolve(boundary_info: Dict, key: str, parametric_manager, default=None):
    val = boundary_info.get(key, default)
    return parametric_manager.resolve_value(val) if parametric_manager is not None else val


def _scalar(spec, default=None):
    if spec is None:
        return default
    if isinstance(spec, dict):
        spec = spec.get("val", default)
    if spec is None:
        return default
    if torch.is_tensor(spec):
        return float(spec.reshape(-1)[0])
    return float(spec)


def _theta(boundary_info: Dict, key: str, default: float) -> float:
    return _scalar(boundary_info.get("fvconfig", {}).get("theta_pde", {}).get(key), default)


def _partner_patch_idx(boundary_info: Dict, patch_name: str) -> tuple:
    patch_dict = boundary_info.get("patch_dict", {})
    partner = boundary_info.get("partner_patch", boundary_info.get("inlet_patch"))
    if partner is None:
        raise ValueError(f"Rack BC on '{patch_name}' requires 'partner_patch'")
    if partner not in patch_dict:
        raise ValueError(f"Rack BC on '{patch_name}': partner patch '{partner}' not in patch_dict")
    return str(partner), int(patch_dict[partner]["patch_idx"])


def _volumetric_rate(boundary_info: Dict, parametric_manager, patch_name: str) -> float:
    volumetric = _scalar(_resolve(boundary_info, "volumetricFlowRate", parametric_manager))
    mass = _scalar(_resolve(boundary_info, "massFlowRate", parametric_manager))
    servers = _scalar(_resolve(boundary_info, "nServers", parametric_manager), 1.0)
    rho = _scalar(_resolve(boundary_info, "rho", parametric_manager), _theta(boundary_info, "rho", 1.1614))
    if volumetric is not None:
        return servers * volumetric
    if mass is not None:
        return mass / rho
    raise ValueError(f"Rack BC on '{patch_name}' requires volumetricFlowRate or massFlowRate")


class _RackVelocityBase(BoundaryCondition):
    def __init__(
        self,
        patch_name: str,
        patch_idx: int,
        boundary_info: Dict,
        parametric_manager: ParametricManager = None,
        *,
        bc_type: BCType,
    ):
        self.flow_rate = _volumetric_rate(boundary_info, parametric_manager, patch_name)
        self.value = _resolve(
            boundary_info, "value", parametric_manager,
            {"value_type": "uniform", "val": [0.0, 0.0, 0.0]},
        )
        super().__init__(patch_name, patch_idx, VARType.VELOCITY, bc_type)

    def init_field(self, uvw_cpd_cell, fvpatch, **kwargs):
        s = _as_int(fvpatch.start_idx_cpd_cell[self.patch_idx])
        e = _as_int(fvpatch.end_idx_cpd_cell[self.patch_idx])
        uvw_cpd_cell[..., s:e, 0:3] = Functionbase(
            value_type=self.value["value_type"],
            value=self.value["val"],
            num_element=e - s,
            channel_dim=3,
        ).eval(device=uvw_cpd_cell.device)
        return uvw_cpd_cell


class Rackinletvelocity(_RackVelocityBase):
    """Rack intake velocity descriptor (enforced in Warp via ``U_RACK_PRESERVE``)."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        super().__init__(patch_name, patch_idx, boundary_info, parametric_manager, bc_type=BCType.MIXED)


class Rackoutletvelocity(_RackVelocityBase):
    """Rack discharge velocity descriptor (enforced in Warp via ``U_FIXED`` or ``U_RACK_PRESERVE``)."""

    def __init__(self, patch_name: str, patch_idx: int, boundary_info: Dict,
                 parametric_manager: ParametricManager = None):
        self.tangential = str(boundary_info.get("tangential", "zero")).lower()
        if self.tangential not in ("zero", "preserve"):
            raise ValueError(f"Rack discharge '{patch_name}': tangential must be 'zero' or 'preserve'")
        super().__init__(patch_name, patch_idx, boundary_info, parametric_manager, bc_type=BCType.FIRST)


class Rackoutlettemperature(BoundaryCondition):
    """Rack discharge temperature descriptor (IBBM pointwise or CBBM mean, enforced in Warp)."""

    def __init__(
        self,
        patch_name: str,
        patch_idx: int,
        boundary_info: Dict,
        parametric_manager: ParametricManager = None,
    ):
        super().__init__(patch_name, patch_idx, VARType.TEMPERATURE, BCType.COUPLED)
        self.model = str(boundary_info.get("model", "ibbm")).lower()
        if self.model not in ("ibbm", "cbbm"):
            raise ValueError(f"Rack discharge '{patch_name}': model must be 'ibbm' or 'cbbm'")
        self.lag_mean = bool(boundary_info.get("lagMean", False))
        if self.lag_mean and self.model != "cbbm":
            raise ValueError(f"Rack discharge '{patch_name}': lagMean applies only to model='cbbm'")
        self._lagged_mean = None
        self.partner_patch_name, self.partner_patch_idx = _partner_patch_idx(boundary_info, patch_name)

        delta = _scalar(_resolve(boundary_info, "deltaT", parametric_manager))
        if delta is None:
            heat = _scalar(_resolve(boundary_info, "Q_W", parametric_manager))
            if heat is None:
                raise ValueError(f"Rack discharge '{patch_name}' requires deltaT or Q_W")
            rho = _scalar(_resolve(boundary_info, "rho", parametric_manager), _theta(boundary_info, "rho", 1.1614))
            cp = _scalar(_resolve(boundary_info, "Cp", parametric_manager), _theta(boundary_info, "Cp", 1005.0))
            delta = heat / (cp * rho * _volumetric_rate(boundary_info, parametric_manager, patch_name))
        self.deltaT = float(delta)
        self.value = _resolve(
            boundary_info, "value", parametric_manager, {"value_type": "uniform", "val": 0.0}
        )
        self.tol_rel = float(boundary_info.get("pairingTolerance", 1e-3))
        self._pair_cache = None

    def init_field(self, t_cpd_cell, fvpatch, **kwargs):
        s = _as_int(fvpatch.start_idx_cpd_cell[self.patch_idx])
        e = _as_int(fvpatch.end_idx_cpd_cell[self.patch_idx])
        t_cpd_cell[..., s:e, 0:1] = Functionbase(
            value_type=self.value["value_type"],
            value=self.value["val"],
            num_element=e - s,
            channel_dim=1,
        ).eval(device=t_cpd_cell.device)
        return t_cpd_cell

    def _pairing(self, integrator, patch_idx_offset: int):
        fvpatch_batch = integrator.fvpatch_batch
        if self._pair_cache is not None and self._pair_cache[0] is fvpatch_batch:
            return self._pair_cache[1]
        pairing = get_rack_patch_pair_indices(
            fvpatch_batch,
            integrator.neighbor_cpd_cell,
            integrator.face_pos,
            self.patch_idx + patch_idx_offset,
            self.partner_patch_idx + patch_idx_offset,
            face_normal=getattr(integrator, "unv_face", None),
            tol_rel=self.tol_rel,
        )
        self._pair_cache = (fvpatch_batch, pairing)
        return pairing
