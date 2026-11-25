"""SIMPLE-equivalent local pseudo-time step for steady-state ODIL."""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["local_pseudo_time_scale"]


@torch.no_grad()
def local_pseudo_time_scale(
    integrator: Any,
    phi_cpd_cell: torch.Tensor,
    *,
    relax_momentum: float,
    relax_energy: float,
) -> torch.Tensor:
    """Return ``dt_solver / dtau_P`` per interior cell for ``[u, v, w, T]``, shape ``[N, 4]``."""
    for name, relax in (("momentum", relax_momentum), ("energy", relax_energy)):
        if not 0.0 < relax < 1.0:
            raise ValueError(f"pseudo_time relax_{name} must lie in (0, 1), got {relax}")
    solver = integrator._solver
    owner, neighbour = solver.neighbor_cpd_cell
    area = solver.face_area[:, 0]
    distance = torch.linalg.vector_norm(
        solver.cpd_cell_pos[neighbour] - solver.cpd_cell_pos[owner], dim=-1
    )
    velocity = phi_cpd_cell[:, 0:3]
    face_flux = (
        0.5 * (velocity[owner] + velocity[neighbour]) * solver.unv_face
    ).sum(dim=-1) * area
    outflow_owner = face_flux.clamp_min(0.0)
    outflow_neighbour = (-face_flux).clamp_min(0.0)

    num_cpd = phi_cpd_cell.shape[0]

    def diagonal(gamma: torch.Tensor) -> torch.Tensor:
        coefficient = gamma * area / distance
        a_p = torch.zeros(num_cpd, device=phi_cpd_cell.device, dtype=phi_cpd_cell.dtype)
        a_p.index_add_(0, owner, coefficient + outflow_owner)
        a_p.index_add_(0, neighbour, coefficient + outflow_neighbour)
        return a_p[solver._interior_cell_indices]

    volume = solver.cell_volume[:, 0]
    dt_solver = solver.dt_cell[:, 0]
    scale_u = dt_solver * diagonal(solver.nu_coeff[:, 0]) * (
        (1.0 - relax_momentum) / relax_momentum
    ) / volume
    scale_t = dt_solver * diagonal(solver.alpha_coeff[:, 0]) * (
        (1.0 - relax_energy) / relax_energy
    ) / volume
    return torch.stack((scale_u, scale_u, scale_u, scale_t), dim=-1)
