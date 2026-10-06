"""Face reconstruction and surface-normal-gradient kernels (OpenFOAM-style)."""

import torch


def batched_matvec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Contract ``mat[..., M, 3]`` with ``vec[..., 3, 1]`` -> ``[..., M, 1]``."""
    return (mat * vec.transpose(-1, -2)).sum(-1, keepdim=True)


def linear_upwind_face_value(
    phi_cpd_cell,
    grad_phi_cpd_cell,
    *,
    face_flux,
    face_owner,
    face_neighbor,
    owner_to_face,
    neighbor_to_face,
    boundary_cpd_indices,
    boundary_face_mask,
) -> torch.Tensor:
    """``linearUpwind``: the upwind cell value extrapolated to the face with the upwind
    cell gradient; non-cyclic boundary faces take their ghost-cell value.
    """
    phi_C = phi_cpd_cell.index_select(-2, face_owner)
    phi_F = phi_cpd_cell.index_select(-2, face_neighbor)
    grad_C = grad_phi_cpd_cell.index_select(-3, face_owner)
    grad_F = grad_phi_cpd_cell.index_select(-3, face_neighbor)

    flux_positive = face_flux > 0
    phi_upwind = torch.where(flux_positive, phi_C, phi_F)
    grad_upwind = torch.where(flux_positive.unsqueeze(-1), grad_C, grad_F)
    r_upwind = torch.where(flux_positive.unsqueeze(-1), owner_to_face, neighbor_to_face)
    phi_face = phi_upwind + batched_matvec(grad_upwind, r_upwind).squeeze(-1)

    mask = boundary_face_mask.reshape((1,) * (phi_face.ndim - 2) + (-1, 1)).expand_as(phi_face)
    boundary_values = phi_cpd_cell.index_select(-2, boundary_cpd_indices)
    boundary_values_full = torch.zeros_like(phi_face).masked_scatter(mask, boundary_values.reshape(-1))
    return torch.where(mask, boundary_values_full, phi_face)


def linear_corrected_face_value(
    phi_cpd_cell,
    grad_phi_cpd_cell,
    *,
    cell_pos,
    face_pos,
    face_owner,
    face_neighbor,
    owner_weight,
    neighbor_weight,
) -> torch.Tensor:
    """``Linear corrected``: distance-weighted linear interpolation plus a gradient correction
    for the offset between the face centre and the weighted cell-centre midpoint.
    """
    phi_cpd_cell_expanded = phi_cpd_cell[:, :, None]
    phi_face = (phi_cpd_cell_expanded[face_owner] * owner_weight
                + phi_cpd_cell_expanded[face_neighbor] * neighbor_weight).squeeze(2)
    grad_linear_face = (owner_weight * grad_phi_cpd_cell[face_owner]
                        + neighbor_weight * grad_phi_cpd_cell[face_neighbor])
    r_hat_face = (cell_pos[face_owner, :, None] * owner_weight
                  + cell_pos[face_neighbor, :, None] * neighbor_weight).squeeze(2)
    correction = batched_matvec(grad_linear_face, (face_pos - r_hat_face)[:, :, None]).squeeze(2)
    return phi_face + correction


def corrected_sn_grad(
    phi_cpd_cell,
    *,
    face_owner,
    face_neighbor,
    non_orth_delta_coeffs,
    grad_cell=None,
    owner_weight=None,
    neighbor_weight=None,
    correction_vectors=None,
) -> torch.Tensor:
    """``corrected`` snGrad: ``(phi_F - phi_C) * nonOrthDeltaCoeffs`` plus the
    non-orthogonal correction from the linearly interpolated cell gradient (omitted when
    ``grad_cell`` is None, i.e. ``uncorrected``).
    """
    snGrad_val = (phi_cpd_cell[face_neighbor] - phi_cpd_cell[face_owner]) * non_orth_delta_coeffs
    if grad_cell is not None:
        grad_face = grad_cell[face_owner] * owner_weight + grad_cell[face_neighbor] * neighbor_weight
        snGrad_val = snGrad_val + (grad_face * correction_vectors[:, None, :]).sum(dim=-1)
    return snGrad_val
