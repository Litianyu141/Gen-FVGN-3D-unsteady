"""Incompressible Navier-Stokes residual (pimpleFoam-like), state ``[u, v, w, p]``.

``ext``: the viscous flux uses the interpolated WLSQ face gradient, kept in the autograd
graph, with the compact snGrad on non-cyclic boundary faces.
"""

import torch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVsolver.applications.base_solver import BaseSolver, EPS_SQRT
from FVsolver.FVdiscretization.FVgrad import gradient_reconstruction


class Solver(BaseSolver):

    def __init__(self, params) -> None:
        super().__init__(params)
        self._bounded_U = self._bounded_scheme("U")

    def conserved_form(self, uvwp_hat_cpd_cell, uvw_flux_hat_face, grad_uvwp_hat_cpd_cell,
                       grad_uvwp_flux_hat_face):
        """Continuity and momentum residuals, and the per-graph log loss.

        ``uvwp_hat_cpd_cell`` = ``[u_new(3), u_old(3), p(1)]``; ``uvw_flux_hat_face`` holds the
        new and old face velocity fluxes.
        """
        idx = self._interior_cell_indices
        unsteady_cell = (uvwp_hat_cpd_cell.index_select(-2, idx)[..., 0:3] / self.dt_cell) * self.cell_volume * self.unsteady_coeff
        source_term = self.source_term.expand(-1, 3) + (
            uvwp_hat_cpd_cell.index_select(-2, idx)[..., 3:6] / self.dt_cell) * self.cell_volume * self.unsteady_coeff

        mass_flux_cells_face = self.continuity_flux(uvw_flux_face=uvw_flux_hat_face[..., 0:1])
        convection_flux_face, bounded_source_uvw = self.convection(
            uvw_flux_hat_face,
            uvwp_hat_cpd_cell[..., 0:3], grad_uvwp_hat_cpd_cell[..., 0:3, :],
            uvwp_hat_cpd_cell[..., 3:6], grad_uvwp_hat_cpd_cell[..., 3:6, :],
            self._bounded_U,
        )

        snGrad_uvw_face = self.compute_snGrad(
            uvwp_hat_cpd_cell[..., 0:3], grad_uvwp_hat_cpd_cell[..., 0:3, :], var_type=VARType.VELOCITY)
        grad_uvw_face = torch.where(self._bc_diffusion_face_mask(), snGrad_uvw_face,
                                    grad_uvwp_flux_hat_face[..., 0:3])
        vis_flux_face = self.diffusion_flux(grad_uvw_face)

        snGrad_p_face = self.compute_snGrad(
            uvwp_hat_cpd_cell[..., 6:7], grad_uvwp_hat_cpd_cell[..., 6:7, :], var_type=VARType.PRESSURE)
        if self._momentum_residual_cell:
            grad_p_cell = grad_uvwp_hat_cpd_cell[..., 6, :].index_select(-2, self._facecv_interior_ids)
            P_grad_volume_cell = grad_p_cell * self.theta_PDE[self.batch_cell, 3:4] * self.cell_volume
        else:
            P_grad_volume_cell = 0.

        J_flux_cells_face = (convection_flux_face - vis_flux_face).index_select(-2, self.cells_face)
        total_RHS = scatter(
            (torch.cat((mass_flux_cells_face, J_flux_cells_face), dim=-1)
             * self.directed_area_cells_face)[..., self.mask_non_empty_cells_face, :],
            self.cells_face_ptr[self.mask_non_empty_cells_face],
            dim=-2, dim_size=self.num_cells, reduce="sum",
        )

        loss_cont = torch.sqrt(
            global_add_pool(total_RHS[..., 0:1] ** 2, batch=self.batch_cell, size=self.num_graphs) + EPS_SQRT
        ).sum(dim=-1, keepdim=True)

        bounded_source_interior = (bounded_source_uvw[..., self.mask_interior_cell, :]
                                   if bounded_source_uvw is not None else 0.0)
        mom_residual_pfree = unsteady_cell + total_RHS[..., 1:4] - source_term + bounded_source_interior
        mom_residual_cell = mom_residual_pfree + P_grad_volume_cell
        loss_momentum = self.momentum_loss(mom_residual_cell, mom_residual_pfree, snGrad_p_face)
        loss_pRef = self.pressure_reference_loss(uvwp_hat_cpd_cell[..., 6:7])
        self.loss_components = (("cont", loss_cont), ("mom", loss_momentum), ("pRef", loss_pRef))

        loss_inner = (
            self.loss_cont_weight * loss_cont.sum(dim=-1, keepdim=True)
            + self.loss_mom_weight * self.balance_momentum(loss_momentum).sum(dim=-1, keepdim=True)
            + self.loss_p_Ref_weight * loss_pRef.sum(dim=-1, keepdim=True)
            + 1e-8
        )
        return torch.log(loss_inner).squeeze(-1)

    def forward(self, phi_old_cpd_cell, phi_new_cpd_cell):
        uvw_old_cpd_cell = phi_old_cpd_cell[..., 0:3]
        # boundary conditions write in place into the ghost rows
        need_clone = phi_new_cpd_cell.requires_grad
        fvfield = {
            "uvw_cpd_cell": phi_new_cpd_cell[..., 0:3].clone() if need_clone else phi_new_cpd_cell[..., 0:3],
            "p_cpd_cell": phi_new_cpd_cell[..., 3:4].clone() if need_clone else phi_new_cpd_cell[..., 3:4],
            "uvw_old_cpd_cell": uvw_old_cpd_cell,
        }
        fvfield = self.enforce_bc_1st(fvfield)
        fvfield = self.enforce_cyclic(fvfield)
        fvfield = self.enforce_mixed(fvfield)
        uvw_new_cpd_cell = fvfield["uvw_cpd_cell"]
        p_new_cpd_cell = fvfield["p_cpd_cell"]

        uvwp_hat_cpd_cell = torch.cat(
            (uvw_new_cpd_cell[..., 0:3], uvw_old_cpd_cell[..., 0:3], p_new_cpd_cell[..., 0:1]), dim=-1)
        grad_uvwp_hat_cpd_cell = gradient_reconstruction(
            phi_node=uvwp_hat_cpd_cell,
            edge_index=self.neighbor_cpd_cell_x,
            mask_valid_node=self.mask_interior_cell,
            precompute_Moments=[self.A_cell_to_cell, self.single_B_cell_to_cell],
        )
        fvfield["grad_uvw_cpd_cell"] = grad_uvwp_hat_cpd_cell[..., 0:6, :]
        fvfield["grad_p_cpd_cell"] = grad_uvwp_hat_cpd_cell[..., 6:7, :]
        fvfield = self.enforce_cyclic(fvfield)
        grad_uvwp_hat_cpd_cell = torch.cat((fvfield["grad_uvw_cpd_cell"], fvfield["grad_p_cpd_cell"]), dim=-2)

        # Each interpolation reads its own slice of the cell values and gradients; the slices
        # fix the floating-point summation order of the backward pass to that of the
        # published runs.
        uvwp_hat_face = self.interpolating_phic_to_faces(uvwp_hat_cpd_cell[..., 0:7],
                                                         grad_uvwp_hat_cpd_cell[..., 0:7, :])
        grad_uvwp_hat_face = self.interpolating_gradients_to_faces(uvwp_hat_cpd_cell[..., 0:7],
                                                                   grad_uvwp_hat_cpd_cell[..., 0:7, :])
        uvw_hat_flux_face = torch.cat(
            (torch.sum(uvwp_hat_face[..., 0:3] * self.unv_face, dim=-1, keepdim=True),
             torch.sum(uvwp_hat_face[..., 3:6] * self.unv_face, dim=-1, keepdim=True)), dim=-1)
        grad_uvwp_flux_hat_face = (grad_uvwp_hat_face[..., 0:6, :] * self.unv_face.unsqueeze(-2)).sum(dim=-1)

        fvfield["uvw_cpd_cell"] = uvw_new_cpd_cell
        fvfield["uvw_old_cpd_cell"] = uvw_old_cpd_cell
        fvfield["p_cpd_cell"] = p_new_cpd_cell
        fvfield["uvw_flux_face"] = uvw_hat_flux_face
        fvfield["grad_uvw_flux_face"] = grad_uvwp_flux_hat_face[..., 0:6]
        fvfield = self.enforce_mixed(fvfield)
        uvw_new_cpd_cell = fvfield["uvw_cpd_cell"]
        p_new_cpd_cell = fvfield["p_cpd_cell"]

        loss_total = self.conserved_form(
            uvwp_hat_cpd_cell=torch.cat((uvw_new_cpd_cell, uvw_old_cpd_cell, p_new_cpd_cell), dim=-1),
            uvw_flux_hat_face=fvfield["uvw_flux_face"],
            grad_uvwp_hat_cpd_cell=grad_uvwp_hat_cpd_cell,
            grad_uvwp_flux_hat_face=fvfield["grad_uvw_flux_face"],
        )
        return loss_total, torch.cat((uvw_new_cpd_cell, p_new_cpd_cell), dim=-1)
