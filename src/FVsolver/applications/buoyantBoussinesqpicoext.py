"""Boussinesq buoyant-flow residual (buoyantBoussinesqPimpleFoam-like), state
``[u, v, w, p_rgh, T]``.

``ext``: the viscous and thermal fluxes use the interpolated WLSQ face gradient, kept in the
autograd graph, with the compact snGrad on non-cyclic boundary faces.
"""

import torch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVsolver.applications.base_solver import BoussinesqSolver, EPS_SQRT
from FVsolver.FVdiscretization.FVgrad import gradient_reconstruction


class Solver(BoussinesqSolver):

    def conserved_form(self, uvwpt_hat_cpd_cell, uvw_flux_hat_face, grad_uvwpt_hat_cpd_cell,
                       grad_uvwpt_flux_hat_face):
        """Continuity, momentum and energy residuals, and the per-graph log loss.

        ``uvwpt_hat_cpd_cell`` = ``[u_new(3), u_old(3), p_rgh(1), T_new(1), T_old(1)]``;
        ``uvw_flux_hat_face`` holds the new and old face velocity fluxes.
        """
        unsteady_uvw_cell, unsteady_t_cell, source_term_total, heat_source_term_total = self.time_scheme(
            uvw_old_cell=uvwpt_hat_cpd_cell[..., 3:6],
            uvw_new_cell=uvwpt_hat_cpd_cell[..., 0:3],
            t_old_cell=uvwpt_hat_cpd_cell[..., 8:9],
            t_new_cell=uvwpt_hat_cpd_cell[..., 7:8],
        )
        mass_flux_cells_face = self.continuity_flux(uvw_flux_face=uvw_flux_hat_face[..., 0:1])
        convection_flux_uvw_face, bounded_source_uvw = self.convection(
            uvw_flux_hat_face,
            uvwpt_hat_cpd_cell[..., 0:3], grad_uvwpt_hat_cpd_cell[..., 0:3, :],
            uvwpt_hat_cpd_cell[..., 3:6], grad_uvwpt_hat_cpd_cell[..., 3:6, :],
            self._bounded_U,
        )

        snGrad_p_face = self.compute_snGrad(
            uvwpt_hat_cpd_cell[..., 6:7], grad_uvwpt_hat_cpd_cell[..., 6:7, :], var_type=VARType.PRESSURE_RGH)
        if self._momentum_residual_cell:
            grad_p_rgh_cell = grad_uvwpt_hat_cpd_cell[..., 6, :].index_select(-2, self._facecv_interior_ids)
            P_rgh_grad_volume = grad_p_rgh_cell * self.theta_PDE[self.batch_cell, 3:4] * self.cell_volume
        else:
            P_rgh_grad_volume = 0.0

        bc_face_mask = self._bc_diffusion_face_mask()
        snGrad_uvw_face = self.compute_snGrad(
            uvwpt_hat_cpd_cell[..., 0:3], grad_uvwpt_hat_cpd_cell[..., 0:3, :], var_type=VARType.VELOCITY)
        grad_uvw_face = torch.where(bc_face_mask, snGrad_uvw_face, grad_uvwpt_flux_hat_face[..., 0:3])
        vis_flux_uvw_face = self.diffusion_flux(grad_uvw_face)

        convection_flux_t_face, bounded_source_t = self.convection(
            uvw_flux_hat_face,
            uvwpt_hat_cpd_cell[..., 7:8], grad_uvwpt_hat_cpd_cell[..., 7:8, :],
            uvwpt_hat_cpd_cell[..., 8:9], grad_uvwpt_hat_cpd_cell[..., 8:9, :],
            self._bounded_T,
        )
        snGrad_t_face = self.compute_snGrad(
            uvwpt_hat_cpd_cell[..., 7:8], grad_uvwpt_hat_cpd_cell[..., 7:8, :], var_type=VARType.TEMPERATURE)
        grad_t_face = torch.where(bc_face_mask, snGrad_t_face, grad_uvwpt_flux_hat_face[..., 7:8])
        thermal_diff_flux_face = self.alpha_coeff * grad_t_face

        buoyancy_volume_source = self.buoyancy_source(
            grad_uvwpt_hat_cpd_cell[..., 7, :].index_select(-2, self._facecv_interior_ids))

        J_flux_mom_cells_face = (convection_flux_uvw_face - vis_flux_uvw_face).index_select(-2, self.cells_face)
        J_flux_energy_cells_face = (convection_flux_t_face - thermal_diff_flux_face).index_select(-2, self.cells_face)
        total_RHS = scatter(
            (torch.cat((mass_flux_cells_face, J_flux_mom_cells_face, J_flux_energy_cells_face), dim=-1)
             * self.directed_area_cells_face).index_select(-2, self._non_empty_cells_face_indices),
            self.cells_face_ptr.index_select(0, self._non_empty_cells_face_indices),
            dim=-2, dim_size=self.num_cells, reduce="sum",
        )
        continuity_RHS = total_RHS[..., 0:1]
        momentum_RHS = total_RHS[..., 1:4]
        temperature_RHS = total_RHS[..., 4:5]

        loss_cont = torch.sqrt(
            global_add_pool(continuity_RHS ** 2, batch=self.batch_cell, size=self.num_graphs) + EPS_SQRT)

        bounded_source_uvw_interior = (bounded_source_uvw.index_select(-2, self._interior_cell_indices)
                                       if bounded_source_uvw is not None else 0.0)
        residual_momentum_pfree = (unsteady_uvw_cell + momentum_RHS - source_term_total
                                   - buoyancy_volume_source + bounded_source_uvw_interior)
        residual_momentum = residual_momentum_pfree + P_rgh_grad_volume
        loss_momentum = self.momentum_loss(residual_momentum, residual_momentum_pfree, snGrad_p_face)

        bounded_source_t_interior = (bounded_source_t.index_select(-2, self._interior_cell_indices)
                                     if bounded_source_t is not None else 0.0)
        residual_energy = unsteady_t_cell + temperature_RHS - heat_source_term_total + bounded_source_t_interior
        loss_energy = torch.sqrt(
            global_add_pool(residual_energy ** 2, batch=self.batch_cell, size=self.num_graphs) + EPS_SQRT)

        loss_p_rghRef = self.pressure_reference_loss(uvwpt_hat_cpd_cell[..., 6:7])
        self.loss_components = (("cont", loss_cont), ("mom", loss_momentum), ("energy", loss_energy),
                                ("p_rghRef", loss_p_rghRef))

        loss_inner = (
            self.loss_cont_weight * loss_cont.sum(dim=-1, keepdim=True)
            + self.loss_mom_weight * self.balance_momentum(loss_momentum).sum(dim=-1, keepdim=True)
            + self.loss_energy_weight * loss_energy.sum(dim=-1, keepdim=True)
            + self.loss_p_Ref_weight * loss_p_rghRef.sum(dim=-1, keepdim=True)
        )
        return torch.log(loss_inner).squeeze(-1)

    def forward(self, phi_old_cpd_cell, phi_new_cpd_cell):
        uvw_old_cpd_cell = phi_old_cpd_cell[..., 0:3]
        t_old_cpd_cell = phi_old_cpd_cell[..., 4:5]
        uvw_new_cpd_cell = phi_new_cpd_cell[..., 0:3]
        p_rgh_new_cpd_cell = phi_new_cpd_cell[..., 3:4]
        t_new_cpd_cell = phi_new_cpd_cell[..., 4:5]
        if phi_new_cpd_cell.requires_grad:
            # boundary conditions write in place into the ghost rows
            uvw_new_cpd_cell = uvw_new_cpd_cell.clone()
            p_rgh_new_cpd_cell = p_rgh_new_cpd_cell.clone()
            t_new_cpd_cell = t_new_cpd_cell.clone()

        fvfield = {
            "uvw_cpd_cell": uvw_new_cpd_cell,
            "p_rgh_cpd_cell": p_rgh_new_cpd_cell,
            "t_cpd_cell": t_new_cpd_cell,
            "uvw_old_cpd_cell": uvw_old_cpd_cell,
        }
        fvfield = self.enforce_bc_1st(fvfield)
        fvfield = self.enforce_cyclic(fvfield)
        fvfield = self.enforce_mixed(fvfield)
        uvw_new_cpd_cell = fvfield["uvw_cpd_cell"]
        p_rgh_new_cpd_cell = fvfield["p_rgh_cpd_cell"]
        t_new_cpd_cell = fvfield["t_cpd_cell"]

        uvwpt_hat_cpd_cell = torch.cat(
            (uvw_new_cpd_cell, uvw_old_cpd_cell, p_rgh_new_cpd_cell, t_new_cpd_cell, t_old_cpd_cell), dim=-1)
        grad_uvwpt_hat_cpd_cell = gradient_reconstruction(
            phi_node=uvwpt_hat_cpd_cell,
            edge_index=self.neighbor_cpd_cell_x,
            mask_valid_node=self.mask_interior_cell,
            precompute_Moments=[self.A_cell_to_cell, self.single_B_cell_to_cell],
        )
        fvfield["grad_uvw_cpd_cell"] = grad_uvwpt_hat_cpd_cell[..., 0:6, :]
        fvfield["grad_p_rgh_cpd_cell"] = grad_uvwpt_hat_cpd_cell[..., 6:7, :]
        fvfield["grad_t_cpd_cell"] = grad_uvwpt_hat_cpd_cell[..., 7:9, :]
        fvfield = self.enforce_cyclic(fvfield)
        grad_uvwpt_hat_cpd_cell = torch.cat(
            (fvfield["grad_uvw_cpd_cell"], fvfield["grad_p_rgh_cpd_cell"], fvfield["grad_t_cpd_cell"]), dim=-2)

        # Both interpolations read one slice of the cell gradient, so autograd sums their
        # contributions before the other uses of the gradient: this fixes the floating-point
        # summation order of the backward pass to that of the published runs.
        wlsq_grad = grad_uvwpt_hat_cpd_cell[..., 0:9, :]
        uvwpt_hat_face = self.interpolating_phic_to_faces(uvwpt_hat_cpd_cell, wlsq_grad)
        grad_uvwpt_hat_face = self.interpolating_gradients_to_faces(uvwpt_hat_cpd_cell, wlsq_grad)
        uvw_hat_flux_face = torch.cat(
            ((uvwpt_hat_face[..., 0:3] * self.unv_face).sum(dim=-1, keepdim=True),
             (uvwpt_hat_face[..., 3:6] * self.unv_face).sum(dim=-1, keepdim=True)), dim=-1)
        grad_uvwpt_flux_hat_face = (grad_uvwpt_hat_face * self.unv_face.unsqueeze(-2)).sum(dim=-1)

        fvfield["uvw_cpd_cell"] = uvw_new_cpd_cell
        fvfield["uvw_old_cpd_cell"] = uvw_old_cpd_cell
        fvfield["p_rgh_cpd_cell"] = p_rgh_new_cpd_cell
        fvfield["t_cpd_cell"] = t_new_cpd_cell
        fvfield["uvw_flux_face"] = uvw_hat_flux_face
        fvfield["grad_uvw_flux_face"] = grad_uvwpt_flux_hat_face[..., 0:6]
        fvfield["grad_p_rgh_flux_face"] = grad_uvwpt_flux_hat_face[..., 6:7]
        fvfield["grad_t_flux_face"] = grad_uvwpt_flux_hat_face[..., 7:9]
        # Dirichlet conditions again, then the Neumann ones on the face fluxes
        fvfield = self.enforce_bc_1st(fvfield)
        fvfield = self.enforce_mixed(fvfield)
        uvw_new_cpd_cell = fvfield["uvw_cpd_cell"]
        p_rgh_new_cpd_cell = fvfield["p_rgh_cpd_cell"]
        t_new_cpd_cell = fvfield["t_cpd_cell"]

        loss_total = self.conserved_form(
            uvwpt_hat_cpd_cell=torch.cat(
                (uvw_new_cpd_cell, uvw_old_cpd_cell, p_rgh_new_cpd_cell, t_new_cpd_cell, t_old_cpd_cell), dim=-1),
            uvw_flux_hat_face=fvfield["uvw_flux_face"],
            grad_uvwpt_hat_cpd_cell=grad_uvwpt_hat_cpd_cell,
            grad_uvwpt_flux_hat_face=torch.cat(
                (fvfield["grad_uvw_flux_face"], fvfield["grad_p_rgh_flux_face"], fvfield["grad_t_flux_face"]),
                dim=-1),
        )
        return loss_total, torch.cat((uvw_new_cpd_cell, p_rgh_new_cpd_cell, t_new_cpd_cell), dim=-1)
