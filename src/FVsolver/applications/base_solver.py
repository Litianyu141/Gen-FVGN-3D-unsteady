"""Machinery shared by the finite-volume residual solvers."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple

import torch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from FVsolver.FVdiscretization.FVflux import FV_flux

# Weight of the NEW time level in the convection term for each ``ddtSchemes`` entry:
# "imex" averages the two levels (Crank-Nicolson), "implicit" is backward Euler.
# Diffusion, pressure and sources are implicit in both.
CONVECTION_TIME_WEIGHT = {"imex": 0.5, "implicit": 1.0}
MOMENTUM_RESIDUALS = ("cell", "face_cv", "cell+face_cv")

EPS_SQRT = 1e-16


class BaseSolver(FV_flux, ABC):
    """Boundary-condition application, scheme selection, momentum rows on face control
    volumes and the pressure-reference loss of the discretisations.

    Fields are carried on compound cells ``[N_cpd, C]`` (interior cells, then one ghost
    cell per boundary face); residuals live on the interior cells.
    """

    def __init__(self, params: Any) -> None:
        super().__init__()
        self.params = params
        fvsolution = params.fvconfig.get("fvSolution", {})
        self.loss_cont_weight = fvsolution.get("loss_cont", 1)
        self.loss_mom_weight = fvsolution.get("loss_mom", 1)
        self.loss_energy_weight = fvsolution.get("loss_energy", 1)
        self.loss_p_Ref_weight = fvsolution.get("loss_pRef", 0.01)
        self.loss_mom_adaptive_balance = bool(fvsolution.get("loss_mom_adaptive_balance", False))
        self.loss_components = ()

        fvschemes = params.fvschemes
        ddt = str(fvschemes["ddtSchemes"]).strip().lower()
        if ddt not in CONVECTION_TIME_WEIGHT:
            raise ValueError(f"[fvSchemes] ddtSchemes = {ddt!r}; expected one of {sorted(CONVECTION_TIME_WEIGHT)}")
        self._conv_theta = CONVECTION_TIME_WEIGHT[ddt]

        mom_res = str(fvschemes.get("momentumResidual", "cell")).lower()
        if mom_res not in MOMENTUM_RESIDUALS:
            raise ValueError(f"[fvSchemes] momentumResidual = {mom_res!r}; expected one of {MOMENTUM_RESIDUALS}")
        self._momentum_residual_cell = mom_res in ("cell", "cell+face_cv")
        self._momentum_residual_face = mom_res in ("face_cv", "cell+face_cv")

    def _bounded_scheme(self, field: str) -> bool:
        """Whether ``divSchemes.<field>`` is the bounded linear-upwind scheme."""
        words = str(self.params.fvschemes.get("divSchemes", {}).get(field, "LinearUpwind")).split()
        bounded = words[0].lower() == "bounded"
        if words[bounded:] != ["LinearUpwind"]:
            raise ValueError(f"[fvSchemes] divSchemes.{field} = {' '.join(words)!r}; "
                             'expected "LinearUpwind" or "bounded LinearUpwind"')
        return bounded

    # ---------------------------------------------------------------- registration
    def register_properties(self, fv_graph: Any, force_register: bool = False) -> None:
        """Cache the geometry and the per-graph PDE coefficients of a batched graph."""
        self.register_geometrics(fv_graph, force_register=force_register)
        self._build_facecv_geom()

        graph_Index = fv_graph.graph_Index
        theta = graph_Index.theta_PDE
        self.theta_PDE = theta
        self.unsteady_coeff = theta[self.batch_cell, 0:1]
        self.continuity_eq_coeff = theta[:, 1:2]
        self.convection_coeff = theta[self.batch_face, 2:3]
        self.grad_p_coeff = theta[self.batch_face, 3:4]
        self.nu_coeff = theta[self.batch_face, 4:5]
        self.source_term = (theta[self.batch_cell, 5:6] * self.cell_volume).detach()
        self.dt_cell = theta[self.batch_cell, 6:7]

        self.pRefCell = graph_Index.pRefCell[graph_Index.pRefCell_valid_mask]
        self.pRefValue = graph_Index.pRefValue[graph_Index.pRefCell_valid_mask]

    def _build_facecv_geom(self):
        """Static geometry of the momentum rows on face control volumes."""
        if getattr(self, "_facecv_geom_sig", None) == self._geom_mesh_sig:
            return
        C = self.C_senders.long().view(-1)
        F = self.F_receivers.long().view(-1)
        mask_int = self.mask_interior_cell.view(-1)
        face_interior = (mask_int[C] & mask_int[F]).view(-1, 1)
        interior_ids = mask_int.nonzero(as_tuple=True)[0]
        vol_cpd = torch.zeros(self.num_cpd_cells, 1, device=self.cell_volume.device, dtype=self.cell_volume.dtype)
        vol_cpd = self._fill_cyclic_ghosts(vol_cpd.index_copy(0, interior_ids, self.cell_volume))
        V_C, V_F = vol_cpd[C], vol_cpd[F]
        # a cyclic face is two-sided once its ghost carries the partner's values
        face_cyclic = (self.mask_boundary_face & ~self.mask_boundary_non_cyclic_face).view(-1, 1)
        self._facecv_C = C
        self._facecv_F = F
        self._facecv_face_twosided = face_interior | face_cyclic
        self._facecv_active_face = ~self.mask_empty_face.view(-1, 1).bool()
        self._facecv_interior_ids = interior_ids
        self._facecv_Vface = torch.where(self._facecv_face_twosided, 0.5 * (V_C + V_F), V_C)
        self._facecv_geom_sig = self._geom_mesh_sig

    def _cyclic_ghost_partner_ids(self):
        """``(ghost, partner)`` compound-cell indices of every cyclic face."""
        sig = self._geom_mesh_sig
        if getattr(self, "_cyclic_ghost_partner_sig", None) == sig:
            return self._cyclic_ghost_partner
        from FVdomain.SetBC.onthefly.setCyclic import get_cyclic_patch_pair_indices

        fvpatch = self.fvpatch_batch
        pairing = getattr(fvpatch, "neighbor_patch_idx", None)
        ghosts, partners = [], []
        if pairing is not None:
            for patch in range(len(pairing)):
                if int(pairing[patch]) < 0:   # -1 marks a non-cyclic patch
                    continue
                _, source_boundary, target_owner, _ = get_cyclic_patch_pair_indices(
                    fvpatch, self.neighbor_cpd_cell, patch)
                ghosts.append(source_boundary.reshape(-1))
                partners.append(target_owner.reshape(-1))
        if ghosts:
            pair = (torch.cat(ghosts).long(), torch.cat(partners).long())
        else:
            empty = torch.zeros(0, dtype=torch.long, device=self.cell_volume.device)
            pair = (empty, empty)
        self._cyclic_ghost_partner = pair
        self._cyclic_ghost_partner_sig = sig
        return pair

    def _fill_cyclic_ghosts(self, x_cpd):
        """Copy every cyclic ghost row (axis -2) from its periodic partner."""
        ghost, partner = self._cyclic_ghost_partner_ids()
        if ghost.numel() == 0:
            return x_cpd
        return x_cpd.index_copy(-2, ghost, x_cpd.index_select(-2, partner))

    # ---------------------------------------------------------- boundary conditions
    def _apply_bcs(self, groups, offsets, fvfield):
        for case_i, conditions in enumerate(groups):
            for bc in conditions:
                fvfield = bc.eval(fvfield, self, patch_idx_offset=offsets[case_i])
        return fvfield

    def enforce_bc_1st(self, fvfield: dict) -> dict:
        """Dirichlet conditions."""
        return self._apply_bcs(self.bcpatch_batch.first, self.bcpatch_batch.first_patch_idx_offset, fvfield)

    def enforce_cyclic(self, fvfield: dict) -> dict:
        """Periodic conditions."""
        return self._apply_bcs(self.bcpatch_batch.cyclic, self.bcpatch_batch.cyclic_patch_idx_offset, fvfield)

    def enforce_mixed(self, fvfield: dict) -> dict:
        """Neumann-type conditions (zeroGradient, symmetry, fixedFluxPressure)."""
        return self._apply_bcs(self.bcpatch_batch.mixed, self.bcpatch_batch.mixed_patch_idx_offset, fvfield)

    def _bc_diffusion_face_mask(self) -> torch.Tensor:
        """``[N_face, 1]``: non-cyclic boundary faces, where the compact snGrad replaces the
        interpolated WLSQ face gradient in the diffusion flux."""
        return (self.mask_boundary_non_cyclic_face & ~self.mask_empty_face).view(-1, 1)

    # ------------------------------------------------------------------ residuals
    def convection(self, flux_face, phi_new, grad_new, phi_old, grad_old, bounded):
        """Convective flux of a field with the ``ddtSchemes`` time weighting; ``flux_face``
        carries the new and old face fluxes in columns 0 and 1."""
        conv, source = self.convective_flux(flux_face[..., 0:1], phi_new, grad_new, bounded)
        if self._conv_theta != 1.0:
            th = self._conv_theta
            conv_old, source_old = self.convective_flux(flux_face[..., 1:2], phi_old, grad_old, bounded)
            conv = th * conv + (1.0 - th) * conv_old
            if source is not None:
                source = th * source + (1.0 - th) * source_old
        return conv, source

    def _face_momentum_residual(self, residual_pfree, snGrad_p_face):
        """Face-normal momentum residual on the staggered face control volumes: the
        pressure-free cell residual averaged to the face plus the compact pressure gradient."""
        rhat_int = residual_pfree / self.cell_volume
        rhat_cpd = rhat_int.new_zeros((*rhat_int.shape[:-2], self.num_cpd_cells, 3)).index_copy(
            -2, self._facecv_interior_ids, rhat_int)
        rhat_cpd = self._fill_cyclic_ghosts(rhat_cpd)
        r_C = rhat_cpd.index_select(-2, self._facecv_C)
        r_F = rhat_cpd.index_select(-2, self._facecv_F)
        rhat_face = torch.where(self._facecv_face_twosided, 0.5 * (r_C + r_F), r_C)
        residual = ((self.unv_face * rhat_face).sum(dim=-1, keepdim=True)
                    + self.grad_p_coeff * snGrad_p_face) * self._facecv_Vface
        return torch.where(self._facecv_active_face, residual, torch.zeros_like(residual))

    def momentum_loss(self, residual_cell, residual_pfree, snGrad_p_face):
        """Per-graph L2 norms of the momentum rows selected by ``momentumResidual``."""
        parts = []
        if self._momentum_residual_cell:
            parts.append(torch.sqrt(
                global_add_pool(residual_cell ** 2, batch=self.batch_cell, size=self.num_graphs) + EPS_SQRT))
        if self._momentum_residual_face:
            residual_face = self._face_momentum_residual(residual_pfree, snGrad_p_face)
            parts.append(torch.sqrt(
                global_add_pool(residual_face ** 2, batch=self.batch_face, size=self.num_graphs) + EPS_SQRT))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def balance_momentum(self, loss_momentum):
        """With ``loss_mom_adaptive_balance``, scale every momentum component to the largest."""
        if not self.loss_mom_adaptive_balance:
            return loss_momentum
        detached = loss_momentum.detach()
        return loss_momentum * (torch.amax(detached, dim=-1, keepdim=True) / (detached + 1e-12))

    def pressure_reference_loss(self, p_cpd_cell: torch.Tensor) -> torch.Tensor:
        """``|<p>_V - pRefValue|`` per graph, for graphs without a Dirichlet pressure condition."""
        device = p_cpd_cell.device
        if self.pRefCell.numel() == 0:
            return p_cpd_cell.new_zeros((*p_cpd_cell.shape[:-2], self.num_graphs, 1))
        p_interior = p_cpd_cell.index_select(-2, self._interior_cell_indices)[..., 0:1]
        sum_p_vol = scatter(p_interior * self.cell_volume, self.batch_cell,
                            dim=-2, dim_size=self.num_graphs, reduce="sum")
        sum_vol = scatter(self.cell_volume, self.batch_cell, dim=-2, dim_size=self.num_graphs, reduce="sum")
        mean_p = sum_p_vol / (sum_vol + EPS_SQRT)
        return self._reference_error(mean_p, p_cpd_cell, device)

    def _reference_error(self, mean_p, p_cpd_cell, device):
        batch_pRef = self.batch_cpd_cell[self.pRefCell]
        pRefValue_expanded = p_cpd_cell.new_zeros((self.num_graphs, 1))
        pRefValue_expanded[batch_pRef] = self.pRefValue.view(-1, 1).to(device)
        valid_graph_mask = torch.zeros(self.num_graphs, dtype=torch.bool, device=device)
        valid_graph_mask[batch_pRef] = True
        return torch.sqrt((mean_p - pRefValue_expanded) ** 2 + EPS_SQRT) * valid_graph_mask.view(-1, 1).float()

    def iter_named_residuals(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """``(name, per-graph loss)`` of every residual of the last forward pass."""
        yield from self.loss_components

    @abstractmethod
    def forward(self, phi_old_cpd_cell: torch.Tensor, phi_new_cpd_cell: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the per-graph log residual loss and the boundary-enforced new state."""

    def __call__(self, phi_old_cpd_cell, phi_new_cpd_cell):
        return self.forward(phi_old_cpd_cell, phi_new_cpd_cell)


class BoussinesqSolver(BaseSolver):
    """Incompressible Navier-Stokes with Boussinesq buoyancy and heat transport; the state is
    ``[u, v, w, p_rgh, T]`` with ``p_rgh = p - rho_k g.h``."""

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self._bounded_U = self._bounded_scheme("U")
        self._bounded_T = self._bounded_scheme("T")

    def register_properties(self, fv_graph: Any, force_register: bool = False) -> None:
        super().register_properties(fv_graph, force_register=force_register)
        theta = fv_graph.graph_Index.theta_PDE
        self.alpha_coeff = theta[self.batch_face, 7:8]
        self.heat_source_term = (theta[self.batch_cell, 8:9] * self.cell_volume).detach()
        self.TRef = theta[:, 9:10]
        self.beta = theta[:, 10:11]
        self.gravity = theta[:, 11:14]
        self.hRef = theta[:, 14:15]

    def time_scheme(self, uvw_old_cell, uvw_new_cell, t_old_cell, t_new_cell):
        """Euler time derivatives of U and T; the old level enters as a source."""
        idx = self._interior_cell_indices
        V, c = self.cell_volume, self.unsteady_coeff
        unsteady_uvw_cell = (uvw_new_cell.index_select(-2, idx)[..., 0:3] / self.dt_cell) * V * c
        source_term_total = self.source_term.expand(-1, 3) + (
            uvw_old_cell.index_select(-2, idx)[..., 0:3] / self.dt_cell) * V * c
        unsteady_t_cell = (t_new_cell.index_select(-2, idx)[..., 0:1] / self.dt_cell) * V * c
        heat_source_term_total = self.heat_source_term + (
            t_old_cell.index_select(-2, idx)[..., 0:1] / self.dt_cell) * V * c
        return unsteady_uvw_cell, unsteady_t_cell, source_term_total, heat_source_term_total

    def buoyancy_source(self, grad_T_cell: torch.Tensor) -> torch.Tensor:
        """``gh beta grad(T) V``: the buoyancy force of the p_rgh formulation."""
        beta_cell = self.beta[self.batch_cell]
        g_cell = self.gravity[self.batch_cell]
        ghRef_cell = torch.norm(g_cell, dim=1, keepdim=True) * self.hRef[self.batch_cell]
        cell_pos_interior = self.cpd_cell_pos.index_select(-2, self._facecv_interior_ids)
        gh_cell = (cell_pos_interior * g_cell).sum(dim=1, keepdim=True) - ghRef_cell
        return gh_cell * beta_cell * grad_T_cell * self.cell_volume
