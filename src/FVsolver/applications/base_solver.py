"""Warp-backed base classes for unsteady incompressible and Boussinesq FVM solvers."""

from typing import Any, Optional, Tuple

import torch
from torch_geometric.utils import scatter

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVdomain.SetBC.onthefly.setCyclic import get_cyclic_patch_pair_indices
from FVsolver.FVdiscretization.ddt_schemes import parse_ddt_scheme
from FVsolver.FVdiscretization.FVInterpolation import FV_Interpolation
from FVsolver.applications.incompressible_warp import (
    warp_solver_enforce_bcs_only,
    warp_solver_forward,
)


class BaseSolver(FV_Interpolation):
    """Base unsteady incompressible FVM solver executing through NVIDIA Warp."""

    LOSS_COMPONENT_NAMES: Tuple[Optional[str], ...] = ("cont", "mom", "phi_jump", "pRef")
    HAS_ENERGY: bool = False
    PRESSURE_VAR_TYPE = VARType.PRESSURE
    _GRAD_TOKENS: Tuple[str, ...] = ("leastSquares",)

    def __init__(self, params: Any) -> None:
        super().__init__()
        self.params = params
        solver_cfg = params.fvconfig.get("solver", {})
        self.dim = int(solver_cfg.get("dim", 3))

        fvsol = params.fvconfig.get("fvSolution", {})
        self.loss_cont_weight = fvsol.get("loss_cont", 1.0)
        self.loss_mom_weight = fvsol.get("loss_mom", 1.0)
        self.loss_phi_jump_weight = fvsol.get("loss_phi_jump", 0.0)
        self.loss_p_Ref_weight = fvsol.get("loss_pRef", 0.01)
        self.loss_mass_balance_weight = fvsol.get("loss_massBalance", 0.0)
        self.loss_mom_adaptive_balance = bool(
            fvsol.get("loss_mom_adaptive_balance", self.HAS_ENERGY)
        )
        self._max_balance_weight = float(fvsol.get("loss_mom_balance_max_weight", 10.0))

        self._parse_schemes()

    @staticmethod
    def _split_grad_token(scheme) -> Tuple[str, Optional[str]]:
        parts = str(scheme).split()
        if len(parts) >= 2 and parts[-1] in BaseSolver._GRAD_TOKENS:
            return " ".join(parts[:-1]), parts[-1]
        return str(scheme), None

    def _parse_schemes(self) -> None:
        fvs = self.params.fvschemes
        div_schemes = fvs.get("divSchemes", {})
        self.U_scheme, _ = self._split_grad_token(div_schemes.get("U", "Linear"))

        sngrad = fvs.get("snGradSchemes")
        if sngrad is not None:
            self._snGrad_scheme, _ = self._split_grad_token(sngrad)
        else:
            lap = str(fvs.get("laplacianSchemes", "")).lower()
            self._snGrad_scheme = "corrected" if "corrected" in lap else "uncorrected"

        bds = str(fvs.get("boundaryDiffusionScheme", "snGrad")).lower()
        if bds != "sngrad":
            raise ValueError(f"Unsupported boundaryDiffusionScheme={bds!r}; only 'snGrad' is supported")
        self._boundary_diffusion_sngrad = True
        self._cyclic_diffusion_sngrad = str(fvs.get("cyclicDiffusionScheme", "wlsq")).lower() == "sngrad"
        self._diffusion_sngrad_face = self._boundary_diffusion_sngrad or self._cyclic_diffusion_sngrad

        mom_res = str(fvs.get("momentumResidual", "cell")).lower()
        if mom_res not in ("cell", "face_cv", "cell+face_cv", "face_cv+cell"):
            raise ValueError(f"Unknown momentumResidual={mom_res!r}")
        self._momentum_residual_face = mom_res in ("face_cv", "cell+face_cv", "face_cv+cell")
        self._momentum_residual_cell = mom_res in ("cell", "cell+face_cv", "face_cv+cell")

        self._facecv_avg = str(fvs.get("faceCvAvg", "half")).lower()
        if self._facecv_avg not in ("half", "upwind"):
            raise ValueError(f"Unknown faceCvAvg={self._facecv_avg!r}")

        self._facecv_boundary_rows = str(fvs.get("faceCvBoundaryRows", "all"))
        if self._facecv_boundary_rows not in ("none", "pressureDirichlet", "all"):
            raise ValueError(f"Unknown faceCvBoundaryRows={self._facecv_boundary_rows!r}")

        self._facecv_buoyancy = str(fvs.get("faceCvBuoyancy", "cell")).lower()
        if self._facecv_buoyancy not in ("cell", "face_sngrad"):
            raise ValueError(f"Unknown faceCvBuoyancy={self._facecv_buoyancy!r}")

        mi = str(fvs.get("momentumInterpolation", "none")).lower()
        if mi not in ("none", "mwi"):
            raise ValueError(f"Unknown momentumInterpolation={mi!r}")
        self._momentum_interpolation_mwi = mi == "mwi"

        mi_coeff = str(fvs.get("momentumInterpolationCoefficient", "implicit")).lower()
        if mi_coeff not in ("implicit", "frozen"):
            raise ValueError(f"Unknown momentumInterpolationCoefficient={mi_coeff!r}")
        self._mwi_frozen_coefficient = mi_coeff == "frozen"

        try:
            ddt = parse_ddt_scheme(fvs["ddtSchemes"])
            self._conv_theta = 1.0 - 0.5 * ddt.off_centering
        except (KeyError, ValueError, TypeError):
            self._conv_theta = 1.0

    def _bc_diffusion_face_mask(self) -> torch.Tensor:
        m = self.mask_boundary_non_cyclic_face
        if getattr(self, "mask_empty_face", None) is not None:
            m = m & (~self.mask_empty_face)
        if self._cyclic_diffusion_sngrad:
            m = m | (self.mask_boundary_face & (~self.mask_boundary_non_cyclic_face))
        return m.view(-1, 1)

    def _cyclic_ghost_partner_ids(self):
        sig = getattr(self, "_geom_mesh_sig", None)
        cached = getattr(self, "_cyclic_ghost_partner", None)
        if cached is not None and getattr(self, "_cyclic_ghost_partner_sig", None) == sig:
            return cached

        fvpatch = self.fvpatch_batch
        pairing = getattr(fvpatch, "neighbor_patch_idx", None)
        if pairing is None:
            pairing = getattr(fvpatch, "neighbour_patch_idx", None)
        ghosts, partners = [], []
        if pairing is not None:
            for patch in range(len(pairing)):
                if int(pairing[patch]) < 0:
                    continue
                _, src_ghost, tgt_owner, _ = get_cyclic_patch_pair_indices(
                    fvpatch, self.neighbor_cpd_cell, patch
                )
                ghosts.append(src_ghost.reshape(-1))
                partners.append(tgt_owner.reshape(-1))
        if ghosts:
            pair = (torch.cat(ghosts).long(), torch.cat(partners).long())
        else:
            empty = torch.zeros(0, dtype=torch.long, device=self.cell_volume.device)
            pair = (empty, empty)
        self._cyclic_ghost_partner = pair
        self._cyclic_ghost_partner_sig = sig
        return pair

    def _fill_cyclic_ghosts(self, x_cpd: torch.Tensor) -> torch.Tensor:
        ghost, partner = self._cyclic_ghost_partner_ids()
        if ghost.numel() == 0:
            return x_cpd
        return x_cpd.index_copy(-2, ghost, x_cpd.index_select(-2, partner))

    def _build_facecv_geom(self) -> None:
        sig = getattr(self, "_geom_mesh_sig", None)
        if sig is not None and getattr(self, "_facecv_geom_sig", None) == sig:
            return
        C = self.C_senders.long().view(-1)
        F = self.F_receivers.long().view(-1)
        mask_int = self.mask_interior_cell.view(-1)
        face_interior = (mask_int[C] & mask_int[F]).view(-1, 1)
        interior_ids = mask_int.nonzero(as_tuple=True)[0]
        vol_cpd = torch.zeros(
            self.num_cpd_cells, 1,
            device=self.cell_volume.device, dtype=self.cell_volume.dtype,
        ).index_copy(0, interior_ids, self.cell_volume)
        vol_cpd = self._fill_cyclic_ghosts(vol_cpd)
        V_C, V_F = vol_cpd[C], vol_cpd[F]
        face_cyclic = (self.mask_boundary_face & ~self.mask_boundary_non_cyclic_face).view(-1, 1)
        face_twosided = face_interior | face_cyclic

        self._facecv_C = C
        self._facecv_F = F
        self._facecv_face_interior = face_interior
        self._facecv_face_twosided = face_twosided
        self._facecv_active_face = self.facecv_active_face_mask(
            face_interior, self.PRESSURE_VAR_TYPE, self._facecv_boundary_rows
        )
        self._facecv_interior_ids = interior_ids
        self._facecv_Vface = torch.where(face_twosided, 0.5 * (V_C + V_F), V_C)
        self._facecv_geom_sig = sig

    def register_properties(self, fv_graph: Any, force_register: bool = False) -> None:
        self.register_geometrics(fv_graph, force_register=force_register)
        self._build_facecv_geom()

        graph_Index = fv_graph.graph_Index
        self.theta_PDE = graph_Index.theta_PDE
        self.unsteady_coeff = graph_Index.theta_PDE[self.batch_cell, 0:1]
        self.continuity_eq_coeff = graph_Index.theta_PDE[:, 1:2]
        self.convection_coeff = graph_Index.theta_PDE[self.batch_face, 2:3]
        self.grad_p_coeff = graph_Index.theta_PDE[self.batch_face, 3:4]
        self.nu_coeff = graph_Index.theta_PDE[self.batch_face, 4:5]
        self.source_term_base = (graph_Index.theta_PDE[self.batch_cell, 5:6] * self.cell_volume).detach()
        self.source_term = self.source_term_base
        self.dt_cell = graph_Index.theta_PDE[self.batch_cell, 6:7]

        self.pRefCell_valid_mask = graph_Index.pRefCell_valid_mask
        self.pRefCell = graph_Index.pRefCell[self.pRefCell_valid_mask]
        self.pRefValue = graph_Index.pRefValue[self.pRefCell_valid_mask]

    def loss_setpRef(
        self,
        p_cpd_cell: Optional[torch.Tensor] = None,
        *,
        p_rgh_cpd_cell: Optional[torch.Tensor] = None,
        t_cpd_cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Volume-weighted mean pressure reference constraint per graph."""
        del t_cpd_cell
        p_tensor = p_cpd_cell if p_cpd_cell is not None else p_rgh_cpd_cell
        if p_tensor is None:
            raise ValueError("loss_setpRef requires p_cpd_cell or p_rgh_cpd_cell")
        leading = p_tensor.shape[:-2]
        loss = p_tensor.new_zeros((*leading, self.num_graphs, 1))
        if self.pRefCell.numel() == 0:
            return loss
        p_int = p_tensor[..., self.mask_interior_cell, 0:1]
        vol = self.cell_volume
        sum_pv = scatter(p_int * vol, self.batch_cell, dim=-2, dim_size=self.num_graphs, reduce="sum")
        sum_v = scatter(vol, self.batch_cell, dim=0, dim_size=self.num_graphs, reduce="sum")
        mean_p = sum_pv / (sum_v + 1e-12)
        valid = self.pRefCell_valid_mask
        diff = mean_p[..., valid, 0:1] - self.pRefValue.view(-1, 1)
        loss[..., valid, 0:1] = diff.pow(2)
        return loss

    loss_setp_rghRef = loss_setpRef

    def enforce_bcs_only(
        self,
        phi_old_cpd_cell: torch.Tensor,
        phi_new_cpd_cell: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``phi_new_cpd_cell`` onto the boundary condition manifold in Warp."""
        return warp_solver_enforce_bcs_only(self, phi_old_cpd_cell, phi_new_cpd_cell)

    def forward(
        self,
        phi_old_cpd_cell: torch.Tensor,
        phi_new_cpd_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate FVM residuals and return ``(loss_per_graph, phi_new_bc_enforced)``."""
        return warp_solver_forward(self, phi_old_cpd_cell, phi_new_cpd_cell)

    def __call__(
        self,
        phi_old_cpd_cell: torch.Tensor,
        phi_new_cpd_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(phi_old_cpd_cell, phi_new_cpd_cell)

    def get_separated_losses(self) -> Tuple[Optional[torch.Tensor], ...]:
        return (
            getattr(self, "loss_cont_component", None),
            getattr(self, "loss_momentum_component", None),
            getattr(self, "loss_phi_jump_component", None),
            getattr(self, "loss_pRef_component", None),
        )

    def get_normalization_factors(self):
        if hasattr(self, "norm_factor_cont"):
            return (
                self.norm_factor_cont,
                self.norm_factor_mom,
                getattr(self, "norm_factor_energy", None),
            )
        return None, None, None

    def iter_named_residuals(self):
        components = self.get_separated_losses()
        if not components:
            return
        labels = getattr(self, "LOSS_COMPONENT_NAMES", ())
        if len(labels) != len(components):
            labels = tuple(f"loss_{i}" for i in range(len(components)))
        for name, tensor in zip(labels, components):
            if name is not None and tensor is not None:
                yield name, tensor


class BoussinesqSolver(BaseSolver):
    """Base unsteady Boussinesq incompressible + thermal transport solver (Warp backend)."""

    LOSS_COMPONENT_NAMES: Tuple[Optional[str], ...] = ("cont", "mom", "energy", "phi_jump", "p_rghRef")
    HAS_ENERGY: bool = True
    PRESSURE_VAR_TYPE = VARType.PRESSURE_RGH

    def __init__(self, params: Any) -> None:
        fvsol = params.fvconfig.get("fvSolution", {})
        self.loss_energy_weight = fvsol.get("loss_energy", 1.0)
        super().__init__(params)

    def _parse_schemes(self) -> None:
        super()._parse_schemes()
        div_schemes = self.params.fvschemes.get("divSchemes", {})
        self.T_scheme, _ = self._split_grad_token(div_schemes.get("T", "Linear"))

    def register_properties(self, fv_graph: Any, force_register: bool = False) -> None:
        super().register_properties(fv_graph, force_register=force_register)
        theta = fv_graph.graph_Index.theta_PDE
        self.alpha_coeff = theta[self.batch_face, 7:8]
        self.heat_source_term = (theta[self.batch_cell, 8:9] * self.cell_volume).detach()
        self.TRef = theta[:, 9:10]
        self.beta = theta[:, 10:11]
        self.gravity = theta[:, 11:14]
        self.hRef = theta[:, 14:15]

    def gh_face(self) -> torch.Tensor:
        g_f = self.gravity[self.batch_face]
        href_f = self.hRef[self.batch_face]
        return (self.face_pos * g_f).sum(dim=1, keepdim=True) - torch.norm(g_f, dim=1, keepdim=True) * href_f

    def get_separated_losses(self) -> Tuple[Optional[torch.Tensor], ...]:
        return (
            getattr(self, "loss_cont_component", None),
            getattr(self, "loss_momentum_component", None),
            getattr(self, "loss_energy_component", None),
            getattr(self, "loss_phi_jump_component", None),
            getattr(self, "loss_p_rghRef_component", None),
        )
