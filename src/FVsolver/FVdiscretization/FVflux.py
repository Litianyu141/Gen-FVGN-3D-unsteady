"""Face fluxes of the convection, diffusion and continuity terms."""

from typing import Optional, Tuple

import torch
from torch_geometric.utils import scatter

from FVsolver.FVdiscretization.FVInterpolation import FV_Interpolation


class FV_flux(FV_Interpolation):
    """Convective, diffusive and mass fluxes on the faces of the cached mesh."""

    def convective_flux(
        self,
        uvw_flux_face: torch.Tensor,
        phi_cpd_cell: torch.Tensor,
        grad_phi_cpd_cell: torch.Tensor,
        bounded: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Linear-upwind convective flux ``rho (u.n) phi_f`` per face and, for a bounded
        scheme, the cell source ``-div(u) phi`` (OpenFOAM ``bounded Gauss``).
        """
        phi_face = self.div_phic_to_faces(
            uvw_flux_face=uvw_flux_face,
            phi_cpd_cell=phi_cpd_cell,
            grad_phi_cpd_cell=grad_phi_cpd_cell,
        )
        convection_flux_face = self.convection_coeff * uvw_flux_face * phi_face
        bounded_source = (self.bounded_convection_source(uvw_flux_face, phi_cpd_cell)
                          if bounded else None)
        return convection_flux_face, bounded_source

    def bounded_convection_source(self, uvw_flux_face: torch.Tensor, phi_cpd_cell: torch.Tensor) -> torch.Tensor:
        """``-(sum_f u_f.S_f) phi_P`` on the interior cells, in compound-cell layout."""
        volumetric_flux_cells_face = (
            uvw_flux_face.index_select(-2, self.cells_face) * self.directed_area_cells_face
        )
        div_phi_volume = scatter(
            volumetric_flux_cells_face, self.cells_face_ptr,
            dim=-2, dim_size=self.num_cells, reduce="sum",
        )
        interior_source = -div_phi_volume * phi_cpd_cell[..., self.mask_interior_cell, :]
        return scatter(
            interior_source, self._interior_cell_indices,
            dim=-2, dim_size=phi_cpd_cell.shape[-2], reduce="sum",
        )

    def diffusion_flux(self, grad_uvw_flux_face, diffusivity=None):
        """Diffusive flux ``nu * snGrad`` per face."""
        nu_face = self.nu_coeff if diffusivity is None else diffusivity
        return grad_uvw_flux_face * nu_face

    def continuity_flux(self, uvw_flux_face):
        """Mass flux of each face, gathered to the cell-face incidence list."""
        return (self.continuity_eq_coeff[self.batch_face] * uvw_flux_face).index_select(-2, self.cells_face)
