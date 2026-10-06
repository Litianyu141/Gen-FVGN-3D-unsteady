"""Mesh-geometry cache and cell-to-face interpolation of the finite-volume operators."""

import torch

from Utils.utilities import FaceType
from FVsolver.FVdiscretization.schemes import (
    batched_matvec,
    corrected_sn_grad,
    linear_corrected_face_value,
    linear_upwind_face_value,
)


class FV_Interpolation:
    """Caches the geometry of a batched mesh graph and interpolates cell fields to faces.

    Cell fields live on compound cells (``cpd_cell``): the interior cells followed by one
    ghost cell per boundary face.  Interpolation weights follow Moukalled et al.
    """

    def __init__(self):
        self.fvpatch_batch = None
        self.bcpatch_batch = None
        self.registered = False

    def register_geometrics(self, fv_graph, force_register=False):
        """Cache the geometric quantities of ``fv_graph``; a no-op when the mesh is unchanged."""
        graph_node = fv_graph.graph_node
        graph_face = fv_graph.graph_face
        graph_cell = fv_graph.graph_cell
        graph_cell_x = fv_graph.graph_cell_x

        self.fvpatch_batch = fv_graph.fvpatch_batch
        self.bcpatch_batch = fv_graph.bcpatch_batch

        cpd = graph_cell.cpd_cell_pos.reshape(-1)
        n = cpd.numel()
        mesh_sig = (
            (int(graph_node._num_nodes), int(graph_face._num_faces),
             int(graph_cell._num_cells), int(graph_cell._num_cpd_cells),
             int(graph_cell.neighbor_cpd_cell.shape[-1]),
             int(graph_cell_x.neighbor_cpd_cell_x.shape[-1])),
            (int(graph_cell.neighbor_cpd_cell.sum()),
             int(graph_cell_x.neighbor_cpd_cell_x.sum()),
             float(cpd[0]), float(cpd[n // 3]), float(cpd[2 * n // 3]), float(cpd[-1])),
        )
        if self.registered and getattr(self, "_geom_mesh_sig", None) == mesh_sig:
            return True
        self._geom_mesh_sig = mesh_sig

        self.num_faces = graph_face._num_faces
        self.num_cells = graph_cell._num_cells
        self.num_cpd_cells = graph_cell._num_cpd_cells
        self.num_graphs = graph_cell._num_graphs

        self.cells_face = graph_face.cells_face.view(-1)
        self.cells_face_ptr = graph_cell.cells_face_ptr.view(-1)
        self.cell_volume = graph_cell.cell_volume.view(-1, 1)
        cells_face_unv = graph_cell.cells_face_unv.view(-1, 3)
        self.face_type = graph_face.face_type.view(-1)
        self.face_pos = graph_face.pos.view(-1, 3)
        self.face_area = graph_face.face_area.view(-1, 1)
        self.unv_face = graph_face.face_normal.view(-1, 3)
        self.cell_type = graph_cell.cell_type.view(-1)
        self.cpd_cell_pos = graph_cell.cpd_cell_pos.view(-1, 3)

        # WLSQ stencil and its precomputed normal matrices
        self.neighbor_cpd_cell_x = graph_cell_x.neighbor_cpd_cell_x
        self.A_cell_to_cell = graph_cell_x.A_cell_to_cell
        self.single_B_cell_to_cell = graph_cell_x.single_B_cell_to_cell

        # face owner (C) and neighbour (F) compound cells
        self.neighbor_cpd_cell = graph_cell.neighbor_cpd_cell
        self.C_senders = self.neighbor_cpd_cell[0]
        self.F_receivers = self.neighbor_cpd_cell[1]

        self.mask_interior_cell = self.cell_type == FaceType.NORMAL
        self.mask_boundary_cell = ~self.mask_interior_cell
        self._interior_cell_indices = self.mask_interior_cell.nonzero(as_tuple=False).squeeze(-1)
        self.mask_interior_face = self.face_type == FaceType.NORMAL
        self.mask_boundary_face = ~self.mask_interior_face
        self.mask_boundary_non_cyclic_face = self.mask_boundary_face & (self.face_type != FaceType.CYCLIC)
        mask_boundary_non_cyclic_cpd_cell = self.mask_boundary_cell & (self.cell_type != FaceType.CYCLIC)
        self._boundary_non_cyclic_cpd_indices = mask_boundary_non_cyclic_cpd_cell.nonzero(
            as_tuple=False).squeeze(-1)

        # face area signed by the outward direction of each cell's face
        undirected_unv_cells_face = self.unv_face[self.cells_face]
        direction_cells_face = torch.sign((cells_face_unv * undirected_unv_cells_face).sum(dim=1, keepdim=True))
        self.directed_area_cells_face = (direction_cells_face * (self.face_area[self.cells_face])).view(-1, 1)

        self.batch_face = graph_face.batch
        self.batch_cell = graph_cell.batch[self.mask_interior_cell]
        self.batch_cpd_cell = graph_cell.batch

        self.mask_empty_face = self.face_type == FaceType.EMPTY
        self.mask_non_empty_cells_face = ~self.mask_empty_face[self.cells_face]
        self._non_empty_cells_face_indices = self.mask_non_empty_cells_face.nonzero(as_tuple=False).squeeze(-1)

        self.calc_geometry_coeff()
        self.registered = True

    def calc_geometry_coeff(self):
        """Interpolation weights ``gC``/``gF``, centre-to-face vectors and the snGrad
        coefficients (OpenFOAM ``nonOrthDeltaCoeffs`` and ``nonOrthCorrectionVectors``).
        """
        self.CF = (self.cpd_cell_pos[self.F_receivers] - self.cpd_cell_pos[self.C_senders])[:, :, None]
        self.dCF = self.CF.norm(dim=1, keepdim=True)
        self.eCF = self.CF / self.dCF

        dfF = torch.norm(self.cpd_cell_pos[self.F_receivers] - self.face_pos, dim=1, keepdim=True)[:, :, None]
        self.gC = dfF / self.dCF
        self.gF = 1.0 - self.gC

        self.Cf = (self.face_pos - self.cpd_cell_pos[self.C_senders])[:, :, None]
        self.Ff = (self.face_pos - self.cpd_cell_pos[self.F_receivers])[:, :, None]

        CF_vec = self.CF.squeeze(2)
        n_dot_d = torch.sum(CF_vec * self.unv_face, dim=1, keepdim=True)
        dCF_mag = self.dCF.squeeze(2)
        self.nonOrthDeltaCoeffs = 1.0 / torch.clamp(n_dot_d, min=0.05 * dCF_mag)
        self.correctionVectors = self.unv_face - CF_vec * self.nonOrthDeltaCoeffs

    def _invalidate_boundary_caches(self) -> None:
        """Drop the per-variable boundary masks when the BC batch or the geometry changes."""
        source = (id(self.bcpatch_batch), id(getattr(self, "correctionVectors", None)))
        if self.__dict__.get("_boundary_cache_source") != source:
            self._boundary_cache_source = source
            self._dirichlet_face_keepalive = (self.bcpatch_batch, getattr(self, "correctionVectors", None))
            self._dirichlet_face_cache = {}
            self._sngrad_operand_cache = {}

    def dirichlet_boundary_face_mask(self, var_type) -> torch.Tensor:
        """``[N_face]`` bool: non-cyclic boundary faces whose ghost holds a prescribed value."""
        self._invalidate_boundary_caches()
        cache = self._dirichlet_face_cache
        key = int(var_type)
        if key in cache:
            return cache[key]

        mask = torch.zeros_like(self.mask_boundary_non_cyclic_face)
        batch, patches = self.bcpatch_batch, self.fvpatch_batch
        for case_i, conditions in enumerate(batch.first):
            offset = int(batch.first_patch_idx_offset[case_i])
            for bc in conditions:
                if int(bc.varType) != key:
                    continue
                patch = int(bc.patch_idx) + offset
                mask[int(patches.start_idx_face[patch]):int(patches.end_idx_face[patch])] = True
        mask = mask & self.mask_boundary_non_cyclic_face
        cache[key] = mask
        return mask

    def _snGrad_operands(self, var_type):
        """``(owner_weight, neighbour_weight, correction_vectors)`` of the corrected snGrad.

        On a Dirichlet face the ghost value sits on the face, so the face gradient is the
        owner's; on other non-cyclic boundary faces the correction is dropped.
        """
        self._invalidate_boundary_caches()
        cache = self._sngrad_operand_cache
        key = None if var_type is None else int(var_type)
        if key in cache:
            return cache[key]

        boundary = self.mask_boundary_non_cyclic_face
        dirichlet = (torch.zeros_like(boundary) if var_type is None
                     else self.dirichlet_boundary_face_mask(var_type))
        owner = torch.where(dirichlet.view(-1, 1, 1), torch.ones_like(self.gC), self.gC)
        neighbour = torch.where(dirichlet.view(-1, 1, 1), torch.zeros_like(self.gF), self.gF)
        vectors = torch.where((boundary & ~dirichlet).view(-1, 1),
                              torch.zeros_like(self.correctionVectors), self.correctionVectors)
        cache[key] = (owner, neighbour, vectors)
        return cache[key]

    def compute_snGrad(self, phi_cpd_cell, grad_phi_cpd_cell=None, var_type=None):
        """Corrected surface-normal gradient ``[N_face, C]``; uncorrected without a gradient."""
        owner_weight, neighbor_weight, correction_vectors = self._snGrad_operands(var_type)
        return corrected_sn_grad(
            phi_cpd_cell,
            face_owner=self.C_senders,
            face_neighbor=self.F_receivers,
            non_orth_delta_coeffs=self.nonOrthDeltaCoeffs,
            grad_cell=grad_phi_cpd_cell,
            owner_weight=owner_weight,
            neighbor_weight=neighbor_weight,
            correction_vectors=correction_vectors,
        )

    def interpolating_gradients_to_faces(self, phi_cpd_cell, grad_phi_cpd_cell):
        """Face gradients ``[N_face, C, 3]``: linear interpolation of the cell gradients with
        the over-relaxed correction along the centre-to-centre direction on non-orthogonal
        faces (Moukalled et al., Eq. 9.46-9.48).
        """
        phi_cpd_cell_expanded = phi_cpd_cell[:, :, None]
        grad_f_hat = (grad_phi_cpd_cell[self.C_senders] * self.gC
                      + grad_phi_cpd_cell[self.F_receivers] * self.gF)

        eCF_hat = self.eCF.squeeze(2)
        cos_theta = torch.sum(self.unv_face * eCF_hat, dim=1, keepdim=True).abs().clamp(min=0.0, max=1.0)
        non_orth_factor = torch.sqrt((1.0 - cos_theta ** 2).clamp_min(0.0))
        non_orth_weight = (non_orth_factor > 1e-5).to(non_orth_factor.dtype)

        gradient_dot_eCF = batched_matvec(grad_f_hat, self.eCF)
        phi_difference_normalized = (phi_cpd_cell_expanded[self.F_receivers]
                                     - phi_cpd_cell_expanded[self.C_senders]) / self.dCF
        correction = (phi_difference_normalized - gradient_dot_eCF) * (self.eCF.transpose(1, 2))
        grad_f_hat += correction * non_orth_weight.unsqueeze(2)
        return grad_f_hat

    def interpolating_phic_to_faces(self, phi_cpd_cell, grad_phi_cpd_cell):
        """Face values ``[N_face, C]`` by gradient-corrected linear interpolation."""
        return linear_corrected_face_value(
            phi_cpd_cell, grad_phi_cpd_cell,
            cell_pos=self.cpd_cell_pos,
            face_pos=self.face_pos,
            face_owner=self.C_senders,
            face_neighbor=self.F_receivers,
            owner_weight=self.gC,
            neighbor_weight=self.gF,
        )

    def div_phic_to_faces(self, uvw_flux_face, phi_cpd_cell, grad_phi_cpd_cell):
        """Convected face values ``[N_face, C]`` by the linear-upwind scheme."""
        return linear_upwind_face_value(
            phi_cpd_cell, grad_phi_cpd_cell,
            face_flux=uvw_flux_face,
            face_owner=self.C_senders,
            face_neighbor=self.F_receivers,
            owner_to_face=self.Cf,
            neighbor_to_face=self.Ff,
            boundary_cpd_indices=self._boundary_non_cyclic_cpd_indices,
            boundary_face_mask=self.mask_boundary_non_cyclic_face,
        )
