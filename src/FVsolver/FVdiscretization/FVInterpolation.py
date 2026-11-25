"""Mesh geometry registration, caching, and face/boundary geometric coefficients for Warp FVM."""

import logging
import os

import torch
from torch_geometric.utils import scatter

from Utils.utilities import FaceType

logger = logging.getLogger(__name__)


class FV_Interpolation:
    """Register mesh geometry and cache face interpolation / snGrad geometric coefficients."""

    def __init__(self):
        self.gC = None
        self.gF = None
        self.CF = None
        self.dCF = None
        self.eCF = None
        self.Cf = None
        self.Ff = None
        self.deltaCoeffs = None
        self.nonOrthDeltaCoeffs = None
        self.correctionVectors = None
        self.C_senders = None
        self.F_receivers = None
        self.registered = False

    def register_geometrics(self, fv_graph, force_register: bool = False):
        if self.registered and not force_register:
            return True

        graph_node = fv_graph.graph_node
        graph_face = fv_graph.graph_face
        graph_cell = fv_graph.graph_cell
        graph_cell_x = fv_graph.graph_cell_x

        self.fvpatch_batch = fv_graph.fvpatch_batch
        self.bcpatch_batch = fv_graph.bcpatch_batch

        shape_sig = (
            int(graph_node._num_nodes),
            int(graph_face._num_faces),
            int(graph_cell._num_cells),
            int(graph_cell._num_cpd_cells),
            int(graph_cell.neighbor_cpd_cell.shape[-1]),
            int(graph_cell_x.neighbor_cpd_cell_x.shape[-1]),
        )
        batch_sig = getattr(fv_graph, "_mesh_batch_sig", None)
        if batch_sig is not None and os.getenv("GEOM_CACHE_SELFCHECK") != "1":
            mesh_sig = ("batch_sig", batch_sig, shape_sig)
        else:
            cpd = graph_cell.cpd_cell_pos.reshape(-1)
            nc = cpd.numel()
            content_sig = (
                int(graph_cell.neighbor_cpd_cell.sum()),
                int(graph_cell_x.neighbor_cpd_cell_x.sum()),
                float(cpd[0]),
                float(cpd[nc // 3]),
                float(cpd[2 * nc // 3]),
                float(cpd[-1]),
            )
            mesh_sig = (shape_sig, content_sig)

        if self.registered and getattr(self, "_geom_mesh_sig", None) == mesh_sig:
            return True
        self._geom_mesh_sig = mesh_sig

        self.num_nodes = graph_node._num_nodes
        self.num_faces = graph_face._num_faces
        self.num_cells = graph_cell._num_cells
        self.num_cpd_cells = graph_cell._num_cpd_cells
        self.num_graphs = graph_cell._num_graphs

        self.node_pos = graph_node.pos
        self.cells_node = graph_node.cells_node.view(-1)
        self.cells_node_ptr = graph_cell.cells_node_ptr.view(-1)
        self.cells_face = graph_face.cells_face.view(-1)
        self.cells_face_ptr = graph_cell.cells_face_ptr.view(-1)
        self.cell_volume = graph_cell.cell_volume.view(-1, 1)
        self.cells_face_unv = graph_cell.cells_face_unv.view(-1, 3)
        self.face_type = graph_face.face_type.view(-1)
        self.face_pos = graph_face.pos.view(-1, 3)
        self.face_area = graph_face.face_area.view(-1, 1)
        self.face_node = graph_node.face_node.view(-1)
        self.face_node_ptr = graph_face.face_node_ptr.view(-1)
        self.unv_face = graph_face.face_normal.view(-1, 3)
        self.cell_type = graph_cell.cell_type.view(-1)
        self.cpd_cell_pos = graph_cell.cpd_cell_pos.view(-1, 3)

        self.neighbor_cpd_cell_x = graph_cell_x.neighbor_cpd_cell_x
        self.A_cell_to_cell = graph_cell_x.A_cell_to_cell
        self.single_B_cell_to_cell = graph_cell_x.single_B_cell_to_cell

        self.neighbor_cpd_cell = graph_cell.neighbor_cpd_cell
        self.cpd_owner = self.C_senders = self.neighbor_cpd_cell[0]
        self.cpd_neighbor = self.F_receivers = self.neighbor_cpd_cell[1]

        self.mask_valid_cyclic_cpd_cell = self.cell_type == FaceType.CYCLIC
        self.mask_non_cyclic_cpd_cell = ~self.mask_valid_cyclic_cpd_cell
        self.mask_interior_cell = self.cell_type == FaceType.NORMAL
        self.mask_boundary_cell = ~self.mask_interior_cell
        self._interior_cell_indices = self.mask_interior_cell.nonzero(as_tuple=False).squeeze(-1)

        self.mask_interior_face = self.face_type == FaceType.NORMAL
        self.mask_boundary_face = ~self.mask_interior_face
        self.mask_boundary_non_cyclic_face = self.mask_boundary_face & (self.face_type != FaceType.CYCLIC)
        self.mask_boundary_non_cyclic_cpd_cell = self.mask_boundary_cell & (self.cell_type != FaceType.CYCLIC)

        self.cells_face_surface_vec = self.cells_face_unv * self.face_area[self.cells_face]
        direction = torch.sign((self.cells_face_unv * self.unv_face[self.cells_face]).sum(dim=1, keepdim=True))
        self.directed_area_cells_face = (direction * self.face_area[self.cells_face]).view(-1, 1)

        self.batch_face = graph_face.batch
        self.batch_cell = graph_cell.batch[self.mask_interior_cell]
        self.batch_cpd_cell = graph_cell.batch

        self.mask_empty_cpd_cell = self.cell_type == FaceType.EMPTY
        self.mask_non_empty_cpd_cell = ~self.mask_empty_cpd_cell
        self.mask_empty_face = self.face_type == FaceType.EMPTY
        self.mask_non_empty_cells_face = ~self.mask_empty_face[self.cells_face]

        if self.mask_empty_face.any():
            z_coords = fv_graph.graph_node.pos[:, 2]
            node_batch = fv_graph.graph_node.batch
            if self.num_graphs > 1 and node_batch is not None:
                z_max = scatter(z_coords, node_batch, dim=0, dim_size=self.num_graphs, reduce="max")
                z_min = scatter(z_coords, node_batch, dim=0, dim_size=self.num_graphs, reduce="min")
                z_thick = torch.abs(z_max - z_min)
                has_empty = scatter(
                    self.mask_empty_face.long(), self.batch_face, dim=0, dim_size=self.num_graphs, reduce="max"
                ) > 0
                valid = has_empty & (z_thick > 1e-12)
                if valid.any():
                    if valid.all() and bool((z_thick == z_thick[0]).all()):
                        zt = z_thick[0]
                        self.face_area = self.face_area / zt
                        self.cell_volume = self.cell_volume / zt
                        self.directed_area_cells_face = self.directed_area_cells_face / zt
                        self.cells_face_surface_vec = self.cells_face_surface_vec / zt
                    else:
                        z_div = torch.where(valid, z_thick, torch.ones_like(z_thick)).unsqueeze(-1)
                        z_face = z_div[self.batch_face]
                        self.face_area = self.face_area / z_face
                        self.cell_volume = self.cell_volume / z_div[self.batch_cell]
                        z_cf = z_face[self.cells_face]
                        self.directed_area_cells_face = self.directed_area_cells_face / z_cf
                        self.cells_face_surface_vec = self.cells_face_surface_vec / z_cf
            else:
                z_thick = torch.abs(z_coords.max() - z_coords.min())
                if z_thick > 1e-12:
                    self.face_area = self.face_area / z_thick
                    self.cell_volume = self.cell_volume / z_thick
                    self.directed_area_cells_face = self.directed_area_cells_face / z_thick
                    self.cells_face_surface_vec = self.cells_face_surface_vec / z_thick

        self.calc_geometry_coeff()
        self.registered = True
        return True

    def calc_geometry_coeff(self):
        """Compute distance-based face interpolation weights and snGrad geometry vectors."""
        self.CF = (self.cpd_cell_pos[self.F_receivers] - self.cpd_cell_pos[self.C_senders])[:, :, None]
        self.dCF = self.CF.norm(dim=1, keepdim=True)
        self.eCF = self.CF / self.dCF

        dfF = torch.norm(self.cpd_cell_pos[self.F_receivers] - self.face_pos, dim=1, keepdim=True)[:, :, None]
        self.gC = dfF / self.dCF
        self.gF = 1.0 - self.gC

        self.Cf = (self.face_pos - self.cpd_cell_pos[self.C_senders])[:, :, None]
        self.Ff = (self.face_pos - self.cpd_cell_pos[self.F_receivers])[:, :, None]

        CF_vec = self.CF.squeeze(2)
        dCF_mag = self.dCF.squeeze(2)
        n_dot_d = torch.sum(CF_vec * self.unv_face, dim=1, keepdim=True)
        self.deltaCoeffs = 1.0 / torch.clamp(dCF_mag, min=1e-12)
        self.nonOrthDeltaCoeffs = 1.0 / torch.clamp(n_dot_d, min=0.05 * dCF_mag)
        self.correctionVectors = self.unv_face - CF_vec * self.nonOrthDeltaCoeffs

    def _invalidate_boundary_caches(self) -> None:
        source = (id(self.bcpatch_batch), id(getattr(self, "correctionVectors", None)))
        if self.__dict__.get("_boundary_cache_source") != source:
            self._boundary_cache_source = source
            self._dirichlet_face_keepalive = (
                self.bcpatch_batch,
                getattr(self, "correctionVectors", None),
            )
            self._dirichlet_face_cache = {}
            self._sngrad_operand_cache = {}

    def dirichlet_boundary_face_mask(self, var_type) -> torch.Tensor:
        """Boolean ``[N_face]`` mask of non-cyclic boundary faces with a FIRST (Dirichlet) BC for ``var_type``."""
        self._invalidate_boundary_caches()
        key = int(var_type)
        if key in self._dirichlet_face_cache:
            return self._dirichlet_face_cache[key]

        mask = torch.zeros_like(self.mask_boundary_non_cyclic_face)
        batch, patches = self.bcpatch_batch, self.fvpatch_batch
        if batch is not None and patches is not None:
            for case_i, conditions in enumerate(batch.first):
                offset = int(batch.first_patch_idx_offset[case_i])
                for bc in conditions:
                    if int(bc.varType) != key:
                        continue
                    patch = int(bc.patch_idx) + offset
                    mask[int(patches.start_idx_face[patch]) : int(patches.end_idx_face[patch])] = True
        mask = mask & self.mask_boundary_non_cyclic_face
        self._dirichlet_face_cache[key] = mask
        return mask

    def facecv_active_face_mask(self, face_interior, pressure_var_type, boundary_rows: str) -> torch.Tensor:
        """Boolean ``[N_face, 1]`` mask of faces carrying a face-CV momentum equation."""
        interior_only = (
            face_interior
            & self.mask_interior_face.view(-1, 1).bool()
            & (~self.mask_empty_face.view(-1, 1).bool())
        )
        if boundary_rows == "none":
            return interior_only
        if boundary_rows == "pressureDirichlet":
            return interior_only | self.dirichlet_boundary_face_mask(pressure_var_type).view(-1, 1)
        if boundary_rows == "all":
            return ~self.mask_empty_face.view(-1, 1).bool()
        raise ValueError(f"faceCvBoundaryRows={boundary_rows!r}: expected 'none', 'pressureDirichlet', or 'all'")

    def _snGrad_operands(self, var_type):
        """Return ``(owner_weight, neighbour_weight, correction_vectors)`` for boundary-aware corrected snGrad."""
        self._invalidate_boundary_caches()
        key = None if var_type is None else int(var_type)
        if key in self._sngrad_operand_cache:
            return self._sngrad_operand_cache[key]

        boundary = self.mask_boundary_non_cyclic_face
        dirichlet = (
            torch.zeros_like(boundary)
            if var_type is None
            else self.dirichlet_boundary_face_mask(var_type)
        )
        owner = torch.where(dirichlet.view(-1, 1, 1), torch.ones_like(self.gC), self.gC)
        neighbour = torch.where(dirichlet.view(-1, 1, 1), torch.zeros_like(self.gF), self.gF)
        vectors = torch.where(
            (boundary & ~dirichlet).view(-1, 1),
            torch.zeros_like(self.correctionVectors),
            self.correctionVectors,
        )
        self._sngrad_operand_cache[key] = (owner, neighbour, vectors)
        return self._sngrad_operand_cache[key]
