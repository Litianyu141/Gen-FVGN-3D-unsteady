"""Batching rules of the heterogeneous mesh graph."""

import torch
from torch_geometric.data import HeteroData

# Index attributes are offset by the size of the entity set they point into when graphs are
# batched; everything else is concatenated as is.
_OFFSET_BY = {
    "cells_face": "_num_faces",
    "cells_face_ptr": "_num_cells",
    "neighbor_cpd_cell": "_num_cpd_cells",
    "neighbor_cpd_cell_x": "_num_cpd_cells",
    "gnn_edge_indices": "_num_cpd_cells",       # senders of the receiver-sorted edges
    "gnn_edge_sort_idx": "_num_twoway_edges",   # positions in the two-way edge list
    "pRefCell": "_num_cpd_cells",
}
_NO_OFFSET = (
    "gnn_edge_counts", "pRefCell_valid_mask", "pRefValue", "pos", "A_cell_to_cell",
    "single_B_cell_to_cell", "cell_volume", "graph_index", "theta_PDE", "x", "cell_type", "face_area",
    "face_type", "cells_face_unv", "_num_nodes", "_num_faces", "_num_cells", "_num_cpd_cells",
    "_num_twoway_edges",
)
_CAT_DIM = {
    "neighbor_cpd_cell_x": 1, "neighbor_cpd_cell": 1,
    **{k: 0 for k in (
        "x", "pos", "gnn_edge_counts", "gnn_edge_indices", "gnn_edge_sort_idx",
        "graph_index", "pRefCell", "pRefCell_valid_mask", "pRefValue", "cell_type", "cell_volume", "face_area",
        "face_type", "cells_face_unv", "cells_face", "cells_face_ptr",
        "A_cell_to_cell", "single_B_cell_to_cell", "_num_nodes", "_num_faces", "_num_cells", "_num_cpd_cells",
        "_num_twoway_edges")},
}


class HeteroGraph(HeteroData):
    """``HeteroData`` with the mesh's offset and concatenation rules."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in _OFFSET_BY:
            return getattr(self, _OFFSET_BY[key], 0)
        if key in _NO_OFFSET:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in _CAT_DIM:
            return _CAT_DIM[key]
        return super().__cat_dim__(key, value, *args, **kwargs)

    def to(self, *args, exclude_keys=None, **kwargs):
        """Move every tensor except ``exclude_keys`` to a device or dtype."""
        exclude_keys = set(exclude_keys or [])
        for store in self.stores:
            for key, item in store.items():
                if isinstance(item, torch.Tensor) and key not in exclude_keys:
                    store[key] = item.to(*args, **kwargs)
        return self
