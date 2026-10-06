"""A pool slot as a heterogeneous graph (nodes, faces, cells, compound cells) and the
attribute views the solver reads from a batch of them."""

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from FVdomain.Graph_data import HeteroGraph


class HeteroGraphDataset(InMemoryDataset):
    """The slots of a ``MmapDataPool`` as graphs."""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        """``(graph, fvPatch, boundary conditions)`` of slot ``idx``."""
        pool_loss_mode = getattr(self.base_dataset, "loss_mode", "fvm")
        slot = self.base_dataset.slots[idx]
        meta = slot.meta_objects
        hetero_graph = HeteroGraph()

        def _t(key, dtype=torch.float32):  # a writable copy of a shared static mesh tensor
            return slot.get_mesh_tensor(key).clone().to(dtype)

        # nodes
        node_pos = _t("node|node_pos", torch.float32)
        hetero_graph['node'].pos = node_pos
        hetero_graph['node'].num_nodes = node_pos.shape[0]

        # faces
        face_area = _t("face|face_area", torch.float32)
        face_type = _t("face|face_type", torch.int64).squeeze()
        face_pos = _t("face|face_pos", torch.float32)
        face_normal = _t("face|face_normal", torch.float32)

        hetero_graph['face'].pos = face_pos
        hetero_graph['face'].face_area = face_area
        hetero_graph['face'].face_type = face_type
        hetero_graph['face'].face_normal = face_normal
        hetero_graph['face'].num_nodes = face_pos.shape[0]

        # cells
        cell_volume = _t("cell|cell_volume", torch.float32)
        cells_face_unv = _t("cells_face_normal", torch.float32)
        hetero_graph['cell'].cell_volume = cell_volume
        hetero_graph['cell'].num_nodes = cell_volume.shape[0]
        hetero_graph['cell'].cells_face_unv = cells_face_unv

        # compound cells (interior cells, then one ghost cell per boundary face)
        cpd_cell_pos = _t("cpd|cell_pos", torch.float32)
        cell_type = _t("cpd|cell_type", torch.int64)

        # the slot's current state: the previous prediction, or the initial state after a reset
        phi_cpd_cell = slot.phi_tensor.clone()
        init_phi_cpd_cell = meta["init_phi_cpd_cell"].to(torch.float32)

        is_data_driven = (pool_loss_mode == "data_driven" and slot.has_labels)
        phi_GT_cpd_cell = None
        if is_data_driven:
            t_target = min(slot.time_step, slot.n_timesteps - 1)
            phi_GT_np = np.asarray(slot.get_label_at(t_target)).copy()
            phi_GT_cpd_cell = torch.from_numpy(phi_GT_np).to(torch.float32)

        hetero_graph['cpd_cell'].pos = cpd_cell_pos
        hetero_graph['cpd_cell'].cell_type = cell_type
        hetero_graph['cpd_cell'].x = phi_cpd_cell
        hetero_graph['cpd_cell'].init_x = init_phi_cpd_cell
        hetero_graph['cpd_cell'].slot_idx = torch.tensor([idx], dtype=torch.long)
        hetero_graph['cpd_cell'].num_nodes = cpd_cell_pos.shape[0]

        # per graph
        theta_PDE = meta["theta_PDE"].to(torch.float32)
        pRefCell_valid_mask = torch.tensor(
            [meta["fvconfig"]["fvSolution"]["pRefCell_valid_mask"]]
        ).bool()
        raw_pRefCell = meta["fvconfig"]["fvSolution"]["pRefCell"]
        pRefCell = torch.tensor([max(0, raw_pRefCell)]).long()
        pRefValue = torch.tensor(
            [meta["fvconfig"]["fvSolution"]["pRefValue"]]
        ).to(torch.float32)

        hetero_graph['graph'].theta_PDE = theta_PDE
        hetero_graph['graph'].graph_index = torch.tensor([idx], dtype=torch.long)
        hetero_graph['graph'].num_nodes = 1
        hetero_graph['graph'].pRefCell_valid_mask = pRefCell_valid_mask
        hetero_graph['graph'].pRefCell = pRefCell
        hetero_graph['graph'].pRefValue = pRefValue

        # connectivity
        hetero_graph['cell', 'bounded_by', 'face'].cells_face = _t("cells_face", torch.int64)
        hetero_graph['cell', 'bounded_by', 'face'].cells_face_ptr = _t("cells_face_ptr", torch.int64)

        neighbor_cpd_cell = _t("cpd|neighbor_cell", torch.int64)
        hetero_graph['cpd_cell', 'neighbors', 'cpd_cell'].neighbor_cpd_cell = neighbor_cpd_cell

        neighbor_cpd_cell_x = _t("cpd|neighbor_cell_x", torch.int64)
        A_cell_to_cell = _t("A_cell_to_cell", torch.float32)
        single_B_cell_to_cell = _t("single_B_cell_to_cell", torch.float32)

        hetero_graph['cpd_cell', 'wlsq_neighbors', 'cpd_cell'].neighbor_cpd_cell_x = neighbor_cpd_cell_x
        hetero_graph['cpd_cell'].A_cell_to_cell = A_cell_to_cell
        hetero_graph['cpd_cell'].single_B_cell_to_cell = single_B_cell_to_cell

        gnn_edge_counts = _t("gnn|edge_counts", torch.int64)
        gnn_edge_indices = _t("gnn|edge_indices", torch.int64)
        gnn_edge_sort_idx = _t("gnn|edge_sort_idx", torch.int64)

        hetero_graph['cpd_cell'].gnn_edge_counts = gnn_edge_counts
        hetero_graph['cpd_cell'].gnn_edge_indices = gnn_edge_indices
        hetero_graph['cpd_cell'].gnn_edge_sort_idx = gnn_edge_sort_idx

        # entity counts, for the index offsets of batching
        hetero_graph._num_nodes = node_pos.shape[0]
        hetero_graph._num_faces = face_pos.shape[0]
        hetero_graph._num_cells = cell_volume.shape[0]
        hetero_graph._num_cpd_cells = cpd_cell_pos.shape[0]
        hetero_graph._num_patches = len(meta['patch_dict'])
        hetero_graph._num_twoway_edges = gnn_edge_sort_idx.shape[0]

        # data loss: the label of the next time step
        if phi_GT_cpd_cell is not None:
            hetero_graph['cpd_cell'].phi_GT_cpd_cell = phi_GT_cpd_cell

        fvpatch = meta['fvpatch']
        bcpatch = meta['bcpatch']

        return hetero_graph, fvpatch, bcpatch

class HeteroFVGraph:
    """A batch of graphs with attribute views by entity (``graph_face``, ``graph_cell``,
    ``graph_cell_x`` for the least-squares stencil, ``graph_Index`` per graph), the batched
    fvPatch and the boundary conditions."""

    def __init__(self, hetero_data, fvpatch_batch, bcpatch_batch):
        self._hetero_data = hetero_data
        self.fvpatch_batch = fvpatch_batch
        self.bcpatch_batch = bcpatch_batch
        self._create_compatibility_views()

    def _create_compatibility_views(self):
        self.graph_node = self._NodeView(self._hetero_data)
        self.graph_face = self._FaceView(self._hetero_data)
        self.graph_cell = self._CellView(self._hetero_data)
        self.graph_cell_x = self._CellXView(self._hetero_data)
        self.graph_Index = self._IndexView(self._hetero_data) # per graph


    class _NodeView:
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        @property
        def pos(self):
            return self._hetero_data['node'].pos

        @property
        def _num_nodes(self):
            return self._hetero_data['node'].num_nodes


    class _FaceView:
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        @property
        def pos(self):
            return self._hetero_data['face'].pos

        @property
        def face_area(self):
            return self._hetero_data['face'].face_area

        @property
        def face_type(self):
            return self._hetero_data['face'].face_type

        @property
        def face_normal(self):
            return self._hetero_data['face'].face_normal

        @property
        def cells_face(self):
            return self._hetero_data['cell', 'bounded_by', 'face'].cells_face

        @property
        def _num_faces(self):
            return self._hetero_data['face'].num_nodes

        @property
        def batch(self):
            return getattr(self._hetero_data['face'], 'batch', None)


    class _CellView:
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        @property
        def x(self):
            return self._hetero_data['cpd_cell'].x

        @x.setter
        def x(self, value):
            self._hetero_data['cpd_cell'].x = value

        @property
        def init_x(self):
            return self._hetero_data['cpd_cell'].init_x

        @property
        def neighbor_cpd_cell(self):
            return self._hetero_data['cpd_cell', 'neighbors', 'cpd_cell'].neighbor_cpd_cell

        @property
        def cell_volume(self):
            return self._hetero_data['cell'].cell_volume

        @property
        def cell_type(self):
            return self._hetero_data['cpd_cell'].cell_type

        @property
        def slot_idx(self):
            return self._hetero_data['cpd_cell'].slot_idx

        @property
        def cells_face_unv(self):
            return self._hetero_data['cell'].cells_face_unv

        @property
        def cpd_cell_pos(self):
            return self._hetero_data['cpd_cell'].pos

        @property
        def cells_face_ptr(self):
            return self._hetero_data['cell', 'bounded_by', 'face'].cells_face_ptr

        @property
        def _num_graphs(self):
            return self._hetero_data['cpd_cell'].batch.max().item() + 1

        @property
        def _num_cells(self):
            return self._hetero_data['cell'].num_nodes

        @property
        def _num_cpd_cells(self):
            return self._hetero_data['cpd_cell'].num_nodes

        @property
        def batch(self):
            return getattr(self._hetero_data['cpd_cell'], 'batch', None)

        @property
        def gnn_edge_counts(self):
            return self._hetero_data['cpd_cell'].gnn_edge_counts

        @property
        def gnn_edge_indices(self):
            return self._hetero_data['cpd_cell'].gnn_edge_indices

        @property
        def gnn_edge_sort_idx(self):
            return self._hetero_data['cpd_cell'].gnn_edge_sort_idx

        @property
        def phi_GT_cpd_cell(self):
            return self._hetero_data['cpd_cell'].phi_GT_cpd_cell


    class _CellXView:
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        @property
        def neighbor_cpd_cell_x(self):
            return self._hetero_data['cpd_cell', 'wlsq_neighbors', 'cpd_cell'].neighbor_cpd_cell_x

        @property
        def A_cell_to_cell(self):
            return self._hetero_data['cpd_cell'].A_cell_to_cell

        @property
        def single_B_cell_to_cell(self):
            return self._hetero_data['cpd_cell'].single_B_cell_to_cell


    class _IndexView:
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        @property
        def theta_PDE(self):
            return self._hetero_data['graph'].theta_PDE

        @property
        def pRefCell_valid_mask(self):
            return self._hetero_data['graph'].pRefCell_valid_mask

        @property
        def pRefCell(self):
            return self._hetero_data['graph'].pRefCell

        @property
        def pRefValue(self):
            return self._hetero_data['graph'].pRefValue

    def to(self, *args, exclude_keys=None, **kwargs):
        """Move the underlying hetero data to device/dtype."""
        self._hetero_data = self._hetero_data.to(*args, exclude_keys=exclude_keys, **kwargs)
        self._create_compatibility_views()
        return self
