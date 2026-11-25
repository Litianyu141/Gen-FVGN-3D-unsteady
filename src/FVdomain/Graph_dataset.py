"""Heterogeneous mesh graph dataset and compatibility wrapper."""

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from FVdomain.Graph_data import HeteroGraph


class HeteroGraphDataset(InMemoryDataset):
    """Build ``(HeteroGraph, fvPatch, bcpatch)`` triples from ``MmapDataPool`` or ``HeteroDataPool``."""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def meta_pool(self):
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.meta_pool)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self.get(i) for i in idx]
        return self.get(idx)

    def get(self, idx):
        if hasattr(self.base_dataset, "slots"):
            return self._get_mmap(idx)
        return self._get_inmemory(idx)

    def _get_inmemory(self, idx):
        data = self.meta_pool[idx]
        g = HeteroGraph()

        node_pos = data["node|node_pos"].to(torch.float32)
        g["node"].pos = node_pos
        g["node"].num_nodes = node_pos.shape[0]

        face_pos = data["face|face_pos"].to(torch.float32)
        g["face"].pos = face_pos
        g["face"].face_area = data["face|face_area"].to(torch.float32)
        g["face"].face_type = data["face|face_type"].long().squeeze()
        g["face"].face_normal = data["face|face_normal"].to(torch.float32)
        g["face"].num_nodes = face_pos.shape[0]

        cell_volume = data["cell|cell_volume"].to(torch.float32)
        g["cell"].cell_volume = cell_volume
        g["cell"].cells_face_unv = data["cells_face_normal"].to(torch.float32)
        g["cell"].num_nodes = cell_volume.shape[0]

        cpd_pos = data["cpd|cell_pos"].to(torch.float32)
        global_idx = data["global_idx"].long()
        g["cpd_cell"].pos = cpd_pos
        g["cpd_cell"].cell_type = data["cpd|cell_type"].long()
        g["cpd_cell"].x = self.base_dataset.phi_cpd_cell_pool[global_idx]
        g["cpd_cell"].init_x = data["init_phi_cpd_cell"].to(torch.float32)
        g["cpd_cell"].global_idx = global_idx
        g["cpd_cell"].num_nodes = cpd_pos.shape[0]

        fvsol = data["fvconfig"]["fvSolution"]
        g["graph"].theta_PDE = data["theta_PDE"].to(torch.float32)
        g["graph"].case_name = torch.tensor([ord(c) for c in data["case_name"]], dtype=torch.long)
        g["graph"].graph_index = torch.tensor([idx], dtype=torch.long)
        g["graph"].num_nodes = 1
        g["graph"].pRefCell_valid_mask = torch.tensor([fvsol["pRefCell_valid_mask"]]).bool()
        g["graph"].pRefCell = torch.tensor([max(0, fvsol["pRefCell"])]).long()
        g["graph"].pRefValue = torch.tensor([fvsol["pRefValue"]]).to(torch.float32)
        g["graph"].time_step = self.base_dataset.time_step_pool[idx]
        g["graph"].dt = torch.tensor(data["fvconfig"]["theta_pde"]["dt"]["val"]).to(torch.float32)

        g["face", "contains", "node"].face_node = data["face_node"].long()
        g["cell", "contains", "node"].cells_node = data["cells_node"].long()
        g["cell", "contains", "node"].cells_node_ptr = data["cells_node_ptr"].long()
        g["cell", "bounded_by", "face"].cells_face = data["cells_face"].long()
        g["cell", "bounded_by", "face"].cells_face_ptr = data["cells_face_ptr"].long()
        g["face", "bounds", "cell"].face_node_ptr = data["face_node_ptr"].long()
        g["cpd_cell", "neighbors", "cpd_cell"].neighbor_cpd_cell = data["cpd|neighbor_cell"].long()

        cyc = data["cyclic_face"].long()
        g["face"].cyclic_face = cyc.reshape(2, 0) if cyc.numel() == 0 else cyc

        single_b = data["single_B_cell_to_cell"].to(torch.float32)
        if single_b.ndim == 2:
            single_b = single_b.unsqueeze(-1)
        g["cpd_cell", "wlsq_neighbors", "cpd_cell"].neighbor_cpd_cell_x = data["cpd|neighbor_cell_x"].long()
        g["cpd_cell"].A_cell_to_cell = data["A_cell_to_cell"].to(torch.float32)
        g["cpd_cell"].single_B_cell_to_cell = single_b

        gnn_sort = data["gnn|edge_sort_idx"].long()
        g["cpd_cell"].gnn_edge_counts = data["gnn|edge_counts"].long()
        g["cpd_cell"].gnn_edge_indices = data["gnn|edge_indices"].long()
        g["cpd_cell"].gnn_edge_sort_idx = gnn_sort

        g._num_nodes = node_pos.shape[0]
        g._num_faces = face_pos.shape[0]
        g._num_cells = cell_volume.shape[0]
        g._num_cpd_cells = cpd_pos.shape[0]
        g._num_patches = len(data["patch_dict"])
        g._num_twoway_edges = gnn_sort.shape[0]
        return g, data["fvpatch"], data["bcpatch"]

    def _get_mmap(self, idx):
        pool_loss_mode = getattr(self.base_dataset, "loss_mode", "fvm")
        slot = self.base_dataset.slots[idx]
        meta = slot.meta_objects
        g = HeteroGraph()

        def _t(key, dtype=torch.float32):
            if slot._shared_mesh.has_tensor(key):
                return slot.get_mesh_tensor(key).clone().to(dtype)
            return meta[key].to(dtype)

        node_pos = _t("node|node_pos", torch.float32)
        g["node"].pos = node_pos
        g["node"].num_nodes = node_pos.shape[0]

        face_pos = _t("face|face_pos", torch.float32)
        g["face"].pos = face_pos
        g["face"].face_area = _t("face|face_area", torch.float32)
        g["face"].face_type = _t("face|face_type", torch.int64).squeeze()
        g["face"].face_normal = _t("face|face_normal", torch.float32)
        g["face"].num_nodes = face_pos.shape[0]

        cell_volume = _t("cell|cell_volume", torch.float32)
        g["cell"].cell_volume = cell_volume
        g["cell"].cells_face_unv = _t("cells_face_normal", torch.float32)
        g["cell"].num_nodes = cell_volume.shape[0]

        cpd_pos = _t("cpd|cell_pos", torch.float32)
        g["cpd_cell"].pos = cpd_pos
        g["cpd_cell"].cell_type = _t("cpd|cell_type", torch.int64)
        g["cpd_cell"].x = slot.phi_tensor.clone()
        g["cpd_cell"].init_x = meta["init_phi_cpd_cell"].to(torch.float32)
        g["cpd_cell"].slot_idx = torch.tensor([idx], dtype=torch.long)
        g["cpd_cell"].num_nodes = cpd_pos.shape[0]

        if pool_loss_mode == "data_driven" and slot.has_labels:
            t_target = min(slot.time_step, slot.n_timesteps - 1)
            phi_gt_np = np.asarray(slot.get_label_at(t_target)).copy()
            t_offset = getattr(slot, "_label_T_offset", None)
            if t_offset is not None:
                phi_gt_np[:, -1] += t_offset
            g["cpd_cell"].phi_GT_cpd_cell = torch.from_numpy(phi_gt_np).to(torch.float32)

        fvsol = meta["fvconfig"]["fvSolution"]
        g["graph"].theta_PDE = meta["theta_PDE"].to(torch.float32)
        g["graph"].case_name = torch.tensor([ord(c) for c in meta["case_name"]], dtype=torch.long)
        g["graph"].graph_index = torch.tensor([idx], dtype=torch.long)
        g["graph"].num_nodes = 1
        g["graph"].pRefCell_valid_mask = torch.tensor([fvsol["pRefCell_valid_mask"]]).bool()
        g["graph"].pRefCell = torch.tensor([max(0, fvsol["pRefCell"])]).long()
        g["graph"].pRefValue = torch.tensor([fvsol["pRefValue"]]).to(torch.float32)
        g["graph"].time_step = torch.tensor(slot.time_step, dtype=torch.float32)
        g["graph"].dt = torch.tensor(meta["fvconfig"]["theta_pde"]["dt"]["val"]).to(torch.float32)

        g["face", "contains", "node"].face_node = _t("face_node", torch.int64)
        g["cell", "contains", "node"].cells_node = _t("cells_node", torch.int64)
        g["cell", "contains", "node"].cells_node_ptr = _t("cells_node_ptr", torch.int64)
        g["cell", "bounded_by", "face"].cells_face = _t("cells_face", torch.int64)
        g["cell", "bounded_by", "face"].cells_face_ptr = _t("cells_face_ptr", torch.int64)
        g["face", "bounds", "cell"].face_node_ptr = _t("face_node_ptr", torch.int64)
        g["cpd_cell", "neighbors", "cpd_cell"].neighbor_cpd_cell = _t("cpd|neighbor_cell", torch.int64)

        cyc = _t("cyclic_face", torch.int64)
        g["face"].cyclic_face = cyc.reshape(2, 0) if cyc.numel() == 0 else cyc

        single_b = _t("single_B_cell_to_cell", torch.float32)
        if single_b.ndim == 2:
            single_b = single_b.unsqueeze(-1)
        g["cpd_cell", "wlsq_neighbors", "cpd_cell"].neighbor_cpd_cell_x = _t("cpd|neighbor_cell_x", torch.int64)
        g["cpd_cell"].A_cell_to_cell = _t("A_cell_to_cell", torch.float32)
        g["cpd_cell"].single_B_cell_to_cell = single_b

        gnn_sort = _t("gnn|edge_sort_idx", torch.int64)
        g["cpd_cell"].gnn_edge_counts = _t("gnn|edge_counts", torch.int64)
        g["cpd_cell"].gnn_edge_indices = _t("gnn|edge_indices", torch.int64)
        g["cpd_cell"].gnn_edge_sort_idx = gnn_sort

        g._num_nodes = node_pos.shape[0]
        g._num_faces = face_pos.shape[0]
        g._num_cells = cell_volume.shape[0]
        g._num_cpd_cells = cpd_pos.shape[0]
        g._num_patches = len(meta["patch_dict"])
        g._num_twoway_edges = gnn_sort.shape[0]
        return g, meta["fvpatch"], meta["bcpatch"]


class HeteroFVGraph:
    """Compatibility wrapper exposing node, face, cell, WLSQ, and graph-level views."""

    def __init__(self, hetero_data, fvpatch_batch, bcpatch_batch):
        self._hetero_data = hetero_data
        self.fvpatch_batch = fvpatch_batch
        self.bcpatch_batch = bcpatch_batch
        self._create_views()

    def _create_views(self):
        self.graph_node = self._NodeView(self._hetero_data)
        self.graph_face = self._FaceView(self._hetero_data)
        self.graph_cell = self._CellView(self._hetero_data)
        self.graph_cell_x = self._CellXView(self._hetero_data)
        self.graph_Index = self._IndexView(self._hetero_data)

    class _NodeView:
        def __init__(self, hd):
            self._hd = hd

        @property
        def pos(self): return self._hd["node"].pos
        @property
        def num_nodes(self): return self._hd["node"].num_nodes
        @property
        def _num_nodes(self): return self._hd["node"].num_nodes
        @property
        def face_node(self): return self._hd["face", "contains", "node"].face_node
        @property
        def cells_node(self): return self._hd["cell", "contains", "node"].cells_node
        @property
        def batch(self): return getattr(self._hd["node"], "batch", None)

    class _FaceView:
        def __init__(self, hd):
            self._hd = hd

        @property
        def pos(self): return self._hd["face"].pos
        @property
        def face_area(self): return self._hd["face"].face_area
        @property
        def face_type(self): return self._hd["face"].face_type
        @property
        def face_normal(self): return self._hd["face"].face_normal
        @property
        def cyclic_face(self): return self._hd["face"].cyclic_face
        @property
        def cells_face(self): return self._hd["cell", "bounded_by", "face"].cells_face
        @property
        def face_node_ptr(self): return self._hd["face", "bounds", "cell"].face_node_ptr
        @property
        def _num_faces(self): return self._hd["face"].num_nodes
        @property
        def batch(self): return getattr(self._hd["face"], "batch", None)

    class _CellView:
        def __init__(self, hd):
            self._hd = hd

        @property
        def x(self): return self._hd["cpd_cell"].x
        @x.setter
        def x(self, val): self._hd["cpd_cell"].x = val
        @property
        def init_x(self): return self._hd["cpd_cell"].init_x
        @init_x.setter
        def init_x(self, val): self._hd["cpd_cell"].init_x = val
        @property
        def pos(self): return self._hd["cpd_cell"].pos
        @property
        def cpd_cell_pos(self): return self._hd["cpd_cell"].pos
        @property
        def neighbor_cpd_cell(self): return self._hd["cpd_cell", "neighbors", "cpd_cell"].neighbor_cpd_cell
        @neighbor_cpd_cell.setter
        def neighbor_cpd_cell(self, val): self._hd["cpd_cell", "neighbors", "cpd_cell"].neighbor_cpd_cell = val
        @property
        def cell_volume(self): return self._hd["cell"].cell_volume
        @property
        def cell_type(self): return self._hd["cpd_cell"].cell_type
        @property
        def global_idx(self): return self._hd["cpd_cell"].global_idx
        @property
        def slot_idx(self):
            si = getattr(self._hd["cpd_cell"], "slot_idx", None)
            return si if si is not None else self._hd["cpd_cell"].global_idx
        @property
        def cells_face_unv(self): return self._hd["cell"].cells_face_unv
        @property
        def cells_node_ptr(self): return self._hd["cell", "contains", "node"].cells_node_ptr
        @property
        def cells_face_ptr(self): return self._hd["cell", "bounded_by", "face"].cells_face_ptr
        @property
        def _num_graphs(self):
            ptr = getattr(self._hd["cpd_cell"], "ptr", None)
            if ptr is not None:
                return int(ptr.shape[0]) - 1
            if "graph" in self._hd.node_types:
                theta = getattr(self._hd["graph"], "theta_PDE", None)
                if theta is not None and theta.ndim >= 2:
                    return int(theta.shape[0])
            batch = getattr(self._hd["cpd_cell"], "batch", None)
            return int(batch.max().item()) + 1 if batch is not None else 1
        @property
        def _num_cells(self): return self._hd["cell"].num_nodes
        @property
        def _num_cpd_cells(self): return self._hd["cpd_cell"].num_nodes
        @property
        def batch(self): return getattr(self._hd["cpd_cell"], "batch", None)
        @property
        def gnn_edge_counts(self): return self._hd["cpd_cell"].gnn_edge_counts
        @property
        def gnn_edge_indices(self): return self._hd["cpd_cell"].gnn_edge_indices
        @property
        def gnn_edge_sort_idx(self): return self._hd["cpd_cell"].gnn_edge_sort_idx
        @property
        def phi_GT_cpd_cell(self):
            val = getattr(self._hd["cpd_cell"], "phi_GT_cpd_cell", None)
            if val is None:
                raise AttributeError("phi_GT_cpd_cell not set")
            return val

    class _CellXView:
        def __init__(self, hd):
            self._hd = hd

        @property
        def neighbor_cpd_cell_x(self): return self._hd["cpd_cell", "wlsq_neighbors", "cpd_cell"].neighbor_cpd_cell_x
        @property
        def A_cell_to_cell(self): return self._hd["cpd_cell"].A_cell_to_cell
        @property
        def single_B_cell_to_cell(self): return self._hd["cpd_cell"].single_B_cell_to_cell
        @property
        def batch(self): return getattr(self._hd["cpd_cell"], "batch", None)

    class _IndexView:
        def __init__(self, hd):
            self._hd = hd

        @property
        def theta_PDE(self): return self._hd["graph"].theta_PDE
        @property
        def time_step(self): return self._hd["graph"].time_step
        @property
        def dt(self): return self._hd["graph"].dt
        @property
        def case_name(self): return self._hd["graph"].case_name
        @property
        def pRefCell_valid_mask(self): return self._hd["graph"].pRefCell_valid_mask
        @property
        def pRefCell(self): return self._hd["graph"].pRefCell
        @property
        def pRefValue(self): return self._hd["graph"].pRefValue
        @property
        def graph_index(self): return self._hd["graph"].graph_index
        @property
        def batch(self): return getattr(self._hd["graph"], "batch", None)

    def to(self, *args, exclude_keys=None, **kwargs):
        self._hetero_data = self._hetero_data.to(*args, exclude_keys=exclude_keys, **kwargs)
        self._create_views()
        return self

    def cuda(self, device=None, exclude_keys=None):
        return self.to(device or "cuda", exclude_keys=exclude_keys)
