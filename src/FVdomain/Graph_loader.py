"""CPU batching loader for ``HeteroFVGraph`` (with optional single-slot static-graph cache)."""

import logging
import os
import random as _random
from queue import Queue
from threading import Thread
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch as HeteroBatch, HeteroData

from FVdomain.Graph_dataset import HeteroFVGraph, HeteroGraphDataset
from FVdomain.SetBC.FvPatch import fvPatch, fvPatchBatch
from FVdomain.SetBC.onthefly.BCbase import BoundaryConditionBatch

logger = logging.getLogger(__name__)


def graph_collate_fn(batch):
    return HeteroBatch.from_data_list(batch)


def fvpatch_collate_fn(fvpatch_list: List[fvPatch], graph_data_list: List[HeteroData]):
    return fvPatchBatch.from_fvpatch_list(fvpatch_list, graph_data_list)


def bcpatch_collate_fn(bcpatch_dict_list: List[dict], fvpatch_list: List[fvPatch]):
    return BoundaryConditionBatch.from_bcpatch_list(bcpatch_dict_list, fvpatch_list)


class HeteroGraphLoader:
    """DataLoader returning ``HeteroFVGraph`` batches."""

    def __init__(
        self,
        base_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=0,
        device=None,
    ):
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.device = device if os.getenv("ENABLE_GRAPH_CACHE") == "1" else None
        self._graph_cache = {}
        self.hetero_dataset = HeteroGraphDataset(base_dataset)
        self.dataset_size = len(self.base_dataset.meta_pool)
        self.indices = list(range(self.dataset_size))

    def _build_one_batch(self, batch_indices):
        if (
            self.device is not None
            and self.batch_size == 1
            and hasattr(self.base_dataset, "slots")
        ):
            idx = batch_indices[0]
            slot = self.base_dataset.slots[idx]
            cached = self._graph_cache.get(idx)
            if cached is None or cached._cache_meta_ref is not slot.meta_objects:
                cached = self._build_one_batch_cpu(batch_indices)
                cached._cache_meta_ref = slot.meta_objects
                self._graph_cache[idx] = cached
            else:
                self._refresh_dynamic_fields(cached, idx)
            device_hd = cached._hetero_data.to(
                self.device, exclude_keys=["slot_idx", "graph_index", "global_idx"]
            )
            return HeteroFVGraph(device_hd, cached.fvpatch_batch, cached.bcpatch_batch)
        return self._build_one_batch_cpu(batch_indices)

    def _build_one_batch_cpu(self, batch_indices):
        graph_data_list, fvpatch_list, bcpatch_list = [], [], []
        for idx in batch_indices:
            gd, fvp, bcp = self.hetero_dataset[idx]
            graph_data_list.append(gd)
            fvpatch_list.append(fvp)
            bcpatch_list.append(bcp)
        return HeteroFVGraph(
            graph_collate_fn(graph_data_list),
            fvpatch_collate_fn(fvpatch_list, graph_data_list),
            bcpatch_collate_fn(bcpatch_list, fvpatch_list),
        )

    def _refresh_dynamic_fields(self, fv_graph, idx):
        slot = self.base_dataset.slots[idx]
        hd = fv_graph._hetero_data
        hd["cpd_cell"].x = slot.phi_tensor.clone()
        hd["graph"].time_step = torch.tensor(float(slot.time_step), dtype=torch.float32)
        if getattr(self.base_dataset, "loss_mode", "fvm") == "data_driven" and getattr(slot, "has_labels", False):
            t_target = min(slot.time_step, slot.n_timesteps - 1)
            phi_gt_np = np.asarray(slot.get_label_at(t_target)).copy()
            t_offset = getattr(slot, "_label_T_offset", None)
            if t_offset is not None:
                phi_gt_np[:, -1] += t_offset
            hd["cpd_cell"].phi_GT_cpd_cell = torch.from_numpy(phi_gt_np).to(torch.float32)

    def __iter__(self):
        indices = self.indices.copy() if self.shuffle else self.indices
        if self.shuffle:
            _random.shuffle(indices)
        all_batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.prefetch_factor <= 0:
            for batch_indices in all_batches:
                yield self._build_one_batch(batch_indices)
            return

        q: Queue = Queue(maxsize=self.prefetch_factor)

        def _producer():
            for batch_indices in all_batches:
                q.put(self._build_one_batch(batch_indices))
            q.put(None)

        t = Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
