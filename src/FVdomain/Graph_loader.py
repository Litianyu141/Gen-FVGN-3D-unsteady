"""Batches of pool slots as ``HeteroFVGraph`` objects."""

import logging
import random
from typing import List

from torch_geometric.data import Batch as HeteroBatch

from FVdomain.Graph_dataset import HeteroFVGraph, HeteroGraphDataset
from FVdomain.SetBC.FvPatch import fvPatch, fvPatchBatch
from FVdomain.SetBC.onthefly.BCbase import BoundaryConditionBatch

logger = logging.getLogger(__name__)


def graph_collate_fn(batch):
    return HeteroBatch.from_data_list(batch)


def fvpatch_collate_fn(fvpatch_list: List[fvPatch], graph_data_list: List):
    return fvPatchBatch.from_fvpatch_list(fvpatch_list, graph_data_list)


def bcpatch_collate_fn(bcpatch_dict_list: List[dict], fvpatch_list: List[fvPatch]):
    return BoundaryConditionBatch.from_bcpatch_list(bcpatch_dict_list, fvpatch_list)


class HeteroGraphLoader:
    """Iterates over the pool's slots in (shuffled) batches of ``batch_size``."""

    def __init__(self, base_dataset, batch_size=1, shuffle=True):
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hetero_dataset = HeteroGraphDataset(base_dataset)
        self.dataset_size = len(self.base_dataset.meta_pool)
        self.indices = list(range(self.dataset_size))
        logger.info(f"Loader initialized: {self.dataset_size} cases, batch_size={batch_size}, shuffle={shuffle}")

    def _build_one_batch(self, batch_indices):
        graphs = [self.hetero_dataset[idx] for idx in batch_indices]
        graph_data_list = [g[0] for g in graphs]
        fvpatch_lists = [g[1] for g in graphs]
        bcpatch_lists = [g[2] for g in graphs]
        return HeteroFVGraph(
            graph_collate_fn(graph_data_list),
            fvpatch_collate_fn(fvpatch_lists, graph_data_list),
            bcpatch_collate_fn(bcpatch_lists, fvpatch_lists),
        )

    def __iter__(self):
        indices = self.indices.copy() if self.shuffle else self.indices
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            yield self._build_one_batch(indices[i:i + self.batch_size])
