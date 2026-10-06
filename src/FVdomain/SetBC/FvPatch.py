"""Boundary patches of a case and of a batch of cases."""

from typing import Any, Dict, List

import torch


class fvPatch:
    """Face and compound-cell index ranges of every boundary patch of one case.  A cyclic
    patch stores the index of its partner patch, every other patch -1."""

    def __init__(self):
        self.patch_idx = []
        self.start_idx_face = []
        self.end_idx_face = []
        self.start_idx_cpd_cell = []
        self.end_idx_cpd_cell = []
        self.neighbor_patch_idx = []
        self.num_patches = 0
        self.patch_dict = None

    def register(self, patch_dict: Dict[str, Dict[str, Any]]):
        self.patch_dict = patch_dict
        partners = []
        for patch_idx, (patch_name, patch_info) in enumerate(patch_dict.items()):
            patch_info["patch_idx"] = patch_idx
            self.patch_idx.append(torch.tensor([patch_idx]))
            self.start_idx_face.append(torch.tensor(patch_info["start_idx_face"]))
            self.end_idx_face.append(torch.tensor(patch_info["end_idx_face"]))
            self.start_idx_cpd_cell.append(torch.tensor(patch_info["start_idx_cpd_cell"]))
            self.end_idx_cpd_cell.append(torch.tensor(patch_info["end_idx_cpd_cell"]))
            if patch_info.get("neighbour_patch") is not None:
                partners.append((patch_idx, patch_info["neighbour_patch"]))
            self.neighbor_patch_idx.append(torch.tensor([-1]))
            self.num_patches += 1
        for patch_idx, partner_name in partners:
            self.neighbor_patch_idx[patch_idx] = torch.tensor([patch_dict[partner_name]["patch_idx"]])

        self.patch_idx = torch.stack(self.patch_idx).long()
        self.start_idx_face = torch.stack(self.start_idx_face).long()
        self.end_idx_face = torch.stack(self.end_idx_face).long()
        self.start_idx_cpd_cell = torch.stack(self.start_idx_cpd_cell).long()
        self.end_idx_cpd_cell = torch.stack(self.end_idx_cpd_cell).long()
        self.neighbor_patch_idx = torch.stack(self.neighbor_patch_idx).long()
        return self


class fvPatchBatch:
    """The patches of a batch of cases, with face, cell and patch indices offset into the
    batched graph."""

    def __init__(self):
        self.patch_idx = []
        self.start_idx_face = []
        self.end_idx_face = []
        self.start_idx_cpd_cell = []
        self.end_idx_cpd_cell = []
        self.neighbor_patch_idx = []
        self.num_cases = 0

    @classmethod
    def from_fvpatch_list(cls, fvpatch_list: List[fvPatch], graph_data_list: List) -> "fvPatchBatch":
        batch = cls()
        last_num_faces = last_num_cpd_cells = last_num_patches = 0
        for fvpatch, graph_data_i in zip(fvpatch_list, graph_data_list):
            batch.patch_idx.append(fvpatch.patch_idx + last_num_patches)
            batch.start_idx_face.append(fvpatch.start_idx_face + last_num_faces)
            batch.end_idx_face.append(fvpatch.end_idx_face + last_num_faces)
            batch.start_idx_cpd_cell.append(fvpatch.start_idx_cpd_cell + last_num_cpd_cells)
            batch.end_idx_cpd_cell.append(fvpatch.end_idx_cpd_cell + last_num_cpd_cells)
            partner = fvpatch.neighbor_patch_idx.clone()
            partner[partner >= 0] += last_num_patches
            batch.neighbor_patch_idx.append(partner)
            last_num_faces += graph_data_i._num_faces
            last_num_cpd_cells += graph_data_i._num_cpd_cells
            last_num_patches += fvpatch.num_patches

        batch.patch_idx = torch.cat(batch.patch_idx)
        batch.start_idx_face = torch.cat(batch.start_idx_face)
        batch.end_idx_face = torch.cat(batch.end_idx_face)
        batch.start_idx_cpd_cell = torch.cat(batch.start_idx_cpd_cell)
        batch.end_idx_cpd_cell = torch.cat(batch.end_idx_cpd_cell)
        batch.neighbor_patch_idx = torch.cat(batch.neighbor_patch_idx)
        batch.num_cases = len(fvpatch_list)
        return batch
