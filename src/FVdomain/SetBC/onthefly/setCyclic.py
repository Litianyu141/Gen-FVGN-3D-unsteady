"""Periodic (cyclic) patch pairs."""

import logging

import torch

logger = logging.getLogger(__name__)


def _as_int(value) -> int:
    return int(value.view(-1)[0].item()) if torch.is_tensor(value) else int(value)


def get_cyclic_patch_pair_indices(fvpatch_like, neighbor_cpd_cell: torch.Tensor, patch_idx: int):
    """``(owner, ghost)`` compound cells of a cyclic patch's faces and of its partner's."""
    source = _as_int(patch_idx)
    target = _as_int(fvpatch_like.neighbor_patch_idx[source])
    pair_index = neighbor_cpd_cell.long()
    source_owner, source_boundary = pair_index[
        :, _as_int(fvpatch_like.start_idx_face[source]):_as_int(fvpatch_like.end_idx_face[source])]
    target_owner, target_boundary = pair_index[
        :, _as_int(fvpatch_like.start_idx_face[target]):_as_int(fvpatch_like.end_idx_face[target])]
    return source_owner, source_boundary, target_owner, target_boundary


def apply_cyclic_face_values(field, fvpatch_like, neighbor_cpd_cell, patch_idx, *, gradient=False):
    """Copy the partner patch's owner values (``[..., N, C]``, or gradients ``[..., N, C, 3]``)
    into this patch's ghost cells."""
    if field is None:
        return None
    _, source_boundary, target_owner, _ = get_cyclic_patch_pair_indices(fvpatch_like, neighbor_cpd_cell, patch_idx)
    if gradient:
        field[..., source_boundary, :, :] = field[..., target_owner, :, :]
    else:
        field[..., source_boundary, :] = field[..., target_owner, :]
    return field


def prepare_cyclic_boundary_geometry(mesh):
    """Place every cyclic ghost cell at the mirror image of its owner about the face centre
    (``2 x_f - x_P``)."""
    patch_dict = mesh["patch_dict"]
    cpd_cell_pos = mesh["cpd|cell_pos"].clone()
    neighbor_cpd_cell = mesh["cpd|neighbor_cell"].long()
    face_pos = mesh["face|face_pos"]

    processed_pairs = set()
    total_cyclic_faces = 0
    for patch_name, patch_info in patch_dict.items():
        is_cyclic = ("cyclic" in str(patch_info.get("type", "")).lower()
                     or "cyclic" in str(patch_info.get("geom_type", "")).lower())
        if not is_cyclic:
            continue
        neighbour_patch_name = patch_info["neighbour_patch"]
        pair_key = tuple(sorted([patch_name, neighbour_patch_name]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        for info in (patch_info, patch_dict[neighbour_patch_name]):
            start, end = int(info["start_idx_face"]), int(info["end_idx_face"])
            cpd_cell_pos[neighbor_cpd_cell[1, start:end]] = (
                2 * face_pos[start:end] - cpd_cell_pos[neighbor_cpd_cell[0, start:end]])
            total_cyclic_faces += (end - start) if info is patch_info else 0

    if total_cyclic_faces > 0:
        mesh["cpd|cell_pos"] = cpd_cell_pos
        logger.info(f"Prepared extrapolated cyclic ghost geometry for {total_cyclic_faces} face pairs")
    return mesh
