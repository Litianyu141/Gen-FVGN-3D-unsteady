"""VTK output of cell-centred states: the interior mesh as VTU, interior and boundary
patches as VTM, and ParaView time series (PVD)."""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
import torch

_VTK_TRIANGLE, _VTK_QUAD, _VTK_POLYGON = 5, 9, 7


def _numpy(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def _cell_array(a):
    """``[n, 1]`` is written as a scalar array."""
    a = _numpy(a)
    return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a


def interior_grid(node_pos, pv_cells_node, pv_cells_type, cell_data=None):
    grid = pv.UnstructuredGrid(_numpy(pv_cells_node), _numpy(pv_cells_type), _numpy(node_pos))
    for name, values in (cell_data or {}).items():
        grid.cell_data[name] = _cell_array(values)
    return grid


def patch_grid(mesh, start_face, end_face):
    """Surface grid of the faces ``[start_face, end_face)``, with its own node numbering."""
    face_node = torch.as_tensor(mesh["face_node"]).long().squeeze()
    face_edge_ptr = torch.as_tensor(mesh["face_edge_ptr"]).long().squeeze()
    face_indices = torch.arange(start_face, end_face)
    in_patch = torch.isin(face_edge_ptr, face_indices)
    global_to_local = torch.full((end_face,), -1, dtype=torch.long)
    global_to_local[face_indices] = torch.arange(end_face - start_face, dtype=torch.long)
    local_face = global_to_local[face_edge_ptr[in_patch]]
    unique_nodes, local_node = torch.unique(face_node[in_patch], return_inverse=True)
    points = _numpy(mesh["node|node_pos"])[unique_nodes.numpy()]
    local_node = local_node.numpy()

    cells, types, offset = [], [], 0
    for n in torch.bincount(local_face, minlength=end_face - start_face).tolist():
        if n == 0:
            continue
        cells += [n] + local_node[offset:offset + n].tolist()
        types.append({3: _VTK_TRIANGLE, 4: _VTK_QUAD}.get(n, _VTK_POLYGON))
        offset += n
    return pv.UnstructuredGrid(np.array(cells, dtype=np.int64), np.array(types, dtype=np.uint8), points)


def export_vtm(path, mesh, mask_interior, cell_fields):
    """Interior mesh (block ``internal_mesh``) and one block per boundary patch.

    ``cell_fields`` maps an array name to a state ``[N_cpd, k]``: its interior rows go to the
    interior block, a patch's ghost-cell rows (one per boundary face) to the patch block."""
    mb = pv.MultiBlock()
    mb.append(interior_grid(mesh["node|node_pos"], mesh["pv_cells_node"], mesh["pv_cells_type"],
                            {name: values[mask_interior] for name, values in cell_fields.items()}), "internal_mesh")
    fvpatch = mesh["fvpatch"]
    for patch_name, info in mesh["patch_dict"].items():
        start_face, end_face = int(info["start_idx_face"]), int(info["end_idx_face"])
        if end_face == start_face:
            continue
        start_cpd = int(fvpatch.start_idx_cpd_cell[info["patch_idx"]])
        end_cpd = int(fvpatch.end_idx_cpd_cell[info["patch_idx"]])
        grid = patch_grid(mesh, start_face, end_face)
        for name, values in cell_fields.items():
            grid.cell_data[name] = _cell_array(values[start_cpd:end_cpd])
        mb.append(grid, patch_name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mb.save(path)
    return path


def write_pvd(path, entries):
    """ParaView collection of ``(time, file name relative to the .pvd)`` entries."""
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    for t, name in entries:
        ET.SubElement(collection, "DataSet", timestep=str(t), group="", part="0", file=name)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
