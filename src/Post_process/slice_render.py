"""Planar slices of a cell-centred state for matplotlib: the cell data are interpolated to
the mesh nodes on the full 3D grid, then sliced and triangulated (Gouraud shading)."""
import numpy as np
import pyvista as pv

import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri  # noqa: E402


def build_pv_slice(node_pos, pv_cells_node, pv_cells_type, field_dict, slice_axis="z", slice_pos=None):
    """``(triangulation, {name: node values on the slice})`` of the slice normal to
    ``slice_axis`` at ``slice_pos`` (default: the grid centre); ``(None, None)`` if empty."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    normal = [0, 0, 0]
    normal[axis_idx] = 1
    other_dims = [d for d in range(3) if d != axis_idx]

    grid = pv.UnstructuredGrid(pv_cells_node, pv_cells_type, np.asarray(node_pos))
    origin = np.zeros(3)
    origin[axis_idx] = grid.center[axis_idx] if slice_pos is None else slice_pos
    for name, arr in field_dict.items():
        grid.cell_data[name] = arr
    sliced = grid.cell_data_to_point_data().slice(normal=normal, origin=origin)
    if sliced.n_cells == 0:
        return None, None
    sliced_tri = sliced.triangulate()
    pts_2d = np.asarray(sliced_tri.points)[:, other_dims]
    tri = mtri.Triangulation(pts_2d[:, 0], pts_2d[:, 1], triangles=sliced_tri.faces.reshape(-1, 4)[:, 1:4])
    return tri, {name: np.asarray(sliced_tri.point_data[name]) for name in field_dict}


def zoom_bounds(bounds, zoom_factor):
    xmin, xmax, ymin, ymax = bounds
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin) / zoom_factor
    hy = 0.5 * (ymax - ymin) / zoom_factor
    return (xc - hx, xc + hx, yc - hy, yc + hy)


def draw_panel(ax, tri, values, vmin, vmax, cmap_name):
    mesh = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh
