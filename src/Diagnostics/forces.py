"""Drag and lift coefficients of a wall patch from a cell state."""
import tomllib

import h5py
import numpy as np


def wall_geometry(h5_path, patchinfo_path, patch):
    """Everything the force integral needs from the mesh, for one wall patch."""
    with open(patchinfo_path, "rb") as f:
        p = tomllib.load(f)["patch"][patch]
    f0, nf, g0 = p["start_idx_face"], p["n_faces"], p["start_idx_cpd_cell"]
    with h5py.File(h5_path, "r") as f:
        area = np.asarray(f["face|face_area"])[f0:f0 + nf].reshape(-1)
        fpos = np.asarray(f["face|face_pos"])[f0:f0 + nf]
        owner = np.asarray(f["cpd|neighbor_cell"])[0, f0:f0 + nf]
        cpos = np.asarray(f["cpd|cell_pos"])
        nodez = np.asarray(f["node|node_pos"])[:, 2]
    to_cell = cpos[owner] - fpos
    d = np.linalg.norm(to_cell, axis=1)
    return dict(area=area, nrm=to_cell / d[:, None], d=d, owner=owner, g0=g0, nf=nf,
                span=float(nodez.max() - nodez.min()))


def cd_cl_from_phi(phi, geom, *, ip, iu, iv, mu, rho, u_ref, diameter):
    """``(Cd, Cl, Cd_pressure)`` of one state ``[N_cpd, C]``: wall pressure from the ghost
    cells, wall shear from the owner-cell tangential velocity over the wall distance."""
    nf, g0, owner, nrm, d, area = geom["nf"], geom["g0"], geom["owner"], geom["nrm"], geom["d"], geom["area"]
    p_face = phi[g0:g0 + nf, ip]
    U3 = np.zeros((nf, 3))
    U3[:, 0] = phi[owner, iu]
    U3[:, 1] = phi[owner, iv]
    U_t = U3 - (U3 * nrm).sum(1, keepdims=True) * nrm
    Fp = ((-p_face[:, None] * nrm) * area[:, None]).sum(0)
    Fv = ((mu * U_t / d[:, None]) * area[:, None]).sum(0)
    q = 0.5 * rho * u_ref ** 2 * diameter * geom["span"]
    F = Fp + Fv
    return float(F[0] / q), float(F[1] / q), float(Fp[0] / q)
