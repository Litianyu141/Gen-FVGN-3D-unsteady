#!/usr/bin/env python3
"""Convert OpenFOAM results into the label store the networks read: one float32 ``.mmap``
``[time step, compound cell, (u, v, w, p_rgh, T)]`` per case, ``labels_manifest.json`` and, when
the training split is converted, the z-score ``statistics.json`` of the training cases.

    python src/entries/prepare_labels.py --runs experiments/Cylinder/openfoam/runs \\
        --h5 experiments/Cylinder/PCNO/hybrid-fvm/h5 --out experiments/Cylinder/NeuralOperator/labels \\
        --splits train val test --stats interior

The first ``N_cells`` rows hold the cells; the boundary faces follow, in the patch ranges of
``<h5>/patchinfo.toml``, with the patch face values.  A case contributes the first ``n_timesteps``
(``<h5>/meta.json``) written times after t = 0; parallel runs must be reconstructed first.
``--cases`` converts named cases instead of the ``meta.json`` splits.  ``--stats`` takes the mean
and std over the cells only (``interior``, cylinder) or over all rows (``all``, data center).
"""
import argparse
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv

CHANNELS = ("u", "v", "w", "p_rgh", "T")


def reader_of(case_dir):
    """Reader of a serial or reconstructed case.  Decomposed results are refused: read per
    processor, the cells come in processor order, not in the mesh numbering of the h5 mesh."""
    foam = case_dir / "case.foam"
    foam.touch()
    reconstructed = any(d.is_dir() and d.name.replace(".", "", 1).isdigit() and float(d.name) > 0
                        for d in case_dir.iterdir())
    if not reconstructed:
        raise RuntimeError(f"{case_dir}: no reconstructed time directories (run reconstructPar)")
    return pv.OpenFOAMReader(str(foam))


def fields(block):
    U = np.asarray(block.cell_data["U"], dtype=np.float32)
    return [U[:, 0], U[:, 1], U[:, 2], np.asarray(block.cell_data["p_rgh"], dtype=np.float32),
            np.asarray(block.cell_data["T"], dtype=np.float32)]


def convert_case(job):
    case_dir, mmap_path, patches, n_cells, n_cpd, n_steps = job
    reader = reader_of(case_dir)
    times = [t for t in sorted(reader.time_values) if t > 1e-9]
    if len(times) < n_steps:
        raise RuntimeError(f"{case_dir}: {len(times)} time steps written, {n_steps} needed")
    out = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(n_steps, n_cpd, len(CHANNELS)))
    for k, t in enumerate(times[:n_steps]):
        reader.set_active_time_value(t)
        mesh = reader.read()
        for c, values in enumerate(fields(mesh["internalMesh"])):
            out[k, :n_cells, c] = values
        boundary = mesh["boundary"]
        for name, p in patches.items():
            if name not in boundary.keys():
                continue
            s, e = p["start_idx_cpd_cell"], p["end_idx_cpd_cell"]
            n = min(boundary[name].n_cells, e - s)
            for c, values in enumerate(fields(boundary[name])):
                out[k, s:s + n, c] = values[:n]
    out.flush()
    return case_dir.name


def statistics(paths, shape, rows):
    """Mean and std per channel over all time steps of ``paths`` and their first ``rows`` rows."""
    total, total_sq, count = np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS)), 0
    for path in paths:
        data = np.memmap(path, dtype="float32", mode="r", shape=shape)
        for t0 in range(0, shape[0], 100):
            chunk = data[t0:t0 + 100, :rows].astype(np.float64)
            total += chunk.sum(axis=(0, 1))
            total_sq += (chunk * chunk).sum(axis=(0, 1))
            count += chunk.shape[0] * chunk.shape[1]
    mean = total / count
    return mean, np.sqrt(total_sq / count - mean ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, type=Path, help="directory of the OpenFOAM case directories")
    ap.add_argument("--h5", required=True, type=Path, help="case directory with patchinfo.toml and meta.json")
    ap.add_argument("--out", required=True, type=Path, help="label store to write")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cases", nargs="+", help="case names, in place of --splits")
    ap.add_argument("--stats", choices=["interior", "all"], default="interior")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    meta = json.load(open(a.h5 / "meta.json"))
    with open(a.h5 / "patchinfo.toml", "rb") as f:
        patchinfo = tomllib.load(f)
    patches = patchinfo["patch"]
    n_cells = patchinfo["metadata"]["num_cells"]
    n_cpd = max([n_cells] + [p["end_idx_cpd_cell"] for p in patches.values()])
    n_steps = meta["n_timesteps"]
    names = a.cases or [n for s in a.splits for n in meta["split"].get(s, [])]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.runs / n, a.out / f"case_{i:03d}.mmap", patches, n_cells, n_cpd, n_steps) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, name in enumerate(pool.map(convert_case, jobs), 1):
            print(f"[{k}/{len(jobs)}] {name}", flush=True)

    shape = [n_steps, n_cpd, len(CHANNELS)]
    manifest = {
        "benchmark": meta.get("benchmark"), "n_cases": len(names), "n_timesteps_per_case": n_steps,
        "dt": meta["dt"], "num_cpd_cells": n_cpd, "num_interior_cells": n_cells, "num_channels": len(CHANNELS),
        "channel_order": list(CHANNELS), "dtype": "float32",
        "cases": [{"case_id": i, "case_name": n, "mmap_file": f"case_{i:03d}.mmap", "shape": shape}
                  for i, n in enumerate(names)],
    }
    with open(a.out / "labels_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    train = meta["split"].get("train", [])
    if not a.cases and "train" in a.splits and train:
        mean, std = statistics([a.out / f"case_{names.index(n):03d}.mmap" for n in train], shape,
                               n_cells if a.stats == "interior" else n_cpd)
        with open(a.out / "statistics.json", "w") as f:
            json.dump({"channel_order": list(CHANNELS), "mean": mean.tolist(), "std": std.tolist(),
                       "n_train_cases": len(train), "rows": a.stats}, f, indent=2)
    print(f"label store written to {a.out}")


if __name__ == "__main__":
    main()
