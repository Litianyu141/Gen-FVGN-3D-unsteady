#!/usr/bin/env python3
"""Autoregressive rollout of a checkpoint trained with the FVM-residual loss.

    cd experiments/<case>/PCNO/<arm>
    python ../../../../src/entries/rollout_with_fvm.py --dataset_dir h5 \\
        --checkpoint logger/<run name>/<date-time>/states/2.state --n_steps 500 --rollout_re 100

The architecture is read from ``states/commandline_args.json`` beside the checkpoint (an
architecture flag given here overrides it).  Outputs go to
``<run>/rollout_results/rollout_<k>[_Re<Re>]/``: ``step_<n>.vtu`` snapshots and their
``rollout.pvd`` time series, ``timing.json`` and, on request, ``forces.csv`` (``--forces``)
and ``errors.csv`` (``--labels_dir``).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import tomllib
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from Diagnostics.forces import cd_cl_from_phi, wall_geometry  # noqa: E402
from Diagnostics.rollout_errors import RolloutErrors  # noqa: E402
from Diagnostics.rollout_timing import StepTimer  # noqa: E402
from FVdomain.Graph_loader import HeteroGraphLoader  # noqa: E402
from FVdomain.Graph_pool import MmapDataPool  # noqa: E402
from NNmodels.importer import NNmodel  # noqa: E402
from Post_process.vtk_export import interior_grid, write_pvd  # noqa: E402
from Utils import get_param  # noqa: E402
from Utils.utilities import FaceType  # noqa: E402

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")


def cli():
    p = argparse.ArgumentParser(description="FVM-loss rollout", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_dir", required=True, help="case h5 directory")
    p.add_argument("--checkpoint", default=None, help="a .state file; default: the newest under --logger_dir")
    p.add_argument("--logger_dir", default=None, help="default: <dataset_dir>/../logger")
    p.add_argument("--n_steps", type=int, default=300)
    p.add_argument("--rollout_re", type=float, default=None,
                   help="Reynolds number imposed at the inlet; default: the case's own draw")
    p.add_argument("--save_times", default=None,
                   help="comma-separated physical times to export (the last step always is)")
    p.add_argument("--case_params", default=None,
                   help="JSON dict of boundary parameters of the case, e.g. a data-centre case's "
                        "{\"flowRate_rack\": ..., \"deltaT\": ...}")
    p.add_argument("--forces", default=None, metavar="PATCH",
                   help="wall patch: Cd and Cl of every step into forces.csv (needs --rollout_re)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--labels_dir", default=None,
                   help="label store: per-step errors against case --label_case into errors.csv")
    p.add_argument("--label_case", default=None, help="case_name in the label manifest, e.g. Re=81.0033")
    p.add_argument("--label_offsets", default=None,
                   help="JSON dict of per-channel constants added to the prediction before the "
                        "comparison, e.g. '{\"T\": 300.15}' for T solved in T - 300.15 K")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=1)
    for k in get_param.ARCH_KEYS:
        p.add_argument(f"--{k}", default=None)
    return p.parse_known_args()[0]


def newest_checkpoint(logger_dir):
    """Newest ``.state`` by modification time (the file stems cycle, ``epoch % 3``)."""
    hits = [(os.path.getmtime(os.path.join(root, f)), os.path.join(root, f))
            for root, _, files in os.walk(logger_dir) for f in files if f.endswith(".state")]
    if not hits:
        raise FileNotFoundError(f"no .state file under {logger_dir}")
    return max(hits)[1]


def main():
    a = cli()
    ckpt = a.checkpoint or newest_checkpoint(
        a.logger_dir or os.path.join(os.path.dirname(os.path.abspath(a.dataset_dir)), "logger"))

    # architecture flags given on the command line (either spelling) override the checkpoint's
    given = {t.lstrip("-").split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
    argv_arch = [f"--{k}" for k in get_param.ARCH_KEYS if k in given]
    sys.argv = [sys.argv[0]] + sum(([f"--{k}", str(getattr(a, k))] for k in get_param.ARCH_KEYS if k in given), [])
    params = get_param.params()
    get_param.apply_checkpoint_arch(params, ckpt, argv=argv_arch)
    params.dataset_dir = a.dataset_dir
    params.dataset_size = 1
    params.batch_size = 1
    params.n_epochs = a.n_steps
    params.load_date_time = None
    params.load_index = None
    params.resume_state = False

    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), "rollout_results",
        f"rollout_{os.path.splitext(os.path.basename(ckpt))[0]}" + (f"_Re{a.rollout_re:g}" if a.rollout_re else ""))
    os.makedirs(out_dir, exist_ok=True)
    state_dir = tempfile.mkdtemp(prefix="fvm_rollout_")  # pool state, removed at the end

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(a.dataset_dir, "fvconfig.toml"), "rb") as f:
        pde = tomllib.load(f)["theta_pde"]
    dt = float(pde["dt"]["val"])

    u_in = None
    overrides = None
    if a.rollout_re is not None:  # needs constant mu and rho in fvconfig.toml
        mu, rho, D = float(pde["mu"]["val"]), float(pde["rho"]["val"]), float(pde["L"])
        u_in = a.rollout_re * mu / (rho * D)
        overrides = {"inlet_velocity": u_in}
    if a.case_params:
        overrides = {**(overrides or {}), **json.loads(a.case_params)}

    t0 = time.time()
    datasets = MmapDataPool(params=params, device=device, state_save_dir=state_dir)
    params = datasets.load_mesh_to_mmap(dataset_dir=a.dataset_dir, split=params.split, loss_mode=params.loss_mode,
                                        param_overrides=overrides)
    loader = HeteroGraphLoader(base_dataset=datasets, batch_size=1, shuffle=False)
    pool_load_s = time.time() - t0
    print(f"[rollout] pool loaded in {pool_load_s:.1f}s"
          + (f"; inlet velocity {u_in:.4f} (Re {a.rollout_re:g})" if u_in else "")
          + (f"; case parameters {a.case_params}" if a.case_params else ""))

    model = NNmodel(params).to(device)
    model.eval()
    model.load_checkpoint(ckpt, device=device)
    print(f"[rollout] loaded {ckpt}")

    forces = None
    if a.forces:
        if a.rollout_re is None:
            raise SystemExit("--forces needs --rollout_re (U_ref = Re nu / D)")
        geom = wall_geometry(glob.glob(os.path.join(a.dataset_dir, "*.h5"))[0],
                             os.path.join(a.dataset_dir, "patchinfo.toml"), a.forces)
        ch = list(params.channel_order)
        forces = dict(geom=geom, ip=ch.index("p_rgh") if "p_rgh" in ch else ch.index("p"),
                      iu=ch.index("u"), iv=ch.index("v"), rows=[], path=os.path.join(out_dir, "forces.csv"))
        print(f"[rollout] forces on patch {a.forces!r}: {geom['nf']} faces, span {geom['span']:.3f}, "
              f"U_ref {u_in:.4f} -> {forces['path']}")

    labels = None
    if a.labels_dir:
        with open(os.path.join(a.labels_dir, "labels_manifest.json")) as f:
            man = json.load(f)
        label_case = next(c for c in man["cases"] if c["case_name"] == a.label_case)
        if a.n_steps > label_case["shape"][0]:
            raise SystemExit(f"--n_steps {a.n_steps} exceeds the label length {label_case['shape'][0]}")
        labels = np.memmap(os.path.join(a.labels_dir, label_case["mmap_file"]), dtype=man["dtype"], mode="r",
                           shape=tuple(label_case["shape"]))
        errors = None
        print(f"[rollout] per-step errors vs {label_case['mmap_file']} ({a.label_case})")

    save_steps = {a.n_steps}
    if a.save_times:
        for t in (float(x) for x in a.save_times.split(",")):
            s = int(round(t / dt))
            if 0 < s <= a.n_steps:
                save_steps.add(s)
    print(f"[rollout] dt={dt}, steps={a.n_steps}, export at {sorted(save_steps)} -> {out_dir}")

    with h5py.File(glob.glob(os.path.join(a.dataset_dir, "*.h5"))[0], "r") as f:
        mesh = {k: torch.from_numpy(np.asarray(f[k]))
                for k in ("node|node_pos", "pv_cells_node", "pv_cells_type", "cpd|cell_type")}
    interior = (mesh["cpd|cell_type"].long() == FaceType.NORMAL).reshape(-1).to(device)

    timer = StepTimer(device)
    exported = []
    with torch.no_grad():
        for step in range(1, a.n_steps + 1):
            # as in training: the loader reads the pool state, the forward advances it,
            # refresh_pool writes it back
            timer.start()
            fv_graph = next(iter(loader))
            if fv_graph._hetero_data["cpd_cell"].x.device.type != device.type:
                fv_graph = fv_graph.to(device, exclude_keys=["slot_idx", "graph_index"])
            _, phi = model(fv_graph=fv_graph)
            phi = phi.detach()
            datasets.refresh_pool(phi_new_cpd_cell=phi.cpu(), slot_idx=fv_graph.graph_cell.slot_idx)
            timer.stop()

            if forces is not None:
                cd, cl, cdp = cd_cl_from_phi(phi.cpu().numpy(), forces["geom"], ip=forces["ip"], iu=forces["iu"],
                                             iv=forces["iv"], mu=mu, rho=rho, u_ref=u_in, diameter=D)
                forces["rows"].append((step, step * dt, cd, cl, cdp))

            if labels is not None:
                if errors is None:
                    errors = RolloutErrors(labels, interior.nonzero().reshape(-1).to(phi.device), man["channel_order"],
                                           device=phi.device,
                                           offsets=json.loads(a.label_offsets) if a.label_offsets else None)
                errors.step(step, phi[interior], dt)

            if step % 50 == 0 or step in save_steps or step <= 3:
                q = phi[interior]
                print(f"  step {step}/{a.n_steps} t={step * dt:.3f}  "
                      f"|U|max={q[:, 0:3].abs().max():.4e} |p|max={q[:, 3:4].abs().max():.4e}")

            if step in save_steps:
                q = phi.cpu()[interior.cpu()]
                data = {"cell_Velocity": q[:, 0:3], "cell_Pressure": q[:, 3:4]}
                if q.shape[1] > 4:
                    data["cell_Temperature"] = q[:, 4:5]
                name = f"step_{step:04d}.vtu"
                interior_grid(mesh["node|node_pos"], mesh["pv_cells_node"], mesh["pv_cells_type"], data).save(
                    os.path.join(out_dir, name))
                exported.append((np.float32(step * dt), name))

    write_pvd(os.path.join(out_dir, "rollout.pvd"), exported)
    if forces is not None:
        with open(forces["path"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "t", "Cd", "Cl", "Cd_pressure"])
            w.writerows(forces["rows"])
        r = np.array([x[1:] for x in forces["rows"]])
        half = len(r) // 2
        print(f"[rollout] forces: Cd mean (2nd half) {r[half:, 1].mean():.4f}  "
              f"Cl amplitude {(r[half:, 2].max() - r[half:, 2].min()) / 2:.4f}")

    tsum = timer.write(os.path.join(out_dir, "timing.json"), checkpoint=os.path.abspath(ckpt), net=params.net,
                       n_params=sum(p.numel() for p in model.parameters()), n_cells=int(interior.sum()),
                       pool_load_s=round(pool_load_s, 3), rollout_re=a.rollout_re, label_case=a.label_case)
    print(f"[rollout] timing: {tsum['step_median_s']:.4f} s/step (median, steady), "
          f"peak alloc {tsum['peak_allocated_gib'] or 0:.2f} GiB")

    if labels is not None and errors is not None:
        errors.write(os.path.join(out_dir, "errors.csv"))
        last = errors.rows[-1]
        print("[rollout] final-step nRMSE " + " ".join(f"{c}={last[f'nrmse_{c}']:.3f}" for c in errors.ch))
    datasets.close()
    shutil.rmtree(state_dir, ignore_errors=True)
    print(f"[rollout] done -> {out_dir}")


if __name__ == "__main__":
    main()
