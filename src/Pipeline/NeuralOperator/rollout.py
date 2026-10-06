"""Autoregressive rollout of a data-loss checkpoint on one case, with errors against the labels."""
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Diagnostics.rollout_timing import StepTimer  # noqa: E402
from FVdomain.Graph_dataset import HeteroFVGraph, HeteroGraphDataset  # noqa: E402
from FVdomain.Graph_loader import bcpatch_collate_fn, fvpatch_collate_fn, graph_collate_fn  # noqa: E402
from FVdomain.Graph_pool import MmapDataPool, _build_merged_mesh  # noqa: E402
from NNmodels.importer import NNmodel  # noqa: E402
from Pipeline.NeuralOperator.objective import (  # noqa: E402
    case_parameters_for, create_dd_solver, load_case_parameters, load_channel_stats)
from Post_process.slice_render import build_pv_slice, draw_panel, zoom_bounds  # noqa: E402
from Post_process.vtk_export import export_vtm  # noqa: E402
from Utils import get_param  # noqa: E402
from Utils.utilities import FaceType  # noqa: E402


def build_fv_graph(dataset, idx):
    graph_data, fvpatch_list, bcpatch_list = dataset.get(idx)
    return HeteroFVGraph(graph_collate_fn([graph_data]), fvpatch_collate_fn([fvpatch_list], [graph_data]),
                         bcpatch_collate_fn([bcpatch_list], [fvpatch_list]))


def setup(args):
    """Load the checkpoint and the evaluation split of the case set with its labels."""
    params = get_param.params()
    params.loss_mode = "data_driven"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {args.checkpoint}")
    # the architecture sets the input width, so it is settled before the mesh is read
    get_param.apply_checkpoint_arch(params, args.checkpoint)

    state_dir = tempfile.mkdtemp(prefix="dd_rollout_")
    pool = MmapDataPool(params=params, device=device, state_save_dir=state_dir, rank=0)
    params = pool.load_mesh_to_mmap(dataset_dir=params.dataset_dir, split=args.eval_split, loss_mode="data_driven")

    model = NNmodel(params).to(device)
    model.load_checkpoint(args.checkpoint, device=device)
    model.eval()
    load_channel_stats(model.data_driven_loss,
                       args.stats_path or os.path.join(params.labels_dir, "statistics.json"))
    model._dd_solver = create_dd_solver(params, model.integrator, model.data_driven_loss)

    cell_type = pool.slots[0].get_mesh_tensor("cpd|cell_type").clone().long().squeeze()
    return dict(
        params=params, pool=pool, model=model, dataset=HeteroGraphDataset(base_dataset=pool), device=device,
        mask_int=(cell_type == FaceType.NORMAL).numpy(), run_dir=str(Path(args.checkpoint).parent.parent),
        args=args, case_parameters=load_case_parameters(params.dataset_dir), state_dir=state_dir,
    )


def run_single(ctx):
    """Roll the case ``--case-idx`` forward ``--n-steps`` steps from its initial state."""
    args, params = ctx["args"], ctx["params"]
    pool, model, dataset, device, mask_int = ctx["pool"], ctx["model"], ctx["dataset"], ctx["device"], ctx["mask_int"]
    dd_solver = model._dd_solver

    case_idx = min(args.case_idx, len(pool.slots) - 1)
    slot = pool.slots[case_idx]
    n_steps = min(args.n_steps, slot.n_timesteps)
    channel_names = params.channel_order
    case_name = slot.meta_objects.get("case_name", f"case_{case_idx}")
    dt = slot.meta_objects["fvconfig"]["theta_pde"]["dt"]["val"]
    case_params = case_parameters_for(case_name, ctx["case_parameters"])
    if case_params:
        print(f"Case: {case_name}, params: {case_params}")
    print(f"Case: {case_name}, rollout {n_steps} steps, channels: {channel_names}")

    mesh = _build_merged_mesh(slot)
    result_dir = os.path.join(ctx["run_dir"], "rollout_results", case_name)
    vtm_dir = os.path.join(result_dir, "vtm")
    os.makedirs(vtm_dir if not args.no_vtu else result_dir, exist_ok=True)
    series_entries = []

    slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))
    snapshot_steps = sorted({s for s in (args.snapshot_steps or []) if s <= n_steps} | {n_steps})
    snapshots = {}
    moment_rows = []
    timer = StepTimer(device)  # from graph assembly to pool write-back, as in the FVM rollout
    for t in range(n_steps):
        step = t + 1
        timer.start()
        fv_graph = build_fv_graph(dataset, case_idx).to(device, exclude_keys=["slot_idx", "graph_index"])
        if case_params:
            dd_solver._pending_case_params = [case_params]
        with torch.no_grad():
            _, phi_new = model(fv_graph=fv_graph, loss_mode="data_driven")
        phi_cpu = phi_new.detach().cpu()
        slot.update_phi(phi_cpu)
        phi = phi_cpu.numpy()
        slot.increment_time_step()
        timer.stop()

        label = np.asarray(slot.get_label_at(t))  # the state at t = step * dt
        if step in snapshot_steps:
            snapshots[step] = phi.copy()
        if not args.no_vtu and (not args.save_only_last or step == n_steps):
            vtm_name = f"step_{step:04d}.vtm"
            export_vtm(os.path.join(vtm_dir, vtm_name), mesh, mask_int, {
                "pred_U": phi[:, 0:3], "pred_p_rgh": phi[:, 3:4], "pred_T": phi[:, 4:5],
                "GT_U": label[:, 0:3], "GT_p_rgh": label[:, 3:4], "GT_T": label[:, 4:5]})
            series_entries.append({"name": vtm_name, "time": float(step * dt)})

        p = phi[mask_int].astype(np.float64)
        g = label[mask_int].astype(np.float64)
        d = np.abs(p - g)
        row = {"step": step}
        for i, c in enumerate(channel_names):
            row[f"mae_{c}"] = float(d[:, i].mean())
            row[f"l1_diff_{c}"] = float(d[:, i].sum())
            row[f"l2_diff_{c}"] = float(np.linalg.norm(p[:, i] - g[:, i]))
            row[f"l1_label_{c}"] = float(np.abs(g[:, i]).sum())
            row[f"l2_label_{c}"] = float(np.linalg.norm(g[:, i]))
        row["n_cells"] = int(p.shape[0])
        moment_rows.append(row)
        if step % 50 == 0:
            print(f"  Step {step}/{n_steps}: u_RMSE={np.sqrt(np.mean((p[:, 0] - g[:, 0]) ** 2)):.6f}")

    t_summary = timer.write(os.path.join(result_dir, "timing.json"), checkpoint=os.path.abspath(args.checkpoint),
                            case=case_name, n_cells=int(mask_int.sum()))
    print(f"timing: {t_summary['step_median_s']:.4f} s/step (median, steady), "
          f"peak alloc {t_summary.get('peak_allocated_gib') or float('nan'):.2f} GiB")
    with open(os.path.join(result_dir, "error_moments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(moment_rows[0]))
        w.writeheader()
        w.writerows(moment_rows)
    if series_entries:
        with open(os.path.join(vtm_dir, f"{case_name}_rollout.vtm.series"), "w") as f:
            json.dump({"file-series-version": "1.0", "files": series_entries}, f, indent=2)
    if not args.no_png:
        render_snapshots(mesh, mask_int, snapshots, slot, channel_names, case_name, dt, result_dir, args)


def render_snapshots(mesh, mask_int, snapshots, slot, channel_names, case_name, dt, result_dir, args):
    """Prediction, label and |error| of every channel on a slice, one PNG per snapshot step."""
    pv_cells_node, pv_cells_type = np.asarray(mesh["pv_cells_node"]), np.asarray(mesh["pv_cells_type"])
    node_pos = np.asarray(mesh["node|node_pos"])
    n_cells = pv_cells_type.shape[0]
    slice_axis = args.slice_axis
    if slice_axis is None:  # the direction with the fewest distinct cell-centre coordinates
        pos_int = slot.get_mesh_tensor("cpd|cell_pos").numpy()[mask_int]
        slice_axis = "xyz"[int(np.argmin([len(np.unique(np.round(pos_int[:, d], 6))) for d in range(3)]))]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})

    for k, (step, pred) in enumerate(sorted(snapshots.items())):
        label = np.asarray(slot.get_label_at(step - 1))
        fields = {}
        for i, c in enumerate(channel_names):
            fields[f"pred_{c}"] = pred[:n_cells, i]
            fields[f"gt_{c}"] = label[:n_cells, i]
        tri, sliced = build_pv_slice(node_pos, pv_cells_node, pv_cells_type, fields, slice_axis=slice_axis,
                                     slice_pos=args.slice_pos)
        if tri is None:
            print("the slice holds no cells; no PNG written")
            return
        if k == 0:
            xlo, xhi, ylo, yhi = zoom_bounds((tri.x.min(), tri.x.max(), tri.y.min(), tri.y.max()), args.zoom)

        n_cols = len(channel_names)
        fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 10), gridspec_kw={"hspace": 0.05, "wspace": 0.35})
        for col, c in enumerate(channel_names):
            pred_c, gt_c = sliced[f"pred_{c}"], sliced[f"gt_{c}"]
            err_c = np.abs(pred_c - gt_c)
            field_lim = max(float(np.percentile(np.abs(gt_c), 99.2)), 1e-8)
            err_lim = max(float(np.percentile(err_c, 99.5)), 1e-8)
            draw_panel(axes[0, col], tri, pred_c, -field_lim, field_lim, "RdBu_r")
            gm = draw_panel(axes[1, col], tri, gt_c, -field_lim, field_lim, "RdBu_r")
            em = draw_panel(axes[2, col], tri, err_c, 0, err_lim, "magma")
            axes[0, col].set_title(f"${c}$ (RMSE={np.sqrt(np.mean((pred_c - gt_c) ** 2)):.2e})", fontsize=11)
            for row in range(3):
                axes[row, col].set_xlim(xlo, xhi)
                axes[row, col].set_ylim(ylo, yhi)
            fmt = plt.FuncFormatter(lambda x, _: f"{x:.1e}")
            fig.colorbar(gm, ax=[axes[0, col], axes[1, col]], shrink=0.7, pad=0.03, aspect=25).ax.yaxis.set_major_formatter(fmt)
            fig.colorbar(em, ax=axes[2, col], shrink=0.7, pad=0.03, aspect=15).ax.yaxis.set_major_formatter(fmt)
        for row, text in enumerate(["Prediction", "Label (CFD)", "|Error|"]):
            axes[row, 0].set_ylabel(text, fontsize=11)
        fig.suptitle(f"{case_name} | Rollout Step {step} (t={step * dt:.2f}s)", fontsize=14, y=0.98)
        path = os.path.join(result_dir, f"rollout_combined_step{step}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Snapshot PNG: {path}")
