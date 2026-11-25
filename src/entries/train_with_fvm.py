#!/usr/bin/env python3
"""Physics-Constrained Neural Operator (PCNO) FVM-residual training (single-GPU & DDP)."""
import contextlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
import warnings
from math import ceil

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "4"

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if os.getenv("TORCH_COMPILE_DISABLE", "0") == "1":
    torch._dynamo.config.disable = True
torch.set_float32_matmul_precision(os.environ.get("FP32_MATMUL_PRECISION", "high"))

from FVdomain.Graph_loader import HeteroGraphLoader
from FVdomain.Graph_pool import MmapDataPool, reset_owner_rank
from NNmodels.importer import NNmodel
from Utils import get_param
from Utils.Logger import ddp_run_logger
from Utils.OptimizerF import get_optimizer
from Utils.distributed import cleanup_distributed, setup_distributed
from Utils.get_param import get_hyperparam


def main():
    rank, local_rank, world_size = setup_distributed()
    params = get_param.params()
    if world_size == 1 and torch.cuda.is_available():
        gpu_idx = int(getattr(params, "on_gpu", 0))
        if 0 <= gpu_idx < torch.cuda.device_count():
            local_rank = gpu_idx
            torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    total_dataset_size = params.dataset_size
    if total_dataset_size % world_size != 0:
        raise ValueError(
            f"--dataset_size ({total_dataset_size}) must be divisible by "
            f"world_size ({world_size}): it is a global slot count."
        )
    params.dataset_size_global = total_dataset_size
    params.dataset_size = total_dataset_size // world_size

    if rank == 0:
        print(
            f"{world_size} ranks, {total_dataset_size} slots GLOBAL ({params.dataset_size}/rank), "
            f"effective batch = {world_size * params.batch_size * max(1, int(params.grad_accum_steps))}"
        )

    base_seed = 777
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    np.random.seed(base_seed + rank * 1000)
    random.seed(base_seed + rank * 1000)
    torch.cuda.set_per_process_memory_fraction(0.99, local_rank)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    logger, state_save_dir = ddp_run_logger(params, rank, world_size, device, seed=base_seed, head="logger")
    logger.info(
        "Rank %d/%d | %d GPUs, %d cases global, %d/rank",
        rank,
        world_size,
        world_size,
        total_dataset_size,
        params.dataset_size,
    )

    datasets = MmapDataPool(
        params=params,
        device=device,
        state_save_dir=state_save_dir,
        rank=rank,
        ddp_mode=True,
    )

    resume_from = None
    if params.resume_state and params.load_date_time:
        resume_from = os.path.join(
            "logger", get_hyperparam(params), params.load_date_time, "mmap_state"
        )

    params = datasets.load_mesh_to_mmap(
        dataset_dir=params.dataset_dir,
        resume=params.resume_state,
        resume_from=resume_from,
        split=params.split,
        loss_mode=params.loss_mode,
        total_dataset_size=total_dataset_size,
        world_size=world_size,
    )

    if getattr(datasets, "gpu_pool_enabled", False):
        loader = datasets.make_gpu_loader(batch_size=params.batch_size, shuffle=True)
    else:
        loader = HeteroGraphLoader(
            base_dataset=datasets,
            batch_size=params.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
            device=device,
        )

    if world_size > 1:
        dist.barrier()

    logger.info(
        "Rank %d | Loaded %d cases (gpu_pool=%s)",
        rank,
        len(datasets.meta_pool),
        getattr(datasets, "gpu_pool_enabled", False),
    )

    global_plot_number = 0

    model = NNmodel(params).to(device)
    datasets._solver_ref = model.integrator._solver

    ckpt_path = None
    if params.load_date_time is not None:
        ckpt_index = 0 if params.load_index is None else int(params.load_index)
        ckpt_path = os.path.join(
            "logger",
            get_hyperparam(params),
            params.load_date_time,
            "states",
            f"{ckpt_index}.state",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Rank {rank}: resume checkpoint not found: {ckpt_path} (cwd={os.getcwd()})"
            )
        model.load_checkpoint(ckpt_path, optimizer=None, scheduler=None, device=device)
        logger.info("Rank %d | Loaded model weights from %s", rank, ckpt_path)

    fluid_model = (
        DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if world_size > 1
        else model
    )
    fluid_model.train()

    optimizer = get_optimizer(
        optimizer_type=params.optimizer_type,
        opt_parameters=fluid_model.parameters(),
        params=params,
    )

    if ckpt_path is not None and (params.resume_optimizer or params.resume_state):
        ckpt_dicts = torch.load(ckpt_path, map_location=device)
        if "optimizer0" in ckpt_dicts:
            optimizer.load_state_dict(ckpt_dicts["optimizer0"])
            logger.info("Rank %d | Restored optimizer state from %s", rank, ckpt_path)
        elif "optimizer_state_dict" in ckpt_dicts:
            optimizer.load_state_dict(ckpt_dicts["optimizer_state_dict"])
            logger.info("Rank %d | Restored optimizer state from %s", rank, ckpt_path)
        del ckpt_dicts

    params.load_index = 0 if params.load_index is None else int(params.load_index)

    graceful_stop = {"triggered": False}

    def _handle_early_stop(signum, _frame):
        graceful_stop["triggered"] = True
        if rank == 0:
            logger.info("Rank 0 | Received signal %d; will checkpoint cleanly at end of current epoch.", signum)

    signal.signal(signal.SIGUSR1, _handle_early_stop)

    lr_sched = str(getattr(params, "lr_schedule", "constant")).lower()
    base_lr = float(params.lr)
    min_lr = float(getattr(params, "min_lr", 5e-6))
    warmup_ep = max(0, int(getattr(params, "warmup_epochs", 20)))
    start_epoch = max(0, int(getattr(params, "start_epoch", 0)))
    run_date_time = os.path.basename(os.path.normpath(state_save_dir))
    resume_json_path = os.environ.get("RUN_RESUME_JSON", "").strip()

    for local_epoch in range(max(0, params.n_epochs - start_epoch)):
        epoch = start_epoch + local_epoch
        if lr_sched == "cosine":
            if warmup_ep > 0 and epoch < warmup_ep:
                cur_lr = base_lr * (0.2 + 0.8 * (epoch + 1) / warmup_ep)
            else:
                denom = max(1, params.n_epochs - 1 - warmup_ep)
                prog = min(1.0, max(0.0, (epoch - warmup_ep) / denom))
                cur_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

        fluid_model.train()
        refresh_pool = False

        if (
            epoch % ceil(params.average_sequence_length / total_dataset_size) == 0
            and not (local_epoch == 0 and params.resume_state)
        ):
            rst_time = ceil(total_dataset_size / params.average_sequence_length)
            datasets._set_reset_env_flag(flag=True, rst_time=rst_time)

        epoch_start = time.time()
        epoch_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
        batch_count = 0
        case_count = 0
        epoch_residuals_gpu = {}
        accum = max(1, int(getattr(params, "grad_accum_steps", 1)))

        for i_iter in range(params.max_inner_steps):
            if i_iter == params.max_inner_steps - 1:
                refresh_pool = True

            num_batches_in_iter = len(loader)
            for batch_index, fv_graph in enumerate(loader):
                if fv_graph._hetero_data["cpd_cell"].x.device.type != device.type:
                    fv_graph = fv_graph.to(
                        device, exclude_keys=["slot_idx", "global_idx", "graph_index"]
                    )

                win_start = (batch_index // accum) * accum
                win_end = min(win_start + accum, num_batches_in_iter)
                actual_accum_steps = max(1, win_end - win_start)
                is_window_start = batch_index == win_start
                is_window_end = (batch_index + 1) == win_end

                if is_window_start:
                    optimizer.zero_grad()

                sync_ctx = (
                    fluid_model.no_sync()
                    if (not is_window_end and hasattr(fluid_model, "no_sync"))
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    loss_total, phi_new_cpd_cell = fluid_model(
                        fv_graph=fv_graph,
                        is_training=True,
                    )
                    (loss_total / actual_accum_steps).backward()

                if is_window_end:
                    optimizer.step()
                    optimizer.zero_grad()

                cur_bs = int(fv_graph.graph_Index.theta_PDE.shape[0])
                with torch.no_grad():
                    epoch_loss_gpu.add_(loss_total.detach().to(torch.float64), alpha=cur_bs)
                    del loss_total
                    batch_count += 1
                    case_count += cur_bs

                    eps = 1e-30
                    for name, tensor in model.integrator.iter_named_residuals():
                        if tensor.dim() >= 2 and tensor.shape[-1] > 1:
                            for i in range(tensor.shape[-1]):
                                key = f"{name}_{i}"
                                val = torch.log10(tensor[..., i : i + 1] + eps).mean().to(torch.float64)
                                if key not in epoch_residuals_gpu:
                                    epoch_residuals_gpu[key] = val * cur_bs
                                else:
                                    epoch_residuals_gpu[key].add_(val, alpha=cur_bs)
                        else:
                            val = torch.log10(tensor + eps).mean().to(torch.float64)
                            if name not in epoch_residuals_gpu:
                                epoch_residuals_gpu[name] = val * cur_bs
                            else:
                                epoch_residuals_gpu[name].add_(val, alpha=cur_bs)

                if refresh_pool:
                    phi_to_store = (
                        phi_new_cpd_cell.detach()
                        if getattr(datasets, "gpu_pool_enabled", False)
                        else phi_new_cpd_cell.detach().cpu()
                    )
                    phi_to_store = datasets.mirror_project(
                        datasets.gauge_project(
                            phi_to_store,
                            model.integrator.mask_interior_cell,
                            getattr(model.integrator._solver, "residual_momentum", None),
                            getattr(model.integrator._solver, "cell_volume", None),
                            model.integrator.cpd_cell_pos,
                            getattr(model.integrator._solver, "dt_cell", None),
                            batch_cpd_cell=fv_graph.graph_cell.batch,
                        ),
                        model.integrator.cpd_cell_pos,
                        batch_cpd_cell=fv_graph.graph_cell.batch,
                    )
                    datasets.refresh_pool(
                        phi_new_cpd_cell=phi_to_store,
                        slot_idx=fv_graph.graph_cell.slot_idx,
                    )

        gpu_pool_fast = getattr(datasets, "gpu_pool_enabled", False)
        if world_size > 1 and not gpu_pool_fast:
            dist.barrier()

        if datasets.reset_env_flag:
            for _ in range(datasets.rst_time):
                current_plot = global_plot_number
                target_rank = reset_owner_rank(current_plot, total_dataset_size, world_size)
                if rank == target_rank:
                    should_plot = datasets._plot_env and os.environ.get("PLOT_ENV", "1") != "0"
                    if should_plot:
                        datasets.plot_count = current_plot
                    datasets.reset_env(plot=should_plot)
                global_plot_number += 1
            datasets.reset_env_flag = False
            datasets._plot_env = True

        if world_size > 1 and not gpu_pool_fast:
            dist.barrier()

        res_keys = list(epoch_residuals_gpu.keys())
        metrics_vec = torch.stack(
            [
                epoch_loss_gpu,
                torch.tensor(float(case_count), device=device, dtype=torch.float64),
                torch.tensor(float(batch_count), device=device, dtype=torch.float64),
            ]
            + [epoch_residuals_gpu[k] for k in res_keys]
        )
        if world_size > 1:
            dist.all_reduce(metrics_vec)
        metrics_host = metrics_vec.cpu().tolist()
        total_cases = metrics_host[1]
        epoch_loss_avg = metrics_host[0] / total_cases if total_cases > 0 else 0.0
        total_batches = int(metrics_host[2])
        epoch_residuals = {k: metrics_host[3 + i] for i, k in enumerate(res_keys)}

        epoch_time = time.time() - epoch_start
        logger.info(
            "Rank %d | Epoch %d | Time: %.2fs | Loss: %.6f | Batches: %d",
            rank,
            epoch,
            epoch_time,
            epoch_loss_avg,
            total_batches,
        )
        if total_cases > 0 and epoch_residuals:
            res_parts = [f"{k}={v / total_cases:.4f}" for k, v in epoch_residuals.items()]
            logger.info("Rank %d | Epoch %d | Residuals (log10): %s", rank, epoch, " ".join(res_parts))
            if rank == 0:
                logger.log_residuals(
                    epoch=epoch + params.load_index,
                    loss_total=epoch_loss_avg,
                    **{k: v / total_cases for k, v in epoch_residuals.items()},
                )

        stop_flag = os.environ.get("STOP_FLAG_FILE", "").strip()
        stop_now = bool(graceful_stop["triggered"]) or (
            rank == 0 and bool(stop_flag) and os.path.exists(stop_flag)
        )
        if world_size > 1:
            stop_t = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
            dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
            stop_now = bool(int(stop_t.item()) > 0)

        ckpt_every = int(os.environ.get("CKPT_EVERY_EPOCHS", getattr(params, "ckpt_every_epochs", 10)))
        keep_milestone_every = int(os.environ.get("KEEP_MILESTONE_EVERY", getattr(params, "keep_milestone_every", 0)))
        snapshot_every = int(os.environ.get("SNAPSHOT_EVERY_EPOCHS", getattr(params, "snapshot_every_epochs", 50)))
        is_last_epoch = (epoch == params.n_epochs - 1) or stop_now
        save_rolling = is_last_epoch or (ckpt_every > 0 and epoch % ckpt_every == 0)
        save_milestone = keep_milestone_every > 0 and (
            (epoch > 0 and epoch % keep_milestone_every == 0) or is_last_epoch
        )
        save_snap = is_last_epoch or (snapshot_every > 0 and epoch > 0 and epoch % snapshot_every == 0)

        if save_rolling or save_milestone or save_snap:
            rolling_slot = (epoch // max(ckpt_every, 1)) % 3
            if rank == 0 and (save_rolling or save_milestone):
                model_to_save = fluid_model.module if world_size > 1 else fluid_model
                saved_state_path = None
                if save_rolling:
                    saved_state_path = logger.save_state(
                        model=model_to_save,
                        optimizer=optimizer,
                        scheduler=None,
                        index=str(rolling_slot),
                    )
                if save_milestone:
                    milestone_ep = epoch if start_epoch > 0 else (epoch + params.load_index)
                    milestone_idx = f"epoch_{milestone_ep:04d}"
                    if saved_state_path is not None:
                        milestone_path = os.path.join(
                            os.path.dirname(saved_state_path), f"{milestone_idx}.state"
                        )
                        shutil.copy2(saved_state_path, milestone_path)
                        logger.info("Rank %d | Milestone state saved to %s", rank, milestone_path)
                    else:
                        logger.save_state(
                            model=model_to_save,
                            optimizer=optimizer,
                            scheduler=None,
                            index=milestone_idx,
                        )
            if save_snap:
                snap_slot = (epoch // max(snapshot_every, 1)) % 3
                datasets.save_snapshot(checkpoint_idx=snap_slot, epoch=epoch)
                if is_last_epoch:
                    datasets.close()

            if world_size > 1:
                dist.barrier()

            if rank == 0 and save_rolling and save_snap and resume_json_path:
                os.makedirs(os.path.dirname(os.path.abspath(resume_json_path)), exist_ok=True)
                tmp_rj = f"{resume_json_path}.tmp.{os.getpid()}"
                with open(tmp_rj, "w") as rf:
                    json.dump(
                        {
                            "load_date_time": run_date_time,
                            "load_index": int(rolling_slot),
                            "start_epoch": int(epoch + 1),
                            "state_save_dir": state_save_dir,
                        },
                        rf,
                        indent=2,
                    )
                os.replace(tmp_rj, resume_json_path)

            if stop_now:
                logger.info("Rank %d | Exiting cleanly after saving epoch %d.", rank, epoch)
                break

        elif world_size > 1 and not gpu_pool_fast:
            dist.barrier()

    datasets.close()
    logger.finalize_residuals()
    logger.info("Rank %d | Training complete.", rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
