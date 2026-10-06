#!/usr/bin/env python3
"""Data-loss (supervised) training: the networks regressed onto CFD labels, each pool case
advanced from its initial state by the network's own predictions.

    cd experiments/<case>/NeuralOperator/<arm>
    python ../../../../src/entries/train_with_data_driven.py --dataset_dir h5 --split train \\
        --labels_dir <label store> ...

One GPU as is; several with ``torchrun --nproc_per_node=<n>`` in place of ``python``.
``--dataset_size`` is the number of pool slots over all GPUs.  The label store holds
``labels_manifest.json``, one ``.mmap`` trajectory per case and the z-score statistics
``statistics.json``.  To continue a run, pass its run directory as ``--resume_mmap`` (pool
states and case assignment) and its checkpoint in the environment variable
``RESUME_CHECKPOINT`` (weights; the optimizer state with ``--resume_optimizer true``).
"""
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "4"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from math import ceil  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from FVdomain.Graph_loader import HeteroGraphLoader  # noqa: E402
from FVdomain.Graph_pool import MmapDataPool, load_case_assignment, validate_resume_params  # noqa: E402
from NNmodels.importer import NNmodel  # noqa: E402
from Pipeline.NeuralOperator.objective import (  # noqa: E402
    case_parameters_for, create_dd_solver, load_case_parameters, load_channel_stats)
from Utils import get_param  # noqa: E402
from Utils.Logger import ddp_run_logger  # noqa: E402
from Utils.OptimizerF import get_optimizer  # noqa: E402

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")
if os.getenv("TORCH_COMPILE_DISABLE", "0") == "1":
    torch._dynamo.config.disable = True
torch.set_float32_matmul_precision("high")


def case_names_of(fv_graph, datasets):
    """Case name of every graph in the batch."""
    return [datasets.slots[int(i)].meta_objects["case_name"] for i in fv_graph.graph_cell.slot_idx]


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    params = get_param.params()
    params.loss_mode = "data_driven"
    params.max_inner_steps = 1
    total_dataset_size = params.dataset_size
    if total_dataset_size % world_size != 0:
        raise ValueError(f"--dataset_size ({total_dataset_size}) must be divisible by the number of ranks ({world_size})")
    params.dataset_size = total_dataset_size // world_size
    if rank == 0:
        print(f"{world_size} ranks, {total_dataset_size} slots ({params.dataset_size}/rank)")

    # one model initialisation, a different data stream per rank
    base_seed = 777
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    np.random.seed(base_seed + rank * 1000)
    random.seed(base_seed + rank * 1000)
    torch.cuda.set_per_process_memory_fraction(0.99, local_rank)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    logger, state_save_dir = ddp_run_logger(params, rank, world_size, device, seed=base_seed)

    datasets = MmapDataPool(params=params, device=device, state_save_dir=state_save_dir, rank=rank)
    resume_mmap_dir = params.resume_mmap
    if resume_mmap_dir:
        validate_resume_params(load_case_assignment(resume_mmap_dir), dataset_size=total_dataset_size,
                               world_size=world_size)
    params = datasets.load_mesh_to_mmap(
        dataset_dir=params.dataset_dir, split=params.split, loss_mode="data_driven",
        total_dataset_size=total_dataset_size, world_size=world_size, resume=bool(resume_mmap_dir),
        resume_from=os.path.join(resume_mmap_dir, "mmap_state") if resume_mmap_dir else None)
    loader = HeteroGraphLoader(base_dataset=datasets, batch_size=params.batch_size, shuffle=True)
    logger.info(f"Rank {rank} | {len(datasets.slots)} cases with labels (split {params.split})")
    if world_size > 1:
        dist.barrier()
    global_plot_number = torch.zeros(1, dtype=torch.long, device=device)

    model = NNmodel(params).to(device)
    fluid_model = (DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
                   if world_size > 1 else model)
    fluid_model.train()
    logger.info(f"Rank {rank} | Model params: {sum(p.numel() for p in model.parameters()):,}")

    stats_path = os.path.join(params.labels_dir, "statistics.json")
    load_channel_stats(model.data_driven_loss, stats_path,
                       std_floor=params.fvconfig.get("solver", {}).get("datadriven_std_floor"),
                       log=logger.info if rank == 0 else None)
    logger.info(f"Rank {rank} | Loaded channel stats from {stats_path}")
    dd_solver = create_dd_solver(params, model.integrator, model.data_driven_loss,
                                 log=logger.info if rank == 0 else (lambda *_a, **_k: None))
    model._dd_solver = dd_solver
    case_parameters = load_case_parameters(params.dataset_dir)

    optimizer = get_optimizer(optimizer_type=params.optimizer_type, opt_parameters=fluid_model.parameters(),
                              params=params)
    params.load_index = 0 if params.load_index is None else int(params.load_index)
    resume_ckpt = os.environ.get("RESUME_CHECKPOINT")
    if resume_ckpt and os.path.isfile(resume_ckpt):
        model.load_checkpoint(resume_ckpt, device=device, optimizer=optimizer if params.resume_optimizer else None)
        logger.info(f"Rank {rank} | Loaded model weights from {resume_ckpt}")
    if world_size > 1:
        dist.barrier()

    for epoch in range(params.n_epochs):
        fluid_model.train()
        # resets cycle through the slots of all ranks
        if epoch % ceil(params.average_sequence_length / total_dataset_size) == 0:
            datasets._set_reset_env_flag(flag=True, rst_time=ceil(total_dataset_size / params.average_sequence_length))

        epoch_start = time.time()
        epoch_loss_sum = 0.0
        batch_count = 0
        channel_loss_sums: dict = {}
        for fv_graph in loader:
            fv_graph = fv_graph.to(device, exclude_keys=["slot_idx", "graph_index"])
            if case_parameters:  # data-centre sweep: flow rates and rack heat of each case
                dd_solver._pending_case_params = [case_parameters_for(name, case_parameters)
                                                  for name in case_names_of(fv_graph, datasets)]
            optimizer.zero_grad()
            loss_total, phi_new_cpd_cell = fluid_model(fv_graph=fv_graph, loss_mode="data_driven")
            loss_total.backward()
            optimizer.step()
            epoch_loss_sum += loss_total.item()
            batch_count += 1
            with torch.no_grad():
                for name, val in model.data_driven_loss.get_separated_losses().items():
                    channel_loss_sums[name] = channel_loss_sums.get(name, 0.0) + val
            # the predicted state is the next input of this case (autoregressive pool)
            datasets.refresh_pool(phi_new_cpd_cell=phi_new_cpd_cell.detach().cpu(),
                                  slot_idx=fv_graph.graph_cell.slot_idx)

        if world_size > 1:
            dist.barrier()
        if datasets.reset_env_flag:
            for _ in range(datasets.rst_time):
                if world_size > 1:
                    dist.broadcast(global_plot_number, src=0)
                if rank == global_plot_number.item() % world_size:
                    datasets.reset_env(plot=False)
                if rank == 0:
                    global_plot_number += 1
            datasets.reset_env_flag = False
        if world_size > 1:
            dist.barrier()

        if world_size > 1:
            loss_tensor = torch.tensor([epoch_loss_sum, float(batch_count)], device=device)
            dist.all_reduce(loss_tensor)
            epoch_loss_avg = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)
            total_batches = int(loss_tensor[1].item())
        else:
            epoch_loss_avg = epoch_loss_sum / max(batch_count, 1)
            total_batches = batch_count
        if channel_loss_sums and world_size > 1:
            ch_keys = sorted(channel_loss_sums.keys())
            ch_tensor = torch.tensor([channel_loss_sums[k] for k in ch_keys] + [float(batch_count)], device=device)
            dist.all_reduce(ch_tensor)
            agg_batches = max(ch_tensor[-1].item(), 1)
            channel_loss_sums = {k: ch_tensor[i].item() / agg_batches for i, k in enumerate(ch_keys)}
        else:
            channel_loss_sums = {k: v / max(batch_count, 1) for k, v in channel_loss_sums.items()}

        logger.info(f"Rank {rank} | Epoch {epoch} | Time: {time.time() - epoch_start:.2f}s | "
                    f"Loss: {epoch_loss_avg:.6f} | Batches: {total_batches}")
        if channel_loss_sums:
            logger.info(f"Rank {rank} | Epoch {epoch} | Per-channel L2: "
                        + " ".join(f"{k}={v:.4f}" for k, v in channel_loss_sums.items()))
            if rank == 0:
                logger.log_residuals(epoch=epoch + params.load_index, loss_total=epoch_loss_avg, **channel_loss_sums)

        if (epoch % 10 == 0) or (epoch == params.n_epochs - 1):
            if rank == 0:
                logger.save_state(model=model, optimizer=optimizer, scheduler=None, index=str(epoch % 3))
            datasets.save_snapshot(checkpoint_idx=epoch % 3, epoch=epoch)
            if world_size > 1:
                dist.barrier()
        if world_size > 1:
            dist.barrier()

    datasets.close()
    logger.finalize_residuals()
    logger.info(f"Rank {rank} | Data-driven training done")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
