#!/usr/bin/env python3
"""FVM-loss training.

    cd experiments/<case>/PCNO/<arm>
    python ../../../../src/entries/train_with_fvm.py --dataset_dir h5 --net TransFVGN_v2 ...

One GPU as is; several with ``torchrun --nproc_per_node=<n>`` in place of ``python``.
``--dataset_size`` is the number of pool slots over all GPUs.  Run outputs go to
``logger/<run name>/<date-time>/``; ``--resume_state true --load_date_time <date-time>
--load_index <k>`` continues a run from checkpoint ``k`` and the pool snapshot saved with it.
"""
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "4"
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
from FVdomain.Graph_pool import MmapDataPool, reset_owner_rank  # noqa: E402
from NNmodels.importer import NNmodel  # noqa: E402
from Utils import get_param  # noqa: E402
from Utils.Logger import ddp_run_logger  # noqa: E402
from Utils.OptimizerF import get_optimizer  # noqa: E402
from Utils.get_param import get_hyperparam  # noqa: E402

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")
if os.getenv("TORCH_COMPILE_DISABLE", "0") == "1":
    torch._dynamo.config.disable = True
torch.set_float32_matmul_precision("high")


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    params = get_param.params()
    total_dataset_size = params.dataset_size
    if total_dataset_size % world_size != 0:
        raise ValueError(f"--dataset_size ({total_dataset_size}) must be divisible by the number of ranks ({world_size})")
    params.dataset_size = total_dataset_size // world_size
    if rank == 0:
        print(f"{world_size} ranks, {total_dataset_size} slots ({params.dataset_size}/rank), "
              f"{world_size * params.batch_size} cases per optimizer step")

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
    resume_from = None
    if params.resume_state and params.load_date_time:
        resume_from = os.path.join("logger", get_hyperparam(params), params.load_date_time, "mmap_state")
    params = datasets.load_mesh_to_mmap(
        dataset_dir=params.dataset_dir, resume=params.resume_state, resume_from=resume_from, split=params.split,
        loss_mode=params.loss_mode, total_dataset_size=total_dataset_size, world_size=world_size)
    loader = HeteroGraphLoader(base_dataset=datasets, batch_size=params.batch_size, shuffle=True)
    if world_size > 1:
        dist.barrier()
    logger.info(f"Rank {rank} | Loaded {len(datasets.meta_pool)} cases")
    global_plot_number = torch.zeros(1, dtype=torch.long, device=device)

    model = NNmodel(params).to(device)
    if params.load_date_time is not None:
        ckpt_path = os.path.join("logger", get_hyperparam(params), params.load_date_time, "states",
                                 f"{0 if params.load_index is None else int(params.load_index)}.state")
        model.load_checkpoint(ckpt_path, device=device)
        logger.info(f"Rank {rank} | Loaded model weights from {ckpt_path}")
    fluid_model = (DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
                   if world_size > 1 else model)
    fluid_model.train()
    optimizer = get_optimizer(optimizer_type=params.optimizer_type, opt_parameters=fluid_model.parameters(),
                              params=params)
    params.load_index = 0 if params.load_index is None else int(params.load_index)

    for epoch in range(params.n_epochs):
        fluid_model.train()
        refresh_pool = False
        # resets cycle through the slots of all ranks
        if epoch % ceil(params.average_sequence_length / total_dataset_size) == 0:
            datasets._set_reset_env_flag(flag=True, rst_time=ceil(total_dataset_size / params.average_sequence_length))

        epoch_start = time.time()
        epoch_loss_sum = 0.0
        batch_count = 0
        epoch_residuals = {}
        for i_iter in range(params.max_inner_steps):
            if i_iter == params.max_inner_steps - 1:
                refresh_pool = True
            for fv_graph in loader:
                if fv_graph._hetero_data["cpd_cell"].x.device.type != device.type:
                    fv_graph = fv_graph.to(device, exclude_keys=["global_idx", "graph_index"])
                optimizer.zero_grad()
                loss_total, phi_new_cpd_cell = fluid_model(fv_graph=fv_graph)
                loss_total.backward()
                optimizer.step()
                epoch_loss_sum += loss_total.item()
                batch_count += 1

                with torch.no_grad():
                    for name, tensor in model.integrator.iter_named_residuals():
                        if tensor.dim() >= 2 and tensor.shape[-1] > 1:
                            for i in range(tensor.shape[-1]):
                                epoch_residuals[f"{name}_{i}"] = (
                                    epoch_residuals.get(f"{name}_{i}", 0.0)
                                    + torch.log10(tensor[..., i:i + 1] + 1e-30).mean().item())
                        else:
                            epoch_residuals[name] = (epoch_residuals.get(name, 0.0)
                                                     + torch.log10(tensor + 1e-30).mean().item())

                if refresh_pool:
                    datasets.refresh_pool(phi_new_cpd_cell=phi_new_cpd_cell.detach().cpu(),
                                          slot_idx=fv_graph.graph_cell.slot_idx)

        if world_size > 1:
            dist.barrier()
        if datasets.reset_env_flag:
            for _ in range(datasets.rst_time):
                # the reset counter is global; the rank holding that slot resets it
                if world_size > 1:
                    dist.broadcast(global_plot_number, src=0)
                current_plot = global_plot_number.item()
                if rank == reset_owner_rank(current_plot, total_dataset_size, world_size):
                    if datasets._plot_env:
                        datasets.plot_count = current_plot
                    datasets.reset_env(plot=datasets._plot_env)
                if rank == 0:
                    global_plot_number += 1
            datasets.reset_env_flag = False
            datasets._plot_env = True
        if world_size > 1:
            dist.barrier()

        if world_size > 1:
            loss_tensor = torch.tensor([epoch_loss_sum, float(batch_count)], device=device)
            dist.all_reduce(loss_tensor)
            epoch_loss_avg = loss_tensor[0].item() / loss_tensor[1].item()
            total_batches = int(loss_tensor[1].item())
        else:
            epoch_loss_avg = epoch_loss_sum / batch_count if batch_count > 0 else 0.0
            total_batches = batch_count
        total_res_batches = batch_count
        if epoch_residuals and world_size > 1:
            res_keys = list(epoch_residuals.keys())
            res_vals = torch.tensor([epoch_residuals[k] for k in res_keys] + [float(batch_count)],
                                    device=device, dtype=torch.float64)
            dist.all_reduce(res_vals)
            total_res_batches = int(res_vals[-1].item())
            for i, k in enumerate(res_keys):
                epoch_residuals[k] = res_vals[i].item()

        logger.info(f"Rank {rank} | Epoch {epoch} | Time: {time.time() - epoch_start:.2f}s | "
                    f"Loss: {epoch_loss_avg:.6f} | Batches: {total_batches}")
        if total_res_batches > 0 and epoch_residuals:
            logger.info(f"Rank {rank} | Epoch {epoch} | Residuals (log10): "
                        + " ".join(f"{k}={v / total_res_batches:.4f}" for k, v in epoch_residuals.items()))
            if rank == 0:
                logger.log_residuals(epoch=epoch, loss_total=epoch_loss_avg,
                                     **{k: v / total_res_batches for k, v in epoch_residuals.items()})

        if (epoch % 10 == 0) or (epoch == params.n_epochs - 1):
            if rank == 0:
                logger.save_state(model=fluid_model.module if world_size > 1 else fluid_model,
                                  optimizer=None, scheduler=None, index=str(epoch % 3))
            datasets.save_snapshot(checkpoint_idx=epoch % 3, epoch=epoch)
            if world_size > 1:
                dist.barrier()
        if world_size > 1:
            dist.barrier()

    datasets.close()
    logger.finalize_residuals()
    logger.info(f"Rank {rank} | Training done")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
