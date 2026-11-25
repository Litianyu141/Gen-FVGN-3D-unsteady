"""Distributed training helpers supporting ``torchrun``, Slurm ``srun``, and single-GPU runs."""

import os
import subprocess
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _resolve_slurm_master_addr() -> str:
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if nodelist:
        try:
            out = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip().splitlines()
            if out:
                return out[0].strip()
        except Exception:
            pass
    return "127.0.0.1"


def resolve_distributed_env() -> Tuple[int, int, int]:
    """Resolve ``(rank, local_rank, world_size)`` from ``torchrun`` or ``srun``."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        n_vis = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % n_vis))
    else:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """Initialize distributed training and return ``(rank, device_local_rank, world_size)``."""
    rank, local_rank, world_size = resolve_distributed_env()
    device_ordinal = (
        0
        if (torch.cuda.is_available() and torch.cuda.device_count() == 1)
        else local_rank
    )

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(device_ordinal)
    os.environ["WORLD_SIZE"] = str(world_size)

    if dist.is_initialized():
        return dist.get_rank(), device_ordinal, dist.get_world_size()

    if world_size == 1:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(device_ordinal)
        return rank, device_ordinal, world_size

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = _resolve_slurm_master_addr()
    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if job_id.isdigit():
            os.environ["MASTER_PORT"] = str(15000 + (int(job_id) % 40000))
        else:
            os.environ["MASTER_PORT"] = "29500"

    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(device_ordinal)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, device_ordinal, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    return rt


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_string(
    string: Optional[str], src: int = 0, device: Optional[torch.device] = None
) -> Optional[str]:
    if not dist.is_initialized():
        return string
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 1024
    if get_rank() == src:
        padded = (string or "").ljust(max_len)[:max_len]
        tensor = torch.tensor([ord(c) for c in padded], dtype=torch.long, device=device)
    else:
        tensor = torch.zeros(max_len, dtype=torch.long, device=device)

    dist.broadcast(tensor, src=src)
    return "".join(chr(c) for c in tensor.tolist()).strip()


def wrap_model_ddp(
    model: torch.nn.Module, local_rank: int, find_unused_parameters: bool = False
) -> DDP:
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def sync_model_params(model: torch.nn.Module, src: int = 0) -> None:
    if not dist.is_initialized():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=src)
