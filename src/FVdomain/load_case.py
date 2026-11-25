"""Unified case-loading API for PCNO training and unsteady ODIL solves."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from FVdomain.Graph_loader import HeteroGraphLoader
from FVdomain.Graph_pool import HeteroDataPool, MmapDataPool
from FVdomain.Load_mesh import H5CFDdataset
from FVsolver.integrator import Integrator


@dataclass
class LoadedCase:
    params: Any
    pool: Any
    loader: Any
    fv_graph: Any = None
    meta_data: Any = None
    case_name: str | None = None
    graph_index: int | None = None
    integrator: Any = None
    solver: Any = None


def load_mesh_only(h5_path, params=None):
    """Load a raw H5 mesh and initialize its boundary conditions without creating a pool."""
    if params is None:
        params = SimpleNamespace(fourier_num_freqs=0)
    return H5CFDdataset(params=params, case_list=[str(h5_path)])[0]


def load_case(
    dataset_dir,
    params,
    *,
    stage="solver",
    pool="inmemory",
    device=None,
    batch_size=1,
    shuffle=False,
    loader_device=None,
    resume=False,
    resume_from=None,
    split=None,
    loss_mode=None,
    total_dataset_size=None,
    world_size=None,
    param_overrides=None,
    fvconfig_overrides=(),
):
    if stage not in ("graph", "solver"):
        raise ValueError(f"Unknown stage {stage!r}; expected 'graph' or 'solver'")

    if pool in ("inmemory", "legacy"):
        datasets = HeteroDataPool(params=params, device=device, state_save_dir=params.state_save_dir)
        params = datasets.load_mesh_to_cpu(dataset_dir=dataset_dir)
    elif pool == "mmap":
        datasets = MmapDataPool(params=params, device=device, state_save_dir=params.state_save_dir)
        params = datasets.load_mesh_to_mmap(
            dataset_dir=dataset_dir,
            resume=resume,
            resume_from=resume_from,
            split=split if split is not None else getattr(params, "split", "all"),
            loss_mode=loss_mode if loss_mode is not None else getattr(params, "loss_mode", "fvm"),
            total_dataset_size=total_dataset_size,
            world_size=world_size if world_size is not None else 1,
            param_overrides=param_overrides,
        )
    else:
        raise ValueError(f"Unknown pool strategy {pool!r}; expected 'inmemory' or 'mmap'")

    if fvconfig_overrides:
        from Utils.toml_utils import apply_overrides

        params.fvconfig_overrides = apply_overrides(params.fvconfig, fvconfig_overrides)
        for item in params.fvconfig_overrides:
            sec, _, name = item.split(" = ")[0].partition(".")
            if sec == "fvSolution" and "." not in name and hasattr(params, name):
                setattr(params, name, params.fvconfig[sec][name])

    loader_kwargs = dict(
        base_dataset=datasets,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=shuffle,
    )
    if loader_device is not None:
        loader_kwargs["device"] = loader_device
    loader = HeteroGraphLoader(**loader_kwargs)

    case = LoadedCase(params=params, pool=datasets, loader=loader)
    if stage == "graph":
        return case

    fv_graph = next(iter(loader))
    graph_index = int(fv_graph.graph_Index.graph_index.cpu().item())
    meta_data = datasets.meta_pool[graph_index]
    case.case_name = str(meta_data["case_name"])
    case.graph_index = graph_index
    case.meta_data = meta_data
    case.fv_graph = fv_graph.to(device, exclude_keys=["global_idx", "graph_index"])

    integrator = Integrator(params)
    integrator.register_properties(case.fv_graph, force_register=True)
    case.integrator = integrator
    case.solver = getattr(integrator, "_solver", integrator)
    return case


def solver_params(dataset_dir, *, state_save_dir=None):
    """Build a standalone parameter namespace for solver-only runs (ODIL)."""
    from Utils import get_param

    params = get_param.params(argv=[])
    params.on_gpu = 0
    params.net = "TransFVGN_v2"
    params.dataset_dir = dataset_dir
    params.load_date_time = None
    params.load_index = None
    params.dataset_size = 1
    params.batch_size = 1
    params.integrator = "implicit"
    params.optimizer_type = "adam"
    params.lr = 1e-3
    params.weight_decay = 0.0
    params.scheduler_type = "linear_decay"
    params.lr_min = params.lr * 0.1
    params.lr_step_size = 5000
    params.lr_factor = 0.5
    params.lr_linear_slope = None
    params.state_save_dir = state_save_dir
    return params


def load_solver_case(dataset_dir, *, device, overrides=(), state_save_dir=None):
    """Load one case and return a registered Warp FVM solver ready for ODIL."""
    params = solver_params(dataset_dir, state_save_dir=state_save_dir)
    return load_case(
        dataset_dir,
        params,
        stage="solver",
        pool="inmemory",
        device=device,
        fvconfig_overrides=overrides,
    )
