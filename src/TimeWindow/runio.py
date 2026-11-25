"""Run-directory naming, provenance, and VTM export I/O for FIFO-window ODIL."""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Any, Mapping, Sequence, Tuple

import h5py
import torch

from TimeWindow.config import (
    _EXPORT_MODES,
    FIFOConfig,
    get_odil_window_steps,
    read_fifo_config,
    validate_case_assets,
)
from TimeWindow.window import _DEFAULT_FIELD_LAYOUT, FieldLayout


def _float_tag(value: float) -> str:
    """Format a float compactly for a filesystem-safe experiment id."""
    text = f"{float(value):.8g}"
    text = re.sub(r"e([+-])0+(\d+)", r"e\1\2", text)
    return text.replace("+", "").replace(".", "p")


def _sanitize_experiment_tag(value: str) -> str:
    """Keep optional user labels readable without allowing path traversal."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return sanitized.strip("._-")


def _read_case_name(mesh_path: str) -> str:
    """Read the case name from the H5 mesh's root attribute before mesh load."""
    with h5py.File(mesh_path, "r") as h5_file:
        if "case_name" not in h5_file.attrs:
            raise ValueError(f"H5 file {mesh_path} has no 'case_name' root attribute.")
        case_name = h5_file.attrs["case_name"]
    if isinstance(case_name, bytes):
        case_name = case_name.decode("utf-8")
    if not isinstance(case_name, str) or not case_name.strip():
        raise ValueError(
            f"H5 file {mesh_path} has invalid case_name={case_name!r} (must be non-empty)."
        )
    return case_name


def _read_static_case_identity(dataset_dir: str) -> Tuple[int, float, str, str]:
    """Read naming fields that are available before the mesh is loaded."""
    config_path = os.path.join(dataset_dir, "fvconfig.toml")
    with open(config_path, "rb") as stream:
        fvconfig = tomllib.load(stream)
    window_steps = get_odil_window_steps(fvconfig)
    fifo_config = read_fifo_config(fvconfig)
    dt_value = float(fvconfig["theta_pde"]["dt"]["val"])
    momentum_scheme = str(
        fvconfig["fvSchemes"].get("momentumResidual", "cell")
    ).lower()
    if momentum_scheme in ("cell+face_cv", "face_cv+cell"):
        scheme_tag = "cell-faceCV"
    else:
        scheme_tag = _sanitize_experiment_tag(momentum_scheme)
    scheme_tag = f"CN{_float_tag(fifo_config.crank_nicolson_off_centering)}-{scheme_tag}"
    case_name = _read_case_name(validate_case_assets(dataset_dir))
    return window_steps, dt_value, scheme_tag, case_name


def build_run_directory_name(
    *,
    timestamp: str,
    window_steps: int,
    dt_value: float,
    learning_rate: float,
    physical_steps: int,
    iterations: int,
    seed: int,
    scheme_tag: str,
    case_label: str,
    execution_tag: str = "",
    experiment_tag: str = "",
) -> str:
    """Build the descriptive run-folder convention."""
    parts = [
        timestamp,
        _sanitize_experiment_tag(case_label),
        f"W{window_steps}",
        _sanitize_experiment_tag(scheme_tag),
        f"dt{_float_tag(dt_value)}",
        f"lr{_float_tag(learning_rate)}",
        f"steps{physical_steps}",
        f"iter{iterations}",
        f"seed{seed}",
    ]
    sanitized_execution_tag = _sanitize_experiment_tag(execution_tag)
    if sanitized_execution_tag:
        parts.append(sanitized_execution_tag)
    optional_tag = _sanitize_experiment_tag(experiment_tag)
    if optional_tag:
        parts.append(optional_tag)
    return "_".join(parts)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root(case_dir: str) -> str:
    here = os.path.abspath(case_dir)
    while here != os.path.dirname(here):
        if os.path.isdir(os.path.join(here, "src", "FVdomain")):
            return here
        here = os.path.dirname(here)
    return os.path.abspath(os.path.join(case_dir, os.pardir, os.pardir))


def _git_text(*args: str, case_dir: str) -> str | None:
    repository_root = _repository_root(case_dir)
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\r\n")


def snapshot_case_inputs(
    *,
    logger: Any,
    args: argparse.Namespace,
    invocation_args: Sequence[str],
    mesh_path: str,
    case_name: str,
    window_steps: int,
    dt_value: float,
    momentum_scheme: str,
    num_interior: int,
    device: torch.device,
    fifo_config: FIFOConfig,
    case_dir: str,
    re_target: float | None = None,
    layout: FieldLayout | None = None,
) -> str:
    """Copy case-local inputs and write a reproducible run manifest."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT

    snapshot_dir = os.path.join(logger.saving_path, "case_snapshot")
    snapshot_h5_dir = os.path.join(snapshot_dir, "h5")
    os.makedirs(snapshot_h5_dir, exist_ok=True)

    entry_script = os.path.basename(sys.argv[0])
    entry_source = os.path.join(case_dir, entry_script)
    if os.path.isfile(entry_source):
        shutil.copy2(entry_source, os.path.join(snapshot_dir, entry_script))
    source_files = sorted(
        glob.glob(os.path.join(case_dir, "*.py"))
        + glob.glob(os.path.join(case_dir, "*.sh"))
    )
    for source in source_files:
        filename = os.path.basename(source)
        if filename == entry_script:
            continue
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(snapshot_dir, filename))
    for filename in ("fvconfig.toml", "patchinfo.toml"):
        source = os.path.join(case_dir, "h5", filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(snapshot_h5_dir, filename))
    bc_src = os.path.join(case_dir, "h5", "bc")
    if os.path.isdir(bc_src):
        shutil.copytree(
            bc_src,
            os.path.join(snapshot_h5_dir, "bc"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    mesh_sha256 = _sha256_file(mesh_path)
    git_status = _git_text(
        "status", "--short", "--untracked-files=all", case_dir=case_dir
    )
    manifest = {
        "schema_version": 1,
        "run_id": os.path.basename(logger.saving_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [
            sys.executable,
            os.path.abspath(sys.argv[0]),
            *invocation_args,
        ],
        "cli": vars(args),
        "case": {
            "name": case_name,
            "Re": re_target,
            "dt": dt_value,
            "window_steps": window_steps,
            "window_strategy": "fifo",
            "rolling_scheme": fifo_config.rolling_scheme,
            "fifo_stride": fifo_config.stride,
            "physical_steps": fifo_config.physical_steps,
            "max_head_iterations": fifo_config.max_head_iterations,
            "min_snapshot_updates": fifo_config.min_snapshot_updates,
            "time_vectorized": True,
            "time_loss_reduction": "weighted_sum",
            "temporal_scheme": fifo_config.temporal_scheme,
            "crank_nicolson_off_centering": fifo_config.crank_nicolson_off_centering,
            "spatial_time_level": "current",
            "head_loss_weight": fifo_config.head_loss_weight,
            "future_loss_decay": fifo_config.future_loss_decay,
            "momentum_residual": momentum_scheme,
            "fields": list(layout.names),
            "adam_update_scale": dict(zip(layout.names, fifo_config.adam_update_scale)),
            "tail_extrapolation": fifo_config.tail_extrapolation,
            "num_interior_cells": num_interior,
            "residual_control": [
                {
                    "field": criterion.field_name,
                    "component": criterion.component,
                    "tolerance": criterion.tolerance,
                    "relTol": criterion.relative_tolerance,
                }
                for criterion in fifo_config.criteria
            ],
            "openfoam_reference_control": {
                "enabled": fifo_config.reference_control.enabled,
                "foam_path": fifo_config.reference_control.foam_path,
                "resolved_foam_path": (
                    os.path.abspath(
                        os.path.join(case_dir, fifo_config.reference_control.foam_path)
                    )
                    if fifo_config.reference_control.enabled
                    else None
                ),
                "check_stride": fifo_config.reference_control.check_stride,
                "acceptance_only": True,
                "used_in_loss_or_gradient": False,
                "thresholds": {
                    "max_u_normalized_rmse": fifo_config.reference_control.max_u_normalized_rmse,
                    "max_annulus_u_normalized_rmse": fifo_config.reference_control.max_annulus_u_normalized_rmse,
                    "max_annulus_u_error_p95": fifo_config.reference_control.max_annulus_u_error_p95,
                    "max_wake_u_normalized_rmse": fifo_config.reference_control.max_wake_u_normalized_rmse,
                    "max_p_normalized_rmse": fifo_config.reference_control.max_p_normalized_rmse,
                    "max_t_normalized_rmse": fifo_config.reference_control.max_t_normalized_rmse,
                    "min_u_correlation": fifo_config.reference_control.min_u_correlation,
                    "min_p_correlation": fifo_config.reference_control.min_p_correlation,
                    "min_t_correlation": fifo_config.reference_control.min_t_correlation,
                    "max_w_rms": fifo_config.reference_control.max_w_rms,
                    "max_w_error_rms": fifo_config.reference_control.max_w_error_rms,
                    "max_temperature_out_of_bounds_fraction": (
                        fifo_config.reference_control.max_temperature_out_of_bounds_fraction
                    ),
                    "max_recirculation_length_difference": (
                        fifo_config.reference_control.max_recirculation_length_difference
                    ),
                    "temperature_min": fifo_config.reference_control.temperature_min,
                    "temperature_max": fifo_config.reference_control.temperature_max,
                    "temperature_bound_tolerance": (
                        fifo_config.reference_control.temperature_bound_tolerance
                    ),
                },
            },
        },
        "mesh": {
            "path": os.path.relpath(mesh_path, _repository_root(case_dir)),
            "size_bytes": os.path.getsize(mesh_path),
            "sha256": mesh_sha256,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "fvm_backend": "warp",
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD", case_dir=case_dir),
            "status_short": git_status.splitlines() if git_status else [],
        },
        "case_snapshot": os.path.relpath(snapshot_dir, logger.saving_path),
    }
    manifest_path = os.path.join(logger.saving_path, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    logger.info(
        "Case snapshot saved to %s; mesh sha256=%s",
        snapshot_dir,
        mesh_sha256,
    )
    return manifest_path


def should_export_popped_snapshot(export_mode: str) -> bool:
    """Return whether an accepted FIFO head should be written to VTM."""
    if export_mode == "popped":
        return True
    if export_mode == "none":
        return False
    raise ValueError(f"export_mode must be one of {_EXPORT_MODES}, got {export_mode!r}")


def export_results(
    *,
    phi_cpd_cell: torch.Tensor,
    meta_data: Mapping[str, Any],
    mask_interior_cell: torch.Tensor,
    logger: Any,
    case_name: str,
    physical_step: int,
    physical_time: float,
) -> None:
    """Export one BC-enforced state as a VTM MultiBlock (internal mesh + patches)."""
    from Post_process.vtk_export import export_vtm

    theta_pde = meta_data["fvconfig"]["theta_pde"]
    re_raw = theta_pde.get("Re")
    re_tag = f"_Re={float(re_raw):.4f}" if isinstance(re_raw, (int, float)) and not isinstance(re_raw, bool) else ""
    dt_value = float(theta_pde["dt"]["val"])
    filename = f"NO.{physical_step:03d}_{case_name}_t={physical_time:.4f}s{re_tag}_dt={dt_value:.3f}"
    output_dir = os.path.join(logger.saving_path, "rollout_results", case_name, filename)
    os.makedirs(output_dir, exist_ok=True)

    phi_cpu = phi_cpd_cell.detach().cpu()
    mask_cpu = mask_interior_cell.detach().cpu().bool().reshape(-1)
    cell_fields = {"U": phi_cpu[:, 0:3]}
    if phi_cpu.shape[1] >= 5:
        cell_fields["p_rgh"] = phi_cpu[:, 3:4]
        cell_fields["T"] = phi_cpu[:, 4:5]
    else:
        cell_fields["p"] = phi_cpu[:, 3:4]

    export_vtm(
        os.path.join(output_dir, f"{filename}.vtm"),
        meta_data,
        mask_cpu,
        cell_fields,
    )
