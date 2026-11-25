"""FIFO time-window hyperparameters, residual-control config, and CLI flags."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from FVsolver.FVdiscretization.ddt_schemes import parse_ddt_scheme
from TimeWindow.window import _FIELD_NAMES, resolve_fvconfig_field_layout

_ROLLING_SCHEMES = ("RollingWithGrad", "RollingNoGrad")
_EXPORT_MODES = ("popped", "none")


@dataclass(frozen=True)
class ResidualCriterion:
    """One OpenFOAM-style residualControl entry."""

    field_name: str
    component: str
    tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class ResidualCheck:
    """Per-field details and the combined nonlinear convergence decision."""

    passed: bool
    field_passed: Mapping[str, bool]
    absolute_passed: Mapping[str, bool]
    relative_passed: Mapping[str, bool]
    current: Mapping[str, float]
    initial: Mapping[str, float]
    relative: Mapping[str, float]


@dataclass(frozen=True)
class ReferenceGeometry:
    """Cylinder geometry and physical scales for the OpenFOAM reference gate."""

    velocity_scale: float
    pressure_scale: float
    cylinder_center_x: float
    cylinder_center_y: float
    cylinder_diameter: float
    centerline_half_width: float
    annulus_outer_radius_diameters: float
    wake_end_x_diameters: float
    wake_half_width_diameters: float


@dataclass(frozen=True)
class ReferenceControlConfig:
    """Read-only same-time OpenFOAM trajectory acceptance thresholds."""

    enabled: bool
    foam_path: str
    check_stride: int
    max_u_normalized_rmse: float
    max_annulus_u_normalized_rmse: float
    max_annulus_u_error_p95: float
    max_wake_u_normalized_rmse: float
    max_p_normalized_rmse: float
    max_t_normalized_rmse: float
    min_u_correlation: float
    min_p_correlation: float
    min_t_correlation: float
    max_w_rms: float
    max_w_error_rms: float
    max_temperature_out_of_bounds_fraction: float
    max_recirculation_length_difference: float
    temperature_min: float
    temperature_max: float
    temperature_bound_tolerance: float
    geometry: ReferenceGeometry


@dataclass(frozen=True)
class FIFOConfig:
    """Validated FIFO and optimizer hyperparameters loaded from fvconfig.toml."""

    enabled: bool
    window_steps: int
    stride: int
    physical_steps: int
    max_head_iterations: int
    check_interval: int
    min_head_iterations: int
    consecutive_checks: int
    learning_rate: float
    minimum_learning_rate: float
    lr_decay_iterations: int
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    adam_update_scale: Tuple[float, ...]
    grad_clip_norm: float
    head_loss_weight: float
    future_loss_decay: float
    min_snapshot_updates: int
    tail_extrapolation: float
    tail_init_std: float
    preserve_adam_state: bool
    eager_verify_on_pop: bool
    export_mode: str
    residual_reduction: str
    rolling_scheme: str
    temporal_scheme: str
    crank_nicolson_off_centering: float
    criteria: Tuple[ResidualCriterion, ...]
    reference_control: ReferenceControlConfig
    pseudo_time_enabled: bool = False
    pseudo_relax_momentum: float = 0.7
    pseudo_relax_energy: float = 0.7
    export_stride: int = 1


def _require_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_float(
    name: str,
    value: Any,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    valid = result > minimum if strictly_positive else result >= minimum
    if not valid:
        relation = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value!r}")
    return result


def _require_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_channel_scales(
    name: str,
    value: Any,
    field_names: Tuple[str, ...] = _FIELD_NAMES,
) -> Tuple[float, ...]:
    """Validate one non-negative optimizer update scale per ODIL field."""
    if not isinstance(value, (list, tuple)) or len(value) != len(field_names):
        raise ValueError(
            f"{name} must contain {len(field_names)} {field_names} values, got {value!r}"
        )
    result = tuple(
        _require_float(f"{name}[{field_name}]", scale)
        for field_name, scale in zip(field_names, value)
    )
    if not any(result):
        raise ValueError(f"{name} cannot freeze every field")
    return result


def read_fifo_config(fvconfig: Mapping[str, Any]) -> FIFOConfig:
    """Load all FIFO/ODIL controls from ``fvSolution.fifo``."""
    fv_solution = fvconfig.get("fvSolution", {})
    raw_fifo = fv_solution.get("fifo")
    if not isinstance(raw_fifo, Mapping):
        raise ValueError("fvSolution.fifo must be a TOML table")
    field_layout = resolve_fvconfig_field_layout(fvconfig)

    enabled = raw_fifo.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("fvSolution.fifo.enabled must be boolean")
    window_steps = _require_int(
        "fvSolution.fifo.window_steps",
        raw_fifo.get("window_steps", fv_solution.get("odil_window_steps", 1)),
    )
    legacy_window = fv_solution.get("odil_window_steps", window_steps)
    if legacy_window != window_steps:
        raise ValueError(
            "fvSolution.odil_window_steps and fifo.window_steps disagree: "
            f"{legacy_window!r} vs {window_steps!r}"
        )
    stride = _require_int("fvSolution.fifo.stride", raw_fifo.get("stride", 1))
    if stride != 1:
        raise ValueError("FIFO currently requires stride=1")
    physical_steps = _require_int(
        "fvSolution.fifo.physical_steps", raw_fifo.get("physical_steps", 50)
    )
    max_head_iterations = _require_int(
        "fvSolution.fifo.max_head_iterations",
        raw_fifo.get("max_head_iterations", 20_000),
    )
    check_interval = _require_int(
        "fvSolution.fifo.check_interval", raw_fifo.get("check_interval", 1)
    )
    min_head_iterations = _require_int(
        "fvSolution.fifo.min_head_iterations",
        raw_fifo.get("min_head_iterations", 1),
        minimum=0,
    )
    consecutive_checks = _require_int(
        "fvSolution.fifo.consecutive_checks",
        raw_fifo.get("consecutive_checks", 1),
    )
    learning_rate = _require_float(
        "fvSolution.fifo.learning_rate",
        raw_fifo.get("learning_rate", 1.0e-4),
        strictly_positive=True,
    )
    minimum_learning_rate = _require_float(
        "fvSolution.fifo.minimum_learning_rate",
        raw_fifo.get("minimum_learning_rate", 5.0e-6),
        strictly_positive=True,
    )
    if minimum_learning_rate > learning_rate:
        raise ValueError("FIFO minimum_learning_rate cannot exceed learning_rate")
    adam_beta1 = _require_float(
        "fvSolution.fifo.adam_beta1", raw_fifo.get("adam_beta1", 0.9)
    )
    adam_beta2 = _require_float(
        "fvSolution.fifo.adam_beta2", raw_fifo.get("adam_beta2", 0.999)
    )
    if not 0.0 <= adam_beta1 < 1.0 or not 0.0 <= adam_beta2 < 1.0:
        raise ValueError("FIFO Adam betas must satisfy 0 <= beta < 1")
    adam_eps = _require_float(
        "fvSolution.fifo.adam_eps",
        raw_fifo.get("adam_eps", 1.0e-8),
        strictly_positive=True,
    )
    adam_update_scale = _require_channel_scales(
        "fvSolution.fifo.adam_update_scale",
        raw_fifo.get("adam_update_scale", [1.0] * field_layout.num_channels),
        field_layout.names,
    )
    grad_clip_norm = _require_float(
        "fvSolution.fifo.grad_clip_norm",
        raw_fifo.get("grad_clip_norm", 1.0),
    )
    head_loss_weight = _require_float(
        "fvSolution.fifo.head_loss_weight",
        raw_fifo.get("head_loss_weight", 1.0),
        strictly_positive=True,
    )
    future_loss_decay = _require_float(
        "fvSolution.fifo.future_loss_decay",
        raw_fifo.get("future_loss_decay", 1.0),
        strictly_positive=True,
    )
    if future_loss_decay > 1.0:
        raise ValueError("fvSolution.fifo.future_loss_decay must be <= 1")
    min_snapshot_updates = _require_int(
        "fvSolution.fifo.min_snapshot_updates",
        raw_fifo.get("min_snapshot_updates", 0),
        minimum=0,
    )
    tail_extrapolation = _require_float(
        "fvSolution.fifo.tail_extrapolation",
        raw_fifo.get("tail_extrapolation", 1.0),
    )
    tail_init_std = _require_float(
        "fvSolution.fifo.tail_init_std",
        raw_fifo.get("tail_init_std", 1.0e-5),
    )
    preserve_adam_state = raw_fifo.get("preserve_adam_state", True)
    eager_verify_on_pop = raw_fifo.get("eager_verify_on_pop", True)
    if not isinstance(preserve_adam_state, bool):
        raise ValueError("fvSolution.fifo.preserve_adam_state must be boolean")
    if not isinstance(eager_verify_on_pop, bool):
        raise ValueError("fvSolution.fifo.eager_verify_on_pop must be boolean")
    export_mode = str(raw_fifo.get("export_mode", "popped")).lower()
    if export_mode not in _EXPORT_MODES:
        raise ValueError(
            f"fvSolution.fifo.export_mode must be one of {_EXPORT_MODES}, got {export_mode!r}"
        )
    residual_reduction = str(raw_fifo.get("residual_reduction", "max")).lower()
    if residual_reduction != "max":
        raise ValueError("FIFO residual_reduction currently requires 'max'")

    rolling_raw = str(raw_fifo.get("rolling_scheme", "RollingWithGrad")).strip()
    rolling_lookup = {
        name.lower().replace("_", "").replace("-", ""): name
        for name in _ROLLING_SCHEMES
    }
    rolling_scheme = rolling_lookup.get(
        rolling_raw.lower().replace("_", "").replace("-", "")
    )
    if rolling_scheme is None:
        raise ValueError(
            f"fvSolution.fifo.rolling_scheme must be one of {_ROLLING_SCHEMES}, got {rolling_raw!r}"
        )

    lr_decay_iterations = _require_int(
        "fvSolution.fifo.lr_decay_iterations",
        raw_fifo.get("lr_decay_iterations", max_head_iterations),
    )

    steady_state = False
    ddt_raw = str((fvconfig.get("fvSchemes") or {}).get("ddtSchemes", "")).strip()
    if ddt_raw:
        try:
            steady_state = parse_ddt_scheme(ddt_raw).is_steady_state
        except ValueError:
            steady_state = False
    temporal_scheme_raw = str(
        raw_fifo.get("temporal_scheme", "CrankNicolson")
    ).strip()
    temporal_scheme_key = (
        temporal_scheme_raw.lower().replace("_", "").replace("-", "")
    )
    if steady_state:
        temporal_scheme = "steady"
    elif temporal_scheme_key in ("backward", "bdf2"):
        temporal_scheme = "backward"
    elif temporal_scheme_key in ("cranknicolson", "cn"):
        temporal_scheme = "CrankNicolson"
    else:
        raise ValueError(
            "fvSolution.fifo.temporal_scheme must be 'backward' or "
            f"'CrankNicolson', got {temporal_scheme_raw!r}"
        )
    crank_nicolson_off_centering = _require_float(
        "fvSolution.fifo.crank_nicolson_off_centering",
        raw_fifo.get("crank_nicolson_off_centering", 0.5),
    )
    if temporal_scheme in ("backward", "steady"):
        crank_nicolson_off_centering = 0.0
    elif (
        abs(crank_nicolson_off_centering - 0.5) > 1.0e-12
        and abs(crank_nicolson_off_centering) > 1.0e-12
    ):
        raise ValueError(
            "FIFO is calibrated to exact OpenFOAM CrankNicolson 0.5; "
            "set crank_nicolson_off_centering=0.5 (or 0.0 for Euler)"
        )
    spatial_ddt_scheme = str(
        fvconfig.get("fvSchemes", {}).get("ddtSchemes", "")
    ).strip().lower()
    if not steady_state and spatial_ddt_scheme != "implicit":
        raise ValueError(
            "FIFO CrankNicolson/backward requires fvSchemes.ddtSchemes="
            "'implicit' so spatial fluxes are evaluated at the current time level"
        )

    raw_control = raw_fifo.get("residualControl")
    if not isinstance(raw_control, Mapping) or not raw_control:
        raise ValueError("fvSolution.fifo.residualControl must be a non-empty table")
    criteria: List[ResidualCriterion] = []
    valid_components = {"cont", "mom"}
    if field_layout.has_energy:
        valid_components.add("energy")
    for field_name, raw_entry in raw_control.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"FIFO residualControl.{field_name} must be a TOML table"
            )
        component = str(raw_entry.get("component", ""))
        if component not in valid_components:
            raise ValueError(
                f"FIFO residualControl.{field_name}.component must be one of "
                f"{sorted(valid_components)}, got {component!r}"
            )
        tolerance = _require_float(
            f"FIFO residualControl.{field_name}.tolerance",
            raw_entry.get("tolerance"),
            strictly_positive=True,
        )
        relative_tolerance = _require_float(
            f"FIFO residualControl.{field_name}.relTol",
            raw_entry.get("relTol", 0.0),
            minimum=-1.0,
        )
        criteria.append(
            ResidualCriterion(
                field_name=str(field_name),
                component=component,
                tolerance=tolerance,
                relative_tolerance=relative_tolerance,
            )
        )
    components = [criterion.component for criterion in criteria]
    if len(set(components)) != len(components):
        raise ValueError("FIFO residualControl components must be unique")

    raw_reference = raw_fifo.get("referenceControl", {})
    if not isinstance(raw_reference, Mapping):
        raise ValueError("fvSolution.fifo.referenceControl must be a TOML table")
    reference_enabled = raw_reference.get("enabled", False)
    if not isinstance(reference_enabled, bool):
        raise ValueError("fvSolution.fifo.referenceControl.enabled must be boolean")
    raw_geometry = raw_reference.get("geometry", {})
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("fvSolution.fifo.referenceControl.geometry must be a TOML table")
    reference_geometry = ReferenceGeometry(
        velocity_scale=_require_float(
            "fvSolution.fifo.referenceControl.geometry.velocity_scale",
            raw_geometry.get("velocity_scale", 1.8),
            strictly_positive=True,
        ),
        pressure_scale=_require_float(
            "fvSolution.fifo.referenceControl.geometry.pressure_scale",
            raw_geometry.get("pressure_scale", 3.24),
            strictly_positive=True,
        ),
        cylinder_center_x=_require_finite_float(
            "fvSolution.fifo.referenceControl.geometry.cylinder_center_x",
            raw_geometry.get("cylinder_center_x", 0.0),
        ),
        cylinder_center_y=_require_finite_float(
            "fvSolution.fifo.referenceControl.geometry.cylinder_center_y",
            raw_geometry.get("cylinder_center_y", 0.0),
        ),
        cylinder_diameter=_require_float(
            "fvSolution.fifo.referenceControl.geometry.cylinder_diameter",
            raw_geometry.get("cylinder_diameter", 0.1),
            strictly_positive=True,
        ),
        centerline_half_width=_require_float(
            "fvSolution.fifo.referenceControl.geometry.centerline_half_width",
            raw_geometry.get("centerline_half_width", 0.01),
            strictly_positive=True,
        ),
        annulus_outer_radius_diameters=_require_float(
            "fvSolution.fifo.referenceControl.geometry.annulus_outer_radius_diameters",
            raw_geometry.get("annulus_outer_radius_diameters", 1.0),
            strictly_positive=True,
        ),
        wake_end_x_diameters=_require_float(
            "fvSolution.fifo.referenceControl.geometry.wake_end_x_diameters",
            raw_geometry.get("wake_end_x_diameters", 10.0),
            strictly_positive=True,
        ),
        wake_half_width_diameters=_require_float(
            "fvSolution.fifo.referenceControl.geometry.wake_half_width_diameters",
            raw_geometry.get("wake_half_width_diameters", 3.0),
            strictly_positive=True,
        ),
    )
    reference_control = ReferenceControlConfig(
        enabled=reference_enabled,
        foam_path=str(raw_reference.get("foam_path", "")),
        check_stride=_require_int(
            "fvSolution.fifo.referenceControl.check_stride",
            raw_reference.get("check_stride", 1),
        ),
        max_u_normalized_rmse=_require_float(
            "fvSolution.fifo.referenceControl.max_u_normalized_rmse",
            raw_reference.get("max_u_normalized_rmse", 0.06),
        ),
        max_annulus_u_normalized_rmse=_require_float(
            "fvSolution.fifo.referenceControl.max_annulus_u_normalized_rmse",
            raw_reference.get("max_annulus_u_normalized_rmse", 0.05),
        ),
        max_annulus_u_error_p95=_require_float(
            "fvSolution.fifo.referenceControl.max_annulus_u_error_p95",
            raw_reference.get("max_annulus_u_error_p95", 0.10),
        ),
        max_wake_u_normalized_rmse=_require_float(
            "fvSolution.fifo.referenceControl.max_wake_u_normalized_rmse",
            raw_reference.get("max_wake_u_normalized_rmse", 0.03),
        ),
        max_p_normalized_rmse=_require_float(
            "fvSolution.fifo.referenceControl.max_p_normalized_rmse",
            raw_reference.get("max_p_normalized_rmse", 0.22),
        ),
        max_t_normalized_rmse=_require_float(
            "fvSolution.fifo.referenceControl.max_t_normalized_rmse",
            raw_reference.get("max_t_normalized_rmse", 0.025),
        ),
        min_u_correlation=_require_finite_float(
            "fvSolution.fifo.referenceControl.min_u_correlation",
            raw_reference.get("min_u_correlation", 0.96),
        ),
        min_p_correlation=_require_finite_float(
            "fvSolution.fifo.referenceControl.min_p_correlation",
            raw_reference.get("min_p_correlation", 0.95),
        ),
        min_t_correlation=_require_finite_float(
            "fvSolution.fifo.referenceControl.min_t_correlation",
            raw_reference.get("min_t_correlation", 0.96),
        ),
        max_w_rms=_require_float(
            "fvSolution.fifo.referenceControl.max_w_rms",
            raw_reference.get("max_w_rms", 5.0e-5),
        ),
        max_w_error_rms=_require_float(
            "fvSolution.fifo.referenceControl.max_w_error_rms",
            raw_reference.get("max_w_error_rms", 5.0e-5),
        ),
        max_temperature_out_of_bounds_fraction=_require_float(
            "fvSolution.fifo.referenceControl.max_temperature_out_of_bounds_fraction",
            raw_reference.get("max_temperature_out_of_bounds_fraction", 1.0e-3),
        ),
        max_recirculation_length_difference=_require_float(
            "fvSolution.fifo.referenceControl.max_recirculation_length_difference",
            raw_reference.get("max_recirculation_length_difference", 0.02),
        ),
        temperature_min=_require_finite_float(
            "fvSolution.fifo.referenceControl.temperature_min",
            raw_reference.get("temperature_min", 0.0),
        ),
        temperature_max=_require_finite_float(
            "fvSolution.fifo.referenceControl.temperature_max",
            raw_reference.get("temperature_max", 1.0),
        ),
        temperature_bound_tolerance=_require_float(
            "fvSolution.fifo.referenceControl.temperature_bound_tolerance",
            raw_reference.get("temperature_bound_tolerance", 0.02),
        ),
        geometry=reference_geometry,
    )
    if reference_control.enabled and not reference_control.foam_path:
        raise ValueError(
            "fvSolution.fifo.referenceControl.foam_path is required when enabled"
        )
    if not -1.0 <= reference_control.min_u_correlation <= 1.0:
        raise ValueError("referenceControl.min_u_correlation must lie in [-1,1]")
    if not -1.0 <= reference_control.min_p_correlation <= 1.0:
        raise ValueError("referenceControl.min_p_correlation must lie in [-1,1]")
    if not -1.0 <= reference_control.min_t_correlation <= 1.0:
        raise ValueError("referenceControl.min_t_correlation must lie in [-1,1]")
    if reference_control.temperature_min > reference_control.temperature_max:
        raise ValueError("referenceControl temperature_min cannot exceed temperature_max")

    export_stride = _require_int(
        "fvSolution.fifo.export_stride", raw_fifo.get("export_stride", 1)
    )
    raw_pseudo = raw_fifo.get("pseudo_time", {}) or {}
    if not isinstance(raw_pseudo, dict):
        raise ValueError("fvSolution.fifo.pseudo_time must be a TOML table")
    pseudo_time_enabled = bool(raw_pseudo.get("enabled", False))
    pseudo_relax_momentum = _require_float(
        "fvSolution.fifo.pseudo_time.relax_U", raw_pseudo.get("relax_U", 0.7)
    )
    pseudo_relax_energy = _require_float(
        "fvSolution.fifo.pseudo_time.relax_T", raw_pseudo.get("relax_T", 0.7)
    )
    if pseudo_time_enabled:
        if temporal_scheme != "steady":
            raise ValueError(
                "fvSolution.fifo.pseudo_time needs fvSchemes.ddtSchemes = 'steadyState'"
            )
        for key, value in (("relax_U", pseudo_relax_momentum), ("relax_T", pseudo_relax_energy)):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"fvSolution.fifo.pseudo_time.{key} must lie in (0, 1), got {value}"
                )

    return FIFOConfig(
        enabled=enabled,
        window_steps=window_steps,
        stride=stride,
        physical_steps=physical_steps,
        max_head_iterations=max_head_iterations,
        check_interval=check_interval,
        min_head_iterations=min_head_iterations,
        consecutive_checks=consecutive_checks,
        learning_rate=learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        adam_update_scale=adam_update_scale,
        grad_clip_norm=grad_clip_norm,
        head_loss_weight=head_loss_weight,
        future_loss_decay=future_loss_decay,
        min_snapshot_updates=min_snapshot_updates,
        tail_extrapolation=tail_extrapolation,
        tail_init_std=tail_init_std,
        preserve_adam_state=preserve_adam_state,
        eager_verify_on_pop=eager_verify_on_pop,
        export_mode=export_mode,
        residual_reduction=residual_reduction,
        rolling_scheme=rolling_scheme,
        lr_decay_iterations=lr_decay_iterations,
        temporal_scheme=temporal_scheme,
        crank_nicolson_off_centering=crank_nicolson_off_centering,
        criteria=tuple(criteria),
        reference_control=reference_control,
        pseudo_time_enabled=pseudo_time_enabled,
        pseudo_relax_momentum=pseudo_relax_momentum,
        pseudo_relax_energy=pseudo_relax_energy,
        export_stride=export_stride,
    )


def format_effective_config(argv, *configs) -> str:
    """Render argv plus every resolved config dataclass, one line per config."""
    lines = ["argv: " + " ".join(argv)]
    for cfg in configs:
        fields = sorted(vars(cfg).items())
        rendered = " ".join(f"{k}={v!r}" for k, v in fields)
        lines.append(f"EffectiveConfig {type(cfg).__name__}: {rendered}")
    return "\n".join(lines)


def get_odil_window_steps(fvconfig: Mapping[str, Any]) -> int:
    """Read and validate the ODIL time-window width from fvconfig."""
    fv_solution = fvconfig.get("fvSolution", {})
    raw_fifo = fv_solution.get("fifo", {})
    raw_value = (
        raw_fifo.get("window_steps", fv_solution.get("odil_window_steps", 1))
        if isinstance(raw_fifo, Mapping)
        else fv_solution.get("odil_window_steps", 1)
    )
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(
            f"fvSolution.fifo.window_steps must be an integer >= 1, got {raw_value!r}"
        )
    if raw_value < 1:
        raise ValueError(f"fvSolution.fifo.window_steps must be >= 1, got {raw_value}")
    return raw_value


def validate_case_assets(dataset_dir: str) -> str:
    """Require the case directory layout with exactly one H5 mesh and patchinfo.toml."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"FIFO-window dataset directory is missing: {dataset_dir}")

    root_h5_files = sorted(
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if name.lower().endswith(".h5")
        and os.path.isfile(os.path.join(dataset_dir, name))
    )
    nested_h5_files = []
    for root, _, files in os.walk(dataset_dir):
        if os.path.abspath(root) == os.path.abspath(dataset_dir):
            continue
        nested_h5_files.extend(
            os.path.join(root, name)
            for name in files
            if name.lower().endswith(".h5")
        )

    patchinfo_path = os.path.join(dataset_dir, "patchinfo.toml")
    if len(root_h5_files) != 1 or nested_h5_files or not os.path.isfile(patchinfo_path):
        discovered = root_h5_files + sorted(nested_h5_files)
        raise FileNotFoundError(
            "A FIFO-window run requires exactly one h5/*.h5 and h5/patchinfo.toml. "
            f"Discovered H5 files: {discovered!r}; "
            f"patchinfo exists: {os.path.isfile(patchinfo_path)}"
        )
    return root_h5_files[0]


def apply_fifo_cli_overrides(
    config: FIFOConfig, args: argparse.Namespace
) -> FIFOConfig:
    """Apply explicitly supplied CLI overrides to a parsed ``FIFOConfig``."""
    updates: Dict[str, Any] = {}
    mapping = {
        "max_head_iterations": "max_head_iterations",
        "n_physical_steps": "physical_steps",
        "export_mode": "export_mode",
        "lr": "learning_rate",
        "minimum_lr": "minimum_learning_rate",
        "adam_eps": "adam_eps",
        "grad_clip_norm": "grad_clip_norm",
        "head_loss_weight": "head_loss_weight",
        "future_loss_decay": "future_loss_decay",
        "tail_init_std": "tail_init_std",
        "cn_off_centering": "crank_nicolson_off_centering",
    }
    for argument_name, config_name in mapping.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            updates[config_name] = value
    if getattr(args, "reference_control", None) is not None:
        updates["reference_control"] = replace(
            config.reference_control, enabled=args.reference_control
        )
    if getattr(args, "eager_verify", None) is not None:
        updates["eager_verify_on_pop"] = bool(args.eager_verify)
    if getattr(args, "preserve_adam_state", None) is not None:
        updates["preserve_adam_state"] = bool(args.preserve_adam_state)
    if getattr(args, "rel_tol_scale", None) is not None:
        scale = _require_float(
            "rel_tol_scale", args.rel_tol_scale, strictly_positive=True
        )
        updates["criteria"] = tuple(
            replace(
                criterion,
                relative_tolerance=criterion.relative_tolerance * scale,
            )
            for criterion in config.criteria
        )
    overridden = replace(config, **updates)
    _require_int("max_head_iterations", overridden.max_head_iterations)
    _require_int("physical_steps", overridden.physical_steps)
    _require_float("learning_rate", overridden.learning_rate, strictly_positive=True)
    _require_float(
        "minimum_learning_rate",
        overridden.minimum_learning_rate,
        strictly_positive=True,
    )
    if overridden.minimum_learning_rate > overridden.learning_rate:
        raise ValueError("minimum_lr cannot exceed lr")
    _require_float("adam_eps", overridden.adam_eps, strictly_positive=True)
    _require_float("grad_clip_norm", overridden.grad_clip_norm)
    _require_float("head_loss_weight", overridden.head_loss_weight, strictly_positive=True)
    _require_float("future_loss_decay", overridden.future_loss_decay, strictly_positive=True)
    if overridden.future_loss_decay > 1.0:
        raise ValueError("future_loss_decay must be <= 1")
    _require_float("tail_init_std", overridden.tail_init_std)
    return overridden


def add_fifo_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare shared CLI flags for FIFO-window ODIL runs."""
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.getcwd(), "h5"),
        type=str,
        help="Case h5 directory containing mesh .h5, patchinfo.toml, fvconfig.toml, and bc/.",
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--exp-name", default="", type=str)
    parser.add_argument(
        "--csv-flush-rows",
        default=25,
        type=int,
        help="Flush residuals.csv after this many optimization records.",
    )
    parser.add_argument(
        "--print-freq",
        default=None,
        type=int,
        help="Print/log cadence; unset uses the optimizer default.",
    )
    parser.add_argument(
        "--cn-off-centering",
        default=None,
        type=float,
        help="Crank-Nicolson off-centering (0.5 = pure CN, 0.0 = implicit Euler).",
    )
    parser.add_argument(
        "--max-head-iterations",
        "--n-epochs",
        dest="max_head_iterations",
        default=None,
        type=int,
        help="Override FIFO max iterations while one snapshot is queue head.",
    )
    parser.add_argument(
        "--n-physical-steps",
        default=None,
        type=int,
        help="Override the number of accepted/popped physical snapshots.",
    )
    parser.add_argument(
        "--export-mode",
        choices=_EXPORT_MODES,
        default=None,
        help="Override FIFO VTM export policy: popped heads or none.",
    )
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--minimum-lr", default=None, type=float)
    parser.add_argument("--adam-eps", default=None, type=float)
    parser.add_argument("--grad-clip-norm", default=None, type=float)
    parser.add_argument("--head-loss-weight", default=None, type=float)
    parser.add_argument("--future-loss-decay", default=None, type=float)
    parser.add_argument(
        "--tail-init-std", "--delta-init-std", dest="tail_init_std",
        default=None, type=float,
    )
    parser.add_argument(
        "--preserve-adam-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fvSolution.fifo.preserve_adam_state.",
    )
    parser.add_argument(
        "--eager-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fvSolution.fifo.eager_verify_on_pop.",
    )
    parser.add_argument(
        "--rel-tol-scale",
        default=None,
        type=float,
        help="Multiply every residualControl relTol by this factor.",
    )
    parser.add_argument(
        "--reference-control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the detached OpenFOAM acceptance gate.",
    )
    parser.add_argument(
        "--backend",
        choices=("warp",),
        default="warp",
        help="FVM solver backend (Warp only).",
    )
