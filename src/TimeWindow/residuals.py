"""OpenFOAM-style FIFO residual-control evaluation and display reductions."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from TimeWindow.config import ResidualCheck, ResidualCriterion
from TimeWindow.window import _LOG_EPS, WindowEvaluation


def evaluate_openfoam_residual_control(
    *,
    current: Mapping[str, float],
    initial: Mapping[str, float],
    criteria: Sequence[ResidualCriterion],
) -> ResidualCheck:
    """Apply OpenFOAM PIMPLE's per-field ``absCheck || relCheck`` rule."""
    field_passed: Dict[str, bool] = {}
    absolute_passed: Dict[str, bool] = {}
    relative_passed: Dict[str, bool] = {}
    relative: Dict[str, float] = {}
    for criterion in criteria:
        name = criterion.field_name
        if name not in current or name not in initial:
            raise KeyError(f"Missing residual value for configured field {name!r}")
        current_value = float(current[name])
        initial_value = float(initial[name])
        if not np.isfinite(current_value) or not np.isfinite(initial_value):
            raise FloatingPointError(f"Non-finite FIFO residual for {name}")
        ratio = current_value / max(initial_value, _LOG_EPS)
        abs_pass = current_value < criterion.tolerance
        rel_pass = ratio < criterion.relative_tolerance
        absolute_passed[name] = abs_pass
        relative_passed[name] = rel_pass
        field_passed[name] = abs_pass or rel_pass
        relative[name] = ratio
    return ResidualCheck(
        passed=bool(criteria) and all(field_passed.values()),
        field_passed=field_passed,
        absolute_passed=absolute_passed,
        relative_passed=relative_passed,
        current={name: float(value) for name, value in current.items()},
        initial={name: float(value) for name, value in initial.items()},
        relative=relative,
    )


def _component_log10_per_time(component: torch.Tensor) -> torch.Tensor:
    """Return one display residual per time step from ``[T,B,...]``."""
    detached = component.detach()
    if detached.ndim < 2:
        raise ValueError(
            f"Vectorized residual components must start with [T,B], got {tuple(detached.shape)}"
        )
    if detached.ndim == 2:
        per_time_graph = detached
    else:
        per_time_graph = detached.flatten(start_dim=2).sum(dim=-1)
    return torch.log10(per_time_graph.clamp_min(_LOG_EPS)).mean(dim=1)


def _component_log10(component: torch.Tensor) -> torch.Tensor:
    """Scalar mean display summary across the time window."""
    return _component_log10_per_time(component).mean()


def residual_control_values_by_time(
    evaluation: WindowEvaluation,
    criteria: Sequence[ResidualCriterion],
) -> Mapping[str, torch.Tensor]:
    """Reduce each configured component to normalized OpenFOAM-like max residuals."""
    values: Dict[str, torch.Tensor] = {}
    time_steps = len(evaluation.states)
    for criterion in criteria:
        component = evaluation.residual_components.get(criterion.component)
        if component is None:
            raise KeyError(
                f"FVM evaluation did not return component {criterion.component!r}"
            )
        if component.ndim < 2 or component.shape[0] != time_steps:
            raise RuntimeError(
                f"Residual component {criterion.component!r} must preserve "
                f"[T,B,...], got {tuple(component.shape)}"
            )
        raw = component.detach().abs()
        scale = evaluation.residual_scales.get(criterion.component)
        if scale is not None:
            scale = scale.detach()
            while scale.ndim < raw.ndim:
                scale = scale.unsqueeze(-1)
            if scale.shape[-1] != raw.shape[-1]:
                scale = scale.mean(dim=-1, keepdim=True)
            raw = raw / scale.clamp_min(1.0e-30)
        values[criterion.field_name] = raw.flatten(1).amax(1)
    return values


def residual_values_at(
    values_by_time: Mapping[str, torch.Tensor], time_index: int
) -> Dict[str, float]:
    """Materialize one queue snapshot's residuals as plain floats."""
    return {
        name: float(values[time_index].item())
        for name, values in values_by_time.items()
    }
