"""Optimizer-agnostic FIFO receding-horizon queue over an unsteady time window."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Tuple

import torch

from TimeWindow.config import FIFOConfig, ResidualCheck
from TimeWindow.pseudo_time import local_pseudo_time_scale
from TimeWindow.residuals import (
    evaluate_openfoam_residual_control,
    residual_values_at,
)
from TimeWindow.window import (
    FieldLayout,
    WindowCore,
    WindowEvaluation,
    _time_derivative_fields,
    evaluate_time_window,
    initialize_window_delta,
    resolve_fvconfig_field_layout,
)


def fifo_time_loss_weights(
    config: FIFOConfig, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build fixed receding-horizon weights without reducing the time axis."""
    offsets = torch.arange(config.window_steps, device=device, dtype=dtype)
    weights = torch.pow(
        torch.as_tensor(config.future_loss_decay, device=device, dtype=dtype),
        offsets,
    )
    weights[0] *= config.head_loss_weight
    return weights


@torch.no_grad()
def shift_fifo_parameter_and_adam_state_(
    *,
    delta_phi_interior: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    tail_init_std: float,
    preserve_adam_state: bool,
    tail_extrapolation: float = 0.0,
) -> None:
    """Pop delta[0], shift rows forward, and extrapolate the newest increment."""
    if delta_phi_interior.ndim != 3 or delta_phi_interior.shape[0] < 1:
        raise ValueError("FIFO shifting requires delta shape [W>=1,N,C]")
    predicted_tail = delta_phi_interior[-1].clone().mul_(tail_extrapolation)
    delta_phi_interior[:-1].copy_(delta_phi_interior[1:].clone())
    delta_phi_interior[-1].copy_(predicted_tail)
    if tail_init_std > 0:
        delta_phi_interior[-1].add_(
            torch.randn_like(delta_phi_interior[-1]), alpha=tail_init_std
        )
    delta_phi_interior.grad = None

    state = optimizer.state.get(delta_phi_interior)
    if not state:
        return
    if not preserve_adam_state:
        state.clear()
        return
    for state_value in state.values():
        if (
            isinstance(state_value, torch.Tensor)
            and state_value.ndim >= 1
            and state_value.shape[0] == delta_phi_interior.shape[0]
        ):
            state_value[:-1].copy_(state_value[1:].clone())
            state_value[-1].zero_()


class FIFOWindowDriver:
    """Fixed-W receding-horizon queue with one-dt FIFO advancement."""

    def __init__(
        self,
        *,
        phi_initial_cpd_cell: torch.Tensor,
        integrator: Any,
        params: Any,
        logger: Any,
        fifo_config: FIFOConfig,
        window_core: WindowCore,
        reference_gate: Any | None = None,
    ) -> None:
        if not fifo_config.enabled:
            raise ValueError("This runner requires fvSolution.fifo.enabled=true")
        if fifo_config.window_steps < 2 and fifo_config.temporal_scheme != "steady":
            raise ValueError("FIFO mode requires window_steps >= 2")
        self.integrator = integrator
        self.params = params
        self.logger = logger
        self.config = fifo_config
        self.window_core = window_core
        self.reference_gate = reference_gate
        self.field_layout: FieldLayout = resolve_fvconfig_field_layout(
            getattr(params, "fvconfig", {}) or {}
        )
        self.interior_indices = integrator.mask_interior_cell.nonzero(
            as_tuple=True
        )[0]
        self.phi_committed_cpd_cell = phi_initial_cpd_cell.detach().clone()
        self.delta_phi_interior = initialize_window_delta(
            fifo_config.window_steps,
            self.interior_indices.numel(),
            self.phi_committed_cpd_cell.shape[-1],
            device=self.phi_committed_cpd_cell.device,
            dtype=self.phi_committed_cpd_cell.dtype,
            init_std=fifo_config.tail_init_std,
            layout=self.field_layout,
        )
        self.optimizer = self._make_optimizer()
        self.optimizer_updates = 0
        self.residual_record_index = 0
        self.popped_steps = 0
        self.head_iterations = 0
        self.convergence_streak = 0
        self.queue_initial_residuals: List[Dict[str, float] | None] = [
            None
        ] * fifo_config.window_steps
        self.queue_update_counts = [0] * fifo_config.window_steps
        self.time_loss_weights = fifo_time_loss_weights(
            fifo_config,
            device=self.delta_phi_interior.device,
            dtype=self.delta_phi_interior.dtype,
        )
        self.start_time = time.time()
        self.last_evaluation: WindowEvaluation | None = None
        self.last_reference_result: Any | None = None
        self.dt_value = float(self.params.fvconfig["theta_pde"]["dt"]["val"])
        self.committed_ddt_interior = self.delta_phi_interior.new_zeros(
            (
                self.interior_indices.numel(),
                self.field_layout.num_time_derivative_fields,
            )
        )
        self.phi_committed_prev_cpd_cell = self.phi_committed_cpd_cell.clone()
        self.cn_history_valid = torch.zeros(
            (), dtype=torch.bool, device=self.delta_phi_interior.device
        )
        self._set_learning_rate()
        self.pseudo_time_scale: torch.Tensor | None = None
        self._refresh_pseudo_time_scale()
        if self.delta_phi_interior.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.delta_phi_interior.device)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _make_optimizer()"
        )

    def _set_learning_rate(self) -> float | None:
        return None

    def _log_queue_shift(self) -> None:
        pass

    def _refresh_pseudo_time_scale(self) -> None:
        if not self.config.pseudo_time_enabled:
            return
        self.pseudo_time_scale = local_pseudo_time_scale(
            self.integrator,
            self.phi_committed_cpd_cell,
            relax_momentum=self.config.pseudo_relax_momentum,
            relax_energy=self.config.pseudo_relax_energy,
        )
        scale = self.pseudo_time_scale
        self.logger.info(
            "Pseudo-time (SIMPLE relax U=%.3g T=%.3g): dt/dtau_P "
            "U in [%.3e, %.3e], T in [%.3e, %.3e]",
            self.config.pseudo_relax_momentum,
            self.config.pseudo_relax_energy,
            float(scale[:, 0].min()), float(scale[:, 0].max()),
            float(scale[:, 3].min()), float(scale[:, 3].max()),
        )

    def _evaluate(self, *, compiled: bool) -> WindowEvaluation:
        use_core = compiled and self.config.temporal_scheme != "steady"
        return evaluate_time_window(
            integrator=self.integrator,
            phi_window_start_cpd_cell=self.phi_committed_cpd_cell,
            delta_phi_interior=self.delta_phi_interior,
            interior_indices=self.interior_indices,
            core_evaluator=self.window_core if use_core else None,
            time_loss_weights=self.time_loss_weights,
            dt_value=self.dt_value,
            crank_nicolson_off_centering=self.config.crank_nicolson_off_centering,
            rolling_with_grad=(self.config.rolling_scheme == "RollingWithGrad"),
            use_backward=(self.config.temporal_scheme == "backward"),
            use_steady=(self.config.temporal_scheme == "steady"),
            committed_ddt_interior=self.committed_ddt_interior,
            cn_history_valid=self.cn_history_valid,
            phi_prev2_cpd_cell=self.phi_committed_prev_cpd_cell,
            layout=self.field_layout,
            pseudo_time_scale=self.pseudo_time_scale,
        )

    def _commit_time_derivative(self, popped_state: torch.Tensor) -> None:
        """Advance the detached CN ddt seed once per accepted head."""
        old_interior = self.phi_committed_cpd_cell.index_select(0, self.interior_indices)
        new_interior = popped_state.index_select(0, self.interior_indices)
        q_head = (
            _time_derivative_fields(new_interior, self.field_layout)
            - _time_derivative_fields(old_interior, self.field_layout)
        ) / self.dt_value
        off_centering = self.config.crank_nicolson_off_centering
        valid = self.cn_history_valid.to(dtype=q_head.dtype)
        seed = (
            valid * self.committed_ddt_interior.to(dtype=q_head.dtype)
            + (1.0 - valid) * q_head
        )
        ddt_head = (1.0 + off_centering) * q_head - off_centering * seed
        with torch.no_grad():
            self.committed_ddt_interior.copy_(ddt_head.detach())
            self.cn_history_valid.fill_(True)

    def _initialize_new_snapshot_residuals(
        self, values_by_time: Mapping[str, torch.Tensor]
    ) -> None:
        for queue_index, initial in enumerate(self.queue_initial_residuals):
            if initial is None:
                self.queue_initial_residuals[queue_index] = residual_values_at(
                    values_by_time, queue_index
                )

    def _head_check(
        self, values_by_time: Mapping[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], ResidualCheck]:
        current = residual_values_at(values_by_time, 0)
        initial = self.queue_initial_residuals[0]
        if initial is None:
            raise RuntimeError("FIFO head has no stored initial residual")
        return current, evaluate_openfoam_residual_control(
            current=current,
            initial=initial,
            criteria=self.config.criteria,
        )

    def _shift_after_pop(self) -> None:
        if self.config.temporal_scheme == "steady":
            with torch.no_grad():
                self.delta_phi_interior.zero_()
            self.delta_phi_interior.grad = None
            self.queue_initial_residuals = [None] * len(self.queue_initial_residuals)
            self.head_iterations = 0
            self.convergence_streak = 0
            self.last_reference_result = None
            self._refresh_pseudo_time_scale()
            return
        shift_fifo_parameter_and_adam_state_(
            delta_phi_interior=self.delta_phi_interior,
            optimizer=self.optimizer,
            tail_init_std=self.config.tail_init_std,
            preserve_adam_state=self.config.preserve_adam_state,
            tail_extrapolation=self.config.tail_extrapolation,
        )
        self.queue_initial_residuals = self.queue_initial_residuals[1:] + [None]
        self.queue_update_counts = self.queue_update_counts[1:] + [0]
        self.head_iterations = 0
        self.convergence_streak = 0
        self.last_reference_result = None
        self._set_learning_rate()
        self._log_queue_shift()
