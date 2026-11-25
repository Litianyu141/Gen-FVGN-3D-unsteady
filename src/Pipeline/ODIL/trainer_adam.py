"""ODIL-Adam trainers: FIFO receding-horizon and fixed-slab PDE optimization."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Tuple

import torch

from TimeWindow.config import ResidualCheck
from TimeWindow.fifo_adam import FIFOAdamWindowDriver
from TimeWindow.residuals import residual_control_values_by_time
from TimeWindow.window import (
    WindowCore,
    WindowEvaluation,
    _FIELD_NAMES,
    evaluate_time_window,
    initialize_window_delta,
)


class WindowedPDETrainer:
    """Optimize consecutive physical steps as coupled fixed-slab ODIL windows."""

    def __init__(
        self,
        *,
        phi_initial_cpd_cell: torch.Tensor,
        integrator: Any,
        params: Any,
        logger: Any,
        window_steps: int,
        delta_init_std: float,
        window_core: WindowCore | None = None,
    ) -> None:
        self.integrator = integrator
        self.params = params
        self.logger = logger
        self.window_steps = window_steps
        self.delta_init_std = delta_init_std
        self.window_core = window_core
        self.interior_indices = integrator.mask_interior_cell.nonzero(
            as_tuple=True
        )[0]
        self.phi_committed_cpd_cell = phi_initial_cpd_cell.detach().clone()
        self.last_evaluation: WindowEvaluation | None = None
        self.optimizer_updates = 0
        self.residual_record_index = 0
        self.start_time = time.time()

    def _make_optimizer_and_scheduler(
        self, delta_phi_interior: torch.nn.Parameter
    ) -> Tuple[Any, Any]:
        from Utils.OptimizerF import get_optimizer
        from Utils.scheduler import get_scheduler

        optimizer = get_optimizer(
            self.params.optimizer_type, [delta_phi_interior], self.params
        )
        scheduler = get_scheduler(
            self.params.scheduler_type,
            optimizer,
            lr_start=self.params.lr,
            lr_min=min(5.0e-6, self.params.lr),
            n_epochs=self.params.n_epochs,
        )
        return optimizer, scheduler

    @staticmethod
    def _metric_value(logs: Mapping[str, torch.Tensor], name: str) -> float:
        metric = logs.get(name)
        return float(metric.item()) if metric is not None else float("nan")

    def _residual_row(
        self,
        *,
        record_type: str,
        window_start: int,
        active_steps: int,
        iteration_in_window: int,
        evaluation: WindowEvaluation,
        learning_rate: float,
        delta_phi_interior: torch.Tensor,
    ) -> Dict[str, Any]:
        logs = evaluation.log10_components
        delta_abs_max = delta_phi_interior.detach().abs().amax(dim=(0, 1))
        final_interior = evaluation.states[-1].detach()[self.interior_indices]
        loss_by_time = evaluation.loss_by_time.detach().reshape(-1)
        state_device = evaluation.states[-1].device
        if state_device.type == "cuda":
            gib = float(1024**3)
            cuda_metrics = {
                "cuda_memory_allocated_gib": float(
                    torch.cuda.memory_allocated(state_device) / gib
                ),
                "cuda_memory_reserved_gib": float(
                    torch.cuda.memory_reserved(state_device) / gib
                ),
                "cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(state_device) / gib
                ),
                "cuda_peak_reserved_gib": float(
                    torch.cuda.max_memory_reserved(state_device) / gib
                ),
            }
        else:
            cuda_metrics = {
                "cuda_memory_allocated_gib": 0.0,
                "cuda_memory_reserved_gib": 0.0,
                "cuda_peak_allocated_gib": 0.0,
                "cuda_peak_reserved_gib": 0.0,
            }
        row: Dict[str, Any] = {
            "iteration": int(self.residual_record_index),
            "record_type": record_type,
            "optimizer_updates_total": int(self.optimizer_updates),
            "window_index": int(window_start // self.window_steps + 1),
            "window_start_step": int(window_start + 1),
            "window_end_step": int(window_start + active_steps),
            "window_active_steps": int(active_steps),
            "iteration_in_window": int(iteration_in_window),
            "loss_total_ln": float(evaluation.loss.detach().item()),
            "loss_time_mean_ln": float(loss_by_time.mean().item()),
            "loss_time_worst_ln": float(loss_by_time.max().item()),
            "loss_cont_log10": self._metric_value(logs, "cont"),
            "loss_cont_worst_log10": self._metric_value(logs, "cont_window_worst"),
            "loss_mom_log10": self._metric_value(logs, "mom"),
            "loss_mom_worst_log10": self._metric_value(logs, "mom_window_worst"),
            "loss_energy_log10": self._metric_value(logs, "energy"),
            "loss_energy_worst_log10": self._metric_value(logs, "energy_window_worst"),
            "loss_phi_jump_log10": self._metric_value(logs, "phi_jump"),
            "loss_phi_jump_worst_log10": self._metric_value(logs, "phi_jump_window_worst"),
            "loss_p_rghRef_log10": self._metric_value(logs, "p_rghRef"),
            "learning_rate": float(learning_rate),
            "delta_u_abs_max": float(delta_abs_max[0].item()),
            "delta_v_abs_max": float(delta_abs_max[1].item()),
            "delta_w_abs_max": float(delta_abs_max[2].item()),
            "delta_p_rgh_abs_max": float(delta_abs_max[3].item()),
            "delta_T_abs_max": float(delta_abs_max[4].item()),
            "state_U_abs_max": float(final_interior[:, :3].abs().max().item()),
            "state_p_rgh_min": float(final_interior[:, 3].min().item()),
            "state_p_rgh_max": float(final_interior[:, 3].max().item()),
            "state_T_min": float(final_interior[:, 4].min().item()),
            "state_T_max": float(final_interior[:, 4].max().item()),
            "elapsed_seconds": float(time.time() - self.start_time),
            **cuda_metrics,
        }
        self.residual_record_index += 1
        return row

    def _record_evaluation(
        self,
        *,
        record_type: str,
        window_start: int,
        active_steps: int,
        iteration_in_window: int,
        evaluation: WindowEvaluation,
        learning_rate: float,
        delta_phi_interior: torch.Tensor,
    ) -> None:
        row = self._residual_row(
            record_type=record_type,
            window_start=window_start,
            active_steps=active_steps,
            iteration_in_window=iteration_in_window,
            evaluation=evaluation,
            learning_rate=learning_rate,
            delta_phi_interior=delta_phi_interior,
        )
        self.logger.log_residuals(**row)

    def _log_iteration(
        self,
        *,
        window_start: int,
        active_steps: int,
        iteration: int,
        evaluation: WindowEvaluation,
        learning_rate: float,
        delta_phi_interior: torch.Tensor,
    ) -> None:
        self._record_evaluation(
            record_type="pre_step",
            window_start=window_start,
            active_steps=active_steps,
            iteration_in_window=iteration + 1,
            evaluation=evaluation,
            learning_rate=learning_rate,
            delta_phi_interior=delta_phi_interior,
        )

        is_checkpoint = (
            iteration < 5
            or (iteration + 1) % self.params.print_freq == 0
            or iteration + 1 == self.params.n_epochs
        )
        if not is_checkpoint:
            return

        logs = evaluation.log10_components
        delta_abs_max = delta_phi_interior.detach().abs().amax(dim=(0, 1))
        mean_text = " ".join(
            f"{name}={float(logs[name].item()):+.3f}"
            for name in ("cont", "mom", "energy")
            if name in logs
        )
        worst_text = " ".join(
            f"{name}={float(logs[f'{name}_window_worst'].item()):+.3f}"
            for name in ("cont", "mom", "energy")
            if f"{name}_window_worst" in logs
        )
        delta_text = ", ".join(
            f"d{name}={float(value.item()):.2e}"
            for name, value in zip(_FIELD_NAMES, delta_abs_max)
        )
        self.logger.info(
            "ODIL window %04d:%04d iter %5d | mean log10[%s] | "
            "worst log10[%s] | %s | lr=%.2e",
            window_start + 1,
            window_start + active_steps,
            iteration + 1,
            mean_text,
            worst_text,
            delta_text,
            learning_rate,
        )

    def optimize_window(
        self, *, window_start: int, active_steps: int
    ) -> Tuple[torch.Tensor, ...]:
        """Jointly optimize one slab and commit its final BC-enforced state."""
        delta_phi_interior = initialize_window_delta(
            active_steps,
            self.interior_indices.numel(),
            self.phi_committed_cpd_cell.shape[1],
            device=self.phi_committed_cpd_cell.device,
            dtype=self.phi_committed_cpd_cell.dtype,
            init_std=self.delta_init_std,
        )
        if delta_phi_interior.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(delta_phi_interior.device)
        optimizer, scheduler = self._make_optimizer_and_scheduler(delta_phi_interior)

        self.logger.info(
            "ODIL time slab %d:%d | Parameter shape=%s",
            window_start + 1,
            window_start + active_steps,
            tuple(delta_phi_interior.shape),
        )
        completed_iterations = 0
        current_lr = float(optimizer.param_groups[0]["lr"])
        for iteration in range(self.params.n_epochs):
            optimizer.zero_grad()
            evaluation = evaluate_time_window(
                integrator=self.integrator,
                phi_window_start_cpd_cell=self.phi_committed_cpd_cell,
                delta_phi_interior=delta_phi_interior,
                interior_indices=self.interior_indices,
                core_evaluator=self.window_core,
            )
            if not torch.isfinite(evaluation.loss):
                raise FloatingPointError(
                    f"Non-finite ODIL loss in window starting at step "
                    f"{window_start + 1}, iteration {iteration + 1}"
                )

            evaluation.loss.backward()
            grad = delta_phi_interior.grad
            if grad is None or not torch.isfinite(grad).all():
                raise FloatingPointError(
                    "ODIL window delta has a missing or non-finite gradient"
                )

            if self.params.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [delta_phi_interior], self.params.grad_clip_norm
                )
            current_lr = float(optimizer.param_groups[0]["lr"])
            self._log_iteration(
                window_start=window_start,
                active_steps=active_steps,
                iteration=iteration,
                evaluation=evaluation,
                learning_rate=current_lr,
                delta_phi_interior=delta_phi_interior,
            )

            optimizer.step()
            completed_iterations = iteration + 1
            self.optimizer_updates += 1
            mom_log10 = evaluation.log10_components.get("mom")
            if self.params.scheduler_type.lower() == "plateau":
                if mom_log10 is None:
                    raise RuntimeError("Plateau scheduler requires the momentum residual")
                scheduler.step(float(mom_log10.item()))
            else:
                scheduler.step()

            worst_metrics = [
                evaluation.log10_components.get(f"{name}_window_worst")
                for name in ("cont", "mom", "energy")
            ]
            if all(metric is not None for metric in worst_metrics) and all(
                float(metric.item()) <= self.params.early_stop_threshold
                for metric in worst_metrics
                if metric is not None
            ):
                self.logger.info(
                    "Early stop: all worst-step cont/mom/energy log10 residuals reached %.3f",
                    self.params.early_stop_threshold,
                )
                break

        with torch.no_grad():
            final_evaluation = evaluate_time_window(
                integrator=self.integrator,
                phi_window_start_cpd_cell=self.phi_committed_cpd_cell,
                delta_phi_interior=delta_phi_interior,
                interior_indices=self.interior_indices,
            )
        if not torch.isfinite(final_evaluation.loss):
            raise FloatingPointError("Optimizer produced a non-finite final ODIL window loss")
        if any(not torch.isfinite(state).all() for state in final_evaluation.states):
            raise FloatingPointError("Optimizer produced a non-finite final ODIL window state")
        self._record_evaluation(
            record_type="window_final",
            window_start=window_start,
            active_steps=active_steps,
            iteration_in_window=completed_iterations,
            evaluation=final_evaluation,
            learning_rate=current_lr,
            delta_phi_interior=delta_phi_interior,
        )
        final_logs = final_evaluation.log10_components
        self.logger.info(
            "ODIL window %04d:%04d final | mean log10[cont=%+.3f "
            "mom=%+.3f energy=%+.3f] | worst log10[cont=%+.3f "
            "mom=%+.3f energy=%+.3f]%s",
            window_start + 1,
            window_start + active_steps,
            self._metric_value(final_logs, "cont"),
            self._metric_value(final_logs, "mom"),
            self._metric_value(final_logs, "energy"),
            self._metric_value(final_logs, "cont_window_worst"),
            self._metric_value(final_logs, "mom_window_worst"),
            self._metric_value(final_logs, "energy_window_worst"),
            (
                " | CUDA peak allocated/reserved="
                f"{torch.cuda.max_memory_allocated(delta_phi_interior.device) / 1024**3:.2f}/"
                f"{torch.cuda.max_memory_reserved(delta_phi_interior.device) / 1024**3:.2f} GiB"
                if delta_phi_interior.device.type == "cuda"
                else ""
            ),
        )
        self.logger.finalize_residuals()
        final_states = tuple(state.detach().clone() for state in final_evaluation.states)
        self.phi_committed_cpd_cell = final_states[-1]
        self.last_evaluation = WindowEvaluation(
            loss=final_evaluation.loss.detach(),
            loss_by_time=final_evaluation.loss_by_time.detach(),
            states=final_states,
            log10_components={
                key: value.detach()
                for key, value in final_evaluation.log10_components.items()
            },
            residual_components={
                key: value.detach()
                for key, value in final_evaluation.residual_components.items()
            },
            residual_scales=final_evaluation.residual_scales,
        )
        return final_states


class FIFOWindowedPDETrainer(FIFOAdamWindowDriver):
    """Fixed-W receding-horizon ODIL with one-dt FIFO advancement."""

    @staticmethod
    def _metric_value(logs: Mapping[str, torch.Tensor], name: str) -> float:
        metric = logs.get(name)
        return float(metric.item()) if metric is not None else float("nan")

    def _cuda_metrics(self) -> Dict[str, float]:
        device = self.delta_phi_interior.device
        if device.type != "cuda":
            return {
                "cuda_memory_allocated_gib": 0.0,
                "cuda_memory_reserved_gib": 0.0,
                "cuda_peak_allocated_gib": 0.0,
                "cuda_peak_reserved_gib": 0.0,
            }
        gib = float(1024**3)
        return {
            "cuda_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / gib),
            "cuda_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / gib),
            "cuda_peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / gib),
            "cuda_peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / gib),
        }

    def _record(
        self,
        *,
        record_type: str,
        evaluation: WindowEvaluation,
        check: ResidualCheck,
        eligible: bool,
        accepted: bool,
        learning_rate: float,
        reference_result: Any | None = None,
    ) -> None:
        logs = evaluation.log10_components
        delta_abs_max = self.delta_phi_interior.detach().abs().amax(dim=(0, 1))
        head_interior = evaluation.states[0].detach()[self.interior_indices]
        loss_by_time = evaluation.loss_by_time.detach().reshape(-1)
        pressure = self.field_layout.pressure_index
        temperature = (
            self.field_layout.names.index("T")
            if self.field_layout.has_energy
            else None
        )
        row: Dict[str, Any] = {
            "iteration": self.residual_record_index,
            "record_type": record_type,
            "optimizer_updates_total": self.optimizer_updates,
            "physical_steps_popped": self.popped_steps,
            "queue_head_step": self.popped_steps + 1,
            "queue_tail_step": self.popped_steps + self.config.window_steps,
            "window_steps": self.config.window_steps,
            "fifo_stride": self.config.stride,
            "head_iterations": self.head_iterations,
            "head_snapshot_updates": self.queue_update_counts[0],
            "head_check_eligible": eligible,
            "head_all_fields_passed": check.passed,
            "head_accepted": accepted,
            "reference_checked": reference_result is not None,
            "reference_passed": (
                bool(reference_result.passed) if reference_result is not None else False
            ),
            "reference_u_nrmse": float("nan"),
            "reference_annulus_u_nrmse": float("nan"),
            "reference_annulus_u_error_p95": float("nan"),
            "reference_annulus_cell_count": 0,
            "reference_wake_u_nrmse": float("nan"),
            "reference_wake_cell_count": 0,
            "reference_p_nrmse": float("nan"),
            "reference_T_nrmse": float("nan"),
            "reference_u_correlation": float("nan"),
            "reference_p_correlation": float("nan"),
            "reference_T_correlation": float("nan"),
            "reference_w_rms": float("nan"),
            "reference_w_error_rms": float("nan"),
            "reference_T_oob_fraction": float("nan"),
            "reference_recirculation_error": float("nan"),
            "convergence_streak": self.convergence_streak,
            "consecutive_checks_required": self.config.consecutive_checks,
            "loss_total_ln": float(evaluation.loss.detach().item()),
            "loss_time_mean_ln": float(loss_by_time.mean().item()),
            "loss_time_worst_ln": float(loss_by_time.max().item()),
            "loss_cont_log10": self._metric_value(logs, "cont"),
            "loss_cont_worst_log10": self._metric_value(logs, "cont_window_worst"),
            "loss_mom_log10": self._metric_value(logs, "mom"),
            "loss_mom_worst_log10": self._metric_value(logs, "mom_window_worst"),
            "loss_energy_log10": self._metric_value(logs, "energy"),
            "loss_energy_worst_log10": self._metric_value(logs, "energy_window_worst"),
            "loss_phi_jump_log10": self._metric_value(logs, "phi_jump"),
            "loss_phi_jump_worst_log10": self._metric_value(logs, "phi_jump_window_worst"),
            "learning_rate": learning_rate,
            "delta_u_abs_max": float(delta_abs_max[0].item()),
            "delta_v_abs_max": float(delta_abs_max[1].item()),
            "delta_w_abs_max": float(delta_abs_max[2].item()),
            "delta_p_rgh_abs_max": float(delta_abs_max[pressure].item()),
            **(
                {"delta_T_abs_max": float(delta_abs_max[temperature].item())}
                if temperature is not None
                else {}
            ),
            "head_U_abs_max": float(head_interior[:, :3].abs().max().item()),
            "head_p_rgh_min": float(head_interior[:, pressure].min().item()),
            "head_p_rgh_max": float(head_interior[:, pressure].max().item()),
            **(
                {
                    "head_T_min": float(head_interior[:, temperature].min().item()),
                    "head_T_max": float(head_interior[:, temperature].max().item()),
                }
                if temperature is not None
                else {}
            ),
            "elapsed_seconds": float(time.time() - self.start_time),
            **self._adam_row_age_metrics(),
            **self._cuda_metrics(),
        }
        if reference_result is not None:
            row.update(
                reference_u_nrmse=reference_result.u_normalized_rmse,
                reference_annulus_u_nrmse=reference_result.annulus_u_normalized_rmse,
                reference_annulus_u_error_p95=reference_result.annulus_u_error_p95,
                reference_annulus_cell_count=reference_result.annulus_cell_count,
                reference_wake_u_nrmse=reference_result.wake_u_normalized_rmse,
                reference_wake_cell_count=reference_result.wake_cell_count,
                reference_p_nrmse=reference_result.p_normalized_rmse,
                reference_T_nrmse=reference_result.t_normalized_rmse,
                reference_u_correlation=reference_result.u_correlation,
                reference_p_correlation=reference_result.p_correlation,
                reference_T_correlation=reference_result.t_correlation,
                reference_w_rms=reference_result.w_rms,
                reference_w_error_rms=reference_result.w_error_rms,
                reference_T_oob_fraction=reference_result.temperature_out_of_bounds_fraction,
                reference_recirculation_error=reference_result.recirculation_length_difference,
            )
        criterion_by_name = {
            criterion.field_name: criterion for criterion in self.config.criteria
        }
        for field_name in criterion_by_name:
            prefix = f"head_{field_name}"
            row[f"{prefix}_residual"] = check.current[field_name]
            row[f"{prefix}_initial"] = check.initial[field_name]
            row[f"{prefix}_relative"] = check.relative[field_name]
            row[f"{prefix}_abs_passed"] = check.absolute_passed[field_name]
            row[f"{prefix}_rel_passed"] = check.relative_passed[field_name]
            row[f"{prefix}_passed"] = check.field_passed[field_name]
            row[f"{prefix}_tolerance"] = criterion_by_name[field_name].tolerance
            row[f"{prefix}_relTol"] = criterion_by_name[field_name].relative_tolerance
        self.logger.log_residuals(**row)
        self.residual_record_index += 1

    def _log_progress(
        self,
        *,
        check: ResidualCheck,
        accepted: bool,
        learning_rate: float,
    ) -> None:
        is_checkpoint = (
            self.head_iterations < 5
            or self.head_iterations % self.params.print_freq == 0
            or accepted
            or self.head_iterations == self.config.max_head_iterations
        )
        if not is_checkpoint:
            return
        residual_text = " ".join(
            f"{criterion.field_name}={check.current[criterion.field_name]:.3e}"
            f"/{check.relative[criterion.field_name]:.3e}"
            f"({'Y' if check.field_passed[criterion.field_name] else 'N'})"
            for criterion in self.config.criteria
        )
        self.logger.info(
            "FIFO head t%04d queue=%04d:%04d iter=%5d age=%5d | residual/relative "
            "[%s] | streak=%d/%d | lr=%.2e%s",
            self.popped_steps + 1,
            self.popped_steps + 1,
            self.popped_steps + self.config.window_steps,
            self.head_iterations,
            self.queue_update_counts[0],
            residual_text,
            self.convergence_streak,
            self.config.consecutive_checks,
            learning_rate,
            " | ACCEPT" if accepted else "",
        )

    def run(
        self,
        *,
        total_physical_steps: int,
        on_pop: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Optimize until ``total_physical_steps`` FIFO heads are accepted."""
        if total_physical_steps < 1:
            raise ValueError("total_physical_steps must be positive")
        self.logger.info(
            "FIFO ODIL start: target_steps=%d W=%d stride=1 max_head_iter=%d "
            "Parameter shape=%s",
            total_physical_steps,
            self.config.window_steps,
            self.config.max_head_iterations,
            tuple(self.delta_phi_interior.shape),
        )
        while self.popped_steps < total_physical_steps:
            self.optimizer.zero_grad(set_to_none=True)
            evaluation = self._evaluate(compiled=True)
            if not torch.isfinite(evaluation.loss):
                raise FloatingPointError("Non-finite FIFO ODIL loss")
            values_by_time = residual_control_values_by_time(
                evaluation, self.config.criteria
            )
            self._initialize_new_snapshot_residuals(values_by_time)
            _, check = self._head_check(values_by_time)
            eligible = (
                self.head_iterations >= self.config.min_head_iterations
                and self.queue_update_counts[0] >= self.config.min_snapshot_updates
                and self.head_iterations % self.config.check_interval == 0
            )
            if eligible:
                self.convergence_streak = (
                    self.convergence_streak + 1 if check.passed else 0
                )
            accepted = (
                eligible
                and check.passed
                and self.convergence_streak >= self.config.consecutive_checks
            )
            accepted_evaluation = evaluation
            accepted_check = check
            reference_result = None
            if accepted and self.config.eager_verify_on_pop:
                with torch.no_grad():
                    eager_evaluation = self._evaluate(compiled=False)
                eager_values = residual_control_values_by_time(
                    eager_evaluation, self.config.criteria
                )
                _, eager_check = self._head_check(eager_values)
                if eager_check.passed:
                    accepted_evaluation = eager_evaluation
                    accepted_check = eager_check
                else:
                    mismatch = "; ".join(
                        f"{criterion.field_name}: core="
                        f"{check.current[criterion.field_name]:.4e}"
                        f"(rel {check.relative[criterion.field_name]:.4f})"
                        f" verify={eager_check.current[criterion.field_name]:.4e}"
                        f"(rel {eager_check.relative[criterion.field_name]:.4f})"
                        f" tol={criterion.tolerance:.1e}"
                        f" relTol={criterion.relative_tolerance:.3g}"
                        for criterion in self.config.criteria
                    )
                    self.logger.warning(
                        "FIFO head passed but verification did not; continuing | %s",
                        mismatch,
                    )
                    accepted = False
                    self.convergence_streak = 0

            next_physical_step = self.popped_steps + 1
            should_check_reference = (
                self.reference_gate is not None
                and next_physical_step % self.config.reference_control.check_stride == 0
            )
            if accepted and should_check_reference:
                reference_result = self.reference_gate.evaluate(
                    accepted_evaluation.states[0][self.interior_indices],
                    next_physical_step * self.dt_value,
                )
                self.last_reference_result = reference_result
                if reference_result.passed:
                    self.logger.info(
                        "OpenFOAM gate t%04d PASS: U/p/T NRMSE="
                        "%.4g/%.4g/%.4g corr=%.4g/%.4g/%.4g "
                        "annulus U/p95=%.4g/%.4g wake U=%.4g "
                        "w/werr=%.3g/%.3g T_oob=%.3g recirc_err=%.4g",
                        next_physical_step,
                        reference_result.u_normalized_rmse,
                        reference_result.p_normalized_rmse,
                        reference_result.t_normalized_rmse,
                        reference_result.u_correlation,
                        reference_result.p_correlation,
                        reference_result.t_correlation,
                        reference_result.annulus_u_normalized_rmse,
                        reference_result.annulus_u_error_p95,
                        reference_result.wake_u_normalized_rmse,
                        reference_result.w_rms,
                        reference_result.w_error_rms,
                        reference_result.temperature_out_of_bounds_fraction,
                        reference_result.recirculation_length_difference,
                    )
                else:
                    accepted = False
                    self.convergence_streak = 0
                    self.logger.warning(
                        "OpenFOAM gate t%04d rejected residual-converged head: %s",
                        next_physical_step,
                        "; ".join(reference_result.failures),
                    )

            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            should_record = (
                self.head_iterations < 5
                or self.head_iterations % self.config.check_interval == 0
                or accepted
                or self.head_iterations == self.config.max_head_iterations
            )
            if should_record:
                self._record(
                    record_type="fifo_pop" if accepted else "pre_step",
                    evaluation=accepted_evaluation if accepted else evaluation,
                    check=accepted_check if accepted else check,
                    eligible=eligible,
                    accepted=accepted,
                    learning_rate=learning_rate,
                    reference_result=reference_result,
                )
            self._log_progress(
                check=accepted_check if accepted else check,
                accepted=accepted,
                learning_rate=learning_rate,
            )

            if accepted:
                popped_state = accepted_evaluation.states[0].detach().clone()
                if not torch.isfinite(popped_state).all():
                    raise FloatingPointError("Accepted FIFO state contains NaN/Inf")
                self._commit_time_derivative(popped_state)
                self.phi_committed_prev_cpd_cell = self.phi_committed_cpd_cell
                self.phi_committed_cpd_cell = popped_state
                self.popped_steps += 1
                self.last_evaluation = accepted_evaluation
                self.logger.info(
                    "FIFO POP t%04d after %d head updates; committed queue head",
                    self.popped_steps,
                    self.head_iterations,
                )
                if on_pop is not None:
                    on_pop(self.popped_steps, popped_state)
                self.logger.finalize_residuals()
                if self.popped_steps >= total_physical_steps:
                    break
                self._shift_after_pop()
                continue

            if self.head_iterations >= self.config.max_head_iterations:
                detail = ", ".join(
                    f"{criterion.field_name}={check.current[criterion.field_name]:.6e} "
                    f"(tol={criterion.tolerance:.1e}, "
                    f"relative={check.relative[criterion.field_name]:.6e}, "
                    f"relTol={criterion.relative_tolerance:.3g})"
                    for criterion in self.config.criteria
                )
                if (
                    self.last_reference_result is not None
                    and not self.last_reference_result.passed
                ):
                    detail += "; OpenFOAM gate: " + "; ".join(
                        self.last_reference_result.failures
                    )
                self.logger.error(
                    "FIFO convergence failure at t%04d after %d updates: %s",
                    self.popped_steps + 1,
                    self.head_iterations,
                    detail,
                )
                self.logger.finalize_residuals()
                raise RuntimeError(
                    "FIFO queue head did not satisfy residualControl within "
                    f"{self.config.max_head_iterations} optimizer updates at "
                    f"physical step {self.popped_steps + 1}: {detail}"
                )

            evaluation.loss.backward()
            grad = self.delta_phi_interior.grad
            if grad is None or not torch.isfinite(grad).all():
                raise FloatingPointError("FIFO delta has a missing or non-finite gradient")
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [self.delta_phi_interior], self.config.grad_clip_norm
                )
            self.optimizer.step()
            self.optimizer_updates += 1
            self.head_iterations += 1
            self.queue_update_counts = [count + 1 for count in self.queue_update_counts]
            self._set_learning_rate()

        self.logger.info(
            "FIFO ODIL complete: popped %d physical snapshots with %d Adam updates",
            self.popped_steps,
            self.optimizer_updates,
        )
        return self.phi_committed_cpd_cell
