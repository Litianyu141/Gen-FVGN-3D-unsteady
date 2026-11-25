#!/usr/bin/env python3
"""FIFO-windowed ODIL entry: direct Warp FVM residual optimization without a network."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Sequence

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _make_sync_free_adam_trainer_cls(base_cls: type) -> type:
    """Subclass ``FIFOWindowedPDETrainer`` to avoid GPU-CPU syncs between checkpoints."""
    import torch
    from TimeWindow.fifo_adam import fifo_linear_learning_rate
    from TimeWindow.residuals import residual_control_values_by_time

    class _SyncFreeFIFOAdamTrainer(base_cls):
        def __init__(self, **kwargs: Any) -> None:
            self._ages_gpu: torch.Tensor | None = None
            super().__init__(**kwargs)
            self._sync_ages_gpu()
            self._set_learning_rate()

        def _sync_ages_gpu(self) -> None:
            shape = (-1, *([1] * (self.delta_phi_interior.ndim - 1)))
            self._ages_gpu = torch.tensor(
                self.queue_update_counts,
                device=self.delta_phi_interior.device,
                dtype=self.delta_phi_interior.dtype,
            ).reshape(shape)

        def _set_learning_rate(self) -> float:
            head_lr = fifo_linear_learning_rate(self.config, self.queue_update_counts[0])
            if self._ages_gpu is None:
                return super()._set_learning_rate()
            horizon = float(max(1, self.config.lr_decay_iterations))
            progress = (self._ages_gpu / horizon).clamp(0.0, 1.0)
            row_lr = self.config.learning_rate + (
                self.config.minimum_learning_rate - self.config.learning_rate
            ) * progress
            for group in self.optimizer.param_groups:
                group["row_lr"] = row_lr
                group["lr"] = head_lr
            return head_lr

        def _shift_after_pop(self) -> None:
            super()._shift_after_pop()
            if self.config.temporal_scheme != "steady":
                self._sync_ages_gpu()
                self._set_learning_rate()

        def _fast_loss_only(self) -> torch.Tensor:
            if (
                self.window_core is not None
                and self.config.temporal_scheme != "steady"
            ):
                core_outputs = self.window_core(
                    self.phi_committed_cpd_cell,
                    self.delta_phi_interior,
                    self.interior_indices,
                    self.committed_ddt_interior,
                    self.cn_history_valid,
                    self.phi_committed_prev_cpd_cell,
                )
                loss_by_time = core_outputs[1]
                return (
                    loss_by_time
                    * self.time_loss_weights.to(dtype=loss_by_time.dtype)[:, None]
                ).sum(dim=0).mean()
            return self._evaluate(compiled=False).loss

        def run(
            self,
            *,
            total_physical_steps: int,
            on_pop: Callable[[int, torch.Tensor], None] | None = None,
        ) -> torch.Tensor:
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
            check_iv = self.config.check_interval
            print_iv = self.params.print_freq
            max_iter = self.config.max_head_iterations
            min_head = self.config.min_head_iterations
            min_snap = self.config.min_snapshot_updates
            clip_norm = self.config.grad_clip_norm

            while self.popped_steps < total_physical_steps:
                need_init = any(x is None for x in self.queue_initial_residuals)
                eligible = (
                    self.head_iterations >= min_head
                    and self.queue_update_counts[0] >= min_snap
                    and self.head_iterations % check_iv == 0
                )
                should_record = (
                    self.head_iterations < 5
                    or self.head_iterations % check_iv == 0
                    or self.head_iterations == max_iter
                )
                is_checkpoint = (
                    self.head_iterations < 5
                    or self.head_iterations % print_iv == 0
                    or self.head_iterations == max_iter
                )
                at_budget = self.head_iterations >= max_iter

                if not (need_init or eligible or should_record or is_checkpoint or at_budget):
                    self.optimizer.zero_grad(set_to_none=True)
                    scalar_loss = self._fast_loss_only()
                    scalar_loss.backward()
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [self.delta_phi_interior], clip_norm
                        )
                    self.optimizer.step()
                    self.optimizer_updates += 1
                    self.head_iterations += 1
                    self.queue_update_counts = [c + 1 for c in self.queue_update_counts]
                    if self._ages_gpu is not None:
                        self._ages_gpu.add_(1.0)
                    self._set_learning_rate()
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                evaluation = self._evaluate(compiled=True)
                if not torch.isfinite(evaluation.loss):
                    raise FloatingPointError("Non-finite FIFO ODIL loss")
                values_by_time = residual_control_values_by_time(
                    evaluation, self.config.criteria
                )
                self._initialize_new_snapshot_residuals(values_by_time)
                _, check = self._head_check(values_by_time)
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
                if should_record or accepted:
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

                if self.head_iterations >= max_iter:
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
                        f"{max_iter} optimizer updates at "
                        f"physical step {self.popped_steps + 1}: {detail}"
                    )

                evaluation.loss.backward()
                grad = self.delta_phi_interior.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise FloatingPointError("FIFO delta has a missing or non-finite gradient")
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [self.delta_phi_interior], clip_norm
                    )
                self.optimizer.step()
                self.optimizer_updates += 1
                self.head_iterations += 1
                self.queue_update_counts = [c + 1 for c in self.queue_update_counts]
                if self._ages_gpu is not None:
                    self._ages_gpu.add_(1.0)
                self._set_learning_rate()

            self.logger.info(
                "FIFO ODIL complete: popped %d physical snapshots with %d Adam updates",
                self.popped_steps,
                self.optimizer_updates,
            )
            return self.phi_committed_cpd_cell

    _SyncFreeFIFOAdamTrainer.__name__ = base_cls.__name__
    return _SyncFreeFIFOAdamTrainer


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    from TimeWindow.torch_env import setup_torch_environment

    setup_torch_environment()

    from Pipeline.ODIL import trainer_class
    from Pipeline.ODIL.config import build_odil_parser, resolve_optimizer_defaults
    from TimeWindow.config import format_effective_config
    from TimeWindow.session import FIFOWindowSession

    parser = build_odil_parser()
    args = resolve_optimizer_defaults(parser.parse_args(argv_list))

    session = FIFOWindowSession(args, argv_list)
    fifo_config = session.resolve_fifo_config()
    logger = session.open_logger(
        logger_subdir="ODIL-Adam",
        execution_tag="FIFO-s1-warp",
        iterations=fifo_config.max_head_iterations,
    )
    logger.info(format_effective_config(argv_list, fifo_config))
    logger.info("Experiment id: %s", session.run_directory)
    logger.info(
        "Residual monitor: %s (flush every %d rows)",
        logger.residual_csv_path,
        args.csv_flush_rows,
    )

    session.load_case()
    base_cls = trainer_class(args.optimizer)
    cls = (
        _make_sync_free_adam_trainer_cls(base_cls)
        if args.optimizer == "adam"
        else base_cls
    )
    trainer = cls(**session.trainer_kwargs())
    session.announce()
    return session.run(trainer)


if __name__ == "__main__":
    raise SystemExit(main())
