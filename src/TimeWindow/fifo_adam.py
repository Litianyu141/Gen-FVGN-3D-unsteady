"""Row-aged ``FIFOAdam`` optimizer and queue driver for FIFO-window ODIL."""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch

from TimeWindow.config import FIFOConfig, _require_channel_scales
from TimeWindow.fifo import FIFOWindowDriver
from TimeWindow.window import _FIELD_NAMES


def fifo_linear_learning_rate(config: FIFOConfig, head_iterations: int) -> float:
    """Linear LR schedule keyed to each snapshot's update age."""
    horizon = max(1, config.lr_decay_iterations)
    progress = min(max(head_iterations, 0) / horizon, 1.0)
    return config.learning_rate + (
        config.minimum_learning_rate - config.learning_rate
    ) * progress


class FIFOAdam(torch.optim.Optimizer):
    """Adam with one bias-correction age per FIFO time row."""

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        update_scale: Sequence[float] | None = None,
        field_names: Tuple[str, ...] = _FIELD_NAMES,
    ) -> None:
        if lr <= 0 or eps <= 0:
            raise ValueError("FIFOAdam lr and eps must be positive")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("FIFOAdam betas must satisfy 0 <= beta < 1")
        if update_scale is None:
            update_scale = (1.0,) * len(field_names)
        validated_scale = _require_channel_scales(
            "FIFOAdam.update_scale", update_scale, field_names
        )
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                update_scale=validated_scale,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("FIFOAdam does not support sparse gradients")
                if parameter.ndim < 1:
                    raise ValueError("FIFOAdam parameters must have a FIFO axis")
                state = self.state[parameter]
                if not state:
                    state["row_step"] = torch.zeros(
                        (parameter.shape[0],) + (1,) * (parameter.ndim - 1),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if parameter.shape[-1] != len(group["update_scale"]):
                        raise ValueError(
                            "FIFOAdam update_scale must match the Parameter's last axis, "
                            f"got {len(group['update_scale'])} and {parameter.shape[-1]}"
                        )
                    state["update_scale"] = parameter.new_tensor(
                        group["update_scale"]
                    ).reshape((1,) * (parameter.ndim - 1) + (-1,))

                row_step = state["row_step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                row_step.add_(1)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias_correction1 = 1.0 - torch.pow(beta1, row_step)
                bias_correction2 = 1.0 - torch.pow(beta2, row_step)
                denominator = (
                    exp_avg_sq.sqrt() / bias_correction2.sqrt()
                ).add_(group["eps"])
                update = exp_avg / denominator / bias_correction1
                row_lr = group.get("row_lr")
                if row_lr is None:
                    parameter.add_(update * state["update_scale"], alpha=-group["lr"])
                else:
                    parameter.sub_(update * state["update_scale"] * row_lr)
        return loss


class FIFOAdamWindowDriver(FIFOWindowDriver):
    """FIFO queue paired with ``FIFOAdam`` and per-snapshot LR scheduling."""

    def _make_optimizer(self) -> FIFOAdam:
        fifo_config = self.config
        optimizer = FIFOAdam(
            [self.delta_phi_interior],
            lr=fifo_config.learning_rate,
            betas=(fifo_config.adam_beta1, fifo_config.adam_beta2),
            eps=fifo_config.adam_eps,
            update_scale=fifo_config.adam_update_scale,
            field_names=self.field_layout.names,
        )
        self.logger.info(
            "FIFOAdam: lr=%.3e betas=(%.4g, %.4g) eps=%.3e; "
            "row-wise bias-correction age enabled; update_scale=%s",
            fifo_config.learning_rate,
            fifo_config.adam_beta1,
            fifo_config.adam_beta2,
            fifo_config.adam_eps,
            tuple(fifo_config.adam_update_scale),
        )
        return optimizer

    def _set_learning_rate(self) -> float:
        row_rates = [
            fifo_linear_learning_rate(self.config, age)
            for age in self.queue_update_counts
        ]
        row_lr = self.delta_phi_interior.new_tensor(row_rates).reshape(
            -1, *([1] * (self.delta_phi_interior.ndim - 1))
        )
        for group in self.optimizer.param_groups:
            group["row_lr"] = row_lr
            group["lr"] = row_rates[0]
        return row_rates[0]

    def _adam_row_age_metrics(self) -> Dict[str, float]:
        state = self.optimizer.state.get(self.delta_phi_interior, {})
        row_step = state.get("row_step")
        if not isinstance(row_step, torch.Tensor) or row_step.numel() == 0:
            return {
                "adam_head_row_step": 0.0,
                "adam_tail_row_step": 0.0,
            }
        ages = row_step.detach().reshape(row_step.shape[0], -1)[:, 0]
        return {
            "adam_head_row_step": float(ages[0].item()),
            "adam_tail_row_step": float(ages[-1].item()),
        }

    def _log_queue_shift(self) -> None:
        ages = self._adam_row_age_metrics()
        self.logger.info(
            "FIFO shift complete: new head Adam age=%.0f, fresh tail age=%.0f",
            ages["adam_head_row_step"],
            ages["adam_tail_row_step"],
        )
