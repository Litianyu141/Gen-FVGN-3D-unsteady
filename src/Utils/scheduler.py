"""Learning-rate schedulers for PCNO and ODIL optimization."""

import logging
import math

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

logger = logging.getLogger(__name__)


class CustomExponentialLR(_LRScheduler):
    """Exponential decay from ``lr_start`` to ``lr_min`` over ``total_epochs``."""

    def __init__(self, optimizer, lr_start, lr_min, decay_power, total_epochs, last_epoch=-1):
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.decay_power = decay_power
        self.total_epochs = max(1, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr_start
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.lr_start for _ in self.optimizer.param_groups]
        progress = min(self.last_epoch / self.total_epochs, 1.0)
        lr_cur = self.lr_min + (self.lr_start - self.lr_min) * math.exp(-self.decay_power * progress)
        return [lr_cur for _ in self.optimizer.param_groups]


class PolynomialDecayLR(_LRScheduler):
    """Polynomial decay from ``lr_start`` to ``lr_min`` over ``total_epochs``."""

    def __init__(self, optimizer, lr_start, lr_min, concavity, total_epochs, last_epoch=-1):
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.concavity = concavity
        self.total_epochs = max(1, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr_start
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.lr_start for _ in self.optimizer.param_groups]
        progress = min(self.last_epoch / self.total_epochs, 1.0)
        decay_factor = progress if self.concavity == 1.0 else 1.0 - ((1.0 - progress) ** (1.0 / self.concavity))
        lr_cur = self.lr_min + (self.lr_start - self.lr_min) * (1.0 - decay_factor)
        return [lr_cur for _ in self.optimizer.param_groups]


class LinearDecayLR(_LRScheduler):
    """Linear decay from ``lr_start`` down to ``lr_min`` over ``total_epochs``."""

    def __init__(self, optimizer, lr_start, lr_min, total_epochs, last_epoch=-1):
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.total_epochs = max(1, total_epochs)
        self.slope = -(lr_start - lr_min) / self.total_epochs
        for group in optimizer.param_groups:
            group["lr"] = lr_start
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        lr_cur = max(self.lr_min, min(self.lr_start, self.lr_start + self.slope * step))
        return [lr_cur for _ in self.optimizer.param_groups]


def get_scheduler(scheduler_type, optimizer, *args, **kwargs):
    """Instantiate a learning-rate scheduler by name."""
    del args
    if not kwargs.get("use_lr_scheduler", True):
        return None

    scheduler_type = str(scheduler_type).lower()

    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("lr_factor", 0.5),
            patience=kwargs.get("lr_patience", 500),
            threshold=kwargs.get("lr_threshold", 1e-6),
            threshold_mode=kwargs.get("threshold_mode", "rel"),
            min_lr=kwargs.get("lr_min_lr", 1e-6),
        )
    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("lr_step_size", 1000),
            gamma=kwargs.get("lr_factor", 0.5),
        )
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("lr_T_max", kwargs.get("n_epochs", 40000)),
            eta_min=kwargs.get("lr_min_lr", kwargs.get("lr_min", 1e-5)),
        )
    if scheduler_type == "cosine_restart":
        eta_max = kwargs.get("lr_max_lr", 1e-2)
        for group in optimizer.param_groups:
            group["lr"] = eta_max
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("lr_T_0", 2000),
            T_mult=kwargs.get("lr_T_mult", 1),
            eta_min=kwargs.get("lr_min_lr", 5e-5),
        )
    if scheduler_type == "exponential":
        return ExponentialLR(optimizer, gamma=kwargs.get("lr_gamma", 0.999))
    if scheduler_type == "custom_exponential":
        return CustomExponentialLR(
            optimizer,
            lr_start=kwargs.get("lr_start", 1.0),
            lr_min=kwargs.get("lr_min", 1e-5),
            decay_power=kwargs.get("decay_power", 2.0),
            total_epochs=kwargs.get("n_epochs", 40000),
        )
    if scheduler_type == "polynomial_decay":
        return PolynomialDecayLR(
            optimizer,
            lr_start=kwargs.get("lr_start", 1.0),
            lr_min=kwargs.get("lr_min", 1e-5),
            concavity=kwargs.get("concavity", 2.0),
            total_epochs=kwargs.get("n_epochs", 40000),
        )
    if scheduler_type in ("linear_decay", "linear"):
        return LinearDecayLR(
            optimizer,
            lr_start=kwargs.get("lr_start", 1.0),
            lr_min=kwargs.get("lr_min", 1e-5),
            total_epochs=kwargs.get("n_epochs", 40000),
        )

    raise ValueError(
        f"Unsupported scheduler_type={scheduler_type!r}; expected one of "
        "['plateau', 'step', 'cosine', 'cosine_restart', 'exponential', "
        "'custom_exponential', 'polynomial_decay', 'linear_decay']"
    )
