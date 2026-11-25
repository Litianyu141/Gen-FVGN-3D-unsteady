"""ODIL: FIFO-windowed direct optimization of unsteady Warp FVM residuals."""

from Pipeline.ODIL.config import OPTIMIZERS


def trainer_class(optimizer: str) -> type:
    """Return the FIFO trainer class implementing ``--optimizer <optimizer>``."""
    if optimizer == "adam":
        from Pipeline.ODIL.trainer_adam import FIFOWindowedPDETrainer

        return FIFOWindowedPDETrainer
    raise ValueError(
        f"no ODIL trainer for --optimizer {optimizer!r}; implemented: {OPTIMIZERS}"
    )


__all__ = ["OPTIMIZERS", "trainer_class"]
