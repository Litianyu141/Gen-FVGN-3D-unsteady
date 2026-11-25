"""PyTorch runtime environment setup for FIFO-window ODIL runs.

Call ``setup_torch_environment()`` once before constructing any trainer.
"""
import warnings


def setup_torch_environment() -> None:
    warnings.filterwarnings(
        "ignore", message=".*Sparse CSR tensor support is in beta state.*"
    )
    import torch._functorch.config as _functorch_config

    _functorch_config.donated_buffer = False
