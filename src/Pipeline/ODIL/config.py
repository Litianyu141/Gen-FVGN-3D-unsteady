"""CLI configuration for FIFO-window ODIL runs."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from TimeWindow.config import add_fifo_window_arguments

OPTIMIZERS = ("adam",)


def build_odil_parser() -> argparse.ArgumentParser:
    """Build the FIFO-window ODIL command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "ODIL entry: FIFO-windowed direct optimization of the unsteady "
            "Warp FVM residual."
        )
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        default="adam",
        help="First-order optimizer advancing the FIFO time window.",
    )
    add_fifo_window_arguments(parser)
    return parser


_OPTIMIZER_DYNAMIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "print_freq": {"adam": 250},
}


def resolve_optimizer_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill per-optimizer dynamic defaults left as ``None`` by ``build_odil_parser``."""
    for dest, per_optimizer in _OPTIMIZER_DYNAMIC_DEFAULTS.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, per_optimizer[args.optimizer])
    return args
