"""Factory and explicit-time-axis wrapper for Warp FVM solvers."""

from typing import Any, Iterator, Literal, Tuple

import torch

from FVsolver.applications import create_solver
from FVsolver.applications.base_solver import BaseSolver

GraphReduction = Literal["mean", "sum"]


def sum_time_loss(loss_by_time: torch.Tensor, *, graph_reduction: GraphReduction = "mean") -> torch.Tensor:
    """Sum physical time ``T`` and reduce graph batch ``B`` from ``[T, B]`` loss tensor."""
    if not isinstance(loss_by_time, torch.Tensor) or loss_by_time.ndim != 2:
        raise ValueError(f"loss_by_time must be a [T, B] tensor, got {type(loss_by_time).__name__}")
    if loss_by_time.shape[0] < 1 or loss_by_time.shape[1] < 1:
        raise ValueError(f"Empty loss_by_time shape {tuple(loss_by_time.shape)}")
    loss_per_graph = loss_by_time.sum(dim=0)
    if graph_reduction == "mean":
        return loss_per_graph.mean()
    if graph_reduction == "sum":
        return loss_per_graph.sum()
    raise ValueError(f"graph_reduction must be 'mean' or 'sum', got {graph_reduction!r}")


class Integrator:
    """Thin wrapper around the active Warp FVM solver preserving ``[T, B]`` loss semantics."""

    def __init__(self, params: Any) -> None:
        self.params = params
        self.solver_type = params.fvconfig["solver"]["type"]
        self._solver: BaseSolver = create_solver(self.solver_type, params)
        self._last_time_steps = 1
        self._last_input_ndim = 2

    def register_properties(self, fv_graph: Any, force_register: bool = False) -> None:
        self._solver.register_properties(fv_graph, force_register=force_register)

    def _validate_state_pair(self, phi_old: torch.Tensor, phi_new: torch.Tensor) -> None:
        if phi_old.ndim not in (2, 3) or phi_new.ndim != phi_old.ndim:
            raise ValueError(f"States must share rank 2 or 3, got {tuple(phi_old.shape)} and {tuple(phi_new.shape)}")
        if phi_old.shape != phi_new.shape:
            raise ValueError(f"State shape mismatch: {tuple(phi_old.shape)} vs {tuple(phi_new.shape)}")
        if phi_new.ndim == 3 and phi_new.shape[0] < 1:
            raise ValueError("Time axis T must be >= 1")

    def _normalize_loss_shape(self, loss_per_graph: torch.Tensor) -> torch.Tensor:
        B = int(self._solver.num_graphs)
        if self._last_input_ndim == 2:
            if tuple(loss_per_graph.shape) != (B,):
                raise RuntimeError(f"Expected loss shape ({B},), got {tuple(loss_per_graph.shape)}")
            return loss_per_graph.unsqueeze(0)
        expected = (self._last_time_steps, B)
        if tuple(loss_per_graph.shape) != expected:
            raise RuntimeError(f"Expected loss shape {expected}, got {tuple(loss_per_graph.shape)}")
        return loss_per_graph

    def __call__(
        self,
        phi_old_cpd_cell: torch.Tensor,
        phi_new_cpd_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_state_pair(phi_old_cpd_cell, phi_new_cpd_cell)
        self._last_input_ndim = phi_new_cpd_cell.ndim
        self._last_time_steps = int(phi_new_cpd_cell.shape[0]) if phi_new_cpd_cell.ndim == 3 else 1
        loss_per_graph, state = self._solver(phi_old_cpd_cell, phi_new_cpd_cell)
        return self._normalize_loss_shape(loss_per_graph), state

    def enforce_bcs_only(
        self,
        phi_old_cpd_cell: torch.Tensor,
        phi_new_cpd_cell: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_state_pair(phi_old_cpd_cell, phi_new_cpd_cell)
        return self._solver.enforce_bcs_only(phi_old_cpd_cell, phi_new_cpd_cell)

    def _component_with_time_axis(self, component: Any) -> Any:
        if not isinstance(component, torch.Tensor):
            return component
        if self._last_input_ndim == 2:
            return component.unsqueeze(0)
        return component

    def get_separated_losses(self) -> Tuple[Any, ...]:
        components = self._solver.get_separated_losses()
        if components is None:
            return ()
        return tuple(self._component_with_time_axis(v) for v in components)

    def iter_named_residuals(self) -> Iterator[Tuple[str, Any]]:
        for name, comp in self._solver.iter_named_residuals():
            yield name, self._component_with_time_axis(comp)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self._solver, name)
