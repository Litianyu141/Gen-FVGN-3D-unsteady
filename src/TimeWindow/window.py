"""FIFO time-window construction: deltas, BC-consistent states, and Warp ddt cores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Tuple

import torch


@dataclass(frozen=True)
class FieldLayout:
    """State vector channel layout keyed by ``solver.type``."""

    names: Tuple[str, ...]
    time_derivative_indices: Tuple[int, ...]
    pressure_index: int
    pref_attr: str
    has_energy: bool

    @property
    def num_channels(self) -> int:
        return len(self.names)

    @property
    def num_time_derivative_fields(self) -> int:
        return len(self.time_derivative_indices)

    @property
    def scalar_transport_indices(self) -> Tuple[int, ...]:
        return tuple(index for index in self.time_derivative_indices if index >= 3)


FIELD_LAYOUTS: Dict[str, FieldLayout] = {
    "buoyantboussinesqpicoext": FieldLayout(
        names=("u", "v", "w", "p_rgh", "T"),
        time_derivative_indices=(0, 1, 2, 4),
        pressure_index=3,
        pref_attr="loss_p_rghRef_component",
        has_energy=True,
    ),
    "buoyantboussinesqpicocmpt": FieldLayout(
        names=("u", "v", "w", "p_rgh", "T"),
        time_derivative_indices=(0, 1, 2, 4),
        pressure_index=3,
        pref_attr="loss_p_rghRef_component",
        has_energy=True,
    ),
    "picoext": FieldLayout(
        names=("u", "v", "w", "p"),
        time_derivative_indices=(0, 1, 2),
        pressure_index=3,
        pref_attr="loss_pRef_component",
        has_energy=False,
    ),
    "picocmpt": FieldLayout(
        names=("u", "v", "w", "p"),
        time_derivative_indices=(0, 1, 2),
        pressure_index=3,
        pref_attr="loss_pRef_component",
        has_energy=False,
    ),
}


def resolve_field_layout(solver_type: str) -> FieldLayout:
    """Map a ``[solver] type`` string onto its ODIL field layout."""
    key = str(solver_type).lower()
    if key not in FIELD_LAYOUTS:
        raise ValueError(
            f"No ODIL field layout for solver.type={solver_type!r}; "
            f"known: {sorted(FIELD_LAYOUTS)}"
        )
    return FIELD_LAYOUTS[key]


_DEFAULT_FIELD_LAYOUT = FIELD_LAYOUTS["buoyantboussinesqpicoext"]
_FIELD_NAMES = _DEFAULT_FIELD_LAYOUT.names
_TIME_DERIVATIVE_FIELD_COUNT = _DEFAULT_FIELD_LAYOUT.num_time_derivative_fields
_LOG_EPS = 1.0e-30


def resolve_fvconfig_field_layout(fvconfig: Mapping[str, Any]) -> FieldLayout:
    """Return the field layout for a parsed ``fvconfig.toml``."""
    solver_type = (fvconfig.get("solver") or {}).get("type")
    if solver_type is None:
        return _DEFAULT_FIELD_LAYOUT
    return resolve_field_layout(solver_type)


def resolve_integrator_field_layout(integrator: Any) -> FieldLayout:
    """Return the field layout for a live integrator from its ``solver_type``."""
    solver_type = getattr(integrator, "solver_type", None)
    if solver_type is None:
        return _DEFAULT_FIELD_LAYOUT
    return resolve_field_layout(solver_type)


@dataclass(frozen=True)
class WindowEvaluation:
    """Differentiable loss, chained states, and detached display metrics."""

    loss: torch.Tensor
    loss_by_time: torch.Tensor
    states: Tuple[torch.Tensor, ...]
    log10_components: Mapping[str, torch.Tensor]
    residual_components: Mapping[str, torch.Tensor]
    residual_scales: Mapping[str, torch.Tensor | None] = field(default_factory=dict)


WindowCore = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, ...],
]


def time_window_slices(total_steps: int, window_steps: int) -> List[Tuple[int, int]]:
    """Return ``(zero_based_start, active_size)`` for all time slabs."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if window_steps < 1:
        raise ValueError(f"window_steps must be >= 1, got {window_steps}")
    return [
        (start, min(window_steps, total_steps - start))
        for start in range(0, total_steps, window_steps)
    ]


def initialize_window_delta(
    window_steps: int,
    num_interior_cells: int,
    num_channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    init_std: float,
    layout: FieldLayout | None = None,
) -> torch.nn.Parameter:
    """Create the interior-only ODIL unknown for one physical-time slab."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT
    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    if num_interior_cells < 1:
        raise ValueError("num_interior_cells must be positive")
    if num_channels != layout.num_channels:
        raise ValueError(
            f"The ODIL window expects {layout.num_channels} "
            f"{list(layout.names)} channels, got {num_channels}"
        )
    if init_std < 0:
        raise ValueError(f"init_std must be non-negative, got {init_std}")

    shape = (window_steps, num_interior_cells, num_channels)
    if init_std == 0:
        values = torch.zeros(shape, device=device, dtype=dtype)
    else:
        values = init_std * torch.randn(shape, device=device, dtype=dtype)
    return torch.nn.Parameter(values)


def _validate_combined_momentum_layout(
    integrator: Any, named_residuals: Mapping[str, torch.Tensor]
) -> None:
    """Verify the configured cell+face_cv residual exposes dim+1 channels."""
    params = getattr(integrator, "params", None)
    schemes = getattr(params, "fvschemes", {}) if params is not None else {}
    mode = str(schemes.get("momentumResidual", "cell")).lower()
    if mode not in ("cell+face_cv", "face_cv+cell"):
        return
    momentum = named_residuals.get("mom")
    if momentum is None:
        raise RuntimeError("Combined momentum mode did not expose a 'mom' residual")
    solver_config = getattr(params, "fvconfig", {}).get("solver", {})
    expected_channels = int(solver_config.get("dim", 3)) + 1
    if momentum.ndim != 3 or momentum.shape[-1] != expected_channels:
        raise RuntimeError(
            "cell+face_cv momentum residual must have [T,B,dim+1] shape; "
            f"expected {expected_channels} channels, got {tuple(momentum.shape)}"
        )


def validate_spatial_only_window_bcs(integrator: Any) -> None:
    """Reject history-dependent BCs in the vectorized parallel window projection."""
    bcpatch_batch = getattr(integrator, "bcpatch_batch", None)
    if bcpatch_batch is None:
        raise RuntimeError("Integrator has no registered boundary-condition batch")

    history_dependent: List[str] = []
    for group_name in ("first", "mixed", "cyclic", "empty", "coupled"):
        for case_conditions in getattr(bcpatch_batch, group_name, ()):
            for condition in case_conditions:
                if bool(getattr(condition, "depends_on_old_state", False)):
                    history_dependent.append(
                        f"{type(condition).__name__}({condition.patch_name})"
                    )
    if history_dependent:
        raise ValueError(
            "The parallel window projection requires spatial-only "
            "boundary conditions; history-dependent conditions were found: "
            + ", ".join(history_dependent)
        )


def _time_derivative_fields(
    phi_cpd_cell: torch.Tensor, layout: FieldLayout | None = None
) -> torch.Tensor:
    """Return the fields on which the ddt operator acts, per layout."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT
    indices = layout.time_derivative_indices
    if indices == (0, 1, 2, 4):
        return torch.cat((phi_cpd_cell[..., 0:3], phi_cpd_cell[..., 4:5]), dim=-1)
    if indices == (0, 1, 2):
        return phi_cpd_cell[..., 0:3]
    raise ValueError(
        f"No time-derivative gather is implemented for time_derivative_indices={indices}"
    )


def crank_nicolson_window_derivative(
    *,
    phi_old_cpd_cell: torch.Tensor,
    phi_new_cpd_cell: torch.Tensor,
    interior_indices: torch.Tensor,
    dt_value: float,
    committed_ddt_interior: torch.Tensor,
    history_valid: torch.Tensor,
    off_centering: float,
    layout: FieldLayout | None = None,
) -> torch.Tensor:
    """Build OpenFOAM's ddt(CrankNicolson 0.5) recurrence without a time loop."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT
    expected_ddt_fields = layout.num_time_derivative_fields
    if phi_old_cpd_cell.ndim != 3 or phi_new_cpd_cell.ndim != 3:
        raise ValueError("CrankNicolson window states must have shape [W,N,C]")
    if phi_old_cpd_cell.shape != phi_new_cpd_cell.shape:
        raise ValueError("CrankNicolson old/new window state shapes must match")
    if (
        committed_ddt_interior.ndim != 2
        or committed_ddt_interior.shape[-1] != expected_ddt_fields
    ):
        raise ValueError(
            "committed_ddt_interior must have shape "
            f"[N_interior,{expected_ddt_fields}], got {tuple(committed_ddt_interior.shape)}"
        )
    if committed_ddt_interior.shape[0] != interior_indices.numel():
        raise ValueError("CrankNicolson history/interior-index size mismatch")
    if history_valid.numel() != 1:
        raise ValueError("CrankNicolson history_valid must be a scalar tensor")
    if dt_value <= 0:
        raise ValueError("CrankNicolson dt_value must be positive")

    old_interior = phi_old_cpd_cell.index_select(1, interior_indices)
    new_interior = phi_new_cpd_cell.index_select(1, interior_indices)
    q = (
        _time_derivative_fields(new_interior, layout)
        - _time_derivative_fields(old_interior, layout)
    ) / dt_value
    time_steps = q.shape[0]
    time_index = torch.arange(time_steps, device=q.device)
    lag = time_index[:, None] - time_index[None, :]
    lower = lag >= 0
    decay = -off_centering
    powers = torch.pow(decay, lag.clamp_min(0)).to(q.dtype)
    recurrence = torch.where(
        lower,
        (1.0 + off_centering) * powers,
        torch.zeros_like(powers),
    )
    derivative = torch.einsum("ij,jnd->ind", recurrence, q)

    valid = history_valid.to(dtype=q.dtype).reshape(1, 1)
    seed = (
        valid * committed_ddt_interior.to(dtype=q.dtype)
        + (1.0 - valid) * q[0]
    )
    history_power = torch.pow(decay, time_index + 1).to(q.dtype).reshape(-1, 1, 1)
    return derivative + history_power * seed.unsqueeze(0)


def _old_state_from_time_derivative(
    *,
    phi_old_cpd_cell: torch.Tensor,
    phi_new_cpd_cell: torch.Tensor,
    interior_indices: torch.Tensor,
    time_derivative_interior: torch.Tensor,
    dt_value: float,
    layout: FieldLayout | None = None,
) -> torch.Tensor:
    """Encode a prescribed [U(,T)] derivative into the two-state FVM API."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT
    old_interior = phi_old_cpd_cell.index_select(1, interior_indices)
    new_interior = phi_new_cpd_cell.index_select(1, interior_indices)
    indices = layout.time_derivative_indices
    if indices == (0, 1, 2, 4):
        effective_interior = torch.cat(
            (
                new_interior[..., 0:3] - dt_value * time_derivative_interior[..., 0:3],
                old_interior[..., 3:4],
                new_interior[..., 4:5] - dt_value * time_derivative_interior[..., 3:4],
            ),
            dim=-1,
        )
    elif indices == (0, 1, 2):
        effective_interior = torch.cat(
            (
                new_interior[..., 0:3] - dt_value * time_derivative_interior[..., 0:3],
                old_interior[..., 3:4],
            ),
            dim=-1,
        )
    else:
        raise ValueError(
            f"No old-state encoding is implemented for time_derivative_indices={indices}"
        )
    return phi_old_cpd_cell.index_copy(1, interior_indices, effective_interior)


def backward_window_derivative(
    *,
    phi_new_cpd_cell: torch.Tensor,
    phi_prev1_cpd_cell: torch.Tensor,
    phi_prev2_cpd_cell: torch.Tensor,
    interior_indices: torch.Tensor,
    dt_value: float,
    history_valid: torch.Tensor,
    layout: FieldLayout | None = None,
) -> torch.Tensor:
    """OpenFOAM ``backward`` (BDF2) second-order ddt over the window."""
    if phi_new_cpd_cell.ndim != 3:
        raise ValueError("backward window states must have shape [W,N,C]")
    if dt_value <= 0:
        raise ValueError("backward dt_value must be positive")

    time_steps = phi_new_cpd_cell.shape[0]
    time_index = torch.arange(time_steps, device=phi_new_cpd_cell.device).reshape(-1, 1, 1)

    prev1 = torch.where(
        time_index == 0,
        phi_prev1_cpd_cell.unsqueeze(0),
        torch.roll(phi_new_cpd_cell, shifts=1, dims=0),
    )
    prev2 = torch.where(
        time_index == 0,
        phi_prev2_cpd_cell.unsqueeze(0),
        torch.where(
            time_index == 1,
            phi_prev1_cpd_cell.unsqueeze(0),
            torch.roll(phi_new_cpd_cell, shifts=2, dims=0),
        ),
    )

    new_interior = _time_derivative_fields(
        phi_new_cpd_cell.index_select(1, interior_indices), layout
    )
    prev1_interior = _time_derivative_fields(
        prev1.index_select(1, interior_indices), layout
    )
    prev2_interior = _time_derivative_fields(
        prev2.index_select(1, interior_indices), layout
    )

    second_order = (3.0 * new_interior - 4.0 * prev1_interior + prev2_interior) / (
        2.0 * dt_value
    )
    euler = (new_interior - prev1_interior) / dt_value
    use_euler = torch.logical_and(
        torch.logical_not(history_valid.reshape(1, 1, 1)),
        time_index == 0,
    )
    return torch.where(use_euler, euler, second_order)


def _evaluate_time_window_core(
    integrator: Any,
    phi_window_start_cpd_cell: torch.Tensor,
    delta_phi_interior: torch.Tensor,
    interior_indices: torch.Tensor,
    committed_ddt_interior: torch.Tensor,
    cn_history_valid: torch.Tensor,
    phi_prev2_cpd_cell: torch.Tensor,
    *,
    dt_value: float,
    crank_nicolson_off_centering: float,
    rolling_with_grad: bool = True,
    use_backward: bool = False,
    use_steady: bool = False,
    layout: FieldLayout | None = None,
    pseudo_time_scale: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate all W snapshots of the ODIL time window via Warp FVM kernels."""
    layout = layout if layout is not None else _DEFAULT_FIELD_LAYOUT
    time_steps = delta_phi_interior.shape[0]
    if rolling_with_grad:
        cumulative_delta = delta_phi_interior.cumsum(dim=0)
    else:
        exclusive_prefix = delta_phi_interior.cumsum(dim=0) - delta_phi_interior
        cumulative_delta = exclusive_prefix.detach() + delta_phi_interior
    base_states = phi_window_start_cpd_cell.unsqueeze(0).expand(time_steps, -1, -1)
    candidate_states = torch.index_add(
        base_states,
        dim=1,
        index=interior_indices,
        source=cumulative_delta,
    )

    initial_old = phi_window_start_cpd_cell.unsqueeze(0).expand_as(candidate_states)
    candidate_states_bced = integrator.enforce_bcs_only(
        phi_old_cpd_cell=initial_old,
        phi_new_cpd_cell=candidate_states,
    )
    if use_steady:
        old_states = candidate_states_bced
        if pseudo_time_scale is not None:
            new_interior = _time_derivative_fields(
                candidate_states_bced.index_select(1, interior_indices), layout
            )
            base_interior = _time_derivative_fields(
                phi_window_start_cpd_cell.index_select(0, interior_indices), layout
            )
            old_states = _old_state_from_time_derivative(
                phi_old_cpd_cell=candidate_states_bced,
                phi_new_cpd_cell=candidate_states_bced,
                interior_indices=interior_indices,
                time_derivative_interior=pseudo_time_scale
                * (new_interior - base_interior.unsqueeze(0)),
                dt_value=1.0,
                layout=layout,
            )
        loss_by_time, states_bced = integrator(
            phi_old_cpd_cell=old_states,
            phi_new_cpd_cell=candidate_states_bced,
        )
        scalar_loss = loss_by_time.sum(dim=0).mean()
        return scalar_loss, loss_by_time, states_bced

    time_index = torch.arange(time_steps, device=candidate_states_bced.device)
    old_states = torch.where(
        time_index.reshape(-1, 1, 1) == 0,
        phi_window_start_cpd_cell.unsqueeze(0),
        torch.roll(candidate_states_bced, shifts=1, dims=0),
    )
    if not rolling_with_grad:
        old_states = old_states.detach()
    if use_backward:
        time_derivative = backward_window_derivative(
            phi_new_cpd_cell=candidate_states_bced,
            phi_prev1_cpd_cell=phi_window_start_cpd_cell,
            phi_prev2_cpd_cell=phi_prev2_cpd_cell,
            interior_indices=interior_indices,
            dt_value=dt_value,
            history_valid=cn_history_valid,
            layout=layout,
        )
        old_states = _old_state_from_time_derivative(
            phi_old_cpd_cell=old_states,
            phi_new_cpd_cell=candidate_states_bced,
            interior_indices=interior_indices,
            time_derivative_interior=time_derivative,
            dt_value=dt_value,
            layout=layout,
        )
    elif crank_nicolson_off_centering > 0.0:
        time_derivative = crank_nicolson_window_derivative(
            phi_old_cpd_cell=old_states,
            phi_new_cpd_cell=candidate_states_bced,
            interior_indices=interior_indices,
            dt_value=dt_value,
            committed_ddt_interior=committed_ddt_interior,
            history_valid=cn_history_valid,
            off_centering=crank_nicolson_off_centering,
            layout=layout,
        )
        if not rolling_with_grad:
            q_now = (
                _time_derivative_fields(
                    candidate_states_bced.index_select(1, interior_indices), layout
                )
                - _time_derivative_fields(
                    old_states.index_select(1, interior_indices), layout
                )
            ) / dt_value
            diagonal_coefficient = 1.0 + crank_nicolson_off_centering
            time_derivative = time_derivative.detach() + diagonal_coefficient * (
                q_now - q_now.detach()
            )
        old_states = _old_state_from_time_derivative(
            phi_old_cpd_cell=old_states,
            phi_new_cpd_cell=candidate_states_bced,
            interior_indices=interior_indices,
            time_derivative_interior=time_derivative,
            dt_value=dt_value,
            layout=layout,
        )
    loss_by_time, states_bced = integrator(
        phi_old_cpd_cell=old_states,
        phi_new_cpd_cell=candidate_states_bced,
    )
    scalar_loss = loss_by_time.sum(dim=0).mean()
    return scalar_loss, loss_by_time, states_bced


def _unpack_norm_factors(
    norms: Any,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Normalize 2-tuple or 3-tuple solver normFactor returns to ``(cont, mom, energy)``."""
    if norms is None:
        return None, None, None
    if len(norms) == 2:
        return norms[0], norms[1], None
    return norms[0], norms[1], norms[2]


class _LazyLog10Components(Mapping[str, torch.Tensor]):
    """Compute display-only ``log10`` residual summaries lazily on demand."""

    def __init__(
        self,
        named: Mapping[str, torch.Tensor],
        time_steps: int,
        reducer: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._named = named
        self._time_steps = time_steps
        self._reducer = reducer
        self._cache: Dict[str, torch.Tensor] | None = None

    def _materialize(self) -> Dict[str, torch.Tensor]:
        if self._cache is None:
            out: Dict[str, torch.Tensor] = {}
            for name, component in self._named.items():
                per_time = self._reducer(component)
                if tuple(per_time.shape) != (self._time_steps,):
                    raise RuntimeError(
                        f"Residual {name!r} did not preserve time axis: "
                        f"got {tuple(per_time.shape)}, expected {(self._time_steps,)}"
                    )
                out[name] = per_time.mean()
                out[f"{name}_window_worst"] = per_time.max()
            self._cache = out
        return self._cache

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._materialize()[key]

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


def make_window_core(
    integrator: Any,
    *,
    dt_value: float,
    crank_nicolson_off_centering: float,
    rolling_with_grad: bool = True,
    use_backward: bool = False,
    layout: FieldLayout | None = None,
) -> WindowCore:
    """Bind the Warp-backed integrator for low-overhead FIFO window evaluation."""
    layout = (
        layout if layout is not None
        else resolve_integrator_field_layout(integrator)
    )
    solver = integrator._solver
    if not hasattr(solver, "get_normalization_factors"):
        raise RuntimeError(
            f"Solver {getattr(integrator, 'solver_type', '?')!r} does not expose "
            "get_normalization_factors()"
        )

    for method_name in ("enforce_bcs_only", "forward"):
        fn = getattr(solver, method_name, None)
        if fn is not None and not getattr(fn, "_odil_compiler_disabled", False):
            wrapped = torch.compiler.disable(fn)
            setattr(wrapped, "_odil_compiler_disabled", True)
            setattr(solver, method_name, wrapped)

    def bound_core(
        phi_window_start_cpd_cell: torch.Tensor,
        delta_phi_interior: torch.Tensor,
        interior_indices: torch.Tensor,
        committed_ddt_interior: torch.Tensor,
        cn_history_valid: torch.Tensor,
        phi_prev2_cpd_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        scalar_loss, loss_by_time, states_bced = _evaluate_time_window_core(
            integrator,
            phi_window_start_cpd_cell,
            delta_phi_interior,
            interior_indices,
            committed_ddt_interior,
            cn_history_valid,
            phi_prev2_cpd_cell,
            dt_value=dt_value,
            crank_nicolson_off_centering=crank_nicolson_off_centering,
            rolling_with_grad=rolling_with_grad,
            use_backward=use_backward,
            layout=layout,
        )
        norm_cont, norm_mom, norm_energy = _unpack_norm_factors(
            solver.get_normalization_factors()
        )
        loss_energy = (
            getattr(solver, "loss_energy_component", None)
            if layout.has_energy
            else None
        )
        loss_pref = getattr(solver, layout.pref_attr, None)
        return (
            scalar_loss,
            loss_by_time,
            states_bced,
            solver.loss_cont_component,
            solver.loss_momentum_component,
            loss_energy,
            getattr(solver, "loss_phi_jump_component", None),
            loss_pref,
            norm_cont,
            norm_mom,
            norm_energy if layout.has_energy else None,
        )

    return bound_core


def _normalization_factors(
    integrator: Any,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """OpenFOAM-style equation scales, or ``None`` when unavailable."""
    solver = getattr(integrator, "_solver", None)
    factory = getattr(solver, "get_normalization_factors", None)
    if factory is None:
        return None, None, None
    return _unpack_norm_factors(factory())


def evaluate_time_window(
    *,
    integrator: Any,
    phi_window_start_cpd_cell: torch.Tensor,
    delta_phi_interior: torch.Tensor,
    interior_indices: torch.Tensor,
    core_evaluator: WindowCore | None = None,
    time_loss_weights: torch.Tensor | None = None,
    dt_value: float = 1.0,
    crank_nicolson_off_centering: float = 0.0,
    rolling_with_grad: bool = True,
    use_backward: bool = False,
    use_steady: bool = False,
    committed_ddt_interior: torch.Tensor | None = None,
    cn_history_valid: torch.Tensor | None = None,
    phi_prev2_cpd_cell: torch.Tensor | None = None,
    layout: FieldLayout | None = None,
    pseudo_time_scale: torch.Tensor | None = None,
) -> WindowEvaluation:
    """Evaluate all coupled time steps in one ODIL window."""
    from TimeWindow.residuals import _component_log10_per_time

    layout = (
        layout if layout is not None
        else resolve_integrator_field_layout(integrator)
    )
    if delta_phi_interior.ndim != 3:
        raise ValueError(
            f"delta_phi_interior must have shape [W,N_interior,C], got {tuple(delta_phi_interior.shape)}"
        )
    if interior_indices.ndim != 1:
        raise ValueError("interior_indices must be one-dimensional")
    if delta_phi_interior.shape[1] != interior_indices.numel():
        raise ValueError(
            f"Interior dimension mismatch: delta has {delta_phi_interior.shape[1]} rows "
            f"but mask has {interior_indices.numel()}"
        )
    if delta_phi_interior.shape[2] != phi_window_start_cpd_cell.shape[1]:
        raise ValueError("Delta/state channel counts do not match")

    time_steps = int(delta_phi_interior.shape[0])
    derivative_history = committed_ddt_interior
    if derivative_history is None:
        derivative_history = delta_phi_interior.new_zeros(
            (interior_indices.numel(), layout.num_time_derivative_fields)
        )
    if cn_history_valid is None:
        cn_history_valid = torch.zeros(
            (), dtype=torch.bool, device=delta_phi_interior.device
        )
    prev2_state = phi_prev2_cpd_cell
    if prev2_state is None:
        prev2_state = phi_window_start_cpd_cell
    if derivative_history.device != delta_phi_interior.device:
        raise ValueError("CrankNicolson history and delta must share a device")
    if cn_history_valid.device != delta_phi_interior.device:
        raise ValueError("CrankNicolson history flag and delta must share a device")
    if pseudo_time_scale is not None and not use_steady:
        raise ValueError("pseudo_time_scale applies to ddtSchemes steadyState only")
    if use_steady and core_evaluator is not None:
        raise NotImplementedError(
            "ddtSchemes steadyState uses the direct window path (core_evaluator=None)"
        )

    if core_evaluator is None:
        scalar_loss, loss_by_time, states_bced = _evaluate_time_window_core(
            integrator,
            phi_window_start_cpd_cell,
            delta_phi_interior,
            interior_indices,
            derivative_history,
            cn_history_valid,
            prev2_state,
            dt_value=dt_value,
            crank_nicolson_off_centering=crank_nicolson_off_centering,
            rolling_with_grad=rolling_with_grad,
            use_backward=use_backward,
            use_steady=use_steady,
            layout=layout,
            pseudo_time_scale=pseudo_time_scale,
        )
    else:
        core_outputs = core_evaluator(
            phi_window_start_cpd_cell,
            delta_phi_interior,
            interior_indices,
            derivative_history,
            cn_history_valid,
            prev2_state,
        )
        if len(core_outputs) != 11:
            raise RuntimeError(
                f"A bound window core must return eleven tensors, got {len(core_outputs)}"
            )
        (
            scalar_loss,
            loss_by_time,
            states_bced,
            loss_cont,
            loss_mom,
            loss_energy,
            loss_phi_jump,
            loss_p_rgh_ref,
            norm_cont,
            norm_mom,
            norm_energy,
        ) = core_outputs

    if tuple(loss_by_time.shape) != (time_steps, 1):
        raise RuntimeError(
            f"The fixed single-case window must return loss [T,1], got {tuple(loss_by_time.shape)}"
        )
    if time_loss_weights is not None:
        if time_loss_weights.ndim != 1 or time_loss_weights.shape[0] != time_steps:
            raise ValueError(
                f"time_loss_weights must have shape [W], got {tuple(time_loss_weights.shape)} for W={time_steps}"
            )
        if time_loss_weights.device != loss_by_time.device:
            raise ValueError("time_loss_weights and loss_by_time must share a device")
        scalar_loss = (
            loss_by_time * time_loss_weights.to(dtype=loss_by_time.dtype)[:, None]
        ).sum(dim=0).mean()

    if core_evaluator is None:
        named = dict(integrator.iter_named_residuals())
        norm_cont, norm_mom, norm_energy = _normalization_factors(integrator)
    else:
        pref_key = (
            "p_rghRef"
            if layout.pref_attr == "loss_p_rghRef_component"
            else "pRef"
        )
        named = {
            "cont": loss_cont,
            "mom": loss_mom,
        }
        if layout.has_energy and loss_energy is not None:
            named["energy"] = loss_energy
        if loss_phi_jump is not None:
            named["phi_jump"] = loss_phi_jump
        if loss_p_rgh_ref is not None:
            named[pref_key] = loss_p_rgh_ref

    residual_scales: Dict[str, torch.Tensor | None] = {
        "cont": norm_cont,
        "mom": norm_mom,
    }
    if layout.has_energy:
        residual_scales["energy"] = norm_energy

    _validate_combined_momentum_layout(integrator, named)
    for name, component in named.items():
        if component.ndim < 2 or int(component.shape[0]) != time_steps:
            raise RuntimeError(
                f"Residual {name!r} did not preserve time axis: "
                f"got {tuple(component.shape)}, expected leading T={time_steps}"
            )
    summary_logs: Mapping[str, torch.Tensor] = _LazyLog10Components(
        named, time_steps, _component_log10_per_time
    )
    return WindowEvaluation(
        loss=scalar_loss,
        loss_by_time=loss_by_time,
        states=tuple(states_bced.unbind(dim=0)),
        log10_components=summary_logs,
        residual_components=named,
        residual_scales=residual_scales,
    )
