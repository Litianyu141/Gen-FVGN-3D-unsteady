"""Run session for FIFO-window ODIL: case loading, logger, gates, and export."""

from __future__ import annotations

from datetime import datetime
import os
import random
import sys
import time
import tomllib
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from FVdomain.load_case import load_case
from TimeWindow.config import (
    FIFOConfig,
    apply_fifo_cli_overrides,
    get_odil_window_steps,
    read_fifo_config,
    validate_case_assets,
)
from TimeWindow.reference_gate import build_openfoam_reference_gate
from TimeWindow.runio import (
    _read_static_case_identity,
    build_run_directory_name,
    export_results,
    should_export_popped_snapshot,
    snapshot_case_inputs,
)
from TimeWindow.window import (
    FieldLayout,
    make_window_core,
    resolve_field_layout,
    validate_spatial_only_window_bcs,
)
from Utils.Logger import Logger


def validate_window_channels(params: Any) -> Tuple[int, FieldLayout]:
    """Return the FIFO window width and field layout resolved from ``[solver] type``."""
    fvconfig = params.fvconfig
    solver_type = str(fvconfig["solver"]["type"])
    layout = resolve_field_layout(solver_type)
    momentum_scheme = str(
        fvconfig["fvSchemes"].get("momentumResidual", "cell")
    ).lower()
    if momentum_scheme not in ("cell+face_cv", "face_cv+cell"):
        raise ValueError(
            "FIFO-window Navier-Stokes runs require fvSchemes.momentumResidual="
            f"cell+face_cv, got {momentum_scheme!r} (solver.type={solver_type!r})"
        )
    return get_odil_window_steps(fvconfig), layout


def scalar_reynolds(fvconfig: Any) -> float | None:
    """Return ``theta_pde.Re`` if it is a plain scalar, otherwise ``None``."""
    value = (fvconfig.get("theta_pde") or {}).get("Re")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class FIFOWindowSession:
    """Lifecycle controller for one FIFO-window ODIL run."""

    def __init__(self, args: Any, invocation_args: Sequence[str]) -> None:
        if args.print_freq < 1 or args.csv_flush_rows < 1:
            raise ValueError("print-freq and csv-flush-rows must be >= 1")
        self.args = args
        self.invocation_args = list(invocation_args)
        self.case_dir = os.path.dirname(os.path.abspath(args.dataset_dir))
        self.mesh_path = validate_case_assets(args.dataset_dir)
        with open(os.path.join(args.dataset_dir, "fvconfig.toml"), "rb") as handle:
            self.fvconfig = tomllib.load(handle)

    def resolve_fifo_config(
        self, *, physical_steps: int | None = None
    ) -> FIFOConfig:
        """Resolve ``[fvSolution.fifo]`` with CLI overrides applied."""
        args = self.args
        fvconfig = self.fvconfig
        if args.cn_off_centering is not None:
            fvconfig["fvSolution"]["fifo"]["crank_nicolson_off_centering"] = (
                args.cn_off_centering
            )

        fifo_config = apply_fifo_cli_overrides(read_fifo_config(fvconfig), args)
        if physical_steps is not None:
            fifo_config = type(fifo_config)(
                **{**fifo_config.__dict__, "physical_steps": physical_steps}
            )
        if not fifo_config.enabled:
            raise ValueError("This entry requires fvSolution.fifo.enabled=true")

        args.max_head_iterations = fifo_config.max_head_iterations
        args.n_physical_steps = fifo_config.physical_steps
        args.export_mode = fifo_config.export_mode
        args.lr = fifo_config.learning_rate
        args.minimum_lr = fifo_config.minimum_learning_rate
        args.adam_eps = fifo_config.adam_eps
        args.grad_clip_norm = fifo_config.grad_clip_norm
        args.head_loss_weight = fifo_config.head_loss_weight
        args.future_loss_decay = fifo_config.future_loss_decay
        args.tail_init_std = fifo_config.tail_init_std
        args.reference_control = fifo_config.reference_control.enabled
        self.fifo_config = fifo_config
        return fifo_config

    def open_logger(
        self, *, logger_subdir: str, execution_tag: str, iterations: int
    ) -> Logger:
        """Seed RNGs and create ``<case>/Logger/<subdir>/<run>/``."""
        args = self.args
        fifo_config = self.fifo_config
        from Utils import get_param

        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            params = get_param.params()
        finally:
            sys.argv = original_argv

        params.on_gpu = args.gpu
        params.net = "TransFVGN_v2"
        params.dataset_dir = args.dataset_dir
        params.n_epochs = fifo_config.max_head_iterations
        params.n_physical_steps = fifo_config.physical_steps
        params.print_freq = args.print_freq
        params.dataset_size = 1
        params.batch_size = 1
        params.integrator = "implicit"
        params.optimizer_type = "adam"
        params.scheduler_type = "fifo_linear_per_head"
        params.lr = fifo_config.learning_rate
        params.adam_eps = fifo_config.adam_eps
        params.weight_decay = 0.0
        params.grad_clip_norm = fifo_config.grad_clip_norm
        params.early_stop_threshold = float("-inf")
        params.load_date_time = None
        params.load_index = None

        seed = int(args.seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.set_per_process_memory_fraction(0.99, device=args.gpu)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )

        (
            self.static_window_steps,
            self.static_dt,
            scheme_tag,
            self.case_label,
        ) = _read_static_case_identity(args.dataset_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_directory = build_run_directory_name(
            timestamp=timestamp,
            window_steps=self.static_window_steps,
            dt_value=self.static_dt,
            learning_rate=fifo_config.learning_rate,
            physical_steps=fifo_config.physical_steps,
            iterations=iterations,
            seed=seed,
            scheme_tag=scheme_tag,
            case_label=self.case_label,
            execution_tag=execution_tag,
            experiment_tag=args.exp_name,
        )
        saving_path = os.path.join(
            self.case_dir, "Logger", logger_subdir, self.run_directory
        )
        logger = Logger(
            logger_subdir,
            use_csv=True,
            params=params,
            copy_code=True,
            seed=seed,
            saving_path=saving_path,
        )
        logger.residual_buffer_size = args.csv_flush_rows
        self.params = params
        self.logger = logger
        return logger

    def load_case(self) -> None:
        """Build Warp integrator, window core, reference gate, manifest, and initial state."""
        args = self.args
        fifo_config = self.fifo_config
        logger = self.logger
        params = self.params
        device = self.device
        case_dir = self.case_dir
        mesh_path = self.mesh_path
        case_label = self.case_label

        logger.info("Loading case %s from %s on %s ...", case_label, mesh_path, device)
        load_start = time.time()
        params.state_save_dir = logger.saving_path
        case = load_case(
            args.dataset_dir, params, stage="solver", pool="legacy", device=device
        )
        params = case.params
        integrator = case.integrator
        meta_data = case.meta_data
        case_name = case.case_name
        validate_spatial_only_window_bcs(integrator)
        window_steps, layout = validate_window_channels(params)
        re_value = scalar_reynolds(params.fvconfig)

        dt_value = float(params.fvconfig["theta_pde"]["dt"]["val"])
        window_core = make_window_core(
            integrator,
            dt_value=dt_value,
            crank_nicolson_off_centering=fifo_config.crank_nicolson_off_centering,
            rolling_with_grad=(fifo_config.rolling_scheme == "RollingWithGrad"),
            use_backward=(fifo_config.temporal_scheme == "backward"),
        )
        logger.info("FVM backend: NVIDIA Warp")
        logger.info(
            "Temporal unrolling: rolling_scheme=%s (%s)",
            fifo_config.rolling_scheme,
            "one graph spans the window; a single backward couples all W snapshots"
            if fifo_config.rolling_scheme == "RollingWithGrad"
            else "snapshots detached; no gradient crosses a snapshot boundary",
        )
        if params.phi_num_channels != layout.num_channels:
            raise ValueError(
                f"solver.type={params.fvconfig['solver']['type']!r} declares "
                f"{layout.num_channels} fields {list(layout.names)}, but the loaded "
                f"case carries {params.phi_num_channels} channels"
            )

        num_interior = int(integrator.mask_interior_cell.sum().item())
        interior_indices = integrator.mask_interior_cell.nonzero(as_tuple=True)[0]
        momentum_scheme = str(params.fvconfig["fvSchemes"]["momentumResidual"])
        if (
            window_steps != self.static_window_steps
            or dt_value != self.static_dt
            or case_name != case_label
        ):
            raise RuntimeError(
                "The loaded case identity differs from fvconfig.toml used in the run id"
            )
        logger.info(
            "Loaded %s in %.2fs: %sdt=%g, interior=%d, window=%d, fields=%s, "
            "scheme=%s, time=%s %.3g",
            case_name,
            time.time() - load_start,
            "" if re_value is None else f"Re={re_value:g}, ",
            dt_value,
            num_interior,
            window_steps,
            ",".join(layout.names),
            momentum_scheme,
            fifo_config.temporal_scheme,
            fifo_config.crank_nicolson_off_centering,
        )
        reference_gate, resolved_reference_path = build_openfoam_reference_gate(
            config=fifo_config.reference_control,
            candidate_cell_centers=integrator.cpd_cell_pos.index_select(
                0, interior_indices
            ),
            total_physical_steps=fifo_config.physical_steps,
            dt_value=dt_value,
            case_dir=case_dir,
        )
        if reference_gate is not None:
            control = fifo_config.reference_control
            logger.info(
                "OpenFOAM acceptance gate: %s every %d dt; max NRMSE U/p/T="
                "%.3g/%.3g/%.3g, min corr=%.3g/%.3g/%.3g, "
                "annulus U/p95<=%.3g/%.3g, wake U<=%.3g, max w/werr=%.3g/%.3g",
                resolved_reference_path,
                control.check_stride,
                control.max_u_normalized_rmse,
                control.max_p_normalized_rmse,
                control.max_t_normalized_rmse,
                control.min_u_correlation,
                control.min_p_correlation,
                control.min_t_correlation,
                control.max_annulus_u_normalized_rmse,
                control.max_annulus_u_error_p95,
                control.max_wake_u_normalized_rmse,
                control.max_w_rms,
                control.max_w_error_rms,
            )
        manifest_path = snapshot_case_inputs(
            logger=logger,
            args=args,
            invocation_args=self.invocation_args,
            mesh_path=mesh_path,
            case_name=case_name,
            window_steps=window_steps,
            dt_value=dt_value,
            momentum_scheme=momentum_scheme,
            num_interior=num_interior,
            device=device,
            fifo_config=fifo_config,
            case_dir=case_dir,
            re_target=re_value,
            layout=layout,
        )
        logger.info("Run manifest: %s", manifest_path)
        logger.info(
            "FIFO controls: stride=%d max_head_iter=%d checks=%d consecutive=%d "
            "min_snapshot_updates=%d lr=%.2e->%.2e preserve_adam=%s eager_verify=%s",
            fifo_config.stride,
            fifo_config.max_head_iterations,
            fifo_config.check_interval,
            fifo_config.consecutive_checks,
            fifo_config.min_snapshot_updates,
            fifo_config.learning_rate,
            fifo_config.minimum_learning_rate,
            fifo_config.preserve_adam_state,
            fifo_config.eager_verify_on_pop,
        )
        logger.info(
            "FIFO channel update scales [%s]; time weights head=%.3g decay=%.3g; "
            "tail extrapolation=%.3g noise_std=%.3g",
            ", ".join(
                f"{name}={scale:g}"
                for name, scale in zip(layout.names, fifo_config.adam_update_scale)
            ),
            fifo_config.head_loss_weight,
            fifo_config.future_loss_decay,
            fifo_config.tail_extrapolation,
            fifo_config.tail_init_std,
        )
        for criterion in fifo_config.criteria:
            logger.info(
                "FIFO residualControl %s -> %s: tolerance=%.3e relTol=%.3g",
                criterion.field_name,
                criterion.component,
                criterion.tolerance,
                criterion.relative_tolerance,
            )

        phi_initial = meta_data["init_phi_cpd_cell"].to(
            device=device, dtype=torch.float32
        )
        expected_shape = (integrator.num_cpd_cells, layout.num_channels)
        if tuple(phi_initial.shape) != expected_shape:
            raise ValueError(
                f"Initial state has shape {tuple(phi_initial.shape)}, expected {expected_shape}"
            )

        self.params = params
        self.integrator = integrator
        self.meta_data = meta_data
        self.case_name = case_name
        self.layout = layout
        self.window_steps = window_steps
        self.dt_value = dt_value
        self.window_core = window_core
        self.reference_gate = reference_gate
        self.phi_initial = phi_initial

    def trainer_kwargs(self) -> Dict[str, Any]:
        return dict(
            phi_initial_cpd_cell=self.phi_initial,
            integrator=self.integrator,
            params=self.params,
            logger=self.logger,
            fifo_config=self.fifo_config,
            window_core=self.window_core,
            reference_gate=self.reference_gate,
        )

    def announce(self) -> None:
        self.logger.info(
            "FIFO window rollout: %d popped snapshots (W=%d, stride=1, dt=%g)",
            self.fifo_config.physical_steps,
            self.window_steps,
            self.dt_value,
        )
        self.logger.info("VTM export mode: %s", self.fifo_config.export_mode)

    def run(self, trainer: Any) -> int:
        fifo_config = self.fifo_config
        logger = self.logger
        integrator = self.integrator
        meta_data = self.meta_data
        case_name = self.case_name
        dt_value = self.dt_value
        layout = self.layout

        def on_fifo_pop(physical_step: int, state: torch.Tensor) -> None:
            if not should_export_popped_snapshot(fifo_config.export_mode):
                return
            if (
                physical_step % fifo_config.export_stride
                and physical_step != fifo_config.physical_steps
            ):
                return
            export_results(
                phi_cpd_cell=state,
                meta_data=meta_data,
                mask_interior_cell=integrator.mask_interior_cell,
                logger=logger,
                case_name=case_name,
                physical_step=physical_step,
                physical_time=physical_step * dt_value,
            )

        try:
            trainer.run(
                total_physical_steps=fifo_config.physical_steps,
                on_pop=on_fifo_pop,
            )
        finally:
            logger.finalize_residuals()

        final_interior = trainer.phi_committed_cpd_cell[integrator.mask_interior_cell]
        logger.info("FIFO window rollout complete")
        ranges = [f"|U|max={float(final_interior[:, 0:3].abs().max()):.6e}"]
        for index in (layout.pressure_index, *layout.scalar_transport_indices):
            column = final_interior[:, index]
            ranges.append(
                f"{layout.names[index]}=[{float(column.min()):.6e},"
                f"{float(column.max()):.6e}]"
            )
        logger.info("Final ranges: %s", ", ".join(ranges))
        return 0
