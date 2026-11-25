"""Read-only OpenFOAM acceptance gate for an in-memory ODIL snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol, Tuple

import numpy as np

from TimeWindow.config import ReferenceControlConfig


@dataclass(frozen=True)
class ReferenceFrame:
    """CPU arrays for one OpenFOAM internal-mesh time."""

    centers: np.ndarray
    volumes: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray


class ReferenceSource(Protocol):
    """Small seam that also permits PyVista-free synthetic tests."""

    @property
    def time_values(self) -> np.ndarray: ...

    def frame(self, physical_time: float) -> ReferenceFrame: ...


@dataclass(frozen=True)
class ReferenceGateThresholds:
    """All acceptance limits; ``None`` disables the corresponding check."""

    max_u_normalized_rmse: float | None
    max_annulus_u_normalized_rmse: float | None
    max_annulus_u_error_p95: float | None
    max_wake_u_normalized_rmse: float | None
    max_p_normalized_rmse: float | None
    max_t_normalized_rmse: float | None
    min_u_correlation: float | None
    min_p_correlation: float | None
    min_t_correlation: float | None
    max_w_rms: float | None
    max_w_error_rms: float | None
    max_temperature_out_of_bounds_fraction: float | None
    max_recirculation_length_difference: float | None

    def __post_init__(self) -> None:
        upper_names = (
            "max_u_normalized_rmse",
            "max_annulus_u_normalized_rmse",
            "max_annulus_u_error_p95",
            "max_wake_u_normalized_rmse",
            "max_p_normalized_rmse",
            "max_t_normalized_rmse",
            "max_w_rms",
            "max_w_error_rms",
            "max_temperature_out_of_bounds_fraction",
            "max_recirculation_length_difference",
        )
        for name in upper_names:
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name in ("min_u_correlation", "min_p_correlation", "min_t_correlation"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not -1.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [-1, 1], got {value}")


@dataclass(frozen=True)
class ReferenceGateResult:
    """A complete, immutable decision for one candidate snapshot."""

    passed: bool
    physical_time: float
    reference_time: float
    u_normalized_rmse: float
    annulus_cell_count: int
    annulus_u_normalized_rmse: float
    annulus_u_error_p95: float
    wake_cell_count: int
    wake_u_normalized_rmse: float
    p_normalized_rmse: float
    t_normalized_rmse: float
    u_correlation: float
    p_correlation: float
    t_correlation: float
    w_rms: float
    w_error_rms: float
    temperature_out_of_bounds_fraction: float
    reference_recirculation_length: float
    candidate_recirculation_length: float
    recirculation_length_difference: float
    candidate_pressure_gauge_removed: float
    reference_pressure_gauge_removed: float
    mapping_method: str
    mapping_max_center_distance: float
    failures: tuple[str, ...]


def _as_cpu_numpy(value: object, *, name: str) -> np.ndarray:
    """Detach torch-like input and make a finite float64 CPU copy."""
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    result = np.asarray(cpu_value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)
    denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(candidate_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(reference, candidate, rtol=1e-12, atol=1e-14) else 0.0
    return float(np.clip(np.dot(ref_centered, candidate_centered) / denominator, -1.0, 1.0))


def _gauge_pressure(pressure: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, float]:
    volume_sum = float(np.sum(volumes))
    if not np.isfinite(volume_sum) or volume_sum <= 0.0:
        raise ValueError("Reference cell volumes must have a positive finite sum")
    mean = float(np.sum(pressure * volumes) / volume_sum)
    return pressure - mean, mean


def _recirculation_length(
    centers: np.ndarray,
    ux: np.ndarray,
    *,
    cylinder_center_x: float,
    cylinder_center_y: float,
    cylinder_diameter: float,
    centerline_half_width: float,
) -> float:
    rear_x = cylinder_center_x + 0.5 * cylinder_diameter
    mask = (
        (centers[:, 0] > rear_x)
        & (np.abs(centers[:, 1] - cylinder_center_y) <= centerline_half_width)
        & (ux < 0.0)
    )
    return max(0.0, float(np.max(centers[mask, 0])) - rear_x) if np.any(mask) else 0.0


def _cylinder_region_masks(
    centers: np.ndarray,
    *,
    cylinder_center_x: float,
    cylinder_center_y: float,
    cylinder_diameter: float,
    annulus_outer_radius_diameters: float,
    wake_end_x_diameters: float,
    wake_half_width_diameters: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a near-wall annulus and a downstream wake without that annulus."""

    x = centers[:, 0] - cylinder_center_x
    y = centers[:, 1] - cylinder_center_y
    radius = np.hypot(x, y)
    inner_radius = 0.5 * cylinder_diameter
    outer_radius = annulus_outer_radius_diameters * cylinder_diameter
    annulus = (radius > inner_radius) & (radius <= outer_radius)
    wake = (
        (x > inner_radius)
        & (x <= wake_end_x_diameters * cylinder_diameter)
        & (np.abs(y) <= wake_half_width_diameters * cylinder_diameter)
        & (radius > outer_radius)
    )
    if not np.any(annulus):
        raise ValueError(
            "Cylinder annulus contains no reference cells; check the cylinder "
            "geometry and cell-center mapping"
        )
    if not np.any(wake):
        raise ValueError(
            "Cylinder wake region contains no reference cells; check the wake "
            "extent and cell-center mapping"
        )
    return annulus, wake


def _region_u_error_metrics(
    reference_u: np.ndarray,
    candidate_u: np.ndarray,
    mask: np.ndarray,
    velocity_scale: float,
) -> tuple[float, float]:
    """Return vector NRMSE and the 95th percentile of ``|delta U|/Uin``."""

    error_magnitude = np.linalg.norm(candidate_u[mask] - reference_u[mask], axis=1)
    normalized_rmse = float(np.sqrt(np.mean(error_magnitude**2)) / velocity_scale)
    normalized_p95 = float(np.quantile(error_magnitude, 0.95) / velocity_scale)
    return normalized_rmse, normalized_p95


def _cell_mapping(
    reference_centers: np.ndarray,
    candidate_centers: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, str, float]:
    if reference_centers.shape != candidate_centers.shape:
        raise ValueError(
            "OpenFOAM and candidate cell-center shapes differ: "
            f"{reference_centers.shape} versus {candidate_centers.shape}"
        )
    ordered_distances = np.linalg.norm(reference_centers - candidate_centers, axis=1)
    ordered_max = float(np.max(ordered_distances, initial=0.0))
    if ordered_max <= tolerance:
        return np.arange(len(reference_centers), dtype=np.int64), "cell_order", ordered_max

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Cell order differs; scipy is required for a one-to-one center mapping"
        ) from exc
    distances, indices = cKDTree(candidate_centers).query(reference_centers, k=1)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Nearest-center mapping is not one-to-one")
    maximum = float(np.max(distances, initial=0.0))
    if maximum > tolerance:
        raise ValueError(
            f"Cell-center mapping error {maximum:.6g} exceeds tolerance {tolerance:.6g}"
        )
    return np.asarray(indices, dtype=np.int64), "nearest_center", maximum


def _internal_mesh(dataset: object) -> object:
    """Locate internalMesh without depending on the comparison CLI namespace."""
    import pyvista as pv

    if not isinstance(dataset, pv.MultiBlock):
        if not isinstance(dataset, pv.DataSet):
            raise TypeError(f"Expected a VTK dataset, got {type(dataset).__name__}")
        return dataset
    for name in ("internalMesh", "internal_mesh"):
        if name in dataset.keys() and isinstance(dataset[name], pv.DataSet):
            return dataset[name]
    candidates: list[object] = []

    def visit(block: object) -> None:
        if isinstance(block, pv.MultiBlock):
            for child in block:
                visit(child)
        elif isinstance(block, pv.DataSet) and "U" in block.cell_data:
            candidates.append(block)

    visit(dataset)
    if not candidates:
        raise ValueError("Could not find an internal cell mesh containing field 'U'")
    return max(candidates, key=lambda block: block.n_cells)


def _cell_array(mesh: object, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in mesh.cell_data:
            result = np.asarray(mesh.cell_data[name], dtype=np.float64)
            return result[:, 0] if result.ndim == 2 and result.shape[1] == 1 else result
    raise KeyError(f"None of the required cell arrays {names} exists")


class OpenFOAMReferenceSource:
    """PyVista-backed source retaining one reader and one decoded frame."""

    def __init__(self, foam_path: str | Path) -> None:
        import pyvista as pv

        path = Path(foam_path).expanduser().resolve()
        if path.is_dir():
            markers = sorted(path.glob("*.foam"))
            if len(markers) != 1:
                raise ValueError(f"Expected exactly one .foam marker in {path}, found {len(markers)}")
            path = markers[0]
        if not path.is_file() or path.suffix.lower() != ".foam":
            raise FileNotFoundError(f"OpenFOAM .foam marker does not exist: {path}")
        self.path = path
        self._reader = pv.OpenFOAMReader(str(path))
        if hasattr(self._reader, "cell_to_point_creation"):
            self._reader.cell_to_point_creation = False
        self._cached_time: float | None = None
        self._cached_frame: ReferenceFrame | None = None

    @property
    def time_values(self) -> np.ndarray:
        return np.asarray(self._reader.time_values, dtype=np.float64)

    def frame(self, physical_time: float) -> ReferenceFrame:
        if self._cached_time == physical_time and self._cached_frame is not None:
            return self._cached_frame
        self._reader.set_active_time_value(physical_time)
        mesh = _internal_mesh(self._reader.read())
        centers = np.asarray(mesh.cell_centers().points, dtype=np.float64).copy()
        sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
        volumes = np.abs(np.asarray(sized.cell_data["Volume"], dtype=np.float64)).copy()
        if float(np.sum(volumes)) <= 0.0:
            volumes = np.ones(mesh.n_cells, dtype=np.float64)
        frame = ReferenceFrame(
            centers=centers,
            volumes=volumes,
            velocity=_cell_array(mesh, ("U",)).copy(),
            pressure=_cell_array(mesh, ("p_rgh", "p")).copy(),
            temperature=_cell_array(mesh, ("T",)).copy(),
        )
        self._cached_time = physical_time
        self._cached_frame = frame
        return frame


class OpenFOAMReferenceGate:
    """Compare an interior ``[cell, (u,v,w,p,T)]`` state to OpenFOAM."""

    def __init__(
        self,
        foam_path: str | Path | None,
        candidate_cell_centers: object,
        thresholds: ReferenceGateThresholds,
        *,
        velocity_scale: float,
        pressure_scale: float,
        cylinder_center_x: float,
        cylinder_center_y: float,
        cylinder_diameter: float,
        centerline_half_width: float,
        annulus_outer_radius_diameters: float,
        wake_end_x_diameters: float,
        wake_half_width_diameters: float,
        temperature_scale: float = 1.0,
        time_tolerance: float = 1e-8,
        center_tolerance: float = 2e-4,
        temperature_min: float = 0.0,
        temperature_max: float = 1.0,
        temperature_bound_tolerance: float = 1e-8,
        source: ReferenceSource | None = None,
    ) -> None:
        self._candidate_centers = _as_cpu_numpy(
            candidate_cell_centers, name="candidate_cell_centers"
        )
        if self._candidate_centers.ndim != 2 or self._candidate_centers.shape[1] != 3:
            raise ValueError(
                "candidate_cell_centers must have shape [N_cells,3], got "
                f"{self._candidate_centers.shape}"
            )
        for name, value in (
            ("velocity_scale", velocity_scale),
            ("pressure_scale", pressure_scale),
            ("temperature_scale", temperature_scale),
            ("time_tolerance", time_tolerance),
            ("center_tolerance", center_tolerance),
            ("cylinder_diameter", cylinder_diameter),
            ("centerline_half_width", centerline_half_width),
            ("annulus_outer_radius_diameters", annulus_outer_radius_diameters),
            ("wake_end_x_diameters", wake_end_x_diameters),
            ("wake_half_width_diameters", wake_half_width_diameters),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if temperature_min > temperature_max:
            raise ValueError("temperature_min must not exceed temperature_max")
        if annulus_outer_radius_diameters <= 0.5:
            raise ValueError("annulus_outer_radius_diameters must be greater than 0.5")
        if wake_end_x_diameters <= 0.5:
            raise ValueError("wake_end_x_diameters must be greater than 0.5")
        self.thresholds = thresholds
        self.velocity_scale = float(velocity_scale)
        self.pressure_scale = float(pressure_scale)
        self.temperature_scale = float(temperature_scale)
        self.time_tolerance = float(time_tolerance)
        self.center_tolerance = float(center_tolerance)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.temperature_bound_tolerance = float(temperature_bound_tolerance)
        self.cylinder_center_x = float(cylinder_center_x)
        self.cylinder_center_y = float(cylinder_center_y)
        self.cylinder_diameter = float(cylinder_diameter)
        self.centerline_half_width = float(centerline_half_width)
        self.annulus_outer_radius_diameters = float(annulus_outer_radius_diameters)
        self.wake_end_x_diameters = float(wake_end_x_diameters)
        self.wake_half_width_diameters = float(wake_half_width_diameters)
        if source is None:
            if foam_path is None:
                raise ValueError("foam_path is required when source is not supplied")
            source = OpenFOAMReferenceSource(foam_path)
        self._source = source
        self._mapping: np.ndarray | None = None
        self._mapping_method = ""
        self._mapping_max_distance = float("nan")
        self._reference_centers: np.ndarray | None = None
        self._cached_reference_time: float | None = None
        self._cached_reference_frame: ReferenceFrame | None = None

    def _match_time(self, physical_time: float) -> float:
        available = np.asarray(self._source.time_values, dtype=np.float64)
        if available.size == 0 or not np.all(np.isfinite(available)):
            raise ValueError("OpenFOAM source has no finite times")
        index = int(np.argmin(np.abs(available - physical_time)))
        matched = float(available[index])
        if abs(matched - physical_time) > self.time_tolerance:
            raise ValueError(
                f"OpenFOAM t={physical_time:g} is unavailable; closest is t={matched:g} "
                f"(tolerance {self.time_tolerance:g})"
            )
        return matched

    def _ensure_mapping(self, reference_centers: np.ndarray) -> None:
        if self._mapping is None:
            mapping, method, maximum = _cell_mapping(
                reference_centers, self._candidate_centers, self.center_tolerance
            )
            self._mapping = mapping
            self._mapping_method = method
            self._mapping_max_distance = maximum
            self._reference_centers = reference_centers.copy()
            return
        assert self._reference_centers is not None
        if reference_centers.shape != self._reference_centers.shape:
            raise ValueError("OpenFOAM internalMesh cell centers changed between reference frames")
        drift = np.linalg.norm(reference_centers - self._reference_centers, axis=1)
        if float(np.max(drift, initial=0.0)) > self.center_tolerance:
            raise ValueError("OpenFOAM internalMesh cell centers changed between reference frames")

    def evaluate(self, interior_state: object, physical_time: float) -> ReferenceGateResult:
        """Evaluate one candidate without retaining a tensor or gradient graph."""
        state = _as_cpu_numpy(interior_state, name="interior_state")
        if state.shape != (len(self._candidate_centers), 5):
            raise ValueError(
                "interior_state must have shape [N_cells,5] ordered as (u,v,w,p,T), got "
                f"{state.shape}"
            )
        matched_time = self._match_time(float(physical_time))
        if self._cached_reference_time == matched_time and self._cached_reference_frame is not None:
            frame = self._cached_reference_frame
        else:
            frame = self._source.frame(matched_time)
            self._cached_reference_time = matched_time
            self._cached_reference_frame = frame
        centers = _as_cpu_numpy(frame.centers, name="reference centers")
        volumes = _as_cpu_numpy(frame.volumes, name="reference volumes").reshape(-1)
        reference_u = _as_cpu_numpy(frame.velocity, name="reference U")
        reference_p_raw = _as_cpu_numpy(frame.pressure, name="reference p").reshape(-1)
        reference_t = _as_cpu_numpy(frame.temperature, name="reference T").reshape(-1)
        cell_count = len(centers)
        if (
            centers.shape != (cell_count, 3)
            or volumes.shape != (cell_count,)
            or reference_u.shape != (cell_count, 3)
            or reference_p_raw.shape != (cell_count,)
            or reference_t.shape != (cell_count,)
        ):
            raise ValueError("OpenFOAM reference frame contains inconsistent cell-array shapes")
        self._ensure_mapping(centers)
        assert self._mapping is not None
        candidate = state[self._mapping]
        candidate_u = candidate[:, :3]
        candidate_p, candidate_gauge = _gauge_pressure(candidate[:, 3], volumes)
        reference_p, reference_gauge = _gauge_pressure(reference_p_raw, volumes)
        candidate_t = candidate[:, 4]

        u_nrmse = float(
            np.sqrt(np.mean(np.sum((candidate_u - reference_u) ** 2, axis=1)))
            / self.velocity_scale
        )

        annulus_or_wake_needed = (
            self.thresholds.max_annulus_u_normalized_rmse is not None
            or self.thresholds.max_annulus_u_error_p95 is not None
            or self.thresholds.max_wake_u_normalized_rmse is not None
        )
        if annulus_or_wake_needed:
            annulus_mask, wake_mask = _cylinder_region_masks(
                centers,
                cylinder_center_x=self.cylinder_center_x,
                cylinder_center_y=self.cylinder_center_y,
                cylinder_diameter=self.cylinder_diameter,
                annulus_outer_radius_diameters=self.annulus_outer_radius_diameters,
                wake_end_x_diameters=self.wake_end_x_diameters,
                wake_half_width_diameters=self.wake_half_width_diameters,
            )
            annulus_u_nrmse, annulus_u_error_p95 = _region_u_error_metrics(
                reference_u, candidate_u, annulus_mask, self.velocity_scale
            )
            wake_u_nrmse, _ = _region_u_error_metrics(
                reference_u, candidate_u, wake_mask, self.velocity_scale
            )
            annulus_cell_count = int(np.count_nonzero(annulus_mask))
            wake_cell_count = int(np.count_nonzero(wake_mask))
        else:
            annulus_u_nrmse = float("nan")
            annulus_u_error_p95 = float("nan")
            wake_u_nrmse = float("nan")
            annulus_cell_count = 0
            wake_cell_count = 0
        p_nrmse = float(np.sqrt(np.mean((candidate_p - reference_p) ** 2)) / self.pressure_scale)
        t_nrmse = float(np.sqrt(np.mean((candidate_t - reference_t) ** 2)) / self.temperature_scale)
        u_corr = _correlation(reference_u, candidate_u)
        p_corr = _correlation(reference_p, candidate_p)
        t_corr = _correlation(reference_t, candidate_t)
        w_rms = float(np.sqrt(np.mean(candidate_u[:, 2] ** 2)))
        w_error_rms = float(np.sqrt(np.mean((candidate_u[:, 2] - reference_u[:, 2]) ** 2)))
        lower = self.temperature_min - self.temperature_bound_tolerance
        upper = self.temperature_max + self.temperature_bound_tolerance
        t_oob = float(np.mean((candidate_t < lower) | (candidate_t > upper)))
        if self.thresholds.max_recirculation_length_difference is not None:
            reference_recirc = _recirculation_length(
                centers,
                reference_u[:, 0],
                cylinder_center_x=self.cylinder_center_x,
                cylinder_center_y=self.cylinder_center_y,
                cylinder_diameter=self.cylinder_diameter,
                centerline_half_width=self.centerline_half_width,
            )
            candidate_recirc = _recirculation_length(
                centers,
                candidate_u[:, 0],
                cylinder_center_x=self.cylinder_center_x,
                cylinder_center_y=self.cylinder_center_y,
                cylinder_diameter=self.cylinder_diameter,
                centerline_half_width=self.centerline_half_width,
            )
            recirc_difference = abs(candidate_recirc - reference_recirc)
        else:
            reference_recirc = float("nan")
            candidate_recirc = float("nan")
            recirc_difference = float("nan")

        failures: list[str] = []
        checks = (
            ("U.normalized_rmse", u_nrmse, self.thresholds.max_u_normalized_rmse, "max"),
            (
                "annulus.U.normalized_rmse",
                annulus_u_nrmse,
                self.thresholds.max_annulus_u_normalized_rmse,
                "max",
            ),
            (
                "annulus.U.error_p95",
                annulus_u_error_p95,
                self.thresholds.max_annulus_u_error_p95,
                "max",
            ),
            (
                "wake.U.normalized_rmse",
                wake_u_nrmse,
                self.thresholds.max_wake_u_normalized_rmse,
                "max",
            ),
            ("p.normalized_rmse", p_nrmse, self.thresholds.max_p_normalized_rmse, "max"),
            ("T.normalized_rmse", t_nrmse, self.thresholds.max_t_normalized_rmse, "max"),
            ("U.correlation", u_corr, self.thresholds.min_u_correlation, "min"),
            ("p.correlation", p_corr, self.thresholds.min_p_correlation, "min"),
            ("T.correlation", t_corr, self.thresholds.min_t_correlation, "min"),
            ("w.rms", w_rms, self.thresholds.max_w_rms, "max"),
            (
                "w.error_rms",
                w_error_rms,
                self.thresholds.max_w_error_rms,
                "max",
            ),
            (
                "T.out_of_bounds_fraction",
                t_oob,
                self.thresholds.max_temperature_out_of_bounds_fraction,
                "max",
            ),
            (
                "recirculation.length_difference",
                recirc_difference,
                self.thresholds.max_recirculation_length_difference,
                "max",
            ),
        )
        for label, actual, limit, direction in checks:
            if limit is None:
                continue
            failed = actual > limit if direction == "max" else actual < limit
            if failed:
                operator = ">" if direction == "max" else "<"
                failures.append(f"{label}={actual:.6g} {operator} {limit:.6g}")

        return ReferenceGateResult(
            passed=not failures,
            physical_time=float(physical_time),
            reference_time=matched_time,
            u_normalized_rmse=u_nrmse,
            annulus_cell_count=annulus_cell_count,
            annulus_u_normalized_rmse=annulus_u_nrmse,
            annulus_u_error_p95=annulus_u_error_p95,
            wake_cell_count=wake_cell_count,
            wake_u_normalized_rmse=wake_u_nrmse,
            p_normalized_rmse=p_nrmse,
            t_normalized_rmse=t_nrmse,
            u_correlation=u_corr,
            p_correlation=p_corr,
            t_correlation=t_corr,
            w_rms=w_rms,
            w_error_rms=w_error_rms,
            temperature_out_of_bounds_fraction=t_oob,
            reference_recirculation_length=reference_recirc,
            candidate_recirculation_length=candidate_recirc,
            recirculation_length_difference=recirc_difference,
            candidate_pressure_gauge_removed=candidate_gauge,
            reference_pressure_gauge_removed=reference_gauge,
            mapping_method=self._mapping_method,
            mapping_max_center_distance=self._mapping_max_distance,
            failures=tuple(failures),
        )


def build_openfoam_reference_gate(
    *,
    config: ReferenceControlConfig,
    candidate_cell_centers: object,
    total_physical_steps: int,
    dt_value: float,
    case_dir: str | Path,
) -> Tuple[Any | None, str | None]:
    """Create the detached same-time acceptance gate and validate coverage."""

    if not config.enabled:
        return None, None

    foam_path = os.path.abspath(os.path.join(str(case_dir), config.foam_path))
    source = OpenFOAMReferenceSource(foam_path)
    required_time = total_physical_steps * dt_value
    available_times = source.time_values
    if available_times.size == 0 or np.min(np.abs(available_times - required_time)) > 1e-8:
        maximum = float(available_times.max()) if available_times.size else float("nan")
        raise ValueError(
            "OpenFOAM reference does not cover the requested ODIL trajectory: "
            f"need t={required_time:g}, available maximum={maximum:g}"
        )
    thresholds = ReferenceGateThresholds(
        max_u_normalized_rmse=config.max_u_normalized_rmse,
        max_annulus_u_normalized_rmse=config.max_annulus_u_normalized_rmse,
        max_annulus_u_error_p95=config.max_annulus_u_error_p95,
        max_wake_u_normalized_rmse=config.max_wake_u_normalized_rmse,
        max_p_normalized_rmse=config.max_p_normalized_rmse,
        max_t_normalized_rmse=config.max_t_normalized_rmse,
        min_u_correlation=config.min_u_correlation,
        min_p_correlation=config.min_p_correlation,
        min_t_correlation=config.min_t_correlation,
        max_w_rms=config.max_w_rms,
        max_w_error_rms=config.max_w_error_rms,
        max_temperature_out_of_bounds_fraction=(
            config.max_temperature_out_of_bounds_fraction
        ),
        max_recirculation_length_difference=(
            config.max_recirculation_length_difference
        ),
    )
    gate = OpenFOAMReferenceGate(
        foam_path=None,
        candidate_cell_centers=candidate_cell_centers,
        thresholds=thresholds,
        velocity_scale=config.geometry.velocity_scale,
        pressure_scale=config.geometry.pressure_scale,
        cylinder_center_x=config.geometry.cylinder_center_x,
        cylinder_center_y=config.geometry.cylinder_center_y,
        cylinder_diameter=config.geometry.cylinder_diameter,
        centerline_half_width=config.geometry.centerline_half_width,
        annulus_outer_radius_diameters=config.geometry.annulus_outer_radius_diameters,
        wake_end_x_diameters=config.geometry.wake_end_x_diameters,
        wake_half_width_diameters=config.geometry.wake_half_width_diameters,
        temperature_min=config.temperature_min,
        temperature_max=config.temperature_max,
        temperature_bound_tolerance=config.temperature_bound_tolerance,
        source=source,
    )
    return gate, foam_path


__all__ = [
    "OpenFOAMReferenceGate",
    "OpenFOAMReferenceSource",
    "ReferenceFrame",
    "ReferenceGateResult",
    "ReferenceGateThresholds",
    "ReferenceSource",
    "build_openfoam_reference_gate",
]
