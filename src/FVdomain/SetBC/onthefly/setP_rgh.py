"""Buoyant modified pressure (p_rgh) boundary condition descriptors."""

from typing import Dict

from FVdomain.SetBC.onthefly.BCbase import BCType, BoundaryCondition, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.scalar import (
    ScalarCalculated,
    ScalarCyclic,
    ScalarEmpty,
    ScalarFixedValue,
    ScalarZeroGradient,
)


class Empty(ScalarEmpty):
    VAR = VARType.PRESSURE_RGH


class Cyclic(ScalarCyclic):
    VAR = VARType.PRESSURE_RGH


class Fixedvalue(ScalarFixedValue):
    VAR = VARType.PRESSURE_RGH


class Zerogradient(ScalarZeroGradient):
    VAR = VARType.PRESSURE_RGH


class Symmetry(ScalarZeroGradient):
    VAR = VARType.PRESSURE_RGH


class Calculated(ScalarCalculated):
    VAR = VARType.PRESSURE_RGH


class Fixedfluxpressure(BoundaryCondition):
    """Buoyant wall p_rgh condition (enforced in Warp via ``P_FIXEDFLUX`` / ``G_FIXEDFLUX_P``)."""

    APPLY_LAST = True

    def __init__(
        self,
        patch_name: str,
        patch_idx: int,
        boundary_info: Dict,
        parametric_manager: ParametricManager = None,
    ):
        raw_val = boundary_info.get("value", {"value_type": "uniform", "val": 0.0})
        self.value = parametric_manager.resolve_value(raw_val) if parametric_manager is not None else raw_val
        super().__init__(patch_name, patch_idx, VARType.PRESSURE_RGH, BCType.MIXED)

    def init_field(self, p_rgh_cpd_cell, fvpatch, **kwargs):
        s = fvpatch.start_idx_cpd_cell[self.patch_idx]
        e = fvpatch.end_idx_cpd_cell[self.patch_idx]
        p_rgh_cpd_cell[s:e, 0:1] = Functionbase(
            value_type=self.value["value_type"],
            value=self.value["val"],
            num_element=e - s,
            channel_dim=1,
        ).eval(device=p_rgh_cpd_cell.device)
        return p_rgh_cpd_cell
