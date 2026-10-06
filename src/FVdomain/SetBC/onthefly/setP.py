"""Kinematic pressure boundary conditions."""

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVdomain.SetBC.onthefly.scalar import ScalarZeroGradient


class Zerogradient(ScalarZeroGradient):
    FIELD, VAR = "p_cpd_cell", VARType.PRESSURE
