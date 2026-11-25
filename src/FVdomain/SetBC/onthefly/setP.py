"""Kinematic pressure (p) boundary condition descriptors."""

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVdomain.SetBC.onthefly.scalar import (
    ScalarCalculated,
    ScalarCyclic,
    ScalarEmpty,
    ScalarFixedValue,
    ScalarZeroGradient,
)


class Empty(ScalarEmpty):
    VAR = VARType.PRESSURE


class Cyclic(ScalarCyclic):
    VAR = VARType.PRESSURE


class Fixedvalue(ScalarFixedValue):
    VAR = VARType.PRESSURE


class Zerogradient(ScalarZeroGradient):
    VAR = VARType.PRESSURE


class Symmetry(ScalarZeroGradient):
    VAR = VARType.PRESSURE


class Calculated(ScalarCalculated):
    VAR = VARType.PRESSURE
