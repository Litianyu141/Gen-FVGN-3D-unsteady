"""Temperature (T) boundary condition descriptors."""

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVdomain.SetBC.onthefly.scalar import (
    ScalarCalculated,
    ScalarCyclic,
    ScalarEmpty,
    ScalarFixedValue,
    ScalarZeroGradient,
)
from FVdomain.SetBC.onthefly.setRack import Rackoutlettemperature


class Empty(ScalarEmpty):
    VAR = VARType.TEMPERATURE


class Cyclic(ScalarCyclic):
    VAR = VARType.TEMPERATURE


class Fixedvalue(ScalarFixedValue):
    VAR = VARType.TEMPERATURE


class Zerogradient(ScalarZeroGradient):
    VAR = VARType.TEMPERATURE


class Symmetry(ScalarZeroGradient):
    VAR = VARType.TEMPERATURE


class Calculated(ScalarCalculated):
    VAR = VARType.TEMPERATURE


class Exprfixedvalue(Fixedvalue):
    """Fallback for exprFixedValue when no case-local ``bc/setT.py`` override is provided."""
