"""Temperature boundary conditions."""

from FVdomain.SetBC.onthefly.BCbase import VARType
from FVdomain.SetBC.onthefly.scalar import ScalarCyclic, ScalarFixedValue, ScalarZeroGradient


class Cyclic(ScalarCyclic):
    FIELD, VAR = "t_cpd_cell", VARType.TEMPERATURE


class Fixedvalue(ScalarFixedValue):
    FIELD, VAR = "t_cpd_cell", VARType.TEMPERATURE


class Zerogradient(ScalarZeroGradient):
    FIELD, VAR = "t_cpd_cell", VARType.TEMPERATURE


class Symmetry(ScalarZeroGradient):
    """Symmetry plane: zero normal gradient."""
    FIELD, VAR = "t_cpd_cell", VARType.TEMPERATURE
