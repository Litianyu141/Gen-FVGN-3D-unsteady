"""Finite-volume discretisations, one module per OpenFOAM-style application."""

import importlib

SOLVER_MODULES = {
    "picoext": "picoext",
    "buoyantboussinesqpicoext": "buoyantBoussinesqpicoext",
    "buoyantboussinesqpicocmpt": "buoyantBoussinesqpicocmpt",
}


def create_solver(params):
    """Instantiate the discretisation selected by the case's ``[solver] type``."""
    solver_type = str(params.fvconfig["solver"]["type"])
    module_name = SOLVER_MODULES.get(solver_type.lower())
    if module_name is None:
        raise ValueError(f"unknown [solver] type {solver_type!r}; supported: {sorted(SOLVER_MODULES)}")
    return importlib.import_module(f"{__name__}.{module_name}").Solver(params)
