"""Registry and factory for Warp-accelerated unsteady incompressible / Boussinesq FVM solvers."""

import importlib

SOLVER_MODULES = {
    "picoext": "FVsolver.applications.picoext",
    "picocmpt": "FVsolver.applications.picocmpt",
    "buoyantboussinesqpicoext": "FVsolver.applications.buoyantBoussinesqpicoext",
    "buoyantboussinesqpicocmpt": "FVsolver.applications.buoyantBoussinesqpicocmpt",
}


def create_solver(*args):
    """Instantiate the FVM solver configured in ``params.fvconfig["solver"]["type"]``."""
    if len(args) == 1:
        params = args[0]
        solver_type = params.fvconfig["solver"]["type"]
    elif len(args) == 2:
        solver_type, params = args
    else:
        raise TypeError(f"create_solver expects (params) or (solver_type, params), got {len(args)} args")

    key = str(solver_type).lower()
    module_path = SOLVER_MODULES.get(key)
    if module_path is None:
        raise ValueError(f"Unknown solver type {solver_type!r}; supported: {list(SOLVER_MODULES)}")
    return importlib.import_module(module_path).Solver(params)
