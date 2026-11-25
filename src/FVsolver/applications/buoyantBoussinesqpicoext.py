"""Unsteady buoyant Boussinesq FVM solver (extended WLSQ gradient coupling, Warp backend)."""

from FVsolver.applications.base_solver import BoussinesqSolver


class Solver(BoussinesqSolver):
    """Buoyant Boussinesq ``[u, v, w, p_rgh, T]`` solver with attached WLSQ gradients."""
