"""Unsteady incompressible Navier-Stokes FVM solver (extended WLSQ gradient coupling, Warp backend)."""

from FVsolver.applications.base_solver import BaseSolver


class Solver(BaseSolver):
    """Incompressible ``[u, v, w, p]`` solver with attached WLSQ gradients."""
