"""Unsteady incompressible Navier-Stokes FVM solver (compact deferred-correction gradient, Warp backend)."""

from FVsolver.applications.base_solver import BaseSolver


class Solver(BaseSolver):
    """Incompressible ``[u, v, w, p]`` solver with compact snGrad diffusion."""
