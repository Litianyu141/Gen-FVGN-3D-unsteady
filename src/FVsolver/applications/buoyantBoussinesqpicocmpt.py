"""Unsteady buoyant Boussinesq FVM solver (compact deferred-correction gradient, Warp backend)."""

from FVsolver.applications.base_solver import BoussinesqSolver


class Solver(BoussinesqSolver):
    """Buoyant Boussinesq ``[u, v, w, p_rgh, T]`` solver with compact snGrad diffusion."""
