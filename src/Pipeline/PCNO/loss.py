"""FVM-residual objective for physics-constrained neural operator (PCNO) training."""

from FVsolver.integrator import sum_time_loss


def fvm_objective(solver, *, phi_old_cpd_cell, phi_new_cpd_cell):
    """Evaluate the finite-volume residual and return ``(scalar_loss, phi_new_bced)``."""
    loss_by_time, phi_new_bced = solver(phi_old_cpd_cell, phi_new_cpd_cell)
    return sum_time_loss(loss_by_time), phi_new_bced
