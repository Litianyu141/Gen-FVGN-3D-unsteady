"""The FVM-loss objective: the finite-volume residual of the network's own update."""


def fvm_objective(solver, *, phi_old_cpd_cell, phi_new_cpd_cell):
    """Mean over the batched graphs of the per-graph log residual, and the boundary-enforced
    new state."""
    loss_per_graph, phi_new_cpd_cell = solver(phi_old_cpd_cell, phi_new_cpd_cell)
    return loss_per_graph.mean(), phi_new_cpd_cell
