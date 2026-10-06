"""The data-loss objective: z-scored L2 distance to the CFD labels after boundary enforcement."""

import json
from pathlib import Path

import torch

from Pipeline.NeuralOperator.loss import apply_std_floor, zscore


class DataDrivenObjective:
    """Applies the network increment to the interior cells, imposes the case's boundary
    conditions with the FVM solver's own boundary objects, and scores the state against the
    labels in z-scored units.  The state is ``[u, v, w, p_rgh, T]``."""

    def __init__(self, fvm_solver, loss_fn):
        self.solver = fvm_solver
        self.loss_fn = loss_fn
        self._pending_case_params = None

    def inject_case_params(self, case_params_per_graph) -> None:
        """Set the flow rates and rack temperature rise of each graph's data-centre case
        (entry ``i`` for graph ``i`` of the batch; None leaves a graph as it is)."""
        for case_params, bc_list in zip(case_params_per_graph, self.solver.bcpatch_batch.first):
            if not case_params:
                continue
            flow_cold = case_params.get("flowRate_cold")
            flow_rack = case_params.get("flowRate_rack")
            delta_t = case_params.get("deltaT")
            for bc in bc_list:
                if hasattr(bc, "volumetricFlowRate"):
                    if bc.patch_name == "coldair" and flow_cold is not None:
                        bc.volumetricFlowRate["val"] = flow_cold
                    if bc.patch_name in ("racks_inlet", "racks_outlet") and flow_rack is not None:
                        bc.volumetricFlowRate["val"] = flow_rack
                if hasattr(bc, "deltaT") and bc.patch_name == "racks_outlet" and delta_t is not None:
                    bc.deltaT = delta_t

    def forward(self, phi_old_cpd_cell, delta_phi_cpd_cell, phi_GT_cpd_cell, mask_interior, batch_idx):
        phi_new = phi_old_cpd_cell.clone()
        phi_new[mask_interior] = delta_phi_cpd_cell[mask_interior] + phi_old_cpd_cell[mask_interior]

        if self._pending_case_params is not None:
            self.inject_case_params(self._pending_case_params)
            self._pending_case_params = None

        fvfield = {
            "uvw_cpd_cell": phi_new[:, :3],
            "p_rgh_cpd_cell": phi_new[:, 3:4],
            "t_cpd_cell": phi_new[:, 4:5],
        }
        fvfield = self.solver.enforce_bc_1st(fvfield)
        fvfield = self.solver.enforce_cyclic(fvfield)
        fvfield = self.solver.enforce_mixed(fvfield)
        phi_new = torch.cat([fvfield["uvw_cpd_cell"], fvfield["p_rgh_cpd_cell"], fvfield["t_cpd_cell"]], dim=-1)

        dd_mean = self.loss_fn.mean.to(phi_new.device)
        dd_std = self.loss_fn.std.to(phi_new.device)
        loss = self.loss_fn(
            phi_pred=zscore(phi_new, dd_mean, dd_std),
            phi_ref=zscore(phi_GT_cpd_cell, dd_mean, dd_std),
            mask_interior=mask_interior,
            batch_idx=batch_idx,
        )
        return loss, phi_new.clone()


def create_dd_solver(params, fvm_solver, loss_fn, log=print):
    """The boundary-enforcing data-loss objective of a Boussinesq case."""
    solver_type = str(params.fvconfig["solver"]["type"]).lower()
    if "buoyantboussinesq" not in solver_type:
        raise ValueError(f"the data loss supports the Boussinesq solvers only, not {solver_type!r}")
    log("[data_driven] boundary conditions enforced on the predicted state")
    return DataDrivenObjective(fvm_solver, loss_fn)


def data_driven_objective(*, dd_solver, mask_interior, graph_cell, phi_old_cpd_cell, delta_phi):
    """Return ``(loss, new state)`` of the data-loss family."""
    return dd_solver.forward(
        phi_old_cpd_cell=phi_old_cpd_cell,
        delta_phi_cpd_cell=delta_phi,
        phi_GT_cpd_cell=graph_cell.phi_GT_cpd_cell,
        mask_interior=mask_interior.bool(),
        batch_idx=graph_cell.batch,
    )


def load_channel_stats(loss_fn, stats_path, std_floor=None, log=None):
    """Give ``loss_fn`` the z-score statistics of ``statistics.json`` (``mean``, ``std`` per
    channel), with the ``[solver] datadriven_std_floor`` of the training configuration."""
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if std_floor:
        apply_std_floor(std, stats.get("channel_order"), std_floor, log=log)
    loss_fn.set_channel_stats(mean, std)


def load_case_parameters(dataset_dir):
    """Per-case boundary parameters, ``case_parameters`` of ``<dataset_dir>/meta.json``."""
    meta_path = Path(dataset_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f).get("case_parameters", {})


def case_parameters_for(case_name, case_parameters):
    """Entry of ``case_name`` in ``case_parameters`` (keys ``case_<n>``), or None."""
    try:
        norm_cn = "case_" + str(int(case_name.split("_")[-1]))
    except (ValueError, IndexError):
        norm_cn = case_name
    return case_parameters.get(norm_cn) or case_parameters.get(case_name)
