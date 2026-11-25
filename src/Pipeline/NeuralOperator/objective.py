"""Supervised data-loss objective: z-scored L2 error after Warp boundary enforcement."""

import json
from pathlib import Path

import torch

from Pipeline.NeuralOperator.loss import apply_std_floor, zscore


class DataDrivenObjective:
    """Apply the network increment on interior cells, enforce BCs in Warp, and score in z-scored units."""

    def __init__(self, fvm_solver, loss_fn):
        self.solver = fvm_solver
        self.loss_fn = loss_fn
        self._pending_case_params = None

    def inject_case_params(self, case_params_per_graph) -> None:
        """Set per-graph flow rates and rack temperature rise before Warp BC refresh."""
        bcpatch = self.solver.bcpatch_batch
        bc_groups = [bcpatch.first, bcpatch.coupled]
        for g_idx, case_params in enumerate(case_params_per_graph):
            if not case_params:
                continue
            flow_cold = case_params.get("flowRate_cold")
            flow_rack = case_params.get("flowRate_rack")
            delta_t = case_params.get("deltaT")
            for group in bc_groups:
                if g_idx >= len(group):
                    continue
                for bc in group[g_idx]:
                    if hasattr(bc, "volumetricFlowRate") and bc.volumetricFlowRate is not None:
                        if bc.patch_name == "coldair" and flow_cold is not None:
                            bc.volumetricFlowRate["val"] = flow_cold
                        if bc.patch_name in ("racks_inlet", "racks_outlet") and flow_rack is not None:
                            bc.volumetricFlowRate["val"] = flow_rack
                    if hasattr(bc, "flow_rate") and bc.patch_name in ("racks_inlet", "racks_outlet") and flow_rack is not None:
                        bc.flow_rate = flow_rack
                    if hasattr(bc, "deltaT") and bc.patch_name == "racks_outlet" and delta_t is not None:
                        if isinstance(bc.deltaT, dict):
                            bc.deltaT["val"] = delta_t
                        else:
                            bc.deltaT = delta_t

    def forward(self, phi_old_cpd_cell, delta_phi_cpd_cell, phi_GT_cpd_cell, mask_interior, batch_idx):
        phi_new = phi_old_cpd_cell.clone()
        phi_new[mask_interior] = delta_phi_cpd_cell[mask_interior] + phi_old_cpd_cell[mask_interior]

        if self._pending_case_params is not None:
            self.inject_case_params(self._pending_case_params)
            self._pending_case_params = None

        phi_new = self.solver.enforce_bcs_only(
            phi_old_cpd_cell=phi_old_cpd_cell,
            phi_new_cpd_cell=phi_new,
        )

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
    """Create the Warp boundary-enforcing data-loss objective for a Boussinesq case."""
    solver_type = str(params.fvconfig["solver"]["type"]).lower()
    if "buoyantboussinesq" not in solver_type:
        raise ValueError(f"the data loss supports the Boussinesq solvers only, not {solver_type!r}")
    log("[data_driven] Warp boundary conditions enforced on the predicted state")
    return DataDrivenObjective(fvm_solver, loss_fn)


def data_driven_objective(
    *,
    dd_solver=None,
    data_driven_loss=None,
    integrator=None,
    mask_interior=None,
    graph_cell=None,
    phi_old_cpd_cell=None,
    delta_phi=None,
    phi_new_cpd_cell=None,
):
    """Return ``(loss, new_state)`` for the supervised data-driven objective."""
    if mask_interior is None and integrator is not None:
        mask_interior = integrator.mask_interior_cell
    mask_bool = mask_interior.bool()

    if dd_solver is not None:
        return dd_solver.forward(
            phi_old_cpd_cell=phi_old_cpd_cell,
            delta_phi_cpd_cell=delta_phi,
            phi_GT_cpd_cell=graph_cell.phi_GT_cpd_cell,
            mask_interior=mask_bool,
            batch_idx=graph_cell.batch,
        )

    if data_driven_loss is None:
        raise ValueError("data_driven_objective requires either dd_solver or data_driven_loss")
    phi_pred = phi_new_cpd_cell if phi_new_cpd_cell is not None else (phi_old_cpd_cell + delta_phi)
    dd_mean = data_driven_loss.mean.to(phi_pred.device)
    dd_std = data_driven_loss.std.to(phi_pred.device)
    loss = data_driven_loss(
        phi_pred=zscore(phi_pred, dd_mean, dd_std),
        phi_ref=zscore(graph_cell.phi_GT_cpd_cell, dd_mean, dd_std),
        mask_interior=mask_bool,
        batch_idx=graph_cell.batch,
    )
    return loss, phi_pred.clone()


def load_channel_stats(loss_fn, stats_path, std_floor=None, log=None):
    """Load z-score statistics from ``statistics.json`` onto ``loss_fn``."""
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if std_floor:
        apply_std_floor(std, stats.get("channel_order"), std_floor, log=log)
    loss_fn.set_channel_stats(mean, std)


def load_case_parameters(dataset_dir):
    """Read ``case_parameters`` from ``meta.json`` under ``dataset_dir`` or nearby ``openfoam/``."""
    ds = Path(dataset_dir)
    candidates = (
        ds / "meta.json",
        ds.parent / "openfoam" / "meta.json",
        ds.parent.parent / "openfoam" / "meta.json",
        ds.parent.parent.parent / "openfoam" / "meta.json",
    )
    for meta_path in candidates:
        if meta_path.is_file():
            with open(meta_path) as f:
                return json.load(f).get("case_parameters", {})
    return {}


def case_parameters_for(case_name, case_parameters):
    """Lookup ``case_name`` in ``case_parameters`` (normalized to ``case_<n>``)."""
    try:
        norm_cn = "case_" + str(int(case_name.split("_")[-1]))
    except (ValueError, IndexError):
        norm_cn = case_name
    return case_parameters.get(norm_cn) or case_parameters.get(case_name)
