"""Model wrapper: static mesh feature caching, architecture dispatch, and loss routing."""

import contextlib
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from FVdomain.SetBC.onthefly.BCbase import SOLVER_VARS, VAR_CONFIG
from FVsolver.integrator import Integrator
from NNmodels.common.attention import compute_graph_ptr
from NNmodels.common.message_passing import counts_to_indptr
from Utils.utilities import FaceType

_LEGACY_KEY_RENAMES = (
    ("processpr_list.", "processors."),
    (".gn_blocks_list.", ".gn_blocks."),
    (".TransBlock.Attn.", ".attn.attn."),
    (".TransBlock.", ".attn."),
)


def remap_legacy_key(key: str) -> str:
    for old, new in _LEGACY_KEY_RENAMES:
        key = key.replace(old, new)
    return key


def remap_legacy_state_dict(state_dict: dict) -> OrderedDict:
    """Map pre-refactor checkpoint parameter keys onto the current module layout."""
    cleaned = OrderedDict(
        (k, v) for k, v in state_dict.items() if not k.startswith("_delta_normalizer.")
    )
    out = OrderedDict((remap_legacy_key(k), v) for k, v in cleaned.items())
    if len(out) != len(cleaned):
        raise RuntimeError("Legacy key remap collided; refusing to load merged state_dict")
    meta = getattr(state_dict, "_metadata", None)
    if meta is not None:
        out._metadata = OrderedDict((remap_legacy_key(k), v) for k, v in meta.items())
    return out


class NNmodel(nn.Module):
    """Build node/edge features, step the chosen simulator, and evaluate the FVM or data loss."""

    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        if params.net in ("TransFVGN_v2", "TransFVGN"):
            from NNmodels.TransFVGN.model import Simulator
        elif params.net == "GNN":
            from NNmodels.GNN.model import Simulator
        elif params.net == "Transolver":
            from NNmodels.Transolver.model import Simulator
        elif params.net == "TransolverPlus":
            from NNmodels.TransolverPlus.model import Simulator
        else:
            raise ValueError(
                f"Unknown net {params.net!r}; expected TransFVGN_v2, GNN, Transolver, or TransolverPlus"
            )

        self.simulator = Simulator(
            message_passing_num=params.message_passing_num,
            node_input_size=params.num_channels,
            edge_input_size=params.num_channels + 4,
            node_output_size=params.phi_num_channels,
            hidden_size=params.hidden_size,
            params=params,
        )
        self.num_cell_types = len(FaceType)
        self.integrator = Integrator(params=params)

        if not getattr(params, "channel_order", None):
            params.channel_order = self._derive_channel_order(params)
        if not getattr(params, "phi_num_channels", None):
            params.phi_num_channels = len(params.channel_order)

        from Pipeline.NeuralOperator.loss import DataDrivenLoss

        self.data_driven_loss = DataDrivenLoss(params)
        self._dd_solver = None
        self._static_mesh_feat_cache: OrderedDict = OrderedDict()
        self.apply(self._init_weights)

    @staticmethod
    def _derive_channel_order(params):
        fvconfig = getattr(params, "fvconfig", {}) or {}
        solver_type = (fvconfig.get("solver", {}) or {}).get("type", "")
        solver_vars = SOLVER_VARS.get(solver_type)
        if not solver_vars:
            return ["u", "v", "w", "p_rgh", "T"]
        names: list[str] = []
        for var in solver_vars:
            ch = VAR_CONFIG.get(var, 1)
            if var == "U" and ch == 3:
                names.extend(["u", "v", "w"])
            elif ch == 1:
                names.append(var)
            else:
                names.extend([f"{var}_{i}" for i in range(ch)])
        return names

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_or_build_static_mesh_meta(self, fv_graph, graph_cell, graph_Index):
        """Cache static per-batch mesh encodings, CSR indptr, and ragged graph_ptr."""
        dev = graph_cell.x.device
        num_graphs = int(graph_Index.theta_PDE.shape[0])
        fourier_num_freqs = int(getattr(self.params, "fourier_num_freqs", 0))
        solver_obj = getattr(self.integrator, "_solver", None)
        geom_sig = getattr(solver_obj, "_geom_mesh_sig", None)
        batch_sig = getattr(fv_graph, "_mesh_batch_sig", None)
        n_cpd_total = int(graph_cell.cpd_cell_pos.shape[0])
        solver_cfg = (
            getattr(solver_obj, "fvconfig", {}).get("solver", {})
            if solver_obj is not None
            else {}
        ) or getattr(self.params, "fvconfig", {}).get("solver", {})
        mask_mode = solver_cfg.get("gnn_output_mask", "interior")

        sig_id = (
            batch_sig
            if batch_sig is not None
            else (geom_sig if geom_sig is not None else int(graph_cell.cpd_cell_pos.data_ptr()))
        )
        cache_key = (dev, num_graphs, n_cpd_total, sig_id, fourier_num_freqs, mask_mode)
        cached = self._static_mesh_feat_cache.get(cache_key)
        if cached is not None:
            self._static_mesh_feat_cache.move_to_end(cache_key)
            return cached

        cell_type_int = graph_cell.cell_type.long().squeeze(-1)
        cell_type_onehot = torch.nn.functional.one_hot(
            cell_type_int, num_classes=self.num_cell_types
        ).float()

        same_mesh = bool(getattr(fv_graph, "_same_mesh", False)) or (num_graphs == 1)
        cpd_ptr_raw = getattr(fv_graph, "_cpd_ptr", None)
        if cpd_ptr_raw is None and hasattr(fv_graph, "_hetero_data"):
            cpd_ptr_raw = getattr(fv_graph._hetero_data["cpd_cell"], "ptr", None)
        graph_ptr = compute_graph_ptr(
            graph_cell.batch,
            n_cpd_total,
            dev,
            num_graphs=num_graphs,
            equal_counts=same_mesh,
            graph_ptr=cpd_ptr_raw,
        )

        fourier_feat = None
        if fourier_num_freqs > 0:
            if same_mesh and num_graphs > 1 and n_cpd_total % num_graphs == 0:
                n_single = n_cpd_total // num_graphs
                f_single = self._compute_fourier_features(
                    graph_cell.cpd_cell_pos[:n_single], None, fourier_num_freqs
                )
                fourier_feat = f_single.repeat(num_graphs, 1)
            else:
                fourier_feat = self._compute_fourier_features(
                    graph_cell.cpd_cell_pos, graph_cell.batch, fourier_num_freqs
                )

        senders, receivers = graph_cell.neighbor_cpd_cell
        pos_diff = graph_cell.cpd_cell_pos[senders] - graph_cell.cpd_cell_pos[receivers]
        pos_norm = torch.linalg.vector_norm(pos_diff, dim=-1, keepdim=True)
        rel_pos_norm = torch.cat([pos_diff, pos_norm], dim=-1)

        gnn_edge_indptr = counts_to_indptr(graph_cell.gnn_edge_counts)

        output_mask = self._select_gnn_output_mask()
        solver_int_idx = getattr(solver_obj, "_interior_cell_indices", None)
        if mask_mode == "interior" and solver_int_idx is not None:
            output_indices = solver_int_idx
        elif mask_mode == "all":
            output_indices = None
        else:
            output_indices = output_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)

        entry = {
            "cell_type_onehot": cell_type_onehot,
            "fourier_feat": fourier_feat,
            "rel_pos_norm": rel_pos_norm,
            "gnn_edge_indptr": gnn_edge_indptr,
            "graph_ptr": graph_ptr,
            "num_graphs": num_graphs,
            "equal_counts": same_mesh,
            "output_indices": output_indices,
        }
        if len(self._static_mesh_feat_cache) >= 32:
            self._static_mesh_feat_cache.popitem(last=False)
        self._static_mesh_feat_cache[cache_key] = entry
        return entry

    def update_x_attr(self, graph_cell, graph_Index, static_meta=None):
        if static_meta is not None:
            cell_type_onehot = static_meta["cell_type_onehot"]
        else:
            cell_type_int = graph_cell.cell_type.long().squeeze(-1)
            cell_type_onehot = torch.nn.functional.one_hot(
                cell_type_int, num_classes=self.num_cell_types
            ).float()
        graph_cell.cell_type_onehot = cell_type_onehot

        theta_graph = graph_Index.theta_PDE
        if getattr(self.params, "theta_encoding", "row_std") == "signed_log":
            theta_norm = torch.sign(theta_graph) * torch.log10(1.0 + theta_graph.abs() / 1e-6) / 8.0
        else:
            theta_mean = theta_graph.mean(dim=-1, keepdim=True)
            theta_std = theta_graph.std(dim=-1, keepdim=True) + 1e-6
            theta_norm = (theta_graph - theta_mean) / theta_std
        theta_per_cell = (
            theta_norm[graph_cell.batch]
            if graph_cell.batch is not None
            else theta_norm.expand(graph_cell.x.shape[0], -1)
        )

        features = [graph_cell.x, graph_cell.init_x, cell_type_onehot, theta_per_cell]
        fourier_num_freqs = int(getattr(self.params, "fourier_num_freqs", 0))
        if fourier_num_freqs > 0:
            fourier_feat = (
                static_meta["fourier_feat"]
                if static_meta is not None and static_meta["fourier_feat"] is not None
                else self._compute_fourier_features(
                    graph_cell.cpd_cell_pos, graph_cell.batch, fourier_num_freqs
                )
            )
            features.append(fourier_feat)

        graph_cell.x = torch.cat(features, dim=1)
        return graph_cell

    @staticmethod
    def _compute_fourier_features(cpd_cell_pos, batch, num_freqs):
        """Domain-normalized sinusoidal positional features ``[N, 6 * num_freqs]``."""
        if batch is not None:
            pos_min = scatter(cpd_cell_pos, batch, dim=0, reduce="min")
            pos_max = scatter(cpd_cell_pos, batch, dim=0, reduce="max")
            pos_range = (pos_max - pos_min).clamp(min=1e-8)
            pos_norm = (cpd_cell_pos - pos_min[batch]) / pos_range[batch]
        else:
            pos_min = cpd_cell_pos.min(dim=0).values
            pos_max = cpd_cell_pos.max(dim=0).values
            pos_range = (pos_max - pos_min).clamp(min=1e-8)
            pos_norm = (cpd_cell_pos - pos_min) / pos_range
        freqs = torch.arange(1, num_freqs + 1, device=cpd_cell_pos.device, dtype=cpd_cell_pos.dtype)
        angles = 2.0 * math.pi * pos_norm.unsqueeze(-1) * freqs.view(1, 1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(cpd_cell_pos.shape[0], -1)

    def update_edge_attr(self, graph, static_meta=None):
        """Build undirected edge features ``[x_s - x_r, pos_s - pos_r, |pos_s - pos_r|]``."""
        senders, receivers = graph.neighbor_cpd_cell
        x_diff = graph.x[senders] - graph.x[receivers]
        if static_meta is not None:
            graph.edge_attr = torch.cat([x_diff, static_meta["rel_pos_norm"]], dim=-1)
            return graph
        pos_diff = graph.cpd_cell_pos[senders] - graph.cpd_cell_pos[receivers]
        pos_norm = torch.linalg.vector_norm(pos_diff, dim=-1, keepdim=True)
        graph.edge_attr = torch.cat([x_diff, pos_diff, pos_norm], dim=-1)
        return graph

    def _select_gnn_output_mask(self):
        solver = getattr(self.integrator, "_solver", None)
        solver_cfg = (
            getattr(solver, "fvconfig", {}).get("solver", {}) if solver is not None else {}
        ) or getattr(self.params, "fvconfig", {}).get("solver", {})
        mask_mode = solver_cfg.get("gnn_output_mask", "interior")
        if mask_mode == "interior":
            return self.integrator.mask_interior_cell
        if mask_mode == "non_empty":
            return self.integrator.mask_non_empty_cpd_cell
        if mask_mode == "non_cyclic":
            return self.integrator.mask_non_cyclic_cpd_cell & self.integrator.mask_non_empty_cpd_cell
        if mask_mode == "all":
            return torch.ones_like(self.integrator.mask_interior_cell, dtype=torch.bool)
        raise ValueError(
            f"Unknown solver.gnn_output_mask={mask_mode!r}; expected interior, non_empty, non_cyclic, or all"
        )

    def forward(self, fv_graph, is_training=True, loss_mode: str = "fvm"):
        self.integrator.register_properties(fv_graph, force_register=True)

        graph_cell = fv_graph.graph_cell
        graph_Index = fv_graph.graph_Index
        phi_old_cpd_cell = graph_cell.x.clone()

        static_meta = self._get_or_build_static_mesh_meta(fv_graph, graph_cell, graph_Index)
        graph_cell = self.update_x_attr(graph_cell, graph_Index, static_meta=static_meta)
        graph_cell = self.update_edge_attr(graph_cell, static_meta=static_meta)
        output_mask = self._select_gnn_output_mask()

        env_amp = os.environ.get("FVGN_AMP_GNN", None)
        amp_cfg = getattr(self.params, "amp_gnn", "fp32")
        if env_amp is not None:
            amp_on = env_amp.strip().lower() in ("1", "true", "yes", "bf16")
        elif isinstance(amp_cfg, bool):
            amp_on = amp_cfg
        else:
            amp_on = str(amp_cfg).strip().lower() in ("1", "true", "yes", "bf16")
        use_amp_gnn = amp_on and graph_cell.x.is_cuda
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp_gnn
            else contextlib.nullcontext()
        )
        sim_x = graph_cell.x.to(torch.bfloat16) if use_amp_gnn else graph_cell.x
        sim_edge_attr = graph_cell.edge_attr.to(torch.bfloat16) if use_amp_gnn else graph_cell.edge_attr

        with amp_ctx:
            delta_phi = self.simulator(
                sim_x,
                sim_edge_attr,
                edge_index=graph_cell.neighbor_cpd_cell,
                batch=graph_cell.batch,
                mask_output_cell=output_mask,
                pos=graph_cell.cpd_cell_pos,
                gnn_edge_counts=graph_cell.gnn_edge_counts,
                gnn_edge_indices=graph_cell.gnn_edge_indices,
                gnn_edge_sort_idx=graph_cell.gnn_edge_sort_idx,
                gnn_edge_indptr=static_meta["gnn_edge_indptr"],
                graph_ptr=static_meta["graph_ptr"],
                num_graphs=static_meta["num_graphs"],
                equal_counts=static_meta["equal_counts"],
                output_indices=static_meta["output_indices"],
            )
        if delta_phi.dtype != phi_old_cpd_cell.dtype:
            delta_phi = delta_phi.to(phi_old_cpd_cell.dtype)

        phi_new_cpd_cell = delta_phi + phi_old_cpd_cell

        if loss_mode == "fvm":
            from Pipeline.PCNO.loss import fvm_objective

            loss_total, phi_new_cpd_cell = fvm_objective(
                self.integrator,
                phi_old_cpd_cell=phi_old_cpd_cell,
                phi_new_cpd_cell=phi_new_cpd_cell,
            )
        elif loss_mode == "data_driven":
            from Pipeline.NeuralOperator.objective import data_driven_objective

            loss_total, phi_new_cpd_cell = data_driven_objective(
                dd_solver=self._dd_solver,
                data_driven_loss=self.data_driven_loss,
                integrator=self.integrator,
                graph_cell=graph_cell,
                phi_old_cpd_cell=phi_old_cpd_cell,
                delta_phi=delta_phi,
                phi_new_cpd_cell=phi_new_cpd_cell,
            )
        else:
            raise ValueError(f"Unknown loss_mode={loss_mode!r}; expected 'fvm' or 'data_driven'")

        return loss_total, phi_new_cpd_cell.detach()

    def load_checkpoint(self, ckpdir=None, *, optimizer=None, scheduler=None, device=None):
        """Load weights strictly (remapping legacy key names when present)."""
        dicts = torch.load(ckpdir, map_location=device)
        self.load_state_dict(remap_legacy_state_dict(dicts["model"]), strict=True)

        if optimizer is not None:
            if "optimizer0" in dicts:
                opts = optimizer if isinstance(optimizer, list) else [optimizer]
                for i, o in enumerate(opts):
                    if f"optimizer{i}" in dicts:
                        o.load_state_dict(dicts[f"optimizer{i}"])
            elif "optimizer_state_dict" in dicts and not isinstance(optimizer, list):
                optimizer.load_state_dict(dicts["optimizer_state_dict"])
            elif "optimizer" in dicts and not isinstance(optimizer, list):
                optimizer.load_state_dict(dicts["optimizer"])

        if scheduler is not None and "scheduler0" in dicts:
            scheds = scheduler if isinstance(scheduler, list) else [scheduler]
            for i, s in enumerate(scheds):
                if f"scheduler{i}" in dicts:
                    s.load_state_dict(dicts[f"scheduler{i}"])

    def save_checkpoint(self, path, optimizer=None, scheduler=None):
        to_save = {"model": self.state_dict()}
        if optimizer is not None:
            opts = optimizer if isinstance(optimizer, list) else [optimizer]
            for i, o in enumerate(opts):
                to_save[f"optimizer{i}"] = o.state_dict()
        if scheduler is not None:
            scheds = scheduler if isinstance(scheduler, list) else [scheduler]
            for i, s in enumerate(scheds):
                to_save[f"scheduler{i}"] = s.state_dict()
        torch.save(to_save, path)
