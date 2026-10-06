"""The trainable model: node and edge features, the selected network, and the objective."""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from FVsolver.applications import create_solver
from Utils.utilities import FaceType

# Parameter names of checkpoints written before the network modules were consolidated.
_LEGACY_KEY_RENAMES = (
    ("processpr_list.", "processors."),
    (".gn_blocks_list.", ".gn_blocks."),
    (".TransBlock.Attn.", ".attn.attn."),
    (".TransBlock.", ".attn."),
)


def remap_legacy_key(key):
    for old, new in _LEGACY_KEY_RENAMES:
        key = key.replace(old, new)
    return key


def remap_legacy_state_dict(state_dict):
    """Rename the parameters of an older checkpoint to the current module names."""
    out = OrderedDict((remap_legacy_key(k), v) for k, v in state_dict.items()
                      if not k.startswith("_delta_normalizer."))
    meta = getattr(state_dict, "_metadata", None)
    if meta is not None:
        out._metadata = OrderedDict((remap_legacy_key(k), v) for k, v in meta.items())
    return out


class NNmodel(nn.Module):
    """Predicts the state increment of every interior cell from the current state."""

    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        if params.net == "TransFVGN_v2":
            from NNmodels.TransFVGN.model import Simulator
        elif params.net == "GNN":
            from NNmodels.GNN.model import Simulator
        else:
            from NNmodels.Transolver.model import Simulator

        self.simulator = Simulator(
            node_input_size=params.num_channels,
            edge_input_size=params.num_channels + 4,  # + relative position (3) and its norm (1)
            node_output_size=params.phi_num_channels,
            params=params,
        )
        self.num_cell_types = len(FaceType)
        # the finite-volume discretisation: residual of the FVM loss, boundary conditions
        self.integrator = create_solver(params)

        if getattr(params, "channel_order", None) is None:
            params.channel_order = self._derive_channel_order(params)
        from Pipeline.NeuralOperator.loss import DataDrivenLoss
        self.data_driven_loss = DataDrivenLoss(params)
        self._dd_solver = None

        self.apply(self._init_weights)

    @staticmethod
    def _derive_channel_order(params):
        """Scalar channel names of the solver's state, e.g. ``[u, v, w, p_rgh, T]``."""
        from FVdomain.SetBC.onthefly.BCbase import SOLVER_VARS, VAR_CONFIG

        names = []
        for var in SOLVER_VARS[params.fvconfig["solver"]["type"]]:
            names.extend(["u", "v", "w"] if VAR_CONFIG[var] == 3 else [var])
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

    def update_x_attr(self, graph_cell, graph_Index):
        """Node features: state, initial state, cell type, normalised PDE coefficients and
        the Fourier encoding of the centroid."""
        cell_type_onehot = torch.nn.functional.one_hot(
            graph_cell.cell_type.long().squeeze(-1), num_classes=self.num_cell_types).float()
        graph_cell.cell_type_onehot = cell_type_onehot

        theta_graph = graph_Index.theta_PDE
        theta_mean = theta_graph.mean(dim=-1, keepdim=True)
        theta_std = theta_graph.std(dim=-1, keepdim=True) + 1e-6
        theta_per_cell = ((theta_graph - theta_mean) / theta_std)[graph_cell.batch]

        features = [graph_cell.x, graph_cell.init_x, cell_type_onehot, theta_per_cell]
        if self.params.fourier_num_freqs > 0:
            features.append(self._compute_fourier_features(
                graph_cell.cpd_cell_pos, graph_cell.batch, self.params.fourier_num_freqs))
        graph_cell.x = torch.cat(features, dim=1)
        return graph_cell

    @staticmethod
    def _compute_fourier_features(cpd_cell_pos, batch, num_freqs):
        """``sin``/``cos(2 pi k x)``, k = 1..num_freqs, of the per-graph min-max normalised
        centroid coordinates."""
        pos_min = scatter(cpd_cell_pos, batch, dim=0, reduce="min")
        pos_max = scatter(cpd_cell_pos, batch, dim=0, reduce="max")
        pos_range = (pos_max - pos_min).clamp(min=1e-8)
        pos_normalized = (cpd_cell_pos - pos_min[batch]) / pos_range[batch]
        freqs = torch.arange(1, num_freqs + 1, device=cpd_cell_pos.device, dtype=cpd_cell_pos.dtype)
        angles = 2.0 * math.pi * pos_normalized.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(cpd_cell_pos.shape[0], -1)

    @staticmethod
    def update_edge_attr(graph):
        """Edge features: feature difference, relative position and its length."""
        senders, receivers = graph.neighbor_cpd_cell
        pos_diff = graph.cpd_cell_pos[senders] - graph.cpd_cell_pos[receivers]
        pos_norm = torch.linalg.vector_norm(pos_diff, dim=-1, keepdim=True)
        graph.edge_attr = torch.cat([graph.x[senders] - graph.x[receivers], pos_diff, pos_norm], dim=-1)
        return graph

    def forward(self, fv_graph, loss_mode: str = "fvm"):
        """Return the loss and the new state ``[N_cpd, C]`` (boundary conditions applied)."""
        self.integrator.register_properties(fv_graph, force_register=True)
        graph_cell = fv_graph.graph_cell
        phi_old_cpd_cell = graph_cell.x.clone()
        graph_cell = self.update_x_attr(graph_cell, fv_graph.graph_Index)
        graph_cell = self.update_edge_attr(graph_cell)

        # zero on the ghost cells, which the boundary conditions fill
        delta_phi = self.simulator(
            graph_cell.x,
            graph_cell.edge_attr,
            edge_index=graph_cell.neighbor_cpd_cell,
            batch=graph_cell.batch,
            mask_output_cell=self.integrator.mask_interior_cell,
            gnn_edge_counts=graph_cell.gnn_edge_counts,
            gnn_edge_indices=graph_cell.gnn_edge_indices,
            gnn_edge_sort_idx=graph_cell.gnn_edge_sort_idx,
        )
        phi_new_cpd_cell = delta_phi + phi_old_cpd_cell

        if loss_mode == "fvm":
            from Pipeline.PCNO.loss import fvm_objective
            return fvm_objective(self.integrator, phi_old_cpd_cell=phi_old_cpd_cell,
                                 phi_new_cpd_cell=phi_new_cpd_cell)
        from Pipeline.NeuralOperator.objective import data_driven_objective
        return data_driven_objective(
            dd_solver=self._dd_solver,
            mask_interior=self.integrator.mask_interior_cell,
            graph_cell=graph_cell,
            phi_old_cpd_cell=phi_old_cpd_cell,
            delta_phi=delta_phi,
        )

    def load_checkpoint(self, path, device=None, optimizer=None):
        """Load the weights of ``path`` (and the optimizer state, when one is given)."""
        state = torch.load(path, map_location=device, weights_only=False)  # holds the optimizer state too
        self.load_state_dict(remap_legacy_state_dict(state["model"]), strict=True)
        if optimizer is not None and "optimizer0" in state:
            optimizer.load_state_dict(state["optimizer0"])

    def save_checkpoint(self, path, optimizer=None, scheduler=None):
        to_save = {"model": self.state_dict()}
        if optimizer is not None:
            for i, o in enumerate(optimizer if isinstance(optimizer, list) else [optimizer]):
                to_save[f"optimizer{i}"] = o.state_dict()
        torch.save(to_save, path)
