"""Warp-accelerated CSR message-passing blocks shared by GNN and TransFVGN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from NNmodels.common.message_passing_warp import (
    warp_edge_gather_add,
    warp_node_twohop_agg,
)


def counts_to_indptr(counts: torch.Tensor) -> torch.Tensor:
    """Convert per-node edge counts ``[N]`` to CSR row pointers ``[N + 1]``."""
    return F.pad(counts.cumsum(0), (1, 0))


def mlp(in_size, hidden_size, out_size, *, lay_norm=True, num_layer=2):
    layers = [nn.Linear(in_size, hidden_size), nn.GELU()]
    for _ in range(num_layer - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
    layers.append(nn.Linear(hidden_size, out_size))
    if lay_norm:
        layers.append(nn.LayerNorm(out_size))
    return nn.Sequential(*layers)


class NodeBlock(nn.Module):
    """Node update using fused two-hop Warp CSR aggregation."""

    def __init__(self, hidden_size, custom_func):
        super().__init__()
        self.net = custom_func
        self.hidden_size = hidden_size

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        gnn_edge_counts,
        gnn_edge_indices,
        gnn_edge_sort_idx,
        gnn_edge_indptr=None,
    ):
        if not edge_attr.is_cuda:
            raise RuntimeError("NodeBlock requires CUDA tensors (Warp backend).")
        if gnn_edge_indptr is None:
            gnn_edge_indptr = counts_to_indptr(gnn_edge_counts)
        node_avg_neighbor = warp_node_twohop_agg(
            edge_attr, gnn_edge_indptr, gnn_edge_indices, gnn_edge_sort_idx,
        )
        return self.net(torch.cat([node_avg_neighbor, x], dim=1))


class EdgeBlock(nn.Module):
    """Edge update via algebraically factorized GEMMs + Warp gather-add."""

    def __init__(self, hidden_size, custom_func):
        super().__init__()
        self.net = custom_func
        self.hidden_size = hidden_size

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        gnn_edge_indptr,
        gnn_edge_sort_idx,
    ):
        if not edge_attr.is_cuda:
            raise RuntimeError("EdgeBlock requires CUDA tensors (Warp backend).")
        senders, receivers = edge_index
        lin0 = self.net[0]
        H = self.hidden_size
        W = lin0.weight
        W_sr = torch.cat((W[:, :H], W[:, H : 2 * H]), dim=0)
        h_sr = F.linear(x, W_sr)
        h_e = F.linear(edge_attr, W[:, 2 * H :], lin0.bias)
        out = warp_edge_gather_add(
            h_sr, h_e, senders, receivers, gnn_edge_indptr, gnn_edge_sort_idx,
        )
        for idx in range(1, len(self.net)):
            out = self.net[idx](out)
        return out


class Encoder(nn.Module):
    """Encode node features and undirected edges (both orientations concatenated)."""

    def __init__(self, node_input_size, edge_input_size, hidden_size):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even")
        self.nb_encoder = mlp(node_input_size, hidden_size, hidden_size)
        self.eb_encoder = mlp(edge_input_size, hidden_size, hidden_size // 2)

    def forward(self, x, edge_attr):
        node_features = self.nb_encoder(x)
        rev = torch.cat((-edge_attr[:, :-1], edge_attr[:, -1:]), dim=1)
        twoway = self.eb_encoder(torch.cat((edge_attr, rev), dim=0))
        edge_features = torch.cat(torch.chunk(twoway, 2, dim=0), dim=1)
        return node_features, edge_features


class GnBlock(nn.Module):
    """One residual message-passing step: edge update followed by node update."""

    def __init__(self, hidden_size):
        super().__init__()
        self.nb_module = NodeBlock(hidden_size, mlp(hidden_size + hidden_size // 2, hidden_size, hidden_size))
        self.eb_module = EdgeBlock(hidden_size, mlp(3 * hidden_size, hidden_size, hidden_size))

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        gnn_edge_counts,
        gnn_edge_indices,
        gnn_edge_sort_idx,
        gnn_edge_indptr=None,
    ):
        if gnn_edge_indptr is None:
            gnn_edge_indptr = counts_to_indptr(gnn_edge_counts)
        de = self.eb_module(
            x, edge_index, edge_attr,
            gnn_edge_indptr=gnn_edge_indptr,
            gnn_edge_sort_idx=gnn_edge_sort_idx,
        )
        dx = self.nb_module(
            x, edge_index, de,
            gnn_edge_counts=gnn_edge_counts,
            gnn_edge_indices=gnn_edge_indices,
            gnn_edge_sort_idx=gnn_edge_sort_idx,
            gnn_edge_indptr=gnn_edge_indptr,
        )
        return x + dx, edge_attr + de


class Decoder(nn.Module):
    def __init__(self, hidden_size, node_output_size):
        super().__init__()
        self.node_decode_module = mlp(hidden_size, hidden_size, node_output_size, lay_norm=False)

    def forward(self, x):
        return self.node_decode_module(x)
