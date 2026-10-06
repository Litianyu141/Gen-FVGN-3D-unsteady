"""Encoder, message-passing block and decoder of the graph networks.

Edges carry two halves, the forward and the reverse direction of a face; node updates
aggregate over a CSR layout of the two-way edges (``gnn_edge_counts``, ``gnn_edge_indices``,
``gnn_edge_sort_idx``, built with the mesh).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import segment


def mlp(in_size, hidden_size, out_size, layer_norm=True, num_layer=2):
    """``num_layer`` Linear+GELU layers, a Linear output layer and an optional LayerNorm."""
    layers = [nn.Linear(in_size, hidden_size), nn.GELU()]
    for _ in range(num_layer - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
    layers.append(nn.Linear(hidden_size, out_size))
    if layer_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))
    return nn.Sequential(*layers)


class NodeBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, edge_attr, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx):
        """Sum the two-way edge features at each node, average that sum over the node's
        neighbours, and update the node from it and its own features."""
        indptr = F.pad(gnn_edge_counts.cumsum(0), (1, 0))
        half_dim = edge_attr.shape[1] // 2
        twoway_edge_attr = torch.stack([edge_attr[:, :half_dim], edge_attr[:, half_dim:]], dim=1).reshape(-1, half_dim)
        node_agg_received_edges = segment(twoway_edge_attr[gnn_edge_sort_idx], ptr=indptr, reduce="sum")
        node_avg_neighbor_node = segment(node_agg_received_edges[gnn_edge_indices], ptr=indptr, reduce="mean")
        return self.net(torch.cat([node_avg_neighbor_node, x], dim=1))


class EdgeBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, edge_index, edge_attr):
        senders, receivers = edge_index
        return self.net(torch.cat([x[senders], x[receivers], edge_attr], dim=1))


class Encoder(nn.Module):
    """Node MLP; edge MLP applied to both directions of each face (relative position negated)."""

    def __init__(self, node_input_size, edge_input_size, hidden_size):
        super().__init__()
        self.nb_encoder = mlp(node_input_size, hidden_size, hidden_size)
        self.eb_encoder = mlp(edge_input_size, hidden_size, hidden_size // 2)

    def forward(self, x, edge_attr):
        node_features = self.nb_encoder(x)
        reverse_edge_features = torch.cat((-edge_attr[:, :-1], edge_attr[:, -1:]), dim=1)
        twoway_edge_features = self.eb_encoder(torch.cat((edge_attr, reverse_edge_features), dim=0))
        edge_features = torch.cat(torch.chunk(twoway_edge_features, 2, dim=0), dim=1)
        return node_features, edge_features


class GnBlock(nn.Module):
    """Edge update, then node update; both residual."""

    def __init__(self, hidden_size):
        super().__init__()
        self.nb_module = NodeBlock(mlp(hidden_size + hidden_size // 2, hidden_size, hidden_size))
        self.eb_module = EdgeBlock(mlp(3 * hidden_size, hidden_size, hidden_size))

    def forward(self, x, edge_index, edge_attr, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx):
        updated_edge_attr = self.eb_module(x, edge_index, edge_attr)
        updated_x = self.nb_module(x, updated_edge_attr, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx)
        return x + updated_x, edge_attr + updated_edge_attr


class Decoder(nn.Module):
    def __init__(self, hidden_size, node_output_size):
        super().__init__()
        self.node_decode_module = mlp(hidden_size, hidden_size, node_output_size, layer_norm=False)

    def forward(self, x):
        return self.node_decode_module(x)
