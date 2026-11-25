"""Precomputed first-order weighted least-squares (WLSQ) normal matrices for mesh loading."""

import torch
from torch_geometric.utils import scatter


def compute_normal_matrix(node_pos: torch.Tensor, edge_index: torch.Tensor):
    """Assemble per-cell WLSQ normal matrices ``A`` [N, 3, 3] and weighted displacements ``B`` [2E, 3, 1]."""
    twoway_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    sender, receiver = twoway_edge_index[0], twoway_edge_index[1]
    diff = node_pos[sender] - node_pos[receiver]
    disp = diff.unsqueeze(2)
    weight = (1.0 / torch.norm(diff, dim=1, keepdim=True)).unsqueeze(2)
    A = scatter(
        torch.matmul(disp * weight, disp.transpose(1, 2)),
        receiver,
        dim=0,
        dim_size=node_pos.shape[0],
        reduce="sum",
    )
    B = weight * disp
    return A, B
