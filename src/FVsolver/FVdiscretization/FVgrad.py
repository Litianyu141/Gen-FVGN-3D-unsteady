"""First-order weighted least-squares (WLSQ) gradient reconstruction at cell centroids."""

import torch
from torch_geometric.utils import scatter


def compute_normal_matrix(node_pos, edge_index):
    """WLSQ normal matrices of a stencil.

    ``edge_index`` [2, E] lists every stencil pair once; both directions are assembled.
    Returns ``A`` [N, 3, 3] and the inverse-distance-weighted displacements ``B`` [2E, 3, 1]
    (forward edges first, flipped edges second).
    """
    twoway_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    sender, receiver = twoway_edge_index[0], twoway_edge_index[1]
    node_pos_diff_on_edge = node_pos[sender] - node_pos[receiver]

    displacement = node_pos_diff_on_edge.unsqueeze(2)
    weight = (1 / torch.norm(node_pos_diff_on_edge, dim=1, keepdim=True) ** 1).unsqueeze(2)
    A = scatter(
        torch.matmul(displacement * weight, displacement.transpose(1, 2)),
        receiver, dim=0, dim_size=node_pos.shape[0], reduce="sum",
    )
    B = weight * displacement
    return A, B


@torch.compile(mode="default", dynamic=True)
def gradient_reconstruction(phi_node, edge_index, mask_valid_node, precompute_Moments):
    """WLSQ gradient ``[N, C, 3]`` of a cell field ``[N, C]``.

    ``precompute_Moments`` is ``[A, B_one_way]`` from :func:`compute_normal_matrix`.  Cells
    outside ``mask_valid_node`` (the boundary ghost cells) get a zero gradient.
    """
    A, oneway_B = precompute_Moments
    twoway_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    sender, receiver = twoway_edge_index[0], twoway_edge_index[1]

    half = oneway_B.shape[0]
    B = torch.cat((oneway_B, oneway_B), dim=0)
    B[half:, 0:3] *= -1
    B_phi = scatter(
        B * (phi_node[sender] - phi_node[receiver]).unsqueeze(1),
        receiver, dim=0, dim_size=phi_node.shape[0], reduce="sum",
    )

    mask = mask_valid_node[:, None, None]
    eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype).expand(A.shape[0], -1, -1)
    A_solve = torch.where(mask, A, eye)
    B_solve = torch.where(mask, B_phi, torch.zeros_like(B_phi))
    return torch.linalg.solve_ex(A_solve, B_solve)[0].transpose(1, 2).clone()
