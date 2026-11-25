"""Pure Transolver (Wu et al., ICML 2024) using Warp ragged Physics-Attention."""

import torch
import torch.nn as nn

from NNmodels.common.attention import AttentionBlock, MLP, compute_graph_ptr


class Simulator(nn.Module):
    _accepts_cached_graph_meta = True

    def __init__(
        self,
        message_passing_num,
        node_input_size,
        edge_input_size,
        node_output_size,
        hidden_size,
        params,
        plus=False,
    ):
        super().__init__()
        self.node_output_size = node_output_size
        self.preprocess = MLP(
            node_input_size, hidden_size * 2, hidden_size,
            n_layers=0, res=False, act="gelu",
        )
        self.blocks = nn.ModuleList([
            AttentionBlock(
                num_heads=int(getattr(params, "num_heads", 8)),
                hidden_dim=hidden_size,
                dropout=0.0,
                act="gelu",
                mlp_ratio=2,
                slice_num=int(getattr(params, "slice_num", 32)),
                plus=plus,
            )
            for _ in range(int(getattr(params, "attn_processor_num", 4)))
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, node_output_size),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.compile(mode="default", dynamic=True)
    def _compiled_forward(self, x, batch, output_indices, graph_ptr, num_graphs: int):
        fx = self.preprocess(x)
        for blk in self.blocks:
            fx = blk(fx, batch=batch, num_graphs=num_graphs, graph_ptr=graph_ptr)
        if output_indices is None:
            return self.decoder(fx)
        decoded = self.decoder(fx.index_select(0, output_indices))
        out = fx.new_zeros((fx.shape[0], self.node_output_size), dtype=decoded.dtype)
        return out.index_copy(0, output_indices, decoded)

    def forward(
        self,
        x,
        edge_attr,
        edge_index=None,
        batch=None,
        mask_output_cell=None,
        pos=None,
        gnn_edge_counts=None,
        gnn_edge_indices=None,
        gnn_edge_sort_idx=None,
        gnn_edge_indptr=None,
        num_graphs: int = None,
        equal_counts: bool = False,
        output_indices=None,
        graph_ptr=None,
    ):
        if output_indices is None and mask_output_cell is not None:
            if mask_output_cell.dtype == torch.bool:
                output_indices = mask_output_cell.view(-1).nonzero(as_tuple=False).squeeze(-1)
            else:
                output_indices = mask_output_cell.view(-1).long()

        graph_ptr = compute_graph_ptr(
            batch, x.shape[0], x.device,
            num_graphs=num_graphs, equal_counts=equal_counts, graph_ptr=graph_ptr,
        )
        B = int(graph_ptr.shape[0]) - 1
        return self._compiled_forward(x, batch, output_indices, graph_ptr, B)
