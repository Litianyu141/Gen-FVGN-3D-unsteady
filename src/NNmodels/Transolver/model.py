"""Transolver (``--net Transolver``): Physics-Attention blocks only, no mesh connectivity."""
import torch
import torch.nn as nn

from NNmodels.common.attention import MLP, AttentionBlock


class Simulator(nn.Module):
    def __init__(self, node_input_size, edge_input_size, node_output_size, params):
        super().__init__()
        self.node_output_size = node_output_size
        hidden_size = params.hidden_size
        self.preprocess = MLP(node_input_size, hidden_size * 2, hidden_size)
        self.blocks = nn.ModuleList([
            AttentionBlock(num_heads=params.num_heads, hidden_dim=hidden_size, mlp_ratio=2, slice_num=params.slice_num)
            for _ in range(params.attn_processor_num)])
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, node_output_size))
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
    def forward(self, x, edge_attr, edge_index, batch, mask_output_cell, gnn_edge_counts, gnn_edge_indices,
                gnn_edge_sort_idx):
        """Increment of every cell, zero outside ``mask_output_cell``."""
        fx = self.preprocess(x)
        for blk in self.blocks:
            fx = blk(fx, batch)
        out = fx.new_zeros((fx.shape[0], self.node_output_size))
        out[mask_output_cell] = self.decoder(fx[mask_output_cell])
        return out
