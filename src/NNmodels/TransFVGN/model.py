"""TransFVGN (``--net TransFVGN_v2``): message passing followed by Physics-Attention in every
processor; ``--net GNN`` is the same network without the attention."""
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _act_checkpoint

from NNmodels.common.attention import AttentionBlock
from NNmodels.common.message_passing import Decoder, Encoder, GnBlock

# NN_ACT_CHECKPOINT=1 recomputes the processors in the backward pass (less GPU memory)
_ACT_CHECKPOINT = os.environ.get("NN_ACT_CHECKPOINT", "0") == "1"


class Processor(nn.Module):
    """``message_passing_num`` GN blocks, then attention on the residual sum."""

    def __init__(self, message_passing_num, hidden_size, slice_num, num_heads, use_attention=True):
        super().__init__()
        self.gn_blocks = nn.ModuleList([GnBlock(hidden_size) for _ in range(message_passing_num)])
        self.attn = AttentionBlock(num_heads=num_heads, hidden_dim=hidden_size, mlp_ratio=2,
                                   slice_num=slice_num) if use_attention else None

    def forward(self, x, edge_index, edge_attr, batch, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx):
        fx, f_edge_attr = x, edge_attr
        for blk in self.gn_blocks:
            fx, f_edge_attr = blk(fx, edge_index, f_edge_attr, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx)
        fx = x + fx
        return (self.attn(fx, batch) if self.attn is not None else fx), f_edge_attr


class Simulator(nn.Module):
    use_attention = True

    def __init__(self, node_input_size, edge_input_size, node_output_size, params):
        super().__init__()
        self.node_output_size = node_output_size
        self.encoder = Encoder(node_input_size, edge_input_size, params.hidden_size)
        self.processors = nn.ModuleList([
            Processor(params.message_passing_num, params.hidden_size, params.slice_num, params.num_heads,
                      use_attention=self.use_attention)
            for _ in range(params.attn_processor_num)])
        self.decoder = Decoder(params.hidden_size, node_output_size)

    @torch.compile(mode="default", dynamic=True)
    def forward(self, x, edge_attr, edge_index, batch, mask_output_cell, gnn_edge_counts, gnn_edge_indices,
                gnn_edge_sort_idx):
        """Increment of every cell, zero outside ``mask_output_cell``."""
        fx, f_edge_attr = self.encoder(x, edge_attr)
        use_ckpt = _ACT_CHECKPOINT and self.training and torch.is_grad_enabled()
        for proc in self.processors:
            args = (fx, edge_index, f_edge_attr, batch, gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx)
            fx, f_edge_attr = _act_checkpoint(proc, *args, use_reentrant=False) if use_ckpt else proc(*args)
        out = fx.new_zeros((fx.shape[0], self.node_output_size))
        out[mask_output_cell] = self.decoder(fx[mask_output_cell])
        return out
