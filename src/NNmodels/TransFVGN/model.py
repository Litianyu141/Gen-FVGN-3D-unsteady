"""Hybrid message passing + Warp Physics-Attention (TransFVGN / TransFVGN_v2)."""

import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _act_checkpoint

try:
    import torch._inductor.config as _inductor_cfg
    if hasattr(_inductor_cfg, "triton") and hasattr(_inductor_cfg.triton, "mix_order_reduction"):
        _inductor_cfg.triton.mix_order_reduction = False
except Exception:
    pass

from NNmodels.common.attention import AttentionBlock, compute_graph_ptr
from NNmodels.common.message_passing import Decoder, Encoder, GnBlock, counts_to_indptr


class Processor(nn.Module):
    """Message-passing GN blocks followed by optional Physics-Attention."""

    def __init__(
        self,
        message_passing_num,
        hidden_size,
        slice_num=32,
        num_heads=8,
        plus=False,
        use_attention=True,
    ):
        super().__init__()
        if message_passing_num < 1:
            raise ValueError("message_passing_num must be greater than 0")
        self.gn_blocks = nn.ModuleList(
            [GnBlock(hidden_size=hidden_size) for _ in range(message_passing_num)]
        )
        self.attn = (
            AttentionBlock(
                num_heads=num_heads,
                hidden_dim=hidden_size,
                dropout=0.0,
                act="gelu",
                mlp_ratio=2,
                slice_num=slice_num,
                plus=plus,
            )
            if use_attention
            else None
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        gnn_edge_counts,
        gnn_edge_indices,
        gnn_edge_sort_idx,
        gnn_edge_indptr=None,
        num_graphs=None,
        graph_ptr=None,
    ):
        fx, f_edge_attr = x, edge_attr
        for blk in self.gn_blocks:
            fx, f_edge_attr = blk(
                fx,
                edge_index,
                f_edge_attr,
                gnn_edge_counts=gnn_edge_counts,
                gnn_edge_indices=gnn_edge_indices,
                gnn_edge_sort_idx=gnn_edge_sort_idx,
                gnn_edge_indptr=gnn_edge_indptr,
            )
        fx = x + fx
        if self.attn is not None:
            fx = self.attn(fx, batch=batch, num_graphs=num_graphs, graph_ptr=graph_ptr)
        return fx, f_edge_attr


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
        use_attention=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_output_size = node_output_size
        self.use_attention = use_attention
        self.encoder = Encoder(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            hidden_size=hidden_size,
        )
        self.processors = nn.ModuleList([
            Processor(
                message_passing_num=message_passing_num,
                hidden_size=hidden_size,
                slice_num=int(getattr(params, "slice_num", 32)),
                num_heads=int(getattr(params, "num_heads", 8)),
                plus=plus,
                use_attention=use_attention,
            )
            for _ in range(int(getattr(params, "attn_processor_num", 4)))
        ])
        self.decoder = Decoder(hidden_size=hidden_size, node_output_size=node_output_size)

    @torch.compile(mode="default", dynamic=True)
    def _compiled_forward(
        self,
        x,
        edge_attr,
        edge_index,
        batch,
        output_indices,
        gnn_edge_counts,
        gnn_edge_indices,
        gnn_edge_sort_idx,
        gnn_edge_indptr,
        graph_ptr,
        num_graphs: int,
    ):
        fx, f_edge_attr = self.encoder(x, edge_attr)
        for proc in self.processors:
            fx, f_edge_attr = proc(
                fx,
                edge_index,
                f_edge_attr,
                batch,
                gnn_edge_counts,
                gnn_edge_indices,
                gnn_edge_sort_idx,
                gnn_edge_indptr=gnn_edge_indptr,
                num_graphs=num_graphs,
                graph_ptr=graph_ptr,
            )
        if output_indices is None:
            return self.decoder(fx)
        decoded = self.decoder(fx.index_select(0, output_indices))
        out = fx.new_zeros((fx.shape[0], self.node_output_size), dtype=decoded.dtype)
        return out.index_copy(0, output_indices, decoded)

    def _checkpointed_forward(
        self,
        x,
        edge_attr,
        edge_index,
        batch,
        output_indices,
        gnn_edge_counts,
        gnn_edge_indices,
        gnn_edge_sort_idx,
        gnn_edge_indptr,
        graph_ptr,
        num_graphs: int,
    ):
        fx, f_edge_attr = self.encoder(x, edge_attr)
        for proc in self.processors:
            fx, f_edge_attr = _act_checkpoint(
                proc,
                fx,
                edge_index,
                f_edge_attr,
                batch,
                gnn_edge_counts,
                gnn_edge_indices,
                gnn_edge_sort_idx,
                gnn_edge_indptr,
                num_graphs,
                graph_ptr,
                use_reentrant=False,
            )
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
        """Return full-sized increment, zero outside ``mask_output_cell``."""
        if gnn_edge_indptr is None and gnn_edge_counts is not None:
            gnn_edge_indptr = counts_to_indptr(gnn_edge_counts)

        if output_indices is None and mask_output_cell is not None:
            if mask_output_cell.dtype == torch.bool:
                output_indices = mask_output_cell.view(-1).nonzero(as_tuple=False).squeeze(-1)
            else:
                output_indices = mask_output_cell.view(-1).long()

        if self.use_attention:
            graph_ptr = compute_graph_ptr(
                batch, x.shape[0], x.device,
                num_graphs=num_graphs, equal_counts=equal_counts, graph_ptr=graph_ptr,
            )
            B = int(graph_ptr.shape[0]) - 1
        else:
            B = 1

        env_ckpt = os.environ.get("NN_ACT_CHECKPOINT")
        if env_ckpt is not None:
            use_ckpt = env_ckpt == "1" and self.training and torch.is_grad_enabled()
        else:
            use_ckpt = False
            if self.training and torch.is_grad_enabled() and x.is_cuda and edge_index is not None:
                bytes_per_edge_hidden = (
                    38
                    if (x.dtype == torch.bfloat16 or torch.is_autocast_enabled())
                    else 12 * x.element_size()
                )
                gnn_bytes = (
                    int(edge_index.shape[1])
                    * int(self.hidden_size)
                    * len(self.processors)
                    * bytes_per_edge_hidden
                )
                fvm_bytes = int(x.shape[0]) * 8192
                total_mem = torch.cuda.get_device_properties(x.device).total_memory
                if (gnn_bytes + fvm_bytes) > int(0.72 * total_mem):
                    use_ckpt = True

        if use_ckpt:
            return self._checkpointed_forward(
                x, edge_attr, edge_index, batch, output_indices,
                gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx,
                gnn_edge_indptr, graph_ptr, B,
            )
        return self._compiled_forward(
            x, edge_attr, edge_index, batch, output_indices,
            gnn_edge_counts, gnn_edge_indices, gnn_edge_sort_idx,
            gnn_edge_indptr, graph_ptr, B,
        )
