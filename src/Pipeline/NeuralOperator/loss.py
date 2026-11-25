"""Z-scored L2 loss of the data-loss baseline."""

from typing import Optional

import torch
from torch_geometric.utils import scatter


def zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def apply_std_floor(ch_std, channel_order, std_floor_cfg, *, log=None):
    """Raise the z-score std of nearly constant channels to ``[solver] datadriven_std_floor``."""
    for ch_name, floor_val in dict(std_floor_cfg or {}).items():
        idx = list(channel_order).index(ch_name)
        old_std = float(ch_std[idx])
        ch_std[idx] = max(old_std, float(floor_val))
        if log is not None:
            log(f"[datadriven_std_floor] channel {ch_name!r}: std {old_std:.4g} -> {float(ch_std[idx]):.4g}")
    return ch_std


class DataDrivenLoss:
    """Per graph, the sum over channels of the L2 norm over interior cells of the z-scored
    error; averaged over the graphs of the batch."""

    def __init__(self, params):
        self.channel_names = list(params.channel_order)
        self.loss_per_channel: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def set_channel_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.detach()
        self.std = std.detach()

    def __call__(self, phi_pred, phi_ref, mask_interior, batch_idx) -> torch.Tensor:
        pred_i = phi_pred[mask_interior]
        ref_i = phi_ref[mask_interior]
        b_i = batch_idx[mask_interior]
        num_graphs = int(b_i.max().item()) + 1
        per_graph_sum = scatter((pred_i - ref_i) ** 2, b_i, dim=0, dim_size=num_graphs, reduce="sum")
        per_graph_l2 = torch.sqrt(per_graph_sum + 1e-12)
        self.loss_per_channel = per_graph_l2.detach()
        return per_graph_l2.sum(dim=1).mean()

    def get_separated_losses(self) -> dict:
        mean_per_ch = self.loss_per_channel.mean(dim=0)
        return {name: mean_per_ch[i].item() for i, name in enumerate(self.channel_names)}
