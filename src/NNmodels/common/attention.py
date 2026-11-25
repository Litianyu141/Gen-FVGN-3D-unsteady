"""Warp-accelerated ragged Physics-Attention shared across all model families."""

from typing import Optional

import torch
import torch.nn as nn

from NNmodels.common.attention_warp import ragged_physics_attention_warp

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, hidden_size, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Unknown activation {act!r}")
        act_cls = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, hidden_size), act_cls())
        self.linear_post = nn.Linear(hidden_size, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


def compute_graph_ptr(
    batch: Optional[torch.Tensor],
    n_total: int,
    device: torch.device,
    num_graphs: Optional[int] = None,
    equal_counts: bool = False,
    graph_ptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build CSR ``graph_ptr[B + 1]`` (``int32`` on ``device``) with zero CPU-GPU syncs."""
    if graph_ptr is not None:
        if graph_ptr.device == device and graph_ptr.dtype == torch.int32:
            return graph_ptr
        return graph_ptr.to(device=device, dtype=torch.int32)

    if batch is None or num_graphs == 1:
        ptr = torch.zeros((2,), dtype=torch.int32, device=device)
        ptr[1] = int(n_total)
        return ptr

    if equal_counts and num_graphs is not None and n_total % int(num_graphs) == 0:
        B = int(num_graphs)
        return torch.arange(0, B + 1, dtype=torch.int32, device=device) * (n_total // B)

    counts = (
        torch.bincount(batch)
        if num_graphs is None
        else torch.bincount(batch, minlength=int(num_graphs))
    )
    ptr = torch.zeros((counts.shape[0] + 1,), dtype=torch.int32, device=device)
    ptr[1:] = counts.cumsum(0).to(torch.int32)
    return ptr


class PhysicsAttention(nn.Module):
    """Fused Warp ragged Physics-Attention (Transolver & Transolver++)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, plus=False):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dim_head, self.plus = heads, dim_head, plus
        self.slice_num = slice_num
        self.softmax = nn.Softmax(dim=-1)

        self.in_project_x = nn.Linear(dim, inner)
        if not plus:
            self.in_project_fx = nn.Linear(dim, inner)
            self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        else:
            self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
            self.proj_temperature = nn.Sequential(
                nn.Linear(dim_head, slice_num), nn.GELU(),
                nn.Linear(slice_num, 1), nn.GELU(),
            )
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_qkv = nn.Linear(dim_head, dim_head * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        equal_counts: bool = False,
        graph_ptr: Optional[torch.Tensor] = None,
        gumbel_seed: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("PhysicsAttention requires CUDA tensors (Warp backend).")

        if gumbel_seed is None:
            gumbel_seed = getattr(self, "_fixed_gumbel_seed", None)

        N, _ = x.shape
        H, D = self.heads, self.dim_head
        ptr = compute_graph_ptr(
            batch, N, x.device,
            num_graphs=num_graphs, equal_counts=equal_counts, graph_ptr=graph_ptr,
        )

        x_mid = self.in_project_x(x).view(N, H, D)
        if self.plus:
            temp = torch.clamp(
                self.proj_temperature(x_mid) + self.bias.view(1, H, 1),
                min=0.01,
            ).squeeze(-1)
            src = None
        else:
            temp = self.temperature.view(H)
            src = self.in_project_fx(x).view(N, H, D)

        out = ragged_physics_attention_warp(
            x_mid=x_mid,
            src=src,
            W_slice=self.in_project_slice.weight,
            b_slice=self.in_project_slice.bias,
            temp=temp,
            to_qkv_fn=self.to_qkv,
            graph_ptr=ptr,
            plus=self.plus,
            gumbel_seed=gumbel_seed,
        )
        return self.to_out(out)


RaggedPhysicsAttention = PhysicsAttention


class AttentionBlock(nn.Module):
    """LayerNorm -> PhysicsAttention -> LayerNorm -> MLP, both residual."""

    def __init__(self, num_heads, hidden_dim, dropout=0.0, act="gelu",
                 mlp_ratio=4, slice_num=32, plus=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, plus=plus,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)

    def forward(
        self,
        fx: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        equal_counts: bool = False,
        graph_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fx = self.attn(
            self.ln_1(fx),
            batch=batch,
            num_graphs=num_graphs,
            equal_counts=equal_counts,
            graph_ptr=graph_ptr,
        ) + fx
        return self.mlp(self.ln_2(fx)) + fx
