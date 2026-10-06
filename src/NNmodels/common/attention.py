"""Physics-Attention (Transolver) and the attention block shared by the networks."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Linear, GELU, Linear."""

    def __init__(self, n_input, hidden_size, n_output):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_input, hidden_size), nn.GELU())
        self.linear_post = nn.Linear(hidden_size, n_output)

    def forward(self, x):
        return self.linear_post(self.linear_pre(x))


class PhysicsAttention(nn.Module):
    """Cells are softly assigned to ``slice_num`` slices per head; attention runs between the
    slice tokens and is scattered back to the cells with the same weights."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dim_head = heads, dim_head
        self.softmax = nn.Softmax(dim=-1)
        self.in_project_x = nn.Linear(dim, inner)
        self.in_project_fx = nn.Linear(dim, inner)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_qkv = nn.Linear(dim_head, dim_head * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    @staticmethod
    def _grid_layout(batch, n_total, device):
        """Number of graphs, cells per graph, and the padded-grid index of every cell (None
        when all graphs have the same number of cells).  A method of its own: inline in
        ``forward``, the graph break at ``int(batch.max())`` makes torch.compile (2.10) emit
        code with an unbound size symbol."""
        B = int(batch.max()) + 1
        counts = torch.bincount(batch, minlength=B)
        n_max = int(counts.max())
        if int(counts.min()) == n_max:
            return B, n_max, None
        offs = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        return B, n_max, batch * n_max + (torch.arange(n_total, device=device) - offs[batch])

    def forward(self, x, batch):
        N = x.shape[0]
        H, D = self.heads, self.dim_head
        # graphs of unequal size are padded to n_max cells; padded rows get zero slice weight
        B, n_max, pad_idx = self._grid_layout(batch, N, x.device)

        def to_grid(flat):
            if pad_idx is None:
                return flat.view(B, n_max, H, D).permute(0, 2, 1, 3)
            g = flat.new_zeros(B * n_max, H, D)
            g[pad_idx] = flat
            return g.view(B, n_max, H, D).permute(0, 2, 1, 3)

        x_mid = to_grid(self.in_project_x(x).view(N, H, D))                          # B H N D
        w = self.softmax(self.in_project_slice(x_mid) / self.temperature)            # B H N G
        src = to_grid(self.in_project_fx(x).view(N, H, D))
        if pad_idx is not None:
            keep = torch.zeros(B * n_max, dtype=torch.bool, device=x.device)
            keep[pad_idx] = True
            w = w * keep.view(B, n_max, 1, 1).permute(0, 2, 1, 3)

        slice_norm = w.sum(2)                                                        # B H G
        token = torch.einsum("bhnd,bhng->bhgd", src, w)
        token = token / (slice_norm.unsqueeze(-1) + 1e-5)
        q, k, v = self.to_qkv(token).chunk(3, dim=-1)
        out_token = torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * D ** -0.5), v)

        out = torch.einsum("bhgd,bhng->bhnd", out_token, w)                         # B H N D
        out = out.permute(0, 2, 1, 3).reshape(B * n_max, H * D)
        if pad_idx is not None:
            out = out[pad_idx]
        return self.to_out(out)


class AttentionBlock(nn.Module):
    """LayerNorm, Physics-Attention, LayerNorm, MLP; both stages residual."""

    def __init__(self, num_heads, hidden_dim, mlp_ratio=4, slice_num=32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                     slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)

    def forward(self, fx, batch):
        fx = self.attn(self.ln_1(fx), batch) + fx
        return self.mlp(self.ln_2(fx)) + fx
