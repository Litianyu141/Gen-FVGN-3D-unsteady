"""Zero-atomic NVIDIA Warp message-passing kernels with full ``torch.compile`` support."""

from typing import Dict, Tuple

import torch
import warp as wp

wp.config.quiet = True
wp.set_module_options({"enable_backward": False})


def _wp_from_torch(t: torch.Tensor, dtype=None):
    return wp.from_torch(t, dtype=dtype, requires_grad=False)


def _make_mp_kernels(io_t, acc_t):
    @wp.kernel
    def edge_gather_add_fwd_kernel(
        h_sr: wp.array2d(dtype=io_t),
        h_e: wp.array2d(dtype=io_t),
        senders: wp.array(dtype=wp.int64),
        receivers: wp.array(dtype=wp.int64),
        H: int,
        out: wp.array2d(dtype=io_t),
    ):
        e, c = wp.tid()
        u = int(senders[e])
        v = int(receivers[e])
        val = acc_t(h_e[e, c]) + acc_t(h_sr[u, c]) + acc_t(h_sr[v, H + c])
        out[e, c] = io_t(val)

    @wp.kernel
    def edge_gather_add_bwd_sr_kernel(
        grad_out: wp.array2d(dtype=io_t),
        indptr: wp.array(dtype=wp.int64),
        sort_idx: wp.array(dtype=wp.int64),
        H: int,
        grad_h_sr: wp.array2d(dtype=io_t),
    ):
        i, c = wp.tid()
        start = int(indptr[i])
        end = int(indptr[i + 1])
        acc_s = acc_t(0.0)
        acc_r = acc_t(0.0)
        for k in range(start, end):
            sidx = int(sort_idx[k])
            e = sidx >> 1
            if (sidx & 1) == 1:
                acc_s += acc_t(grad_out[e, c])
            else:
                acc_r += acc_t(grad_out[e, c])
        grad_h_sr[i, c] = io_t(acc_s)
        grad_h_sr[i, H + c] = io_t(acc_r)

    @wp.kernel
    def node_stage1_sum_fwd_kernel(
        twoway_edge_attr: wp.array2d(dtype=io_t),
        indptr: wp.array(dtype=wp.int64),
        sort_idx: wp.array(dtype=wp.int64),
        node_agg: wp.array2d(dtype=io_t),
    ):
        i, c = wp.tid()
        start = int(indptr[i])
        end = int(indptr[i + 1])
        acc = acc_t(0.0)
        for k in range(start, end):
            acc += acc_t(twoway_edge_attr[int(sort_idx[k]), c])
        node_agg[i, c] = io_t(acc)

    @wp.kernel
    def node_stage2_mean_fwd_kernel(
        node_agg: wp.array2d(dtype=io_t),
        indptr: wp.array(dtype=wp.int64),
        edge_indices: wp.array(dtype=wp.int64),
        out: wp.array2d(dtype=io_t),
    ):
        i, c = wp.tid()
        start = int(indptr[i])
        end = int(indptr[i + 1])
        deg = end - start
        if deg <= 0:
            out[i, c] = io_t(acc_t(0.0))
            return
        acc = acc_t(0.0)
        for k in range(start, end):
            acc += acc_t(node_agg[int(edge_indices[k]), c])
        out[i, c] = io_t(acc / acc_t(deg))

    @wp.kernel
    def node_stage2_mean_bwd_kernel(
        grad_out: wp.array2d(dtype=io_t),
        indptr: wp.array(dtype=wp.int64),
        edge_indices: wp.array(dtype=wp.int64),
        grad_node_agg: wp.array2d(dtype=io_t),
    ):
        i, c = wp.tid()
        start = int(indptr[i])
        end = int(indptr[i + 1])
        acc = acc_t(0.0)
        for k in range(start, end):
            j = int(edge_indices[k])
            deg_j = int(indptr[j + 1]) - int(indptr[j])
            if deg_j > 0:
                acc += acc_t(grad_out[j, c]) / acc_t(deg_j)
        grad_node_agg[i, c] = io_t(acc)

    @wp.kernel
    def node_stage1_sum_bwd_kernel(
        grad_node_agg: wp.array2d(dtype=io_t),
        indptr: wp.array(dtype=wp.int64),
        sort_idx: wp.array(dtype=wp.int64),
        grad_twoway_edge_attr: wp.array2d(dtype=io_t),
    ):
        i, c = wp.tid()
        start = int(indptr[i])
        end = int(indptr[i + 1])
        val = grad_node_agg[i, c]
        for k in range(start, end):
            grad_twoway_edge_attr[int(sort_idx[k]), c] = val

    return (
        edge_gather_add_fwd_kernel,
        edge_gather_add_bwd_sr_kernel,
        node_stage1_sum_fwd_kernel,
        node_stage2_mean_fwd_kernel,
        node_stage2_mean_bwd_kernel,
        node_stage1_sum_bwd_kernel,
    )


_KERNEL_REGISTRY: Dict[torch.dtype, Tuple] = {}
_DTYPE_MAP = {
    torch.float32: (wp.float32, wp.float32),
    torch.float64: (wp.float64, wp.float64),
    torch.bfloat16: (wp.bfloat16, wp.float32),
}


def _get_kernels(dtype: torch.dtype):
    ks = _KERNEL_REGISTRY.get(dtype)
    if ks is None:
        if dtype not in _DTYPE_MAP:
            raise TypeError(f"Unsupported dtype for Warp message passing: {dtype}")
        wp.init()
        io_t, acc_t = _DTYPE_MAP[dtype]
        ks = _make_mp_kernels(io_t, acc_t)
        _KERNEL_REGISTRY[dtype] = ks
    return ks


@torch.library.custom_op("fvgn_mp::edge_gather_add", mutates_args=())
def _edge_gather_add_op(
    h_sr: torch.Tensor,
    h_e: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    indptr: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    h_sr_c = h_sr.contiguous()
    h_e_c = h_e.contiguous()
    senders_c = senders.contiguous()
    receivers_c = receivers.contiguous()
    E, H = h_e_c.shape
    out = torch.empty_like(h_e_c)
    if E == 0 or H == 0:
        return out
    io_t, _ = _DTYPE_MAP[h_e_c.dtype]
    fwd_k = _get_kernels(h_e_c.dtype)[0]
    stream = wp.stream_from_torch(h_e_c.device)
    wp.launch(
        fwd_k,
        dim=(E, H),
        inputs=[
            _wp_from_torch(h_sr_c, dtype=io_t),
            _wp_from_torch(h_e_c, dtype=io_t),
            _wp_from_torch(senders_c, dtype=wp.int64),
            _wp_from_torch(receivers_c, dtype=wp.int64),
            int(H),
            _wp_from_torch(out, dtype=io_t),
        ],
        device=f"cuda:{h_e_c.device.index}",
        stream=stream,
    )
    return out


@_edge_gather_add_op.register_fake
def _edge_gather_add_fake(h_sr, h_e, senders, receivers, indptr, sort_idx):
    return torch.empty_like(h_e)


@torch.library.custom_op("fvgn_mp::edge_gather_add_bwd_sr", mutates_args=())
def _edge_gather_add_bwd_sr_op(
    grad_out: torch.Tensor,
    indptr: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    g_c = grad_out.contiguous()
    indptr_c = indptr.contiguous()
    sort_c = sort_idx.contiguous()
    num_nodes = indptr_c.shape[0] - 1
    H = g_c.shape[1]
    grad_h_sr = torch.empty((num_nodes, 2 * H), device=g_c.device, dtype=g_c.dtype)
    if num_nodes == 0 or H == 0:
        return grad_h_sr
    io_t, _ = _DTYPE_MAP[g_c.dtype]
    bwd_k = _get_kernels(g_c.dtype)[1]
    stream = wp.stream_from_torch(g_c.device)
    wp.launch(
        bwd_k,
        dim=(num_nodes, H),
        inputs=[
            _wp_from_torch(g_c, dtype=io_t),
            _wp_from_torch(indptr_c, dtype=wp.int64),
            _wp_from_torch(sort_c, dtype=wp.int64),
            int(H),
            _wp_from_torch(grad_h_sr, dtype=io_t),
        ],
        device=f"cuda:{g_c.device.index}",
        stream=stream,
    )
    return grad_h_sr


@_edge_gather_add_bwd_sr_op.register_fake
def _edge_gather_add_bwd_sr_fake(grad_out, indptr, sort_idx):
    num_nodes = indptr.shape[0] - 1
    H = grad_out.shape[1]
    return torch.empty((num_nodes, 2 * H), device=grad_out.device, dtype=grad_out.dtype)


def _setup_edge_gather_add_ctx(ctx, inputs, output):
    _h_sr, _h_e, _senders, _receivers, indptr, sort_idx = inputs
    ctx.save_for_backward(indptr, sort_idx)


def _backward_edge_gather_add(ctx, grad_out):
    indptr, sort_idx = ctx.saved_tensors
    need_sr = ctx.needs_input_grad[0]
    need_e = ctx.needs_input_grad[1]
    g_sr = _edge_gather_add_bwd_sr_op(grad_out, indptr, sort_idx) if need_sr else None
    return (
        g_sr,
        grad_out if need_e else None,
        None,
        None,
        None,
        None,
    )


_edge_gather_add_op.register_autograd(
    _backward_edge_gather_add, setup_context=_setup_edge_gather_add_ctx
)


@torch.library.custom_op("fvgn_mp::node_twohop_agg", mutates_args=())
def _node_twohop_agg_op(
    edge_attr: torch.Tensor,
    indptr: torch.Tensor,
    edge_indices: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    e_c = edge_attr.contiguous()
    indptr_c = indptr.contiguous()
    idx_c = edge_indices.contiguous()
    sort_c = sort_idx.contiguous()
    half_dim = e_c.shape[1] // 2
    N = indptr_c.shape[0] - 1
    out = torch.empty((N, half_dim), device=e_c.device, dtype=e_c.dtype)
    if N == 0 or half_dim == 0:
        return out
    twoway = e_c.view(-1, half_dim)
    node_agg = torch.empty((N, half_dim), device=e_c.device, dtype=e_c.dtype)
    io_t, _ = _DTYPE_MAP[e_c.dtype]
    _, _, k_s1_fwd, k_s2_fwd, _, _ = _get_kernels(e_c.dtype)
    dev_str = f"cuda:{e_c.device.index}"
    stream = wp.stream_from_torch(e_c.device)
    wp.launch(
        k_s1_fwd,
        dim=(N, half_dim),
        inputs=[
            _wp_from_torch(twoway, dtype=io_t),
            _wp_from_torch(indptr_c, dtype=wp.int64),
            _wp_from_torch(sort_c, dtype=wp.int64),
            _wp_from_torch(node_agg, dtype=io_t),
        ],
        device=dev_str,
        stream=stream,
    )
    wp.launch(
        k_s2_fwd,
        dim=(N, half_dim),
        inputs=[
            _wp_from_torch(node_agg, dtype=io_t),
            _wp_from_torch(indptr_c, dtype=wp.int64),
            _wp_from_torch(idx_c, dtype=wp.int64),
            _wp_from_torch(out, dtype=io_t),
        ],
        device=dev_str,
        stream=stream,
    )
    return out


@_node_twohop_agg_op.register_fake
def _node_twohop_agg_fake(edge_attr, indptr, edge_indices, sort_idx):
    half_dim = edge_attr.shape[1] // 2
    N = indptr.shape[0] - 1
    return torch.empty((N, half_dim), device=edge_attr.device, dtype=edge_attr.dtype)


@torch.library.custom_op("fvgn_mp::node_twohop_agg_bwd", mutates_args=())
def _node_twohop_agg_bwd_op(
    grad_out: torch.Tensor,
    indptr: torch.Tensor,
    edge_indices: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    g_c = grad_out.contiguous()
    indptr_c = indptr.contiguous()
    idx_c = edge_indices.contiguous()
    sort_c = sort_idx.contiguous()
    N, half_dim = g_c.shape
    num_edges = sort_c.shape[0] // 2
    grad_edge = torch.zeros((num_edges, 2 * half_dim), device=g_c.device, dtype=g_c.dtype)
    if N == 0 or num_edges == 0 or half_dim == 0:
        return grad_edge
    grad_node_agg = torch.empty((N, half_dim), device=g_c.device, dtype=g_c.dtype)
    grad_twoway = grad_edge.view(-1, half_dim)
    io_t, _ = _DTYPE_MAP[g_c.dtype]
    _, _, _, _, k_s2_bwd, k_s1_bwd = _get_kernels(g_c.dtype)
    dev_str = f"cuda:{g_c.device.index}"
    stream = wp.stream_from_torch(g_c.device)
    wp.launch(
        k_s2_bwd,
        dim=(N, half_dim),
        inputs=[
            _wp_from_torch(g_c, dtype=io_t),
            _wp_from_torch(indptr_c, dtype=wp.int64),
            _wp_from_torch(idx_c, dtype=wp.int64),
            _wp_from_torch(grad_node_agg, dtype=io_t),
        ],
        device=dev_str,
        stream=stream,
    )
    wp.launch(
        k_s1_bwd,
        dim=(N, half_dim),
        inputs=[
            _wp_from_torch(grad_node_agg, dtype=io_t),
            _wp_from_torch(indptr_c, dtype=wp.int64),
            _wp_from_torch(sort_c, dtype=wp.int64),
            _wp_from_torch(grad_twoway, dtype=io_t),
        ],
        device=dev_str,
        stream=stream,
    )
    return grad_edge


@_node_twohop_agg_bwd_op.register_fake
def _node_twohop_agg_bwd_fake(grad_out, indptr, edge_indices, sort_idx):
    half_dim = grad_out.shape[1]
    num_edges = sort_idx.shape[0] // 2
    return torch.empty((num_edges, 2 * half_dim), device=grad_out.device, dtype=grad_out.dtype)


def _setup_node_twohop_agg_ctx(ctx, inputs, output):
    _edge_attr, indptr, edge_indices, sort_idx = inputs
    ctx.save_for_backward(indptr, edge_indices, sort_idx)


def _backward_node_twohop_agg(ctx, grad_out):
    indptr, edge_indices, sort_idx = ctx.saved_tensors
    if not ctx.needs_input_grad[0]:
        return (None, None, None, None)
    grad_edge = _node_twohop_agg_bwd_op(grad_out, indptr, edge_indices, sort_idx)
    return (grad_edge, None, None, None)


_node_twohop_agg_op.register_autograd(
    _backward_node_twohop_agg, setup_context=_setup_node_twohop_agg_ctx
)


def warp_edge_gather_add(
    h_sr: torch.Tensor,
    h_e: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    indptr: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    """Fused ``h_e[e] + h_sr[senders[e], :H] + h_sr[receivers[e], H:]`` with zero-atomic CSR backward."""
    return _edge_gather_add_op(
        h_sr,
        h_e,
        senders.to(torch.int64),
        receivers.to(torch.int64),
        indptr.to(torch.int64),
        sort_idx.to(torch.int64),
    )


def warp_node_twohop_agg(
    edge_attr: torch.Tensor,
    indptr: torch.Tensor,
    edge_indices: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    """Fused two-hop CSR sum -> neighbor-mean aggregation with zero-atomic backward."""
    return _node_twohop_agg_op(
        edge_attr,
        indptr.to(torch.int64),
        edge_indices.to(torch.int64),
        sort_idx.to(torch.int64),
    )
