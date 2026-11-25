"""Memory-efficient and compute-fused ragged Physics-Attention kernels in NVIDIA Warp."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import warp as wp

wp.config.quiet = True
wp.set_module_options({"enable_backward": False})

_POOL_RED_SPLITS = 64
_WEIGHT_RED_SPLITS = 128
_TEMP_RED_SPLITS = 256


def _wp_from_torch(t: torch.Tensor, dtype=None):
    return wp.from_torch(t, dtype=dtype, requires_grad=False)


def _choose_chunk_size(n_total: int) -> int:
    if n_total < 4096:
        return 32
    if n_total < 262144:
        return 64
    return 128


def _make_attn_kernels(io_t, acc_t):
    @wp.func
    def split_start(total: int, s: int, n_splits: int) -> int:
        q = total // n_splits
        r = total - q * n_splits
        return q * s + (r * s) // n_splits

    @wp.func
    def gumbel_from_counter(seed: wp.uint32, n: int, h: int, g: int):
        k1 = wp.uint32(n) * wp.uint32(0x9E3779B9)
        k2 = (wp.uint32(h) * wp.uint32(0x85EBCA6B)) ^ (wp.uint32(g) * wp.uint32(0xC2B2AE35))
        x = seed ^ k1 ^ k2
        x = (x ^ (x >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
        x = (x ^ (x >> wp.uint32(15))) * wp.uint32(0x846CA68B)
        x = x ^ (x >> wp.uint32(16))
        u = (wp.float32(x >> wp.uint32(9)) + wp.float32(0.5)) * wp.float32(1.1920928955078125e-07)
        return acc_t(-wp.log(-wp.log(acc_t(u))))

    @wp.func
    def find_graph_idx(graph_ptr: wp.array(dtype=wp.int32), B: int, n: int) -> int:
        lo = int(0)
        hi = B
        while lo + 1 < hi:
            mid = (lo + hi) >> 1
            if graph_ptr[mid] <= n:
                lo = mid
            else:
                hi = mid
        return lo

    @wp.func
    def eval_tau(
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        n: int,
        h: int,
        temp_per_node: int,
    ):
        if temp_per_node == 1:
            return temp_node[n, h]
        return temp_head[h]

    @wp.func
    def eval_logit_general(
        x_mid: wp.array3d(dtype=io_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        seed_arr: wp.array(dtype=wp.int32),
        n: int,
        h: int,
        g: int,
        D: int,
        tau: acc_t,
        is_plus: int,
    ):
        dot_val = b_slice[g]
        for d in range(D):
            dot_val += acc_t(x_mid[n, h, d]) * W_slice[g, d]
        if is_plus == 1:
            dot_val += gumbel_from_counter(wp.uint32(seed_arr[0]), n, h, g)
        return dot_val / tau

    @wp.kernel
    def build_chunk_offsets_kernel(
        graph_ptr: wp.array(dtype=wp.int32),
        B: int,
        chunk_size: int,
        graph_chunk_off: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        if tid > 0:
            return
        off = int(0)
        graph_chunk_off[0] = 0
        for b in range(B):
            cnt = graph_ptr[b + 1] - graph_ptr[b]
            nc = int(0)
            if cnt > 0:
                nc = (cnt + chunk_size - 1) // chunk_size
            off += nc
            graph_chunk_off[b + 1] = off

    @wp.kernel
    def fill_chunks_kernel(
        graph_ptr: wp.array(dtype=wp.int32),
        graph_chunk_off: wp.array(dtype=wp.int32),
        B: int,
        chunk_size: int,
        chunk_start: wp.array(dtype=wp.int32),
        chunk_end: wp.array(dtype=wp.int32),
        chunk_graph: wp.array(dtype=wp.int32),
    ):
        cid = wp.tid()
        tot = graph_chunk_off[B]
        if cid >= tot:
            chunk_start[cid] = 0
            chunk_end[cid] = 0
            chunk_graph[cid] = -1
            return
        b = find_graph_idx(graph_chunk_off, B, cid)
        lc = cid - graph_chunk_off[b]
        s = graph_ptr[b] + lc * chunk_size
        e = wp.min(s + chunk_size, graph_ptr[b + 1])
        chunk_start[cid] = s
        chunk_end[cid] = e
        chunk_graph[cid] = b

    @wp.kernel
    def lse_fwd_kernel(
        x_mid: wp.array3d(dtype=io_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        seed_arr: wp.array(dtype=wp.int32),
        G: int,
        D: int,
        is_plus: int,
        temp_per_node: int,
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
    ):
        n, h = wp.tid()
        tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
        inv_tau = acc_t(1.0) / tau
        seed_u = wp.uint32(0)
        if is_plus == 1:
            seed_u = wp.uint32(seed_arr[0])

        m = acc_t(-1.0e30)
        s = acc_t(0.0)

        if D == 16:
            x0 = acc_t(x_mid[n, h, 0])
            x1 = acc_t(x_mid[n, h, 1])
            x2 = acc_t(x_mid[n, h, 2])
            x3 = acc_t(x_mid[n, h, 3])
            x4 = acc_t(x_mid[n, h, 4])
            x5 = acc_t(x_mid[n, h, 5])
            x6 = acc_t(x_mid[n, h, 6])
            x7 = acc_t(x_mid[n, h, 7])
            x8 = acc_t(x_mid[n, h, 8])
            x9 = acc_t(x_mid[n, h, 9])
            x10 = acc_t(x_mid[n, h, 10])
            x11 = acc_t(x_mid[n, h, 11])
            x12 = acc_t(x_mid[n, h, 12])
            x13 = acc_t(x_mid[n, h, 13])
            x14 = acc_t(x_mid[n, h, 14])
            x15 = acc_t(x_mid[n, h, 15])
            for g in range(G):
                dot_val = (
                    b_slice[g]
                    + x0 * W_slice[g, 0] + x1 * W_slice[g, 1]
                    + x2 * W_slice[g, 2] + x3 * W_slice[g, 3]
                    + x4 * W_slice[g, 4] + x5 * W_slice[g, 5]
                    + x6 * W_slice[g, 6] + x7 * W_slice[g, 7]
                    + x8 * W_slice[g, 8] + x9 * W_slice[g, 9]
                    + x10 * W_slice[g, 10] + x11 * W_slice[g, 11]
                    + x12 * W_slice[g, 12] + x13 * W_slice[g, 13]
                    + x14 * W_slice[g, 14] + x15 * W_slice[g, 15]
                )
                if is_plus == 1:
                    dot_val += gumbel_from_counter(seed_u, n, h, g)
                z = dot_val * inv_tau
                if g == 0:
                    m = z
                    s = acc_t(1.0)
                elif z > m:
                    s = s * wp.exp(m - z) + acc_t(1.0)
                    m = z
                else:
                    s += wp.exp(z - m)
        else:
            for g in range(G):
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                if g == 0:
                    m = z
                    s = acc_t(1.0)
                elif z > m:
                    s = s * wp.exp(m - z) + acc_t(1.0)
                    m = z
                else:
                    s += wp.exp(z - m)

        lse_max[n, h] = m
        lse_inv_sum[n, h] = acc_t(1.0) / s

    @wp.kernel
    def slice_pool_chunk_kernel(
        x_mid: wp.array3d(dtype=io_t),
        src: wp.array3d(dtype=io_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        seed_arr: wp.array(dtype=wp.int32),
        chunk_start: wp.array(dtype=wp.int32),
        chunk_end: wp.array(dtype=wp.int32),
        chunk_graph: wp.array(dtype=wp.int32),
        D: int,
        is_plus: int,
        temp_per_node: int,
        write_norm: int,
        partial_norm: wp.array3d(dtype=acc_t),
        partial_num: wp.array4d(dtype=acc_t),
    ):
        cid, h, g, dtile = wp.tid()
        d_base = dtile * 16
        rem = D - d_base
        if chunk_graph[cid] < 0:
            if write_norm == 1 and dtile == 0:
                partial_norm[cid, h, g] = acc_t(0.0)
            for k in range(16):
                if k < rem:
                    partial_num[cid, h, g, d_base + k] = acc_t(0.0)
            return

        s_idx = chunk_start[cid]
        e_idx = chunk_end[cid]
        seed_u = wp.uint32(0)
        if is_plus == 1:
            seed_u = wp.uint32(seed_arr[0])

        acc_norm = acc_t(0.0)
        a0 = acc_t(0.0)
        a1 = acc_t(0.0)
        a2 = acc_t(0.0)
        a3 = acc_t(0.0)
        a4 = acc_t(0.0)
        a5 = acc_t(0.0)
        a6 = acc_t(0.0)
        a7 = acc_t(0.0)
        a8 = acc_t(0.0)
        a9 = acc_t(0.0)
        a10 = acc_t(0.0)
        a11 = acc_t(0.0)
        a12 = acc_t(0.0)
        a13 = acc_t(0.0)
        a14 = acc_t(0.0)
        a15 = acc_t(0.0)

        if D == 16:
            bg = b_slice[g]
            w0 = W_slice[g, 0]
            w1 = W_slice[g, 1]
            w2 = W_slice[g, 2]
            w3 = W_slice[g, 3]
            w4 = W_slice[g, 4]
            w5 = W_slice[g, 5]
            w6 = W_slice[g, 6]
            w7 = W_slice[g, 7]
            w8 = W_slice[g, 8]
            w9 = W_slice[g, 9]
            w10 = W_slice[g, 10]
            w11 = W_slice[g, 11]
            w12 = W_slice[g, 12]
            w13 = W_slice[g, 13]
            w14 = W_slice[g, 14]
            w15 = W_slice[g, 15]
            inv_tau_h = acc_t(1.0)
            if temp_per_node == 0:
                inv_tau_h = acc_t(1.0) / temp_head[h]
            for n in range(s_idx, e_idx):
                dot_val = (
                    bg
                    + acc_t(x_mid[n, h, 0]) * w0 + acc_t(x_mid[n, h, 1]) * w1
                    + acc_t(x_mid[n, h, 2]) * w2 + acc_t(x_mid[n, h, 3]) * w3
                    + acc_t(x_mid[n, h, 4]) * w4 + acc_t(x_mid[n, h, 5]) * w5
                    + acc_t(x_mid[n, h, 6]) * w6 + acc_t(x_mid[n, h, 7]) * w7
                    + acc_t(x_mid[n, h, 8]) * w8 + acc_t(x_mid[n, h, 9]) * w9
                    + acc_t(x_mid[n, h, 10]) * w10 + acc_t(x_mid[n, h, 11]) * w11
                    + acc_t(x_mid[n, h, 12]) * w12 + acc_t(x_mid[n, h, 13]) * w13
                    + acc_t(x_mid[n, h, 14]) * w14 + acc_t(x_mid[n, h, 15]) * w15
                )
                if is_plus == 1:
                    dot_val += gumbel_from_counter(seed_u, n, h, g)
                z = dot_val * inv_tau_h
                if temp_per_node == 1:
                    z = dot_val / temp_node[n, h]
                wt = wp.exp(z - lse_max[n, h]) * lse_inv_sum[n, h]
                acc_norm += wt
                a0 += wt * acc_t(src[n, h, 0])
                a1 += wt * acc_t(src[n, h, 1])
                a2 += wt * acc_t(src[n, h, 2])
                a3 += wt * acc_t(src[n, h, 3])
                a4 += wt * acc_t(src[n, h, 4])
                a5 += wt * acc_t(src[n, h, 5])
                a6 += wt * acc_t(src[n, h, 6])
                a7 += wt * acc_t(src[n, h, 7])
                a8 += wt * acc_t(src[n, h, 8])
                a9 += wt * acc_t(src[n, h, 9])
                a10 += wt * acc_t(src[n, h, 10])
                a11 += wt * acc_t(src[n, h, 11])
                a12 += wt * acc_t(src[n, h, 12])
                a13 += wt * acc_t(src[n, h, 13])
                a14 += wt * acc_t(src[n, h, 14])
                a15 += wt * acc_t(src[n, h, 15])
        elif rem >= 16:
            for n in range(s_idx, e_idx):
                tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                wt = wp.exp(z - lse_max[n, h]) * lse_inv_sum[n, h]
                if dtile == 0:
                    acc_norm += wt
                a0 += wt * acc_t(src[n, h, d_base + 0])
                a1 += wt * acc_t(src[n, h, d_base + 1])
                a2 += wt * acc_t(src[n, h, d_base + 2])
                a3 += wt * acc_t(src[n, h, d_base + 3])
                a4 += wt * acc_t(src[n, h, d_base + 4])
                a5 += wt * acc_t(src[n, h, d_base + 5])
                a6 += wt * acc_t(src[n, h, d_base + 6])
                a7 += wt * acc_t(src[n, h, d_base + 7])
                a8 += wt * acc_t(src[n, h, d_base + 8])
                a9 += wt * acc_t(src[n, h, d_base + 9])
                a10 += wt * acc_t(src[n, h, d_base + 10])
                a11 += wt * acc_t(src[n, h, d_base + 11])
                a12 += wt * acc_t(src[n, h, d_base + 12])
                a13 += wt * acc_t(src[n, h, d_base + 13])
                a14 += wt * acc_t(src[n, h, d_base + 14])
                a15 += wt * acc_t(src[n, h, d_base + 15])
        else:
            for n in range(s_idx, e_idx):
                tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                wt = wp.exp(z - lse_max[n, h]) * lse_inv_sum[n, h]
                if dtile == 0:
                    acc_norm += wt
                if rem > 0:
                    a0 += wt * acc_t(src[n, h, d_base + 0])
                if rem > 1:
                    a1 += wt * acc_t(src[n, h, d_base + 1])
                if rem > 2:
                    a2 += wt * acc_t(src[n, h, d_base + 2])
                if rem > 3:
                    a3 += wt * acc_t(src[n, h, d_base + 3])
                if rem > 4:
                    a4 += wt * acc_t(src[n, h, d_base + 4])
                if rem > 5:
                    a5 += wt * acc_t(src[n, h, d_base + 5])
                if rem > 6:
                    a6 += wt * acc_t(src[n, h, d_base + 6])
                if rem > 7:
                    a7 += wt * acc_t(src[n, h, d_base + 7])
                if rem > 8:
                    a8 += wt * acc_t(src[n, h, d_base + 8])
                if rem > 9:
                    a9 += wt * acc_t(src[n, h, d_base + 9])
                if rem > 10:
                    a10 += wt * acc_t(src[n, h, d_base + 10])
                if rem > 11:
                    a11 += wt * acc_t(src[n, h, d_base + 11])
                if rem > 12:
                    a12 += wt * acc_t(src[n, h, d_base + 12])
                if rem > 13:
                    a13 += wt * acc_t(src[n, h, d_base + 13])
                if rem > 14:
                    a14 += wt * acc_t(src[n, h, d_base + 14])
                if rem > 15:
                    a15 += wt * acc_t(src[n, h, d_base + 15])

        if write_norm == 1 and dtile == 0:
            partial_norm[cid, h, g] = acc_norm
        if rem >= 16:
            partial_num[cid, h, g, d_base + 0] = a0
            partial_num[cid, h, g, d_base + 1] = a1
            partial_num[cid, h, g, d_base + 2] = a2
            partial_num[cid, h, g, d_base + 3] = a3
            partial_num[cid, h, g, d_base + 4] = a4
            partial_num[cid, h, g, d_base + 5] = a5
            partial_num[cid, h, g, d_base + 6] = a6
            partial_num[cid, h, g, d_base + 7] = a7
            partial_num[cid, h, g, d_base + 8] = a8
            partial_num[cid, h, g, d_base + 9] = a9
            partial_num[cid, h, g, d_base + 10] = a10
            partial_num[cid, h, g, d_base + 11] = a11
            partial_num[cid, h, g, d_base + 12] = a12
            partial_num[cid, h, g, d_base + 13] = a13
            partial_num[cid, h, g, d_base + 14] = a14
            partial_num[cid, h, g, d_base + 15] = a15
        else:
            if rem > 0:
                partial_num[cid, h, g, d_base + 0] = a0
            if rem > 1:
                partial_num[cid, h, g, d_base + 1] = a1
            if rem > 2:
                partial_num[cid, h, g, d_base + 2] = a2
            if rem > 3:
                partial_num[cid, h, g, d_base + 3] = a3
            if rem > 4:
                partial_num[cid, h, g, d_base + 4] = a4
            if rem > 5:
                partial_num[cid, h, g, d_base + 5] = a5
            if rem > 6:
                partial_num[cid, h, g, d_base + 6] = a6
            if rem > 7:
                partial_num[cid, h, g, d_base + 7] = a7
            if rem > 8:
                partial_num[cid, h, g, d_base + 8] = a8
            if rem > 9:
                partial_num[cid, h, g, d_base + 9] = a9
            if rem > 10:
                partial_num[cid, h, g, d_base + 10] = a10
            if rem > 11:
                partial_num[cid, h, g, d_base + 11] = a11
            if rem > 12:
                partial_num[cid, h, g, d_base + 12] = a12
            if rem > 13:
                partial_num[cid, h, g, d_base + 13] = a13
            if rem > 14:
                partial_num[cid, h, g, d_base + 14] = a14
            if rem > 15:
                partial_num[cid, h, g, d_base + 15] = a15

    @wp.kernel
    def pool_reduce_stage1_kernel(
        partial_norm: wp.array3d(dtype=acc_t),
        partial_num: wp.array4d(dtype=acc_t),
        graph_chunk_off: wp.array(dtype=wp.int32),
        H: int,
        G: int,
        D: int,
        n_splits: int,
        normalize: int,
        stage1_norm: wp.array2d(dtype=acc_t),
        stage1_num: wp.array3d(dtype=acc_t),
    ):
        bhg, s = wp.tid()
        hg = H * G
        b = bhg // hg
        rem_hg = bhg - b * hg
        h = rem_hg // G
        g = rem_hg - h * G

        c0 = graph_chunk_off[b]
        cnt = graph_chunk_off[b + 1] - c0
        cid_beg = c0 + split_start(cnt, s, n_splits)
        cid_end = c0 + split_start(cnt, s + 1, n_splits)

        if normalize == 1:
            acc_n = acc_t(0.0)
            for cid in range(cid_beg, cid_end):
                acc_n += partial_norm[cid, h, g]
            stage1_norm[bhg, s] = acc_n

        if D == 16:
            v0 = acc_t(0.0)
            v1 = acc_t(0.0)
            v2 = acc_t(0.0)
            v3 = acc_t(0.0)
            v4 = acc_t(0.0)
            v5 = acc_t(0.0)
            v6 = acc_t(0.0)
            v7 = acc_t(0.0)
            v8 = acc_t(0.0)
            v9 = acc_t(0.0)
            v10 = acc_t(0.0)
            v11 = acc_t(0.0)
            v12 = acc_t(0.0)
            v13 = acc_t(0.0)
            v14 = acc_t(0.0)
            v15 = acc_t(0.0)
            for cid in range(cid_beg, cid_end):
                v0 += partial_num[cid, h, g, 0]
                v1 += partial_num[cid, h, g, 1]
                v2 += partial_num[cid, h, g, 2]
                v3 += partial_num[cid, h, g, 3]
                v4 += partial_num[cid, h, g, 4]
                v5 += partial_num[cid, h, g, 5]
                v6 += partial_num[cid, h, g, 6]
                v7 += partial_num[cid, h, g, 7]
                v8 += partial_num[cid, h, g, 8]
                v9 += partial_num[cid, h, g, 9]
                v10 += partial_num[cid, h, g, 10]
                v11 += partial_num[cid, h, g, 11]
                v12 += partial_num[cid, h, g, 12]
                v13 += partial_num[cid, h, g, 13]
                v14 += partial_num[cid, h, g, 14]
                v15 += partial_num[cid, h, g, 15]
            stage1_num[bhg, s, 0] = v0
            stage1_num[bhg, s, 1] = v1
            stage1_num[bhg, s, 2] = v2
            stage1_num[bhg, s, 3] = v3
            stage1_num[bhg, s, 4] = v4
            stage1_num[bhg, s, 5] = v5
            stage1_num[bhg, s, 6] = v6
            stage1_num[bhg, s, 7] = v7
            stage1_num[bhg, s, 8] = v8
            stage1_num[bhg, s, 9] = v9
            stage1_num[bhg, s, 10] = v10
            stage1_num[bhg, s, 11] = v11
            stage1_num[bhg, s, 12] = v12
            stage1_num[bhg, s, 13] = v13
            stage1_num[bhg, s, 14] = v14
            stage1_num[bhg, s, 15] = v15
        else:
            for d in range(D):
                acc_v = acc_t(0.0)
                for cid in range(cid_beg, cid_end):
                    acc_v += partial_num[cid, h, g, d]
                stage1_num[bhg, s, d] = acc_v

    @wp.kernel
    def pool_reduce_stage2_kernel(
        stage1_norm: wp.array2d(dtype=acc_t),
        stage1_num: wp.array3d(dtype=acc_t),
        H: int,
        G: int,
        D: int,
        n_splits: int,
        normalize: int,
        inv_denom: wp.array3d(dtype=acc_t),
        token_acc: wp.array4d(dtype=acc_t),
        token_out: wp.array4d(dtype=io_t),
    ):
        b, h, g = wp.tid()
        bhg = (b * H + h) * G + g
        scale = acc_t(1.0)
        if normalize == 1:
            norm = acc_t(0.0)
            for s in range(n_splits):
                norm += stage1_norm[bhg, s]
            scale = acc_t(1.0) / (norm + acc_t(1.0e-5))
            inv_denom[b, h, g] = scale

        for d in range(D):
            acc_v = acc_t(0.0)
            for s in range(n_splits):
                acc_v += stage1_num[bhg, s, d]
            val = acc_v * scale
            token_acc[b, h, g, d] = val
            if normalize == 1:
                token_out[b, h, g, d] = io_t(val)

    @wp.kernel
    def slice_broadcast_fwd_kernel(
        x_mid: wp.array3d(dtype=io_t),
        out_token: wp.array4d(dtype=acc_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        graph_ptr: wp.array(dtype=wp.int32),
        seed_arr: wp.array(dtype=wp.int32),
        B: int,
        G: int,
        D: int,
        is_plus: int,
        temp_per_node: int,
        out: wp.array3d(dtype=io_t),
    ):
        n, h, dtile = wp.tid()
        d_base = dtile * 16
        rem = D - d_base
        b = find_graph_idx(graph_ptr, B, n)
        tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
        inv_tau = acc_t(1.0) / tau
        m_nh = lse_max[n, h]
        inv_s = lse_inv_sum[n, h]
        seed_u = wp.uint32(0)
        if is_plus == 1:
            seed_u = wp.uint32(seed_arr[0])

        a0 = acc_t(0.0)
        a1 = acc_t(0.0)
        a2 = acc_t(0.0)
        a3 = acc_t(0.0)
        a4 = acc_t(0.0)
        a5 = acc_t(0.0)
        a6 = acc_t(0.0)
        a7 = acc_t(0.0)
        a8 = acc_t(0.0)
        a9 = acc_t(0.0)
        a10 = acc_t(0.0)
        a11 = acc_t(0.0)
        a12 = acc_t(0.0)
        a13 = acc_t(0.0)
        a14 = acc_t(0.0)
        a15 = acc_t(0.0)

        if D == 16:
            x0 = acc_t(x_mid[n, h, 0])
            x1 = acc_t(x_mid[n, h, 1])
            x2 = acc_t(x_mid[n, h, 2])
            x3 = acc_t(x_mid[n, h, 3])
            x4 = acc_t(x_mid[n, h, 4])
            x5 = acc_t(x_mid[n, h, 5])
            x6 = acc_t(x_mid[n, h, 6])
            x7 = acc_t(x_mid[n, h, 7])
            x8 = acc_t(x_mid[n, h, 8])
            x9 = acc_t(x_mid[n, h, 9])
            x10 = acc_t(x_mid[n, h, 10])
            x11 = acc_t(x_mid[n, h, 11])
            x12 = acc_t(x_mid[n, h, 12])
            x13 = acc_t(x_mid[n, h, 13])
            x14 = acc_t(x_mid[n, h, 14])
            x15 = acc_t(x_mid[n, h, 15])
            for g in range(G):
                dot_val = (
                    b_slice[g]
                    + x0 * W_slice[g, 0] + x1 * W_slice[g, 1]
                    + x2 * W_slice[g, 2] + x3 * W_slice[g, 3]
                    + x4 * W_slice[g, 4] + x5 * W_slice[g, 5]
                    + x6 * W_slice[g, 6] + x7 * W_slice[g, 7]
                    + x8 * W_slice[g, 8] + x9 * W_slice[g, 9]
                    + x10 * W_slice[g, 10] + x11 * W_slice[g, 11]
                    + x12 * W_slice[g, 12] + x13 * W_slice[g, 13]
                    + x14 * W_slice[g, 14] + x15 * W_slice[g, 15]
                )
                if is_plus == 1:
                    dot_val += gumbel_from_counter(seed_u, n, h, g)
                wt = wp.exp(dot_val * inv_tau - m_nh) * inv_s
                a0 += wt * out_token[b, h, g, 0]
                a1 += wt * out_token[b, h, g, 1]
                a2 += wt * out_token[b, h, g, 2]
                a3 += wt * out_token[b, h, g, 3]
                a4 += wt * out_token[b, h, g, 4]
                a5 += wt * out_token[b, h, g, 5]
                a6 += wt * out_token[b, h, g, 6]
                a7 += wt * out_token[b, h, g, 7]
                a8 += wt * out_token[b, h, g, 8]
                a9 += wt * out_token[b, h, g, 9]
                a10 += wt * out_token[b, h, g, 10]
                a11 += wt * out_token[b, h, g, 11]
                a12 += wt * out_token[b, h, g, 12]
                a13 += wt * out_token[b, h, g, 13]
                a14 += wt * out_token[b, h, g, 14]
                a15 += wt * out_token[b, h, g, 15]
        elif rem >= 16:
            for g in range(G):
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                wt = wp.exp(z - m_nh) * inv_s
                a0 += wt * out_token[b, h, g, d_base + 0]
                a1 += wt * out_token[b, h, g, d_base + 1]
                a2 += wt * out_token[b, h, g, d_base + 2]
                a3 += wt * out_token[b, h, g, d_base + 3]
                a4 += wt * out_token[b, h, g, d_base + 4]
                a5 += wt * out_token[b, h, g, d_base + 5]
                a6 += wt * out_token[b, h, g, d_base + 6]
                a7 += wt * out_token[b, h, g, d_base + 7]
                a8 += wt * out_token[b, h, g, d_base + 8]
                a9 += wt * out_token[b, h, g, d_base + 9]
                a10 += wt * out_token[b, h, g, d_base + 10]
                a11 += wt * out_token[b, h, g, d_base + 11]
                a12 += wt * out_token[b, h, g, d_base + 12]
                a13 += wt * out_token[b, h, g, d_base + 13]
                a14 += wt * out_token[b, h, g, d_base + 14]
                a15 += wt * out_token[b, h, g, d_base + 15]
        else:
            for g in range(G):
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                wt = wp.exp(z - m_nh) * inv_s
                if rem > 0:
                    a0 += wt * out_token[b, h, g, d_base + 0]
                if rem > 1:
                    a1 += wt * out_token[b, h, g, d_base + 1]
                if rem > 2:
                    a2 += wt * out_token[b, h, g, d_base + 2]
                if rem > 3:
                    a3 += wt * out_token[b, h, g, d_base + 3]
                if rem > 4:
                    a4 += wt * out_token[b, h, g, d_base + 4]
                if rem > 5:
                    a5 += wt * out_token[b, h, g, d_base + 5]
                if rem > 6:
                    a6 += wt * out_token[b, h, g, d_base + 6]
                if rem > 7:
                    a7 += wt * out_token[b, h, g, d_base + 7]
                if rem > 8:
                    a8 += wt * out_token[b, h, g, d_base + 8]
                if rem > 9:
                    a9 += wt * out_token[b, h, g, d_base + 9]
                if rem > 10:
                    a10 += wt * out_token[b, h, g, d_base + 10]
                if rem > 11:
                    a11 += wt * out_token[b, h, g, d_base + 11]
                if rem > 12:
                    a12 += wt * out_token[b, h, g, d_base + 12]
                if rem > 13:
                    a13 += wt * out_token[b, h, g, d_base + 13]
                if rem > 14:
                    a14 += wt * out_token[b, h, g, d_base + 14]
                if rem > 15:
                    a15 += wt * out_token[b, h, g, d_base + 15]

        if rem >= 16:
            out[n, h, d_base + 0] = io_t(a0)
            out[n, h, d_base + 1] = io_t(a1)
            out[n, h, d_base + 2] = io_t(a2)
            out[n, h, d_base + 3] = io_t(a3)
            out[n, h, d_base + 4] = io_t(a4)
            out[n, h, d_base + 5] = io_t(a5)
            out[n, h, d_base + 6] = io_t(a6)
            out[n, h, d_base + 7] = io_t(a7)
            out[n, h, d_base + 8] = io_t(a8)
            out[n, h, d_base + 9] = io_t(a9)
            out[n, h, d_base + 10] = io_t(a10)
            out[n, h, d_base + 11] = io_t(a11)
            out[n, h, d_base + 12] = io_t(a12)
            out[n, h, d_base + 13] = io_t(a13)
            out[n, h, d_base + 14] = io_t(a14)
            out[n, h, d_base + 15] = io_t(a15)
        else:
            if rem > 0:
                out[n, h, d_base + 0] = io_t(a0)
            if rem > 1:
                out[n, h, d_base + 1] = io_t(a1)
            if rem > 2:
                out[n, h, d_base + 2] = io_t(a2)
            if rem > 3:
                out[n, h, d_base + 3] = io_t(a3)
            if rem > 4:
                out[n, h, d_base + 4] = io_t(a4)
            if rem > 5:
                out[n, h, d_base + 5] = io_t(a5)
            if rem > 6:
                out[n, h, d_base + 6] = io_t(a6)
            if rem > 7:
                out[n, h, d_base + 7] = io_t(a7)
            if rem > 8:
                out[n, h, d_base + 8] = io_t(a8)
            if rem > 9:
                out[n, h, d_base + 9] = io_t(a9)
            if rem > 10:
                out[n, h, d_base + 10] = io_t(a10)
            if rem > 11:
                out[n, h, d_base + 11] = io_t(a11)
            if rem > 12:
                out[n, h, d_base + 12] = io_t(a12)
            if rem > 13:
                out[n, h, d_base + 13] = io_t(a13)
            if rem > 14:
                out[n, h, d_base + 14] = io_t(a14)
            if rem > 15:
                out[n, h, d_base + 15] = io_t(a15)


    @wp.kernel
    def bwd_fused_node_d16_kernel(
        x_mid: wp.array3d(dtype=io_t),
        src: wp.array3d(dtype=io_t),
        dout: wp.array3d(dtype=io_t),
        U_pool: wp.array4d(dtype=acc_t),
        c_pool: wp.array3d(dtype=acc_t),
        V_bcast: wp.array4d(dtype=acc_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        graph_ptr: wp.array(dtype=wp.int32),
        seed_arr: wp.array(dtype=wp.int32),
        B: int,
        G: int,
        is_plus: int,
        temp_per_node: int,
        has_pool: int,
        has_bcast: int,
        reuse_x_as_src: int,
        dot_ww_buf: wp.array2d(dtype=acc_t),
        d_temp_node: wp.array2d(dtype=acc_t),
        d_x_mid: wp.array3d(dtype=io_t),
        d_src: wp.array3d(dtype=io_t),
    ):
        n, h = wp.tid()
        b = find_graph_idx(graph_ptr, B, n)
        tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
        inv_tau = acc_t(1.0) / tau
        m_nh = lse_max[n, h]
        inv_s = lse_inv_sum[n, h]
        seed_u = wp.uint32(0)
        if is_plus == 1:
            seed_u = wp.uint32(seed_arr[0])

        x0 = acc_t(x_mid[n, h, 0])
        x1 = acc_t(x_mid[n, h, 1])
        x2 = acc_t(x_mid[n, h, 2])
        x3 = acc_t(x_mid[n, h, 3])
        x4 = acc_t(x_mid[n, h, 4])
        x5 = acc_t(x_mid[n, h, 5])
        x6 = acc_t(x_mid[n, h, 6])
        x7 = acc_t(x_mid[n, h, 7])
        x8 = acc_t(x_mid[n, h, 8])
        x9 = acc_t(x_mid[n, h, 9])
        x10 = acc_t(x_mid[n, h, 10])
        x11 = acc_t(x_mid[n, h, 11])
        x12 = acc_t(x_mid[n, h, 12])
        x13 = acc_t(x_mid[n, h, 13])
        x14 = acc_t(x_mid[n, h, 14])
        x15 = acc_t(x_mid[n, h, 15])

        s0 = acc_t(0.0)
        s1 = acc_t(0.0)
        s2 = acc_t(0.0)
        s3 = acc_t(0.0)
        s4 = acc_t(0.0)
        s5 = acc_t(0.0)
        s6 = acc_t(0.0)
        s7 = acc_t(0.0)
        s8 = acc_t(0.0)
        s9 = acc_t(0.0)
        s10 = acc_t(0.0)
        s11 = acc_t(0.0)
        s12 = acc_t(0.0)
        s13 = acc_t(0.0)
        s14 = acc_t(0.0)
        s15 = acc_t(0.0)
        if has_pool == 1:
            if reuse_x_as_src == 1:
                s0 = x0
                s1 = x1
                s2 = x2
                s3 = x3
                s4 = x4
                s5 = x5
                s6 = x6
                s7 = x7
                s8 = x8
                s9 = x9
                s10 = x10
                s11 = x11
                s12 = x12
                s13 = x13
                s14 = x14
                s15 = x15
            else:
                s0 = acc_t(src[n, h, 0])
                s1 = acc_t(src[n, h, 1])
                s2 = acc_t(src[n, h, 2])
                s3 = acc_t(src[n, h, 3])
                s4 = acc_t(src[n, h, 4])
                s5 = acc_t(src[n, h, 5])
                s6 = acc_t(src[n, h, 6])
                s7 = acc_t(src[n, h, 7])
                s8 = acc_t(src[n, h, 8])
                s9 = acc_t(src[n, h, 9])
                s10 = acc_t(src[n, h, 10])
                s11 = acc_t(src[n, h, 11])
                s12 = acc_t(src[n, h, 12])
                s13 = acc_t(src[n, h, 13])
                s14 = acc_t(src[n, h, 14])
                s15 = acc_t(src[n, h, 15])

        do0 = acc_t(0.0)
        do1 = acc_t(0.0)
        do2 = acc_t(0.0)
        do3 = acc_t(0.0)
        do4 = acc_t(0.0)
        do5 = acc_t(0.0)
        do6 = acc_t(0.0)
        do7 = acc_t(0.0)
        do8 = acc_t(0.0)
        do9 = acc_t(0.0)
        do10 = acc_t(0.0)
        do11 = acc_t(0.0)
        do12 = acc_t(0.0)
        do13 = acc_t(0.0)
        do14 = acc_t(0.0)
        do15 = acc_t(0.0)
        if has_bcast == 1:
            do0 = acc_t(dout[n, h, 0])
            do1 = acc_t(dout[n, h, 1])
            do2 = acc_t(dout[n, h, 2])
            do3 = acc_t(dout[n, h, 3])
            do4 = acc_t(dout[n, h, 4])
            do5 = acc_t(dout[n, h, 5])
            do6 = acc_t(dout[n, h, 6])
            do7 = acc_t(dout[n, h, 7])
            do8 = acc_t(dout[n, h, 8])
            do9 = acc_t(dout[n, h, 9])
            do10 = acc_t(dout[n, h, 10])
            do11 = acc_t(dout[n, h, 11])
            do12 = acc_t(dout[n, h, 12])
            do13 = acc_t(dout[n, h, 13])
            do14 = acc_t(dout[n, h, 14])
            do15 = acc_t(dout[n, h, 15])

        dot_ww = acc_t(0.0)
        sum_wwz = acc_t(0.0)
        sum_wz = acc_t(0.0)

        for g in range(G):
            dot_val = (
                b_slice[g]
                + x0 * W_slice[g, 0] + x1 * W_slice[g, 1]
                + x2 * W_slice[g, 2] + x3 * W_slice[g, 3]
                + x4 * W_slice[g, 4] + x5 * W_slice[g, 5]
                + x6 * W_slice[g, 6] + x7 * W_slice[g, 7]
                + x8 * W_slice[g, 8] + x9 * W_slice[g, 9]
                + x10 * W_slice[g, 10] + x11 * W_slice[g, 11]
                + x12 * W_slice[g, 12] + x13 * W_slice[g, 13]
                + x14 * W_slice[g, 14] + x15 * W_slice[g, 15]
            )
            if is_plus == 1:
                dot_val += gumbel_from_counter(seed_u, n, h, g)
            z = dot_val * inv_tau
            wt = wp.exp(z - m_nh) * inv_s
            bw = acc_t(0.0)
            if has_pool == 1:
                bw += (
                    c_pool[b, h, g]
                    + s0 * U_pool[b, h, g, 0] + s1 * U_pool[b, h, g, 1]
                    + s2 * U_pool[b, h, g, 2] + s3 * U_pool[b, h, g, 3]
                    + s4 * U_pool[b, h, g, 4] + s5 * U_pool[b, h, g, 5]
                    + s6 * U_pool[b, h, g, 6] + s7 * U_pool[b, h, g, 7]
                    + s8 * U_pool[b, h, g, 8] + s9 * U_pool[b, h, g, 9]
                    + s10 * U_pool[b, h, g, 10] + s11 * U_pool[b, h, g, 11]
                    + s12 * U_pool[b, h, g, 12] + s13 * U_pool[b, h, g, 13]
                    + s14 * U_pool[b, h, g, 14] + s15 * U_pool[b, h, g, 15]
                )
            if has_bcast == 1:
                bw += (
                    do0 * V_bcast[b, h, g, 0] + do1 * V_bcast[b, h, g, 1]
                    + do2 * V_bcast[b, h, g, 2] + do3 * V_bcast[b, h, g, 3]
                    + do4 * V_bcast[b, h, g, 4] + do5 * V_bcast[b, h, g, 5]
                    + do6 * V_bcast[b, h, g, 6] + do7 * V_bcast[b, h, g, 7]
                    + do8 * V_bcast[b, h, g, 8] + do9 * V_bcast[b, h, g, 9]
                    + do10 * V_bcast[b, h, g, 10] + do11 * V_bcast[b, h, g, 11]
                    + do12 * V_bcast[b, h, g, 12] + do13 * V_bcast[b, h, g, 13]
                    + do14 * V_bcast[b, h, g, 14] + do15 * V_bcast[b, h, g, 15]
                )
            wbw = wt * bw
            dot_ww += wbw
            sum_wwz += wbw * z
            sum_wz += wt * z

        dot_ww_buf[n, h] = dot_ww
        d_temp_node[n, h] = -(sum_wwz - dot_ww * sum_wz) * inv_tau

        dx0 = acc_t(0.0)
        dx1 = acc_t(0.0)
        dx2 = acc_t(0.0)
        dx3 = acc_t(0.0)
        dx4 = acc_t(0.0)
        dx5 = acc_t(0.0)
        dx6 = acc_t(0.0)
        dx7 = acc_t(0.0)
        dx8 = acc_t(0.0)
        dx9 = acc_t(0.0)
        dx10 = acc_t(0.0)
        dx11 = acc_t(0.0)
        dx12 = acc_t(0.0)
        dx13 = acc_t(0.0)
        dx14 = acc_t(0.0)
        dx15 = acc_t(0.0)

        ds0 = acc_t(0.0)
        ds1 = acc_t(0.0)
        ds2 = acc_t(0.0)
        ds3 = acc_t(0.0)
        ds4 = acc_t(0.0)
        ds5 = acc_t(0.0)
        ds6 = acc_t(0.0)
        ds7 = acc_t(0.0)
        ds8 = acc_t(0.0)
        ds9 = acc_t(0.0)
        ds10 = acc_t(0.0)
        ds11 = acc_t(0.0)
        ds12 = acc_t(0.0)
        ds13 = acc_t(0.0)
        ds14 = acc_t(0.0)
        ds15 = acc_t(0.0)

        for g in range(G):
            dot_val = (
                b_slice[g]
                + x0 * W_slice[g, 0] + x1 * W_slice[g, 1]
                + x2 * W_slice[g, 2] + x3 * W_slice[g, 3]
                + x4 * W_slice[g, 4] + x5 * W_slice[g, 5]
                + x6 * W_slice[g, 6] + x7 * W_slice[g, 7]
                + x8 * W_slice[g, 8] + x9 * W_slice[g, 9]
                + x10 * W_slice[g, 10] + x11 * W_slice[g, 11]
                + x12 * W_slice[g, 12] + x13 * W_slice[g, 13]
                + x14 * W_slice[g, 14] + x15 * W_slice[g, 15]
            )
            if is_plus == 1:
                dot_val += gumbel_from_counter(seed_u, n, h, g)
            wt = wp.exp(dot_val * inv_tau - m_nh) * inv_s
            bw = acc_t(0.0)
            if has_pool == 1:
                u0 = U_pool[b, h, g, 0]
                u1 = U_pool[b, h, g, 1]
                u2 = U_pool[b, h, g, 2]
                u3 = U_pool[b, h, g, 3]
                u4 = U_pool[b, h, g, 4]
                u5 = U_pool[b, h, g, 5]
                u6 = U_pool[b, h, g, 6]
                u7 = U_pool[b, h, g, 7]
                u8 = U_pool[b, h, g, 8]
                u9 = U_pool[b, h, g, 9]
                u10 = U_pool[b, h, g, 10]
                u11 = U_pool[b, h, g, 11]
                u12 = U_pool[b, h, g, 12]
                u13 = U_pool[b, h, g, 13]
                u14 = U_pool[b, h, g, 14]
                u15 = U_pool[b, h, g, 15]
                bw += (
                    c_pool[b, h, g]
                    + s0 * u0 + s1 * u1 + s2 * u2 + s3 * u3
                    + s4 * u4 + s5 * u5 + s6 * u6 + s7 * u7
                    + s8 * u8 + s9 * u9 + s10 * u10 + s11 * u11
                    + s12 * u12 + s13 * u13 + s14 * u14 + s15 * u15
                )
                ds0 += wt * u0
                ds1 += wt * u1
                ds2 += wt * u2
                ds3 += wt * u3
                ds4 += wt * u4
                ds5 += wt * u5
                ds6 += wt * u6
                ds7 += wt * u7
                ds8 += wt * u8
                ds9 += wt * u9
                ds10 += wt * u10
                ds11 += wt * u11
                ds12 += wt * u12
                ds13 += wt * u13
                ds14 += wt * u14
                ds15 += wt * u15
            if has_bcast == 1:
                bw += (
                    do0 * V_bcast[b, h, g, 0] + do1 * V_bcast[b, h, g, 1]
                    + do2 * V_bcast[b, h, g, 2] + do3 * V_bcast[b, h, g, 3]
                    + do4 * V_bcast[b, h, g, 4] + do5 * V_bcast[b, h, g, 5]
                    + do6 * V_bcast[b, h, g, 6] + do7 * V_bcast[b, h, g, 7]
                    + do8 * V_bcast[b, h, g, 8] + do9 * V_bcast[b, h, g, 9]
                    + do10 * V_bcast[b, h, g, 10] + do11 * V_bcast[b, h, g, 11]
                    + do12 * V_bcast[b, h, g, 12] + do13 * V_bcast[b, h, g, 13]
                    + do14 * V_bcast[b, h, g, 14] + do15 * V_bcast[b, h, g, 15]
                )
            bar_l = wt * (bw - dot_ww) * inv_tau
            dx0 += bar_l * W_slice[g, 0]
            dx1 += bar_l * W_slice[g, 1]
            dx2 += bar_l * W_slice[g, 2]
            dx3 += bar_l * W_slice[g, 3]
            dx4 += bar_l * W_slice[g, 4]
            dx5 += bar_l * W_slice[g, 5]
            dx6 += bar_l * W_slice[g, 6]
            dx7 += bar_l * W_slice[g, 7]
            dx8 += bar_l * W_slice[g, 8]
            dx9 += bar_l * W_slice[g, 9]
            dx10 += bar_l * W_slice[g, 10]
            dx11 += bar_l * W_slice[g, 11]
            dx12 += bar_l * W_slice[g, 12]
            dx13 += bar_l * W_slice[g, 13]
            dx14 += bar_l * W_slice[g, 14]
            dx15 += bar_l * W_slice[g, 15]

        if has_pool == 1 and reuse_x_as_src == 1:
            dx0 += ds0
            dx1 += ds1
            dx2 += ds2
            dx3 += ds3
            dx4 += ds4
            dx5 += ds5
            dx6 += ds6
            dx7 += ds7
            dx8 += ds8
            dx9 += ds9
            dx10 += ds10
            dx11 += ds11
            dx12 += ds12
            dx13 += ds13
            dx14 += ds14
            dx15 += ds15

        d_x_mid[n, h, 0] = io_t(dx0)
        d_x_mid[n, h, 1] = io_t(dx1)
        d_x_mid[n, h, 2] = io_t(dx2)
        d_x_mid[n, h, 3] = io_t(dx3)
        d_x_mid[n, h, 4] = io_t(dx4)
        d_x_mid[n, h, 5] = io_t(dx5)
        d_x_mid[n, h, 6] = io_t(dx6)
        d_x_mid[n, h, 7] = io_t(dx7)
        d_x_mid[n, h, 8] = io_t(dx8)
        d_x_mid[n, h, 9] = io_t(dx9)
        d_x_mid[n, h, 10] = io_t(dx10)
        d_x_mid[n, h, 11] = io_t(dx11)
        d_x_mid[n, h, 12] = io_t(dx12)
        d_x_mid[n, h, 13] = io_t(dx13)
        d_x_mid[n, h, 14] = io_t(dx14)
        d_x_mid[n, h, 15] = io_t(dx15)

        if has_pool == 1 and reuse_x_as_src == 0:
            d_src[n, h, 0] = io_t(ds0)
            d_src[n, h, 1] = io_t(ds1)
            d_src[n, h, 2] = io_t(ds2)
            d_src[n, h, 3] = io_t(ds3)
            d_src[n, h, 4] = io_t(ds4)
            d_src[n, h, 5] = io_t(ds5)
            d_src[n, h, 6] = io_t(ds6)
            d_src[n, h, 7] = io_t(ds7)
            d_src[n, h, 8] = io_t(ds8)
            d_src[n, h, 9] = io_t(ds9)
            d_src[n, h, 10] = io_t(ds10)
            d_src[n, h, 11] = io_t(ds11)
            d_src[n, h, 12] = io_t(ds12)
            d_src[n, h, 13] = io_t(ds13)
            d_src[n, h, 14] = io_t(ds14)
            d_src[n, h, 15] = io_t(ds15)

    @wp.kernel
    def bwd_dot_and_dtemp_general_kernel(
        x_mid: wp.array3d(dtype=io_t),
        src: wp.array3d(dtype=io_t),
        dout: wp.array3d(dtype=io_t),
        U_pool: wp.array4d(dtype=acc_t),
        c_pool: wp.array3d(dtype=acc_t),
        V_bcast: wp.array4d(dtype=acc_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        graph_ptr: wp.array(dtype=wp.int32),
        seed_arr: wp.array(dtype=wp.int32),
        B: int,
        G: int,
        D: int,
        is_plus: int,
        temp_per_node: int,
        has_pool: int,
        has_bcast: int,
        dot_ww_buf: wp.array2d(dtype=acc_t),
        d_temp_node: wp.array2d(dtype=acc_t),
    ):
        n, h = wp.tid()
        b = find_graph_idx(graph_ptr, B, n)
        tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
        inv_tau = acc_t(1.0) / tau
        m_nh = lse_max[n, h]
        inv_s = lse_inv_sum[n, h]

        dot_ww = acc_t(0.0)
        sum_wwz = acc_t(0.0)
        sum_wz = acc_t(0.0)

        for g in range(G):
            z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
            wt = wp.exp(z - m_nh) * inv_s
            bw = acc_t(0.0)
            if has_pool == 1:
                bw += c_pool[b, h, g]
                for d in range(D):
                    bw += acc_t(src[n, h, d]) * U_pool[b, h, g, d]
            if has_bcast == 1:
                for d in range(D):
                    bw += acc_t(dout[n, h, d]) * V_bcast[b, h, g, d]
            wbw = wt * bw
            dot_ww += wbw
            sum_wwz += wbw * z
            sum_wz += wt * z

        dot_ww_buf[n, h] = dot_ww
        d_temp_node[n, h] = -(sum_wwz - dot_ww * sum_wz) * inv_tau

    @wp.kernel
    def bwd_dx_dsrc_general_kernel(
        x_mid: wp.array3d(dtype=io_t),
        src: wp.array3d(dtype=io_t),
        dout: wp.array3d(dtype=io_t),
        U_pool: wp.array4d(dtype=acc_t),
        c_pool: wp.array3d(dtype=acc_t),
        V_bcast: wp.array4d(dtype=acc_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        dot_ww_buf: wp.array2d(dtype=acc_t),
        graph_ptr: wp.array(dtype=wp.int32),
        seed_arr: wp.array(dtype=wp.int32),
        B: int,
        G: int,
        D: int,
        is_plus: int,
        temp_per_node: int,
        has_pool: int,
        has_bcast: int,
        reuse_x_as_src: int,
        d_x_mid: wp.array3d(dtype=io_t),
        d_src: wp.array3d(dtype=io_t),
    ):
        n, h, dtile = wp.tid()
        d_base = dtile * 16
        rem = D - d_base
        b = find_graph_idx(graph_ptr, B, n)
        tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
        inv_tau = acc_t(1.0) / tau
        m_nh = lse_max[n, h]
        inv_s = lse_inv_sum[n, h]
        dot_ww = dot_ww_buf[n, h]

        for k in range(16):
            if k < rem:
                d_idx = d_base + k
                dx_k = acc_t(0.0)
                ds_k = acc_t(0.0)
                for g in range(G):
                    z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                    wt = wp.exp(z - m_nh) * inv_s
                    bw = acc_t(0.0)
                    if has_pool == 1:
                        bw += c_pool[b, h, g]
                        for d2 in range(D):
                            bw += acc_t(src[n, h, d2]) * U_pool[b, h, g, d2]
                        ds_k += wt * U_pool[b, h, g, d_idx]
                    if has_bcast == 1:
                        for d2 in range(D):
                            bw += acc_t(dout[n, h, d2]) * V_bcast[b, h, g, d2]
                    bar_l = wt * (bw - dot_ww) * inv_tau
                    dx_k += bar_l * W_slice[g, d_idx]
                if has_pool == 1 and reuse_x_as_src == 1:
                    dx_k += ds_k
                d_x_mid[n, h, d_idx] = io_t(dx_k)
                if has_pool == 1 and reuse_x_as_src == 0:
                    d_src[n, h, d_idx] = io_t(ds_k)

    @wp.kernel
    def bwd_weight_chunk_kernel(
        x_mid: wp.array3d(dtype=io_t),
        src: wp.array3d(dtype=io_t),
        dout: wp.array3d(dtype=io_t),
        U_pool: wp.array4d(dtype=acc_t),
        c_pool: wp.array3d(dtype=acc_t),
        V_bcast: wp.array4d(dtype=acc_t),
        W_slice: wp.array2d(dtype=acc_t),
        b_slice: wp.array(dtype=acc_t),
        temp_head: wp.array(dtype=acc_t),
        temp_node: wp.array2d(dtype=acc_t),
        lse_max: wp.array2d(dtype=acc_t),
        lse_inv_sum: wp.array2d(dtype=acc_t),
        dot_ww_buf: wp.array2d(dtype=acc_t),
        seed_arr: wp.array(dtype=wp.int32),
        chunk_start: wp.array(dtype=wp.int32),
        chunk_end: wp.array(dtype=wp.int32),
        chunk_graph: wp.array(dtype=wp.int32),
        D: int,
        is_plus: int,
        temp_per_node: int,
        has_pool: int,
        has_bcast: int,
        partial_dW: wp.array4d(dtype=acc_t),
        partial_db: wp.array3d(dtype=acc_t),
    ):
        cid, h, g, dtile = wp.tid()
        d_base = dtile * 16
        rem = D - d_base
        b = chunk_graph[cid]
        if b < 0:
            if dtile == 0:
                partial_db[cid, h, g] = acc_t(0.0)
            for k in range(16):
                if k < rem:
                    partial_dW[cid, h, g, d_base + k] = acc_t(0.0)
            return

        s_idx = chunk_start[cid]
        e_idx = chunk_end[cid]

        acc_db = acc_t(0.0)
        dw0 = acc_t(0.0)
        dw1 = acc_t(0.0)
        dw2 = acc_t(0.0)
        dw3 = acc_t(0.0)
        dw4 = acc_t(0.0)
        dw5 = acc_t(0.0)
        dw6 = acc_t(0.0)
        dw7 = acc_t(0.0)
        dw8 = acc_t(0.0)
        dw9 = acc_t(0.0)
        dw10 = acc_t(0.0)
        dw11 = acc_t(0.0)
        dw12 = acc_t(0.0)
        dw13 = acc_t(0.0)
        dw14 = acc_t(0.0)
        dw15 = acc_t(0.0)

        if D == 16:
            seed_u = wp.uint32(0)
            if is_plus == 1:
                seed_u = wp.uint32(seed_arr[0])
            bg = b_slice[g]
            w0 = W_slice[g, 0]
            w1 = W_slice[g, 1]
            w2 = W_slice[g, 2]
            w3 = W_slice[g, 3]
            w4 = W_slice[g, 4]
            w5 = W_slice[g, 5]
            w6 = W_slice[g, 6]
            w7 = W_slice[g, 7]
            w8 = W_slice[g, 8]
            w9 = W_slice[g, 9]
            w10 = W_slice[g, 10]
            w11 = W_slice[g, 11]
            w12 = W_slice[g, 12]
            w13 = W_slice[g, 13]
            w14 = W_slice[g, 14]
            w15 = W_slice[g, 15]

            u0 = acc_t(0.0)
            u1 = acc_t(0.0)
            u2 = acc_t(0.0)
            u3 = acc_t(0.0)
            u4 = acc_t(0.0)
            u5 = acc_t(0.0)
            u6 = acc_t(0.0)
            u7 = acc_t(0.0)
            u8 = acc_t(0.0)
            u9 = acc_t(0.0)
            u10 = acc_t(0.0)
            u11 = acc_t(0.0)
            u12 = acc_t(0.0)
            u13 = acc_t(0.0)
            u14 = acc_t(0.0)
            u15 = acc_t(0.0)
            cp_val = acc_t(0.0)
            if has_pool == 1:
                cp_val = c_pool[b, h, g]
                u0 = U_pool[b, h, g, 0]
                u1 = U_pool[b, h, g, 1]
                u2 = U_pool[b, h, g, 2]
                u3 = U_pool[b, h, g, 3]
                u4 = U_pool[b, h, g, 4]
                u5 = U_pool[b, h, g, 5]
                u6 = U_pool[b, h, g, 6]
                u7 = U_pool[b, h, g, 7]
                u8 = U_pool[b, h, g, 8]
                u9 = U_pool[b, h, g, 9]
                u10 = U_pool[b, h, g, 10]
                u11 = U_pool[b, h, g, 11]
                u12 = U_pool[b, h, g, 12]
                u13 = U_pool[b, h, g, 13]
                u14 = U_pool[b, h, g, 14]
                u15 = U_pool[b, h, g, 15]

            v0 = acc_t(0.0)
            v1 = acc_t(0.0)
            v2 = acc_t(0.0)
            v3 = acc_t(0.0)
            v4 = acc_t(0.0)
            v5 = acc_t(0.0)
            v6 = acc_t(0.0)
            v7 = acc_t(0.0)
            v8 = acc_t(0.0)
            v9 = acc_t(0.0)
            v10 = acc_t(0.0)
            v11 = acc_t(0.0)
            v12 = acc_t(0.0)
            v13 = acc_t(0.0)
            v14 = acc_t(0.0)
            v15 = acc_t(0.0)
            if has_bcast == 1:
                v0 = V_bcast[b, h, g, 0]
                v1 = V_bcast[b, h, g, 1]
                v2 = V_bcast[b, h, g, 2]
                v3 = V_bcast[b, h, g, 3]
                v4 = V_bcast[b, h, g, 4]
                v5 = V_bcast[b, h, g, 5]
                v6 = V_bcast[b, h, g, 6]
                v7 = V_bcast[b, h, g, 7]
                v8 = V_bcast[b, h, g, 8]
                v9 = V_bcast[b, h, g, 9]
                v10 = V_bcast[b, h, g, 10]
                v11 = V_bcast[b, h, g, 11]
                v12 = V_bcast[b, h, g, 12]
                v13 = V_bcast[b, h, g, 13]
                v14 = V_bcast[b, h, g, 14]
                v15 = V_bcast[b, h, g, 15]

            inv_tau_h = acc_t(1.0)
            if temp_per_node == 0:
                inv_tau_h = acc_t(1.0) / temp_head[h]

            for n in range(s_idx, e_idx):
                inv_tau = inv_tau_h
                if temp_per_node == 1:
                    inv_tau = acc_t(1.0) / temp_node[n, h]
                x0 = acc_t(x_mid[n, h, 0])
                x1 = acc_t(x_mid[n, h, 1])
                x2 = acc_t(x_mid[n, h, 2])
                x3 = acc_t(x_mid[n, h, 3])
                x4 = acc_t(x_mid[n, h, 4])
                x5 = acc_t(x_mid[n, h, 5])
                x6 = acc_t(x_mid[n, h, 6])
                x7 = acc_t(x_mid[n, h, 7])
                x8 = acc_t(x_mid[n, h, 8])
                x9 = acc_t(x_mid[n, h, 9])
                x10 = acc_t(x_mid[n, h, 10])
                x11 = acc_t(x_mid[n, h, 11])
                x12 = acc_t(x_mid[n, h, 12])
                x13 = acc_t(x_mid[n, h, 13])
                x14 = acc_t(x_mid[n, h, 14])
                x15 = acc_t(x_mid[n, h, 15])
                dot_val = (
                    bg
                    + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
                    + x4 * w4 + x5 * w5 + x6 * w6 + x7 * w7
                    + x8 * w8 + x9 * w9 + x10 * w10 + x11 * w11
                    + x12 * w12 + x13 * w13 + x14 * w14 + x15 * w15
                )
                if is_plus == 1:
                    dot_val += gumbel_from_counter(seed_u, n, h, g)
                wt = wp.exp(dot_val * inv_tau - lse_max[n, h]) * lse_inv_sum[n, h]
                bw = acc_t(0.0)
                if has_pool == 1:
                    bw += (
                        cp_val
                        + acc_t(src[n, h, 0]) * u0 + acc_t(src[n, h, 1]) * u1
                        + acc_t(src[n, h, 2]) * u2 + acc_t(src[n, h, 3]) * u3
                        + acc_t(src[n, h, 4]) * u4 + acc_t(src[n, h, 5]) * u5
                        + acc_t(src[n, h, 6]) * u6 + acc_t(src[n, h, 7]) * u7
                        + acc_t(src[n, h, 8]) * u8 + acc_t(src[n, h, 9]) * u9
                        + acc_t(src[n, h, 10]) * u10 + acc_t(src[n, h, 11]) * u11
                        + acc_t(src[n, h, 12]) * u12 + acc_t(src[n, h, 13]) * u13
                        + acc_t(src[n, h, 14]) * u14 + acc_t(src[n, h, 15]) * u15
                    )
                if has_bcast == 1:
                    bw += (
                        acc_t(dout[n, h, 0]) * v0 + acc_t(dout[n, h, 1]) * v1
                        + acc_t(dout[n, h, 2]) * v2 + acc_t(dout[n, h, 3]) * v3
                        + acc_t(dout[n, h, 4]) * v4 + acc_t(dout[n, h, 5]) * v5
                        + acc_t(dout[n, h, 6]) * v6 + acc_t(dout[n, h, 7]) * v7
                        + acc_t(dout[n, h, 8]) * v8 + acc_t(dout[n, h, 9]) * v9
                        + acc_t(dout[n, h, 10]) * v10 + acc_t(dout[n, h, 11]) * v11
                        + acc_t(dout[n, h, 12]) * v12 + acc_t(dout[n, h, 13]) * v13
                        + acc_t(dout[n, h, 14]) * v14 + acc_t(dout[n, h, 15]) * v15
                    )
                bar_l = wt * (bw - dot_ww_buf[n, h]) * inv_tau
                acc_db += bar_l
                dw0 += bar_l * x0
                dw1 += bar_l * x1
                dw2 += bar_l * x2
                dw3 += bar_l * x3
                dw4 += bar_l * x4
                dw5 += bar_l * x5
                dw6 += bar_l * x6
                dw7 += bar_l * x7
                dw8 += bar_l * x8
                dw9 += bar_l * x9
                dw10 += bar_l * x10
                dw11 += bar_l * x11
                dw12 += bar_l * x12
                dw13 += bar_l * x13
                dw14 += bar_l * x14
                dw15 += bar_l * x15
        else:
            for n in range(s_idx, e_idx):
                tau = eval_tau(temp_head, temp_node, n, h, temp_per_node)
                z = eval_logit_general(x_mid, W_slice, b_slice, seed_arr, n, h, g, D, tau, is_plus)
                wt = wp.exp(z - lse_max[n, h]) * lse_inv_sum[n, h]
                bw = acc_t(0.0)
                if has_pool == 1:
                    bw += c_pool[b, h, g]
                    for d in range(D):
                        bw += acc_t(src[n, h, d]) * U_pool[b, h, g, d]
                if has_bcast == 1:
                    for d in range(D):
                        bw += acc_t(dout[n, h, d]) * V_bcast[b, h, g, d]
                bar_l = wt * (bw - dot_ww_buf[n, h]) / tau
                if dtile == 0:
                    acc_db += bar_l
                if rem > 0:
                    dw0 += bar_l * acc_t(x_mid[n, h, d_base + 0])
                if rem > 1:
                    dw1 += bar_l * acc_t(x_mid[n, h, d_base + 1])
                if rem > 2:
                    dw2 += bar_l * acc_t(x_mid[n, h, d_base + 2])
                if rem > 3:
                    dw3 += bar_l * acc_t(x_mid[n, h, d_base + 3])
                if rem > 4:
                    dw4 += bar_l * acc_t(x_mid[n, h, d_base + 4])
                if rem > 5:
                    dw5 += bar_l * acc_t(x_mid[n, h, d_base + 5])
                if rem > 6:
                    dw6 += bar_l * acc_t(x_mid[n, h, d_base + 6])
                if rem > 7:
                    dw7 += bar_l * acc_t(x_mid[n, h, d_base + 7])
                if rem > 8:
                    dw8 += bar_l * acc_t(x_mid[n, h, d_base + 8])
                if rem > 9:
                    dw9 += bar_l * acc_t(x_mid[n, h, d_base + 9])
                if rem > 10:
                    dw10 += bar_l * acc_t(x_mid[n, h, d_base + 10])
                if rem > 11:
                    dw11 += bar_l * acc_t(x_mid[n, h, d_base + 11])
                if rem > 12:
                    dw12 += bar_l * acc_t(x_mid[n, h, d_base + 12])
                if rem > 13:
                    dw13 += bar_l * acc_t(x_mid[n, h, d_base + 13])
                if rem > 14:
                    dw14 += bar_l * acc_t(x_mid[n, h, d_base + 14])
                if rem > 15:
                    dw15 += bar_l * acc_t(x_mid[n, h, d_base + 15])

        if dtile == 0:
            partial_db[cid, h, g] = acc_db
        if rem >= 16:
            partial_dW[cid, h, g, d_base + 0] = dw0
            partial_dW[cid, h, g, d_base + 1] = dw1
            partial_dW[cid, h, g, d_base + 2] = dw2
            partial_dW[cid, h, g, d_base + 3] = dw3
            partial_dW[cid, h, g, d_base + 4] = dw4
            partial_dW[cid, h, g, d_base + 5] = dw5
            partial_dW[cid, h, g, d_base + 6] = dw6
            partial_dW[cid, h, g, d_base + 7] = dw7
            partial_dW[cid, h, g, d_base + 8] = dw8
            partial_dW[cid, h, g, d_base + 9] = dw9
            partial_dW[cid, h, g, d_base + 10] = dw10
            partial_dW[cid, h, g, d_base + 11] = dw11
            partial_dW[cid, h, g, d_base + 12] = dw12
            partial_dW[cid, h, g, d_base + 13] = dw13
            partial_dW[cid, h, g, d_base + 14] = dw14
            partial_dW[cid, h, g, d_base + 15] = dw15
        else:
            if rem > 0:
                partial_dW[cid, h, g, d_base + 0] = dw0
            if rem > 1:
                partial_dW[cid, h, g, d_base + 1] = dw1
            if rem > 2:
                partial_dW[cid, h, g, d_base + 2] = dw2
            if rem > 3:
                partial_dW[cid, h, g, d_base + 3] = dw3
            if rem > 4:
                partial_dW[cid, h, g, d_base + 4] = dw4
            if rem > 5:
                partial_dW[cid, h, g, d_base + 5] = dw5
            if rem > 6:
                partial_dW[cid, h, g, d_base + 6] = dw6
            if rem > 7:
                partial_dW[cid, h, g, d_base + 7] = dw7
            if rem > 8:
                partial_dW[cid, h, g, d_base + 8] = dw8
            if rem > 9:
                partial_dW[cid, h, g, d_base + 9] = dw9
            if rem > 10:
                partial_dW[cid, h, g, d_base + 10] = dw10
            if rem > 11:
                partial_dW[cid, h, g, d_base + 11] = dw11
            if rem > 12:
                partial_dW[cid, h, g, d_base + 12] = dw12
            if rem > 13:
                partial_dW[cid, h, g, d_base + 13] = dw13
            if rem > 14:
                partial_dW[cid, h, g, d_base + 14] = dw14
            if rem > 15:
                partial_dW[cid, h, g, d_base + 15] = dw15

    @wp.kernel
    def weight_reduce_stage1_kernel(
        partial_dW: wp.array4d(dtype=acc_t),
        partial_db: wp.array3d(dtype=acc_t),
        graph_chunk_off: wp.array(dtype=wp.int32),
        B: int,
        H: int,
        n_splits: int,
        stage1_dW: wp.array3d(dtype=acc_t),
        stage1_db: wp.array2d(dtype=acc_t),
    ):
        g, d, s = wp.tid()
        tot = graph_chunk_off[B]
        total_items = tot * H
        idx_beg = split_start(total_items, s, n_splits)
        idx_end = split_start(total_items, s + 1, n_splits)

        acc_w = acc_t(0.0)
        for idx in range(idx_beg, idx_end):
            cid = idx // H
            h = idx - cid * H
            acc_w += partial_dW[cid, h, g, d]
        stage1_dW[g, d, s] = acc_w

        if d == 0:
            acc_b = acc_t(0.0)
            for idx in range(idx_beg, idx_end):
                cid = idx // H
                h = idx - cid * H
                acc_b += partial_db[cid, h, g]
            stage1_db[g, s] = acc_b

    @wp.kernel
    def weight_reduce_stage2_kernel(
        stage1_dW: wp.array3d(dtype=acc_t),
        stage1_db: wp.array2d(dtype=acc_t),
        n_splits: int,
        d_W_slice: wp.array2d(dtype=acc_t),
        d_b_slice: wp.array(dtype=acc_t),
    ):
        g, d = wp.tid()
        acc_w = acc_t(0.0)
        for s in range(n_splits):
            acc_w += stage1_dW[g, d, s]
        d_W_slice[g, d] = acc_w

        if d == 0:
            acc_b = acc_t(0.0)
            for s in range(n_splits):
                acc_b += stage1_db[g, s]
            d_b_slice[g] = acc_b

    @wp.kernel
    def temp_reduce_stage1_kernel(
        d_temp_node: wp.array2d(dtype=acc_t),
        N: int,
        n_splits: int,
        stage1_dt: wp.array2d(dtype=acc_t),
    ):
        h, s = wp.tid()
        n_beg = split_start(N, s, n_splits)
        n_end = split_start(N, s + 1, n_splits)
        acc_v = acc_t(0.0)
        for n in range(n_beg, n_end):
            acc_v += d_temp_node[n, h]
        stage1_dt[h, s] = acc_v

    @wp.kernel
    def temp_reduce_stage2_kernel(
        stage1_dt: wp.array2d(dtype=acc_t),
        n_splits: int,
        d_temp_head: wp.array(dtype=acc_t),
    ):
        h = wp.tid()
        acc_v = acc_t(0.0)
        for s in range(n_splits):
            acc_v += stage1_dt[h, s]
        d_temp_head[h] = acc_v

    return {
        "build_chunk_offsets": build_chunk_offsets_kernel,
        "fill_chunks": fill_chunks_kernel,
        "lse_fwd": lse_fwd_kernel,
        "pool_chunk": slice_pool_chunk_kernel,
        "pool_reduce_s1": pool_reduce_stage1_kernel,
        "pool_reduce_s2": pool_reduce_stage2_kernel,
        "bcast_fwd": slice_broadcast_fwd_kernel,
        "bwd_fused_d16": bwd_fused_node_d16_kernel,
        "bwd_dot_dtemp_gen": bwd_dot_and_dtemp_general_kernel,
        "bwd_dx_dsrc_gen": bwd_dx_dsrc_general_kernel,
        "bwd_weight_chunk": bwd_weight_chunk_kernel,
        "weight_reduce_s1": weight_reduce_stage1_kernel,
        "weight_reduce_s2": weight_reduce_stage2_kernel,
        "temp_reduce_s1": temp_reduce_stage1_kernel,
        "temp_reduce_s2": temp_reduce_stage2_kernel,
    }


_KERNELS_CACHE: dict = {}


def _get_kernels(io_dtype: torch.dtype):
    cached = _KERNELS_CACHE.get(io_dtype)
    if cached is not None:
        return cached
    wp.init()
    if io_dtype == torch.float64:
        entry = (_make_attn_kernels(wp.float64, wp.float64), torch.float64, False)
    elif io_dtype == torch.bfloat16:
        if hasattr(wp, "bfloat16"):
            entry = (_make_attn_kernels(wp.bfloat16, wp.float32), torch.float32, False)
        else:
            entry = (_make_attn_kernels(wp.float32, wp.float32), torch.float32, True)
    elif io_dtype == torch.float32:
        entry = (_make_attn_kernels(wp.float32, wp.float32), torch.float32, False)
    else:
        raise TypeError(f"Unsupported dtype for Warp RaggedPhysicsAttention: {io_dtype}")
    _KERNELS_CACHE[io_dtype] = entry
    return entry


def _build_chunk_schedule(
    K: dict,
    graph_ptr_i32: torch.Tensor,
    N: int,
    B: int,
    wp_dev,
    stream,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_size = _choose_chunk_size(N)
    m_max = max((N // chunk_size) + B, 1)
    dev = graph_ptr_i32.device
    graph_chunk_off = torch.empty((B + 1,), dtype=torch.int32, device=dev)
    chunk_start = torch.empty((m_max,), dtype=torch.int32, device=dev)
    chunk_end = torch.empty((m_max,), dtype=torch.int32, device=dev)
    chunk_graph = torch.empty((m_max,), dtype=torch.int32, device=dev)

    w_ptr = _wp_from_torch(graph_ptr_i32)
    w_off = _wp_from_torch(graph_chunk_off)
    w_cs = _wp_from_torch(chunk_start)
    w_ce = _wp_from_torch(chunk_end)
    w_cg = _wp_from_torch(chunk_graph)

    wp.launch(
        K["build_chunk_offsets"],
        dim=1,
        inputs=[w_ptr, B, chunk_size],
        outputs=[w_off],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    wp.launch(
        K["fill_chunks"],
        dim=m_max,
        inputs=[w_ptr, w_off, B, chunk_size],
        outputs=[w_cs, w_ce, w_cg],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    return m_max, graph_chunk_off, chunk_start, chunk_end, chunk_graph


def _split_temp_views(
    temp_acc: torch.Tensor,
    N: int,
    H: int,
    temp_per_node: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temp_per_node:
        dummy_head = temp_acc.new_ones((H,))
        return dummy_head, temp_acc.reshape(N, H).contiguous()
    dummy_node = temp_acc.new_ones((1, max(H, 1)))
    return temp_acc.reshape(H).contiguous(), dummy_node


def _launch_pool_reduce_2stage(
    K: dict,
    partial_norm: torch.Tensor,
    partial_num: torch.Tensor,
    g_off: torch.Tensor,
    B: int,
    H: int,
    G: int,
    D: int,
    normalize: int,
    inv_denom: torch.Tensor,
    token_acc: torch.Tensor,
    token_out_raw: torch.Tensor,
    acc_dtype: torch.dtype,
    dev: torch.device,
    wp_dev,
    stream,
) -> None:
    bhg_tot = B * H * G
    n_splits = _POOL_RED_SPLITS
    stage1_norm = (
        torch.empty((bhg_tot, n_splits), dtype=acc_dtype, device=dev)
        if normalize == 1
        else torch.empty((1, 1), dtype=acc_dtype, device=dev)
    )
    stage1_num = torch.empty((bhg_tot, n_splits, D), dtype=acc_dtype, device=dev)

    wp.launch(
        K["pool_reduce_s1"],
        dim=(bhg_tot, n_splits),
        inputs=[
            _wp_from_torch(partial_norm),
            _wp_from_torch(partial_num),
            _wp_from_torch(g_off),
            H,
            G,
            D,
            n_splits,
            normalize,
        ],
        outputs=[_wp_from_torch(stage1_norm), _wp_from_torch(stage1_num)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    wp.launch(
        K["pool_reduce_s2"],
        dim=(B, H, G),
        inputs=[
            _wp_from_torch(stage1_norm),
            _wp_from_torch(stage1_num),
            H,
            G,
            D,
            n_splits,
            normalize,
        ],
        outputs=[
            _wp_from_torch(inv_denom),
            _wp_from_torch(token_acc),
            _wp_from_torch(token_out_raw),
        ],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )


def _launch_unified_node_and_param_bwd(
    K: dict,
    x_io: torch.Tensor,
    src_io: torch.Tensor,
    dout_io: torch.Tensor,
    U_pool: torch.Tensor,
    c_pool: torch.Tensor,
    V_bcast: torch.Tensor,
    W_acc: torch.Tensor,
    b_acc: torch.Tensor,
    temp_head: torch.Tensor,
    temp_node: torch.Tensor,
    lse_m: torch.Tensor,
    lse_is: torch.Tensor,
    ptr_i32: torch.Tensor,
    seed_i32: torch.Tensor,
    g_off: torch.Tensor,
    c_start: torch.Tensor,
    c_end: torch.Tensor,
    c_graph: torch.Tensor,
    m_max: int,
    N: int,
    B: int,
    H: int,
    G: int,
    D: int,
    is_plus: bool,
    temp_per_node: bool,
    has_pool: int,
    has_bcast: int,
    reuse_x_as_src: bool,
    acc_dtype: torch.dtype,
    io_out_dtype: torch.dtype,
    temp_orig: torch.Tensor,
    dev: torch.device,
    wp_dev,
    stream,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_d_tiles = (D + 15) // 16
    dot_ww_buf = torch.empty((N, H), dtype=acc_dtype, device=dev)
    d_temp_node = torch.empty((N, H), dtype=acc_dtype, device=dev)
    d_x_raw = torch.empty((N, H, D), dtype=io_out_dtype, device=dev)
    d_src_raw = (
        torch.empty((N, H, D), dtype=io_out_dtype, device=dev)
        if (has_pool == 1 and not reuse_x_as_src)
        else torch.empty((1, 1, 1), dtype=io_out_dtype, device=dev)
    )

    if D == 16:
        wp.launch(
            K["bwd_fused_d16"],
            dim=(N, H),
            inputs=[
                _wp_from_torch(x_io),
                _wp_from_torch(src_io),
                _wp_from_torch(dout_io),
                _wp_from_torch(U_pool),
                _wp_from_torch(c_pool),
                _wp_from_torch(V_bcast),
                _wp_from_torch(W_acc),
                _wp_from_torch(b_acc),
                _wp_from_torch(temp_head),
                _wp_from_torch(temp_node),
                _wp_from_torch(lse_m),
                _wp_from_torch(lse_is),
                _wp_from_torch(ptr_i32),
                _wp_from_torch(seed_i32),
                B,
                G,
                1 if is_plus else 0,
                1 if temp_per_node else 0,
                has_pool,
                has_bcast,
                1 if reuse_x_as_src else 0,
            ],
            outputs=[
                _wp_from_torch(dot_ww_buf),
                _wp_from_torch(d_temp_node),
                _wp_from_torch(d_x_raw),
                _wp_from_torch(d_src_raw),
            ],
            device=wp_dev,
            stream=stream,
            record_tape=False,
        )
    else:
        wp.launch(
            K["bwd_dot_dtemp_gen"],
            dim=(N, H),
            inputs=[
                _wp_from_torch(x_io),
                _wp_from_torch(src_io),
                _wp_from_torch(dout_io),
                _wp_from_torch(U_pool),
                _wp_from_torch(c_pool),
                _wp_from_torch(V_bcast),
                _wp_from_torch(W_acc),
                _wp_from_torch(b_acc),
                _wp_from_torch(temp_head),
                _wp_from_torch(temp_node),
                _wp_from_torch(lse_m),
                _wp_from_torch(lse_is),
                _wp_from_torch(ptr_i32),
                _wp_from_torch(seed_i32),
                B,
                G,
                D,
                1 if is_plus else 0,
                1 if temp_per_node else 0,
                has_pool,
                has_bcast,
            ],
            outputs=[_wp_from_torch(dot_ww_buf), _wp_from_torch(d_temp_node)],
            device=wp_dev,
            stream=stream,
            record_tape=False,
        )
        wp.launch(
            K["bwd_dx_dsrc_gen"],
            dim=(N, H, n_d_tiles),
            inputs=[
                _wp_from_torch(x_io),
                _wp_from_torch(src_io),
                _wp_from_torch(dout_io),
                _wp_from_torch(U_pool),
                _wp_from_torch(c_pool),
                _wp_from_torch(V_bcast),
                _wp_from_torch(W_acc),
                _wp_from_torch(b_acc),
                _wp_from_torch(temp_head),
                _wp_from_torch(temp_node),
                _wp_from_torch(lse_m),
                _wp_from_torch(lse_is),
                _wp_from_torch(dot_ww_buf),
                _wp_from_torch(ptr_i32),
                _wp_from_torch(seed_i32),
                B,
                G,
                D,
                1 if is_plus else 0,
                1 if temp_per_node else 0,
                has_pool,
                has_bcast,
                1 if reuse_x_as_src else 0,
            ],
            outputs=[_wp_from_torch(d_x_raw), _wp_from_torch(d_src_raw)],
            device=wp_dev,
            stream=stream,
            record_tape=False,
        )

    partial_dW = torch.empty((m_max, H, G, D), dtype=acc_dtype, device=dev)
    partial_db = torch.empty((m_max, H, G), dtype=acc_dtype, device=dev)
    wp.launch(
        K["bwd_weight_chunk"],
        dim=(m_max, H, G, n_d_tiles),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(src_io),
            _wp_from_torch(dout_io),
            _wp_from_torch(U_pool),
            _wp_from_torch(c_pool),
            _wp_from_torch(V_bcast),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(lse_m),
            _wp_from_torch(lse_is),
            _wp_from_torch(dot_ww_buf),
            _wp_from_torch(seed_i32),
            _wp_from_torch(c_start),
            _wp_from_torch(c_end),
            _wp_from_torch(c_graph),
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
            has_pool,
            has_bcast,
        ],
        outputs=[_wp_from_torch(partial_dW), _wp_from_torch(partial_db)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )

    n_w_splits = _WEIGHT_RED_SPLITS
    stage1_dW = torch.empty((G, D, n_w_splits), dtype=acc_dtype, device=dev)
    stage1_db = torch.empty((G, n_w_splits), dtype=acc_dtype, device=dev)
    d_W_acc = torch.empty((G, D), dtype=acc_dtype, device=dev)
    d_b_acc = torch.empty((G,), dtype=acc_dtype, device=dev)

    wp.launch(
        K["weight_reduce_s1"],
        dim=(G, D, n_w_splits),
        inputs=[
            _wp_from_torch(partial_dW),
            _wp_from_torch(partial_db),
            _wp_from_torch(g_off),
            B,
            H,
            n_w_splits,
        ],
        outputs=[_wp_from_torch(stage1_dW), _wp_from_torch(stage1_db)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    del partial_dW, partial_db
    wp.launch(
        K["weight_reduce_s2"],
        dim=(G, D),
        inputs=[
            _wp_from_torch(stage1_dW),
            _wp_from_torch(stage1_db),
            n_w_splits,
        ],
        outputs=[_wp_from_torch(d_W_acc), _wp_from_torch(d_b_acc)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    del stage1_dW, stage1_db

    if temp_per_node:
        d_temp_out = d_temp_node.reshape_as(temp_orig).to(temp_orig.dtype).contiguous()
    else:
        n_t_splits = _TEMP_RED_SPLITS
        stage1_dt = torch.empty((H, n_t_splits), dtype=acc_dtype, device=dev)
        d_temp_head = torch.empty((H,), dtype=acc_dtype, device=dev)
        wp.launch(
            K["temp_reduce_s1"],
            dim=(H, n_t_splits),
            inputs=[_wp_from_torch(d_temp_node), N, n_t_splits],
            outputs=[_wp_from_torch(stage1_dt)],
            device=wp_dev,
            stream=stream,
            record_tape=False,
        )
        wp.launch(
            K["temp_reduce_s2"],
            dim=H,
            inputs=[_wp_from_torch(stage1_dt), n_t_splits],
            outputs=[_wp_from_torch(d_temp_head)],
            device=wp_dev,
            stream=stream,
            record_tape=False,
        )
        d_temp_out = d_temp_head.reshape_as(temp_orig).to(temp_orig.dtype).contiguous()

    return d_x_raw, d_src_raw, d_W_acc, d_b_acc, d_temp_out


def _run_slice_pool_fwd(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    K, acc_dtype, cast_io = _get_kernels(orig_dtype)

    if N == 0:
        token_acc = torch.zeros((B, H, G, D), dtype=acc_dtype, device=dev)
        inv_denom = torch.full((B, H, G), 1.0e5, dtype=acc_dtype, device=dev)
        lse_max = torch.zeros((0, H), dtype=acc_dtype, device=dev)
        lse_inv_sum = torch.zeros((0, H), dtype=acc_dtype, device=dev)
        return token_acc.to(orig_dtype), token_acc, inv_denom, lse_max, lse_inv_sum

    wp_dev = wp.device_from_torch(dev)
    stream = wp.stream_from_torch(dev)

    x_io = (x_mid.to(acc_dtype) if cast_io else x_mid).contiguous()
    src_io = x_io if reuse_x_as_src else (src.to(acc_dtype) if cast_io else src).contiguous()
    W_acc = W_slice.to(device=dev, dtype=acc_dtype).contiguous()
    b_acc = b_slice.to(device=dev, dtype=acc_dtype).contiguous()
    temp_acc = temp.to(device=dev, dtype=acc_dtype).contiguous()
    temp_head, temp_node = _split_temp_views(temp_acc, N, H, temp_per_node)
    ptr_i32 = graph_ptr.to(device=dev, dtype=torch.int32).contiguous()
    seed_i32 = seed_arr.to(device=dev, dtype=torch.int32).reshape(1).contiguous()

    lse_max = torch.empty((N, H), dtype=acc_dtype, device=dev)
    lse_inv_sum = torch.empty((N, H), dtype=acc_dtype, device=dev)

    wp.launch(
        K["lse_fwd"],
        dim=(N, H),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(seed_i32),
            G,
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
        ],
        outputs=[_wp_from_torch(lse_max), _wp_from_torch(lse_inv_sum)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )

    m_max, g_off, c_start, c_end, c_graph = _build_chunk_schedule(
        K, ptr_i32, N, B, wp_dev, stream
    )
    n_d_tiles = (D + 15) // 16
    partial_norm = torch.empty((m_max, H, G), dtype=acc_dtype, device=dev)
    partial_num = torch.empty((m_max, H, G, D), dtype=acc_dtype, device=dev)

    wp.launch(
        K["pool_chunk"],
        dim=(m_max, H, G, n_d_tiles),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(src_io),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(lse_max),
            _wp_from_torch(lse_inv_sum),
            _wp_from_torch(seed_i32),
            _wp_from_torch(c_start),
            _wp_from_torch(c_end),
            _wp_from_torch(c_graph),
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
            1,
        ],
        outputs=[_wp_from_torch(partial_norm), _wp_from_torch(partial_num)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )

    io_out_dtype = acc_dtype if cast_io else orig_dtype
    inv_denom = torch.empty((B, H, G), dtype=acc_dtype, device=dev)
    token_acc = torch.empty((B, H, G, D), dtype=acc_dtype, device=dev)
    token_out_raw = torch.empty((B, H, G, D), dtype=io_out_dtype, device=dev)

    _launch_pool_reduce_2stage(
        K,
        partial_norm,
        partial_num,
        g_off,
        B,
        H,
        G,
        D,
        1,
        inv_denom,
        token_acc,
        token_out_raw,
        acc_dtype,
        dev,
        wp_dev,
        stream,
    )
    del partial_norm, partial_num

    token_out = token_out_raw.to(orig_dtype) if cast_io else token_out_raw
    return token_out, token_acc, inv_denom, lse_max, lse_inv_sum


def _run_slice_pool_bwd(
    d_token: torch.Tensor,
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    K, acc_dtype, cast_io = _get_kernels(orig_dtype)

    if N == 0:
        return (
            torch.zeros_like(x_mid),
            torch.zeros_like(src),
            torch.zeros_like(W_slice),
            torch.zeros_like(b_slice),
            torch.zeros_like(temp),
        )

    wp_dev = wp.device_from_torch(dev)
    stream = wp.stream_from_torch(dev)
    io_out_dtype = acc_dtype if cast_io else orig_dtype

    x_io = (x_mid.to(acc_dtype) if cast_io else x_mid).contiguous()
    src_io = x_io if reuse_x_as_src else (src.to(acc_dtype) if cast_io else src).contiguous()
    dummy_dout = torch.empty((1, 1, 1), dtype=io_out_dtype, device=dev)
    dummy_vbcast = torch.empty((1, 1, 1, 1), dtype=acc_dtype, device=dev)
    W_acc = W_slice.to(device=dev, dtype=acc_dtype).contiguous()
    b_acc = b_slice.to(device=dev, dtype=acc_dtype).contiguous()
    temp_acc = temp.to(device=dev, dtype=acc_dtype).contiguous()
    temp_head, temp_node = _split_temp_views(temp_acc, N, H, temp_per_node)
    ptr_i32 = graph_ptr.to(device=dev, dtype=torch.int32).contiguous()
    seed_i32 = seed_arr.to(device=dev, dtype=torch.int32).reshape(1).contiguous()

    d_tok_acc = d_token.to(device=dev, dtype=acc_dtype)
    U_field = (d_tok_acc * inv_denom.unsqueeze(-1)).contiguous()
    c_field = (-(U_field * token_acc).sum(dim=-1)).contiguous()

    m_max, g_off, c_start, c_end, c_graph = _build_chunk_schedule(
        K, ptr_i32, N, B, wp_dev, stream
    )
    d_x_raw, d_src_raw, d_W_acc, d_b_acc, d_temp_out = _launch_unified_node_and_param_bwd(
        K,
        x_io,
        src_io,
        dummy_dout,
        U_field,
        c_field,
        dummy_vbcast,
        W_acc,
        b_acc,
        temp_head,
        temp_node,
        lse_max.contiguous(),
        lse_inv_sum.contiguous(),
        ptr_i32,
        seed_i32,
        g_off,
        c_start,
        c_end,
        c_graph,
        m_max,
        N,
        B,
        H,
        G,
        D,
        is_plus,
        temp_per_node,
        1,
        0,
        reuse_x_as_src,
        acc_dtype,
        io_out_dtype,
        temp,
        dev,
        wp_dev,
        stream,
    )

    d_x_out = (d_x_raw.to(orig_dtype) if cast_io else d_x_raw).contiguous()
    d_src_out = (
        torch.zeros_like(src)
        if reuse_x_as_src
        else (d_src_raw.to(src.dtype) if cast_io else d_src_raw).contiguous()
    )
    return (
        d_x_out,
        d_src_out,
        d_W_acc.to(W_slice.dtype).contiguous(),
        d_b_acc.to(b_slice.dtype).contiguous(),
        d_temp_out,
    )


def _run_slice_broadcast_fwd(
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> torch.Tensor:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    if N == 0:
        return torch.empty_like(x_mid)

    K, acc_dtype, cast_io = _get_kernels(orig_dtype)
    wp_dev = wp.device_from_torch(dev)
    stream = wp.stream_from_torch(dev)
    io_out_dtype = acc_dtype if cast_io else orig_dtype

    x_io = (x_mid.to(acc_dtype) if cast_io else x_mid).contiguous()
    tok_acc = out_token.to(device=dev, dtype=acc_dtype).contiguous()
    W_acc = W_slice.to(device=dev, dtype=acc_dtype).contiguous()
    b_acc = b_slice.to(device=dev, dtype=acc_dtype).contiguous()
    temp_acc = temp.to(device=dev, dtype=acc_dtype).contiguous()
    temp_head, temp_node = _split_temp_views(temp_acc, N, H, temp_per_node)
    ptr_i32 = graph_ptr.to(device=dev, dtype=torch.int32).contiguous()
    seed_i32 = seed_arr.to(device=dev, dtype=torch.int32).reshape(1).contiguous()

    n_d_tiles = (D + 15) // 16
    out_raw = torch.empty((N, H, D), dtype=io_out_dtype, device=dev)

    wp.launch(
        K["bcast_fwd"],
        dim=(N, H, n_d_tiles),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(tok_acc),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(lse_max.contiguous()),
            _wp_from_torch(lse_inv_sum.contiguous()),
            _wp_from_torch(ptr_i32),
            _wp_from_torch(seed_i32),
            B,
            G,
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
        ],
        outputs=[_wp_from_torch(out_raw)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )
    return (out_raw.to(orig_dtype) if cast_io else out_raw).contiguous()


def _run_slice_broadcast_bwd(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    if N == 0:
        return (
            torch.zeros_like(x_mid),
            torch.zeros_like(out_token),
            torch.zeros_like(W_slice),
            torch.zeros_like(b_slice),
            torch.zeros_like(temp),
        )

    K, acc_dtype, cast_io = _get_kernels(orig_dtype)
    wp_dev = wp.device_from_torch(dev)
    stream = wp.stream_from_torch(dev)
    io_out_dtype = acc_dtype if cast_io else orig_dtype

    x_io = (x_mid.to(acc_dtype) if cast_io else x_mid).contiguous()
    dout_io = (d_out.to(acc_dtype) if cast_io else d_out).contiguous()
    tok_acc = out_token.to(device=dev, dtype=acc_dtype).contiguous()
    W_acc = W_slice.to(device=dev, dtype=acc_dtype).contiguous()
    b_acc = b_slice.to(device=dev, dtype=acc_dtype).contiguous()
    temp_acc = temp.to(device=dev, dtype=acc_dtype).contiguous()
    temp_head, temp_node = _split_temp_views(temp_acc, N, H, temp_per_node)
    ptr_i32 = graph_ptr.to(device=dev, dtype=torch.int32).contiguous()
    seed_i32 = seed_arr.to(device=dev, dtype=torch.int32).reshape(1).contiguous()
    lse_m = lse_max.contiguous()
    lse_is = lse_inv_sum.contiguous()

    m_max, g_off, c_start, c_end, c_graph = _build_chunk_schedule(
        K, ptr_i32, N, B, wp_dev, stream
    )
    n_d_tiles = (D + 15) // 16

    dummy_norm = torch.empty((1, 1, 1), dtype=acc_dtype, device=dev)
    partial_dtok = torch.empty((m_max, H, G, D), dtype=acc_dtype, device=dev)
    wp.launch(
        K["pool_chunk"],
        dim=(m_max, H, G, n_d_tiles),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(dout_io),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(lse_m),
            _wp_from_torch(lse_is),
            _wp_from_torch(seed_i32),
            _wp_from_torch(c_start),
            _wp_from_torch(c_end),
            _wp_from_torch(c_graph),
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
            0,
        ],
        outputs=[_wp_from_torch(dummy_norm), _wp_from_torch(partial_dtok)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )

    dummy_inv_d = torch.empty((1, 1, 1), dtype=acc_dtype, device=dev)
    d_out_token_acc = torch.empty((B, H, G, D), dtype=acc_dtype, device=dev)
    dummy_tok_io = torch.empty((1, 1, 1, 1), dtype=io_out_dtype, device=dev)
    _launch_pool_reduce_2stage(
        K,
        dummy_norm,
        partial_dtok,
        g_off,
        B,
        H,
        G,
        D,
        0,
        dummy_inv_d,
        d_out_token_acc,
        dummy_tok_io,
        acc_dtype,
        dev,
        wp_dev,
        stream,
    )
    del partial_dtok

    dummy_src = torch.empty((1, 1, 1), dtype=io_out_dtype, device=dev)
    dummy_u = torch.empty((1, 1, 1, 1), dtype=acc_dtype, device=dev)
    dummy_c = torch.empty((1, 1, 1), dtype=acc_dtype, device=dev)

    d_x_raw, _, d_W_acc, d_b_acc, d_temp_out = _launch_unified_node_and_param_bwd(
        K,
        x_io,
        dummy_src,
        dout_io,
        dummy_u,
        dummy_c,
        tok_acc,
        W_acc,
        b_acc,
        temp_head,
        temp_node,
        lse_m,
        lse_is,
        ptr_i32,
        seed_i32,
        g_off,
        c_start,
        c_end,
        c_graph,
        m_max,
        N,
        B,
        H,
        G,
        D,
        is_plus,
        temp_per_node,
        0,
        1,
        False,
        acc_dtype,
        io_out_dtype,
        temp,
        dev,
        wp_dev,
        stream,
    )

    d_x_out = (d_x_raw.to(orig_dtype) if cast_io else d_x_raw).contiguous()
    return (
        d_x_out,
        d_out_token_acc.to(out_token.dtype).contiguous(),
        d_W_acc.to(W_slice.dtype).contiguous(),
        d_b_acc.to(b_slice.dtype).contiguous(),
        d_temp_out,
    )


def _run_physics_attn_fused_fwd(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    _, acc_dtype, _ = _get_kernels(orig_dtype)

    if N == 0:
        out_empty = torch.empty((0, H, D), dtype=orig_dtype, device=dev)
        tok_empty = torch.zeros((B, H, G, D), dtype=orig_dtype, device=dev)
        tok_acc_e = torch.zeros((B, H, G, D), dtype=acc_dtype, device=dev)
        inv_d_e = torch.full((B, H, G), 1.0e5, dtype=acc_dtype, device=dev)
        probs_e = torch.zeros((B, H, G, G), dtype=acc_dtype, device=dev)
        lse_e = torch.zeros((0, H), dtype=acc_dtype, device=dev)
        return (
            out_empty,
            tok_empty,
            tok_acc_e,
            inv_d_e,
            tok_acc_e,
            tok_acc_e,
            tok_acc_e,
            probs_e,
            tok_empty,
            lse_e,
            lse_e,
        )

    token, token_acc, inv_denom, lse_max, lse_inv_sum = _run_slice_pool_fwd(
        x_mid,
        src,
        W_slice,
        b_slice,
        temp,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
    )

    tok_f = token.to(device=dev, dtype=acc_dtype)
    wq_f = W_qkv.to(device=dev, dtype=acc_dtype)
    bq_f = b_qkv.to(device=dev, dtype=acc_dtype) if has_b_qkv else None
    qkv = F.linear(tok_f, wq_f, bq_f)
    q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
    q = q_raw.contiguous()
    k = k_raw.contiguous()
    v = v_raw.contiguous()

    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = torch.softmax(scores, dim=-1).contiguous()
    out_token_acc = torch.matmul(attn_probs, v).contiguous()
    out_token = out_token_acc.to(orig_dtype).contiguous()

    out = _run_slice_broadcast_fwd(
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
    )
    return (
        out,
        token.contiguous(),
        token_acc.contiguous(),
        inv_denom.contiguous(),
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max.contiguous(),
        lse_inv_sum.contiguous(),
    )


def _run_physics_attn_fused_bwd(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    token: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_probs: torch.Tensor,
    out_token: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = int(graph_ptr.shape[0]) - 1
    dev = x_mid.device
    orig_dtype = x_mid.dtype
    if N == 0:
        return (
            torch.zeros_like(x_mid),
            torch.zeros_like(src),
            torch.zeros_like(W_slice),
            torch.zeros_like(b_slice),
            torch.zeros_like(temp),
            torch.zeros_like(W_qkv),
            torch.zeros_like(b_qkv),
        )

    K, acc_dtype, cast_io = _get_kernels(orig_dtype)
    wp_dev = wp.device_from_torch(dev)
    stream = wp.stream_from_torch(dev)
    io_out_dtype = acc_dtype if cast_io else orig_dtype

    x_io = (x_mid.to(acc_dtype) if cast_io else x_mid).contiguous()
    src_io = x_io if reuse_x_as_src else (src.to(acc_dtype) if cast_io else src).contiguous()
    dout_io = (d_out.to(acc_dtype) if cast_io else d_out).contiguous()
    tok_bcast_acc = out_token.to(device=dev, dtype=acc_dtype).contiguous()
    W_acc = W_slice.to(device=dev, dtype=acc_dtype).contiguous()
    b_acc = b_slice.to(device=dev, dtype=acc_dtype).contiguous()
    temp_acc = temp.to(device=dev, dtype=acc_dtype).contiguous()
    temp_head, temp_node = _split_temp_views(temp_acc, N, H, temp_per_node)
    ptr_i32 = graph_ptr.to(device=dev, dtype=torch.int32).contiguous()
    seed_i32 = seed_arr.to(device=dev, dtype=torch.int32).reshape(1).contiguous()
    lse_m = lse_max.contiguous()
    lse_is = lse_inv_sum.contiguous()

    m_max, g_off, c_start, c_end, c_graph = _build_chunk_schedule(
        K, ptr_i32, N, B, wp_dev, stream
    )
    n_d_tiles = (D + 15) // 16

    dummy_norm = torch.empty((1, 1, 1), dtype=acc_dtype, device=dev)
    partial_dtok = torch.empty((m_max, H, G, D), dtype=acc_dtype, device=dev)
    wp.launch(
        K["pool_chunk"],
        dim=(m_max, H, G, n_d_tiles),
        inputs=[
            _wp_from_torch(x_io),
            _wp_from_torch(dout_io),
            _wp_from_torch(W_acc),
            _wp_from_torch(b_acc),
            _wp_from_torch(temp_head),
            _wp_from_torch(temp_node),
            _wp_from_torch(lse_m),
            _wp_from_torch(lse_is),
            _wp_from_torch(seed_i32),
            _wp_from_torch(c_start),
            _wp_from_torch(c_end),
            _wp_from_torch(c_graph),
            D,
            1 if is_plus else 0,
            1 if temp_per_node else 0,
            0,
        ],
        outputs=[_wp_from_torch(dummy_norm), _wp_from_torch(partial_dtok)],
        device=wp_dev,
        stream=stream,
        record_tape=False,
    )

    dummy_inv_d = torch.empty((1, 1, 1), dtype=acc_dtype, device=dev)
    d_out_token_acc = torch.empty((B, H, G, D), dtype=acc_dtype, device=dev)
    dummy_tok_io = torch.empty((1, 1, 1, 1), dtype=io_out_dtype, device=dev)
    _launch_pool_reduce_2stage(
        K,
        dummy_norm,
        partial_dtok,
        g_off,
        B,
        H,
        G,
        D,
        0,
        dummy_inv_d,
        d_out_token_acc,
        dummy_tok_io,
        acc_dtype,
        dev,
        wp_dev,
        stream,
    )
    del partial_dtok

    scale = D ** -0.5
    q_acc = q.to(device=dev, dtype=acc_dtype)
    k_acc = k.to(device=dev, dtype=acc_dtype)
    v_acc = v.to(device=dev, dtype=acc_dtype)
    d_attn = torch.matmul(d_out_token_acc, v_acc.transpose(-2, -1))
    d_v = torch.matmul(attn_probs.transpose(-2, -1), d_out_token_acc)
    d_scores = attn_probs * (d_attn - (d_attn * attn_probs).sum(dim=-1, keepdim=True)) * scale
    d_q = torch.matmul(d_scores, k_acc)
    d_k = torch.matmul(d_scores.transpose(-2, -1), q_acc)
    d_qkv = torch.cat([d_q, d_k, d_v], dim=-1)

    wq_f = W_qkv.to(device=dev, dtype=acc_dtype)
    tok_f = token.to(device=dev, dtype=acc_dtype)
    d_token_acc = torch.matmul(d_qkv, wq_f)
    d_W_qkv = (
        torch.matmul(d_qkv.reshape(-1, 3 * D).t(), tok_f.reshape(-1, D))
        .to(W_qkv.dtype)
        .contiguous()
    )
    d_b_qkv = (
        d_qkv.reshape(-1, 3 * D).sum(dim=0).to(b_qkv.dtype).contiguous()
        if has_b_qkv
        else torch.zeros_like(b_qkv).contiguous()
    )

    U_pool = (d_token_acc * inv_denom.unsqueeze(-1)).contiguous()
    c_pool = (-(U_pool * token_acc).sum(dim=-1)).contiguous()

    d_x_raw, d_src_raw, d_W_acc, d_b_acc, d_temp_out = _launch_unified_node_and_param_bwd(
        K,
        x_io,
        src_io,
        dout_io,
        U_pool,
        c_pool,
        tok_bcast_acc,
        W_acc,
        b_acc,
        temp_head,
        temp_node,
        lse_m,
        lse_is,
        ptr_i32,
        seed_i32,
        g_off,
        c_start,
        c_end,
        c_graph,
        m_max,
        N,
        B,
        H,
        G,
        D,
        is_plus,
        temp_per_node,
        1,
        1,
        reuse_x_as_src,
        acc_dtype,
        io_out_dtype,
        temp,
        dev,
        wp_dev,
        stream,
    )

    d_x_out = (d_x_raw.to(orig_dtype) if cast_io else d_x_raw).contiguous()
    d_src_out = (
        torch.zeros_like(src).contiguous()
        if reuse_x_as_src
        else (d_src_raw.to(src.dtype) if cast_io else d_src_raw).contiguous()
    )
    return (
        d_x_out,
        d_src_out,
        d_W_acc.to(W_slice.dtype).contiguous(),
        d_b_acc.to(b_slice.dtype).contiguous(),
        d_temp_out,
        d_W_qkv,
        d_b_qkv,
    )


def _acc_torch_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


@torch.library.custom_op("fvgn::ragged_slice_pool", mutates_args=())
def _ragged_slice_pool_op(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_slice_pool_fwd(
        x_mid,
        src,
        W_slice,
        b_slice,
        temp,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
    )


@_ragged_slice_pool_op.register_fake
def _(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = graph_ptr.shape[0] - 1
    acc_dt = _acc_torch_dtype(x_mid.dtype)
    return (
        torch.empty((B, H, G, D), dtype=x_mid.dtype, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty((B, H, G, D), dtype=acc_dt, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty((B, H, G), dtype=acc_dt, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty((N, H), dtype=acc_dt, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty((N, H), dtype=acc_dt, device=x_mid.device, memory_format=torch.contiguous_format),
    )


def _setup_slice_pool_ctx(ctx, inputs, output):
    (
        x_mid,
        src,
        W_slice,
        b_slice,
        temp,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
    ) = inputs
    _, token_acc, inv_denom, lse_max, lse_inv_sum = output
    ctx.is_plus = bool(is_plus)
    ctx.temp_per_node = bool(temp_per_node)
    ctx.reuse_x_as_src = bool(reuse_x_as_src)
    ctx.mark_non_differentiable(token_acc, inv_denom, lse_max, lse_inv_sum)
    saved_src = x_mid if reuse_x_as_src else src
    ctx.save_for_backward(
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        token_acc,
        inv_denom,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    )


@torch.library.custom_op("fvgn::ragged_slice_pool_bwd", mutates_args=())
def _ragged_slice_pool_bwd_op(
    d_token_out: torch.Tensor,
    x_mid: torch.Tensor,
    saved_src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_slice_pool_bwd(
        d_token_out,
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        token_acc,
        inv_denom,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
    )


@_ragged_slice_pool_bwd_op.register_fake
def _(
    d_token_out: torch.Tensor,
    x_mid: torch.Tensor,
    saved_src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x_mid.shape, dtype=x_mid.dtype, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty(saved_src.shape, dtype=saved_src.dtype, device=saved_src.device, memory_format=torch.contiguous_format),
        torch.empty(W_slice.shape, dtype=W_slice.dtype, device=W_slice.device, memory_format=torch.contiguous_format),
        torch.empty(b_slice.shape, dtype=b_slice.dtype, device=b_slice.device, memory_format=torch.contiguous_format),
        torch.empty(temp.shape, dtype=temp.dtype, device=temp.device, memory_format=torch.contiguous_format),
    )


def _backward_slice_pool(ctx, d_token_out, _d_tacc, _d_inv, _d_lm, _d_lis):
    (
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        token_acc,
        inv_denom,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    ) = ctx.saved_tensors
    d_x, d_src, d_W, d_b, d_temp = _ragged_slice_pool_bwd_op(
        d_token_out,
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        token_acc,
        inv_denom,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        ctx.is_plus,
        ctx.temp_per_node,
        ctx.reuse_x_as_src,
    )
    return (
        d_x if ctx.needs_input_grad[0] else None,
        (None if ctx.reuse_x_as_src else (d_src if ctx.needs_input_grad[1] else None)),
        d_W if ctx.needs_input_grad[2] else None,
        d_b if ctx.needs_input_grad[3] else None,
        d_temp if ctx.needs_input_grad[4] else None,
        None,
        None,
        None,
        None,
        None,
    )


_ragged_slice_pool_op.register_autograd(
    _backward_slice_pool, setup_context=_setup_slice_pool_ctx
)


@torch.library.custom_op("fvgn::ragged_slice_broadcast", mutates_args=())
def _ragged_slice_broadcast_op(
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> torch.Tensor:
    return _run_slice_broadcast_fwd(
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
    )


@_ragged_slice_broadcast_op.register_fake
def _(
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> torch.Tensor:
    return torch.empty(x_mid.shape, dtype=x_mid.dtype, device=x_mid.device, memory_format=torch.contiguous_format)


def _setup_slice_broadcast_ctx(ctx, inputs, output):
    (
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
    ) = inputs
    ctx.is_plus = bool(is_plus)
    ctx.temp_per_node = bool(temp_per_node)
    ctx.save_for_backward(
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    )


@torch.library.custom_op("fvgn::ragged_slice_broadcast_bwd", mutates_args=())
def _ragged_slice_broadcast_bwd_op(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_slice_broadcast_bwd(
        d_out,
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
    )


@_ragged_slice_broadcast_bwd_op.register_fake
def _(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    out_token: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x_mid.shape, dtype=x_mid.dtype, device=x_mid.device, memory_format=torch.contiguous_format),
        torch.empty(out_token.shape, dtype=out_token.dtype, device=out_token.device, memory_format=torch.contiguous_format),
        torch.empty(W_slice.shape, dtype=W_slice.dtype, device=W_slice.device, memory_format=torch.contiguous_format),
        torch.empty(b_slice.shape, dtype=b_slice.dtype, device=b_slice.device, memory_format=torch.contiguous_format),
        torch.empty(temp.shape, dtype=temp.dtype, device=temp.device, memory_format=torch.contiguous_format),
    )


def _backward_slice_broadcast(ctx, d_out):
    (
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    ) = ctx.saved_tensors
    d_x, d_tok, d_W, d_b, d_temp = _ragged_slice_broadcast_bwd_op(
        d_out,
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        ctx.is_plus,
        ctx.temp_per_node,
    )
    return (
        d_x if ctx.needs_input_grad[0] else None,
        d_tok if ctx.needs_input_grad[1] else None,
        d_W if ctx.needs_input_grad[2] else None,
        d_b if ctx.needs_input_grad[3] else None,
        d_temp if ctx.needs_input_grad[4] else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_ragged_slice_broadcast_op.register_autograd(
    _backward_slice_broadcast, setup_context=_setup_slice_broadcast_ctx
)


@torch.library.custom_op("fvgn::ragged_physics_attn_fused", mutates_args=())
def _ragged_physics_attn_fused_op(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _run_physics_attn_fused_fwd(
        x_mid,
        src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
        has_b_qkv,
    )


@_ragged_physics_attn_fused_op.register_fake
def _(
    x_mid: torch.Tensor,
    src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    B = graph_ptr.shape[0] - 1
    acc_dt = _acc_torch_dtype(x_mid.dtype)
    dev = x_mid.device
    cf = torch.contiguous_format
    return (
        torch.empty((N, H, D), dtype=x_mid.dtype, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=x_mid.dtype, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G, G), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((B, H, G, D), dtype=x_mid.dtype, device=dev, memory_format=cf),
        torch.empty((N, H), dtype=acc_dt, device=dev, memory_format=cf),
        torch.empty((N, H), dtype=acc_dt, device=dev, memory_format=cf),
    )


@torch.library.custom_op("fvgn::ragged_physics_attn_fused_bwd", mutates_args=())
def _ragged_physics_attn_fused_bwd_op(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    saved_src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    token: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_probs: torch.Tensor,
    out_token: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _run_physics_attn_fused_bwd(
        d_out,
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        token,
        token_acc,
        inv_denom,
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
        has_b_qkv,
    )


@_ragged_physics_attn_fused_bwd_op.register_fake
def _(
    d_out: torch.Tensor,
    x_mid: torch.Tensor,
    saved_src: torch.Tensor,
    W_slice: torch.Tensor,
    b_slice: torch.Tensor,
    temp: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    token: torch.Tensor,
    token_acc: torch.Tensor,
    inv_denom: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_probs: torch.Tensor,
    out_token: torch.Tensor,
    lse_max: torch.Tensor,
    lse_inv_sum: torch.Tensor,
    graph_ptr: torch.Tensor,
    seed_arr: torch.Tensor,
    is_plus: bool,
    temp_per_node: bool,
    reuse_x_as_src: bool,
    has_b_qkv: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    cf = torch.contiguous_format
    return (
        torch.empty(x_mid.shape, dtype=x_mid.dtype, device=x_mid.device, memory_format=cf),
        torch.empty(saved_src.shape, dtype=saved_src.dtype, device=saved_src.device, memory_format=cf),
        torch.empty(W_slice.shape, dtype=W_slice.dtype, device=W_slice.device, memory_format=cf),
        torch.empty(b_slice.shape, dtype=b_slice.dtype, device=b_slice.device, memory_format=cf),
        torch.empty(temp.shape, dtype=temp.dtype, device=temp.device, memory_format=cf),
        torch.empty(W_qkv.shape, dtype=W_qkv.dtype, device=W_qkv.device, memory_format=cf),
        torch.empty(b_qkv.shape, dtype=b_qkv.dtype, device=b_qkv.device, memory_format=cf),
    )


def _setup_physics_attn_fused_ctx(ctx, inputs, output):
    (
        x_mid,
        src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        graph_ptr,
        seed_arr,
        is_plus,
        temp_per_node,
        reuse_x_as_src,
        has_b_qkv,
    ) = inputs
    (
        _out,
        token,
        token_acc,
        inv_denom,
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max,
        lse_inv_sum,
    ) = output
    ctx.is_plus = bool(is_plus)
    ctx.temp_per_node = bool(temp_per_node)
    ctx.reuse_x_as_src = bool(reuse_x_as_src)
    ctx.has_b_qkv = bool(has_b_qkv)
    ctx.mark_non_differentiable(
        token, token_acc, inv_denom, q, k, v, attn_probs, out_token, lse_max, lse_inv_sum
    )
    saved_src = x_mid if reuse_x_as_src else src
    ctx.save_for_backward(
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        token,
        token_acc,
        inv_denom,
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    )


def _backward_physics_attn_fused(
    ctx,
    d_out,
    _d_tok,
    _d_tacc,
    _d_inv,
    _d_q,
    _d_k,
    _d_v,
    _d_probs,
    _d_otok,
    _d_lm,
    _d_lis,
):
    (
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        token,
        token_acc,
        inv_denom,
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
    ) = ctx.saved_tensors
    d_x, d_src, d_Ws, d_bs, d_temp, d_Wqkv, d_bqkv = _ragged_physics_attn_fused_bwd_op(
        d_out,
        x_mid,
        saved_src,
        W_slice,
        b_slice,
        temp,
        W_qkv,
        b_qkv,
        token,
        token_acc,
        inv_denom,
        q,
        k,
        v,
        attn_probs,
        out_token,
        lse_max,
        lse_inv_sum,
        graph_ptr,
        seed_arr,
        ctx.is_plus,
        ctx.temp_per_node,
        ctx.reuse_x_as_src,
        ctx.has_b_qkv,
    )
    return (
        d_x if ctx.needs_input_grad[0] else None,
        (None if ctx.reuse_x_as_src else (d_src if ctx.needs_input_grad[1] else None)),
        d_Ws if ctx.needs_input_grad[2] else None,
        d_bs if ctx.needs_input_grad[3] else None,
        d_temp if ctx.needs_input_grad[4] else None,
        d_Wqkv if ctx.needs_input_grad[5] else None,
        (d_bqkv if (ctx.has_b_qkv and ctx.needs_input_grad[6]) else None),
        None,
        None,
        None,
        None,
        None,
        None,
    )


_ragged_physics_attn_fused_op.register_autograd(
    _backward_physics_attn_fused, setup_context=_setup_physics_attn_fused_ctx
)


def sample_counter_gumbel(
    seed: int,
    N: int,
    H: int,
    G: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact PyTorch replica of ``gumbel_from_counter`` for unit-test parity verification."""
    n_idx = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1)
    h_idx = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1)
    g_idx = torch.arange(G, device=device, dtype=torch.int64).view(1, 1, G)
    mask32 = 0xFFFFFFFF
    k1 = (n_idx * 0x9E3779B9) & mask32
    k2 = ((h_idx * 0x85EBCA6B) & mask32) ^ ((g_idx * 0xC2B2AE35) & mask32)
    x = (int(seed) & mask32) ^ k1 ^ k2
    x = (((x ^ (x >> 16)) & mask32) * 0x7FEB352D) & mask32
    x = (((x ^ (x >> 15)) & mask32) * 0x846CA68B) & mask32
    x = (x ^ (x >> 16)) & mask32
    u = ((x >> 9).to(torch.float32) + 0.5) * 1.1920928955078125e-07
    u_acc = u.to(torch.float64 if dtype == torch.float64 else torch.float32)
    return (-torch.log(-torch.log(u_acc))).to(dtype)


def ragged_physics_attention_warp(
    x_mid: torch.Tensor,
    src: Optional[torch.Tensor],
    W_slice: torch.Tensor,
    b_slice: Optional[torch.Tensor],
    temp: torch.Tensor,
    to_qkv_fn,
    graph_ptr: torch.Tensor,
    *,
    plus: bool = False,
    gumbel_seed: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Full Pass 1 -> Pass 2 (SDPA) -> Pass 3 ragged Physics-Attention on CUDA."""
    N, H, D = x_mid.shape
    G = W_slice.shape[0]
    dev = x_mid.device
    if b_slice is None:
        b_slice = x_mid.new_zeros((G,))

    reuse_x_as_src = bool(plus) or (src is None)
    dummy_src = x_mid.new_zeros((1, 1, 1)) if reuse_x_as_src else src
    temp_per_node = bool(plus) or (temp.numel() != H)

    if plus:
        if gumbel_seed is None:
            seed_arr = torch.randint(0, 0x7FFFFFFF, (1,), dtype=torch.int32, device=dev)
        elif torch.is_tensor(gumbel_seed):
            seed_arr = gumbel_seed.to(device=dev, dtype=torch.int32).reshape(1)
        else:
            seed_arr = torch.full((1,), int(gumbel_seed), dtype=torch.int32, device=dev)
    else:
        seed_arr = torch.zeros((1,), dtype=torch.int32, device=dev)

    x_mid = x_mid.contiguous()
    dummy_src = dummy_src.contiguous()
    W_slice = W_slice.contiguous()
    b_slice = b_slice.contiguous()
    temp = temp.contiguous()
    graph_ptr = graph_ptr.contiguous()

    if isinstance(to_qkv_fn, torch.nn.Linear):
        W_qkv = to_qkv_fn.weight.contiguous()
        has_b_qkv = to_qkv_fn.bias is not None
        b_qkv = (
            to_qkv_fn.bias.contiguous()
            if has_b_qkv
            else x_mid.new_zeros((3 * D,))
        )
        out = _ragged_physics_attn_fused_op(
            x_mid,
            dummy_src,
            W_slice,
            b_slice,
            temp,
            W_qkv,
            b_qkv,
            graph_ptr,
            seed_arr,
            bool(plus),
            temp_per_node,
            reuse_x_as_src,
            has_b_qkv,
        )[0]
        return out.reshape(N, H * D)

    token, _tok_acc, _inv_denom, lse_max, lse_inv_sum = _ragged_slice_pool_op(
        x_mid,
        dummy_src,
        W_slice,
        b_slice,
        temp,
        graph_ptr,
        seed_arr,
        bool(plus),
        temp_per_node,
        reuse_x_as_src,
    )

    q, k, v = to_qkv_fn(token).chunk(3, dim=-1)
    out_token = F.scaled_dot_product_attention(q, k, v, scale=D ** -0.5).contiguous()

    out = _ragged_slice_broadcast_op(
        x_mid,
        out_token,
        W_slice,
        b_slice,
        temp,
        lse_max.detach(),
        lse_inv_sum.detach(),
        graph_ptr,
        seed_arr,
        bool(plus),
        temp_per_node,
    )
    return out.reshape(N, H * D)
