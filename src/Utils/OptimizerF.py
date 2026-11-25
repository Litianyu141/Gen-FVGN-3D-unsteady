"""Optimizers for PCNO and ODIL training, including batched-QR SOAP."""

from itertools import chain
import logging

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


class SOAP(optim.Optimizer):
    """SOAP optimizer (https://arxiv.org/abs/2409.11321) with batched QR eigenbasis refresh."""

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)

        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        return grad.reshape(new_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            merge_dims = group["merge_dims"]
            max_precond_dim = group["max_precond_dim"]
            precondition_frequency = group["precondition_frequency"]
            precondition_1d = group["precondition_1d"]
            correct_bias = group["correct_bias"]
            normalize_grads = group["normalize_grads"]
            shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else beta2

            qr_pending = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=precondition_frequency,
                        precondition_1d=precondition_1d,
                        shampoo_beta=shampoo_beta,
                        max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims,
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims,
                        precondition_1d=precondition_1d,
                    )
                    continue

                grad_projected = self.project(
                    grad, state, merge_dims=merge_dims, max_precond_dim=max_precond_dim
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))
                denom = exp_avg_sq.sqrt().add_(eps)

                step_size = lr
                if correct_bias:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                norm_grad = self.project_back(
                    exp_avg / denom,
                    state,
                    merge_dims=merge_dims,
                    max_precond_dim=max_precond_dim,
                )

                if normalize_grads:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                if weight_decay > 0.0:
                    p.add_(p, alpha=(-lr * weight_decay))

                self._accumulate_preconditioner(
                    grad,
                    state,
                    max_precond_dim=max_precond_dim,
                    merge_dims=merge_dims,
                    precondition_1d=precondition_1d,
                )
                if (
                    state["step"] > 0
                    and state["step"] % state["precondition_frequency"] == 0
                ):
                    qr_pending.append(state)

            if qr_pending:
                self._refresh_eigenbases_batched(
                    qr_pending,
                    max_precond_dim=max_precond_dim,
                    merge_dims=merge_dims,
                )

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        Q = state["Q"]
        if not merge_dims:
            ndim = grad.dim()
            if ndim == 1:
                mat = Q[0]
                return grad @ mat if len(mat) > 0 else grad
            if ndim == 2:
                q0, q1 = Q[0], Q[1]
                out = grad
                if len(q0) > 0:
                    out = q0.T @ out
                if len(q1) > 0:
                    out = out @ q1
                return out

        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in Q:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        Q = state["Q"]
        if not merge_dims:
            ndim = grad.dim()
            if ndim == 1:
                mat = Q[0]
                return mat @ grad if len(mat) > 0 else grad
            if ndim == 2:
                q0, q1 = Q[0], Q[1]
                out = grad
                if len(q0) > 0:
                    out = q0 @ out
                if len(q1) > 0:
                    out = out @ q1.T
                return out

        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in Q:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _accumulate_preconditioner(
        self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False
    ):
        alpha = 1.0 - state["shampoo_beta"]
        ndim = grad.dim()
        if ndim == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), alpha)
            return
        if ndim == 2 and not merge_dims:
            if grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad @ grad.T, alpha)
            if grad.shape[1] <= max_precond_dim:
                state["GG"][1].lerp_(grad.T @ grad, alpha)
            return
        target = self.merge_dims(grad, max_precond_dim) if merge_dims else grad
        for idx, sh in enumerate(target.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(
                    target,
                    target,
                    dims=[[*chain(range(idx), range(idx + 1, len(target.shape)))]] * 2,
                )
                state["GG"][idx].lerp_(outer_product, alpha)

    def _refresh_eigenbases_batched(self, states, max_precond_dim=10000, merge_dims=False):
        """Batched power-iteration + QR eigenbasis refresh grouped by matrix shape."""
        for state in states:
            if state["Q"] is not None:
                state["exp_avg"] = self.project_back(
                    state["exp_avg"],
                    state,
                    merge_dims=merge_dims,
                    max_precond_dim=max_precond_dim,
                )

        exp_avg_sq_work = []
        meta_per_state = []
        buckets = {}
        new_Q_lists = []

        for s_idx, state in enumerate(states):
            orig_shape = state["exp_avg_sq"].shape
            permuted_shape = None
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
            eas = (
                self.merge_dims(state["exp_avg_sq"], max_precond_dim)
                if merge_dims
                else state["exp_avg_sq"]
            )
            exp_avg_sq_work.append(eas)
            meta_per_state.append((orig_shape, permuted_shape))

            precond_list = state["GG"]
            orth_list = state["Q"]
            new_Q = [None] * len(precond_list)
            for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
                if len(m) == 0:
                    new_Q[ind] = []
                    continue
                key = (m.device, m.shape[0])
                buckets.setdefault(key, []).append(
                    (s_idx, ind, m.data.float(), o.data.float(), m.data.dtype, m.data.device)
                )
            new_Q_lists.append(new_Q)

        for (_dev, _dim), items in buckets.items():
            M_batch = torch.stack([it[2] for it in items], dim=0)
            O_batch = torch.stack([it[3] for it in items], dim=0)
            MO_batch = torch.bmm(M_batch, O_batch)
            est_eig = (O_batch * MO_batch).sum(dim=1)
            sort_idx = torch.argsort(est_eig, dim=1, descending=True)
            gather_idx = sort_idx.unsqueeze(1).expand_as(MO_batch)
            power_iter = torch.gather(MO_batch, 2, gather_idx)
            Q_batch, _ = torch.linalg.qr(power_iter)

            for k, (s_idx, ind, _m, _o, orig_dtype, orig_dev) in enumerate(items):
                exp_avg_sq_work[s_idx] = exp_avg_sq_work[s_idx].index_select(ind, sort_idx[k])
                qk = Q_batch[k]
                if orig_dtype != torch.float32:
                    qk = qk.to(orig_dev, dtype=orig_dtype)
                new_Q_lists[s_idx][ind] = qk

        for s_idx, state in enumerate(states):
            eas = exp_avg_sq_work[s_idx]
            orig_shape, permuted_shape = meta_per_state[s_idx]
            if merge_dims:
                if self._data_format == "channels_last" and len(orig_shape) == 4:
                    eas = eas.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    eas = eas.reshape(orig_shape)
            state["exp_avg_sq"] = eas
            state["Q"] = new_Q_lists[s_idx]
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def update_preconditioner(
        self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False
    ):
        will_refresh = (
            state["Q"] is None
            or (state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0)
        )
        if state["Q"] is not None and will_refresh:
            state["exp_avg"] = self.project_back(
                state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim
            )

        self._accumulate_preconditioner(
            grad,
            state,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
        )

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])

        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

        if state["step"] > 0 and will_refresh:
            state["exp_avg"] = self.project(
                state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim
            )

    def get_orthogonal_matrix(self, mat):
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except Exception:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        precond_list = state["GG"]
        orth_list = state["Q"]
        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
            else:
                float_data = True
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        exp_avg_sq = (
            self.merge_dims(state["exp_avg_sq"], max_precond_dim)
            if merge_dims
            else state["exp_avg_sq"]
        )

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            Q, _ = torch.linalg.qr(m @ o)
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return final


def get_optimizer(optimizer_type, opt_parameters, params):
    """Build an optimizer (``soap``, ``adam``, ``adamw``, ``sgd``, or ``sgd_pure``)."""
    from torch.optim import Adam, AdamW, SGD

    optimizer_type = str(optimizer_type).lower()
    weight_decay = float(getattr(params, "weight_decay", 0.0))

    if optimizer_type == "soap":
        freq = int(getattr(params, "soap_precond_freq", 10))
        optimizer = SOAP(
            opt_parameters,
            lr=params.lr,
            betas=(0.95, 0.95),
            weight_decay=weight_decay,
            precondition_frequency=freq,
            max_precond_dim=10000,
            precondition_1d=False,
            normalize_grads=False,
            eps=1e-8,
            correct_bias=True,
        )
        logger.info("Using SOAP optimizer (lr=%g, freq=%d, wd=%g)", params.lr, freq, weight_decay)
    elif optimizer_type == "adam":
        adam_eps = float(getattr(params, "adam_eps", 1e-8))
        optimizer = Adam(
            opt_parameters,
            lr=params.lr,
            betas=(0.9, 0.999),
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        logger.info("Using Adam optimizer (lr=%g, eps=%g, wd=%g)", params.lr, adam_eps, weight_decay)
    elif optimizer_type == "adamw":
        wd = weight_decay if weight_decay > 0 else 0.01
        optimizer = AdamW(
            opt_parameters,
            lr=params.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        logger.info("Using AdamW optimizer (lr=%g, wd=%g)", params.lr, wd)
    elif optimizer_type == "sgd":
        momentum = float(getattr(params, "sgd_momentum", 0.9))
        optimizer = SGD(
            opt_parameters,
            lr=params.lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        logger.info("Using SGD optimizer (lr=%g, momentum=%g, wd=%g)", params.lr, momentum, weight_decay)
    elif optimizer_type == "sgd_pure":
        optimizer = SGD(opt_parameters, lr=params.lr, momentum=0.0, weight_decay=0.0)
        logger.info("Using pure SGD optimizer (lr=%g)", params.lr)
    else:
        raise ValueError(
            f"Unsupported optimizer_type={optimizer_type!r}; "
            "expected one of ['soap', 'adam', 'adamw', 'sgd', 'sgd_pure']"
        )

    return optimizer
