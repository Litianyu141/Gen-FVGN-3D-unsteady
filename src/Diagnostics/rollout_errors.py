"""Per-step error of a rollout against the CFD labels."""
import numpy as np
import torch


class RolloutErrors:
    """Per step and channel, over the interior cells: the nRMSE ``||pred - label|| / ||label||``
    and the moments ``mae``, ``l1_diff``, ``l2_diff``, ``l1_label``, ``l2_label``.

    ``labels[k - 1]`` holds the state at ``t = k dt``; ``offsets`` are per-channel constants
    added to the prediction first (``{"T": 300.15}`` for a case solved in ``T - 300.15 K``
    against labels in kelvin)."""

    def __init__(self, labels, interior_idx, channels, *, device="cpu", offsets=None):
        self.labels, self.idx, self.ch = labels, interior_idx, list(channels)
        self.offsets = dict(offsets or {})
        self.dev = device
        self.rows = []

    def step(self, step, pred_interior, dt):
        p = pred_interior.float().to(self.dev)
        if self.offsets:
            p = p.clone()
            for c, v in self.offsets.items():
                if c in self.ch:
                    p[:, self.ch.index(c)] += v
        g = torch.from_numpy(np.asarray(self.labels[step - 1])).to(self.dev)[self.idx].float()
        nr = torch.linalg.vector_norm(p - g, dim=0) / torch.linalg.vector_norm(g, dim=0).clamp_min(1e-30)
        row = {"step": step, "t": step * dt}
        for c, v in zip(self.ch, nr.tolist()):
            row[f"nrmse_{c}"] = v
        d = (p - g).abs()
        for i, c in enumerate(self.ch):
            row[f"mae_{c}"] = float(d[:, i].mean())
            row[f"l1_diff_{c}"] = float(d[:, i].sum())
            row[f"l2_diff_{c}"] = float(torch.linalg.vector_norm(p[:, i] - g[:, i]))
            row[f"l1_label_{c}"] = float(g[:, i].abs().sum())
            row[f"l2_label_{c}"] = float(torch.linalg.vector_norm(g[:, i]))
        row["n_cells"] = int(p.shape[0])
        self.rows.append(row)
        return row

    def write(self, csv_path):
        keys = list(self.rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in self.rows:
                f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
