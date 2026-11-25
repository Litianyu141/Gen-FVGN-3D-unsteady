"""Per-step wall clock and peak GPU memory of an autoregressive rollout."""
import json
import time

import numpy as np
import torch


class StepTimer:
    def __init__(self, device, warmup=5):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.warmup = warmup
        self.times = []
        self._t0 = None
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self):
        self._sync()
        self.times.append(time.perf_counter() - self._t0)

    def summary(self, **extra):
        t = np.asarray(self.times, dtype=np.float64)
        w = min(self.warmup, max(len(t) - 1, 0))
        steady = t[w:]
        out = dict(
            n_steps=int(len(t)), warmup_steps=int(w),
            total_loop_s=float(t.sum()), warmup_s=float(t[:w].sum()),
            step_mean_s=float(steady.mean()), step_median_s=float(np.median(steady)),
            step_p95_s=float(np.percentile(steady, 95)),
            step_min_s=float(steady.min()), step_max_s=float(steady.max()),
        )
        if self.cuda:
            out.update(
                device=torch.cuda.get_device_name(self.device),
                peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
            )
        else:
            out.update(device="cpu", peak_allocated_gib=None, peak_reserved_gib=None)
        out.update(extra)
        return out

    def write(self, path, **extra):
        s = self.summary(**extra)
        with open(path, "w") as f:
            json.dump(s, f, indent=1)
        return s
