"""Pure Transolver++ (Luo et al., ICML 2025) using Warp ragged Physics-Attention."""

from NNmodels.Transolver.model import Simulator as _Transolver


class Simulator(_Transolver):
    def __init__(self, *args, **kwargs):
        kwargs["plus"] = True
        super().__init__(*args, **kwargs)
