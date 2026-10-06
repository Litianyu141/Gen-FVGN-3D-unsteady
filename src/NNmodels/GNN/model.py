"""``--net GNN``: TransFVGN without the attention blocks (message passing only)."""
from NNmodels.TransFVGN.model import Simulator as _TransFVGN


class Simulator(_TransFVGN):
    use_attention = False
