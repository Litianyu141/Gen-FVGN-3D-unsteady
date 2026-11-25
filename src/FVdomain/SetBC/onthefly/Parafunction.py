"""Parameters sampled per case (``[parameter]`` of bc/*.toml or fvconfig.toml) and boundary value tensors."""

import copy
import logging
import random
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class ParameterSampler:
    """Sampler supporting ``constant``, ``uniform``, ``choices``, and ``arithmetic`` specs."""

    def __init__(self, name: str, spec: dict, seed: int = None):
        self.name = name
        self.spec = copy.deepcopy(spec)
        self.value_type = spec.get("value_type", "constant")
        self._rng = random.Random(seed)
        self._pool: List = []
        self._current_global_value = None
        self._global_value_set = False
        self._init_pool()

    def _init_pool(self):
        if self.value_type == "arithmetic":
            start, step, end = self.spec["val"]
            n = int((end - start) / step) + 1
            self._pool = [start + i * step for i in range(n)]
            if abs(self._pool[-1] - end) > 1e-12:
                self._pool.append(end)
        elif self.value_type == "choices":
            val = self.spec.get("val", self.spec.get("choices", []))
            self._pool = list(val) if isinstance(val, (list, tuple)) else [val]
        else:
            self._pool = []
        if self._pool:
            self._rng.shuffle(self._pool)

    def sample(self) -> Any:
        if self.value_type in ("constant", "uniform"):
            return self.spec.get("val")
        if self._global_value_set:
            return self._current_global_value
        if not self._pool:
            self._init_pool()
        self._current_global_value = self._pool.pop()
        self._global_value_set = True
        logger.info("Parameter '%s': sampled value %s", self.name, self._current_global_value)
        return self._current_global_value

    def reset(self):
        self._global_value_set = False
        self._current_global_value = None


class ParametricManager:
    """Samplers for one case plus ``$name`` substitution in boundary/PDE specs."""

    def __init__(self, seed: int = None):
        self._samplers: Dict[str, ParameterSampler] = {}
        self._seed = seed

    def register_from_toml(self, parameter_dict: dict):
        for name, spec in parameter_dict.items():
            self._samplers[name] = ParameterSampler(name, spec, self._seed)
            logger.info("Registered parameter '%s': %s", name, spec)

    def has_parameter(self, name: str) -> bool:
        return name in self._samplers

    def get_value(self, name: str) -> Any:
        if name not in self._samplers:
            raise KeyError(f"Parameter '{name}' not registered")
        return self._samplers[name].sample()

    def reset_all(self):
        for sampler in self._samplers.values():
            sampler.reset()

    def override_value(self, name: str, value: Any) -> None:
        if name in self._samplers:
            sampler = self._samplers[name]
            sampler.value_type = "constant"
            sampler.spec = {"value_type": "constant", "val": value}
            sampler._pool = []
            sampler._global_value_set = False
            sampler._current_global_value = value
        else:
            self.register_from_toml({name: {"value_type": "constant", "val": value}})
        logger.info("Parameter '%s' -> %s", name, value)

    def resolve_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.get_value(value[1:]) if value.startswith("$") else value
        if isinstance(value, (list, tuple)):
            resolved = [self.resolve_value(v) for v in value]
            return tuple(resolved) if isinstance(value, tuple) else resolved
        if isinstance(value, dict):
            return {k: self.resolve_value(v) for k, v in value.items()}
        return value

    def current_values(self) -> dict:
        return {name: s._current_global_value for name, s in self._samplers.items()}

    def get_status(self) -> dict:
        return {
            name: {
                "value_type": s.value_type,
                "current_value": s._current_global_value,
            }
            for name, s in self._samplers.items()
        }


_VALUE_TENSOR_CACHE: Dict[Any, torch.Tensor] = {}
_VALUE_TENSOR_CACHE_MAX = 512


class Functionbase:
    """Boundary value evaluator for ``uniform`` and ``constant`` specs."""

    def __init__(self, value_type, value, num_element, channel_dim):
        self.value_type = value_type
        self.value = value
        self.num_element = num_element
        self.channel_dim = channel_dim

    def eval(self, device, **_kwargs):
        if self.value_type not in ("uniform", "constant"):
            raise ValueError(
                f"unknown value_type {self.value_type!r}; expected 'uniform' or 'constant'"
            )
        key = (
            self.value_type,
            repr(self.value),
            int(self.num_element),
            int(self.channel_dim),
            str(device),
        )
        cached = _VALUE_TENSOR_CACHE.get(key)
        if cached is not None:
            return cached
        if self.value_type == "uniform":
            tensor = torch.tensor(self.value, dtype=torch.float32, device=device).view(
                -1, self.channel_dim
            ).repeat(int(self.num_element), 1)
        else:
            tensor = torch.tensor(self.value, dtype=torch.float32, device=device)
        if len(_VALUE_TENSOR_CACHE) >= _VALUE_TENSOR_CACHE_MAX:
            _VALUE_TENSOR_CACHE.pop(next(iter(_VALUE_TENSOR_CACHE)))
        _VALUE_TENSOR_CACHE[key] = tensor
        return tensor
