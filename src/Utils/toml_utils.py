"""Conversion of tomlkit documents to plain Python objects."""

from typing import Any


def toml_to_native(obj: Any) -> Any:
    """Recursively convert tomlkit items to Python types (TOML booleans become 0/1)."""
    obj_type = type(obj).__name__
    if obj_type in ("Integer", "Bool"):
        return int(obj)
    if obj_type == "Float":
        return float(obj)
    if obj_type in ("String", "Key"):
        return str(obj)
    if isinstance(obj, dict):
        return {toml_to_native(k): toml_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [toml_to_native(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    return obj
