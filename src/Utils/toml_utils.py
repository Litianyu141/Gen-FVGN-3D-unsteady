"""Convert tomlkit objects to native Python types and apply CLI ``section.key=value`` overrides."""

import tomllib
from typing import Any, Dict, List


def toml_to_native(obj: Any) -> Any:
    if obj is None:
        return None
    obj_type = type(obj).__name__
    if obj_type == "Bool":
        return bool(obj)
    if obj_type == "Integer":
        return int(obj)
    if obj_type == "Float":
        return float(obj)
    if obj_type in ("String", "Key", "Date", "DateTime", "Time"):
        return str(obj)
    if isinstance(obj, dict):
        return {toml_to_native(k): toml_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [toml_to_native(v) for v in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return obj


def load_toml_native(file_path: str) -> Dict[str, Any]:
    import tomlkit

    with open(file_path, "r", encoding="utf-8") as f:
        return toml_to_native(tomlkit.load(f))


def parse_toml_native(toml_string: str) -> Dict[str, Any]:
    import tomlkit

    return toml_to_native(tomlkit.parse(toml_string))


def parse_override(item: str):
    """Parse ``"section.key=value"`` into ``(["section", "key"], value)``."""
    if "=" not in item:
        raise ValueError(f"Override {item!r}: expected section.key=value")
    path, raw = item.split("=", 1)
    keys = [k for k in path.strip().split(".") if k]
    if len(keys) < 2:
        raise ValueError(f"Override {item!r}: key needs its section, e.g. fvSchemes.ddtSchemes=Euler")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return keys, value


def apply_overrides(config: Dict[str, Any], items) -> List[str]:
    """Apply ``section.key=value`` overrides into ``config`` in place."""
    applied = []
    for item in items or ():
        keys, value = parse_override(item)
        node = config
        for k in keys[:-1]:
            if k not in node or not hasattr(node[k], "get"):
                node[k] = {}
            node = node[k]
        is_new = keys[-1] not in node
        node[keys[-1]] = value
        applied.append(f"{'.'.join(keys)} = {value!r}" + ("  (new key)" if is_new else ""))
    return applied


def add_override_argument(parser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override an fvconfig.toml entry (repeatable, e.g. fvSchemes.faceCvBuoyancy=face_sngrad)",
    )
