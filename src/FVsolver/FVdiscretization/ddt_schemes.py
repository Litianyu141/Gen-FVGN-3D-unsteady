"""Parse OpenFOAM-style ``fvSchemes.ddtSchemes`` for PCNO and FIFO-ODIL."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CRANK_NICOLSON_COEFFICIENT = 0.5

_ALIASES = {
    "euler": "Euler",
    "backward": "backward",
    "bdf2": "backward",
    "cranknicolson": "CrankNicolson",
    "cn": "CrankNicolson",
    "steadystate": "steadyState",
    "steady": "steadyState",
}


@dataclass(frozen=True)
class DdtScheme:
    """Parsed ``ddtSchemes`` entry."""

    name: str
    off_centering: float = 0.0

    @property
    def is_crank_nicolson(self) -> bool:
        return self.name == "CrankNicolson"

    @property
    def is_backward(self) -> bool:
        return self.name == "backward"

    @property
    def is_steady_state(self) -> bool:
        return self.name == "steadyState"

    def __str__(self) -> str:
        if self.is_crank_nicolson:
            return f"CrankNicolson {self.off_centering:g}"
        return self.name


def parse_ddt_scheme(raw) -> DdtScheme:
    """Parse ``ddtSchemes`` (supports ``Euler``, ``CrankNicolson [c]``, ``backward``, ``steadyState``, ``implicit``, ``imex``)."""
    if isinstance(raw, DdtScheme):
        return raw
    text = str(raw).strip()
    if not text:
        raise ValueError("fvSchemes.ddtSchemes is empty; specify e.g. 'Euler', 'CrankNicolson 0.5', or 'backward'")

    tokens = text.replace(";", " ").split()
    key = tokens[0].lower().replace("_", "").replace("-", "")

    if key == "imex":
        return DdtScheme(name="CrankNicolson", off_centering=1.0)
    if key == "implicit":
        return DdtScheme(name="Euler")

    if key not in _ALIASES:
        raise ValueError(
            f"Unknown fvSchemes.ddtSchemes={text!r}; expected 'Euler', 'CrankNicolson [c]', 'backward', or 'steadyState'"
        )
    name = _ALIASES[key]
    if len(tokens) > 2:
        raise ValueError(f"fvSchemes.ddtSchemes={text!r} has too many tokens")

    if name != "CrankNicolson":
        if len(tokens) == 2:
            raise ValueError(f"fvSchemes.ddtSchemes={text!r}: only CrankNicolson takes a coefficient")
        return DdtScheme(name)

    if len(tokens) == 1:
        return DdtScheme(name, DEFAULT_CRANK_NICOLSON_COEFFICIENT)
    try:
        coeff = float(tokens[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CrankNicolson coefficient {tokens[1]!r}") from exc
    if not 0.0 <= coeff <= 1.0:
        raise ValueError(f"CrankNicolson coefficient must be in [0, 1], got {coeff}")
    return DdtScheme(name, coeff)


def ddt_scheme_from_fvconfig(fvconfig) -> DdtScheme:
    schemes = (fvconfig or {}).get("fvSchemes", {})
    if "ddtSchemes" not in schemes:
        raise KeyError("fvconfig.toml is missing [fvSchemes] ddtSchemes")
    return parse_ddt_scheme(schemes["ddtSchemes"])


def effective_old_state(*, phi_old, phi_new, previous_effective_old=None, scheme: DdtScheme):
    """Return effective old state so ``(phi_new - phi_old_eff) / dt`` realizes ``scheme``."""
    if scheme.is_steady_state:
        return phi_new
    if scheme.is_backward:
        raise ValueError("effective_old_state does not implement 'backward'; use TimeWindow for BDF2")
    if scheme.name == "Euler" or scheme.off_centering == 0.0 or previous_effective_old is None:
        return phi_old
    c = scheme.off_centering
    return phi_old - c * (phi_new - phi_old) + c * (phi_old - previous_effective_old)
