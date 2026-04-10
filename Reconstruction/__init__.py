"""
Reconstruction — Modular deconvolution algorithms for satellite imagery.

The package exposes a flat public API at the package root while keeping module
imports lazy, so ``import Reconstruction`` succeeds even when optional
dependencies are not installed.
"""
from __future__ import annotations

import importlib
import importlib.util

__version__: str = "0.1.0"

_HAS_PNP: bool = importlib.util.find_spec("bm3d") is not None
_HAS_RED: bool = _HAS_PNP

_EXPORTS: dict[str, tuple[str, str]] = {
    "set_backend": ("._backend", "set_backend"),
    "WienerDeconv": (".wiener", "WienerDeconv"),
    "wiener_deblur": (".wiener", "wiener_deblur"),
    "RLUnknownBoundary": (".rl_unknown_boundary", "RLUnknownBoundary"),
    "rl_deblur_unknown_boundary": (
        ".rl_unknown_boundary",
        "rl_deblur_unknown_boundary",
    ),
    "LandweberUnknownBoundary": (
        ".landweber_unknown_boundary",
        "LandweberUnknownBoundary",
    ),
    "landweber_deblur_unknown_boundary": (
        ".landweber_unknown_boundary",
        "landweber_deblur_unknown_boundary",
    ),
    "ADMMDeconv": (".admm", "ADMMDeconv"),
    "admm_deblur": (".admm", "admm_deblur"),
    "TVAL3Deconv": (".tval3", "TVAL3Deconv"),
    "tval3_deblur": (".tval3", "tval3_deblur"),
    "FISTADeconv": (".fista", "FISTADeconv"),
    "fista_deblur": (".fista", "fista_deblur"),
    "ChambollePockDeconv": (".chambolle_pock", "ChambollePockDeconv"),
    "chambolle_pock_deblur": (".chambolle_pock", "chambolle_pock_deblur"),
    "PnPADMM": (".pnp_admm", "PnPADMM"),
    "pnp_admm_deblur": (".pnp_admm", "pnp_admm_deblur"),
    "REDDeconv": (".red_admm", "REDDeconv"),
    "red_deblur": (".red_admm", "red_deblur"),
}

__all__ = [
    "set_backend",
    "WienerDeconv",
    "RLUnknownBoundary",
    "LandweberUnknownBoundary",
    "ADMMDeconv",
    "TVAL3Deconv",
    "FISTADeconv",
    "ChambollePockDeconv",
    "wiener_deblur",
    "rl_deblur_unknown_boundary",
    "landweber_deblur_unknown_boundary",
    "admm_deblur",
    "tval3_deblur",
    "fista_deblur",
    "chambolle_pock_deblur",
    "__version__",
]

if _HAS_PNP:
    __all__ += ["PnPADMM", "pnp_admm_deblur"]

if _HAS_RED:
    __all__ += ["REDDeconv", "red_deblur"]


def _rewrite_import_error(symbol: str, exc: ImportError) -> ImportError:
    missing = getattr(exc, "name", "") or ""

    if (
        symbol in {"PnPADMM", "pnp_admm_deblur", "REDDeconv", "red_deblur"}
        and missing.startswith("bm3d")
    ):
        return ImportError(
            f"{symbol} requires the optional 'bm3d' dependency. Install with: "
            "pip install reconstruction[pnp] or pip install bm3d"
        )

    if missing.startswith("skimage"):
        return ImportError(
            "WienerDeconv automatic noise estimation requires the optional "
            "'scikit-image' dependency. Install with: "
            "pip install reconstruction[imaging] or pip install scikit-image"
        )

    if missing.startswith(("RemondoPythonCore", "Shared")):
        return ImportError(
            "Reconstruction solver modules require the shared preprocessing "
            "utilities from 'RemondoPythonCore.Common' (preferred) or "
            "'Shared.Common' (legacy). Install/use the full "
            "RemondoPythonCore package layout before importing solver modules."
        )

    return exc


def __getattr__(name: str):
    if name in {"__version__", "_HAS_PNP", "_HAS_RED"}:
        return globals()[name]

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as exc:
        raise _rewrite_import_error(name, exc) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__) | {"_HAS_PNP", "_HAS_RED"})
