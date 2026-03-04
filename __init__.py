"""
RemondoPythonCore.reconstruction — lazy namespace facade.

Windows note
------------
On Windows, Python's module finder is case-sensitive even on a
case-insensitive filesystem.  The directory stored as ``reconstruction``
(lowercase) is found when the import name is ``reconstruction`` but NOT
when the name is ``Reconstruction`` (capital R).

This facade re-exports the full public API of the inner ``Reconstruction``
package on first attribute access — lazily, so that the dev test suite's
``conftest.py`` can install mock dependencies before any Reconstruction
code is imported.

Usage in production code
------------------------
    from RemondoPythonCore.reconstruction import (
        WienerDeconv, wiener_deblur,
        RLUnknownBoundary, rl_deblur_unknown_boundary,
        # ... etc.
    )
"""
from __future__ import annotations

import importlib as _imp
import importlib.util as _imp_util  # ensure importlib.util is loaded before _backend.py

_INNER = None   # populated on first access


def _get_inner():
    global _INNER
    if _INNER is None:
        _INNER = _imp.import_module(".Reconstruction", __package__)
    return _INNER


def __getattr__(name: str):
    """Lazily delegate attribute lookup to the inner Reconstruction package."""
    inner = _get_inner()
    try:
        val = getattr(inner, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Cache in this module's globals to speed up subsequent accesses.
    globals()[name] = val
    return val


def __dir__():
    inner = _get_inner()
    own = list(globals().keys())
    return sorted(set(own) | set(getattr(inner, "__all__", [])))
