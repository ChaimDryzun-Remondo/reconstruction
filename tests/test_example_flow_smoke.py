"""Module-load smoke test for ``examples/example_flow.py``.

Sprint 4 commit T3.2 rewrote the demo to use synthetic input, a
declarative algorithm spec list, and the corrected
``RemondoPythonCore.external_reconstruction`` import path.  This file
adds a single Level-1 smoke test that catches import-time and
syntax-level regressions without running the demo's __main__ block.

The test does **not** verify pedagogical clarity, structural
readability, or the correctness of the algorithm spec values — those
properties are not testable automatically and require human review.
What this test catches:

  - import errors (e.g. a future regression to a deprecated namespace
    path, paralleling the lowercase ``RemondoPythonCore.reconstruction``
    that broke pre-T3.2);
  - syntax errors;
  - missing names that the demo's structure relies on
    (``image_quality_metrics``, ``_ALGO_SPECS``, ``_run_one_algorithm``);
  - the algorithm spec list shrinking below the nine non-BM3D entries.

A more substantive end-to-end test (Level 2 in T3.2's verification)
was deliberately not added — the demo's purpose is illustration, not
production code, and the structural change required to support an
end-to-end test (extracting __main__ into a callable ``main()``
function) would have shifted what readers encounter when they open the
file.  See `docs/refactoring-audit/NOTES.md` for the disposition.

Marked ``@pytest.mark.monorepo`` because the demo imports from
``RemondoPythonCore.Common.*`` (for ``Image_Quality_Measures``,
``Image_Preprocessing``, ``General_Utilities``) and from
``RemondoPythonCore.external_reconstruction``; both namespaces are only
available when the parent monorepo environment is active.  The default
submodule core-profile test invocation deselects this test; it runs
when the parent monorepo's pytest invocation is used.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

# example_flow.py imports from ``RemondoPythonCore.Common.*`` and
# ``RemondoPythonCore.external_reconstruction.*``.  Both namespaces
# resolve when ``c:/git/`` (the parent of ``RemondoPythonCore/``) is on
# ``sys.path``.  The parent monorepo's pyproject sets
# ``pythonpath = [".."]`` for its own pytest invocations; the submodule's
# pyproject does not, so when this test is collected via the submodule's
# pytest we add the parent-of-RemondoPythonCore explicitly.
#
# Order matters: the submodule's conftest.py installs ``Shared.Common.*``
# mocks (per Sprint 0 W6) which transitively cause ``RemondoPythonCore``
# to be cached in ``sys.modules`` as a namespace package with an empty
# ``__path__``.  We therefore (a) prepend the parent-of-RemondoPythonCore
# to ``sys.path``, (b) drop any stale ``RemondoPythonCore.*`` entries
# from ``sys.modules`` so the namespace package's ``__path__`` is rebuilt
# on next import, and (c) invalidate finder caches.
_REMONDOPYTHONCORE_PARENT = Path(__file__).resolve().parents[3]
if str(_REMONDOPYTHONCORE_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMONDOPYTHONCORE_PARENT))
for _stale in [k for k in list(sys.modules) if k == "RemondoPythonCore"
               or k.startswith("RemondoPythonCore.")]:
    del sys.modules[_stale]
importlib.invalidate_caches()


@pytest.mark.monorepo
def test_example_flow_module_loads_cleanly() -> None:
    """Load ``examples/example_flow.py`` as a module without running its
    __main__ block.  Verifies the module's public surface contains the
    names downstream tooling relies on, plus the expected algorithm
    spec count.
    """
    path = Path(__file__).parent.parent / "examples" / "example_flow.py"
    assert path.exists(), f"example_flow.py not found at {path}"

    spec = importlib.util.spec_from_file_location("example_flow_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__", main block skipped

    assert hasattr(module, "image_quality_metrics"), (
        "image_quality_metrics helper missing"
    )
    assert hasattr(module, "normalize_image"), "normalize_image helper missing"
    assert hasattr(module, "airy_psf"), "airy_psf helper missing"
    assert hasattr(module, "_AlgoSpec"), "_AlgoSpec dataclass missing"
    assert hasattr(module, "_run_one_algorithm"), "_run_one_algorithm executor missing"
    assert hasattr(module, "_ALGO_SPECS"), "_ALGO_SPECS list missing"

    # Nine non-BM3D specs always; eleven if bm3d is installed.
    assert len(module._ALGO_SPECS) >= 9, (
        f"_ALGO_SPECS shrank to {len(module._ALGO_SPECS)} entries; "
        "expected at least 9 (the non-BM3D minimum)"
    )
    assert len(module._ALGO_SPECS) <= 11, (
        f"_ALGO_SPECS grew to {len(module._ALGO_SPECS)} entries; "
        "expected at most 11 (the BM3D-included maximum)"
    )
