"""Module-load smoke test for ``examples/example.py``.

Sprint 4 commit T3.3 rewrote the demo to use Common's ``PSNR`` and
``SSIM`` (replacing the local Wang-2004 simplified implementations) and
preserved the local ``blur_image`` and ``add_awgn`` helpers per the
T3.3 verification's three divergence dispositions.  This file adds a
single Level-1 smoke test that catches import-time and syntax-level
regressions without running the demo's __main__ block.

Parallel structure to ``test_example_flow_smoke.py``: same
bootstrap-via-shared-helper pattern, same monorepo marker, same
non-coverage of pedagogical content.

Marked ``@pytest.mark.monorepo`` because the demo imports from
``RemondoPythonCore.Common.*`` (for ``Image_Quality_Measures``,
``Image_Preprocessing``, ``General_Utilities``) and from
``RemondoPythonCore.external_reconstruction``; both namespaces are
only available when the parent monorepo environment is active.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ._remondopythoncore_bootstrap import bootstrap_remondopythoncore


@pytest.mark.monorepo
def test_example_module_loads_cleanly() -> None:
    """Load ``examples/example.py`` as a module without running its
    __main__ block.  Verifies the demo's helpers, dataclasses, and
    per-algorithm runner functions are present.
    """
    bootstrap_remondopythoncore()

    path = Path(__file__).parent.parent / "examples" / "example.py"
    assert path.exists(), f"example.py not found at {path}"

    spec = importlib.util.spec_from_file_location("example_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec_module: Python 3.13's dataclass
    # decorator does sys.modules.get(cls.__module__).__dict__ during type
    # resolution; an unregistered module yields None and crashes.
    sys.modules["example_smoke"] = module
    try:
        spec.loader.exec_module(module)  # __name__ != "__main__", main block skipped
    finally:
        sys.modules.pop("example_smoke", None)

    # Local helpers preserved per T3.3's three divergence dispositions.
    assert hasattr(module, "blur_image"), "blur_image helper missing (T1.2 deferral)"
    assert hasattr(module, "add_awgn"), "add_awgn helper missing (pedagogical)"
    assert hasattr(module, "airy_psf"), "airy_psf helper missing"
    assert hasattr(module, "Config"), "Config dataclass missing"
    assert hasattr(module, "Result"), "Result dataclass missing"
    assert hasattr(module, "main"), "main function missing"
    assert hasattr(module, "display_results"), "display_results missing"

    # Per-algorithm runner functions for the always-available algorithms.
    for runner in (
        "run_wiener",
        "run_rl_unknown_boundary",
        "run_rl_standard",
        "run_landweber",
        "run_admm",
        "run_tval3",
        "run_fista_tv",
        "run_chambolle_pock",
        "run_pnp_admm",  # gracefully no-op when bm3d is unavailable
        "run_red_admm",  # gracefully no-op when bm3d is unavailable
    ):
        assert hasattr(module, runner), f"missing per-algorithm runner: {runner}"
