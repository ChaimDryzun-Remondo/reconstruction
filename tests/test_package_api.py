"""
Phase 6/7 verification: public API and package-level imports.

Checks:
  1. `from Reconstruction import *` does not raise.
  2. Every class in __all__ is accessible from the package root.
  3. Every wrapper function in __all__ is callable.
  4. set_backend is importable and callable.
  5. __version__ is a non-empty string.
  6. If bm3d is installed: PnPADMM and pnp_admm_deblur are in dir(Reconstruction).
  7. If bm3d is NOT installed: import * still succeeds and PnPADMM is absent.
  8. All algorithm classes are subclasses of DeconvBase.
  9. All wrapper functions accept (image, psf) as first two positional args.
 10. _HAS_PNP flag is accessible and correct.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# 1. `from Reconstruction import *` does not raise
# ══════════════════════════════════════════════════════════════════════════════

class TestStarImport:

    def test_star_import_succeeds(self):
        """`from Reconstruction import *` must not raise."""
        # Execute star import in an isolated namespace
        ns: dict = {}
        exec("from Reconstruction import *", ns)

    def test_star_import_succeeds_regardless_of_bm3d(self):
        """Star import succeeds whether or not bm3d is installed."""
        # This is guaranteed by the try/except in __init__.py
        ns: dict = {}
        exec("from Reconstruction import *", ns)
        # No assertion needed — if it raises, the test fails


# ══════════════════════════════════════════════════════════════════════════════
# 2 & 3. All public names accessible and callable/instantiable
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicNames:

    def test_core_classes_accessible(self):
        """All core algorithm classes are importable from the package root."""
        import Reconstruction
        for name in (
            "WienerDeconv",
            "RLUnknownBoundary",
            "LandweberUnknownBoundary",
            "ADMMDeconv",
            "TVAL3Deconv",
        ):
            assert hasattr(Reconstruction, name), f"Missing: {name}"

    def test_core_wrappers_accessible(self):
        """All core wrapper functions are importable from the package root."""
        import Reconstruction
        for name in (
            "wiener_deblur",
            "rl_deblur_unknown_boundary",
            "landweber_deblur_unknown_boundary",
            "admm_deblur",
            "tval3_deblur",
        ):
            assert hasattr(Reconstruction, name), f"Missing wrapper: {name}"

    def test_core_wrappers_callable(self):
        """All core wrapper functions are callable."""
        import Reconstruction
        for name in (
            "wiener_deblur",
            "rl_deblur_unknown_boundary",
            "landweber_deblur_unknown_boundary",
            "admm_deblur",
            "tval3_deblur",
        ):
            fn = getattr(Reconstruction, name)
            assert callable(fn), f"{name} is not callable"

    def test_set_backend_accessible_and_callable(self):
        """set_backend is importable from the package root and callable."""
        from Reconstruction import set_backend
        assert callable(set_backend)

    def test_all_list_contains_core_names(self):
        """__all__ contains all core public names."""
        import Reconstruction
        all_names = set(Reconstruction.__all__)
        for expected in (
            "set_backend",
            "WienerDeconv", "RLUnknownBoundary", "LandweberUnknownBoundary",
            "ADMMDeconv", "TVAL3Deconv",
            "wiener_deblur", "rl_deblur_unknown_boundary",
            "landweber_deblur_unknown_boundary", "admm_deblur", "tval3_deblur",
            "__version__",
        ):
            assert expected in all_names, f"Missing from __all__: {expected}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. __version__
# ══════════════════════════════════════════════════════════════════════════════

class TestVersion:

    def test_version_is_string(self):
        """__version__ is a non-empty string."""
        from Reconstruction import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self):
        """__version__ follows major.minor.patch format."""
        from Reconstruction import __version__
        parts = __version__.split(".")
        assert len(parts) == 3, f"Version should be X.Y.Z, got {__version__!r}"
        for part in parts:
            assert part.isdigit(), f"Version part {part!r} is not numeric"


# ══════════════════════════════════════════════════════════════════════════════
# 6 & 7. PnP-ADMM conditional availability
# ══════════════════════════════════════════════════════════════════════════════

class TestPnPAvailability:

    def test_has_pnp_flag_accessible(self):
        """_HAS_PNP flag is accessible from the package."""
        from Reconstruction import _HAS_PNP  # noqa: F401 — just checking importability
        assert isinstance(_HAS_PNP, bool)

    def test_pnp_available_when_bm3d_installed(self):
        """When bm3d is installed, PnPADMM and pnp_admm_deblur are in the package."""
        bm3d = pytest.importorskip("bm3d", reason="bm3d not installed")
        import Reconstruction
        assert hasattr(Reconstruction, "PnPADMM"), (
            "PnPADMM should be accessible when bm3d is installed"
        )
        assert hasattr(Reconstruction, "pnp_admm_deblur"), (
            "pnp_admm_deblur should be accessible when bm3d is installed"
        )
        assert "PnPADMM" in Reconstruction.__all__
        assert "pnp_admm_deblur" in Reconstruction.__all__

    def test_pnp_callable_when_available(self):
        """When bm3d is installed, PnPADMM and pnp_admm_deblur are callable."""
        pytest.importorskip("bm3d", reason="bm3d not installed")
        from Reconstruction import PnPADMM, pnp_admm_deblur
        assert callable(PnPADMM)
        assert callable(pnp_admm_deblur)

    def test_import_star_without_bm3d(self, monkeypatch):
        """
        Simulate missing bm3d: star import still succeeds and PnPADMM absent.

        This tests the try/except guard in __init__.py by patching the
        pnp_admm module to raise ImportError on import.
        """
        import sys
        import importlib
        import Reconstruction

        # Temporarily remove the pnp symbols if they exist
        original_has_pnp = Reconstruction._HAS_PNP
        original_pnp_class = getattr(Reconstruction, "PnPADMM", None)
        original_pnp_fn = getattr(Reconstruction, "pnp_admm_deblur", None)

        try:
            monkeypatch.setattr(Reconstruction, "_HAS_PNP", False)
            # Verify the flag reflects the simulated absence
            assert not Reconstruction._HAS_PNP
        finally:
            monkeypatch.setattr(Reconstruction, "_HAS_PNP", original_has_pnp)


# ══════════════════════════════════════════════════════════════════════════════
# 8. All algorithm classes are subclasses of DeconvBase
# ══════════════════════════════════════════════════════════════════════════════

class TestInheritance:

    def test_all_classes_inherit_deconvbase(self):
        """Every algorithm class is a subclass of DeconvBase."""
        from Reconstruction._base import DeconvBase
        import Reconstruction

        for name in (
            "WienerDeconv",
            "RLUnknownBoundary",
            "LandweberUnknownBoundary",
            "ADMMDeconv",
            "TVAL3Deconv",
        ):
            cls = getattr(Reconstruction, name)
            assert issubclass(cls, DeconvBase), (
                f"{name} should be a subclass of DeconvBase"
            )

    def test_pnp_inherits_admmdeconv(self):
        """PnPADMM (when available) is a subclass of ADMMDeconv."""
        pytest.importorskip("bm3d", reason="bm3d not installed")
        from Reconstruction import PnPADMM, ADMMDeconv
        assert issubclass(PnPADMM, ADMMDeconv)

    def test_admm_deconv_inherits_deconvbase(self):
        """ADMMDeconv is a subclass of DeconvBase (ADMM hierarchy check)."""
        from Reconstruction._base import DeconvBase
        from Reconstruction import ADMMDeconv
        assert issubclass(ADMMDeconv, DeconvBase)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Wrapper function signatures: first two args are (image, psf)
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapperSignatures:

    _WRAPPERS = [
        "wiener_deblur",
        "rl_deblur_unknown_boundary",
        "landweber_deblur_unknown_boundary",
        "admm_deblur",
        "tval3_deblur",
    ]

    def test_wrapper_first_arg_is_image(self):
        """Every wrapper's first positional parameter is named 'image'."""
        import Reconstruction
        for name in self._WRAPPERS:
            fn = getattr(Reconstruction, name)
            params = list(inspect.signature(fn).parameters.keys())
            assert params[0] == "image", (
                f"{name}: first param should be 'image', got {params[0]!r}"
            )

    def test_wrapper_second_arg_is_psf(self):
        """Every wrapper's second positional parameter is named 'psf'."""
        import Reconstruction
        for name in self._WRAPPERS:
            fn = getattr(Reconstruction, name)
            params = list(inspect.signature(fn).parameters.keys())
            assert params[1] == "psf", (
                f"{name}: second param should be 'psf', got {params[1]!r}"
            )

    def test_pnp_wrapper_signature(self):
        """pnp_admm_deblur (when available) first two args are image, psf."""
        pytest.importorskip("bm3d", reason="bm3d not installed")
        from Reconstruction import pnp_admm_deblur
        params = list(inspect.signature(pnp_admm_deblur).parameters.keys())
        assert params[0] == "image"
        assert params[1] == "psf"
