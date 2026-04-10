from __future__ import annotations

import importlib
import sys

import pytest


def _clear_reconstruction_modules() -> None:
    for name in list(sys.modules):
        if name == "Reconstruction" or name.startswith("Reconstruction."):
            del sys.modules[name]


class TestImportSmoke:

    def test_import_reconstruction_does_not_eagerly_import_solver_modules(self):
        _clear_reconstruction_modules()

        pkg = importlib.import_module("Reconstruction")

        assert pkg.__name__ == "Reconstruction"
        assert "Reconstruction.wiener" not in sys.modules
        assert "Reconstruction.rl_unknown_boundary" not in sys.modules
        assert "Reconstruction.pnp_admm" not in sys.modules

    def test_import_lowercase_facade_smoke(self):
        module = importlib.import_module("reconstruction")
        assert module.__name__ == "reconstruction"

    def test_wiener_symbol_import_does_not_require_scikit_image(self, monkeypatch):
        _clear_reconstruction_modules()
        monkeypatch.setitem(sys.modules, "skimage", None)
        monkeypatch.setitem(sys.modules, "skimage.restoration", None)

        pkg = importlib.import_module("Reconstruction")
        symbol = getattr(pkg, "WienerDeconv")

        assert symbol.__name__ == "WienerDeconv"

    def test_wiener_auto_sigma_missing_scikit_image_raises_clear_error(
        self,
        monkeypatch,
        test_image,
        gaussian_psf,
    ):
        _clear_reconstruction_modules()
        monkeypatch.setitem(sys.modules, "skimage", None)
        monkeypatch.setitem(sys.modules, "skimage.restoration", None)

        pkg = importlib.import_module("Reconstruction")
        solver = pkg.WienerDeconv(test_image, gaussian_psf)

        with pytest.raises(ImportError, match="scikit-image"):
            solver.deblur()

    def test_optional_bm3d_symbol_raises_clear_error(self, monkeypatch):
        _clear_reconstruction_modules()
        pkg = importlib.import_module("Reconstruction")

        original_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == ".pnp_admm" and package == "Reconstruction":
                raise ModuleNotFoundError("No module named 'bm3d'", name="bm3d")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        with pytest.raises(ImportError, match="requires the optional 'bm3d' dependency"):
            getattr(pkg, "PnPADMM")
