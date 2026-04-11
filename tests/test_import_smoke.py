from __future__ import annotations

import importlib
import sys
import types

import pytest


def _snapshot_reconstruction_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if (
            name == "Reconstruction"
            or name.startswith("Reconstruction.")
            or name == "reconstruction"
            or name.startswith("reconstruction.")
        )
    }


def _clear_reconstruction_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "Reconstruction"
            or name.startswith("Reconstruction.")
            or name == "reconstruction"
            or name.startswith("reconstruction.")
        ):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _restore_reconstruction_modules_after_test():
    snapshot = _snapshot_reconstruction_modules()
    try:
        yield
    finally:
        _clear_reconstruction_modules()
        sys.modules.update(snapshot)


class TestImportSmoke:

    def test_import_uppercase_package_does_not_eagerly_import_solver_modules(self):
        _clear_reconstruction_modules()

        pkg = importlib.import_module("Reconstruction")

        assert pkg.__name__ == "Reconstruction"
        assert "Reconstruction.wiener" not in sys.modules
        assert "reconstruction.Reconstruction.wiener" not in sys.modules
        assert "Reconstruction.rl_unknown_boundary" not in sys.modules
        assert "Reconstruction.pnp_admm" not in sys.modules

    def test_import_lowercase_facade_smoke(self):
        _clear_reconstruction_modules()

        module = importlib.import_module("reconstruction")

        assert module.__name__ == "reconstruction"

    def test_import_uppercase_backend_smoke(self):
        _clear_reconstruction_modules()

        upper_backend = importlib.import_module("Reconstruction._backend")
        lower_backend = importlib.import_module("reconstruction.Reconstruction._backend")

        assert upper_backend is lower_backend

    def test_import_uppercase_base_aliases_lowercase_module(self, monkeypatch):
        _clear_reconstruction_modules()

        fake_base = types.ModuleType("reconstruction.Reconstruction._base")
        fake_base.DeconvBase = type("DeconvBase", (), {})
        monkeypatch.setitem(sys.modules, "reconstruction.Reconstruction._base", fake_base)

        upper_base = importlib.import_module("Reconstruction._base")

        assert upper_base is fake_base
        assert upper_base.DeconvBase is fake_base.DeconvBase

    def test_uppercase_and_lowercase_exports_share_implementation_objects(self):
        _clear_reconstruction_modules()

        upper = importlib.import_module("Reconstruction")
        lower = importlib.import_module("reconstruction")

        assert upper.set_backend is lower.set_backend
        assert upper.WienerDeconv is lower.WienerDeconv

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
            if name == ".pnp_admm" and package in {
                "Reconstruction",
                "reconstruction.Reconstruction",
            }:
                raise ModuleNotFoundError("No module named 'bm3d'", name="bm3d")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        with pytest.raises(ImportError, match="requires the optional 'bm3d' dependency"):
            getattr(pkg, "PnPADMM")
