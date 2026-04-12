from __future__ import annotations

import importlib
import sys

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

    def test_reconstruction_package_has_version(self):
        _clear_reconstruction_modules()

        pkg = importlib.import_module("Reconstruction")

        assert isinstance(pkg.__version__, str)
        assert pkg.__version__  # non-empty

    def test_import_uppercase_backend_smoke(self):
        _clear_reconstruction_modules()

        backend = importlib.import_module("Reconstruction._backend")

        assert callable(getattr(backend, "set_backend", None))

    def test_reconstruction_base_importable(self):
        _clear_reconstruction_modules()

        base = importlib.import_module("Reconstruction._base")

        assert hasattr(base, "DeconvBase")

    def test_reconstruction_exports_are_accessible(self):
        _clear_reconstruction_modules()

        pkg = importlib.import_module("Reconstruction")

        # First access triggers __getattr__ (set_backend is lazy via _EXPORTS,
        # not bound at Reconstruction/__init__.py import time).
        first_ref = pkg.set_backend
        # Second access returns the cached value from globals(), not __getattr__.
        assert pkg.set_backend is first_ref
        assert callable(first_ref)
        assert callable(pkg.WienerDeconv)

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
