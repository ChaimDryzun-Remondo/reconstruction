from __future__ import annotations

import builtins
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


def _snapshot_common_namespace_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if (
            name == "RemondoPythonCore"
            or name.startswith("RemondoPythonCore.")
            or name == "Shared"
            or name.startswith("Shared.")
        )
    }


def _clear_common_namespace_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "RemondoPythonCore"
            or name.startswith("RemondoPythonCore.")
            or name == "Shared"
            or name.startswith("Shared.")
        ):
            del sys.modules[name]


def _install_common_namespace(root: str, *, odd_crop_in_general_utils: bool, label: str) -> dict[str, object]:
    root_mod = types.ModuleType(root)
    root_mod.__path__ = []
    common_mod = types.ModuleType(f"{root}.Common")
    common_mod.__path__ = []
    gen_mod = types.ModuleType(f"{root}.Common.General_Utilities")
    psf_mod = types.ModuleType(f"{root}.Common.PSF_Preprocessing")
    img_mod = types.ModuleType(f"{root}.Common.Image_Preprocessing")

    def _sentinel(name: str):
        def _fn(*args, **kwargs):
            return (label, name, args, kwargs)
        _fn.__name__ = f"{label}_{name}"
        return _fn

    sentinels = {
        "padding": _sentinel("padding"),
        "cropping": _sentinel("cropping"),
        "odd_crop_around_center": _sentinel("odd_crop_around_center"),
        "psf_preprocess": _sentinel("psf_preprocess"),
        "condition_psf": _sentinel("condition_psf"),
        "image_normalization": _sentinel("image_normalization"),
        "validate_image": _sentinel("validate_image"),
        "to_grayscale": _sentinel("to_grayscale"),
    }

    gen_mod.padding = sentinels["padding"]
    gen_mod.cropping = sentinels["cropping"]
    if odd_crop_in_general_utils:
        gen_mod.odd_crop_around_center = sentinels["odd_crop_around_center"]

    psf_mod.psf_preprocess = sentinels["psf_preprocess"]
    psf_mod.condition_psf = sentinels["condition_psf"]

    img_mod.image_normalization = sentinels["image_normalization"]
    img_mod.validate_image = sentinels["validate_image"]
    img_mod.to_grayscale = sentinels["to_grayscale"]
    if not odd_crop_in_general_utils:
        img_mod.odd_crop_around_center = sentinels["odd_crop_around_center"]

    sys.modules[root] = root_mod
    sys.modules[f"{root}.Common"] = common_mod
    sys.modules[f"{root}.Common.General_Utilities"] = gen_mod
    sys.modules[f"{root}.Common.PSF_Preprocessing"] = psf_mod
    sys.modules[f"{root}.Common.Image_Preprocessing"] = img_mod
    return sentinels


def _import_fresh_common_module():
    _clear_reconstruction_modules()
    return importlib.import_module("Reconstruction._common")


@pytest.fixture(autouse=True)
def _restore_reconstruction_modules_after_test():
    snapshot = _snapshot_reconstruction_modules()
    try:
        yield
    finally:
        _clear_reconstruction_modules()
        sys.modules.update(snapshot)


@pytest.fixture(autouse=True)
def _restore_common_namespace_modules_after_test():
    snapshot = _snapshot_common_namespace_modules()
    try:
        yield
    finally:
        _clear_common_namespace_modules()
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


class TestCommonImportContract:

    def test_common_prefers_remondo_namespace_when_available(self):
        _clear_common_namespace_modules()
        remondo = _install_common_namespace(
            "RemondoPythonCore",
            odd_crop_in_general_utils=True,
            label="remondo",
        )
        shared = _install_common_namespace(
            "Shared",
            odd_crop_in_general_utils=False,
            label="shared",
        )

        common = _import_fresh_common_module()

        assert common.padding is remondo["padding"]
        assert common.cropping is remondo["cropping"]
        assert common.odd_crop_around_center is remondo["odd_crop_around_center"]
        assert common.psf_preprocess is remondo["psf_preprocess"]
        assert common.condition_psf is remondo["condition_psf"]
        assert common.image_normalization is remondo["image_normalization"]
        assert common.validate_image is remondo["validate_image"]
        assert common.to_grayscale is remondo["to_grayscale"]
        assert common.padding is not shared["padding"]

    @pytest.mark.xfail(reason="Sprint 5 Q2 — fallback mechanism pending removal; see WORKPLAN.md §8 Q2")
    def test_common_falls_back_to_shared_namespace_when_remondo_absent(self):
        _clear_common_namespace_modules()
        shared = _install_common_namespace(
            "Shared",
            odd_crop_in_general_utils=False,
            label="shared",
        )

        common = _import_fresh_common_module()

        assert common.padding is shared["padding"]
        assert common.cropping is shared["cropping"]
        assert common.odd_crop_around_center is shared["odd_crop_around_center"]
        assert common.psf_preprocess is shared["psf_preprocess"]
        assert common.condition_psf is shared["condition_psf"]
        assert common.image_normalization is shared["image_normalization"]
        assert common.validate_image is shared["validate_image"]
        assert common.to_grayscale is shared["to_grayscale"]

    @pytest.mark.xfail(reason="Sprint 5 Q2 — fallback mechanism pending removal; see WORKPLAN.md §8 Q2")
    def test_common_missing_both_namespaces_raises_clear_error(self):
        _clear_common_namespace_modules()

        with pytest.raises(ImportError, match="shared preprocessing utilities"):
            _import_fresh_common_module()

    @pytest.mark.xfail(reason="Sprint 5 Q2 — fallback mechanism pending removal; see WORKPLAN.md §8 Q2")
    def test_root_solver_symbol_requires_shared_preprocessing_namespace(self):
        _clear_reconstruction_modules()
        _clear_common_namespace_modules()

        pkg = importlib.import_module("Reconstruction")

        with pytest.raises(ImportError, match="shared preprocessing utilities"):
            getattr(pkg, "WienerDeconv")

    def test_common_nested_import_failure_in_preferred_namespace_is_not_mislabeled(
        self,
        monkeypatch,
    ):
        _clear_common_namespace_modules()
        _install_common_namespace(
            "Shared",
            odd_crop_in_general_utils=False,
            label="shared",
        )

        original_import = builtins.__import__

        def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "RemondoPythonCore.Common.General_Utilities":
                raise ModuleNotFoundError(
                    "No module named 'dependency_x'",
                    name="dependency_x",
                )
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _failing_import)

        with pytest.raises(ImportError, match="dependency_x"):
            _import_fresh_common_module()

    def test_root_symbol_import_preserves_nested_common_import_error(self, monkeypatch):
        _clear_reconstruction_modules()
        _clear_common_namespace_modules()

        original_import = builtins.__import__

        def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "RemondoPythonCore.Common.General_Utilities":
                raise ModuleNotFoundError(
                    "No module named 'dependency_x'",
                    name="dependency_x",
                )
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _failing_import)

        pkg = importlib.import_module("Reconstruction")

        with pytest.raises(ImportError, match="dependency_x"):
            getattr(pkg, "WienerDeconv")
