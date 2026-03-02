"""
Phase 4b verification tests for Reconstruction.landweber_unknown_boundary.

Checks:
  1. LandweberUnknownBoundary: init succeeds, correct state attributes.
  2. deblur(): output shape matches original image, dtype is numpy float.
  3. deblur(): all output values are non-negative.
  4. FISTA: t_k update formula is strictly increasing; more iterations
     produce a different result (FISTA accumulates momentum).
  5. Preconditioned vs unpreconditioned: produce different results.
  6. Wrapper: landweber_deblur_unknown_boundary splits kwargs correctly.
  7. Regression: output matches docs/reference/Landweber_Unknown_Boundary.py
     within atol=1e-5.  Both new and reference are constructed with
     htm_floor_frac=0.0 for identical HTM values.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import Reconstruction._backend as backend
from Reconstruction.landweber_unknown_boundary import (
    LandweberUnknownBoundary,
    landweber_deblur_unknown_boundary,
)
from Reconstruction._base import DeconvBase


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend before and after every test."""
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


@pytest.fixture
def lw(blurred_image, gaussian_psf):
    """LandweberUnknownBoundary instance on the blurred test image."""
    return LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberInit:
    """LandweberUnknownBoundary inherits all init logic from DeconvBase."""

    def test_init_succeeds(self, blurred_image, gaussian_psf):
        """Constructor completes without error."""
        obj = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        assert obj is not None

    def test_image_shape_stored(self, lw, blurred_image):
        """self.h and self.w match the input image odd dimensions."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        assert lw.h == OH
        assert lw.w == OW

    def test_full_shape_larger_than_image(self, lw):
        """Padded canvas must be strictly larger than the image."""
        assert lw.full_shape[0] > lw.h
        assert lw.full_shape[1] > lw.w

    def test_full_shape_is_odd(self, lw):
        """Padded canvas dimensions must be odd."""
        assert lw.full_shape[0] % 2 == 1
        assert lw.full_shape[1] % 2 == 1

    def test_lipschitz_positive(self, lw):
        """Lipschitz constant must be strictly positive."""
        assert lw._lipschitz > 0.0

    def test_is_deconvbase_subclass(self):
        """LandweberUnknownBoundary is a subclass of DeconvBase."""
        assert issubclass(LandweberUnknownBoundary, DeconvBase)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Output shape and dtype
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberOutputShape:
    """deblur() must return a numpy array with the original image shape."""

    def test_output_shape_matches_image(self, lw, blurred_image):
        """Output shape matches the (possibly odd-cropped) input shape."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        result = lw.deblur(num_iter=3, lambda_tv=0.0)
        assert result.shape == (OH, OW)

    def test_output_is_numpy_array(self, lw):
        """deblur() must return a numpy.ndarray (CPU)."""
        result = lw.deblur(num_iter=3, lambda_tv=0.0)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_float(self, lw):
        """Output dtype is a float type."""
        result = lw.deblur(num_iter=3, lambda_tv=0.0)
        assert np.issubdtype(result.dtype, np.floating)

    def test_output_shape_single_iter(self, blurred_image, gaussian_psf):
        """Shape is correct even for num_iter=1."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        lw = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = lw.deblur(num_iter=1, lambda_tv=0.0)
        assert result.shape == (OH, OW)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Positivity
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberPositivity:
    """All output values must be non-negative."""

    def test_output_non_negative_no_tv(self, lw):
        """Without TV, output is non-negative."""
        result = lw.deblur(num_iter=10, lambda_tv=0.0)
        assert float(result.min()) >= 0.0

    def test_output_non_negative_with_tv(self, blurred_image, gaussian_psf):
        """With TV regularisation, output is also non-negative."""
        lw = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = lw.deblur(num_iter=10, lambda_tv=0.001)
        assert float(result.min()) >= 0.0

    def test_positivity_floor_applied(self, blurred_image, gaussian_psf):
        """Output values respect the positivity floor."""
        lw = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = lw.deblur(num_iter=5, lambda_tv=0.0,
                           enforce_positivity=True, epsilon_positivity=1e-8)
        assert float(result.min()) >= 0.0

    def test_positivity_disabled_can_be_negative(self, blurred_image, gaussian_psf):
        """enforce_positivity=False may allow negative values on raw data."""
        lw = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        # Just verify it doesn't crash and returns the correct shape.
        result = lw.deblur(num_iter=5, lambda_tv=0.0, enforce_positivity=False)
        assert result.shape == (lw.h, lw.w)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FISTA momentum
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberFISTA:
    """Verify FISTA momentum and adaptive restart behaviour."""

    def test_momentum_formula_yields_increasing_t(self):
        """
        The FISTA update formula t_new = (1 + √(1 + 4 t²)) / 2 is strictly
        increasing: t_{k+1} > t_k for all t_k ≥ 1.
        """
        t_k = 1.0
        for _ in range(20):
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
            assert t_new > t_k, (
                f"FISTA t not increasing: t_new={t_new:.6f} <= t_k={t_k:.6f}"
            )
            t_k = t_new

    def test_more_iterations_change_result(self, blurred_image, gaussian_psf):
        """More iterations with FISTA produce a different result than fewer."""
        lw1 = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        lw2 = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        r2  = lw1.deblur(num_iter=2,  lambda_tv=0.0, tol=0.0,
                         adaptive_restart=False)
        r15 = lw2.deblur(num_iter=15, lambda_tv=0.0, tol=0.0,
                         adaptive_restart=False)
        assert not np.allclose(r2, r15), (
            "Different iteration counts should produce different results"
        )

    def test_adaptive_restart_produces_valid_output(self, blurred_image, gaussian_psf):
        """adaptive_restart=True and False both produce valid numpy arrays."""
        lw_no  = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        lw_yes = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        rno  = lw_no.deblur(num_iter=20, lambda_tv=0.0, tol=0.0,
                            adaptive_restart=False)
        ryes = lw_yes.deblur(num_iter=20, lambda_tv=0.0, tol=0.0,
                             adaptive_restart=True)
        assert isinstance(rno,  np.ndarray)
        assert isinstance(ryes, np.ndarray)
        assert rno.shape  == (lw_no.h,  lw_no.w)
        assert ryes.shape == (lw_yes.h, lw_yes.w)

    def test_estimated_image_updated_after_deblur(self, lw):
        """self.estimated_image must be updated after calling deblur()."""
        init_est = np.array(lw.estimated_image).copy()
        lw.deblur(num_iter=10, lambda_tv=0.0)
        new_est = np.array(lw.estimated_image)
        assert not np.allclose(init_est, new_est), (
            "estimated_image should be updated by deblur()"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Preconditioned vs unpreconditioned
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberPreconditioned:
    """Preconditioned and unpreconditioned variants must differ."""

    def test_preconditioned_vs_unpreconditioned_differ(self, blurred_image,
                                                       gaussian_psf):
        """precondition=True and False produce different outputs."""
        lw_pre  = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        lw_nopre = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        r_pre   = lw_pre.deblur(  num_iter=20, lambda_tv=0.0, precondition=True)
        r_nopre = lw_nopre.deblur(num_iter=20, lambda_tv=0.0, precondition=False)
        assert not np.allclose(r_pre, r_nopre), (
            "Preconditioned and unpreconditioned variants should differ"
        )

    def test_both_variants_produce_valid_arrays(self, blurred_image, gaussian_psf):
        """Both variants return non-negative numpy arrays with correct shape."""
        for pre in (True, False):
            lw = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
            result = lw.deblur(num_iter=5, lambda_tv=0.0, precondition=pre)
            assert isinstance(result, np.ndarray)
            assert result.shape == (lw.h, lw.w)
            assert float(result.min()) >= 0.0

    def test_tv_changes_output(self, blurred_image, gaussian_psf):
        """lambda_tv > 0 produces a different result than lambda_tv = 0."""
        lw_no_tv = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        lw_tv    = LandweberUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        r_no_tv = lw_no_tv.deblur(num_iter=20, lambda_tv=0.0)
        r_tv    = lw_tv.deblur(   num_iter=20, lambda_tv=0.001)
        assert not np.allclose(r_no_tv, r_tv), (
            "TV regularisation should change the output"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Wrapper function
# ══════════════════════════════════════════════════════════════════════════════

class TestLandweberWrapper:
    """landweber_deblur_unknown_boundary() must split kwargs correctly."""

    def test_wrapper_returns_array(self, blurred_image, gaussian_psf):
        """Wrapper returns a numpy array."""
        result = landweber_deblur_unknown_boundary(
            blurred_image, gaussian_psf, num_iter=3, lambda_tv=0.0,
        )
        assert isinstance(result, np.ndarray)

    def test_wrapper_output_shape(self, blurred_image, gaussian_psf):
        """Wrapper output has the correct shape."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        result = landweber_deblur_unknown_boundary(
            blurred_image, gaussian_psf, num_iter=3, lambda_tv=0.0,
        )
        assert result.shape == (OH, OW)

    def test_wrapper_matches_direct_call(self, blurred_image, gaussian_psf):
        """Wrapper result equals direct LandweberUnknownBoundary(...).deblur(...)."""
        obj = LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, paddingMode="Reflect",
        )
        direct = obj.deblur(num_iter=8, lambda_tv=0.0)

        wrapper = landweber_deblur_unknown_boundary(
            blurred_image, gaussian_psf,
            paddingMode="Reflect",   # init kwarg
            num_iter=8,              # deblur kwarg
            lambda_tv=0.0,           # deblur kwarg
        )
        np.testing.assert_allclose(wrapper, direct, atol=1e-6)

    def test_wrapper_init_kwargs_split(self, blurred_image, gaussian_psf):
        """paddingMode (init kwarg) and num_iter (deblur kwarg) are split."""
        result = landweber_deblur_unknown_boundary(
            blurred_image, gaussian_psf,
            paddingMode="Edge",   # init kwarg
            num_iter=3,           # deblur kwarg
            lambda_tv=0.0,
        )
        assert result.shape[0] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. Regression against reference implementation
# ══════════════════════════════════════════════════════════════════════════════

_REF_PATH = (
    Path(__file__).parent.parent / "docs" / "reference"
    / "Landweber_Unknown_Boundary.py"
)


def _load_reference_module():
    """Load docs/reference/Landweber_Unknown_Boundary.py via importlib."""
    if not _REF_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "Landweber_Unknown_Boundary_ref", str(_REF_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _REF_PATH.exists(),
                    reason="Reference file not found")
class TestLandweberRegression:
    """
    Regression: new LandweberUnknownBoundary must match the reference
    within atol=1e-5 for 30 iterations.

    Both new and reference constructors accept htm_floor_frac; passing
    htm_floor_frac=0.0 to both ensures they compute identical HTM arrays
    (clipped only to the absolute floor 1e-12, with no percentage clamp).
    This makes the comparison as clean as possible.
    """

    @pytest.fixture(scope="class")
    def ref_mod(self):
        return _load_reference_module()

    def test_no_tv_30_iter(self, blurred_image, gaussian_psf, ref_mod):
        """No-TV case: 30 iterations, both implementations agree within 1e-5."""
        ref_lw = ref_mod.LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        ref_result = ref_lw.deblur(
            num_iter=30, lambda_tv=0.0, tol=0.0,
        )

        new_lw = LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_lw.deblur(
            num_iter=30, lambda_tv=0.0, tol=0.0,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg="No-TV regression failed: new != reference within atol=1e-5",
        )

    def test_with_tv_30_iter(self, blurred_image, gaussian_psf, ref_mod):
        """TV case: 30 iterations with lambda_tv=0.001, agree within 1e-5."""
        ref_lw = ref_mod.LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        ref_result = ref_lw.deblur(
            num_iter=30, lambda_tv=0.001, tol=0.0,
        )

        new_lw = LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_lw.deblur(
            num_iter=30, lambda_tv=0.001, tol=0.0,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg="TV regression failed: new != reference within atol=1e-5",
        )

    def test_precondition_false_30_iter(self, blurred_image, gaussian_psf,
                                        ref_mod):
        """precondition=False: 30 iterations, agree within 1e-5."""
        ref_lw = ref_mod.LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        ref_result = ref_lw.deblur(
            num_iter=30, lambda_tv=0.001, tol=0.0, precondition=False,
        )

        new_lw = LandweberUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_lw.deblur(
            num_iter=30, lambda_tv=0.001, tol=0.0, precondition=False,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg=(
                "precondition=False regression failed: "
                "new != reference within atol=1e-5"
            ),
        )
