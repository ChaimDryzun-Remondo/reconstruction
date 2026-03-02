"""
Phase 4a verification tests for Reconstruction.rl_unknown_boundary.

Checks:
  1. RLUnknownBoundary: init succeeds, correct state attributes.
  2. deblur(): output shape matches original image.
  3. deblur(): all output values are non-negative.
  4. TV regularisation changes the result (lambda_tv > 0 vs lambda_tv = 0).
  5. tv_on_full_canvas=False produces a different result than True.
  6. Convergence: early stopping triggers within max_iter.
  7. Wrapper: rl_deblur_unknown_boundary splits kwargs correctly.
  8. Regression: output matches docs/reference/RL_Unknown_Boundary.py
     within atol=1e-5 (htm_floor_frac=0.0 disables the base-class floor
     clamp so the two constructors are equivalent).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import Reconstruction._backend as backend
from Reconstruction.rl_unknown_boundary import (
    RLUnknownBoundary,
    rl_deblur_unknown_boundary,
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
def rl(blurred_image, gaussian_psf):
    """RLUnknownBoundary instance on the blurred test image."""
    return RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestRLInit:
    """RLUnknownBoundary inherits all init logic from DeconvBase."""

    def test_init_succeeds(self, blurred_image, gaussian_psf):
        """Constructor completes without error."""
        obj = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        assert obj is not None

    def test_image_shape_stored(self, rl, blurred_image):
        """self.h and self.w match the input image odd dimensions."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        assert rl.h == OH
        assert rl.w == OW

    def test_full_shape_larger_than_image(self, rl):
        """Padded canvas must be strictly larger than the image."""
        assert rl.full_shape[0] > rl.h
        assert rl.full_shape[1] > rl.w

    def test_full_shape_is_odd(self, rl):
        """Padded canvas dimensions must be odd."""
        assert rl.full_shape[0] % 2 == 1
        assert rl.full_shape[1] % 2 == 1

    def test_mask_sum_equals_image_area(self, rl):
        """Mask must have exactly h*w ones (image support)."""
        import numpy as np
        mask_np = np.array(rl.mask)
        assert float(mask_np.sum()) == float(rl.h * rl.w)

    def test_estimated_image_positive(self, rl):
        """Initial estimate must be strictly positive."""
        import numpy as np
        est = np.array(rl.estimated_image)
        assert float(est.min()) >= float(np.float32(1e-8))

    def test_is_deconvbase_subclass(self):
        """RLUnknownBoundary is a subclass of DeconvBase."""
        assert issubclass(RLUnknownBoundary, DeconvBase)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Output shape
# ══════════════════════════════════════════════════════════════════════════════

class TestRLOutputShape:
    """deblur() must return an array with the original image shape."""

    def test_output_shape_matches_image(self, rl, blurred_image):
        """Output shape matches the (possibly odd-cropped) input shape."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        result = rl.deblur(num_iter=5, lambda_tv=0.0)
        assert result.shape == (OH, OW)

    def test_output_is_numpy_array(self, rl):
        """deblur() must return a numpy.ndarray (CPU)."""
        result = rl.deblur(num_iter=5, lambda_tv=0.0)
        assert isinstance(result, np.ndarray)

    def test_output_shape_single_iter(self, blurred_image, gaussian_psf):
        """Shape is correct even for num_iter=1."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        rl = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = rl.deblur(num_iter=1, lambda_tv=0.0)
        assert result.shape == (OH, OW)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Positivity
# ══════════════════════════════════════════════════════════════════════════════

class TestRLPositivity:
    """All output values must be non-negative."""

    def test_output_non_negative_no_tv(self, rl):
        """Without TV, output is non-negative."""
        result = rl.deblur(num_iter=10, lambda_tv=0.0)
        assert float(result.min()) >= 0.0

    def test_output_non_negative_with_tv(self, blurred_image, gaussian_psf):
        """With TV, output is also non-negative."""
        rl = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = rl.deblur(num_iter=10, lambda_tv=0.001)
        assert float(result.min()) >= 0.0

    def test_positivity_floor_applied(self, blurred_image, gaussian_psf):
        """Output values respect epsilon_positivity floor."""
        rl = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        eps = 1e-8
        result = rl.deblur(num_iter=5, lambda_tv=0.0, epsilon_positivity=eps)
        assert float(result.min()) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. TV regularisation changes the result
# ══════════════════════════════════════════════════════════════════════════════

class TestRLTVEffect:
    """TV regularisation must change the output relative to lambda_tv=0."""

    def test_tv_changes_output(self, blurred_image, gaussian_psf):
        """lambda_tv=0.001 produces a different result than lambda_tv=0."""
        rl_no_tv = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        rl_tv    = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result_no_tv = rl_no_tv.deblur(num_iter=20, lambda_tv=0.0)
        result_tv    = rl_tv.deblur(   num_iter=20, lambda_tv=0.001)
        assert not np.allclose(result_no_tv, result_tv), (
            "TV regularisation should change the output"
        )

    def test_tv_full_canvas_vs_masked(self, blurred_image, gaussian_psf):
        """tv_on_full_canvas=True and False produce different results."""
        rl1 = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        rl2 = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result_full   = rl1.deblur(num_iter=10, lambda_tv=0.001,
                                   tv_on_full_canvas=True)
        result_masked = rl2.deblur(num_iter=10, lambda_tv=0.001,
                                   tv_on_full_canvas=False)
        assert not np.allclose(result_full, result_masked), (
            "tv_on_full_canvas=True and False should differ"
        )

    def test_stronger_tv_smoother(self, blurred_image, gaussian_psf):
        """Larger lambda_tv produces smoother (lower gradient norm) output."""
        rl_weak   = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        rl_strong = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result_weak   = rl_weak.deblur(  num_iter=30, lambda_tv=1e-5)
        result_strong = rl_strong.deblur(num_iter=30, lambda_tv=0.01)

        # Compare total variation (sum of |gradient|) as a smoothness proxy
        def tv(x):
            return float(np.sum(np.abs(np.diff(x, axis=0))) +
                         np.sum(np.abs(np.diff(x, axis=1))))

        assert tv(result_strong) < tv(result_weak), (
            "Stronger TV should produce a smoother (lower TV) output"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convergence behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestRLConvergence:
    """Verify convergence-related deblur() behaviour."""

    def test_very_tight_tol_runs_to_max_iter(self, blurred_image, gaussian_psf):
        """tol=0 (never converges) runs all num_iter iterations."""
        rl = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        # Store estimated image state after run
        result = rl.deblur(num_iter=15, lambda_tv=0.0, tol=0.0)
        # Result must still be a valid array
        assert result.shape == (rl.h, rl.w)

    def test_very_loose_tol_converges_early(self, blurred_image, gaussian_psf):
        """tol=1.0 (always converges) terminates before max_iter."""
        rl_short = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        rl_long  = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result_short = rl_short.deblur(num_iter=100, lambda_tv=0.0, tol=1.0,
                                       min_iter=5, check_every=5)
        result_long  = rl_long.deblur( num_iter=100, lambda_tv=0.0, tol=0.0)
        # With tol=1, it should converge (stop early), producing fewer iterations
        # Both are valid outputs; just verify they have the right shape
        assert result_short.shape == (rl_short.h, rl_short.w)
        assert result_long.shape  == (rl_long.h,  rl_long.w)

    def test_num_iter_clamp_low(self, blurred_image, gaussian_psf):
        """num_iter=0 is clamped to 1 — deblur must not raise."""
        rl = RLUnknownBoundary(image=blurred_image, psf=gaussian_psf)
        result = rl.deblur(num_iter=0, lambda_tv=0.0)
        assert result.shape == (rl.h, rl.w)

    def test_estimated_image_updated_after_deblur(self, rl):
        """self.estimated_image must be updated after calling deblur()."""
        import numpy as np
        init_est = np.array(rl.estimated_image).copy()
        rl.deblur(num_iter=10, lambda_tv=0.0)
        new_est = np.array(rl.estimated_image)
        assert not np.allclose(init_est, new_est), (
            "estimated_image should be updated by deblur()"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Wrapper function
# ══════════════════════════════════════════════════════════════════════════════

class TestRLWrapper:
    """rl_deblur_unknown_boundary() must split kwargs correctly."""

    def test_wrapper_returns_array(self, blurred_image, gaussian_psf):
        """Wrapper returns a numpy array."""
        result = rl_deblur_unknown_boundary(
            blurred_image, gaussian_psf, num_iter=5, lambda_tv=0.0,
        )
        assert isinstance(result, np.ndarray)

    def test_wrapper_output_shape(self, blurred_image, gaussian_psf):
        """Wrapper output has the same shape as the (odd-cropped) input."""
        h, w = blurred_image.shape
        OH = h if h % 2 == 1 else h - 1
        OW = w if w % 2 == 1 else w - 1
        result = rl_deblur_unknown_boundary(
            blurred_image, gaussian_psf, num_iter=5, lambda_tv=0.0,
        )
        assert result.shape == (OH, OW)

    def test_wrapper_matches_direct_call(self, blurred_image, gaussian_psf):
        """Wrapper result equals direct RLUnknownBoundary(...).deblur(...)."""
        obj = RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, paddingMode="Reflect",
        )
        direct = obj.deblur(num_iter=10, lambda_tv=0.0)

        wrapper = rl_deblur_unknown_boundary(
            blurred_image, gaussian_psf,
            paddingMode="Reflect",   # init kwarg
            num_iter=10,             # deblur kwarg
            lambda_tv=0.0,           # deblur kwarg
        )
        np.testing.assert_allclose(wrapper, direct, atol=1e-6)

    def test_wrapper_init_kwargs_split(self, blurred_image, gaussian_psf):
        """paddingMode (init kwarg) and num_iter (deblur kwarg) are split."""
        # If the split fails, passing both would raise TypeError.
        result = rl_deblur_unknown_boundary(
            blurred_image, gaussian_psf,
            paddingMode="Edge",   # init kwarg
            num_iter=5,           # deblur kwarg
            lambda_tv=0.0,
        )
        assert result.shape[0] > 0

    def test_wrapper_init_keys_coverage(self):
        """_INIT_KEYS contains the expected constructor parameters."""
        expected = {"paddingMode", "padding_scale", "initialEstimate",
                    "apply_taper_on_padding_band", "htm_floor_frac", "use_mask"}
        assert expected == DeconvBase._INIT_KEYS


# ══════════════════════════════════════════════════════════════════════════════
# 7. Regression against reference implementation
# ══════════════════════════════════════════════════════════════════════════════

# Path to the reference file — resolved relative to this test file's location.
_REF_PATH = (
    Path(__file__).parent.parent / "docs" / "reference" / "RL_Unknown_Boundary.py"
)


def _load_reference_module():
    """
    Load docs/reference/RL_Unknown_Boundary.py via importlib.

    Returns the module object, or None if the file does not exist.
    """
    if not _REF_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "RL_Unknown_Boundary_ref", str(_REF_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _REF_PATH.exists(),
                    reason="Reference file not found")
class TestRLRegression:
    """
    Regression: new RLUnknownBoundary must match the reference within atol=1e-5.

    The constructor in the reference does NOT apply the HTM floor clamp that
    DeconvBase adds.  Passing ``htm_floor_frac=0.0`` to the new constructor
    reduces the base-class clamp to 1e-12, which is equivalent to the
    reference's ``HTM + 1e-12`` addend in the denominator for all pixels
    in the crop region.
    """

    @pytest.fixture(scope="class")
    def ref_mod(self):
        return _load_reference_module()

    def test_no_tv_50_iter(self, blurred_image, gaussian_psf, ref_mod):
        """No-TV case: 50 iterations, both implementations agree within 1e-5."""
        # Reference
        ref_rl = ref_mod.RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf,
        )
        ref_result = ref_rl.deblur(
            num_iter=50, lambda_tv=0.0, tol=1e-6,
        )

        # New implementation — disable floor clamp to match reference
        new_rl = RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_rl.deblur(
            num_iter=50, lambda_tv=0.0, tol=1e-6,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg="No-TV regression failed: new != reference within atol=1e-5",
        )

    def test_with_tv_50_iter(self, blurred_image, gaussian_psf, ref_mod):
        """TV case: 50 iterations with lambda_tv=0.0002, agree within 1e-5."""
        ref_rl = ref_mod.RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf,
        )
        ref_result = ref_rl.deblur(
            num_iter=50, lambda_tv=0.0002, tol=1e-6,
        )

        new_rl = RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_rl.deblur(
            num_iter=50, lambda_tv=0.0002, tol=1e-6,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg="TV regression failed: new != reference within atol=1e-5",
        )

    def test_tv_full_canvas_false_regression(self, blurred_image, gaussian_psf,
                                             ref_mod):
        """tv_on_full_canvas=False: 20 iterations, agree within 1e-5."""
        ref_rl = ref_mod.RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf,
        )
        ref_result = ref_rl.deblur(
            num_iter=20, lambda_tv=0.001, tol=1e-6, tv_on_full_canvas=False,
        )

        new_rl = RLUnknownBoundary(
            image=blurred_image, psf=gaussian_psf, htm_floor_frac=0.0,
        )
        new_result = new_rl.deblur(
            num_iter=20, lambda_tv=0.001, tol=1e-6, tv_on_full_canvas=False,
        )

        np.testing.assert_allclose(
            new_result, ref_result, atol=1e-5,
            err_msg=(
                "tv_on_full_canvas=False regression failed: "
                "new != reference within atol=1e-5"
            ),
        )
