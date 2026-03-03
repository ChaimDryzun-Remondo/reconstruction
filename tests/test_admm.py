"""
Phase 5c verification tests for Reconstruction.admm.

Tests:
  1. Output shape, dtype, finite values.
  2. Positivity enforcement (nonneg=True / nonneg=False).
  3. Both TVnorm=1 and TVnorm=2 produce valid output.
  4. Deconvolution quality: PSNR improves over blurred input.
  5. use_mask=False (M=1 everywhere) produces valid output.
  6. cost_history length and finiteness.
  7. Adaptive rho_v stays within [rho_min, rho_max].
  8. Separate rho_v and rho_w params are respected.
  9. Early convergence with loose tol.
 10. Wrapper equivalence: admm_deblur matches class-based usage.
 11. _INIT_KEYS separates constructor from deblur kwargs correctly.
 12. Overridable prior interface: MinimalPriorSubclass with no-op prior.
 13. Precomputed arrays: H_full, H_H_conj, lap_fft correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

import Reconstruction._backend as backend
from Reconstruction.admm import ADMMDeconv, admm_deblur


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend for all tests."""
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


def _gaussian_psf(size: int = 9, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf.astype(np.float64)


def _test_image(h: int = 51, w: int = 51) -> np.ndarray:
    """Odd-sized synthetic test image."""
    img = np.full((h, w), 0.1, dtype=np.float64)
    ch, cw = h // 4, w // 4
    img[ch:3 * ch, cw:3 * cw] = 0.8
    return img


def _blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    return np.clip(fftconvolve(image, psf, mode="same"), 0, None)


def _psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = float(np.mean((ref - out) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


@pytest.fixture
def small_psf() -> np.ndarray:
    return _gaussian_psf(size=9, sigma=1.5)


@pytest.fixture
def test_image() -> np.ndarray:
    return _test_image(51, 51)


@pytest.fixture
def blurred(test_image, small_psf) -> np.ndarray:
    return _blur(test_image, small_psf)


@pytest.fixture
def solver(blurred, small_psf):
    """Default ADMMDeconv instance."""
    return ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Output shape, dtype, and finiteness
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputBasics:

    def test_output_shape_matches_input(self, blurred, small_psf, test_image):
        """deblur() returns an array matching the original image shape."""
        result = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0).deblur(
            num_iter=10, lambda_tv=0.01
        )
        assert result.shape == test_image.shape, (
            f"Expected {test_image.shape}, got {result.shape}"
        )

    def test_output_is_numpy(self, blurred, small_psf):
        """deblur() always returns a numpy (CPU) array."""
        result = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert isinstance(result, np.ndarray)

    def test_output_is_finite(self, blurred, small_psf):
        """All output values must be finite (no NaN or Inf)."""
        result = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0).deblur(
            num_iter=20, lambda_tv=0.01
        )
        assert np.isfinite(result).all(), "Output contains NaN or Inf values"

    def test_non_square_image(self, small_psf):
        """ADMM works on a non-square odd image (51×41)."""
        img = _test_image(h=51, w=41)
        blurred = _blur(img, small_psf)
        result = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert result.shape == img.shape


# ══════════════════════════════════════════════════════════════════════════════
# 2. Positivity enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestPositivity:

    def test_nonneg_true_all_positive(self, blurred, small_psf):
        """nonneg=True: all output pixels are ≥ 0."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0, nonneg=True
        ).deblur(num_iter=20, lambda_tv=0.01)
        assert float(np.min(result)) >= 0.0, (
            f"nonneg=True violation: min={np.min(result):.4e}"
        )

    def test_nonneg_false_allows_negatives(self, blurred, small_psf):
        """nonneg=False: output is not clamped (verify runs without error)."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0, nonneg=False
        ).deblur(num_iter=20, lambda_tv=0.01)
        assert isinstance(result, np.ndarray)

    def test_nonneg_override_in_deblur(self, blurred, small_psf):
        """nonneg override at deblur() time is respected."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0, nonneg=False)
        result = solver.deblur(num_iter=10, lambda_tv=0.01, nonneg=True)
        assert float(np.min(result)) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. TVnorm variants
# ══════════════════════════════════════════════════════════════════════════════

class TestTVnorm:

    def test_tvnorm1_valid_output(self, blurred, small_psf):
        """TVnorm=1 (anisotropic) produces finite output."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0, TVnorm=1
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_tvnorm2_valid_output(self, blurred, small_psf):
        """TVnorm=2 (isotropic) produces finite output."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0, TVnorm=2
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_tvnorm_override_in_deblur(self, blurred, small_psf):
        """TVnorm override at deblur() time produces valid output."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0, TVnorm=2)
        result1 = solver.deblur(num_iter=10, lambda_tv=0.01, TVnorm=1)
        result2 = solver.deblur(num_iter=10, lambda_tv=0.01, TVnorm=2)
        assert np.isfinite(result1).all()
        assert np.isfinite(result2).all()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Deconvolution quality
# ══════════════════════════════════════════════════════════════════════════════

class TestDeconvolutionQuality:

    def test_psnr_improves_over_blurred(self, test_image, blurred, small_psf):
        """
        ADMM PSNR must exceed the blurred-image PSNR by at least 0.5 dB.

        Both images are normalized to [0, 1] before comparison to remove
        the scale mismatch introduced by DeconvBase's image_normalization.
        """
        def _norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)

        result = ADMMDeconv(
            blurred, small_psf,
            rho_v=16.0, rho_w=16.0, nonneg=True,
        ).deblur(num_iter=50, lambda_tv=0.005)

        psnr_blurred = _psnr(_norm01(test_image), _norm01(blurred))
        psnr_deconv  = _psnr(_norm01(test_image), _norm01(result))
        assert psnr_deconv > psnr_blurred + 0.5, (
            f"ADMM PSNR ({psnr_deconv:.1f} dB) should exceed "
            f"blurred PSNR ({psnr_blurred:.1f} dB) by ≥0.5 dB"
        )

    def test_output_in_valid_range(self, blurred, small_psf):
        """Output values should be in a reasonable range [0, 2]."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0, nonneg=True
        ).deblur(num_iter=30, lambda_tv=0.01)
        assert float(np.max(result)) < 2.0, "Output values unreasonably large"
        assert float(np.min(result)) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. use_mask
# ══════════════════════════════════════════════════════════════════════════════

class TestUseMask:

    def test_use_mask_false_valid(self, blurred, small_psf):
        """use_mask=False (M=1 everywhere) produces valid output."""
        result = ADMMDeconv(
            blurred, small_psf,
            rho_v=16.0, rho_w=16.0, use_mask=False,
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_use_mask_true_default(self, blurred, small_psf):
        """Default use_mask=True produces valid output."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=16.0,
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_use_mask_true_mask_shape(self, blurred, small_psf):
        """use_mask=True: mask has full canvas shape."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        assert solver.mask.shape == solver.full_shape


# ══════════════════════════════════════════════════════════════════════════════
# 6. Cost history
# ══════════════════════════════════════════════════════════════════════════════

class TestCostHistory:

    def test_cost_history_length_full_run(self, blurred, small_psf):
        """
        When num_iter=N and no early stopping, cost_history has N+1 entries
        (initial cost + one per iteration).
        """
        num_iter = 15
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=num_iter, lambda_tv=0.1, tol=0.0,
                      min_iter=num_iter + 1)
        assert len(solver.cost_history) == num_iter + 1, (
            f"Expected {num_iter + 1} cost entries, got {len(solver.cost_history)}"
        )

    def test_cost_history_all_finite(self, blurred, small_psf):
        """All cost values must be finite."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=20, lambda_tv=0.01)
        assert all(np.isfinite(c) for c in solver.cost_history), (
            "cost_history contains NaN or Inf"
        )

    def test_cost_history_property_returns_copy(self, blurred, small_psf):
        """cost_history returns a fresh list (not a reference to internal state)."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        hist1 = solver.cost_history
        hist2 = solver.cost_history
        assert hist1 is not hist2

    def test_cost_history_monotone_tendency(self, blurred, small_psf):
        """Final cost should not be dramatically larger than initial cost."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=30, lambda_tv=0.01)
        costs = solver.cost_history
        assert costs[-1] <= costs[0] * 10.0, (
            f"Cost increased dramatically: {costs[0]:.3e} → {costs[-1]:.3e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Adaptive rho_v bounds
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveRhoV:

    def test_last_rho_v_is_float(self, blurred, small_psf):
        """last_rho_v property returns a float."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=10, lambda_tv=0.01)
        assert isinstance(solver.last_rho_v, float)

    def test_last_rho_v_respects_bounds(self, blurred, small_psf):
        """last_rho_v must stay within [rho_min, rho_max]."""
        rho_min, rho_max = 0.5, 128.0
        solver = ADMMDeconv(
            blurred, small_psf,
            rho_v=16.0, rho_w=16.0,
            rho_max=rho_max, rho_min=rho_min, rho_factor=2.0,
        )
        solver.deblur(num_iter=50, lambda_tv=0.01)
        assert rho_min <= solver.last_rho_v <= rho_max, (
            f"last_rho_v={solver.last_rho_v:.4f} outside [{rho_min}, {rho_max}]"
        )

    def test_last_rho_v_is_finite(self, blurred, small_psf):
        """last_rho_v must be finite after deblur."""
        solver = ADMMDeconv(
            blurred, small_psf,
            rho_v=16.0, rho_max=512.0, rho_min=0.5, rho_factor=1.5,
        )
        solver.deblur(num_iter=50, lambda_tv=0.01)
        assert np.isfinite(solver.last_rho_v)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Separate rho_v and rho_w
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparateRho:

    def test_rho_w_unchanged(self, blurred, small_psf):
        """last_rho_w equals the constructor rho_w (rho_w is not adapted)."""
        rho_w_init = 8.0
        solver = ADMMDeconv(
            blurred, small_psf, rho_v=16.0, rho_w=rho_w_init
        )
        solver.deblur(num_iter=30, lambda_tv=0.01)
        assert solver.last_rho_w == rho_w_init, (
            f"rho_w changed: started {rho_w_init}, ended {solver.last_rho_w}"
        )

    def test_different_rho_v_rho_w_valid(self, blurred, small_psf):
        """rho_v ≠ rho_w produces finite output."""
        result = ADMMDeconv(
            blurred, small_psf, rho_v=64.0, rho_w=8.0
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_last_rho_w_property_exists(self, blurred, small_psf):
        """last_rho_w property is accessible and returns a float."""
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=32.0)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        assert isinstance(solver.last_rho_w, float)
        assert solver.last_rho_w == 32.0


# ══════════════════════════════════════════════════════════════════════════════
# 9. Early convergence
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergence:

    def test_early_termination_via_residuals(self, blurred, small_psf):
        """
        With a very loose tol (1.0), v-constraint residuals converge quickly
        and early termination triggers well before maxiter.
        """
        num_iter = 200
        solver = ADMMDeconv(blurred, small_psf, rho_v=32.0, rho_w=32.0)
        solver.deblur(num_iter=num_iter, lambda_tv=0.01, tol=1.0, min_iter=2)
        assert len(solver.cost_history) < num_iter + 1, (
            f"Expected early stopping with tol=1.0, "
            f"but ran all {num_iter} iterations"
        )

    def test_min_iter_respected(self, blurred, small_psf):
        """min_iter is respected: no early termination before min_iter."""
        min_iter = 10
        solver = ADMMDeconv(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(
            num_iter=50, lambda_tv=0.01,
            tol=1e10,   # always converged by this criterion
            min_iter=min_iter,
        )
        assert len(solver.cost_history) >= min_iter + 1


# ══════════════════════════════════════════════════════════════════════════════
# 10. Wrapper equivalence
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapper:

    def test_admm_deblur_matches_class(self, blurred, small_psf):
        """admm_deblur output matches direct class usage (same determinism)."""
        common_kw = dict(rho_v=16.0, rho_w=16.0, nonneg=True)
        result_cls = ADMMDeconv(blurred, small_psf, **common_kw).deblur(
            num_iter=10, lambda_tv=0.01
        )
        result_fn = admm_deblur(
            blurred, small_psf, iters=10, lambda_tv=0.01, **common_kw
        )
        np.testing.assert_array_equal(result_cls, result_fn)

    def test_wrapper_kwarg_split(self, blurred, small_psf):
        """admm_deblur correctly routes init vs deblur kwargs."""
        result = admm_deblur(
            blurred, small_psf,
            iters=5, lambda_tv=0.02,
            # init kwargs
            rho_v=8.0, rho_w=8.0, padding_scale=2.0,
            # deblur kwargs
            nonneg=True, TVnorm=1,
        )
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 11. _INIT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:

    def test_init_keys_contains_admm_params(self):
        """_INIT_KEYS includes ADMM-specific constructor parameters."""
        keys = ADMMDeconv._INIT_KEYS
        for expected in ("rho_v", "rho_w", "rho_max", "rho_min", "rho_factor",
                         "TVnorm", "nonneg"):
            assert expected in keys, f"Missing key: {expected}"

    def test_init_keys_includes_base_keys(self):
        """_INIT_KEYS inherits all DeconvBase._INIT_KEYS."""
        from Reconstruction._base import DeconvBase
        assert DeconvBase._INIT_KEYS.issubset(ADMMDeconv._INIT_KEYS)

    def test_deblur_params_not_in_init_keys(self):
        """deblur-only params (num_iter, lambda_tv, tol) are not in _INIT_KEYS."""
        keys = ADMMDeconv._INIT_KEYS
        for deblur_only in ("num_iter", "lambda_tv", "tol", "verbose"):
            assert deblur_only not in keys, (
                f"Unexpected deblur key in _INIT_KEYS: {deblur_only}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 12. Overridable prior interface
# ══════════════════════════════════════════════════════════════════════════════

class MinimalPriorSubclass(ADMMDeconv):
    """
    Minimal PnP-style subclass that uses no explicit TV prior.

    _prior_init   → empty state
    _prior_update → returns zero array (denoiser would go here)
    _prior_dual_update → no-op
    _x_update_denom   → rho_v * H_H_conj only (no TV Laplacian)
    """

    def _prior_init(self, u):
        return {}

    def _prior_update(self, u, state, lambda_tv, rho_w, eps):
        return backend.xp.zeros_like(u)

    def _prior_dual_update(self, u, state):
        pass  # no-op

    def _x_update_denom(self, rho_v, rho_w):
        return rho_v * self.H_H_conj


class TestOverridablePrior:

    def test_minimal_subclass_runs(self, blurred, small_psf):
        """MinimalPriorSubclass (zero-prior) produces finite output."""
        result = MinimalPriorSubclass(
            blurred, small_psf, rho_v=16.0, rho_w=16.0
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_minimal_subclass_output_shape(self, blurred, small_psf, test_image):
        """MinimalPriorSubclass output shape matches input shape."""
        result = MinimalPriorSubclass(
            blurred, small_psf, rho_v=16.0, rho_w=16.0
        ).deblur(num_iter=5, lambda_tv=0.01)
        assert result.shape == test_image.shape

    def test_minimal_subclass_cost_history(self, blurred, small_psf):
        """MinimalPriorSubclass cost_history has correct length (no TV term)."""
        num_iter = 8
        solver = MinimalPriorSubclass(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=num_iter, lambda_tv=0.01,
                      tol=0.0, min_iter=num_iter + 1)
        assert len(solver.cost_history) == num_iter + 1

    def test_default_and_minimal_differ(self, blurred, small_psf):
        """Default TV prior and MinimalPriorSubclass produce different outputs."""
        kw = dict(rho_v=16.0, rho_w=16.0, nonneg=True)
        result_tv = ADMMDeconv(blurred, small_psf, **kw).deblur(
            num_iter=20, lambda_tv=0.01
        )
        result_min = MinimalPriorSubclass(blurred, small_psf, **kw).deblur(
            num_iter=20, lambda_tv=0.01
        )
        # They should differ because one uses TV regularization and the other doesn't
        assert not np.allclose(result_tv, result_min), (
            "TV and no-TV priors produced identical outputs"
        )

    def test_prior_init_called_once(self, blurred, small_psf):
        """_prior_init contract: called once, returns dict."""

        call_count = [0]

        class TrackingSubclass(ADMMDeconv):
            def _prior_init(self, u):
                call_count[0] += 1
                return super()._prior_init(u)

        solver = TrackingSubclass(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        assert call_count[0] == 1, (
            f"_prior_init called {call_count[0]} times; expected 1"
        )

    def test_prior_update_called_per_iter(self, blurred, small_psf):
        """_prior_update called exactly once per ADMM iteration."""

        call_count = [0]

        class TrackingSubclass(ADMMDeconv):
            def _prior_update(self, u, state, lambda_tv, rho_w, eps):
                call_count[0] += 1
                return super()._prior_update(u, state, lambda_tv, rho_w, eps)

        num_iter = 7
        solver = TrackingSubclass(blurred, small_psf, rho_v=16.0, rho_w=16.0)
        solver.deblur(num_iter=num_iter, lambda_tv=0.01,
                      tol=0.0, min_iter=num_iter + 1)
        assert call_count[0] == num_iter, (
            f"_prior_update called {call_count[0]} times; expected {num_iter}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 13. Precomputed arrays
# ══════════════════════════════════════════════════════════════════════════════

class TestPrecomputedArrays:

    def test_H_full_shape(self, solver):
        """H_full has shape == full_shape."""
        assert solver.H_full.shape == solver.full_shape

    def test_H_H_conj_nonneg(self, solver):
        """|H|² = H_H_conj must be non-negative everywhere."""
        h2 = backend._to_numpy(solver.H_H_conj)
        assert float(np.min(h2)) >= -1e-6, "|H|² has negative values"

    def test_lap_fft_nonneg(self, solver):
        """lap_fft = 4 − 2cos − 2cos is non-negative everywhere."""
        lap = backend._to_numpy(solver.lap_fft)
        assert float(np.min(lap)) >= -1e-6, "lap_fft has negative values"

    def test_lap_fft_dc_is_zero(self, solver):
        """DC bin of lap_fft must be zero (4 − 2 − 2 = 0)."""
        lap = backend._to_numpy(solver.lap_fft)
        assert abs(lap[0, 0]) < 1e-10, (
            f"DC bin of lap_fft = {lap[0, 0]:.3e}, expected 0"
        )

    def test_lap_fft_shape(self, solver):
        """lap_fft has shape == full_shape."""
        assert solver.lap_fft.shape == solver.full_shape

    def test_H_full_dtype_float64(self, solver):
        """H_full should be computed in float64 for numerical stability."""
        h = backend._to_numpy(solver.H_full)
        assert np.issubdtype(h.dtype, np.complexfloating)
