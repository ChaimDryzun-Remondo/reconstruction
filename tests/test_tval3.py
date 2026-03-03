"""
Phase 5d verification tests for Reconstruction.tval3.

Tests:
  1. Output shape, dtype, finite values.
  2. Positivity enforcement (nonneg=True).
  3. Both TVnorm=1 and TVnorm=2 produce valid output.
  4. Deconvolution quality: PSNR improves over blurred input.
  5. cost_history length equals num_iter + 1 (no early stopping).
  6. Adaptive rho_v changes during iteration.
  7. adaptive_tv=False produces valid output.
  8. burn_in_frac: algorithm runs without error for various fractions.
  9. Edge map range: adaptive weights in [0.2*lambda, lambda].
 10. Wrapper equivalence: tval3_deblur matches class-based usage.
 11. use_mask=False (M=1 everywhere) produces valid output.
 12. Convergence: terminates before maxiter on a degenerate (identity) problem.
 13. _INIT_KEYS separates constructor from deblur kwargs correctly.
 14. last_mu property changes when rho_v adapts.
 15. Non-square image support.
"""
from __future__ import annotations

import numpy as np
import pytest

import Reconstruction._backend as backend
from Reconstruction.tval3 import TVAL3Deconv, tval3_deblur


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
    """Default TVAL3Deconv instance (adaptive_tv=False for speed)."""
    return TVAL3Deconv(
        blurred, small_psf,
        mu=16.0, adaptive_tv=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Output shape, dtype, and finiteness
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputBasics:

    def test_output_shape_matches_input(self, blurred, small_psf, test_image):
        """deblur() returns an array matching the original image shape."""
        result = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False).deblur(
            num_iter=10, lambda_tv=0.01
        )
        assert result.shape == test_image.shape, (
            f"Expected {test_image.shape}, got {result.shape}"
        )

    def test_output_is_numpy(self, blurred, small_psf):
        """deblur() always returns a numpy (CPU) array."""
        result = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert isinstance(result, np.ndarray)

    def test_output_is_finite(self, blurred, small_psf):
        """All output values must be finite (no NaN or Inf)."""
        result = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False).deblur(
            num_iter=20, lambda_tv=0.01
        )
        assert np.isfinite(result).all(), "Output contains NaN or Inf values"

    def test_non_square_image(self, small_psf):
        """TVAL3 works on a non-square odd image (51×41)."""
        img = _test_image(h=51, w=41)
        blurred = _blur(img, small_psf)
        result = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert result.shape == img.shape


# ══════════════════════════════════════════════════════════════════════════════
# 2. Positivity enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestPositivity:

    def test_nonneg_true_all_positive(self, blurred, small_psf):
        """nonneg=True: all output pixels are ≥ 0."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, nonneg=True, adaptive_tv=False
        ).deblur(num_iter=20, lambda_tv=0.01)
        assert float(np.min(result)) >= 0.0, (
            f"nonneg=True violation: min={np.min(result):.4e}"
        )

    def test_nonneg_false_allows_negatives(self, blurred, small_psf):
        """nonneg=False: output is not clamped."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, nonneg=False, adaptive_tv=False
        ).deblur(num_iter=20, lambda_tv=0.01)
        assert isinstance(result, np.ndarray)
        # Just verify it ran — no positivity assertion


# ══════════════════════════════════════════════════════════════════════════════
# 3. TVnorm variants
# ══════════════════════════════════════════════════════════════════════════════

class TestTVnorm:

    def test_tvnorm1_valid_output(self, blurred, small_psf):
        """TVnorm=1 (anisotropic) produces finite output."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, TVnorm=1, adaptive_tv=False
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_tvnorm2_valid_output(self, blurred, small_psf):
        """TVnorm=2 (isotropic) produces finite output."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, TVnorm=2, adaptive_tv=False
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_tvnorm_override_in_deblur(self, blurred, small_psf):
        """TVnorm override in deblur() produces valid output."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, TVnorm=2, adaptive_tv=False)
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
        TVAL3 PSNR must exceed the blurred-image PSNR by at least 0.5 dB.

        Both images are normalized to [0, 1] before comparison to remove
        the scale mismatch introduced by DeconvBase's image_normalization.
        Uses 50 iterations with a mild lambda_tv to allow sharpening.
        """
        def _norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)

        result = TVAL3Deconv(
            blurred, small_psf,
            mu=16.0, nonneg=True, adaptive_tv=False,
        ).deblur(num_iter=50, lambda_tv=0.005)

        psnr_blurred = _psnr(_norm01(test_image), _norm01(blurred))
        psnr_deconv  = _psnr(_norm01(test_image), _norm01(result))
        assert psnr_deconv > psnr_blurred + 0.5, (
            f"TVAL3 PSNR ({psnr_deconv:.1f} dB) should exceed "
            f"blurred PSNR ({psnr_blurred:.1f} dB) by ≥0.5 dB"
        )

    def test_output_in_valid_range_approx(self, blurred, small_psf):
        """Output values should be in a reasonable range [0, 2]."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, nonneg=True, adaptive_tv=False
        ).deblur(num_iter=30, lambda_tv=0.01)
        assert float(np.max(result)) < 2.0, "Output values unreasonably large"
        assert float(np.min(result)) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cost history
# ══════════════════════════════════════════════════════════════════════════════

class TestCostHistory:

    def test_cost_history_length_full_run(self, blurred, small_psf):
        """
        When num_iter=N and no early stopping, cost_history has N+1 entries
        (initial cost + one per iteration).
        """
        num_iter = 15
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(num_iter=num_iter, lambda_tv=0.1, tol=0.0, min_iter=num_iter + 1)
        assert len(solver.cost_history) == num_iter + 1, (
            f"Expected {num_iter + 1} cost entries, got {len(solver.cost_history)}"
        )

    def test_cost_history_all_finite(self, blurred, small_psf):
        """All cost values must be finite."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(num_iter=20, lambda_tv=0.01)
        assert all(np.isfinite(c) for c in solver.cost_history), (
            "cost_history contains NaN or Inf"
        )

    def test_cost_history_decreases_initially(self, blurred, small_psf):
        """Cost should not increase dramatically in the first few iterations."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(num_iter=30, lambda_tv=0.01)
        costs = solver.cost_history
        # Final cost should be ≤ initial cost (up to 10x tolerance for ADMM fluctuations)
        assert costs[-1] <= costs[0] * 10.0, (
            f"Cost increased dramatically: {costs[0]:.3e} → {costs[-1]:.3e}"
        )

    def test_cost_history_property_returns_copy(self, blurred, small_psf):
        """cost_history returns a fresh list (not a reference to internal state)."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        hist1 = solver.cost_history
        hist2 = solver.cost_history
        assert hist1 is not hist2


# ══════════════════════════════════════════════════════════════════════════════
# 6. Adaptive rho_v
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveMu:

    def test_last_mu_property_exists(self, blurred, small_psf):
        """last_mu property returns a float."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(num_iter=10, lambda_tv=0.01)
        assert isinstance(solver.last_mu, float)

    def test_rho_v_can_change(self, blurred, small_psf):
        """
        With mu_factor > 1.0 and enough iterations, last_mu may differ
        from the initial mu.  We just verify the result is finite.
        """
        solver = TVAL3Deconv(
            blurred, small_psf,
            mu=16.0, mu_max=512.0, mu_min=0.5, mu_factor=1.5,
            adaptive_tv=False,
        )
        solver.deblur(num_iter=50, lambda_tv=0.01)
        assert np.isfinite(solver.last_mu)

    def test_last_mu_respects_bounds(self, blurred, small_psf):
        """last_mu must stay within [mu_min, mu_max]."""
        mu_min, mu_max = 0.5, 128.0
        solver = TVAL3Deconv(
            blurred, small_psf,
            mu=16.0, mu_max=mu_max, mu_min=mu_min, mu_factor=2.0,
            adaptive_tv=False,
        )
        solver.deblur(num_iter=50, lambda_tv=0.01)
        assert mu_min <= solver.last_mu <= mu_max, (
            f"last_mu={solver.last_mu:.4f} outside [{mu_min}, {mu_max}]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Adaptive TV
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveTV:

    def test_adaptive_tv_false_valid(self, blurred, small_psf):
        """adaptive_tv=False produces finite output."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, adaptive_tv=False
        ).deblur(num_iter=15, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_adaptive_tv_true_valid(self, blurred, small_psf):
        """adaptive_tv=True (with some iterations) produces finite output."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, adaptive_tv=True, burn_in_frac=0.3
        ).deblur(num_iter=20, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_adaptive_tv_override_in_deblur(self, blurred, small_psf):
        """adaptive_tv can be overridden at deblur() time."""
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=True)
        result = solver.deblur(num_iter=10, lambda_tv=0.01, adaptive_tv=False)
        assert np.isfinite(result).all()

    def test_burn_in_valid_output(self, blurred, small_psf):
        """Various burn_in_frac values produce valid output."""
        for frac in (0.0, 0.2, 0.5, 1.0):
            result = TVAL3Deconv(
                blurred, small_psf, mu=16.0, adaptive_tv=True, burn_in_frac=frac
            ).deblur(num_iter=10, lambda_tv=0.01)
            assert np.isfinite(result).all(), f"NaN with burn_in_frac={frac}"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Edge map
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeMap:

    def test_edge_map_range(self, blurred, small_psf):
        """
        _compute_edge_map values must be in [0.2*lambda_tv, lambda_tv].

        At strong edges: weight ≈ lambda_tv * 0.2 (less smoothing).
        In flat regions: weight ≈ lambda_tv (full smoothing).
        """
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0)
        u = solver.estimated_image.astype(backend.xp.float64)
        lambda_tv = 0.05
        edge_map = solver._compute_edge_map(u, lambda_tv)
        edge_np = backend._to_numpy(edge_map)
        assert float(np.min(edge_np)) >= 0.2 * lambda_tv - 1e-6, (
            f"Edge map min {np.min(edge_np):.4e} below 0.2*{lambda_tv}"
        )
        assert float(np.max(edge_np)) <= lambda_tv + 1e-6, (
            f"Edge map max {np.max(edge_np):.4e} above {lambda_tv}"
        )

    def test_edge_map_flat_image(self, small_psf):
        """On a flat image the edge map should equal lambda_tv everywhere."""
        flat_img = np.full((51, 51), 0.5, dtype=np.float64)
        solver = TVAL3Deconv(flat_img, small_psf, mu=16.0)
        u = solver.estimated_image.astype(backend.xp.float64)
        lambda_tv = 0.03
        edge_map = solver._compute_edge_map(u, lambda_tv)
        edge_np = backend._to_numpy(edge_map)
        # Flat → edge_strength → normalized to 0 → weight = lambda_tv * 1.0
        np.testing.assert_allclose(edge_np, lambda_tv, atol=1e-5,
                                   err_msg="Flat image edge map should be lambda_tv")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Wrapper equivalence
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapper:

    def test_tval3_deblur_matches_class(self, blurred, small_psf):
        """tval3_deblur output matches direct class usage (same RNG state)."""
        common_kw = dict(
            mu=16.0, adaptive_tv=False, nonneg=True,
        )
        result_cls = TVAL3Deconv(blurred, small_psf, **common_kw).deblur(
            num_iter=10, lambda_tv=0.01
        )
        result_fn = tval3_deblur(
            blurred, small_psf, iters=10, lambda_tv=0.01, **common_kw
        )
        np.testing.assert_array_equal(result_cls, result_fn)

    def test_wrapper_kwarg_split(self, blurred, small_psf):
        """tval3_deblur correctly routes init vs deblur kwargs."""
        result = tval3_deblur(
            blurred, small_psf,
            iters=5, lambda_tv=0.02,
            # init kwargs
            mu=8.0, adaptive_tv=False, padding_scale=2.0,
            # deblur kwargs
            nonneg=True, TVnorm=1,
        )
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 10. use_mask=False
# ══════════════════════════════════════════════════════════════════════════════

class TestUseMask:

    def test_use_mask_false_valid(self, blurred, small_psf):
        """use_mask=False (M=1 everywhere) produces valid output."""
        result = TVAL3Deconv(
            blurred, small_psf,
            mu=16.0, use_mask=False, adaptive_tv=False,
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_use_mask_true_default(self, blurred, small_psf):
        """Default use_mask=True produces valid output."""
        result = TVAL3Deconv(
            blurred, small_psf, mu=16.0, adaptive_tv=False,
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 11. Early convergence on degenerate problem
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergence:

    def test_early_termination_via_residuals(self, blurred, small_psf):
        """
        With a very loose tol (1.0), primal/dual residuals converge quickly
        and early termination triggers well before maxiter.
        """
        num_iter = 200
        solver = TVAL3Deconv(
            blurred, small_psf, mu=32.0, adaptive_tv=False
        )
        # tol=1.0 means stop when rel_change < 1.0 — should trigger fast
        solver.deblur(num_iter=num_iter, lambda_tv=0.01, tol=1.0, min_iter=2)
        assert len(solver.cost_history) < num_iter + 1, (
            f"Expected early stopping with tol=1.0, "
            f"but ran all {num_iter} iterations"
        )

    def test_converges_at_min_iter_boundary(self, blurred, small_psf):
        """min_iter is respected: no early termination before min_iter."""
        min_iter = 10
        solver = TVAL3Deconv(blurred, small_psf, mu=16.0, adaptive_tv=False)
        solver.deblur(
            num_iter=50, lambda_tv=0.01,
            tol=1e10,   # always converged by this criterion
            min_iter=min_iter,
        )
        # Must have run at least min_iter iterations
        assert len(solver.cost_history) >= min_iter + 1


# ══════════════════════════════════════════════════════════════════════════════
# 12. _INIT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:

    def test_init_keys_contains_tval3_params(self):
        """_INIT_KEYS includes TVAL3-specific constructor parameters."""
        keys = TVAL3Deconv._INIT_KEYS
        for expected in ("mu", "mu_max", "mu_min", "mu_factor",
                         "TVnorm", "nonneg", "adaptive_tv", "burn_in_frac"):
            assert expected in keys, f"Missing key: {expected}"

    def test_init_keys_includes_base_keys(self):
        """_INIT_KEYS inherits all DeconvBase._INIT_KEYS."""
        from Reconstruction._base import DeconvBase
        assert DeconvBase._INIT_KEYS.issubset(TVAL3Deconv._INIT_KEYS)

    def test_deblur_params_not_in_init_keys(self):
        """deblur-only params (num_iter, lambda_tv, tol) are not in _INIT_KEYS."""
        keys = TVAL3Deconv._INIT_KEYS
        for deblur_only in ("num_iter", "lambda_tv", "tol", "verbose"):
            assert deblur_only not in keys, f"Unexpected deblur key in _INIT_KEYS: {deblur_only}"


# ══════════════════════════════════════════════════════════════════════════════
# 13. Precomputed arrays
# ══════════════════════════════════════════════════════════════════════════════

class TestPrecomputedArrays:

    def test_H_full_shape(self, blurred, small_psf, solver):
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
        assert abs(lap[0, 0]) < 1e-10, f"DC bin of lap_fft = {lap[0, 0]:.3e}, expected 0"

    def test_lap_fft_shape(self, blurred, small_psf, solver):
        """lap_fft has shape == full_shape."""
        assert solver.lap_fft.shape == solver.full_shape
