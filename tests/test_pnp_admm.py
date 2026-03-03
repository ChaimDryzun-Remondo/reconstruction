"""
Phase PnP verification tests for Reconstruction.pnp_admm.

All tests are skipped when the bm3d package is not installed.

Tests:
  1. Construction: PnPADMM can be constructed (verifies bm3d import).
  2. Output shape, dtype, finiteness.
  3. Positivity enforcement (nonneg=True).
  4. Deconvolution quality: PSNR improves over blurred input.
  5. use_mask behaviour (inherited v-update).
  6. _denoise: reduces noise on a synthetic noisy image.
  7. sigma derivation: σ = sigma_scale · √(λ/ρ_z).
  8. sigma_scale effect: higher sigma_scale → smoother output.
  9. _x_update_denom: ρ_v · |H|² + ρ_z (no lap_fft).
 10. _prior_init: returns dict with "z" and "d_z" keys.
 11. _prior_dual_update: d_z increments by (x − z).
 12. cost_history: populated with finite values.
 13. Wrapper equivalence: pnp_admm_deblur matches class-based usage.
 14. _INIT_KEYS: contains rho_z, denoiser_profile, sigma_scale.
 15. Inheritance: PnPADMM is a subclass of ADMMDeconv and DeconvBase.
 16. GPU/CPU transfer: _denoise does not crash when running on CPU.
 17. rho_z property alias for rho_w.
 18. ImportError when bm3d absent (tested via a mock path).
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip entire module when bm3d is not installed
bm3d_module = pytest.importorskip("bm3d", reason="bm3d not installed")

import Reconstruction._backend as backend
from Reconstruction._base import DeconvBase
from Reconstruction.admm import ADMMDeconv
from Reconstruction.pnp_admm import PnPADMM, pnp_admm_deblur, _HAS_BM3D


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


def _test_image(h: int = 41, w: int = 41) -> np.ndarray:
    """Small odd-sized synthetic test image (fast for BM3D)."""
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
    return _gaussian_psf(size=7, sigma=1.2)


@pytest.fixture
def test_image() -> np.ndarray:
    return _test_image(41, 41)


@pytest.fixture
def blurred(test_image, small_psf) -> np.ndarray:
    return _blur(test_image, small_psf)


@pytest.fixture
def solver(blurred, small_psf) -> PnPADMM:
    """Default PnPADMM instance (small params for speed)."""
    return PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Construction
# ══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_constructs_successfully(self, blurred, small_psf):
        """PnPADMM can be constructed when bm3d is installed."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        assert solver is not None

    def test_denoiser_profile_stored(self, blurred, small_psf):
        """Constructor stores denoiser_profile as given."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=2.0,
                         denoiser_profile="lc")
        assert solver.denoiser_profile == "lc"

    def test_sigma_scale_stored(self, blurred, small_psf):
        """Constructor stores sigma_scale as given."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0,
                         sigma_scale=0.75)
        assert solver.sigma_scale == 0.75

    def test_rho_z_property(self, blurred, small_psf):
        """rho_z property equals the constructor rho_z argument."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=4.0)
        assert solver.rho_z == 4.0

    def test_rho_z_maps_to_rho_w(self, blurred, small_psf):
        """rho_z is stored as self.rho_w (parent's second-penalty slot)."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=3.5)
        assert solver.rho_w == 3.5


# ══════════════════════════════════════════════════════════════════════════════
# 2. Output shape, dtype, finiteness
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputBasics:

    def test_output_shape_matches_input(self, blurred, small_psf, test_image):
        """deblur() returns an array matching the original image shape."""
        result = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0).deblur(
            num_iter=3, lambda_tv=0.01
        )
        assert result.shape == test_image.shape, (
            f"Expected {test_image.shape}, got {result.shape}"
        )

    def test_output_is_numpy(self, blurred, small_psf):
        """deblur() always returns a CPU numpy array."""
        result = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0).deblur(
            num_iter=3, lambda_tv=0.01
        )
        assert isinstance(result, np.ndarray)

    def test_output_is_finite(self, blurred, small_psf):
        """All output values must be finite (no NaN or Inf)."""
        result = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert np.isfinite(result).all(), "Output contains NaN or Inf"

    def test_non_square_image(self, small_psf):
        """PnPADMM works on a non-square odd image (41×31)."""
        img = _test_image(h=41, w=31)
        blurred = _blur(img, small_psf)
        result = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0).deblur(
            num_iter=3, lambda_tv=0.01
        )
        assert result.shape == img.shape


# ══════════════════════════════════════════════════════════════════════════════
# 3. Positivity
# ══════════════════════════════════════════════════════════════════════════════

class TestPositivity:

    def test_nonneg_true_all_positive(self, blurred, small_psf):
        """nonneg=True: all output pixels ≥ 0."""
        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=1.0, nonneg=True
        ).deblur(num_iter=5, lambda_tv=0.01)
        assert float(np.min(result)) >= 0.0, (
            f"nonneg=True violation: min={np.min(result):.4e}"
        )

    def test_nonneg_false_runs(self, blurred, small_psf):
        """nonneg=False: output is not clamped (verify runs without error)."""
        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=1.0, nonneg=False
        ).deblur(num_iter=5, lambda_tv=0.01)
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Deconvolution quality
# ══════════════════════════════════════════════════════════════════════════════

class TestDeconvolutionQuality:

    def test_psnr_improves_over_blurred(self, test_image, blurred, small_psf):
        """
        PnP-ADMM PSNR must exceed blurred-image PSNR by at least 0.5 dB.

        Both images normalised to [0, 1] before comparison to remove
        the scale mismatch introduced by DeconvBase's normalisation.
        """
        def _norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)

        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=2.0, nonneg=True,
        ).deblur(num_iter=15, lambda_tv=0.005)

        psnr_blurred = _psnr(_norm01(test_image), _norm01(blurred))
        psnr_deconv  = _psnr(_norm01(test_image), _norm01(result))
        assert psnr_deconv > psnr_blurred + 0.5, (
            f"PnP PSNR ({psnr_deconv:.1f} dB) should exceed "
            f"blurred PSNR ({psnr_blurred:.1f} dB) by ≥0.5 dB"
        )

    def test_output_in_valid_range(self, blurred, small_psf):
        """Output values should be in a reasonable range [0, 2]."""
        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=1.0, nonneg=True
        ).deblur(num_iter=10, lambda_tv=0.01)
        assert float(np.max(result)) < 2.0
        assert float(np.min(result)) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. use_mask (inherited v-update)
# ══════════════════════════════════════════════════════════════════════════════

class TestUseMask:

    def test_use_mask_true_valid(self, blurred, small_psf):
        """use_mask=True (default) produces valid output."""
        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=1.0
        ).deblur(num_iter=3, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_use_mask_false_valid(self, blurred, small_psf):
        """use_mask=False (M=1 everywhere) produces valid output."""
        result = PnPADMM(
            blurred, small_psf, rho_v=1.0, rho_z=1.0, use_mask=False
        ).deblur(num_iter=3, lambda_tv=0.01)
        assert np.isfinite(result).all()

    def test_mask_shape_matches_canvas(self, blurred, small_psf):
        """Mask has the full canvas shape."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        assert solver.mask.shape == solver.full_shape


# ══════════════════════════════════════════════════════════════════════════════
# 6. _denoise method
# ══════════════════════════════════════════════════════════════════════════════

class TestDenoise:

    def test_denoise_reduces_noise(self, blurred, small_psf):
        """_denoise output has lower noise std than the noisy input."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        rng = np.random.default_rng(42)
        clean = np.clip(blurred / (blurred.max() + 1e-8), 0, 1).astype(np.float64)
        noise = rng.normal(0, 0.05, clean.shape)
        noisy = np.clip(clean + noise, 0, 1)

        noisy_xp = backend.xp.array(noisy, dtype=backend.xp.float64)
        denoised_xp = solver._denoise(noisy_xp, sigma=0.05)
        denoised = backend._to_numpy(denoised_xp)

        noise_before = float(np.std(noisy - clean))
        noise_after  = float(np.std(denoised - clean))
        assert noise_after < noise_before, (
            f"BM3D should reduce noise: before={noise_before:.4f}, "
            f"after={noise_after:.4f}"
        )

    def test_denoise_output_shape(self, blurred, small_psf):
        """_denoise preserves array shape."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        arr = backend.xp.array(np.clip(blurred, 0, 1), dtype=backend.xp.float64)
        result = solver._denoise(arr, sigma=0.05)
        assert result.shape == arr.shape

    def test_denoise_output_clipped(self, blurred, small_psf):
        """_denoise output stays in [0, 1]."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        arr = backend.xp.ones(blurred.shape, dtype=backend.xp.float64) * 0.5
        result = backend._to_numpy(solver._denoise(arr, sigma=0.05))
        assert float(np.min(result)) >= 0.0 - 1e-9
        assert float(np.max(result)) <= 1.0 + 1e-9

    def test_denoise_tiny_sigma_noop(self, blurred, small_psf):
        """_denoise with sigma < 1e-6 returns the input unchanged."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        arr = backend.xp.array(np.clip(blurred, 0, 1), dtype=backend.xp.float64)
        result = solver._denoise(arr, sigma=0.0)
        # Should return exactly the same object (early-exit)
        assert result is arr

    def test_denoise_cpu_no_crash(self, blurred, small_psf):
        """On CPU backend, _denoise runs without raising."""
        backend.set_backend("cpu")
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        arr = backend.xp.array(np.clip(blurred, 0, 1), dtype=backend.xp.float64)
        result = solver._denoise(arr, sigma=0.03)
        assert result.shape == arr.shape


# ══════════════════════════════════════════════════════════════════════════════
# 7. sigma derivation
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaDerivation:

    def test_sigma_formula(self, blurred, small_psf):
        """
        The sigma passed to BM3D equals sigma_scale * sqrt(lambda_tv / rho_z).

        We capture the sigma via a subclass that records the last value.
        """
        captured = {}

        class CaptureSigmaPnP(PnPADMM):
            def _denoise(self, image, sigma):
                captured["sigma"] = sigma
                return image  # identity denoiser to avoid BM3D cost

        lambda_tv = 0.04
        rho_z = 4.0
        sigma_scale = 0.5
        solver = CaptureSigmaPnP(
            blurred, small_psf,
            rho_v=1.0, rho_z=rho_z, sigma_scale=sigma_scale
        )
        solver.deblur(num_iter=1, lambda_tv=lambda_tv)

        expected = sigma_scale * np.sqrt(lambda_tv / rho_z)
        assert abs(captured["sigma"] - expected) < 1e-8, (
            f"sigma={captured['sigma']:.6f}, expected={expected:.6f}"
        )

    def test_sigma_floored_at_1e6(self, blurred, small_psf):
        """
        sigma is floored at 1e-6 (prevents near-zero BM3D calls).
        Test with lambda_tv=0 → sigma should equal 1e-6, not 0.
        """
        captured = {}

        class CaptureSigmaPnP(PnPADMM):
            def _denoise(self, image, sigma):
                captured["sigma"] = sigma
                return image

        solver = CaptureSigmaPnP(
            blurred, small_psf, rho_v=1.0, rho_z=1.0, sigma_scale=1.0
        )
        solver.deblur(num_iter=1, lambda_tv=0.0)
        # lambda_tv=0 → sqrt(0/rho_z)=0 → floor to 1e-6
        assert captured["sigma"] >= 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 8. sigma_scale effect
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaScaleEffect:

    def test_higher_sigma_scale_smoother(self, blurred, small_psf):
        """
        Higher sigma_scale → stronger BM3D → smoother output.

        Smoothness measured via std-dev of horizontal finite differences.
        Uses an extreme sigma_scale ratio and sufficient iterations so the
        ADMM dynamics have time to converge to a clearly smoother solution.
        """
        kw = dict(rho_v=1.0, rho_z=1.0, nonneg=True)
        result_low = PnPADMM(
            blurred, small_psf, sigma_scale=0.02, **kw
        ).deblur(num_iter=20, lambda_tv=0.01)
        result_high = PnPADMM(
            blurred, small_psf, sigma_scale=5.0, **kw
        ).deblur(num_iter=20, lambda_tv=0.01)

        std_low  = float(np.std(np.diff(result_low,  axis=1)))
        std_high = float(np.std(np.diff(result_high, axis=1)))
        assert std_high < std_low, (
            f"Higher sigma_scale should produce smoother output: "
            f"std_low={std_low:.4f}, std_high={std_high:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. _x_update_denom
# ══════════════════════════════════════════════════════════════════════════════

class TestXUpdateDenom:

    def test_denom_no_lap_fft(self, solver):
        """
        _x_update_denom returns ρ_v · |H|² + ρ_z (no lap_fft).

        Verify by comparing to the expected formula directly.
        """
        rho_v, rho_z = 2.0, 3.0
        denom = solver._x_update_denom(rho_v, rho_z)
        expected = rho_v * solver.H_H_conj + rho_z
        np.testing.assert_allclose(
            backend._to_numpy(denom),
            backend._to_numpy(expected),
            atol=1e-10,
            err_msg="_x_update_denom mismatch with rho_v * H_H_conj + rho_z",
        )

    def test_denom_differs_from_tv_denom(self, solver):
        """
        PnP denominator differs from TV denominator (which uses lap_fft).

        At most frequency bins (where lap_fft ≠ 0) the two denominators
        differ.
        """
        rho_v, rho_w = 2.0, 3.0
        pnp_denom = solver._x_update_denom(rho_v, rho_w)
        tv_denom   = rho_v * solver.H_H_conj + rho_w * solver.lap_fft
        diff = backend._to_numpy(pnp_denom - tv_denom)
        # They must differ (lap_fft is not identically 0 except at DC)
        assert float(np.max(np.abs(diff))) > 1e-6, (
            "PnP and TV denominators should differ outside DC bin"
        )

    def test_denom_shape(self, solver):
        """_x_update_denom returns array of shape full_shape."""
        denom = solver._x_update_denom(1.0, 1.0)
        assert denom.shape == solver.full_shape

    def test_denom_positive_everywhere(self, solver):
        """denom = ρ_v |H|² + ρ_z must be ≥ ρ_z > 0 everywhere."""
        rho_z = 2.0
        denom = backend._to_numpy(solver._x_update_denom(1.0, rho_z))
        assert float(np.min(denom)) >= rho_z - 1e-10, (
            "denom should be ≥ rho_z everywhere"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. _prior_init
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorInit:

    def test_returns_z_and_d_z(self, solver):
        """_prior_init returns dict with exactly 'z' and 'd_z' keys."""
        u = solver.estimated_image.astype(backend.xp.float64)
        state = solver._prior_init(u)
        assert "z" in state, "State missing 'z' key"
        assert "d_z" in state, "State missing 'd_z' key"

    def test_z_equals_u(self, solver):
        """Initial z equals u (identity initialisation)."""
        u = solver.estimated_image.astype(backend.xp.float64)
        state = solver._prior_init(u)
        np.testing.assert_array_equal(
            backend._to_numpy(state["z"]),
            backend._to_numpy(u),
        )

    def test_d_z_is_zero(self, solver):
        """Initial d_z is all zeros."""
        u = solver.estimated_image.astype(backend.xp.float64)
        state = solver._prior_init(u)
        d_z_np = backend._to_numpy(state["d_z"])
        assert float(np.max(np.abs(d_z_np))) == 0.0

    def test_z_is_copy_not_reference(self, solver):
        """z is an independent copy of u (mutating u doesn't change z)."""
        u = solver.estimated_image.astype(backend.xp.float64).copy()
        state = solver._prior_init(u)
        u_np_before = backend._to_numpy(state["z"]).copy()
        # Mutate u and verify state["z"] is unchanged
        u += 999.0
        u_np_after = backend._to_numpy(state["z"])
        np.testing.assert_array_equal(u_np_before, u_np_after)


# ══════════════════════════════════════════════════════════════════════════════
# 11. _prior_dual_update
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorDualUpdate:

    def test_d_z_increments_correctly(self, solver):
        """d_z += (u - z) after each call to _prior_dual_update."""
        u = solver.estimated_image.astype(backend.xp.float64).copy()
        state = solver._prior_init(u)

        # Perturb z so u - z is non-trivial
        state["z"] = state["z"] - 0.05

        d_z_before = backend._to_numpy(state["d_z"]).copy()
        u_np = backend._to_numpy(u)
        z_np = backend._to_numpy(state["z"])
        expected_increment = u_np - z_np

        solver._prior_dual_update(u, state)

        d_z_after = backend._to_numpy(state["d_z"])
        np.testing.assert_allclose(
            d_z_after,
            d_z_before + expected_increment,
            atol=1e-10,
        )

    def test_dual_update_cumulative(self, solver):
        """Two calls to _prior_dual_update accumulate correctly."""
        u = solver.estimated_image.astype(backend.xp.float64).copy()
        state = solver._prior_init(u)

        # First update: u - z
        solver._prior_dual_update(u, state)
        d_z_1 = backend._to_numpy(state["d_z"]).copy()

        # Second update with a different u
        u2 = u + 0.1
        solver._prior_dual_update(u2, state)
        d_z_2 = backend._to_numpy(state["d_z"])

        # Second d_z should equal d_z_1 + (u2 - z)
        z_np = backend._to_numpy(state["z"])
        u2_np = backend._to_numpy(u2)
        np.testing.assert_allclose(d_z_2, d_z_1 + (u2_np - z_np), atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 12. cost_history
# ══════════════════════════════════════════════════════════════════════════════

class TestCostHistory:

    def test_cost_history_populated(self, blurred, small_psf):
        """cost_history has num_iter + 1 entries on a full run."""
        num_iter = 5
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        solver.deblur(num_iter=num_iter, lambda_tv=0.01,
                      tol=0.0, min_iter=num_iter + 1)
        assert len(solver.cost_history) == num_iter + 1

    def test_cost_history_all_finite(self, blurred, small_psf):
        """All cost values must be finite."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        assert all(np.isfinite(c) for c in solver.cost_history)

    def test_cost_history_data_only(self, blurred, small_psf):
        """
        PnP cost_history contains only data fidelity (no TV term).

        Because the PnP state dict has "z"/"d_z" but not "w_h"/"w_w",
        _compute_admm_cost computes only the data term.  Verify costs
        are positive and finite.
        """
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        solver.deblur(num_iter=5, lambda_tv=0.01)
        for i, c in enumerate(solver.cost_history):
            assert c >= 0.0, f"Negative cost at index {i}: {c}"
            assert np.isfinite(c), f"Non-finite cost at index {i}: {c}"

    def test_cost_history_returns_copy(self, blurred, small_psf):
        """cost_history returns a fresh list each time."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        solver.deblur(num_iter=3, lambda_tv=0.01)
        assert solver.cost_history is not solver.cost_history


# ══════════════════════════════════════════════════════════════════════════════
# 13. Wrapper equivalence
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapper:

    def test_pnp_admm_deblur_matches_class(self, blurred, small_psf):
        """
        pnp_admm_deblur output matches direct class usage to float32 tolerance.

        BM3D may exhibit sub-ULP float32 differences between two independent
        calls (internal parallelism, FFTW plan caching).  Use allclose with
        a tight tolerance rather than exact equality.
        """
        common_kw = dict(rho_v=1.0, rho_z=1.0, nonneg=True)
        result_cls = PnPADMM(blurred, small_psf, **common_kw).deblur(
            num_iter=3, lambda_tv=0.01
        )
        result_fn = pnp_admm_deblur(
            blurred, small_psf, iters=3, lambda_tv=0.01, **common_kw
        )
        np.testing.assert_allclose(result_cls, result_fn, atol=1e-5, rtol=1e-5)

    def test_wrapper_kwarg_split(self, blurred, small_psf):
        """pnp_admm_deblur correctly routes init vs deblur kwargs."""
        result = pnp_admm_deblur(
            blurred, small_psf,
            iters=3, lambda_tv=0.01,
            # init kwargs
            rho_v=1.0, rho_z=2.0, sigma_scale=0.8, padding_scale=2.0,
            # deblur kwargs
            nonneg=True,
        )
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 14. _INIT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:

    def test_init_keys_contains_pnp_params(self):
        """_INIT_KEYS includes PnP-specific constructor parameters."""
        keys = PnPADMM._INIT_KEYS
        for expected in ("rho_z", "denoiser_profile", "sigma_scale"):
            assert expected in keys, f"Missing key: {expected}"

    def test_init_keys_includes_admm_keys(self):
        """_INIT_KEYS inherits all ADMMDeconv._INIT_KEYS."""
        assert ADMMDeconv._INIT_KEYS.issubset(PnPADMM._INIT_KEYS)

    def test_init_keys_includes_base_keys(self):
        """_INIT_KEYS inherits all DeconvBase._INIT_KEYS."""
        assert DeconvBase._INIT_KEYS.issubset(PnPADMM._INIT_KEYS)

    def test_deblur_params_not_in_init_keys(self):
        """deblur-only params (num_iter, lambda_tv, tol) are not in _INIT_KEYS."""
        keys = PnPADMM._INIT_KEYS
        for deblur_only in ("num_iter", "lambda_tv", "tol", "verbose"):
            assert deblur_only not in keys, (
                f"Unexpected deblur key in _INIT_KEYS: {deblur_only}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 15. Inheritance
# ══════════════════════════════════════════════════════════════════════════════

class TestInheritance:

    def test_is_subclass_of_admmdeconv(self):
        """PnPADMM is a subclass of ADMMDeconv."""
        assert issubclass(PnPADMM, ADMMDeconv)

    def test_is_subclass_of_deconvbase(self):
        """PnPADMM is a subclass of DeconvBase."""
        assert issubclass(PnPADMM, DeconvBase)

    def test_instance_of_admmdeconv(self, blurred, small_psf):
        """Constructed PnPADMM instance is an ADMMDeconv."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        assert isinstance(solver, ADMMDeconv)

    def test_instance_of_deconvbase(self, blurred, small_psf):
        """Constructed PnPADMM instance is a DeconvBase."""
        solver = PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
        assert isinstance(solver, DeconvBase)

    def test_has_bm3d_flag_true(self):
        """_HAS_BM3D is True when bm3d is installed (prerequisite for this test)."""
        assert _HAS_BM3D is True


# ══════════════════════════════════════════════════════════════════════════════
# 16. ImportError when bm3d absent
# ══════════════════════════════════════════════════════════════════════════════

class TestImportErrorPath:

    def test_import_error_raised_without_bm3d(self, blurred, small_psf, monkeypatch):
        """
        Verify that the ImportError path is reachable.

        Monkeypatch _HAS_BM3D to False in the pnp_admm module and confirm
        that constructing PnPADMM raises ImportError with the expected
        message.
        """
        import Reconstruction.pnp_admm as pnp_mod
        monkeypatch.setattr(pnp_mod, "_HAS_BM3D", False)
        with pytest.raises(ImportError, match="bm3d"):
            PnPADMM(blurred, small_psf, rho_v=1.0, rho_z=1.0)
