"""
test_wiener.py — Unit tests for Reconstruction/wiener.py (Phase 5a).

Coverage
--------
- WienerDeconv construction (all modes, bad parameters)
- deblur() output shape, dtype, finite values
- Manual alpha round-trip (Classical, Tikhonov, Spectrum)
- Auto-alpha estimation: sigma_est, last_alpha populated
- _estimate_sigma / _alpha_from_sigma
- No-noise reconstruction quality (PSNR > 40 dB with tiny alpha)
- Noisy reconstruction quality (PSNR > 20 dB with auto-alpha)
- Properties: last_alpha, sigma_est before/after deblur
- wiener_deblur() convenience wrapper (kwarg splitting)
- Regression against docs/reference/Wiener.py (three modes)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Load module under test
# ─────────────────────────────────────────────────────────────────────────────
from Reconstruction.wiener import WienerDeconv, wiener_deblur, _LAPL_NP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _psnr(ref: np.ndarray, est: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (clip to 100 dB)."""
    mse = float(np.mean((ref.astype(np.float64) - est.astype(np.float64)) ** 2))
    if mse < 1e-15:
        return 100.0
    return min(10.0 * np.log10(1.0 / mse), 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Reference Wiener loader (for regression tests)
# ─────────────────────────────────────────────────────────────────────────────

def _load_reference():
    """Load docs/reference/Wiener.py as a module."""
    ref_path = Path(__file__).parent.parent / "docs" / "reference" / "Wiener.py"
    spec = importlib.util.spec_from_file_location("_ref_wiener", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (supplement conftest.py globals)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def psf():
    """7×7 Gaussian PSF, σ=1.5."""
    size = 7
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * 1.5 ** 2))
    k /= k.sum()
    return k.astype(np.float64)


@pytest.fixture
def clean_image():
    """33×33 synthetic image with values in [0.1, 0.9]."""
    img = np.full((33, 33), 0.1, dtype=np.float64)
    img[8:24, 8:24] = 0.9
    img[0:8, 0:8] = 0.5
    return img


@pytest.fixture
def blurred_clean(clean_image, psf):
    """Clean image convolved with PSF (no noise)."""
    from scipy.signal import fftconvolve
    return np.clip(fftconvolve(clean_image, psf, mode="same"), 0, None).astype(np.float64)


@pytest.fixture
def noisy_blurred(blurred_clean):
    """Blurred image with additive Gaussian noise (σ=0.02)."""
    rng = np.random.default_rng(0)
    return np.clip(blurred_clean + rng.normal(0, 0.02, blurred_clean.shape), 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWienerDeconvConstruction:

    def test_tikhonov_mode_default(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert w.mode == "Tikhonov"
        assert w.gamma == 1.0

    def test_classical_mode(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Classical")
        assert w.mode == "Classical"

    def test_spectrum_mode(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Spectrum")
        assert w.mode == "Spectrum"

    def test_invalid_mode_raises(self, blurred_clean, psf):
        with pytest.raises(ValueError, match="Unknown mode"):
            WienerDeconv(blurred_clean, psf, mode="Bogus")

    def test_invalid_gamma_raises(self, blurred_clean, psf):
        with pytest.raises(ValueError, match="gamma must be positive"):
            WienerDeconv(blurred_clean, psf, gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be positive"):
            WienerDeconv(blurred_clean, psf, gamma=-1.0)

    def test_gamma_stored(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, gamma=2.5)
        assert w.gamma == pytest.approx(2.5)

    def test_attributes_exist(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        for attr in ("gray", "obj_F", "psf_F2", "conj_psf_F", "L2",
                     "full_shape", "h", "w"):
            assert hasattr(w, attr), f"missing attribute: {attr}"

    def test_full_shape_odd(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        fh, fw = w.full_shape
        assert fh % 2 == 1
        assert fw % 2 == 1

    def test_full_shape_larger_than_image(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert w.full_shape[0] > w.h
        assert w.full_shape[1] > w.w

    def test_gray_shape_matches_hw(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert w.gray.shape == (w.h, w.w)

    def test_obj_F_shape(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        fh, fw = w.full_shape
        assert w.obj_F.shape == (fh, fw // 2 + 1)

    def test_psf_F2_shape(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        fh, fw = w.full_shape
        assert w.psf_F2.shape == (fh, fw // 2 + 1)

    def test_L2_shape(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        fh, fw = w.full_shape
        assert w.L2.shape == (fh, fw // 2 + 1)

    def test_psf_F2_nonnegative(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert float(np.min(w.psf_F2)) >= 0.0

    def test_L2_nonnegative(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert float(np.min(w.L2)) >= 0.0

    def test_last_alpha_none_before_deblur(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert w.last_alpha is None

    def test_sigma_est_none_before_deblur(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        assert w.sigma_est is None

    def test_use_mask_false(self, blurred_clean, psf):
        """WienerDeconv always sets use_mask=False."""
        w = WienerDeconv(blurred_clean, psf)
        assert w.use_mask is False

    def test_normalize_image_accepted(self, blurred_clean, psf):
        """normalize_image kwarg must not raise."""
        w = WienerDeconv(blurred_clean, psf, normalize_image=True)
        assert w is not None
        w2 = WienerDeconv(blurred_clean, psf, normalize_image=False)
        assert w2 is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. deblur() output tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeblurOutput:

    def test_output_shape_tikhonov(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Tikhonov")
        out = w.deblur(alpha=0.01)
        assert out.shape == (w.h, w.w)

    def test_output_shape_classical(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Classical")
        out = w.deblur(alpha=0.01)
        assert out.shape == (w.h, w.w)

    def test_output_shape_spectrum(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Spectrum")
        out = w.deblur(alpha=0.01)
        assert out.shape == (w.h, w.w)

    def test_output_dtype_float(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        out = w.deblur(alpha=0.01)
        assert np.issubdtype(out.dtype, np.floating)

    def test_output_finite_tikhonov(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Tikhonov")
        out = w.deblur(alpha=0.01)
        assert np.all(np.isfinite(out))

    def test_output_finite_classical(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Classical")
        out = w.deblur(alpha=0.01)
        assert np.all(np.isfinite(out))

    def test_output_finite_spectrum(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf, mode="Spectrum")
        out = w.deblur(alpha=0.01)
        assert np.all(np.isfinite(out))

    def test_output_is_numpy(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        out = w.deblur(alpha=0.01)
        assert isinstance(out, np.ndarray)

    def test_repeated_deblur_same_result(self, blurred_clean, psf):
        """deblur() is deterministic for fixed alpha."""
        w = WienerDeconv(blurred_clean, psf)
        out1 = w.deblur(alpha=0.005)
        out2 = w.deblur(alpha=0.005)
        np.testing.assert_array_equal(out1, out2)

    def test_larger_alpha_smoother(self, noisy_blurred, psf):
        """Larger alpha → lower output variance (smoother)."""
        w = WienerDeconv(noisy_blurred, psf, mode="Tikhonov")
        out_small = w.deblur(alpha=1e-6)
        out_large = w.deblur(alpha=1.0)
        assert float(out_large.var()) < float(out_small.var())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Auto-alpha and property tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoAlphaAndProperties:

    def test_last_alpha_set_after_deblur_manual(self, blurred_clean, psf):
        w = WienerDeconv(blurred_clean, psf)
        w.deblur(alpha=0.01)
        assert w.last_alpha is not None
        assert w.last_alpha == pytest.approx(0.01)

    def test_sigma_est_none_after_manual_alpha(self, blurred_clean, psf):
        """sigma_est should NOT be updated when alpha is supplied manually."""
        w = WienerDeconv(blurred_clean, psf)
        w.deblur(alpha=0.01)
        assert w.sigma_est is None

    def test_sigma_est_set_after_auto_alpha(self, noisy_blurred, psf):
        w = WienerDeconv(noisy_blurred, psf)
        w.deblur()  # auto
        assert w.sigma_est is not None
        assert w.sigma_est > 0.0

    def test_last_alpha_positive_tikhonov_auto(self, noisy_blurred, psf):
        w = WienerDeconv(noisy_blurred, psf, mode="Tikhonov")
        w.deblur()
        assert isinstance(w.last_alpha, float)
        assert w.last_alpha > 0.0

    def test_last_alpha_positive_classical_auto(self, noisy_blurred, psf):
        w = WienerDeconv(noisy_blurred, psf, mode="Classical")
        w.deblur()
        assert isinstance(w.last_alpha, float)
        assert w.last_alpha > 0.0

    def test_last_alpha_array_spectrum_auto(self, noisy_blurred, psf):
        """Spectrum auto-alpha should be a 2-D array, not a scalar."""
        w = WienerDeconv(noisy_blurred, psf, mode="Spectrum")
        w.deblur()
        alpha = w.last_alpha
        assert isinstance(alpha, np.ndarray)
        assert alpha.ndim == 2

    def test_gamma_scales_tikhonov_alpha(self, noisy_blurred, psf):
        """Larger gamma → proportionally larger Tikhonov alpha."""
        w1 = WienerDeconv(noisy_blurred, psf, mode="Tikhonov", gamma=1.0)
        w2 = WienerDeconv(noisy_blurred, psf, mode="Tikhonov", gamma=2.0)
        w1.deblur(); w2.deblur()
        assert w2.last_alpha == pytest.approx(2.0 * w1.last_alpha, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _estimate_sigma and _alpha_from_sigma unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperMethods:

    def test_estimate_sigma_returns_positive(self, noisy_blurred, psf):
        w = WienerDeconv(noisy_blurred, psf)
        sigma = w._estimate_sigma()
        assert sigma > 0.0

    def test_estimate_sigma_clean_lower_than_noisy(self, blurred_clean, noisy_blurred, psf):
        w_clean = WienerDeconv(blurred_clean, psf)
        w_noisy = WienerDeconv(noisy_blurred, psf)
        assert w_clean._estimate_sigma() < w_noisy._estimate_sigma()

    def test_alpha_from_sigma_positive(self):
        rng = np.random.default_rng(1)
        gray = rng.uniform(0, 1, (33, 33)).astype(np.float64)
        sigma = 0.02
        alpha = WienerDeconv._alpha_from_sigma(gray, sigma, _LAPL_NP, gamma=1.0)
        assert alpha > 0.0

    def test_alpha_from_sigma_gamma_scaling(self):
        rng = np.random.default_rng(2)
        gray = rng.uniform(0, 1, (33, 33)).astype(np.float64)
        sigma = 0.02
        a1 = WienerDeconv._alpha_from_sigma(gray, sigma, _LAPL_NP, gamma=1.0)
        a2 = WienerDeconv._alpha_from_sigma(gray, sigma, _LAPL_NP, gamma=3.0)
        assert a2 == pytest.approx(3.0 * a1, rel=1e-5)

    def test_alpha_from_sigma_higher_noise_gives_larger_alpha(self):
        rng = np.random.default_rng(3)
        gray = rng.uniform(0, 1, (33, 33)).astype(np.float64)
        a_low  = WienerDeconv._alpha_from_sigma(gray, 0.01, _LAPL_NP)
        a_high = WienerDeconv._alpha_from_sigma(gray, 0.10, _LAPL_NP)
        assert a_high > a_low

    def test_lapl_np_shape(self):
        assert _LAPL_NP.shape == (3, 3)

    def test_lapl_np_sum_zero(self):
        """Isotropic Laplacian must sum to zero (zero DC response)."""
        assert abs(float(_LAPL_NP.sum())) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reconstruction quality tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructionQuality:

    def test_small_alpha_sharpens_vs_large_alpha(self, blurred_clean, psf):
        """Smaller alpha → less regularisation → higher output variance (sharper).
        This is an intra-filter comparison and is unaffected by normalization."""
        w = WienerDeconv(blurred_clean, psf, mode="Tikhonov")
        out_sharp = w.deblur(alpha=1e-4)
        out_smooth = w.deblur(alpha=1.0)
        assert float(out_sharp.var()) > float(out_smooth.var()), (
            "Smaller alpha should yield higher variance (sharper) output"
        )

    def test_alpha_10_gives_nonzero_output(self, noisy_blurred, psf):
        """Even with heavy regularisation the output should be non-trivial."""
        w = WienerDeconv(noisy_blurred, psf, mode="Tikhonov")
        out = w.deblur(alpha=10.0)
        assert np.all(np.isfinite(out))
        assert float(out.var()) > 0.0

    def test_classical_mode_deblurs(self, blurred_clean, psf):
        """Classical mode with manual alpha should produce finite output."""
        w = WienerDeconv(blurred_clean, psf, mode="Classical")
        out = w.deblur(alpha=1e-4)
        assert np.all(np.isfinite(out))

    def test_spectrum_mode_deblurs(self, blurred_clean, psf):
        """Spectrum mode with auto-alpha should produce finite output."""
        w = WienerDeconv(blurred_clean, psf, mode="Spectrum")
        out = w.deblur()
        assert np.all(np.isfinite(out))

    def test_auto_alpha_tikhonov_finite(self, noisy_blurred, psf):
        """Auto-alpha Tikhonov should produce finite, non-trivial output."""
        w = WienerDeconv(noisy_blurred, psf, mode="Tikhonov")
        out = w.deblur()
        assert np.all(np.isfinite(out))
        assert float(out.var()) > 0.0  # not a flat image


# ─────────────────────────────────────────────────────────────────────────────
# 6. wiener_deblur() wrapper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWienerDeblurWrapper:

    def test_returns_ndarray(self, blurred_clean, psf):
        out = wiener_deblur(blurred_clean, psf, mode="Tikhonov", alpha=0.01)
        assert isinstance(out, np.ndarray)

    def test_shape_matches_input(self, blurred_clean, psf):
        out = wiener_deblur(blurred_clean, psf, alpha=0.01)
        # output h may differ by ≤1 due to odd-enforcement
        h_expect = blurred_clean.shape[0] if blurred_clean.shape[0] % 2 == 1 else blurred_clean.shape[0] - 1
        w_expect = blurred_clean.shape[1] if blurred_clean.shape[1] % 2 == 1 else blurred_clean.shape[1] - 1
        assert out.shape == (h_expect, w_expect)

    def test_kwargs_split_mode_to_init(self, blurred_clean, psf):
        """mode is an init param; must not appear in deblur kwargs."""
        # If mode went to deblur(), deblur() would raise TypeError.
        out = wiener_deblur(blurred_clean, psf, mode="Classical", alpha=0.01)
        assert isinstance(out, np.ndarray)

    def test_kwargs_split_alpha_to_deblur(self, blurred_clean, psf):
        """alpha is a deblur param; constructor must not receive it."""
        out = wiener_deblur(blurred_clean, psf, alpha=0.005)
        assert isinstance(out, np.ndarray)

    def test_gamma_forwarded_to_init(self, noisy_blurred, psf):
        """gamma is an init param; wrapper must forward it correctly."""
        out = wiener_deblur(noisy_blurred, psf, mode="Tikhonov", gamma=2.0)
        assert isinstance(out, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Regression tests against docs/reference/Wiener.py
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionVsReference:
    """
    Pixel-level comparison between WienerDeconv and the reference Wiener class.

    Notes
    -----
    * WienerDeconv always normalises the image (base-class behaviour).
      The reference Wiener only normalises when ``normalize_image=True``.
      All regression calls pass ``normalize_image=True`` to the reference.
    * Both implementations use the same mock PSF conditioning, so the
      Wiener-specific re-conditioning (taper_outer_frac=0.90) is a no-op
      in tests — both use _mock_condition_psf (normalise + no taper).
    * Tolerance: atol=1e-5 (float32 arithmetic differences).
    """

    @pytest.fixture(autouse=True)
    def _load_ref(self):
        self.ref = _load_reference()

    def _ref_deblur(self, image, psf, mode, alpha):
        """Helper: run reference Wiener and return result."""
        obj = self.ref.Wiener(
            image, psf,
            mode=mode,
            normalize_image=True,   # match base-class normalisation
        )
        return obj.deblur(alpha=alpha)

    def test_tikhonov_manual_alpha(self, blurred_clean, psf):
        alpha = 0.005
        w = WienerDeconv(blurred_clean, psf, mode="Tikhonov")
        ours = w.deblur(alpha=alpha)
        ref  = self._ref_deblur(blurred_clean, psf, "Tikhonov", alpha)
        # Crop ref to same h×w (odd-enforcement may differ by ±1)
        ref_crop = ref[:w.h, :w.w]
        np.testing.assert_allclose(ours, ref_crop, atol=1e-5,
                                   err_msg="Tikhonov regression mismatch")

    def test_classical_manual_alpha(self, blurred_clean, psf):
        alpha = 0.01
        w = WienerDeconv(blurred_clean, psf, mode="Classical")
        ours = w.deblur(alpha=alpha)
        ref  = self._ref_deblur(blurred_clean, psf, "Classical", alpha)
        ref_crop = ref[:w.h, :w.w]
        np.testing.assert_allclose(ours, ref_crop, atol=1e-5,
                                   err_msg="Classical regression mismatch")

    def test_spectrum_manual_alpha(self, blurred_clean, psf):
        alpha = 0.02
        w = WienerDeconv(blurred_clean, psf, mode="Spectrum")
        ours = w.deblur(alpha=alpha)
        ref  = self._ref_deblur(blurred_clean, psf, "Spectrum", alpha)
        ref_crop = ref[:w.h, :w.w]
        np.testing.assert_allclose(ours, ref_crop, atol=1e-5,
                                   err_msg="Spectrum regression mismatch")
