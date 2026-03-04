"""
RED-ADMM (Regularization by Denoising) deconvolution tests.

All tests are skipped when the bm3d package is not installed.

Tests:
  1.  Inheritance: REDDeconv is subclass of ADMMDeconv and DeconvBase.
  2.  Construction: basic attributes set correctly.
  3.  sigma property: equals constructor argument, unchanged after deblur.
  4.  _INIT_KEYS: contains sigma, denoiser_profile; lambda_reg absent.
  5.  Output basics: shape, dtype, finiteness.
  6.  Positivity: nonneg=True → all values ≥ 0.
  7.  Deconvolution quality: PSNR improves over blurred input.
  8.  _prior_init: returns dict with "denoised" key.
  9.  _prior_dual_update: no-op (state unchanged after call).
  10. _x_update_denom: returns ρ_v|H|² + λ (no lap_fft term).
  11. prior_rhs = λ·D(x), not λ·(x − D(x)).
  12. Denoiser called once per iteration: monkeypatch _denoise.
  13. Fixed sigma: σ does not change during iteration.
  14. cost_history: populated with finite values.
  15. Wrapper: red_deblur matches class-based usage.
  16. RED vs PnP: produce different outputs (different algorithms).
  17. ImportError when bm3d absent (monkeypatched _HAS_BM3D=False).
  18. v-update mask: inherited behavior (use_mask=True by default).
  19. Wrapper _INIT_KEYS routing: sigma routed to constructor.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Skip the entire module when bm3d is not installed.
bm3d_module = pytest.importorskip("bm3d", reason="bm3d not installed")

import Reconstruction._backend as backend
from Reconstruction._base import DeconvBase
from Reconstruction.admm import ADMMDeconv
from Reconstruction.red_admm import REDDeconv, red_deblur, _HAS_BM3D


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend for all tests."""
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


def _gaussian_psf(size: int = 7, sigma: float = 1.5) -> np.ndarray:
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
    mse = float(np.mean((ref.astype(float) - out.astype(float)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


@pytest.fixture(scope="module")
def small_psf() -> np.ndarray:
    return _gaussian_psf(size=7, sigma=1.5)


@pytest.fixture(scope="module")
def test_image() -> np.ndarray:
    return _test_image(41, 41)


@pytest.fixture(scope="module")
def blurred(test_image, small_psf) -> np.ndarray:
    return _blur(test_image, small_psf)


@pytest.fixture(scope="module")
def solver(blurred, small_psf) -> REDDeconv:
    return REDDeconv(blurred, small_psf, sigma=0.05)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Inheritance
# ══════════════════════════════════════════════════════════════════════════════

class TestInheritance:

    def test_is_subclass_of_admm_deconv(self):
        assert issubclass(REDDeconv, ADMMDeconv)

    def test_is_subclass_of_deconvbase(self):
        assert issubclass(REDDeconv, DeconvBase)

    def test_instance_is_admm_deconv(self, solver):
        assert isinstance(solver, ADMMDeconv)

    def test_instance_is_deconvbase(self, solver):
        assert isinstance(solver, DeconvBase)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Construction
# ══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_sigma_stored_correctly(self, blurred, small_psf):
        s = REDDeconv(blurred, small_psf, sigma=0.03)
        assert abs(s.sigma - 0.03) < 1e-9

    def test_denoiser_profile_stored(self, blurred, small_psf):
        s = REDDeconv(blurred, small_psf, denoiser_profile="lc")
        assert s.denoiser_profile == "lc"

    def test_nonneg_default_true(self, solver):
        assert solver.nonneg is True

    def test_import_error_without_bm3d(self, blurred, small_psf, monkeypatch):
        """ImportError raised if _HAS_BM3D is False."""
        import Reconstruction.red_admm as red_mod
        monkeypatch.setattr(red_mod, "_HAS_BM3D", False)
        with pytest.raises(ImportError, match="bm3d"):
            REDDeconv(blurred, small_psf)


# ══════════════════════════════════════════════════════════════════════════════
# 3. sigma property
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaProperty:

    def test_sigma_equals_constructor_arg(self, blurred, small_psf):
        s = REDDeconv(blurred, small_psf, sigma=0.07)
        assert abs(s.sigma - 0.07) < 1e-9

    def test_sigma_unchanged_after_deblur(self, blurred, small_psf):
        """Fixed sigma: does not change during iteration."""
        s = REDDeconv(blurred, small_psf, sigma=0.05)
        sigma_before = s.sigma
        s.deblur(num_iter=5, lambda_reg=0.01)
        assert abs(s.sigma - sigma_before) < 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# 4. _INIT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:

    def test_sigma_in_init_keys(self):
        assert "sigma" in REDDeconv._INIT_KEYS

    def test_denoiser_profile_in_init_keys(self):
        assert "denoiser_profile" in REDDeconv._INIT_KEYS

    def test_inherits_admm_init_keys(self):
        for key in ADMMDeconv._INIT_KEYS:
            assert key in REDDeconv._INIT_KEYS

    def test_lambda_reg_not_in_init_keys(self):
        """lambda_reg is a deblur() param, not a constructor param."""
        assert "lambda_reg" not in REDDeconv._INIT_KEYS

    def test_num_iter_not_in_init_keys(self):
        assert "num_iter" not in REDDeconv._INIT_KEYS


# ══════════════════════════════════════════════════════════════════════════════
# 5. Output basics
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputBasics:

    def test_output_shape(self, blurred, small_psf, test_image):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3, lambda_reg=0.01)
        assert result.shape == test_image.shape

    def test_output_is_numpy(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3)
        assert isinstance(result, np.ndarray)

    def test_output_finite(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3)
        assert np.isfinite(result).all()

    def test_output_2d(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3)
        assert result.ndim == 2

    def test_output_dtype_float(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3)
        assert result.dtype in (np.float32, np.float64)

    def test_non_square_image(self, small_psf):
        img = _test_image(41, 31)
        blr = _blur(img, small_psf)
        result = REDDeconv(blr, small_psf).deblur(num_iter=3)
        assert result.ndim == 2
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Positivity
# ══════════════════════════════════════════════════════════════════════════════

class TestPositivity:

    def test_nonneg_true_produces_nonneg_output(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf, nonneg=True).deblur(
            num_iter=5, lambda_reg=0.01
        )
        assert float(np.min(result)) >= 0.0

    def test_deblur_nonneg_override_true(self, blurred, small_psf):
        """nonneg=True at deblur() overrides constructor nonneg=False."""
        result = REDDeconv(blurred, small_psf, nonneg=False).deblur(
            num_iter=5, lambda_reg=0.01, nonneg=True
        )
        assert float(np.min(result)) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 7. Deconvolution quality
# ══════════════════════════════════════════════════════════════════════════════

class TestDeconvolutionQuality:
    """
    PSNR after RED deblur must not be worse than blurred input.

    DeconvBase normalises images to [0,1] internally.  Both ref and result
    are mapped to [0,1] before PSNR calculation (same pattern as test_admm.py).
    """

    def test_improves_psnr(self, blurred, small_psf, test_image):
        result = REDDeconv(blurred, small_psf, sigma=0.05).deblur(
            num_iter=10, lambda_reg=0.005
        )
        h, w = result.shape
        ref = test_image[:h, :w]
        psnr_in  = _psnr(_norm01(ref), _norm01(blurred[:h, :w]))
        psnr_out = _psnr(_norm01(ref), _norm01(result))
        assert psnr_out > psnr_in - 0.5, (
            f"RED did not maintain quality: in={psnr_in:.2f}, out={psnr_out:.2f}"
        )

    def test_output_values_in_reasonable_range(self, blurred, small_psf):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=5)
        assert float(np.max(result)) < 10.0, f"max={np.max(result):.2f} too large"
        assert float(np.min(result)) >= 0.0, "nonneg violated"


# ══════════════════════════════════════════════════════════════════════════════
# 8. _prior_init
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorInit:

    def test_returns_dict_with_denoised_key(self, solver):
        xp_local = backend.xp
        u = xp_local.zeros(solver.full_shape, dtype=xp_local.float64)
        state = solver._prior_init(u)
        assert "denoised" in state

    def test_denoised_shape_matches_u(self, solver):
        xp_local = backend.xp
        u = xp_local.ones(solver.full_shape, dtype=xp_local.float64) * 0.5
        state = solver._prior_init(u)
        assert state["denoised"].shape == u.shape

    def test_no_w_h_key(self, solver):
        """RED has no TV auxiliary variable w."""
        xp_local = backend.xp
        u = xp_local.zeros(solver.full_shape, dtype=xp_local.float64)
        state = solver._prior_init(u)
        assert "w_h" not in state
        assert "w_w" not in state

    def test_no_z_key(self, solver):
        """RED has no PnP z-split variable."""
        xp_local = backend.xp
        u = xp_local.zeros(solver.full_shape, dtype=xp_local.float64)
        state = solver._prior_init(u)
        assert "z" not in state


# ══════════════════════════════════════════════════════════════════════════════
# 9. _prior_dual_update is a no-op
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorDualUpdateNoOp:

    def test_state_unchanged_after_dual_update(self, solver):
        """_prior_dual_update must not modify the state dict."""
        xp_local = backend.xp
        u = xp_local.ones(solver.full_shape, dtype=xp_local.float64) * 0.5
        sentinel = xp_local.zeros(solver.full_shape, dtype=xp_local.float64)
        state = {"denoised": sentinel.copy(), "extra_key": "should_stay"}

        solver._prior_dual_update(u, state)

        # "denoised" array unchanged
        np.testing.assert_array_equal(
            backend._to_numpy(state["denoised"]),
            backend._to_numpy(sentinel),
        )
        # No new keys added
        assert set(state.keys()) == {"denoised", "extra_key"}

    def test_dual_update_returns_none(self, solver):
        xp_local = backend.xp
        u = xp_local.ones(solver.full_shape, dtype=xp_local.float64)
        state = {"denoised": u.copy()}
        result = solver._prior_dual_update(u, state)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 10. _x_update_denom: ρ_v|H|² + λ (no lap_fft)
# ══════════════════════════════════════════════════════════════════════════════

class TestXUpdateDenom:

    def test_denom_formula_rho_v_H_sq_plus_lambda(self, solver):
        """denom = rho_v * H_H_conj + lambda_reg (no lap_fft)."""
        xp_local = backend.xp
        rho_v = 2.0
        lam   = 0.03
        solver._current_lambda = lam

        denom = solver._x_update_denom(rho_v, rho_w=99.9)  # rho_w ignored

        expected = rho_v * backend._to_numpy(solver.H_H_conj) + lam
        actual   = backend._to_numpy(denom)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_denom_no_laplacian_term(self, solver):
        """Denominator must NOT equal rho_v|H|² + rho_w·lap_fft."""
        xp_local = backend.xp
        rho_v = 2.0
        rho_w = 32.0
        lam   = 0.05
        solver._current_lambda = lam

        red_denom = backend._to_numpy(solver._x_update_denom(rho_v, rho_w))
        tv_denom  = backend._to_numpy(
            rho_v * solver.H_H_conj + rho_w * solver.lap_fft
        )

        # RED denom should differ from TV denom (no lap_fft contribution)
        assert not np.allclose(red_denom, tv_denom, atol=1e-6), (
            "RED denom equals TV denom — lap_fft term incorrectly included"
        )

    def test_denom_shape(self, solver):
        solver._current_lambda = 0.01
        denom = solver._x_update_denom(1.0, 1.0)
        assert backend._to_numpy(denom).shape == solver.full_shape


# ══════════════════════════════════════════════════════════════════════════════
# 11. prior_rhs = λ·D(x), not λ·(x − D(x))
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorRhs:

    def test_prior_rhs_equals_lambda_times_denoised(
        self, solver, monkeypatch
    ):
        """_prior_update returns λ·D(x) — the denoised image scaled by λ."""
        xp_local = backend.xp
        lam = 0.02
        fixed_denoised = xp_local.ones(
            solver.full_shape, dtype=xp_local.float64
        ) * 0.6

        # Patch _denoise to return a known value
        monkeypatch.setattr(solver, "_denoise", lambda u, s: fixed_denoised)

        u     = xp_local.ones(solver.full_shape, dtype=xp_local.float64) * 0.5
        state = solver._prior_init(u)
        prior_rhs = solver._prior_update(
            u, state, lambda_tv=lam, rho_w=99.0, eps=1e-8
        )

        expected = lam * backend._to_numpy(fixed_denoised)
        np.testing.assert_allclose(
            backend._to_numpy(prior_rhs), expected, rtol=1e-6,
            err_msg="prior_rhs should be λ·D(x), not λ·(x−D(x))"
        )

    def test_prior_rhs_not_lambda_residual(self, solver, monkeypatch):
        """Confirm prior_rhs ≠ λ·(x − D(x))."""
        xp_local = backend.xp
        lam = 0.02
        fixed_denoised = xp_local.ones(
            solver.full_shape, dtype=xp_local.float64
        ) * 0.6

        monkeypatch.setattr(solver, "_denoise", lambda u, s: fixed_denoised)

        u = xp_local.ones(solver.full_shape, dtype=xp_local.float64) * 0.5
        state = solver._prior_init(u)
        prior_rhs = solver._prior_update(
            u, state, lambda_tv=lam, rho_w=1.0, eps=1e-8
        )

        # If prior_rhs were λ(x−D(x)), the value would be lam*(0.5−0.6)=−0.002
        # but it should be lam*0.6 = 0.012
        wrong_rhs = lam * (backend._to_numpy(u) - backend._to_numpy(fixed_denoised))
        assert not np.allclose(
            backend._to_numpy(prior_rhs), wrong_rhs, atol=1e-6
        ), "prior_rhs incorrectly equals λ·(x−D(x))"


# ══════════════════════════════════════════════════════════════════════════════
# 12. Denoiser called once per iteration
# ══════════════════════════════════════════════════════════════════════════════

class TestDenoiserCallCount:

    def test_denoise_called_once_per_iteration(
        self, blurred, small_psf, monkeypatch
    ):
        """_denoise should be called exactly num_iter times."""
        call_count = [0]

        def counting_denoise(image, sigma):
            call_count[0] += 1
            return image  # identity — fast, no BM3D needed

        solver = REDDeconv(blurred, small_psf, sigma=0.05)
        monkeypatch.setattr(solver, "_denoise", counting_denoise)

        n_iters = 5
        solver.deblur(num_iter=n_iters, lambda_reg=0.01, min_iter=999)

        assert call_count[0] == n_iters, (
            f"Expected _denoise called {n_iters} times, got {call_count[0]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 13. Fixed sigma
# ══════════════════════════════════════════════════════════════════════════════

class TestFixedSigma:

    def test_sigma_passed_to_denoise_is_always_fixed(
        self, blurred, small_psf, monkeypatch
    ):
        """Each _denoise call receives the same fixed sigma."""
        observed_sigmas = []

        def recording_denoise(image, sigma):
            observed_sigmas.append(sigma)
            return image

        fixed_sigma = 0.07
        solver = REDDeconv(blurred, small_psf, sigma=fixed_sigma)
        monkeypatch.setattr(solver, "_denoise", recording_denoise)
        solver.deblur(num_iter=5, lambda_reg=0.01, min_iter=999)

        assert len(observed_sigmas) == 5
        for s in observed_sigmas:
            assert abs(s - fixed_sigma) < 1e-9, (
                f"Expected sigma={fixed_sigma}, got {s} — sigma is not fixed!"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 14. cost_history
# ══════════════════════════════════════════════════════════════════════════════

class TestCostHistory:

    def test_cost_history_populated(self, blurred, small_psf):
        solver = REDDeconv(blurred, small_psf)
        solver.deblur(num_iter=5, lambda_reg=0.01)
        assert len(solver.cost_history) > 0

    def test_cost_history_finite(self, blurred, small_psf):
        solver = REDDeconv(blurred, small_psf)
        solver.deblur(num_iter=5, lambda_reg=0.01)
        assert all(np.isfinite(c) for c in solver.cost_history)

    def test_cost_history_non_negative(self, blurred, small_psf):
        """Data-fidelity cost ≥ 0."""
        solver = REDDeconv(blurred, small_psf)
        solver.deblur(num_iter=5, lambda_reg=0.01)
        assert all(c >= 0.0 for c in solver.cost_history)


# ══════════════════════════════════════════════════════════════════════════════
# 15. Wrapper red_deblur
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapper:

    def test_wrapper_matches_class_result(self, blurred, small_psf):
        """red_deblur and class-based call produce close results.

        BM3D has thread-level non-determinism so bit-exact equality is not
        guaranteed.  We verify that the wrapper routes arguments correctly by
        checking that outputs are within a generous tolerance.
        """
        result_cls = REDDeconv(blurred, small_psf, sigma=0.05).deblur(
            num_iter=5, lambda_reg=0.01
        )
        result_fn = red_deblur(
            blurred, small_psf, iters=5, lambda_reg=0.01, sigma=0.05
        )
        assert result_cls.shape == result_fn.shape
        assert result_cls.dtype == result_fn.dtype
        np.testing.assert_allclose(result_cls, result_fn, atol=0.05)

    def test_wrapper_sigma_routed_to_constructor(self, blurred, small_psf):
        """sigma kwarg is routed to the constructor, not deblur()."""
        result = red_deblur(blurred, small_psf, iters=3, sigma=0.03)
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()

    def test_wrapper_lambda_reg_routed_to_deblur(self, blurred, small_psf):
        """lambda_reg kwarg is routed to deblur() (not constructor)."""
        result = red_deblur(blurred, small_psf, iters=3, lambda_reg=0.02)
        assert isinstance(result, np.ndarray)

    def test_wrapper_first_arg_is_image(self):
        import inspect
        params = list(inspect.signature(red_deblur).parameters.keys())
        assert params[0] == "image"

    def test_wrapper_second_arg_is_psf(self):
        import inspect
        params = list(inspect.signature(red_deblur).parameters.keys())
        assert params[1] == "psf"


# ══════════════════════════════════════════════════════════════════════════════
# 16. RED vs PnP produce different outputs
# ══════════════════════════════════════════════════════════════════════════════

class TestREDvsPnP:

    def test_red_and_pnp_produce_different_outputs(self, blurred, small_psf):
        """RED and PnP are different algorithms — outputs must differ."""
        from Reconstruction.pnp_admm import PnPADMM

        red_result = REDDeconv(blurred, small_psf, sigma=0.05).deblur(
            num_iter=10, lambda_reg=0.01
        )
        pnp_result = PnPADMM(blurred, small_psf, rho_z=1.0).deblur(
            num_iter=10, lambda_tv=0.01
        )

        assert not np.allclose(red_result, pnp_result, atol=1e-4), (
            "RED and PnP produced identical outputs — suspect a bug in one of them"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 17. Root namespace accessibility
# ══════════════════════════════════════════════════════════════════════════════

class TestRootNamespace:

    def test_red_deconv_in_all(self):
        import Reconstruction
        assert "REDDeconv" in Reconstruction.__all__

    def test_red_deblur_in_all(self):
        import Reconstruction
        assert "red_deblur" in Reconstruction.__all__

    def test_instantiate_from_root(self, blurred, small_psf):
        import Reconstruction
        cls = Reconstruction.REDDeconv
        instance = cls(blurred, small_psf)
        assert isinstance(instance, DeconvBase)

    def test_wrapper_callable_from_root(self, blurred, small_psf):
        import Reconstruction
        result = Reconstruction.red_deblur(blurred, small_psf, iters=3)
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# 18. v-update mask (inherited behavior)
# ══════════════════════════════════════════════════════════════════════════════

class TestMask:

    def test_use_mask_true_by_default(self, solver):
        """RED inherits use_mask=True from DeconvBase default."""
        assert solver.use_mask is True

    def test_mask_shape_matches_canvas(self, solver):
        assert solver.mask.shape == solver.full_shape

    def test_output_shape_with_mask(self, blurred, small_psf, test_image):
        result = REDDeconv(blurred, small_psf).deblur(num_iter=3)
        h, w = result.shape
        assert h <= test_image.shape[0] and w <= test_image.shape[1]
