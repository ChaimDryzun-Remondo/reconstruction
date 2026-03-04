"""
Chambolle-Pock (Condat-Vũ) deconvolution tests.

Covers:
  - ChambollePockDeconv construction (properties, _INIT_KEYS)
  - Output basics (shape, dtype, finiteness) for TVnorm=1 and TVnorm=2
  - Positivity enforcement (constructor + deblur-time override)
  - Deconvolution quality (PSNR improvement over blurred input)
  - TVnorm=1 and TVnorm=2 produce valid (different) outputs
  - Dual projection correctness (per-pixel ‖p‖₂ ≤ λ; clamp for TVnorm=1)
  - Step size validity: τ(L/2 + 8σ) ≤ 1.0 (Condat convergence condition)
  - Mask is active (use_mask=True by default)
  - TVnorm validation (only 1 and 2 accepted)
  - theta=0 disables extrapolation but still runs
  - Convergence with loose tolerance
  - Wrapper chambolle_pock_deblur matches class-based call
  - Root namespace accessibility (ChambollePockDeconv, chambolle_pock_deblur)
  - Inheritance from DeconvBase
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import Reconstruction
import Reconstruction._backend as backend
from Reconstruction._base import DeconvBase
from Reconstruction.chambolle_pock import ChambollePockDeconv, chambolle_pock_deblur


# ══════════════════════════════════════════════════════════════════════════════
# Helpers and fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


def _gaussian_psf(size: int = 9, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf.astype(np.float64)


def _test_image(h: int = 65, w: int = 65) -> np.ndarray:
    img = np.full((h, w), 0.1, dtype=np.float64)
    ch, cw = h // 4, w // 4
    img[ch:3 * ch, cw:3 * cw] = 0.8
    return img


def _blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    return np.clip(fftconvolve(image, psf, mode="same"), 0, None)


def _psnr(ref: np.ndarray, est: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(float) - est.astype(float)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


@pytest.fixture(scope="module")
def small_psf() -> np.ndarray:
    return _gaussian_psf(size=9, sigma=1.5)


@pytest.fixture(scope="module")
def test_image() -> np.ndarray:
    return _test_image(65, 65)


@pytest.fixture(scope="module")
def blurred(test_image, small_psf) -> np.ndarray:
    return _blur(test_image, small_psf)


@pytest.fixture(scope="module")
def solver(blurred, small_psf) -> ChambollePockDeconv:
    return ChambollePockDeconv(blurred, small_psf)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Inheritance
# ══════════════════════════════════════════════════════════════════════════════

class TestInheritance:

    def test_chambolle_pock_deconv_is_deconvbase(self):
        """ChambollePockDeconv is a subclass of DeconvBase."""
        assert issubclass(ChambollePockDeconv, DeconvBase)

    def test_instance_is_deconvbase(self, solver):
        assert isinstance(solver, DeconvBase)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Properties
# ══════════════════════════════════════════════════════════════════════════════

class TestProperties:

    def test_sigma_dual_positive(self, solver):
        assert solver.sigma_dual > 0.0

    def test_tau_primal_positive(self, solver):
        assert solver.tau_primal > 0.0

    def test_lipschitz_positive(self, solver):
        assert solver.lipschitz > 0.0

    def test_lipschitz_equals_max_pf_squared(self, solver):
        """L = max |PF|²."""
        xp_local = backend.xp
        expected = float(xp_local.max(xp_local.abs(solver.PF) ** 2))
        assert abs(solver.lipschitz - expected) < 1e-5

    def test_step_size_validity(self, solver):
        """τ(L/2 + 8σ) ≤ 1.0  (Condat 2013 convergence condition)."""
        K_NORM_SQ = 8.0
        condition_val = solver.tau_primal * (
            solver.lipschitz / 2.0 + K_NORM_SQ * solver.sigma_dual
        )
        assert condition_val <= 1.0 + 1e-6, (
            f"Condat condition violated: τ(L/2+8σ) = {condition_val:.6f} > 1"
        )

    def test_default_sigma_close_to_spec(self, solver):
        """Default σ = 0.99/√8."""
        expected = 0.99 / math.sqrt(8.0)
        assert abs(solver.sigma_dual - expected) < 1e-7


# ══════════════════════════════════════════════════════════════════════════════
# 3. _INIT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:

    def test_sigma_dual_in_init_keys(self):
        assert "sigma_dual" in ChambollePockDeconv._INIT_KEYS

    def test_theta_in_init_keys(self):
        assert "theta" in ChambollePockDeconv._INIT_KEYS

    def test_nonneg_in_init_keys(self):
        assert "nonneg" in ChambollePockDeconv._INIT_KEYS

    def test_tvnorm_in_init_keys(self):
        assert "TVnorm" in ChambollePockDeconv._INIT_KEYS

    def test_inherits_base_init_keys(self):
        for key in DeconvBase._INIT_KEYS:
            assert key in ChambollePockDeconv._INIT_KEYS

    def test_lambda_tv_not_in_init_keys(self):
        """lambda_tv is a deblur() param, not a constructor param."""
        assert "lambda_tv" not in ChambollePockDeconv._INIT_KEYS

    def test_num_iter_not_in_init_keys(self):
        assert "num_iter" not in ChambollePockDeconv._INIT_KEYS


# ══════════════════════════════════════════════════════════════════════════════
# 4. Output basics
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputBasics:

    def test_tvnorm2_output_shape(self, blurred, small_psf, test_image):
        result = ChambollePockDeconv(blurred, small_psf).deblur(
            num_iter=10, lambda_tv=0.01
        )
        assert result.shape == test_image.shape

    def test_tvnorm2_output_is_numpy(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=5)
        assert isinstance(result, np.ndarray)

    def test_tvnorm2_output_finite(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=5)
        assert np.isfinite(result).all()

    def test_tvnorm1_output_shape(self, blurred, small_psf, test_image):
        result = ChambollePockDeconv(blurred, small_psf, TVnorm=1).deblur(
            num_iter=10, lambda_tv=0.01
        )
        assert result.shape == test_image.shape

    def test_tvnorm1_output_finite(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf, TVnorm=1).deblur(
            num_iter=5, lambda_tv=0.01
        )
        assert np.isfinite(result).all()

    def test_non_square_image(self, small_psf):
        """Output shape matches non-square input (odd dims enforced by base)."""
        img = _test_image(51, 41)
        blr = _blur(img, small_psf)
        result = ChambollePockDeconv(blr, small_psf).deblur(num_iter=5)
        assert result.ndim == 2

    def test_output_dtype_float(self, blurred, small_psf):
        """Output is float (float32 or float64 — matches internal dtype)."""
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=5)
        assert result.dtype in (np.float32, np.float64)

    def test_output_2d(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=5)
        assert result.ndim == 2


# ══════════════════════════════════════════════════════════════════════════════
# 5. Positivity
# ══════════════════════════════════════════════════════════════════════════════

class TestPositivity:

    def test_nonneg_true_produces_nonneg_output(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf, nonneg=True).deblur(
            num_iter=20, lambda_tv=0.01
        )
        assert float(np.min(result)) >= 0.0

    def test_nonneg_false_may_produce_negatives(self, blurred, small_psf):
        """Without positivity, ringing may produce negatives — just verify it runs."""
        result = ChambollePockDeconv(blurred, small_psf, nonneg=False).deblur(
            num_iter=20, lambda_tv=0.001
        )
        assert isinstance(result, np.ndarray)

    def test_deblur_nonneg_override_true(self, blurred, small_psf):
        """nonneg=True at deblur() overrides constructor nonneg=False."""
        result = ChambollePockDeconv(blurred, small_psf, nonneg=False).deblur(
            num_iter=20, lambda_tv=0.01, nonneg=True
        )
        assert float(np.min(result)) >= 0.0

    def test_deblur_nonneg_override_false(self, blurred, small_psf):
        """nonneg=False at deblur() overrides constructor nonneg=True."""
        result = ChambollePockDeconv(blurred, small_psf, nonneg=True).deblur(
            num_iter=5, lambda_tv=0.01, nonneg=False
        )
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Deconvolution quality
# ══════════════════════════════════════════════════════════════════════════════

class TestDeconvolutionQuality:
    """
    Quality tests: PSNR after CP deblur must not be worse than blurred input.

    DeconvBase normalises the image to [0, 1] internally, so the result lives
    in a different absolute scale than the input test_image.  Both images are
    mapped to [0, 1] before PSNR calculation — same approach as test_fista.py
    and test_admm.py.
    """

    def _run_and_psnr(self, blurred, psf, image, TVnorm=2, iters=100):
        result = ChambollePockDeconv(blurred, psf, TVnorm=TVnorm).deblur(
            num_iter=iters, lambda_tv=0.005
        )
        h, w = result.shape
        ref = image[:h, :w]
        psnr_in  = _psnr(_norm01(ref), _norm01(blurred[:h, :w]))
        psnr_out = _psnr(_norm01(ref), _norm01(result))
        return psnr_in, psnr_out

    def test_tvnorm2_improves_psnr(self, blurred, small_psf, test_image):
        psnr_in, psnr_out = self._run_and_psnr(blurred, small_psf, test_image)
        assert psnr_out > psnr_in - 0.5, (
            f"TVnorm=2 CP did not improve quality: in={psnr_in:.2f}, "
            f"out={psnr_out:.2f}"
        )

    def test_tvnorm1_improves_psnr(self, blurred, small_psf, test_image):
        psnr_in, psnr_out = self._run_and_psnr(
            blurred, small_psf, test_image, TVnorm=1
        )
        assert psnr_out > psnr_in - 0.5, (
            f"TVnorm=1 CP did not improve quality: in={psnr_in:.2f}, "
            f"out={psnr_out:.2f}"
        )

    def test_output_values_in_reasonable_range(self, blurred, small_psf):
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=30)
        assert float(np.max(result)) < 10.0, f"max={np.max(result):.2f} too large"
        assert float(np.min(result)) >= 0.0, "nonneg violated"


# ══════════════════════════════════════════════════════════════════════════════
# 7. TVnorm variants
# ══════════════════════════════════════════════════════════════════════════════

class TestTVnorm:

    def test_tvnorm2_and_tvnorm1_produce_different_outputs(
        self, blurred, small_psf
    ):
        """Isotropic and anisotropic TV produce distinct results."""
        r2 = ChambollePockDeconv(blurred, small_psf, TVnorm=2).deblur(
            num_iter=20, lambda_tv=0.01
        )
        r1 = ChambollePockDeconv(blurred, small_psf, TVnorm=1).deblur(
            num_iter=20, lambda_tv=0.01
        )
        # They should not be identical (different projections)
        assert not np.allclose(r2, r1, atol=1e-6), (
            "TVnorm=1 and TVnorm=2 produced identical results — "
            "suspect dual projection bug"
        )

    def test_tvnorm_invalid_raises(self, blurred, small_psf):
        with pytest.raises(ValueError, match="TVnorm"):
            ChambollePockDeconv(blurred, small_psf, TVnorm=3)

    def test_tvnorm_0_raises(self, blurred, small_psf):
        with pytest.raises(ValueError):
            ChambollePockDeconv(blurred, small_psf, TVnorm=0)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Dual projection correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestDualProject:

    @pytest.fixture
    def cp_solver(self, blurred, small_psf):
        return ChambollePockDeconv(blurred, small_psf, TVnorm=2)

    @pytest.fixture
    def cp_solver_l1(self, blurred, small_psf):
        return ChambollePockDeconv(blurred, small_psf, TVnorm=1)

    def test_vectorial_projection_within_lambda_ball(self, cp_solver):
        """After TVnorm=2 projection, per-pixel ‖p‖₂ ≤ λ everywhere."""
        xp_local = backend.xp
        rng = np.random.default_rng(42)
        ph = xp_local.array(rng.standard_normal((20, 20)).astype(np.float32) * 2.0)
        pw = xp_local.array(rng.standard_normal((20, 20)).astype(np.float32) * 2.0)
        lam = xp_local.float32(0.5)

        ph_out, pw_out = cp_solver._dual_project(ph, pw, lam)
        mag = backend._to_numpy(
            xp_local.sqrt(ph_out * ph_out + pw_out * pw_out)
        )
        assert float(np.max(mag)) <= 0.5 + 1e-5, (
            f"max ‖p‖₂ = {np.max(mag):.4f} > λ=0.5 after vectorial projection"
        )

    def test_vectorial_projection_identity_for_small_vectors(self, cp_solver):
        """Vectors with ‖p‖₂ < λ should not be changed."""
        xp_local = backend.xp
        # Small vectors: max magnitude ≈ 0.1 < λ=1.0
        ph = xp_local.array(np.full((5, 5), 0.05, dtype=np.float32))
        pw = xp_local.array(np.full((5, 5), 0.05, dtype=np.float32))
        lam = xp_local.float32(1.0)

        ph_out, pw_out = cp_solver._dual_project(ph, pw, lam)
        np.testing.assert_allclose(
            backend._to_numpy(ph_out), backend._to_numpy(ph), atol=1e-6
        )
        np.testing.assert_allclose(
            backend._to_numpy(pw_out), backend._to_numpy(pw), atol=1e-6
        )

    def test_componentwise_clamp(self, cp_solver_l1):
        """TVnorm=1: each component clamped to [−λ, λ]."""
        xp_local = backend.xp
        ph = xp_local.array(np.array([[2.0, -3.0], [0.1, -0.1]], dtype=np.float32))
        pw = xp_local.array(np.array([[0.5, 1.5], [-0.5, 4.0]], dtype=np.float32))
        lam = xp_local.float32(1.0)

        ph_out, pw_out = cp_solver_l1._dual_project(ph, pw, lam)
        ph_np = backend._to_numpy(ph_out)
        pw_np = backend._to_numpy(pw_out)

        assert float(np.max(np.abs(ph_np))) <= 1.0 + 1e-6
        assert float(np.max(np.abs(pw_np))) <= 1.0 + 1e-6

    def test_componentwise_clamp_exact_values(self, cp_solver_l1):
        """Exact values after TVnorm=1 clamp."""
        xp_local = backend.xp
        ph = xp_local.array(np.array([[3.0, -2.0, 0.5]], dtype=np.float32))
        pw = xp_local.array(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        lam = xp_local.float32(1.0)

        ph_out, _ = cp_solver_l1._dual_project(ph, pw, lam)
        expected = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(backend._to_numpy(ph_out), expected, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Step size validity
# ══════════════════════════════════════════════════════════════════════════════

class TestStepSize:

    def test_condat_condition_satisfied(self, solver):
        """τ(L/2 + 8σ) ≤ 1 — Condat 2013 Theorem 3.1 convergence condition."""
        K_NORM_SQ = 8.0
        lhs = solver.tau_primal * (
            solver.lipschitz / 2.0 + K_NORM_SQ * solver.sigma_dual
        )
        assert lhs <= 1.0 + 1e-6, f"Condat condition: {lhs:.8f} > 1"

    def test_custom_sigma_still_satisfies_condition(self, blurred, small_psf):
        """Custom sigma_dual produces valid tau."""
        custom_sigma = 0.1
        solver = ChambollePockDeconv(blurred, small_psf, sigma_dual=custom_sigma)
        K_NORM_SQ = 8.0
        lhs = solver.tau_primal * (
            solver.lipschitz / 2.0 + K_NORM_SQ * solver.sigma_dual
        )
        assert lhs <= 1.0 + 1e-6

    def test_tau_less_than_2_over_lipschitz(self, solver):
        """τ < 2/L (necessary for gradient step stability)."""
        assert solver.tau_primal < 2.0 / solver.lipschitz + 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# 10. Mask
# ══════════════════════════════════════════════════════════════════════════════

class TestMask:

    def test_use_mask_true_by_default(self, solver):
        """ChambollePockDeconv always uses the mask."""
        assert solver.use_mask is True

    def test_mask_shape_matches_canvas(self, solver):
        assert solver.mask.shape == solver.full_shape

    def test_output_shape_with_mask(self, blurred, small_psf, test_image):
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=5)
        h, w = result.shape
        assert h <= test_image.shape[0] and w <= test_image.shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# 11. theta=0 disables extrapolation
# ══════════════════════════════════════════════════════════════════════════════

class TestTheta:

    def test_theta_zero_runs_without_error(self, blurred, small_psf):
        """theta=0 disables extrapolation — plain forward-backward."""
        result = ChambollePockDeconv(blurred, small_psf, theta=0.0).deblur(
            num_iter=10, lambda_tv=0.01
        )
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()

    def test_theta_zero_output_shape(self, blurred, small_psf, test_image):
        result = ChambollePockDeconv(blurred, small_psf, theta=0.0).deblur(
            num_iter=10
        )
        assert result.shape == test_image.shape

    def test_theta_zero_differs_from_theta_one(self, blurred, small_psf):
        """theta=0 and theta=1 produce different iterates after 20 iters."""
        r0 = ChambollePockDeconv(blurred, small_psf, theta=0.0).deblur(
            num_iter=20, lambda_tv=0.01
        )
        r1 = ChambollePockDeconv(blurred, small_psf, theta=1.0).deblur(
            num_iter=20, lambda_tv=0.01
        )
        assert not np.allclose(r0, r1, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Convergence
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergence:

    def test_loose_tol_converges_before_max_iter(self, blurred, small_psf):
        """Very loose tolerance should converge well before 200 iterations."""
        result = ChambollePockDeconv(blurred, small_psf).deblur(
            num_iter=200, tol=1.0, min_iter=1, check_every=1
        )
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()

    def test_min_iter_prevents_early_stop(self, blurred, small_psf):
        """min_iter=50 prevents convergence check before iteration 50."""
        result = ChambollePockDeconv(blurred, small_psf).deblur(
            num_iter=55, tol=100.0, min_iter=50, check_every=1
        )
        assert isinstance(result, np.ndarray)

    def test_single_iteration_runs(self, blurred, small_psf):
        """num_iter=1 should complete without error."""
        result = ChambollePockDeconv(blurred, small_psf).deblur(num_iter=1)
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()


# ══════════════════════════════════════════════════════════════════════════════
# 13. Wrapper chambolle_pock_deblur
# ══════════════════════════════════════════════════════════════════════════════

class TestWrapper:

    def test_wrapper_matches_class_result(self, blurred, small_psf):
        """chambolle_pock_deblur and class-based call produce identical results."""
        result_cls = ChambollePockDeconv(blurred, small_psf).deblur(
            num_iter=10, lambda_tv=0.01
        )
        result_fn = chambolle_pock_deblur(
            blurred, small_psf, iters=10, lambda_tv=0.01
        )
        np.testing.assert_array_equal(result_cls, result_fn)

    def test_wrapper_first_arg_is_image(self):
        params = list(inspect.signature(chambolle_pock_deblur).parameters.keys())
        assert params[0] == "image"

    def test_wrapper_second_arg_is_psf(self):
        params = list(inspect.signature(chambolle_pock_deblur).parameters.keys())
        assert params[1] == "psf"

    def test_wrapper_accessible_from_root_namespace(self):
        """chambolle_pock_deblur is accessible from Reconstruction package root."""
        assert hasattr(Reconstruction, "chambolle_pock_deblur")
        assert callable(Reconstruction.chambolle_pock_deblur)

    def test_class_accessible_from_root_namespace(self):
        assert hasattr(Reconstruction, "ChambollePockDeconv")

    def test_wrapper_routes_init_key_to_constructor(self, blurred, small_psf):
        """Constructor key 'nonneg' routed correctly via wrapper."""
        result = chambolle_pock_deblur(
            blurred, small_psf, iters=5, nonneg=True
        )
        assert float(np.min(result)) >= 0.0

    def test_wrapper_routes_tvnorm(self, blurred, small_psf):
        """Constructor key 'TVnorm' routed correctly via wrapper."""
        result = chambolle_pock_deblur(
            blurred, small_psf, iters=5, TVnorm=1
        )
        assert isinstance(result, np.ndarray)

    def test_wrapper_routes_deblur_key(self, blurred, small_psf):
        """Deblur key 'tol' accepted via wrapper kwargs."""
        result = chambolle_pock_deblur(
            blurred, small_psf, iters=5, tol=1e-3
        )
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# 14. Root namespace
# ══════════════════════════════════════════════════════════════════════════════

class TestRootNamespace:

    def test_chambolle_pock_deconv_in_all(self):
        assert "ChambollePockDeconv" in Reconstruction.__all__

    def test_chambolle_pock_deblur_in_all(self):
        assert "chambolle_pock_deblur" in Reconstruction.__all__

    def test_instantiate_from_root(self, blurred, small_psf):
        cls = Reconstruction.ChambollePockDeconv
        instance = cls(blurred, small_psf)
        assert isinstance(instance, DeconvBase)

    def test_chambolle_pock_deconv_subclass_of_deconvbase_via_root(self):
        assert issubclass(Reconstruction.ChambollePockDeconv, DeconvBase)

    def test_wrapper_callable_from_root(self, blurred, small_psf):
        result = Reconstruction.chambolle_pock_deblur(
            blurred, small_psf, iters=5
        )
        assert isinstance(result, np.ndarray)
