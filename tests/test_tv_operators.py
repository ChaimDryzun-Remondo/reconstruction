"""
Phase 2 verification tests for Reconstruction._tv_operators.

Checks:
  1. Adjointness:  ⟨−∇x, p⟩ = ⟨x, div(p)⟩  for random arrays.
  2. prox_tv identity: prox_tv_chambolle(v, gamma=0) returns v unchanged.
  3. prox_tv denoising: std is reduced ≥ 50% on a constant + noise image.
  4. Multiplicative correction: shape preserved, all values ≥ 0.5 (clamp).
"""
from __future__ import annotations

import numpy as np
import pytest

import Reconstruction._backend as backend
from Reconstruction._tv_operators import (
    backward_div,
    forward_grad,
    prox_tv_chambolle,
    tv_multiplicative_correction,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend before and after every test."""
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Adjointness:  ⟨−∇x, p⟩ = ⟨x, div(p)⟩
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjointness:
    """
    Verify the adjoint pairing  ⟨−∇x, p⟩ = ⟨x, div(p)⟩.

    Tests are run with float64 arrays so that floating-point cancellation
    errors are well below 1e-5 (float64 eps ≈ 2e-16 vs. float32 ≈ 1e-7).
    The Neumann BC operators are algebraically exact in either precision;
    float64 just gives a comfortable margin over the 1e-5 threshold.
    """

    def test_adjointness_random_64x64(self):
        """⟨−∇x, p⟩ ≈ ⟨x, div(p)⟩  for random 64×64 float64 arrays."""
        rng = np.random.default_rng(0)
        x   = rng.random((64, 64))           # float64
        p_h = rng.random((64, 64))
        p_w = rng.random((64, 64))

        dh, dw = forward_grad(x)

        # LHS = ⟨−∇x, p⟩
        lhs = float(np.sum(-dh * p_h) + np.sum(-dw * p_w))
        # RHS = ⟨x, div(p)⟩
        rhs = float(np.sum(x * backward_div(p_h, p_w)))

        assert abs(lhs - rhs) < 1e-5, (
            f"Adjointness violated: |lhs − rhs| = {abs(lhs - rhs):.3e} "
            f"(lhs={lhs:.6f}, rhs={rhs:.6f})"
        )

    def test_adjointness_random_33x47(self):
        """Adjointness holds for a non-square array shape."""
        rng = np.random.default_rng(1)
        x   = rng.random((33, 47))
        p_h = rng.random((33, 47))
        p_w = rng.random((33, 47))

        dh, dw = forward_grad(x)
        lhs = float(np.sum(-dh * p_h) + np.sum(-dw * p_w))
        rhs = float(np.sum(x * backward_div(p_h, p_w)))

        assert abs(lhs - rhs) < 1e-5, (
            f"Non-square adjointness failed: |lhs − rhs| = {abs(lhs - rhs):.3e}"
        )

    def test_forward_grad_neumann_bc(self):
        """forward_grad: last row of dh and last col of dw must be zero."""
        rng = np.random.default_rng(2)
        x = rng.random((16, 16)).astype(np.float32)
        dh, dw = forward_grad(x)
        np.testing.assert_array_equal(dh[-1, :], np.zeros(16, dtype=np.float32),
                                      err_msg="Last row of dh is not zero (Neumann BC)")
        np.testing.assert_array_equal(dw[:, -1], np.zeros(16, dtype=np.float32),
                                      err_msg="Last col of dw is not zero (Neumann BC)")

    def test_forward_grad_interior_values(self):
        """forward_grad interior values are forward differences."""
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        dh, dw = forward_grad(x)
        # Row differences along axis 0
        expected_dh = np.diff(x, axis=0, prepend=np.nan)[1:, :]   # x[1:]-x[:-1]
        np.testing.assert_array_equal(dh[:-1, :], x[1:, :] - x[:-1, :])
        # Column differences along axis 1
        np.testing.assert_array_equal(dw[:, :-1], x[:, 1:] - x[:, :-1])

    def test_backward_div_output_shape(self):
        """backward_div returns an array with the same shape as its inputs."""
        p_h = np.ones((32, 32), dtype=np.float32)
        p_w = np.ones((32, 32), dtype=np.float32)
        div = backward_div(p_h, p_w)
        assert div.shape == (32, 32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Proximal TV: identity at gamma = 0
# ══════════════════════════════════════════════════════════════════════════════

class TestProxTVIdentity:
    """prox_tv_chambolle(v, gamma=0) must return a copy of v unchanged."""

    def test_gamma_zero_returns_copy(self):
        """gamma=0 returns v.copy(), not v itself."""
        rng = np.random.default_rng(3)
        v = rng.random((32, 32)).astype(np.float32)
        result = prox_tv_chambolle(v, gamma=0)
        np.testing.assert_array_equal(result, v)
        assert result is not v, "gamma=0 must return a copy, not the same object"

    def test_negative_gamma_returns_copy(self):
        """gamma < 0 is treated like gamma=0 (identity)."""
        v = np.ones((8, 8), dtype=np.float32)
        result = prox_tv_chambolle(v, gamma=-1.0)
        np.testing.assert_array_equal(result, v)

    def test_gamma_zero_shape_preserved(self):
        """Output shape matches input shape when gamma=0."""
        v = np.zeros((17, 23), dtype=np.float32)
        result = prox_tv_chambolle(v, gamma=0)
        assert result.shape == v.shape


# ══════════════════════════════════════════════════════════════════════════════
# 3. Proximal TV: noise reduction on constant + noise image
# ══════════════════════════════════════════════════════════════════════════════

class TestProxTVDenoising:
    """prox_tv_chambolle must denoise a noisy constant image by ≥ 50%."""

    def test_noise_reduction_50_percent(self):
        """
        Denoising a constant 0.5 image + Gaussian σ=0.05 with
        gamma=0.05, n_inner=100 reduces std by at least 50%.
        """
        rng = np.random.default_rng(42)
        signal = 0.5
        noise_std = 0.05
        v = (np.full((64, 64), signal, dtype=np.float32)
             + rng.normal(0, noise_std, (64, 64)).astype(np.float32))

        std_before = float(np.std(v))
        result = prox_tv_chambolle(v, gamma=0.05, n_inner=100)
        std_after = float(np.std(result))

        reduction = 1.0 - std_after / std_before
        assert reduction >= 0.5, (
            f"Expected ≥50% noise reduction, got {100*reduction:.1f}% "
            f"(std: {std_before:.4f} → {std_after:.4f})"
        )

    def test_output_shape_preserved(self):
        """prox_tv_chambolle preserves array shape."""
        v = np.ones((17, 23), dtype=np.float32)
        result = prox_tv_chambolle(v, gamma=0.01, n_inner=10)
        assert result.shape == v.shape

    def test_stronger_gamma_smoother(self):
        """Larger gamma produces smoother (lower std) output."""
        rng = np.random.default_rng(7)
        v = (np.full((32, 32), 0.5, dtype=np.float32)
             + rng.normal(0, 0.1, (32, 32)).astype(np.float32))

        result_weak  = prox_tv_chambolle(v, gamma=0.001, n_inner=50)
        result_strong = prox_tv_chambolle(v, gamma=0.1,  n_inner=50)

        assert np.std(result_strong) < np.std(result_weak), (
            "Stronger gamma should produce lower std (smoother result)"
        )

    def test_constant_image_is_fixed_point(self):
        """A perfectly constant image is a fixed point of the TV prox."""
        v = np.full((16, 16), 0.5, dtype=np.float32)
        result = prox_tv_chambolle(v, gamma=0.1, n_inner=50)
        # For a constant image TV(v)=0, so prox is identity — should be exact.
        np.testing.assert_allclose(result, v, atol=1e-5,
                                   err_msg="Constant image changed after TV prox")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multiplicative correction: shape and safety clamp
# ══════════════════════════════════════════════════════════════════════════════

class TestTVMultiplicativeCorrection:
    """
    Verify tv_multiplicative_correction:
      - output shape matches input
      - all values ≥ 0.5 (safety clamp)
    """

    def test_shape_preserved(self):
        """Output shape matches the input image shape."""
        rng = np.random.default_rng(5)
        x = rng.random((64, 64)).astype(np.float32)
        correction = tv_multiplicative_correction(x, lambda_tv=0.001)
        assert correction.shape == x.shape, (
            f"Shape mismatch: input {x.shape}, output {correction.shape}"
        )

    def test_clamp_floor_0_5(self):
        """All correction values must be ≥ 0.5 (safety clamp)."""
        rng = np.random.default_rng(6)
        x = rng.random((64, 64)).astype(np.float32)
        correction = tv_multiplicative_correction(x, lambda_tv=0.001)
        min_val = float(np.min(correction))
        assert min_val >= 0.5, (
            f"Correction clamp violated: min value = {min_val:.6f} < 0.5"
        )

    def test_clamp_floor_strong_lambda(self):
        """Clamp holds even for an aggressively large lambda_tv."""
        rng = np.random.default_rng(8)
        x = rng.random((32, 32)).astype(np.float32)
        # Very large lambda would push correction below 0.5 without the clamp.
        correction = tv_multiplicative_correction(x, lambda_tv=100.0)
        assert float(np.min(correction)) >= 0.5, (
            "Safety clamp failed for large lambda_tv"
        )

    def test_flat_image_correction_near_one(self):
        """On a perfectly flat image the correction factor is ≈ 1 everywhere."""
        x = np.full((16, 16), 0.5, dtype=np.float32)
        correction = tv_multiplicative_correction(x, lambda_tv=0.01)
        # Gradient is zero → div(n) = 0 → correction = 1.0
        np.testing.assert_allclose(correction, np.ones_like(correction),
                                   atol=1e-5,
                                   err_msg="Flat image should have correction ≈ 1")

    def test_lambda_zero_correction_is_one(self):
        """lambda_tv=0 should return an array of ones (no correction)."""
        rng = np.random.default_rng(9)
        x = rng.random((16, 16)).astype(np.float32)
        correction = tv_multiplicative_correction(x, lambda_tv=0.0)
        np.testing.assert_allclose(correction, np.ones_like(correction),
                                   atol=1e-5,
                                   err_msg="lambda_tv=0 should give all-ones correction")

    def test_non_square_shape(self):
        """Shape and clamp hold for non-square input."""
        rng = np.random.default_rng(10)
        x = rng.random((33, 57)).astype(np.float32)
        correction = tv_multiplicative_correction(x, lambda_tv=0.01)
        assert correction.shape == (33, 57)
        assert float(np.min(correction)) >= 0.5
