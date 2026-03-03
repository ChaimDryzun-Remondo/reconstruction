"""
Phase 7 cross-algorithm integration smoke tests.

Creates a single blurred test image and runs ALL available algorithms on it
with minimal iterations.  Verifies:
  - All produce output of the same shape as the input.
  - All produce finite (non-NaN, non-Inf) values.
  - All produce output different from the blurred input (deconvolution acted).
  - Wiener (non-iterative) and all iterative methods handle the same image/PSF
    without error.

These are smoke tests — they catch import errors, shape mismatches, and obvious
crashes across the full algorithm suite, NOT deconvolution quality.
"""
from __future__ import annotations

import numpy as np
import pytest

import Reconstruction
import Reconstruction._backend as backend
from Reconstruction._base import DeconvBase


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend for all integration tests."""
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
    img = np.full((h, w), 0.1, dtype=np.float64)
    ch, cw = h // 4, w // 4
    img[ch:3 * ch, cw:3 * cw] = 0.8
    return img


def _blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    return np.clip(fftconvolve(image, psf, mode="same"), 0, None)


@pytest.fixture(scope="module")
def integration_image() -> np.ndarray:
    return _test_image(51, 51)


@pytest.fixture(scope="module")
def integration_psf() -> np.ndarray:
    return _gaussian_psf(size=9, sigma=1.5)


@pytest.fixture(scope="module")
def integration_blurred(integration_image, integration_psf) -> np.ndarray:
    return _blur(integration_image, integration_psf)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build (algorithm_name, result) pairs for all available algorithms
# ══════════════════════════════════════════════════════════════════════════════

def _run_all_algorithms(blurred, psf) -> list[tuple[str, np.ndarray]]:
    """
    Run all available algorithms and collect (name, result) pairs.

    PnP-ADMM is included only when bm3d is installed.
    All algorithms use minimal iterations (5-10) to keep the test fast.
    """
    results = []

    # Wiener (single-pass, no iterations; no extra kwargs needed)
    result = Reconstruction.wiener_deblur(blurred, psf)
    results.append(("Wiener", result))

    # RL (masked)
    result = Reconstruction.rl_deblur_unknown_boundary(
        blurred, psf, num_iter=5, lambda_tv=0.0002
    )
    results.append(("RL-Unknown-Boundary", result))

    # Landweber / FISTA
    result = Reconstruction.landweber_deblur_unknown_boundary(
        blurred, psf, num_iter=5, lambda_tv=0.001
    )
    results.append(("Landweber", result))

    # ADMM-TV
    result = Reconstruction.admm_deblur(
        blurred, psf, iters=5, lambda_tv=0.01
    )
    results.append(("ADMM-TV", result))

    # TVAL3
    result = Reconstruction.tval3_deblur(
        blurred, psf, iters=5, lambda_tv=0.01
    )
    results.append(("TVAL3", result))

    # PnP-ADMM (optional)
    if Reconstruction._HAS_PNP:
        result = Reconstruction.pnp_admm_deblur(
            blurred, psf, iters=3, lambda_tv=0.01,
            rho_v=1.0, rho_z=1.0,
        )
        results.append(("PnP-ADMM", result))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Core smoke tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAllAlgorithmsRun:

    def test_all_algorithms_complete_without_error(
        self, integration_blurred, integration_psf
    ):
        """All algorithms run to completion without raising."""
        results = _run_all_algorithms(integration_blurred, integration_psf)
        assert len(results) >= 5, "At least 5 algorithms should have run"

    def test_all_outputs_are_numpy_arrays(
        self, integration_blurred, integration_psf
    ):
        """All algorithms return numpy arrays."""
        for name, result in _run_all_algorithms(integration_blurred, integration_psf):
            assert isinstance(result, np.ndarray), (
                f"{name}: expected np.ndarray, got {type(result).__name__}"
            )

    def test_all_outputs_same_shape_as_input(
        self, integration_blurred, integration_psf, integration_image
    ):
        """All algorithms return an array with the same shape as the input image."""
        expected_shape = integration_image.shape
        for name, result in _run_all_algorithms(integration_blurred, integration_psf):
            assert result.shape == expected_shape, (
                f"{name}: expected shape {expected_shape}, got {result.shape}"
            )

    def test_all_outputs_finite(
        self, integration_blurred, integration_psf
    ):
        """All algorithm outputs contain only finite values."""
        for name, result in _run_all_algorithms(integration_blurred, integration_psf):
            assert np.isfinite(result).all(), (
                f"{name}: output contains NaN or Inf values"
            )

    def test_all_outputs_differ_from_input(
        self, integration_blurred, integration_psf, integration_image
    ):
        """All outputs differ from the blurred input (deconvolution acted)."""
        # Normalise blurred to match DeconvBase's output scale
        blurred_norm = integration_blurred / (integration_blurred.max() + 1e-8)

        for name, result in _run_all_algorithms(integration_blurred, integration_psf):
            result_norm = result / (result.max() + 1e-8)
            max_diff = float(np.max(np.abs(result_norm - blurred_norm)))
            assert max_diff > 1e-4, (
                f"{name}: output too similar to blurred input "
                f"(max_diff={max_diff:.2e}); deconvolution may not have acted"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Cross-algorithm consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossAlgorithmConsistency:

    def test_all_outputs_in_reasonable_range(
        self, integration_blurred, integration_psf
    ):
        """All outputs are in a reasonable value range (no extreme amplification).

        Thresholds are intentionally loose to accommodate the Wiener filter,
        which is a linear operation and can produce ringing artifacts and
        negative values — both are physically expected.  The check catches
        truly pathological blow-ups (NaN/Inf are already tested separately).
        """
        for name, result in _run_all_algorithms(integration_blurred, integration_psf):
            assert float(np.max(result)) < 1000.0, (
                f"{name}: max output {np.max(result):.2f} unreasonably large"
            )
            assert float(np.min(result)) > -100.0, (
                f"{name}: min output {np.min(result):.4f} unreasonably negative"
            )

    def test_wiener_same_shape_as_iterative(
        self, integration_blurred, integration_psf, integration_image
    ):
        """Wiener (non-iterative) returns same shape as iterative methods."""
        wiener_result = Reconstruction.wiener_deblur(
            integration_blurred, integration_psf
        )
        admm_result = Reconstruction.admm_deblur(
            integration_blurred, integration_psf, iters=5, lambda_tv=0.01
        )
        assert wiener_result.shape == admm_result.shape == integration_image.shape

    def test_class_and_wrapper_produce_same_result(
        self, integration_blurred, integration_psf
    ):
        """Class-based and wrapper-based calls give identical results for ADMM."""
        kw_init = dict(rho_v=16.0, rho_w=16.0)
        kw_deblur = dict(num_iter=5, lambda_tv=0.01)

        result_cls = Reconstruction.ADMMDeconv(
            integration_blurred, integration_psf, **kw_init
        ).deblur(**kw_deblur)

        result_fn = Reconstruction.admm_deblur(
            integration_blurred, integration_psf,
            iters=5, lambda_tv=0.01,
            **kw_init,
        )
        np.testing.assert_array_equal(result_cls, result_fn)

    def test_repeated_calls_same_result(
        self, integration_blurred, integration_psf
    ):
        """The same algorithm called twice on the same input gives identical output."""
        kw = dict(iters=5, lambda_tv=0.01)
        result1 = Reconstruction.tval3_deblur(
            integration_blurred, integration_psf, **kw
        )
        result2 = Reconstruction.tval3_deblur(
            integration_blurred, integration_psf, **kw
        )
        np.testing.assert_array_equal(result1, result2)


# ══════════════════════════════════════════════════════════════════════════════
# Package-level instantiation via root namespace
# ══════════════════════════════════════════════════════════════════════════════

class TestRootNamespaceInstantiation:

    def test_can_instantiate_all_classes_from_root(
        self, integration_blurred, integration_psf
    ):
        """All algorithm classes can be instantiated via Reconstruction.<Class>."""
        classes = [
            ("WienerDeconv", {}),
            ("RLUnknownBoundary", {}),
            ("LandweberUnknownBoundary", {}),
            ("ADMMDeconv", {}),
            ("TVAL3Deconv", {}),
        ]
        for name, extra_kw in classes:
            cls = getattr(Reconstruction, name)
            instance = cls(integration_blurred, integration_psf, **extra_kw)
            assert isinstance(instance, DeconvBase), (
                f"{name} instance should be a DeconvBase"
            )

    def test_pnp_instantiation_from_root(
        self, integration_blurred, integration_psf
    ):
        """PnPADMM can be instantiated from Reconstruction.PnPADMM."""
        pytest.importorskip("bm3d", reason="bm3d not installed")
        PnPADMM = Reconstruction.PnPADMM
        instance = PnPADMM(integration_blurred, integration_psf,
                           rho_v=1.0, rho_z=1.0)
        assert isinstance(instance, DeconvBase)
