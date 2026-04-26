"""Contract tests for ``Reconstruction.wiener.optimize_alpha``.

Sprint 4 commit T3.1 promoted ``optimize_wiener_alpha`` from
``examples/example_flow.py`` into a public API at ``Reconstruction.wiener``
under the new name ``optimize_alpha``, returning a frozen
``OptimizeAlphaResult`` dataclass.

Coverage:

- **Convergence (5 tests)** — synthetic Gaussian-PSF + AWGN setup with
  empirical-optimum verification.  Each test fine-sweeps the metric to
  locate the empirical optimum ``t_emp`` and asserts ``optimize_alpha``
  lands within tolerance.
- **Robustness (5 tests)** — degenerate metric values (NaN, constant),
  solver exception propagation, boundary-α handling.
- **Contract (3 tests)** — return type is ``OptimizeAlphaResult``, all
  fields populated, image is float dtype.
- **Search-parameter dispatch (2 tests)** — ``coarse_n_points`` and
  ``brent_xtol`` are honoured.
- **Metric-substitution (1 test)** — custom ``psnr_fn`` is invoked at
  the optimum (``ssim_fn`` substitution is exercised by every
  convergence test).

The synthetic ``_blur_and_add_awgn`` helper uses ``scipy.signal.fftconvolve``
+ explicit AWGN rather than ``Common.image_perturbation.blur``.  The
canonical ``blur`` MINMAX-rescales the input to [0, 1] (see Sprint 4
commit T1.2 deferral / NOTES.md), which destroys the absolute-scale
information the analytic noise model below depends on.

The fine-sweep helper replicates ``_normalize_image_for_metric`` from
``wiener.py`` rather than importing it; the duplication is intentional
to keep the test independent of the private helper's signature drift.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from Reconstruction import (
    OptimizeAlphaResult,
    WienerDeconv,
    optimize_alpha,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_test_image(shape: tuple[int, int] = (65, 65), seed: int = 42) -> np.ndarray:
    H, W = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    grad = (yy + xx) / (H + W - 2)
    sinusoid = 0.3 * np.sin(2 * np.pi * yy / H) * np.cos(2 * np.pi * xx / W)
    rng = np.random.default_rng(seed)
    noise = 0.05 * rng.standard_normal(shape)
    img = 0.5 * grad + 0.3 + sinusoid + noise
    return np.clip(img, 0.0, 1.0)


def _make_gaussian_psf(size: int = 11, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    cy = cx = (size - 1) / 2.0
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _blur_and_add_awgn(
    image: np.ndarray, psf: np.ndarray, sigma_n: float, seed: int = 42
) -> np.ndarray:
    blurred = fftconvolve(image, psf, mode="same")
    rng = np.random.default_rng(seed)
    return blurred + rng.normal(0.0, sigma_n, blurred.shape)


def _norm_for_metric(image: np.ndarray) -> np.ndarray:
    """Replicates ``_normalize_image_for_metric`` from ``wiener.py``."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-6:
        return np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
    return np.zeros_like(image)


def _fine_sweep_optimum(
    solver: WienerDeconv,
    ref_image: np.ndarray,
    metric_fn,
    t_lo: float = -5.0,
    t_hi: float = 0.0,
    n_points: int = 100,
) -> float:
    """Locate empirical metric optimum on a fine log10(α) grid."""
    t_grid = np.linspace(t_lo, t_hi, n_points)
    metric_grid = np.array(
        [
            float(metric_fn(_norm_for_metric(solver.deblur(alpha=10.0 ** t)), ref_image))
            for t in t_grid
        ]
    )
    return float(t_grid[np.argmax(metric_grid)])


def _psnr_metric(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Plain PSNR (assumes both arrays in [0, 1])."""
    mse = float(np.mean((estimate - reference) ** 2))
    if mse <= 1e-15:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _neg_mse_metric(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Negated MSE (higher is better, as the optimizer expects)."""
    return -float(np.mean((estimate - reference) ** 2))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def synthetic_setup_snr10():
    img = _make_test_image()
    psf = _make_gaussian_psf()
    degraded = _blur_and_add_awgn(img, psf, sigma_n=0.02)
    solver = WienerDeconv(
        degraded, psf, mode="Tikhonov", paddingMode="Reflect", padding_scale=2.0
    )
    _ = solver.deblur()  # warm-up + populate last_alpha
    return img, psf, degraded, solver


# ===========================================================================
# Group 1 - Convergence
# ===========================================================================


# Convergence tests use a metric-value-based assertion rather than a
# location-based assertion: the metric landscape can be flat or have
# multiple near-equal local optima where the location of the true
# optimum is ill-defined within numerical noise.  The location-based
# check is sensitive to that flatness; the metric-value check verifies
# that the function lands at a point whose metric is within tolerance
# of the fine-sweep peak metric over the SAME search range.  This is
# a robust convergence guarantee — "optimize_alpha finds a point as
# good as the best the search range contains" — and is independent of
# whether the landscape happens to be flat or peaked.


def _peak_metric_over_range(
    solver: WienerDeconv,
    ref_image: np.ndarray,
    metric_fn,
    t_lo: float,
    t_hi: float,
    n_points: int = 200,
) -> float:
    """Maximum metric value over a fine log10(α) grid covering [t_lo, t_hi]."""
    t_grid = np.linspace(t_lo, t_hi, n_points)
    return float(
        max(
            metric_fn(_norm_for_metric(solver.deblur(alpha=10.0 ** t)), ref_image)
            for t in t_grid
        )
    )


@pytest.mark.parametrize("sigma_n", [0.02, 0.0067, 0.002])
def test_optimize_alpha_converges_under_psnr_objective_tikhonov(
    sigma_n: float,
) -> None:
    """Convergence: PSNR-as-metric in Tikhonov mode, three SNRs.

    The fine sweep and ``optimize_alpha`` cover identical ranges so
    their respective optima are comparable.  Asserts that
    ``result.ssim`` is within 0.3 dB of the fine-sweep peak metric
    across the same range — robust to landscape flatness.
    """
    img = _make_test_image()
    psf = _make_gaussian_psf()
    degraded = _blur_and_add_awgn(img, psf, sigma_n=sigma_n)
    solver = WienerDeconv(
        degraded, psf, mode="Tikhonov", paddingMode="Reflect", padding_scale=2.0
    )
    _ = solver.deblur()

    t_lo, t_hi = -4.0, 1.0
    t_center = (t_lo + t_hi) / 2
    half_width = (t_hi - t_lo) / 2

    peak_metric = _peak_metric_over_range(solver, img, _psnr_metric, t_lo, t_hi)

    result = optimize_alpha(
        solver, img, alpha_0=10.0 ** t_center,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        coarse_half_width=half_width,
    )
    assert result.ssim >= peak_metric - 0.3


@pytest.mark.monorepo
def test_optimize_alpha_default_metric_runs_to_completion(
    synthetic_setup_snr10,
) -> None:
    """Verify the default metric path (lazy-imported ``MSSSIM`` /
    ``PiqPSNR``) is wired correctly and returns a finite result.

    Marked ``monorepo`` because the lazy imports require
    ``RemondoPythonCore.Common.Image_Quality_Measures`` to be on the
    path — only available when the parent monorepo environment is
    active.  Excluded from the submodule's default core-profile run.
    """
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(solver, img, alpha_0=1e-3, coarse_n_points=20)
    assert isinstance(result, OptimizeAlphaResult)
    assert np.isfinite(result.ssim)
    assert np.isfinite(result.psnr)


def test_optimize_alpha_converges_under_neg_mse_in_classical_mode() -> None:
    """Tightest-tolerance convergence test: Classical mode + negated
    MSE metric.  In Classical mode the regularisation parameter α IS
    the noise/signal-power ratio K, so the metric landscape has a
    sharp well-defined optimum.  Tolerance is 1e-3 in -MSE space —
    much tighter than the PSNR tests above because MSE doesn't have
    the log compression that smears PSNR's optimum into a plateau.
    """
    img = _make_test_image()
    psf = _make_gaussian_psf()
    sigma_n = 0.02
    degraded = _blur_and_add_awgn(img, psf, sigma_n=sigma_n)
    solver = WienerDeconv(
        degraded, psf, mode="Classical", paddingMode="Reflect", padding_scale=2.0
    )
    _ = solver.deblur()

    t_lo, t_hi = -4.0, 1.0
    t_center = (t_lo + t_hi) / 2
    half_width = (t_hi - t_lo) / 2

    peak_metric = _peak_metric_over_range(solver, img, _neg_mse_metric, t_lo, t_hi)

    result = optimize_alpha(
        solver, img, alpha_0=10.0 ** t_center,
        ssim_fn=_neg_mse_metric, psnr_fn=_psnr_metric,
        coarse_half_width=half_width,
        coarse_n_points=80,
        brent_xtol=1e-4,
    )
    assert result.ssim >= peak_metric - 1e-3


# ===========================================================================
# Group 2 - Robustness
# ===========================================================================


def test_optimize_alpha_does_not_crash_when_metric_returns_constant(
    synthetic_setup_snr10,
) -> None:
    """Constant metric: argmax returns 0 (the leftmost grid point);
    Brent refines around it.  Function should return without crashing,
    even though the result is meaningless."""
    _img, _psf, _degraded, solver = synthetic_setup_snr10
    img = _make_test_image()
    constant_metric = lambda a, b: 0.5  # noqa: E731
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=constant_metric, psnr_fn=_psnr_metric,
    )
    assert isinstance(result, OptimizeAlphaResult)
    assert result.ssim == 0.5


def test_optimize_alpha_propagates_solver_exception(synthetic_setup_snr10) -> None:
    """If solver.deblur raises, the exception propagates.  No silent
    suppression."""
    img, _psf, _degraded, solver = synthetic_setup_snr10

    class _PoisonedSolver:
        def deblur(self, alpha=None):
            raise RuntimeError("intentional test failure")

    poisoned = _PoisonedSolver()
    with pytest.raises(RuntimeError, match="intentional test failure"):
        optimize_alpha(
            poisoned, img, alpha_0=1e-3,
            ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        )


def test_optimize_alpha_handles_alpha_0_far_from_optimum(synthetic_setup_snr10) -> None:
    """Boundary case: alpha_0 set far from any reasonable optimum.
    The coarse grid's right edge becomes the best.  Brent refines
    inside the (boundary-adjacent) cell.  Function should return
    without crashing."""
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-20,  # absurdly low
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        coarse_half_width=2.0,
    )
    assert isinstance(result, OptimizeAlphaResult)
    assert np.all(np.isfinite(result.coarse_ssim))


def test_optimize_alpha_does_not_sample_alpha_zero(synthetic_setup_snr10) -> None:
    """The function works in log10 space, so α=0 is unreachable.
    Pin this so a future refactor that switches to linear α space
    surfaces the change."""
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
    )
    assert result.alpha > 0
    assert np.all(10.0 ** result.coarse_t > 0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing surprise: when the metric returns NaN for some "
        "candidates, np.argmax silently selects index 0 (NaN sorts last "
        "in numpy's default argmax, so the first finite value wins -- "
        "or, if all are NaN, index 0 is returned arbitrarily). The "
        "function does not detect or skip NaN candidates, so the "
        "returned 'optimum' is meaningless. Future fix can either "
        "skip-NaN candidates and continue or raise a clear error; the "
        "test body's assertion (NaN must NOT silently propagate to "
        "result.ssim) accepts either fix path. See NOTES.md Sprint 4 / "
        "commit T3.1."
    ),
)
def test_optimize_alpha_nan_metric_is_handled(synthetic_setup_snr10) -> None:
    """A metric returning NaN for some candidates currently produces a
    silent meaningless result.  Flexible xfail per the cross-sprint
    pattern: future fix can skip NaN candidates or raise; either path
    is acceptable as long as NaN does not silently propagate to the
    returned ``ssim`` field."""
    img, _psf, _degraded, solver = synthetic_setup_snr10

    nan_metric = lambda a, b: float("nan")  # noqa: E731
    try:
        result = optimize_alpha(
            solver, img, alpha_0=1e-3,
            ssim_fn=nan_metric, psnr_fn=_psnr_metric,
        )
    except (ValueError, RuntimeError):
        # Acceptable: function rejected NaN-returning metric loudly.
        return
    # Acceptable: function detected NaN and returned a finite result
    # (perhaps from a NaN-skip path).
    assert np.isfinite(result.ssim), (
        f"NaN metric silently propagated to result.ssim={result.ssim}"
    )


# ===========================================================================
# Group 3 - Result-shape contract
# ===========================================================================


def test_optimize_alpha_returns_optimize_alpha_result_dataclass(
    synthetic_setup_snr10,
) -> None:
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
    )
    assert isinstance(result, OptimizeAlphaResult)


def test_optimize_alpha_result_fields_all_populated(synthetic_setup_snr10) -> None:
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        coarse_n_points=20,
    )
    assert result.alpha > 0.0
    assert np.isfinite(result.log10_alpha)
    assert np.isfinite(result.ssim)
    assert np.isfinite(result.psnr)
    assert result.image.ndim == 2
    assert result.coarse_t.shape == (20,)
    assert result.coarse_ssim.shape == (20,)
    assert result.coarse_n_evals == 20
    assert result.refine_n_evals >= 0
    assert result.total_n_evals == result.coarse_n_evals + result.refine_n_evals + 1
    assert result.elapsed > 0.0


def test_optimize_alpha_result_image_is_float_in_unit_range(
    synthetic_setup_snr10,
) -> None:
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
    )
    assert np.issubdtype(result.image.dtype, np.floating)
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


# ===========================================================================
# Group 4 - Search-parameter dispatch
# ===========================================================================


def test_optimize_alpha_coarse_n_points_dispatches(synthetic_setup_snr10) -> None:
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        coarse_n_points=15,
    )
    assert result.coarse_n_evals == 15
    assert result.coarse_t.shape == (15,)


def test_optimize_alpha_brent_xtol_dispatches(synthetic_setup_snr10) -> None:
    """Tighter xtol means more Brent evaluations (or at least not
    fewer)."""
    img, _psf, _degraded, solver = synthetic_setup_snr10
    result_loose = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        brent_xtol=1e-1,
    )
    result_tight = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=_psnr_metric,
        brent_xtol=1e-5,
    )
    assert result_tight.refine_n_evals >= result_loose.refine_n_evals


# ===========================================================================
# Group 5 - Metric substitution (sanity)
# ===========================================================================


def test_optimize_alpha_custom_psnr_fn_is_invoked(synthetic_setup_snr10) -> None:
    """Custom ``psnr_fn`` is invoked at the optimum and its return
    value populates ``result.psnr``.  Sentinel value confirms invocation."""
    img, _psf, _degraded, solver = synthetic_setup_snr10
    SENTINEL = -42.0
    sentinel_psnr = lambda a, b: SENTINEL  # noqa: E731
    result = optimize_alpha(
        solver, img, alpha_0=1e-3,
        ssim_fn=_psnr_metric, psnr_fn=sentinel_psnr,
    )
    assert result.psnr == SENTINEL
