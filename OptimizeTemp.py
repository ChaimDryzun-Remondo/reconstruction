from __future__ import annotations

import time
import logging
from typing import Optional, Callable

import numpy as np
from scipy.optimize import minimize_scalar


import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Callable

import csv

import numpy as np
import tifffile

import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar

from RemondoPythonCore.Common.Image_Preprocessing import to_grayscale, image_normalization
from RemondoPythonCore.Common.PSF_Preprocessing import condition_psf, psf_preprocess
from RemondoPythonCore.Common.IO import load_image
from RemondoPythonCore.Common.General_Utilities import odd_crop
from RemondoPythonCore.Common.Image_Quality_Measures import PiqPSNR, MSSSIM, FSIM, VIF, MSGMSD
from RemondoPythonCore.reconstruction import (
    WienerDeconv,
    RLUnknownBoundary,
    LandweberUnknownBoundary,
    ADMMDeconv,
    TVAL3Deconv,
    FISTADeconv,
    ChambollePockDeconv,
)
from RemondoPythonCore.reconstruction import PnPADMM, REDDeconv


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    lo, hi = image.min(), image.max()
    if hi - lo > 1e-6:
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(image)


def _resolve_metric(
    metric_fn: Optional[Callable],
) -> Callable:
    """Import default MS-GMSD metric if not provided."""
    if metric_fn is None:
        from RemondoPythonCore.Common.Image_Quality_Measures import MSGMSD
        metric_fn = MSGMSD
    return metric_fn


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Coarse sweep over lambda_tv (num_iter fixed)
# ══════════════════════════════════════════════════════════════════════════════

def _sweep_lambda(
    solver,
    ref_image: np.ndarray,
    metric_fn: Callable,
    num_iter: int,
    log_range: tuple[float, float],
    n_points: int,
    tvnorm: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Evaluate MS-GMSD over a log-spaced grid of lambda_tv values.

    Parameters
    ----------
    solver : ADMMDeconv
        Pre-constructed ADMM solver.
    ref_image : np.ndarray
        Ground-truth reference image.
    metric_fn : callable
        metric_fn(reconstructed, reference) → float (lower is better).
    num_iter : int
        Fixed iteration count for this sweep.
    log_range : (float, float)
        (log10_min, log10_max) for lambda_tv.
    n_points : int
        Number of grid points.
    tvnorm : int
        TV norm variant (1 or 2).
    verbose : bool
        Print progress.

    Returns
    -------
    t_grid : ndarray — log10(lambda_tv) values
    scores : ndarray — MS-GMSD at each point
    i_best : int — index of the minimum
    """
    t_grid = np.linspace(log_range[0], log_range[1], n_points)
    scores = np.empty(n_points)

    t0 = time.perf_counter()
    for i, t in enumerate(t_grid):
        lam = 10.0 ** t
        result = solver.deblur(num_iter=num_iter, lambda_tv=lam, TVnorm=tvnorm)
        scores[i] = float(metric_fn(_normalize_image(result), ref_image))

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    λ sweep: {i+1}/{n_points} done ({elapsed:.0f} s)")

    i_best = int(np.argmin(scores))

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"  Stage 1 complete ({elapsed:.0f} s, {n_points} evals @ "
            f"num_iter={num_iter}): "
            f"best log₁₀(λ) = {t_grid[i_best]:.4f} "
            f"(λ = {10.0 ** t_grid[i_best]:.4e}), "
            f"MS-GMSD = {scores[i_best]:.6f}"
        )

    return t_grid, scores, i_best


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Sweep over num_iter (lambda_tv fixed)
# ══════════════════════════════════════════════════════════════════════════════

def _sweep_num_iter(
    solver,
    ref_image: np.ndarray,
    metric_fn: Callable,
    lambda_tv: float,
    iter_range: tuple[int, int],
    iter_step: int,
    tvnorm: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Evaluate MS-GMSD over a range of iteration counts.

    Returns
    -------
    iters : ndarray — iteration counts tested
    scores : ndarray — MS-GMSD at each count
    i_best : int — index of the minimum
    """
    iters = np.arange(iter_range[0], iter_range[1] + 1, iter_step)
    scores = np.empty(len(iters))

    t0 = time.perf_counter()
    for i, n in enumerate(iters):
        result = solver.deblur(
            num_iter=int(n), lambda_tv=lambda_tv, TVnorm=tvnorm
        )
        scores[i] = float(metric_fn(_normalize_image(result), ref_image))

        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"    iter sweep: num_iter={int(n):4d}, "
                f"MS-GMSD={scores[i]:.6f} ({elapsed:.0f} s)"
            )

    i_best = int(np.argmin(scores))

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"  Stage 2 complete ({elapsed:.0f} s, {len(iters)} evals @ "
            f"λ={lambda_tv:.4e}): "
            f"best num_iter = {int(iters[i_best])}, "
            f"MS-GMSD = {scores[i_best]:.6f}"
        )

    return iters, scores, i_best


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Brent refinement of lambda_tv (num_iter fixed at stage-2 optimum)
# ══════════════════════════════════════════════════════════════════════════════

def _refine_lambda(
    solver,
    ref_image: np.ndarray,
    metric_fn: Callable,
    num_iter: int,
    t_center: float,
    half_width: float,
    tvnorm: int,
    xtol: float,
    verbose: bool,
) -> tuple[float, float, int]:
    """
    Refine lambda_tv using Brent's method in a narrow log-space bracket.

    Returns
    -------
    t_opt : float — optimal log10(lambda_tv)
    score_opt : float — MS-GMSD at the optimum
    n_evals : int
    """
    n_evals = 0

    def objective(t: float) -> float:
        nonlocal n_evals
        lam = 10.0 ** t
        result = solver.deblur(num_iter=num_iter, lambda_tv=lam, TVnorm=tvnorm)
        score = float(metric_fn(_normalize_image(result), ref_image))
        n_evals += 1
        return score  # MS-GMSD is already "lower is better"

    opt = minimize_scalar(
        objective,
        bounds=(t_center - half_width, t_center + half_width),
        method="bounded",
        options={"xatol": xtol, "maxiter": 50},
    )

    if verbose:
        print(
            f"  Stage 3 complete ({n_evals} evals @ num_iter={num_iter}): "
            f"log₁₀(λ) = {opt.x:.6f} "
            f"(λ = {10.0 ** opt.x:.6e}), "
            f"MS-GMSD = {opt.fun:.6f}"
        )

    return opt.x, opt.fun, n_evals


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def optimize_admm_tv(
    solver,
    ref_image: np.ndarray,
    metric_fn: Optional[Callable] = None,
    tvnorm: int = 2,
    # Stage 1: lambda sweep
    lambda_log_range: tuple[float, float] = (-5.0, -1.0),
    lambda_n_points: int = 30,
    lambda_sweep_num_iter: int = 200,
    # Stage 2: iteration sweep
    iter_range: tuple[int, int] = (50, 450),
    iter_step: int = 25,
    # Stage 3: lambda refinement
    refine_half_width: float = 0.5,
    refine_xtol: float = 0.01,
    verbose: bool = True,
) -> dict:
    """
    Three-stage optimisation of ADMM-TV: λ_TV sweep → num_iter sweep → λ_TV refinement.

    Parameters
    ----------
    solver : ADMMDeconv
        Pre-constructed solver (constructor already called with image, psf, etc.).
    ref_image : np.ndarray
        Ground-truth reference image.
    metric_fn : callable, optional
        metric_fn(reconstructed, reference) → float, LOWER is better.
        Default: MS-GMSD from RemondoPythonCore.
    tvnorm : int
        TV norm variant passed to solver.deblur().  Default 2 (isotropic).
    lambda_log_range : (float, float)
        Search range for log₁₀(λ_TV) in stage 1.  Default (-5, -1),
        corresponding to λ ∈ [1e-5, 0.1].
    lambda_n_points : int
        Number of coarse grid points in stage 1.  Default 30.
    lambda_sweep_num_iter : int
        Fixed num_iter used during the stage-1 λ sweep.  Should be large
        enough for reasonable convergence but not wastefully large.  Default 200.
    iter_range : (int, int)
        (min_iter, max_iter) for the stage-2 iteration sweep.  Default (50, 450).
    iter_step : int
        Step size for the iteration sweep.  Default 25.
    refine_half_width : float
        Half-width in decades for the stage-3 Brent refinement bracket
        around the stage-1 winner.  Default 0.5.
    refine_xtol : float
        Tolerance in log₁₀(λ) for Brent convergence.  Default 0.01.
    verbose : bool
        Print progress at each stage.

    Returns
    -------
    dict with keys:
        "lambda_tv"          : float   — optimal λ_TV
        "log10_lambda_tv"    : float   — log₁₀ of optimal λ_TV
        "num_iter"           : int     — optimal iteration count
        "msgmsd"             : float   — MS-GMSD at the optimum
        "image"              : ndarray — reconstructed image at the optimum
        "stage1_t"           : ndarray — log₁₀(λ) grid from stage 1
        "stage1_scores"      : ndarray — MS-GMSD values from stage 1
        "stage2_iters"       : ndarray — num_iter values from stage 2
        "stage2_scores"      : ndarray — MS-GMSD values from stage 2
        "stage1_evals"       : int
        "stage2_evals"       : int
        "stage3_evals"       : int
        "total_evals"        : int
        "elapsed"            : float   — total wall time (seconds)
    """
    metric_fn = _resolve_metric(metric_fn)
    t0_wall = time.perf_counter()

    if verbose:
        print(f"ADMM-TV optimisation (three-stage coordinate descent)")
        print(f"  λ range: [{10**lambda_log_range[0]:.1e}, "
              f"{10**lambda_log_range[1]:.1e}], "
              f"{lambda_n_points} pts")
        print(f"  iter range: [{iter_range[0]}, {iter_range[1]}], "
              f"step {iter_step}")
        print(f"  TVnorm = {tvnorm}")
        print()

    # ── Stage 1: Coarse λ_TV sweep at fixed num_iter ─────────────────────
    if verbose:
        print(f"Stage 1: sweeping λ_TV ({lambda_n_points} pts, "
              f"num_iter={lambda_sweep_num_iter})")

    t_grid, s1_scores, i1_best = _sweep_lambda(
        solver, ref_image, metric_fn,
        num_iter=lambda_sweep_num_iter,
        log_range=lambda_log_range,
        n_points=lambda_n_points,
        tvnorm=tvnorm,
        verbose=verbose,
    )
    lambda_stage1 = 10.0 ** t_grid[i1_best]

    if verbose:
        print()

    # ── Stage 2: num_iter sweep at stage-1 λ_TV ──────────────────────────
    if verbose:
        print(f"Stage 2: sweeping num_iter at λ_TV = {lambda_stage1:.4e}")

    s2_iters, s2_scores, i2_best = _sweep_num_iter(
        solver, ref_image, metric_fn,
        lambda_tv=lambda_stage1,
        iter_range=iter_range,
        iter_step=iter_step,
        tvnorm=tvnorm,
        verbose=verbose,
    )
    best_num_iter = int(s2_iters[i2_best])

    if verbose:
        print()

    # ── Stage 3: Refine λ_TV at optimal num_iter via Brent ────────────────
    if verbose:
        print(f"Stage 3: refining λ_TV via Brent at num_iter = {best_num_iter}")

    t_refined, score_refined, s3_evals = _refine_lambda(
        solver, ref_image, metric_fn,
        num_iter=best_num_iter,
        t_center=t_grid[i1_best],
        half_width=refine_half_width,
        tvnorm=tvnorm,
        xtol=refine_xtol,
        verbose=verbose,
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    best_lambda = 10.0 ** t_refined
    best_result = _normalize_image(
        solver.deblur(num_iter=best_num_iter, lambda_tv=best_lambda, TVnorm=tvnorm)
    )
    best_score = float(metric_fn(best_result, ref_image))

    elapsed = time.perf_counter() - t0_wall
    total_evals = lambda_n_points + len(s2_iters) + s3_evals + 1

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"  Optimal λ_TV    = {best_lambda:.6e}  "
              f"(log₁₀ = {t_refined:.4f})")
        print(f"  Optimal num_iter = {best_num_iter}")
        print(f"  MS-GMSD         = {best_score:.6f}")
        print(f"  Evaluations     : {lambda_n_points} (stage 1) "
              f"+ {len(s2_iters)} (stage 2) "
              f"+ {s3_evals} (stage 3) "
              f"+ 1 (final) = {total_evals}")
        print(f"  Wall time       : {elapsed:.0f} s "
              f"({elapsed/60:.1f} min)")
        print(f"{'═' * 65}")

    return {
        "lambda_tv": best_lambda,
        "log10_lambda_tv": t_refined,
        "num_iter": best_num_iter,
        "msgmsd": best_score,
        "image": best_result,
        "stage1_t": t_grid,
        "stage1_scores": s1_scores,
        "stage2_iters": s2_iters,
        "stage2_scores": s2_scores,
        "stage1_evals": lambda_n_points,
        "stage2_evals": len(s2_iters),
        "stage3_evals": s3_evals,
        "total_evals": total_evals,
        "elapsed": elapsed,
    }


input_image_path = r"C:\Users\chaim\Downloads\city_30cm_ROI1.tif"
input_psf_path = r"C:\Users\chaim\Datasets\PSFs\TMA_R1_150_50\detector_psf_tiffs\psf_det_inner_r_0350.000mm_step0005.tif"

noise_sigma: float = 0.01        # std-dev of AWGN (on [0,1] scale)
noise_seed: int = 42
rng = np.random.default_rng(noise_seed)

# Load and preprocess the input image to obtain the reference image
scene_raw, _, _ = load_image(filename=str(input_image_path), trnasform_to_grayscale=True, normlize_image=True)
ref_image = odd_crop(scene_raw)

ref_image = image_normalization(to_grayscale(ref_image))
ref_image = _normalize_image(ref_image)

# Load and preprocess PSF, normalise to sum to 1, and save
psf_raw, _, _ = load_image(filename=str(input_psf_path))
psf_np = psf_preprocess(
    psf=psf_raw,
    center_method="com",
    remove_negatives="clip",
    eps=1e-12,
    enforce_odd_shape=True,
)
psf_np = condition_psf(
    psf=psf_np,
    bg_ring_frac=0.15,
    taper_outer_frac=0.90,
    taper_end_frac=1.0,
)

psf_np /= (psf_np.sum() + 1e-12)  # normalise PSF to sum to 1


# Create blurred image by convolving reference image with PSF (circular-boundary approx) and normalise to [0, 1]
blurred = fftconvolve(ref_image, psf_np, mode="same")

# Add AWGN to the blurred image
degraded = blurred + rng.normal(0.0, noise_sigma, blurred.shape)


solver = TVAL3Deconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)

opt = optimize_admm_tv(
    solver, ref_image,
    lambda_log_range=(-5.0, -1.0),   # narrower range, centred on the stage-3 result
    lambda_n_points=60,
    lambda_sweep_num_iter=200,        # slightly higher pilot
    iter_range=(100, 1000),            # extend the ceiling
    iter_step=100,                     # coarser step to cover wider range
)

print(f"Best λ_TV = {opt['lambda_tv']:.6e}, num_iter = {opt['num_iter']}")
print(f"MS-GMSD = {opt['msgmsd']:.6f}")
best_image = opt["image"]
