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

def show_image(image, title=None):
    plt.imshow(image, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-6:
        return np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
    else:
        return np.zeros_like(image)

def image_quality_metrics(reconstructed, reference):
    """Compute PSNR and MS-SSIM between reconstructed and reference images."""
    psnr = PiqPSNR(reconstructed, reference)
    msssim = MSSSIM(reconstructed, reference)
    fsim = FSIM(reconstructed, reference)
    vif = VIF(reconstructed, reference)
    msgmsd = MSGMSD(reconstructed, reference)
    return psnr, msssim, fsim, vif, msgmsd

def _resolve_metrics(
    ssim_fn: Optional[Callable],
    psnr_fn: Optional[Callable],
) -> tuple[Callable, Callable]:
    """Import default metric functions if not provided."""
    if ssim_fn is None:
        from RemondoPythonCore.Common.Image_Quality_Measures import MSSSIM
        ssim_fn = MSSSIM
    if psnr_fn is None:
        from RemondoPythonCore.Common.Image_Quality_Measures import PiqPSNR
        psnr_fn = PiqPSNR
    return ssim_fn, psnr_fn

def _coarse_search(
    solver,
    ref_image: np.ndarray,
    ssim_fn: Callable,
    t_center: float,
    half_width: float,
    n_points: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    t_grid = np.linspace(t_center - half_width, t_center + half_width, n_points)
    ssim_grid = np.empty(n_points)

    for i, t in enumerate(t_grid):
        result = solver.deblur(alpha=10.0 ** t)
        ssim_grid[i] = float(ssim_fn(normalize_image(result), ref_image))

    i_best = int(np.argmax(ssim_grid))

    if verbose:
        print(
            f"  Coarse search ({n_points} pts): "
            f"best t = {t_grid[i_best]:.4f} "
            f"(α = {10.0 ** t_grid[i_best]:.4e}), "
            f"MS-SSIM = {ssim_grid[i_best]:.6f}"
        )

    return t_grid, ssim_grid, i_best

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Local Brent refinement
# ══════════════════════════════════════════════════════════════════════════════

def _brent_refine(
    solver,
    ref_image: np.ndarray,
    ssim_fn: Callable,
    t_lo: float,
    t_hi: float,
    xtol: float,
    verbose: bool,
) -> tuple[float, float, int]:
    n_evals = 0

    def objective(t: float) -> float:
        nonlocal n_evals
        result = solver.deblur(alpha=10.0 ** t)
        ssim_val = float(ssim_fn(normalize_image(result), ref_image))
        n_evals += 1
        return -ssim_val

    opt = minimize_scalar(
        objective,
        bounds=(t_lo, t_hi),
        method="bounded",
        options={"xatol": xtol, "maxiter": 50},
    )

    t_opt = opt.x
    ssim_opt = -opt.fun

    if verbose:
        print(
            f"  Brent refine ({n_evals} evals): "
            f"t = {t_opt:.6f} "
            f"(α = {10.0 ** t_opt:.6e}), "
            f"MS-SSIM = {ssim_opt:.6f}"
        )

    return t_opt, ssim_opt, n_evals

def optimize_wiener_alpha(
    solver,
    ref_image: np.ndarray,
    alpha_0: float,
    ssim_fn: Optional[Callable] = None,
    psnr_fn: Optional[Callable] = None,
    coarse_half_width: float = 3.0,
    coarse_n_points: int = 80,
    brent_xtol: float = 1e-3,
    verbose: bool = True,
) -> dict:
    ssim_fn, psnr_fn = _resolve_metrics(ssim_fn, psnr_fn)
    t0_wall = time.perf_counter()
    t_center = np.log10(alpha_0)

    if verbose:
        print(f"Wiener α optimisation (two-stage)")
        print(f"  α₀ = {alpha_0:.4e}  (t₀ = {t_center:.4f})")
        print(f"  Coarse: {coarse_n_points} pts in "
              f"[{10.0 ** (t_center - coarse_half_width):.2e}, "
              f"{10.0 ** (t_center + coarse_half_width):.2e}]")

    # ── Stage 1: coarse grid ──────────────────────────────────────────────
    t_grid, ssim_grid, i_best = _coarse_search(
        solver, ref_image, ssim_fn,
        t_center, coarse_half_width, coarse_n_points, verbose,
    )

    # ── Stage 2: Brent refinement in the winning grid cell ────────────────
    dt = t_grid[1] - t_grid[0]  # grid spacing
    t_lo = t_grid[i_best] - dt
    t_hi = t_grid[i_best] + dt

    t_opt, ssim_opt, refine_evals = _brent_refine(
        solver, ref_image, ssim_fn,
        t_lo, t_hi, brent_xtol, verbose,
    )

    # ── Final evaluation at the optimum ───────────────────────────────────
    best_alpha = 10.0 ** t_opt
    best_result = normalize_image(solver.deblur(alpha=best_alpha))
    best_ssim = float(ssim_fn(best_result, ref_image))
    best_psnr = float(psnr_fn(best_result, ref_image))

    elapsed = time.perf_counter() - t0_wall
    total_evals = coarse_n_points + refine_evals + 1

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  Optimal α  = {best_alpha:.6e}  (log₁₀ = {t_opt:.4f})")
        print(f"  MS-SSIM    = {best_ssim:.6f}")
        print(f"  PSNR       = {best_psnr:.2f} dB")
        print(f"  Evals      : {coarse_n_points} coarse "
              f"+ {refine_evals} refine + 1 final = {total_evals}")
        print(f"  Wall time  : {elapsed:.2f} s")
        print(f"{'═' * 60}")

    return {
        "alpha": best_alpha,
        "log10_alpha": t_opt,
        "ssim": best_ssim,
        "psnr": best_psnr,
        "image": best_result,
        "coarse_t": t_grid,
        "coarse_ssim": ssim_grid,
        "coarse_n_evals": coarse_n_points,
        "refine_n_evals": refine_evals,
        "total_n_evals": total_evals,
        "elapsed": elapsed,
    }

input_image_path = r"C:\Users\chaim\Downloads\city_30cm_ROI2.tif"
input_psf_path = r"C:\Users\chaim\Datasets\PSFs\TMA_R1_150_50\detector_psf_tiffs\psf_det_inner_r_0350.000mm_step0005.tif"

output_parent_dir = Path(r"C:\Users\chaim\Results")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = output_parent_dir / f"Reconstruction_{stamp}"
output_dir.mkdir(parents=True, exist_ok=True)

noise_sigma: float = 0.01        # std-dev of AWGN (on [0,1] scale)
noise_seed: int = 42
rng = np.random.default_rng(noise_seed)

# Load and preprocess the input image to obtain the reference image
scene_raw, _, _ = load_image(filename=str(input_image_path), trnasform_to_grayscale=True, normlize_image=True)
ref_image = odd_crop(scene_raw)

ref_image = image_normalization(to_grayscale(ref_image))
ref_image = normalize_image(ref_image)

tifffile.imwrite(str(output_dir / f"reference.tif"), ref_image.astype(np.float32))
show_image(ref_image, title="Reference Image")

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

tifffile.imwrite(str(output_dir / f"PSF.tif"), psf_np.astype(np.float32))
show_image(psf_np, title="PSF")

results = []

# Create blurred image by convolving reference image with PSF (circular-boundary approx) and normalise to [0, 1]
t0 = time.perf_counter()
blurred = fftconvolve(ref_image, psf_np, mode="same")
elapsed = time.perf_counter() - t0
blurred_psnr, blurred_msssim, blurred_fsim, blurred_vif, blurred_msgmsd = image_quality_metrics(blurred, ref_image)
results.append(("Blurred", elapsed, blurred_psnr, blurred_msssim, blurred_fsim, blurred_vif, blurred_msgmsd))
print(f"Blurred Image Quality:\n  PSNR: {blurred_psnr:.2f} dB\n  MS-SSIM: {blurred_msssim:.4f}, FSIM: {blurred_fsim:.4f}, VIF: {blurred_vif:.4f}, MSGMSD: {blurred_msgmsd:.6f}")
tifffile.imwrite(str(output_dir / f"degraded.tif"), blurred.astype(np.float32))
show_image(blurred, title="Degraded Image")	

# Add AWGN to the blurred image
t0 = time.perf_counter()
degraded = blurred + rng.normal(0.0, noise_sigma, blurred.shape)
elapsed = time.perf_counter() - t0
degraded_psnr, degraded_msssim, degraded_fsim, degraded_vif, degraded_msgmsd = image_quality_metrics(degraded, ref_image)
results.append(("Degraded", elapsed, degraded_psnr, degraded_msssim, degraded_fsim, degraded_vif, degraded_msgmsd))
print(f"Degraded Image Quality:\n  PSNR: {degraded_psnr:.2f} dB\n  MS-SSIM: {degraded_msssim:.4f}, FSIM: {degraded_fsim:.4f}, VIF: {degraded_vif:.4f}, MSGMSD: {degraded_msgmsd:.6f}")
tifffile.imwrite(str(output_dir / f"degraded_noisy.tif"), degraded.astype(np.float32))
show_image(degraded, title="Degraded Image with AWGN")

# Wiener deconvolution  (Classical, no regularization)
solver = WienerDeconv(degraded, psf_np, mode="Classical", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
alpha_0 = solver.last_alpha
print(f"Initial α from Wiener formula = {alpha_0:.4e}")
opt = optimize_wiener_alpha(solver, ref_image, alpha_0=alpha_0)
print(f"Optimal α = {opt['alpha']:.6e}, SSIM = {opt['ssim']:.6f}")
result = opt["image"]
print(f"Wiener Deconvolution (Classical) completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
wiener_classical_psnr, wiener_classical_msssim, wiener_classical_fsim, wiener_classical_vif, wiener_classical_msgmsd = image_quality_metrics(result, ref_image)
print(f"Wiener Deconvolution (Classical) Quality:\n  PSNR: {wiener_classical_psnr:.2f} dB\n  MS-SSIM: {wiener_classical_msssim:.4f}, FSIM: {wiener_classical_fsim:.4f}, VIF: {wiener_classical_vif:.4f}, MSGMSD: {wiener_classical_msgmsd:.6f}")
results.append(("Wiener (Classical)", elapsed, wiener_classical_psnr, wiener_classical_msssim, wiener_classical_fsim, wiener_classical_vif, wiener_classical_msgmsd))
tifffile.imwrite(str(output_dir / f"wiener_classical.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Classical)\nElapsed time: {elapsed:.2f} seconds")

# Wiener deconvolution (Tikhonov regularization)
solver = WienerDeconv(degraded, psf_np, mode="Tikhonov", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
print(f"Wiener Deconvolution (Tikhonov) completed in {elapsed:.2f} seconds.")
alpha_0 = solver.last_alpha
print(f"Initial α from Wiener formula = {alpha_0:.4e}")
opt = optimize_wiener_alpha(solver, ref_image, alpha_0=alpha_0)
print(f"Optimal α = {opt['alpha']:.6e}, SSIM = {opt['ssim']:.6f}")
result = opt["image"]
result = normalize_image(result)  # ensure result is in [0, 1] range
wiener_tikhonov_psnr, wiener_tikhonov_msssim, wiener_tikhonov_fsim, wiener_tikhonov_vif, wiener_tikhonov_msgmsd = image_quality_metrics(result, ref_image)
print(f"Wiener Deconvolution (Tikhonov) Quality:\n  PSNR: {wiener_tikhonov_psnr:.2f} dB\n  MS-SSIM: {wiener_tikhonov_msssim:.4f}, FSIM: {wiener_tikhonov_fsim:.4f}, VIF: {wiener_tikhonov_vif:.4f}, MSGMSD: {wiener_tikhonov_msgmsd:.6f}")
results.append(("Wiener (Tikhonov)", elapsed, wiener_tikhonov_psnr, wiener_tikhonov_msssim, wiener_tikhonov_fsim, wiener_tikhonov_vif, wiener_tikhonov_msgmsd))
tifffile.imwrite(str(output_dir / f"wiener_tikhonov.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Tikhonov)\nElapsed time: {elapsed:.2f} seconds")

# Richardson-Lucy deconvolution with unknown boundary conditions
solver = RLUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=500, lambda_tv=1e-3)
elapsed = time.perf_counter() - t0
print(f"Richardson-Lucy Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
rl_psnr, rl_msssim, rl_fsim, rl_vif, rl_msgmsd = image_quality_metrics(result, ref_image)
print(f"Richardson-Lucy Deconvolution Quality:\n  PSNR: {rl_psnr:.2f} dB\n  MS-SSIM: {rl_msssim:.4f}, FSIM: {rl_fsim:.4f}, VIF: {rl_vif:.4f}, MSGMSD: {rl_msgmsd:.6f}")
results.append(("Richardson-Lucy", elapsed, rl_psnr, rl_msssim, rl_fsim, rl_vif, rl_msgmsd))
tifffile.imwrite(str(output_dir / f"richardson_lucy.tif"), result.astype(np.float32))
show_image(result, title=f"Richardson-Lucy Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# Landweber deconvolution with unknown boundary conditions
solver = LandweberUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=1e-3, precondition=True, adaptive_restart=True)
elapsed = time.perf_counter() - t0
print(f"Landweber Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
landweber_psnr, landweber_msssim, landweber_fsim, landweber_vif, landweber_msgmsd = image_quality_metrics(result, ref_image)
print(f"Landweber Deconvolution Quality:\n  PSNR: {landweber_psnr:.2f} dB\n  MS-SSIM: {landweber_msssim:.4f}, FSIM: {landweber_fsim:.4f}, VIF: {landweber_vif:.4f}, MSGMSD: {landweber_msgmsd:.6f}")
results.append(("Landweber", elapsed, landweber_psnr, landweber_msssim, landweber_fsim, landweber_vif, landweber_msgmsd))
tifffile.imwrite(str(output_dir / f"landweber.tif"), result.astype(np.float32))
show_image(result, title=f"Landweber Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# ADMM with Total Variation regularisation
solver = ADMMDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=1500,  lambda_tv=8.9e-04, TVnorm=2)
elapsed = time.perf_counter() - t0
print(f"ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
admm_psnr, admm_msssim, admm_fsim, admm_vif, admm_msgmsd = image_quality_metrics(result, ref_image)
print(f"ADMM Deconvolution Quality:\n  PSNR: {admm_psnr:.2f} dB\n  MS-SSIM: {admm_msssim:.4f}, FSIM: {admm_fsim:.4f}, VIF: {admm_vif:.4f}, MSGMSD: {admm_msgmsd:.6f}")
results.append(("ADMM (TV)", elapsed, admm_psnr, admm_msssim, admm_fsim, admm_vif, admm_msgmsd))
tifffile.imwrite(str(output_dir / f"admm_tv.tif"), result.astype(np.float32))
show_image(result, title=f"ADMM Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# TVAL3 (augmented-Lagrangian TV minimisation)
solver = TVAL3Deconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=1500, lambda_tv=6.0e-04, TVnorm=2, adaptive_tv=True, burn_in_frac=0.2)
elapsed = time.perf_counter() - t0
print(f"TVAL3 Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
tval3_psnr, tval3_msssim, tval3_fsim, tval3_vif, tval3_msgmsd = image_quality_metrics(result, ref_image)
print(f"TVAL3 Deconvolution Quality:\n  PSNR: {tval3_psnr:.2f} dB\n  MS-SSIM: {tval3_msssim:.4f}, FSIM: {tval3_fsim:.4f}, VIF: {tval3_vif:.4f}, MSGMSD: {tval3_msgmsd:.6f}")
results.append(("TVAL3", elapsed, tval3_psnr, tval3_msssim, tval3_fsim, tval3_vif, tval3_msgmsd))
tifffile.imwrite(str(output_dir / f"tval3_tv.tif"), result.astype(np.float32))
show_image(result, title=f"TVAL3 Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# FISTA in TV-regularisation mode
solver = FISTADeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=600, lambda_reg=6e-4, reg_mode="TV")
elapsed = time.perf_counter() - t0
print(f"FISTA Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
fista_psnr, fista_msssim, fista_fsim, fista_vif, fista_msgmsd = image_quality_metrics(result, ref_image)
print(f"FISTA Deconvolution Quality:\n  PSNR: {fista_psnr:.2f} dB\n  MS-SSIM: {fista_msssim:.4f}, FSIM: {fista_fsim:.4f}, VIF: {fista_vif:.4f}, MSGMSD: {fista_msgmsd:.6f}")
results.append(("FISTA (TV)", elapsed, fista_psnr, fista_msssim, fista_fsim, fista_vif, fista_msgmsd))
tifffile.imwrite(str(output_dir / f"fista_tv.tif"), result.astype(np.float32))
show_image(result, title=f"FISTA Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# FISTA in L1-wavelet regularisation mode
solver = FISTADeconv(
    degraded, psf_np,
    wavelet="bior4.4",
    wavelet_levels=4,
    paddingMode="Reflect",
    padding_scale=2.0,
)
t0 = time.perf_counter()
result = solver.deblur(num_iter=650, lambda_reg=8e-4, reg_mode="L1_wavelet")
elapsed = time.perf_counter() - t0
print(f"FISTA-L1-Wavelet Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
fista_wavelet_psnr, fista_wavelet_msssim, fista_wavelet_fsim, fista_wavelet_vif, fista_wavelet_msgmsd = image_quality_metrics(result, ref_image)
print(f"FISTA-L1-Wavelet Deconvolution Quality:\n  PSNR: {fista_wavelet_psnr:.2f} dB\n  MS-SSIM: {fista_wavelet_msssim:.4f}, FSIM: {fista_wavelet_fsim:.4f}, VIF: {fista_wavelet_vif:.4f}, MSGMSD: {fista_wavelet_msgmsd:.6f}")
results.append(("FISTA (L1-Wavelet)", elapsed, fista_wavelet_psnr, fista_wavelet_msssim, fista_wavelet_fsim, fista_wavelet_vif, fista_wavelet_msgmsd))
tifffile.imwrite(str(output_dir / f"fista_l1_wavelet.tif"), result.astype(np.float32))
show_image(result, title=f"FISTA Deconvolution (L1-Wavelet)\nElapsed time: {elapsed:.2f} seconds")

# Chambolle-Pock (Condat-Vũ) primal-dual algorithm in TV-regularisation mode
solver = ChambollePockDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=110, lambda_tv=0.00015)
elapsed = time.perf_counter() - t0
print(f"Chambolle-Pock Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
chambolle_pock_psnr, chambolle_pock_msssim, chambolle_pock_fsim, chambolle_pock_vif, chambolle_pock_msgmsd = image_quality_metrics(result, ref_image)
print(f"Chambolle-Pock Deconvolution Quality:\n  PSNR: {chambolle_pock_psnr:.2f} dB\n  MS-SSIM: {chambolle_pock_msssim:.4f}, FSIM: {chambolle_pock_fsim:.4f}, VIF: {chambolle_pock_vif:.4f}, MSGMSD: {chambolle_pock_msgmsd:.6f}")
results.append(("Chambolle-Pock", elapsed, chambolle_pock_psnr, chambolle_pock_msssim, chambolle_pock_fsim, chambolle_pock_vif, chambolle_pock_msgmsd))
tifffile.imwrite(str(output_dir / f"chambolle_pock_tv.tif"), result.astype(np.float32))
show_image(result, title=f"Chambolle-Pock Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# Plug-and-Play ADMM with BM3D denoiser
solver = PnPADMM(
    degraded, psf_np,
    rho_z=0.5,
    sigma_scale=0.1,
    rho_v=1.0,
    paddingMode="Reflect",
    padding_scale=2.0,
)
t0 = time.perf_counter()
result = solver.deblur(num_iter=8, lambda_tv=0.002)
elapsed = time.perf_counter() - t0
print(f"Plug-and-Play ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
pnp_admm_psnr, pnp_admm_msssim, pnp_admm_fsim, pnp_admm_vif, pnp_admm_msgmsd = image_quality_metrics(result, ref_image)
print(f"Plug-and-Play ADMM Deconvolution Quality:\n  PSNR: {pnp_admm_psnr:.2f} dB\n  MS-SSIM: {pnp_admm_msssim:.4f}, FSIM: {pnp_admm_fsim:.4f}, VIF: {pnp_admm_vif:.4f}, MSGMSD: {pnp_admm_msgmsd:.6f}")
results.append(("PnP-ADMM (BM3D)", elapsed, pnp_admm_psnr, pnp_admm_msssim, pnp_admm_fsim, pnp_admm_vif, pnp_admm_msgmsd))
tifffile.imwrite(str(output_dir / f"pnp_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"Plug-and-Play ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# RED-ADMM with BM3D denoiser
solver = REDDeconv(
    degraded, psf_np,
    sigma=0.005,          # closer to actual noise level
    rho_v=0.5,
    paddingMode="Reflect",
    padding_scale=2.0,
)
t0 = time.perf_counter()
result = solver.deblur(num_iter=3, lambda_reg=0.0001)
elapsed = time.perf_counter() - t0
print(f"RED-ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
red_admm_psnr, red_admm_msssim, red_admm_fsim, red_admm_vif, red_admm_msgmsd = image_quality_metrics(result, ref_image)
print(f"RED-ADMM Deconvolution Quality:\n  PSNR: {red_admm_psnr:.2f} dB\n  MS-SSIM: {red_admm_msssim:.4f}, FSIM: {red_admm_fsim:.4f}, VIF: {red_admm_vif:.4f}, MSGMSD: {red_admm_msgmsd:.6f}")
results.append(("RED-ADMM (BM3D)", elapsed, red_admm_psnr, red_admm_msssim, red_admm_fsim, red_admm_vif, red_admm_msgmsd))
tifffile.imwrite(str(output_dir / f"red_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"RED-ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# Print and save results summary table
header = ("Method", "Elapsed (s)", "PSNR (dB)", "MS-SSIM", "FSIM", "VIF", "MSGMSD")
col_widths = (25, 12, 10, 10, 10, 10, 10)
fmt = f"{{:<{col_widths[0]}}} {{:>{col_widths[1]}}} {{:>{col_widths[2]}}} {{:>{col_widths[3]}}} {{:>{col_widths[4]}}} {{:>{col_widths[5]}}} {{:>{col_widths[6]}}}"
sep = "-" * (sum(col_widths) + len(col_widths) - 1)

print(f"\n{sep}")
print(fmt.format(*header))
print(sep)
for name, t, psnr, msssim, fsim, vif, msgmsd in results:
    print(fmt.format(name, f"{t:.2f}", f"{psnr:.2f}", f"{msssim:.4f}", f"{fsim:.4f}", f"{vif:.4f}", f"{msgmsd:.6f}"))
print(sep)

csv_path = output_dir / "results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for name, t, psnr, msssim, fsim, vif, msgmsd in results:
        writer.writerow([name, f"{t:.2f}", f"{psnr:.2f}", f"{msssim:.4f}", f"{fsim:.4f}", f"{vif:.4f}", f"{msgmsd:.6f}"])
print(f"\nResults saved to {csv_path}")
