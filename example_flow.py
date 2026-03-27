import time
from datetime import datetime
from pathlib import Path

import csv

import numpy as np
import tifffile

import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from RemondoPythonCore.Common.Image_Preprocessing import to_grayscale, image_normalization
from RemondoPythonCore.Common.PSF_Preprocessing import condition_psf, psf_preprocess
from RemondoPythonCore.Common.IO import load_image
from RemondoPythonCore.Common.General_Utilities import odd_crop
from RemondoPythonCore.Common.Image_Quality_Measures import PiqPSNR, MSSSIM
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
    return psnr, msssim

input_image_path = r"C:\Users\chaim\Downloads\city_30cm_ROI1.tif"
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
blurred_psnr, blurred_msssim = image_quality_metrics(blurred, ref_image)
results.append(("Blurred", elapsed, blurred_psnr, blurred_msssim))
print(f"Blurred Image Quality:\n  PSNR: {blurred_psnr:.2f} dB\n  MS-SSIM: {blurred_msssim:.4f}")
tifffile.imwrite(str(output_dir / f"degraded.tif"), blurred.astype(np.float32))
show_image(blurred, title="Degraded Image")	

# Add AWGN to the blurred image
t0 = time.perf_counter()
degraded = blurred + rng.normal(0.0, noise_sigma, blurred.shape)
elapsed = time.perf_counter() - t0
degraded_psnr, degraded_msssim = image_quality_metrics(degraded, ref_image)
results.append(("Degraded", elapsed, degraded_psnr, degraded_msssim))
print(f"Degraded Image Quality:\n  PSNR: {degraded_psnr:.2f} dB\n  MS-SSIM: {degraded_msssim:.4f}")
tifffile.imwrite(str(output_dir / f"degraded_noisy.tif"), degraded.astype(np.float32))
show_image(degraded, title="Degraded Image with AWGN")

# Wiener deconvolution  (Classical, no regularization)
solver = WienerDeconv(degraded, psf_np, mode="Classical", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
print(f"Wiener Deconvolution (Classical) completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
wiener_classical_psnr, wiener_classical_msssim = image_quality_metrics(result, ref_image)
print(f"Wiener Deconvolution (Classical) Quality:\n  PSNR: {wiener_classical_psnr:.2f} dB\n  MS-SSIM: {wiener_classical_msssim:.4f}")
results.append(("Wiener (Classical)", elapsed, wiener_classical_psnr, wiener_classical_msssim))
tifffile.imwrite(str(output_dir / f"wiener_classical.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Classical)\nElapsed time: {elapsed:.2f} seconds")

# Wiener deconvolution (Tikhonov regularization)
solver = WienerDeconv(degraded, psf_np, mode="Tikhonov", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
print(f"Wiener Deconvolution (Tikhonov) completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
wiener_tikhonov_psnr, wiener_tikhonov_msssim = image_quality_metrics(result, ref_image)
print(f"Wiener Deconvolution (Tikhonov) Quality:\n  PSNR: {wiener_tikhonov_psnr:.2f} dB\n  MS-SSIM: {wiener_tikhonov_msssim:.4f}")
results.append(("Wiener (Tikhonov)", elapsed, wiener_tikhonov_psnr, wiener_tikhonov_msssim))
tifffile.imwrite(str(output_dir / f"wiener_tikhonov.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Tikhonov)\nElapsed time: {elapsed:.2f} seconds")

# Richardson-Lucy deconvolution with unknown boundary conditions
solver = RLUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=500, lambda_tv=1e-3)
elapsed = time.perf_counter() - t0
print(f"Richardson-Lucy Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
rl_psnr, rl_msssim = image_quality_metrics(result, ref_image)
print(f"Richardson-Lucy Deconvolution Quality:\n  PSNR: {rl_psnr:.2f} dB\n  MS-SSIM: {rl_msssim:.4f}")
results.append(("Richardson-Lucy", elapsed, rl_psnr, rl_msssim))
tifffile.imwrite(str(output_dir / f"richardson_lucy.tif"), result.astype(np.float32))
show_image(result, title=f"Richardson-Lucy Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# Landweber deconvolution with unknown boundary conditions
solver = LandweberUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=1e-3, precondition=True, adaptive_restart=True)
elapsed = time.perf_counter() - t0
print(f"Landweber Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
landweber_psnr, landweber_msssim = image_quality_metrics(result, ref_image)
print(f"Landweber Deconvolution Quality:\n  PSNR: {landweber_psnr:.2f} dB\n  MS-SSIM: {landweber_msssim:.4f}")
results.append(("Landweber", elapsed, landweber_psnr, landweber_msssim))
tifffile.imwrite(str(output_dir / f"landweber.tif"), result.astype(np.float32))
show_image(result, title=f"Landweber Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# ADMM with Total Variation regularisation
solver = ADMMDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=120,  lambda_tv=0.00015, TVnorm=2)
elapsed = time.perf_counter() - t0
print(f"ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
admm_psnr, admm_msssim = image_quality_metrics(result, ref_image)
print(f"ADMM Deconvolution Quality:\n  PSNR: {admm_psnr:.2f} dB\n  MS-SSIM: {admm_msssim:.4f}")
results.append(("ADMM (TV)", elapsed, admm_psnr, admm_msssim))
tifffile.imwrite(str(output_dir / f"admm_tv.tif"), result.astype(np.float32))
show_image(result, title=f"ADMM Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# TVAL3 (augmented-Lagrangian TV minimisation)
solver = TVAL3Deconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=150, lambda_tv=1e-4, adaptive_tv=True, burn_in_frac=0.1)
elapsed = time.perf_counter() - t0
print(f"TVAL3 Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
tval3_psnr, tval3_msssim = image_quality_metrics(result, ref_image)
print(f"TVAL3 Deconvolution Quality:\n  PSNR: {tval3_psnr:.2f} dB\n  MS-SSIM: {tval3_msssim:.4f}")
results.append(("TVAL3", elapsed, tval3_psnr, tval3_msssim))
tifffile.imwrite(str(output_dir / f"tval3_tv.tif"), result.astype(np.float32))
show_image(result, title=f"TVAL3 Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# FISTA in TV-regularisation mode
solver = FISTADeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=400, lambda_reg=1e-3, reg_mode="TV")
elapsed = time.perf_counter() - t0
print(f"FISTA Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
fista_psnr, fista_msssim = image_quality_metrics(result, ref_image)
print(f"FISTA Deconvolution Quality:\n  PSNR: {fista_psnr:.2f} dB\n  MS-SSIM: {fista_msssim:.4f}")
results.append(("FISTA (TV)", elapsed, fista_psnr, fista_msssim))
tifffile.imwrite(str(output_dir / f"fista_tv.tif"), result.astype(np.float32))
show_image(result, title=f"FISTA Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# Chambolle-Pock (Condat-Vũ) primal-dual algorithm in TV-regularisation mode
solver = ChambollePockDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=120, lambda_tv=0.0001)
elapsed = time.perf_counter() - t0
print(f"Chambolle-Pock Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
chambolle_pock_psnr, chambolle_pock_msssim = image_quality_metrics(result, ref_image)
print(f"Chambolle-Pock Deconvolution Quality:\n  PSNR: {chambolle_pock_psnr:.2f} dB\n  MS-SSIM: {chambolle_pock_msssim:.4f}")
results.append(("Chambolle-Pock", elapsed, chambolle_pock_psnr, chambolle_pock_msssim))
tifffile.imwrite(str(output_dir / f"chambolle_pock_tv.tif"), result.astype(np.float32))
show_image(result, title=f"Chambolle-Pock Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# Plug-and-Play ADMM with BM3D denoiser
solver = PnPADMM(
    degraded, psf_np,
    rho_z=0.5,
    sigma_scale=1.0,
    rho_v=1.0,
    paddingMode="Reflect",
    padding_scale=2.0,
)
t0 = time.perf_counter()
result = solver.deblur(num_iter=2, lambda_tv=0.005)
elapsed = time.perf_counter() - t0
print(f"Plug-and-Play ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
pnp_admm_psnr, pnp_admm_msssim = image_quality_metrics(result, ref_image)
print(f"Plug-and-Play ADMM Deconvolution Quality:\n  PSNR: {pnp_admm_psnr:.2f} dB\n  MS-SSIM: {pnp_admm_msssim:.4f}")
results.append(("PnP-ADMM (BM3D)", elapsed, pnp_admm_psnr, pnp_admm_msssim))
tifffile.imwrite(str(output_dir / f"pnp_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"Plug-and-Play ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# RED-ADMM with BM3D denoiser
solver = REDDeconv(
    degraded, psf_np,
    sigma=0.02,          # closer to actual noise level
    rho_v=1.0,
    paddingMode="Reflect",
    padding_scale=2.0,
)
t0 = time.perf_counter()
result = solver.deblur(num_iter=2, lambda_reg=0.005)
elapsed = time.perf_counter() - t0
print(f"RED-ADMM Deconvolution completed in {elapsed:.2f} seconds.")
result = normalize_image(result)  # ensure result is in [0, 1] range
red_admm_psnr, red_admm_msssim = image_quality_metrics(result, ref_image)
print(f"RED-ADMM Deconvolution Quality:\n  PSNR: {red_admm_psnr:.2f} dB\n  MS-SSIM: {red_admm_msssim:.4f}")
results.append(("RED-ADMM (BM3D)", elapsed, red_admm_psnr, red_admm_msssim))
tifffile.imwrite(str(output_dir / f"red_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"RED-ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# Print and save results summary table
header = ("Method", "Elapsed (s)", "PSNR (dB)", "MS-SSIM")
col_widths = (25, 12, 10, 10)
fmt = f"{{:<{col_widths[0]}}} {{:>{col_widths[1]}}} {{:>{col_widths[2]}}} {{:>{col_widths[3]}}}"
sep = "-" * (sum(col_widths) + len(col_widths) - 1)

print(f"\n{sep}")
print(fmt.format(*header))
print(sep)
for name, t, psnr, msssim in results:
    print(fmt.format(name, f"{t:.2f}", f"{psnr:.2f}", f"{msssim:.4f}"))
print(sep)

csv_path = output_dir / "results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for name, t, psnr, msssim in results:
        writer.writerow([name, f"{t:.2f}", f"{psnr:.2f}", f"{msssim:.4f}"])
print(f"\nResults saved to {csv_path}")
