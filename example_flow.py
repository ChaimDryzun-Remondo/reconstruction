import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from RemondoPythonCore.Common.Image_Preprocessing import to_grayscale, image_normalization
from RemondoPythonCore.Common.PSF_Preprocessing import condition_psf, psf_preprocess
from RemondoPythonCore.Common.IO import load_image
from RemondoPythonCore.Common.General_Utilities import odd_crop

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
	

input_image_path = r"C:\Users\chaim\Downloads\city_30cm_ROI1.tif"
input_psf_path = r"C:\Users\chaim\Datasets\PSFs\TMA_R1_150_50\detector_psf_tiffs\psf_det_inner_r_0350.000mm_step0005.tif"

output_parent_dir = Path(r"C:\Users\chaim\Results")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = output_parent_dir / f"Reconstruction_{stamp}"
output_dir.mkdir(parents=True, exist_ok=True)

noise_sigma: float = 0.01        # std-dev of AWGN (on [0,1] scale)
noise_seed: int = 42
rng = np.random.default_rng(seed)

# Load and preprocess the input image to obtain the reference image
scene_raw, _, _ = load_image(filename=str(input_image_path), trnasform_to_grayscale=True, normlize_image=True)
ref_image = odd_crop(scene_raw)

ref_image = image_normalization(to_grayscale(ref_image))

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

# Create blurred image by convolving reference image with PSF (circular-boundary approx) and normalise to [0, 1]
blurred = np.clip(fftconvolve(image, psf, mode="same"), 0.0, 1.0)
blurred = normalize_image(blurred)
tifffile.imwrite(str(output_dir / f"degraded.tif"), blurred.astype(np.float32))
show_image(blurred, title="Degraded Image")	

# Add AWGN to the blurred image
degraded = blurred + rng.normal(0.0, noise_sigma, blurred.shape)
degraded = np.clip(degraded, 0.0, 1.0)
tifffile.imwrite(str(output_dir / f"degraded_noisy.tif"), degraded.astype(np.float32))
show_image(degraded, title="Degraded Image with AWGN")

# Wiener deconvolution  (Classical, no regularization)
solver = WienerDeconv(degraded, psf_np, mode="Classical", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
print(f"Wiener Deconvolution (Classical) completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"wiener_classical.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Classical)\nElapsed time: {elapsed:.2f} seconds")

# Wiener deconvolution (Tikhonov regularization)
solver = WienerDeconv(degraded, psf_np, mode="Tikhonov", paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur()
elapsed = time.perf_counter() - t0
print(f"Wiener Deconvolution (Tikhonov) completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"wiener_tikhonov.tif"), result.astype(np.float32))
show_image(result, title=f"Wiener Deconvolution (Tikhonov)\nElapsed time: {elapsed:.2f} seconds")

# Richardson-Lucy deconvolution with unknown boundary conditions
solver = RLUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=1e-3)
elapsed = time.perf_counter() - t0
print(f"Richardson-Lucy Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"richardson_lucy.tif"), result.astype(np.float32))
show_image(result, title=f"Richardson-Lucy Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# Landweber deconvolution with unknown boundary conditions
solver = LandweberUnknownBoundary(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=1e-3, precondition=True, adaptive_restart=True)
elapsed = time.perf_counter() - t0
print(f"Landweber Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"landweber.tif"), result.astype(np.float32))
show_image(result, title=f"Landweber Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# ADMM with Total Variation regularisation
solver = ADMMDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250,  lambda_tv=0.01, TVnorm=2)
elapsed = time.perf_counter() - t0
print(f"ADMM Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"admm_tv.tif"), result.astype(np.float32))
show_image(result, title=f"ADMM Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# TVAL3 (augmented-Lagrangian TV minimisation)
solver = TVAL3Deconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=1e-2, adaptive_tv=True, burn_in_frac=0.2)
elapsed = time.perf_counter() - t0
print(f"TVAL3 Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"tval3_tv.tif"), result.astype(np.float32))
show_image(result, title=f"TVAL3 Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# FISTA in TV-regularisation mode
solver = FISTADeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_reg=1e-3, reg_mode="TV")
elapsed = time.perf_counter() - t0
print(f"FISTA Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"fista_tv.tif"), result.astype(np.float32))
show_image(result, title=f"FISTA Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# Chambolle-Pock (Condat-Vũ) primal-dual algorithm in TV-regularisation mode
solver = ChambollePockDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=250, lambda_tv=0.01)
elapsed = time.perf_counter() - t0
print(f"Chambolle-Pock Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"chambolle_pock_tv.tif"), result.astype(np.float32))
show_image(result, title=f"Chambolle-Pock Deconvolution (TV)\nElapsed time: {elapsed:.2f} seconds")

# Plug-and-Play ADMM with BM3D denoiser
solver = PnPADMM(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=min(150, 200), lambda_tv=0.01, sigma_scale=1.0)
elapsed = time.perf_counter() - t0
print(f"Plug-and-Play ADMM Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"pnp_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"Plug-and-Play ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")

# RED-ADMM with BM3D denoiser
solver = REDDeconv(degraded, psf_np, paddingMode="Reflect", padding_scale=2.0)
t0 = time.perf_counter()
result = solver.deblur(num_iter=min(150, 200), lambda_reg=0.01, sigma=0.05)  # BM3D is expensive
elapsed = time.perf_counter() - t0
print(f"RED-ADMM Deconvolution completed in {elapsed:.2f} seconds.")
tifffile.imwrite(str(output_dir / f"red_admm_bm3d.tif"), result.astype(np.float32))
show_image(result, title=f"RED-ADMM Deconvolution\nElapsed time: {elapsed:.2f} seconds")
