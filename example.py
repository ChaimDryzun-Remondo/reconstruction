"""
Reconstruction Package — Example: Deconvolution of a Blurred Noisy Image
=========================================================================

Demonstrates all algorithms in the Reconstruction package on the scikit-image
"rocket" test image, degraded with an Airy-disk PSF and additive white
Gaussian noise (AWGN).

Pipeline:
    1. Load "rocket" from skimage.data, convert to grayscale, normalize to [0, 1].
    2. Enforce odd spatial dimensions (required by DeconvBase).
    3. Synthesise an Airy-disk PSF via the jinc² formula.
    4. Blur the image (circular FFT convolution) and add AWGN.
    5. Run each reconstruction algorithm and collect results.
    6. Compute quality metrics (PSNR, SSIM) against the ground truth.
    7. Display all results in a tiled matplotlib figure.

Usage:
    python reconstruction_example.py

Requirements:
    - numpy, scipy, scikit-image, matplotlib
    - The Reconstruction package importable from Shared.Reconstruction
    - RemondoPythonCore.Common (Image_Preprocessing, General_Utilities)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.signal import fftconvolve

from RemondoPythonCore.Common.Image_Preprocessing import to_grayscale, image_normalization
from RemondoPythonCore.Common.General_Utilities import odd_crop_around_center

from RemondoPythonCore.reconstruction import (
    WienerDeconv,
    RLUnknownBoundary,
    LandweberUnknownBoundary,
    ADMMDeconv,
    TVAL3Deconv,
    FISTADeconv,
    ChambollePockDeconv,
)

# Optional BM3D-based algorithms (require the `bm3d` package)
try:
    from RemondoPythonCore.reconstruction import PnPADMM, REDDeconv
    _HAS_BM3D = True
except ImportError:
    _HAS_BM3D = False

# Optional standard RL (may not exist — fallback to RLUnknownBoundary(use_mask=False))
_HAS_RL_STANDARD = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Central knobs for the example."""
    image_name: str = "rocket"

    # PSF parameters (Airy disk)
    psf_size: tuple[int, int] = (35, 35)
    psf_radius: float = 3.0          # pixels to first zero ring

    # Noise
    noise_sigma: float = 0.01        # std-dev of AWGN (on [0,1] scale)
    noise_seed: int = 42

    # Iterative algorithm defaults
    num_iter: int = 50
    padding_scale: float = 2.0
    padding_mode: str = "Reflect"


CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
# PSF Generation
# ══════════════════════════════════════════════════════════════════════════════

def airy_psf(size: tuple[int, int] = (35, 35), radius: float = 5.0) -> np.ndarray:
    """
    Generate an Airy-disk PSF (jinc² pattern).

    Parameters
    ----------
    size : tuple[int, int]
        (height, width) of the output array.  Must be odd in both dimensions.
    radius : float
        Distance in pixels from the centre to the first zero ring.
        Related to the optical system by  radius ≈ 1.22 λ f/# / pixel_pitch.

    Returns
    -------
    psf : np.ndarray, shape *size*, dtype float64, sum == 1.
    """
    h, w = size
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    r = np.sqrt(x ** 2 + y ** 2)
    arg = np.pi * r / radius
    with np.errstate(invalid="ignore", divide="ignore"):
        psf = np.where(r == 0, 1.0, (2 * j1(arg) / arg) ** 2)
    return psf / psf.sum()


# ══════════════════════════════════════════════════════════════════════════════
# Image Degradation
# ══════════════════════════════════════════════════════════════════════════════

def blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Blur *image* with *psf* using FFT convolution (circular-boundary approx)."""
    blurred = fftconvolve(image, psf, mode="same")
    return np.clip(blurred, 0.0, 1.0)


def add_awgn(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Add AWGN with standard deviation *sigma* and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Quality Metrics
# ══════════════════════════════════════════════════════════════════════════════

def psnr(reference: np.ndarray, estimate: np.ndarray,
         data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((reference - estimate) ** 2)
    if mse < 1e-15:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(reference: np.ndarray, estimate: np.ndarray,
         data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (simplified, full-image).

    Uses the classic Wang et al. (2004) formulation with default constants
    C1 = (K1 * L)^2,  C2 = (K2 * L)^2,  K1=0.01, K2=0.03, L=data_range.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = reference.mean()
    mu_y = estimate.mean()
    sig_x = reference.var()
    sig_y = estimate.var()
    sig_xy = np.mean((reference - mu_x) * (estimate - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    return float(num / den)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm Runners
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """Container for one algorithm's output and diagnostics."""
    name: str
    image: np.ndarray
    psnr_db: float
    ssim_val: float
    elapsed_s: float


def run_wiener(degraded: np.ndarray, psf: np.ndarray,
               cfg: Config) -> Result:
    """
    Wiener deconvolution (direct, non-iterative).

    Three regularisation modes are available: 'classical', 'tikhonov', 'spectrum'.
    We demonstrate 'tikhonov' with λ chosen empirically.
    """
    solver = WienerDeconv(
        degraded, psf,
        mode="Tikhonov",
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur()
    elapsed = time.perf_counter() - t0
    return Result("Wiener (Tikhonov)", result, 0.0, 0.0, elapsed)


def run_rl_unknown_boundary(degraded: np.ndarray, psf: np.ndarray,
                            cfg: Config) -> Result:
    """Richardson-Lucy with unknown-boundary mask correction."""
    solver = RLUnknownBoundary(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter * 5, lambda_tv=1e-3)
    elapsed = time.perf_counter() - t0
    return Result("RL (Unknown Bnd.)", result, 0.0, 0.0, elapsed)


def run_rl_standard(degraded: np.ndarray, psf: np.ndarray,
                    cfg: Config) -> Optional[Result]:
    """Standard RL (no boundary mask) — may not be available."""
    if _HAS_RL_STANDARD:
        solver = RLDeconv(
            degraded, psf,
            paddingMode=cfg.padding_mode,
            padding_scale=cfg.padding_scale,
        )
        t0 = time.perf_counter()
        result = solver.deblur(num_iter=cfg.num_iter*10, lambda_tv=5e-3)
        elapsed = time.perf_counter() - t0
        return Result("RL (Standard)", result, 0.0, 0.0, elapsed)
    else:
        # Fallback: use RLUnknownBoundary with use_mask=False
        solver = RLUnknownBoundary(
            degraded, psf,
            paddingMode=cfg.padding_mode,
            padding_scale=cfg.padding_scale,
            use_mask=False,
        )
        t0 = time.perf_counter()
        result = solver.deblur(num_iter=cfg.num_iter*5, lambda_tv=1e-3)
        elapsed = time.perf_counter() - t0
        return Result("RL (mask=False)", result, 0.0, 0.0, elapsed)


def run_landweber(degraded: np.ndarray, psf: np.ndarray,
                  cfg: Config) -> Result:
    """Landweber / FISTA-accelerated proximal gradient with TV regularisation."""
    solver = LandweberUnknownBoundary(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter*5, lambda_tv=1e-3, precondition=True, adaptive_restart=True)
    elapsed = time.perf_counter() - t0
    return Result("Landweber (FISTA)", result, 0.0, 0.0, elapsed)


def run_admm(degraded: np.ndarray, psf: np.ndarray,
             cfg: Config) -> Result:
    """ADMM with Total Variation regularisation."""
    solver = ADMMDeconv(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter*5,  lambda_tv=0.01, TVnorm=2)
    elapsed = time.perf_counter() - t0
    return Result("ADMM-TV", result, 0.0, 0.0, elapsed)


def run_tval3(degraded: np.ndarray, psf: np.ndarray,
              cfg: Config) -> Result:
    """TVAL3 (augmented-Lagrangian TV minimisation)."""
    solver = TVAL3Deconv(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter*5, lambda_tv=1e-2, adaptive_tv=True, burn_in_frac=0.2)
    elapsed = time.perf_counter() - t0
    return Result("TVAL3", result, 0.0, 0.0, elapsed)


def run_fista_tv(degraded: np.ndarray, psf: np.ndarray,
                 cfg: Config) -> Result:
    """FISTA in TV-regularisation mode."""
    solver = FISTADeconv(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter*4, lambda_reg=1e-3, reg_mode="TV")
    elapsed = time.perf_counter() - t0
    return Result("FISTA (TV)", result, 0.0, 0.0, elapsed)


def run_chambolle_pock(degraded: np.ndarray, psf: np.ndarray,
                       cfg: Config) -> Result:
    """Chambolle-Pock (Condat-Vũ) primal-dual splitting."""
    solver = ChambollePockDeconv(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=cfg.num_iter*4, lambda_tv=0.01)
    elapsed = time.perf_counter() - t0
    return Result("Chambolle-Pock", result, 0.0, 0.0, elapsed)


def run_pnp_admm(degraded: np.ndarray, psf: np.ndarray,
                 cfg: Config) -> Optional[Result]:
    """Plug-and-Play ADMM with BM3D denoiser (optional)."""
    if not _HAS_BM3D:
        logger.warning("PnP-ADMM skipped — bm3d package not installed.")
        return None
    solver = PnPADMM(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=min(cfg.num_iter, 30), lambda_tv=0.01, sigma_scale=1.0)  # BM3D is expensive
    elapsed = time.perf_counter() - t0
    return Result("PnP-ADMM (BM3D)", result, 0.0, 0.0, elapsed)


def run_red_admm(degraded: np.ndarray, psf: np.ndarray,
                 cfg: Config) -> Optional[Result]:
    """RED-ADMM with BM3D denoiser (optional)."""
    if not _HAS_BM3D:
        logger.warning("RED-ADMM skipped — bm3d package not installed.")
        return None
    solver = REDDeconv(
        degraded, psf,
        paddingMode=cfg.padding_mode,
        padding_scale=cfg.padding_scale,
    )
    t0 = time.perf_counter()
    result = solver.deblur(num_iter=min(cfg.num_iter, 30), lambda_reg=0.01, sigma=0.05)  # BM3D is expensive
    elapsed = time.perf_counter() - t0
    return Result("RED-ADMM (BM3D)", result, 0.0, 0.0, elapsed)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def display_results(ground_truth: np.ndarray,
                    degraded: np.ndarray,
                    results: list[Result]) -> None:
    """Show ground truth, degraded image, and all reconstruction results."""
    n = len(results) + 2  # +2 for ground truth and degraded
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.ravel()

    # Ground truth
    axes[0].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Degraded
    psnr_deg = psnr(ground_truth, degraded)
    ssim_deg = ssim(ground_truth, degraded)
    axes[1].imshow(degraded, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Degraded\nPSNR={psnr_deg:.2f} dB  SSIM={ssim_deg:.3f}",
                      fontsize=10)
    axes[1].axis("off")

    # Results
    for i, r in enumerate(results):
        ax = axes[i + 2]
        ax.imshow(r.image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{r.name}\nPSNR={r.psnr_db:.2f} dB  SSIM={r.ssim_val:.3f}"
                     f"\n({r.elapsed_s:.2f} s)",
                     fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Reconstruction Comparison — Airy PSF (r={CFG.psf_radius}), "
        f"AWGN σ={CFG.noise_sigma}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    logger.info("Figure saved to reconstruction_comparison.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Load and preprocess ──────────────────────────────────────────────
    logger.info("Loading '%s' from skimage.data ...", CFG.image_name)
    caller = getattr(ski.data, CFG.image_name)
    image = caller()
    gray = image_normalization(to_grayscale(image))

    h, w = gray.shape
    if h % 2 == 0:
        h -= 1
    if w % 2 == 0:
        w -= 1
    gray = odd_crop_around_center(gray, (h, w))
    logger.info("Ground truth shape: %s, range [%.4f, %.4f]",
                gray.shape, gray.min(), gray.max())

    # ── 2. Generate PSF ─────────────────────────────────────────────────────
    PSF = airy_psf(size=CFG.psf_size, radius=CFG.psf_radius)
    logger.info("PSF shape: %s, sum=%.6f, peak=%.6f",
                PSF.shape, PSF.sum(), PSF.max())

    # ── 3. Degrade the image ────────────────────────────────────────────────
    blurred = blur_image(gray, PSF)
    degraded = add_awgn(blurred, sigma=CFG.noise_sigma, seed=CFG.noise_seed)
    logger.info("Degraded image PSNR=%.2f dB, SSIM=%.3f",
                psnr(gray, degraded), ssim(gray, degraded))

    # ── 4. Run all algorithms ───────────────────────────────────────────────
    runners = [
        ("Wiener",          run_wiener),
        ("RL (Unknown)",    run_rl_unknown_boundary),
        ("RL (Standard)",   run_rl_standard),
        ("Landweber",       run_landweber),
        ("ADMM-TV",         run_admm),
        ("TVAL3",           run_tval3),
        ("FISTA (TV)",      run_fista_tv),
        ("Chambolle-Pock",  run_chambolle_pock),
        ("PnP-ADMM",       run_pnp_admm),
        ("RED-ADMM",        run_red_admm),
    ]

    results: list[Result] = []
    for label, runner in runners:
        logger.info("Running %s ...", label)
        try:
            r = runner(degraded, PSF, CFG)
            if r is None:
                continue
            # Compute quality metrics against the ground truth
            r.psnr_db = psnr(gray, r.image)
            r.ssim_val = ssim(gray, r.image)
            results.append(r)
            logger.info("  %-20s  PSNR=%6.2f dB  SSIM=%.4f  (%.2f s)",
                        r.name, r.psnr_db, r.ssim_val, r.elapsed_s)
        except Exception as e:
            logger.error("  %-20s  FAILED: %s", label, e)

    # ── 5. Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Algorithm':<24s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}  {'Time (s)':>9s}")
    print("-" * 70)
    psnr_deg = psnr(gray, degraded)
    ssim_deg = ssim(gray, degraded)
    print(f"{'(Degraded input)':<24s}  {psnr_deg:10.2f}  {ssim_deg:8.4f}  {'—':>9s}")
    for r in results:
        print(f"{r.name:<24s}  {r.psnr_db:10.2f}  {r.ssim_val:8.4f}  {r.elapsed_s:9.2f}")
    print("=" * 70)

    # ── 6. Visualise ────────────────────────────────────────────────────────
    display_results(gray, degraded, results)


if __name__ == "__main__":
    main()