"""End-to-end reconstruction demo with Wiener α-optimisation.

Demonstrates all reconstruction algorithms in the package on a small
synthetic problem (skimage's ``camera`` test image blurred by an Airy
PSF and corrupted by additive white Gaussian noise).  The two Wiener
variants additionally invoke ``Reconstruction.wiener.optimize_alpha`` to
showcase the two-stage α-optimisation workflow that is the unique-to-
this-demo feature compared to ``examples/example.py``.

Synthetic input rather than real-image / real-PSF input: the demo's
purpose is API illustration rather than a research workflow, and
synthetic input lets the script run on any machine without external
files.  See `docs/refactoring-audit/NOTES.md` Sprint 4 commits T3.1 /
T3.2 for the design rationale.

Usage:
    python example_flow.py

Requirements:
    numpy, scipy, scikit-image, matplotlib, tifffile (optional —
    only used by the package's IO layer if you adapt the demo to read
    your own data).  The PnP-ADMM / RED-ADMM algorithms additionally
    require ``bm3d``; if not installed, those two algorithms are
    skipped with a printed message.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.special import j1

from RemondoPythonCore.Common.Image_Preprocessing import (
    image_normalization,
    to_grayscale,
)
from RemondoPythonCore.Common.General_Utilities import odd_crop_around_center
from RemondoPythonCore.Common.Image_Quality_Measures import (
    FSIM,
    MSGMSD,
    MSSSIM,
    PiqPSNR,
    VIF,
)
from RemondoPythonCore.external_reconstruction import (
    ADMMDeconv,
    ChambollePockDeconv,
    FISTADeconv,
    LandweberUnknownBoundary,
    RLUnknownBoundary,
    TVAL3Deconv,
    WienerDeconv,
    optimize_alpha,
)

try:
    from RemondoPythonCore.external_reconstruction import PnPADMM, REDDeconv
    _HAS_BM3D = True
except ImportError:
    _HAS_BM3D = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def normalize_image(image: np.ndarray) -> np.ndarray:
    """MINMAX-rescale an image to [0, 1] with a degenerate-input safety floor."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-6:
        return np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
    return np.zeros_like(image)


def image_quality_metrics(
    reconstructed: np.ndarray, reference: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Compute the five image-quality metrics reported by this demo."""
    return (
        PiqPSNR(reconstructed, reference),
        MSSSIM(reconstructed, reference),
        FSIM(reconstructed, reference),
        VIF(reconstructed, reference),
        MSGMSD(reconstructed, reference),
    )


def airy_psf(size: tuple[int, int] = (35, 35), radius: float = 3.0) -> np.ndarray:
    """Generate an Airy-disk PSF (jinc² pattern), normalised to sum=1.

    Parameters
    ----------
    size
        ``(height, width)`` of the output array.  Both dimensions must
        be odd for a centred PSF.
    radius
        Distance in pixels from the centre to the first zero ring.
        Related to the optical system by ``radius ≈ 1.22 λ f/# / pixel_pitch``.
    """
    h, w = size
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
    r = np.sqrt(x ** 2 + y ** 2)
    arg = np.pi * r / radius
    with np.errstate(invalid="ignore", divide="ignore"):
        psf = np.where(r == 0, 1.0, (2 * j1(arg) / arg) ** 2)
    return psf / psf.sum()


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm specification (declarative)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class _AlgoSpec:
    """One algorithm's configuration for the eleven-algorithm sweep.

    Variability between algorithms is captured here as data; the
    executor function ``_run_one_algorithm`` runs any spec uniformly.
    """
    name: str
    solver_class: type
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    deblur_kwargs: dict[str, Any] = field(default_factory=dict)
    optimize_alpha_after: bool = False  # True for the two Wiener variants


def _build_algo_specs() -> list[_AlgoSpec]:
    """Build the algorithm spec list, conditionally including the two
    BM3D-dependent algorithms based on ``_HAS_BM3D``."""
    specs: list[_AlgoSpec] = [
        _AlgoSpec(
            "Wiener (Classical)", WienerDeconv,
            {"mode": "Classical", "paddingMode": "Reflect", "padding_scale": 2.0},
            {},
            optimize_alpha_after=True,
        ),
        _AlgoSpec(
            "Wiener (Tikhonov)", WienerDeconv,
            {"mode": "Tikhonov", "paddingMode": "Reflect", "padding_scale": 2.0},
            {},
            optimize_alpha_after=True,
        ),
        _AlgoSpec(
            "Richardson-Lucy", RLUnknownBoundary,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 500, "lambda_tv": 1e-3},
        ),
        _AlgoSpec(
            "Landweber", LandweberUnknownBoundary,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 250, "lambda_tv": 1e-3, "precondition": True, "adaptive_restart": True},
        ),
        _AlgoSpec(
            "ADMM (TV)", ADMMDeconv,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 1500, "lambda_tv": 8.9e-4, "TVnorm": 2},
        ),
        _AlgoSpec(
            "TVAL3", TVAL3Deconv,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 1500, "lambda_tv": 6.0e-4, "TVnorm": 2,
             "adaptive_tv": True, "burn_in_frac": 0.2},
        ),
        _AlgoSpec(
            "FISTA (TV)", FISTADeconv,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 600, "lambda_reg": 6e-4, "reg_mode": "TV"},
        ),
        _AlgoSpec(
            "FISTA (L1-Wavelet)", FISTADeconv,
            {"wavelet": "bior4.4", "wavelet_levels": 4,
             "paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 650, "lambda_reg": 8e-4, "reg_mode": "L1_wavelet"},
        ),
        _AlgoSpec(
            "Chambolle-Pock", ChambollePockDeconv,
            {"paddingMode": "Reflect", "padding_scale": 2.0},
            {"num_iter": 110, "lambda_tv": 0.00015},
        ),
    ]
    if _HAS_BM3D:
        specs.append(
            _AlgoSpec(
                "PnP-ADMM (BM3D)", PnPADMM,
                {"rho_z": 0.5, "sigma_scale": 0.1, "rho_v": 1.0,
                 "paddingMode": "Reflect", "padding_scale": 2.0},
                {"num_iter": 8, "lambda_tv": 0.002},
            )
        )
        specs.append(
            _AlgoSpec(
                "RED-ADMM (BM3D)", REDDeconv,
                {"sigma": 0.005, "rho_v": 0.5,
                 "paddingMode": "Reflect", "padding_scale": 2.0},
                {"num_iter": 3, "lambda_reg": 0.0001},
            )
        )
    return specs


_ALGO_SPECS: list[_AlgoSpec] = _build_algo_specs()


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm executor
# ══════════════════════════════════════════════════════════════════════════════


def _run_one_algorithm(
    spec: _AlgoSpec,
    *,
    degraded: np.ndarray,
    ref_image: np.ndarray,
    psf: np.ndarray,
) -> dict[str, Any]:
    """Run one algorithm spec end-to-end and return its result + metrics.

    Returns a dict with: ``name``, ``elapsed`` (s), ``image`` (final
    deblurred image, MINMAX-rescaled to [0, 1]), ``psnr``, ``msssim``,
    ``fsim``, ``vif``, ``msgmsd``, plus ``alpha_search`` (the
    ``OptimizeAlphaResult`` from ``optimize_alpha`` for the Wiener
    variants, ``None`` otherwise).
    """
    solver = spec.solver_class(degraded, psf, **spec.solver_kwargs)
    t0 = time.perf_counter()
    result = solver.deblur(**spec.deblur_kwargs)
    alpha_search = None
    if spec.optimize_alpha_after:
        alpha_0 = solver.last_alpha
        opt = optimize_alpha(solver, ref_image, alpha_0=alpha_0, verbose=False)
        result = opt.image
        alpha_search = opt
    elapsed = time.perf_counter() - t0
    result = normalize_image(result)
    psnr, msssim, fsim, vif, msgmsd = image_quality_metrics(result, ref_image)
    return {
        "name": spec.name,
        "elapsed": elapsed,
        "image": result,
        "psnr": psnr,
        "msssim": msssim,
        "fsim": fsim,
        "vif": vif,
        "msgmsd": msgmsd,
        "alpha_search": alpha_search,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════


def _display_results(
    ref_image: np.ndarray,
    degraded: np.ndarray,
    runs: list[dict[str, Any]],
) -> None:
    """Single tiled figure: reference + degraded + each algorithm output."""
    n = len(runs) + 2
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.ravel()

    axes[0].imshow(ref_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Reference", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    deg_psnr, deg_msssim, *_ = image_quality_metrics(
        normalize_image(degraded), ref_image
    )
    axes[1].imshow(degraded, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Degraded\nPSNR={deg_psnr:.2f} dB  MS-SSIM={deg_msssim:.3f}",
        fontsize=9,
    )
    axes[1].axis("off")

    for i, run in enumerate(runs):
        ax = axes[i + 2]
        ax.imshow(run["image"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{run['name']}\n"
            f"PSNR={run['psnr']:.2f} dB  MS-SSIM={run['msssim']:.3f}\n"
            f"({run['elapsed']:.2f} s)",
            fontsize=8,
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Reconstruction Comparison — Airy PSF (r=3.0), AWGN σ=0.01",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()


def _display_alpha_trajectories(
    runs: list[dict[str, Any]],
) -> None:
    """Separate small figure showing the Wiener α-optimisation trajectories.

    This is the unique-to-this-demo feature compared to example.py: the
    two Wiener variants invoke ``optimize_alpha`` to find the
    metric-maximising α via a coarse grid plus Brent refinement.  Each
    panel shows the metric values at the coarse grid points and marks
    the final optimum.
    """
    wiener_runs = [r for r in runs if r["alpha_search"] is not None]
    if not wiener_runs:
        return
    fig, axes = plt.subplots(1, len(wiener_runs), figsize=(5 * len(wiener_runs), 4),
                              squeeze=False)
    for ax, run in zip(axes[0], wiener_runs):
        opt = run["alpha_search"]
        ax.plot(opt.coarse_t, opt.coarse_ssim, "o-", label="coarse-grid metric",
                markersize=3)
        ax.axvline(opt.log10_alpha, color="red", linestyle="--",
                   label=f"optimum at log10(α)={opt.log10_alpha:.3f}")
        ax.set_xlabel(r"$\log_{10}(\alpha)$")
        ax.set_ylabel("MS-SSIM")
        ax.set_title(f"{run['name']}: α-search trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()


def _print_summary_table(runs: list[dict[str, Any]]) -> None:
    """Print the metric summary table to stdout."""
    header = ("Method", "Elapsed (s)", "PSNR (dB)", "MS-SSIM",
              "FSIM", "VIF", "MSGMSD")
    col_widths = (25, 12, 10, 10, 10, 10, 10)
    fmt = " ".join(f"{{:>{w}}}" if i else f"{{:<{w}}}"
                   for i, w in enumerate(col_widths))
    sep = "-" * (sum(col_widths) + len(col_widths) - 1)
    print(f"\n{sep}")
    print(fmt.format(*header))
    print(sep)
    for run in runs:
        print(fmt.format(
            run["name"],
            f"{run['elapsed']:.2f}",
            f"{run['psnr']:.2f}",
            f"{run['msssim']:.4f}",
            f"{run['fsim']:.4f}",
            f"{run['vif']:.4f}",
            f"{run['msgmsd']:.6f}",
        ))
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # ── 1. Synthetic input ─────────────────────────────────────────────────
    logger.info("Loading 'camera' from skimage.data ...")
    raw = ski.data.camera()
    gray = image_normalization(to_grayscale(raw))
    h, w = gray.shape
    if h % 2 == 0:
        h -= 1
    if w % 2 == 0:
        w -= 1
    ref_image = odd_crop_around_center(gray, (h, w))
    logger.info(
        "Reference shape: %s, range [%.4f, %.4f]",
        ref_image.shape, ref_image.min(), ref_image.max(),
    )

    psf = airy_psf(size=(35, 35), radius=3.0)
    logger.info("PSF shape: %s, sum=%.6f", psf.shape, psf.sum())

    # ── 2. Degrade ─────────────────────────────────────────────────────────
    from scipy.signal import fftconvolve
    blurred = fftconvolve(ref_image, psf, mode="same")
    rng = np.random.default_rng(42)
    degraded = blurred + rng.normal(0.0, 0.01, blurred.shape)
    logger.info(
        "Degraded image: PSNR=%.2f dB, MS-SSIM=%.4f vs reference",
        *image_quality_metrics(normalize_image(degraded), ref_image)[:2],
    )

    # ── 3. Run all algorithm specs ─────────────────────────────────────────
    if not _HAS_BM3D:
        logger.warning(
            "bm3d not installed; PnP-ADMM and RED-ADMM will be skipped."
        )
    runs: list[dict[str, Any]] = []
    for spec in _ALGO_SPECS:
        logger.info("Running %s ...", spec.name)
        try:
            run = _run_one_algorithm(
                spec, degraded=degraded, ref_image=ref_image, psf=psf,
            )
            runs.append(run)
            logger.info(
                "  %-22s PSNR=%6.2f dB  MS-SSIM=%.4f  (%.2f s)",
                run["name"], run["psnr"], run["msssim"], run["elapsed"],
            )
        except Exception as exc:
            logger.error("  %-22s FAILED: %s", spec.name, exc)

    # ── 4. Report ──────────────────────────────────────────────────────────
    _print_summary_table(runs)
    _display_results(ref_image, degraded, runs)
    _display_alpha_trajectories(runs)
    plt.show()
