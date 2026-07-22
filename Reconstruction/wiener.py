"""
wiener.py — Wiener deconvolution with three regularisation modes.

Solves the frequency-domain restoration problem:

    X̂(f) = H*(f) · Y(f) / D(f)

where the denominator D(f) depends on the chosen mode:

    Classical  :  D = |H|² + K          (K = σ²_n / σ²_x, scalar)
    Tikhonov   :  D = |H|² + α·|L|²     (L = isotropic Laplacian)
    Spectrum   :  D = |H|² + α(f)        (per-frequency S_nn/S_xx)

The filter is non-iterative (single FFT pass), making it fast and
useful as a baseline or as an initial estimate for iterative methods.

No iteration loop, no positivity enforcement, no TV regularisation.

Public API
----------
WienerDeconv : DeconvBase subclass
    Stateful deconvolution object.  Instantiate once, call :meth:`deblur`
    repeatedly with different alpha values (constructor FFTs amortised).
    Repeated calls reuse setup and update diagnostics, but they do not
    warm-start from a previous output iterate because Wiener is not
    iterative.

wiener_deblur : convenience wrapper
    One-shot function.  Creates a ``WienerDeconv``, calls ``deblur``,
    and returns the result. Each wrapper call is a fresh cold start.

References
----------
[1] Wiener deconvolution: https://en.wikipedia.org/wiki/Wiener_deconvolution
[2] Tikhonov & Arsenin (1977), "Solutions of Ill-Posed Problems".
[3] Donoho & Johnstone (1994), "Ideal spatial adaptation via wavelet
    shrinkage", Biometrika 81(3):425–455.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import convolve2d as _cpu_convolve2d

from . import _backend as backend
from ._base import (
    DeconvBase,
    _WIENER_PSF_BG_RING_FRAC,
    _WIENER_PSF_TAPER_END_FRAC,
    _WIENER_PSF_TAPER_OUTER_FRAC,
    _prepare_psf_fft,
    _run_wrapper_deblur,
)
from ._common import padding, cropping, psf_preprocess, condition_psf

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# 3×3 isotropic Laplacian kernel (diagonal-weighted for rotational isotropy).
# Kept as plain NumPy (float64) to avoid precision loss in _alpha_from_sigma
# and to be backend-independent at module level.
#
#   L = (1/6) · [[1, 4, 1],
#                [4,-20, 4],
#                [1, 4, 1]]
_LAPL_NP: np.ndarray = np.array(
    [[1.0, 4.0, 1.0],
     [4.0, -20.0, 4.0],
     [1.0, 4.0, 1.0]],
    dtype=np.float64,
) / 6.0

ModeStr = Literal["Classical", "Spectrum", "Tikhonov"]


# ─────────────────────────────────────────────────────────────────────────────
# WienerDeconv
# ─────────────────────────────────────────────────────────────────────────────

class WienerDeconv(DeconvBase):
    """
    Non-blind Wiener deconvolution with three regularisation strategies.

    The constructor performs all frequency-domain pre-computation that does
    not depend on the regularisation parameter α (image FFT, PSF FFT,
    Laplacian FFT).  :meth:`deblur` is therefore cheap to call repeatedly
    with different α values — useful for parameter sweeps.

    ``WienerDeconv`` is stateful for setup reuse and diagnostics
    (for example ``last_alpha`` and ``sigma_est``), but not for iterate warm
    starts: repeated :meth:`deblur` calls do not read the previous
    ``estimated_image`` as an initial state.

    Inherits image preprocessing (grayscale, normalisation, canvas sizing,
    GPU warm-up) from :class:`DeconvBase` with two fixed overrides:

    * ``use_mask=False`` — Wiener does not support masked data fidelity and is
      therefore not an unknown-boundary masked solver in the package sense.
    * ``apply_taper_on_padding_band=True`` — cosine taper at the image
      boundary suppresses Gibbs ringing in the frequency domain.

    The resulting boundary model is a padded / tapered circular deconvolution
    model: the image is extended to a larger FFT canvas, tapered on the
    padding band, and then restored without an ``M``-restricted fidelity term.

    After calling ``super().__init__()``, Wiener rebuilds the PSF spectrum from
    the original PSF using the same centre-of-mass centering, negative
    clipping, odd-shape enforcement, zero-padding, and `ifftshift` placement
    policy as :class:`DeconvBase`, but with a distinct conditioning preset:

    * iterative-family default:
      `bg_ring_frac=0.15`, `taper_outer_frac=0.20`, `taper_end_frac=0.50`
    * Wiener override:
      `bg_ring_frac=0.15`, `taper_outer_frac=0.90`, `taper_end_frac=1.0`

    So Wiener and the iterative-family solvers do not, by default, operate on
    identical conditioned PSFs.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB/RGBA.
    psf : np.ndarray
        Point spread function before the package preprocessing policy is
        applied. The PSF is centred by centre of mass, negative values are
        clipped, odd shape is enforced, then Wiener's solver-specific
        conditioning preset is applied before zero-padding and `ifftshift`.
    mode : {"Tikhonov", "Classical", "Spectrum"}
        Regularisation strategy.  See module docstring for details.
        Default ``"Tikhonov"``.
    paddingMode : str
        Border extension for the image padding.  Default ``"Reflect"``.
    normalize_image : bool
        Accepted for API compatibility; the base class always normalises the
        image to [0, 1].  Regression tests should pass
        ``normalize_image=True`` to the reference implementation.
    padding_scale : float
        Canvas size multiplier.  Default 2.0.
    gamma : float
        Scaling factor for the auto-estimated α in Tikhonov mode.
        γ > 1 → more regularisation (smoother); γ < 1 → less (sharper).
    initialEstimate : np.ndarray or None
        Initial guess forwarded to ``DeconvBase``; not used by the Wiener
        filter itself but retained for API consistency.
    htm_floor_frac : float
        Floor fraction for H^T M clamping, forwarded to ``DeconvBase``.

    Attributes
    ----------
    gray : xp.ndarray
        Normalised, unpadded grayscale image on the compute device.
    obj_F : xp.ndarray, complex
        rfft2 of the padded, normalised image.
    psf_F2 : xp.ndarray, float
        |OTF|² (squared magnitude of the PSF spectrum).
    conj_psf_F : xp.ndarray, complex
        Conjugate OTF (alias of ``conjPF``).
    L2 : xp.ndarray, float
        |FFT(Laplacian)|², precomputed for Tikhonov mode.

    Notes
    -----
    The object stores the most recent returned image in ``estimated_image``
    for consistency with :class:`DeconvBase`, but Wiener does not use that
    stored image as input to subsequent :meth:`deblur` calls.

    References
    ----------
    [1] A. Beck & M. Teboulle (2009), SIAM J. Imaging Sciences 2(1):183–202.
    [2] A. Chambolle (2004), J. Math. Imaging Vision 20(1–2):89–97.
    [3] B. O'Donoghue & E. Candès (2015), Found. Comput. Math. 15:715–732.
    """

    # Override _INIT_KEYS to add Wiener-specific constructor parameters.
    # (use_mask and apply_taper_on_padding_band are hardcoded; omit them.)
    _INIT_KEYS: frozenset[str] = frozenset({
        "paddingMode",
        "padding_scale",
        "initialEstimate",
        "htm_floor_frac",
        "mode",
        "gamma",
        "normalize_image",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        mode: ModeStr = "Tikhonov",
        paddingMode: str = "Reflect",
        normalize_image: bool = False,  # noqa: ARG002 — accepted, not used
        padding_scale: float = 2.0,
        gamma: float = 1.0,
        initialEstimate: Optional[np.ndarray] = None,
        htm_floor_frac: float = 0.01,
    ) -> None:

        # ── Parameter validation ───────────────────────────────────────────
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        if mode not in ("Classical", "Spectrum", "Tikhonov"):
            raise ValueError(
                f"Unknown mode {mode!r}; choose 'Classical', 'Spectrum', or 'Tikhonov'."
            )

        self.mode: str = mode
        self.gamma: float = float(gamma)

        # ── Base class setup ───────────────────────────────────────────────
        # Hardcoded: use_mask=False (Wiener has no masked data fidelity),
        #            apply_taper_on_padding_band=True (suppress Gibbs ringing).
        super().__init__(
            image,
            psf,
            paddingMode=paddingMode,
            padding_scale=padding_scale,
            initialEstimate=initialEstimate,
            apply_taper_on_padding_band=True,
            use_mask=False,
            htm_floor_frac=htm_floor_frac,
        )

        # ── Re-condition PSF with Wiener-optimised parameters ──────────────
        # The base class uses taper_outer_frac=0.20, taper_end_frac=0.50 (more
        # aggressive, suited for iterative methods that tolerate OTF zeros).
        # Wiener benefits from a wider outer taper (0.90) that preserves more
        # of the OTF magnitude while still suppressing PSF tail noise.
        # Overwrite the PF/conjPF set by the base class.
        self.PF, self.conjPF = _prepare_psf_fft(
            psf,
            self.full_shape,
            bg_ring_frac=_WIENER_PSF_BG_RING_FRAC,
            taper_outer_frac=_WIENER_PSF_TAPER_OUTER_FRAC,
            taper_end_frac=_WIENER_PSF_TAPER_END_FRAC,
            preprocess_fn=psf_preprocess,
            condition_fn=condition_psf,
            padding_fn=padding,
        )
        self.conj_psf_F: "backend.xp.ndarray" = self.conjPF  # alias used by deblur
        self.psf_F2: "backend.xp.ndarray" = backend._freeze(backend.xp.abs(self.PF) ** 2)

        # ── F3: recompute HTM and Lipschitz from the Wiener-specific PF ───
        # The base class set both from the iterative-family PF *before* the
        # PF/conjPF override above, leaving stale values bound to the wrong
        # spectrum.  Wiener itself does not read these fields, but keeping
        # them consistent with the active PF closes a latent trap for any
        # FISTA-style subclass of WienerDeconv.
        htm_raw = backend.irfft2(
            self.conjPF * backend.rfft2(self.mask), s=self.full_shape
        ).astype(backend.xp.float32)
        htm_max = float(backend.xp.max(htm_raw))
        htm_floor = max(htm_floor_frac * htm_max, 1e-12)
        backend.xp.clip(htm_raw, a_min=htm_floor, a_max=None, out=htm_raw)
        self.HTM = backend._freeze(htm_raw)
        self._lipschitz = float(backend.xp.max(self.psf_F2))

        # ── Image spectrum ─────────────────────────────────────────────────
        # rfft2 of the padded, tapered, normalised image.
        self.obj_F: "backend.xp.ndarray" = backend.rfft2(self.image)

        # ── Laplacian spectrum ─────────────────────────────────────────────
        lap_pad: "backend.xp.ndarray" = backend.xp.array(
            padding(
                image=_LAPL_NP.astype(np.float32),
                full_size=self.full_shape,
                Type="Zero",
                apply_taper=False,
            ),
            dtype=backend.xp.float32,
        )
        lap_pad = backend.ifftshift(lap_pad)
        self.L2: "backend.xp.ndarray" = backend._freeze(backend.xp.abs(backend.rfft2(lap_pad)) ** 2)

        # ── Unpadded grayscale ─────────────────────────────────────────────
        # self.image is the full padded canvas; crop back to original size.
        gray_np: np.ndarray = cropping(backend._to_numpy(self.image), (self.h, self.w))
        self.gray: "backend.xp.ndarray" = backend.xp.array(
            gray_np, dtype=backend.xp.float32
        )

        # ── Diagnostic floor & state ───────────────────────────────────────
        self.eps: float = 1e-8
        self._last_alpha: Optional[Union[float, "backend.xp.ndarray"]] = None
        self._sigma_est: Optional[float] = None

    # ── Noise estimation ───────────────────────────────────────────────────

    def _estimate_sigma(self) -> float:
        """
        Estimate additive Gaussian noise σ via MAD of the finest wavelet band.

        Uses the Donoho–Johnstone (1994) estimator as implemented in
        ``skimage.restoration.estimate_sigma``.

        Returns
        -------
        float
            Estimated per-pixel noise standard deviation (normalised units).
        """
        try:
            from skimage.restoration import estimate_sigma
        except ImportError as exc:
            raise ImportError(
                "WienerDeconv automatic noise estimation requires the optional "
                "'scikit-image' dependency. Install with: "
                "pip install reconstruction[imaging] or pip install scikit-image"
            ) from exc

        gray_np = backend._to_numpy(self.gray)
        return 1.5*float(estimate_sigma(gray_np, channel_axis=None, average_sigmas=True))

    # ── Auto alpha for Tikhonov mode ───────────────────────────────────────

    @staticmethod
    def _alpha_from_sigma(
        gray: "np.ndarray | xp.ndarray",
        sigma: float,
        lap_kernel: np.ndarray,
        gamma: float = 1.0,
    ) -> float:
        """
        Estimate the Tikhonov α from noise level σ.

        Derivation
        ----------
        For y = x + n (signal + white noise), applying Laplacian L gives:

            Var(L*y) = Var(L*x) + σ² · ‖L‖²_F

        so  Var(L*x) ≈ Var(L*y) - σ² · ‖L‖²_F.

        Optimal α under a white-noise/Laplacian-prior model:

            α = γ · σ² / Var(L*x)

        Parameters
        ----------
        gray : 2-D array (numpy or cupy), normalised to [0, 1].
        sigma : estimated noise std.
        lap_kernel : Laplacian kernel (numpy float64).
        gamma : regularisation scaling factor.

        Returns
        -------
        float
        """
        # Always use CPU/scipy — result is a scalar; no GPU needed here.
        gray_np = backend._to_numpy(gray).astype(np.float64)
        lap_np = np.asarray(lap_kernel, dtype=np.float64)

        z: np.ndarray = _cpu_convolve2d(gray_np, lap_np, mode="same", boundary="symm")
        var_z: float = float(z.var())
        var_Ln: float = float(sigma ** 2 * float((lap_np ** 2).sum()))
        var_signal: float = max(var_z - var_Ln, float(np.finfo(np.float32).eps))

        return gamma * sigma ** 2 / var_signal

    # ── Deblurring core ────────────────────────────────────────────────────

    def deblur(
        self,
        alpha: Optional[Union[float, np.ndarray]] = None,
        inverse_normalize: bool = False,
    ) -> np.ndarray:
        """
        Apply the Wiener filter and return the deblurred image.

        The Wiener filter in the frequency domain is:

            X̂(f) = H*(f) · Y(f) / D(f)

        where D(f) depends on the chosen mode:

        * **Classical** :  ``D = |H|² + K``  (K = σ²_n / σ²_x, global scalar)
        * **Tikhonov**  :  ``D = |H|² + α·|L|²``  (Laplacian regulariser)
        * **Spectrum**  :  ``D = |H|² + α(f)``  (per-frequency map)

        Assumes stationary Gaussian noise.  The regularisation parameter λ
        controls the noise–resolution trade-off: larger λ → smoother output
        with less noise amplification but more residual blur.

        No iteration, no positivity projection (Wiener may produce negative
        pixels — this is normal for linear filters).

        Parameters
        ----------
        alpha : float, array, or None
            Regularisation parameter.  If None (default), α is estimated
            automatically from the noise level.  For Spectrum mode an array of
            shape ``(full_shape[0], full_shape[1]//2+1)`` may be supplied.
        inverse_normalize : bool
            If ``True``, map the returned cropped grayscale result back to the
            observed image's odd-cropped raw grayscale units. This is
            independent of the constructor-level ``normalize_image``
            compatibility flag; internal Wiener computation still uses the
            package working domain.

        Returns
        -------
        np.ndarray, float32, shape (self.h, self.w)
            Deblurred image cropped to the original field of view.
        """
        # ── Step 1: Determine α ───────────────────────────────────────────
        if alpha is None:
            sigma = self._estimate_sigma()
            self._sigma_est = sigma

            if self.mode == "Tikhonov":
                alpha = self._alpha_from_sigma(self.gray, sigma, _LAPL_NP, self.gamma)
                logger.info("Tikhonov: auto α = %.4e  (σ = %.4e)", alpha, sigma)

            elif self.mode == "Classical":
                # K = σ²_n / σ²_x  where σ²_x ≈ Var(y) - σ²_n.
                var_y = float(self.gray.var())
                sigma_x2 = max(var_y - sigma ** 2, float(np.finfo(np.float32).eps))
                alpha = sigma ** 2 / sigma_x2
                logger.info("Classical: auto α = %.4e  (σ = %.4e)", alpha, sigma)

            else:  # Spectrum
                N = self.full_shape[0] * self.full_shape[1]
                # PSD of observed image: Syy = |Y|² / N
                if backend._use_gpu:
                    from cupyx.scipy.ndimage import uniform_filter as uniform_filter  # type: ignore[import]
                else:
                    from scipy.ndimage import uniform_filter

                Syy: "backend.xp.ndarray" = backend.xp.abs(self.obj_F) ** 2 / N
                Snn_psd: float = sigma ** 2
                # Signal PSD estimate: Sxx = (Syy - Snn) / |H|²
                Sxx: "backend.xp.ndarray" = backend.xp.maximum(
                    (Syy - Snn_psd) / backend.xp.maximum(self.psf_F2, 1e-10),
                    1e-10,
                )
                alpha_map: "backend.xp.ndarray" = Snn_psd / Sxx
                # Smooth in log domain to suppress variance.
                log_alpha = backend.xp.log(alpha_map)
                log_alpha = uniform_filter(log_alpha.real, size=3)
                alpha = backend.xp.exp(log_alpha)

        else:
            # Manual alpha uses the current call's supplied parameter rather
            # than an estimated noise level, so sigma_est reflects that by
            # clearing any value cached from a prior auto-alpha call.
            self._sigma_est = None
            # Manual alpha: ensure on correct device for Spectrum mode.
            if self.mode == "Spectrum":
                alpha = backend.xp.asarray(alpha)

        # Cache for diagnostic access.
        self._last_alpha = alpha

        # ── Step 2: Build filter denominator ─────────────────────────────
        if self.mode == "Tikhonov":
            denom: "backend.xp.ndarray" = self.psf_F2 + alpha * self.L2
            # Floor at eps × max(denom) to prevent blow-up at OTF zeros.
            denom = backend.xp.maximum(denom, self.eps * float(backend.xp.max(denom)))
        elif self.mode == "Classical":
            denom = self.psf_F2 + alpha
        else:  # Spectrum
            denom = self.psf_F2 + alpha

        # ── Step 3: Apply filter ──────────────────────────────────────────
        X_F: "backend.xp.ndarray" = self.conj_psf_F * self.obj_F / denom
        x: "backend.xp.ndarray" = backend.irfft2(X_F, s=self.full_shape)

        # ── Step 4: Crop and return ───────────────────────────────────────
        return self._crop_and_return(x, inverse_normalize=inverse_normalize)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def last_alpha(self) -> Optional[Union[float, np.ndarray]]:
        """
        The α value used in the most recent :meth:`deblur` call.

        Returns a NumPy scalar or array regardless of compute backend.
        Returns ``None`` if :meth:`deblur` has not been called yet.
        """
        if self._last_alpha is None:
            return None
        if isinstance(self._last_alpha, (int, float)):
            return self._last_alpha
        return backend._to_numpy(self._last_alpha)

    @property
    def sigma_est(self) -> Optional[float]:
        """
        Noise σ estimated during the last :meth:`deblur` call.

        Returns ``None`` if alpha was supplied manually (estimation is
        skipped in that case).
        """
        return self._sigma_est


# ─────────────────────────────────────────────────────────────────────────────
# optimize_alpha — two-stage α optimisation against a quality metric
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizeAlphaResult:
    """Result of :func:`optimize_alpha`.

    Attributes
    ----------
    alpha
        Best α found.
    log10_alpha
        ``log10(alpha)`` at the optimum.
    ssim
        Quality metric (typically SSIM) at the optimum.
    psnr
        PSNR at the optimum (reported separately for diagnostics).
    image
        Deblurred image at the optimum, MINMAX-normalised to [0, 1].
    coarse_t
        ``log10(α)`` grid sampled in the coarse stage.
    coarse_ssim
        Metric values at the coarse-grid points.
    coarse_n_evals
        Number of solver calls in the coarse stage.
    refine_n_evals
        Number of solver calls in the Brent refinement stage.
    total_n_evals
        Total solver calls (coarse + refine + 1 final evaluation).
    elapsed
        Wall-clock seconds.
    """

    alpha: float
    log10_alpha: float
    ssim: float
    psnr: float
    image: np.ndarray
    coarse_t: np.ndarray
    coarse_ssim: np.ndarray
    coarse_n_evals: int
    refine_n_evals: int
    total_n_evals: int
    elapsed: float


def _resolve_metrics(
    ssim_fn: Optional[Callable], psnr_fn: Optional[Callable]
) -> tuple[Callable, Callable]:
    """Lazy-import default metric functions if not provided.

    The lazy import preserves the package's "import succeeds even when
    optional dependencies are not installed" pattern (see
    ``Reconstruction/__init__.py``); ``MSSSIM`` / ``PiqPSNR`` pull in
    the ``piq`` torch-backed library, which is not loaded at module
    import time.
    """
    if ssim_fn is None:
        from RemondoPythonCore.Common.Image_Quality_Measures import MSSSIM
        ssim_fn = MSSSIM
    if psnr_fn is None:
        from RemondoPythonCore.Common.Image_Quality_Measures import PiqPSNR
        psnr_fn = PiqPSNR
    return ssim_fn, psnr_fn


def _normalize_image_for_metric(image: np.ndarray) -> np.ndarray:
    """MINMAX-rescale to [0, 1] with a degenerate-input safety floor.

    Hardcoded preprocessing applied to each candidate deblurred image
    before the quality metric is computed.  Matches the behaviour of
    ``optimize_wiener_alpha`` in its pre-promotion form (where the
    function lived in ``examples/example_flow.py`` and the helper was
    named ``normalize_image``).  Callers needing raw-domain metrics
    are not currently supported; see NOTES.md (Sprint 4 / commit T3.1)
    for the future-extension framing.
    """
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-6:
        return np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
    return np.zeros_like(image)


def _coarse_search(
    solver: "WienerDeconv",
    ref_image: np.ndarray,
    ssim_fn: Callable,
    t_center: float,
    half_width: float,
    n_points: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stage 1: uniform log10(α) grid evaluation.

    Returns the grid, the metric values, and the index of the best.
    """
    t_grid = np.linspace(t_center - half_width, t_center + half_width, n_points)
    ssim_grid = np.empty(n_points)

    for i, t in enumerate(t_grid):
        result = solver.deblur(alpha=10.0 ** t)
        ssim_grid[i] = float(ssim_fn(_normalize_image_for_metric(result), ref_image))

    i_best = int(np.argmax(ssim_grid))

    if verbose:
        print(
            f"  Coarse search ({n_points} pts): "
            f"best t = {t_grid[i_best]:.4f} "
            f"(α = {10.0 ** t_grid[i_best]:.4e}), "
            f"metric = {ssim_grid[i_best]:.6f}"
        )

    return t_grid, ssim_grid, i_best


def _brent_refine(
    solver: "WienerDeconv",
    ref_image: np.ndarray,
    ssim_fn: Callable,
    t_lo: float,
    t_hi: float,
    xtol: float,
    verbose: bool,
) -> tuple[float, float, int]:
    """Stage 2: bounded Brent refinement inside the winning grid cell."""
    n_evals = 0

    def objective(t: float) -> float:
        nonlocal n_evals
        result = solver.deblur(alpha=10.0 ** t)
        ssim_val = float(ssim_fn(_normalize_image_for_metric(result), ref_image))
        n_evals += 1
        return -ssim_val

    opt = minimize_scalar(
        objective,
        bounds=(t_lo, t_hi),
        method="bounded",
        options={"xatol": xtol, "maxiter": 50},
    )

    t_opt = float(opt.x)
    ssim_opt = float(-opt.fun)

    if verbose:
        print(
            f"  Brent refine ({n_evals} evals): "
            f"t = {t_opt:.6f} "
            f"(α = {10.0 ** t_opt:.6e}), "
            f"metric = {ssim_opt:.6f}"
        )

    return t_opt, ssim_opt, n_evals


def optimize_alpha(
    solver: "WienerDeconv",
    ref_image: np.ndarray,
    alpha_0: float,
    *,
    ssim_fn: Optional[Callable] = None,
    psnr_fn: Optional[Callable] = None,
    coarse_half_width: float = 3.0,
    coarse_n_points: int = 80,
    brent_xtol: float = 1e-3,
    verbose: bool = False,
) -> OptimizeAlphaResult:
    """Two-stage α optimisation against a quality metric.

    Stage 1 sweeps a uniform ``log10(α)`` grid of width
    ``2 * coarse_half_width`` centred on ``log10(alpha_0)``.  Stage 2
    runs a bounded Brent refinement inside the winning grid cell.
    Both stages call :meth:`WienerDeconv.deblur` repeatedly; the
    constructor's frequency-domain pre-computation is amortised.

    Each candidate deblurred image is MINMAX-rescaled to [0, 1] via
    ``_normalize_image_for_metric`` before being passed to ``ssim_fn``
    or ``psnr_fn``.  This preprocessing is hardcoded; callers needing
    raw-domain metrics should request the parameterisation as a
    follow-up (see NOTES.md).

    Parameters
    ----------
    solver
        Pre-constructed :class:`WienerDeconv` instance.  The constructor
        performs the FFT pre-computation; this function only varies
        ``alpha`` across calls to :meth:`WienerDeconv.deblur`.
    ref_image
        Reference (ground-truth) image used by the quality metric.
        Not mutated.
    alpha_0
        Initial guess for α.  The coarse grid is centred on
        ``log10(alpha_0)``.  A typical starting point is
        ``solver.last_alpha`` after a default-α call.
    ssim_fn
        Quality metric to maximise.  Signature ``f(estimate, ref) -> float``.
        If ``None``, defaults to ``Common.Image_Quality_Measures.MSSSIM``
        (lazy-imported).  Despite the name, any compatible quality
        metric may be passed (e.g. PSNR for an analytic-optimum test).
    psnr_fn
        PSNR-style metric reported in the result for diagnostics; not
        used in optimisation.  Defaults to
        ``Common.Image_Quality_Measures.PiqPSNR`` (lazy-imported).
    coarse_half_width
        Half-width of the coarse log10(α) grid.  Default 3.0 (search
        spans ``[α_0 / 1e3, α_0 * 1e3]``).
    coarse_n_points
        Number of coarse-grid points.  Default 80.
    brent_xtol
        ``xatol`` tolerance for ``scipy.optimize.minimize_scalar`` in the
        refinement stage.  Default 1e-3.
    verbose
        If ``True``, print per-stage progress.  Default ``False`` (silent).

    Returns
    -------
    OptimizeAlphaResult
        Frozen dataclass with the optimum α, its log10, the metric values,
        the deblurred image at the optimum, the search history, and
        diagnostics.

    Notes
    -----
    The function is currently tied to the ``alpha=`` keyword of
    :meth:`WienerDeconv.deblur`.  Generalising to other ``DeconvBase``
    subclasses (RL, FISTA, Landweber, etc.) would require either a
    callable wrapper or a parameter-name keyword; deferred until a
    concrete caller needs it.
    """
    ssim_fn, psnr_fn = _resolve_metrics(ssim_fn, psnr_fn)
    t0_wall = time.perf_counter()
    t_center = float(np.log10(alpha_0))

    if verbose:
        print(f"Wiener α optimisation (two-stage)")
        print(f"  α₀ = {alpha_0:.4e}  (t₀ = {t_center:.4f})")
        print(
            f"  Coarse: {coarse_n_points} pts in "
            f"[{10.0 ** (t_center - coarse_half_width):.2e}, "
            f"{10.0 ** (t_center + coarse_half_width):.2e}]"
        )

    t_grid, ssim_grid, i_best = _coarse_search(
        solver, ref_image, ssim_fn,
        t_center, coarse_half_width, coarse_n_points, verbose,
    )

    dt = float(t_grid[1] - t_grid[0])
    t_lo = float(t_grid[i_best]) - dt
    t_hi = float(t_grid[i_best]) + dt

    t_opt, ssim_opt, refine_evals = _brent_refine(
        solver, ref_image, ssim_fn,
        t_lo, t_hi, brent_xtol, verbose,
    )

    best_alpha = 10.0 ** t_opt
    best_image = _normalize_image_for_metric(solver.deblur(alpha=best_alpha))
    best_ssim = float(ssim_fn(best_image, ref_image))
    best_psnr = float(psnr_fn(best_image, ref_image))

    elapsed = time.perf_counter() - t0_wall
    total_evals = coarse_n_points + refine_evals + 1

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  Optimal α  = {best_alpha:.6e}  (log₁₀ = {t_opt:.4f})")
        print(f"  Metric     = {best_ssim:.6f}")
        print(f"  PSNR       = {best_psnr:.2f} dB")
        print(
            f"  Evals      : {coarse_n_points} coarse "
            f"+ {refine_evals} refine + 1 final = {total_evals}"
        )
        print(f"  Wall time  : {elapsed:.2f} s")
        print(f"{'═' * 60}")

    return OptimizeAlphaResult(
        alpha=best_alpha,
        log10_alpha=t_opt,
        ssim=best_ssim,
        psnr=best_psnr,
        image=best_image,
        coarse_t=t_grid,
        coarse_ssim=ssim_grid,
        coarse_n_evals=coarse_n_points,
        refine_n_evals=refine_evals,
        total_n_evals=total_evals,
        elapsed=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def wiener_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Convenience one-shot wrapper for Wiener deconvolution.

    Splits ``**kwargs`` between the :class:`WienerDeconv` constructor and
    :meth:`~WienerDeconv.deblur` using :attr:`WienerDeconv._INIT_KEYS`.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    **kwargs
        Any parameter accepted by :class:`WienerDeconv` (constructor) or
        :meth:`~WienerDeconv.deblur`.

    Returns
    -------
    np.ndarray
        Deblurred image, float32, shape (H, W) matching the original
        image field of view.
    """
    return _run_wrapper_deblur(
        WienerDeconv,
        WienerDeconv._INIT_KEYS,
        image,
        psf,
        kwargs,
    )
