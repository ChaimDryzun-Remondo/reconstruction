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

wiener_deblur : convenience wrapper
    One-shot function.  Creates a ``WienerDeconv``, calls ``deblur``,
    and returns the result.

References
----------
[1] Wiener deconvolution: https://en.wikipedia.org/wiki/Wiener_deconvolution
[2] Tikhonov & Arsenin (1977), "Solutions of Ill-Posed Problems".
[3] Donoho & Johnstone (1994), "Ideal spatial adaptation via wavelet
    shrinkage", Biometrika 81(3):425–455.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
from scipy.signal import convolve2d as _cpu_convolve2d

from skimage.restoration import estimate_sigma

from ._backend import xp, rfft2, irfft2, ifftshift, _freeze, _to_numpy, _use_gpu
from ._base import DeconvBase
from Shared.Common.General_Utilities import padding, cropping
from Shared.Common.PSF_Preprocessing import psf_preprocess, condition_psf

if _use_gpu:
    from cupyx.scipy.ndimage import uniform_filter as _uniform_filter  # type: ignore[import]
else:
    from scipy.ndimage import uniform_filter as _uniform_filter

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

    Inherits image preprocessing (grayscale, normalisation, canvas sizing,
    GPU warm-up) from :class:`DeconvBase` with two fixed overrides:

    * ``use_mask=False`` — Wiener does not support masked data fidelity.
    * ``apply_taper_on_padding_band=True`` — cosine taper at the image
      boundary suppresses Gibbs ringing in the frequency domain.

    After calling ``super().__init__()``, the PSF is re-conditioned with
    Wiener-optimised parameters (wider outer taper, ``taper_outer_frac=0.90``)
    that preserve more of the OTF compared to the iterative-algorithm default.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB/RGBA.
    psf : np.ndarray
        Point spread function.  Need not be normalised; negative values are
        clipped; odd shape is enforced internally.
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
        htm_floor_frac: float = 0.05,
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
        psf_np: np.ndarray = psf_preprocess(
            psf=psf,
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
        psf_pad: "xp.ndarray" = xp.array(
            padding(image=psf_np, full_size=self.full_shape, Type="Zero", apply_taper=False),
            dtype=xp.float32,
        )
        psf_pad = ifftshift(psf_pad)

        # Overwrite the PF/conjPF set by the base class.
        self.PF = _freeze(rfft2(psf_pad))
        self.conjPF = _freeze(self.PF.conj())
        self.conj_psf_F: "xp.ndarray" = self.conjPF  # alias used by deblur
        self.psf_F2: "xp.ndarray" = _freeze(xp.abs(self.PF) ** 2)

        # ── Image spectrum ─────────────────────────────────────────────────
        # rfft2 of the padded, tapered, normalised image.
        self.obj_F: "xp.ndarray" = rfft2(self.image)

        # ── Laplacian spectrum ─────────────────────────────────────────────
        lap_pad: "xp.ndarray" = xp.array(
            padding(
                image=_LAPL_NP.astype(np.float32),
                full_size=self.full_shape,
                Type="Zero",
                apply_taper=False,
            ),
            dtype=xp.float32,
        )
        lap_pad = ifftshift(lap_pad)
        self.L2: "xp.ndarray" = _freeze(xp.abs(rfft2(lap_pad)) ** 2)

        # ── Unpadded grayscale ─────────────────────────────────────────────
        # self.image is the full padded canvas; crop back to original size.
        gray_np: np.ndarray = cropping(_to_numpy(self.image), (self.h, self.w))
        self.gray: "xp.ndarray" = xp.array(gray_np, dtype=xp.float32)

        # ── Diagnostic floor & state ───────────────────────────────────────
        self.eps: float = 1e-8
        self._last_alpha: Optional[Union[float, "xp.ndarray"]] = None
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
        gray_np = _to_numpy(self.gray)
        return float(estimate_sigma(gray_np, channel_axis=None, average_sigmas=True))

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
        gray_np = _to_numpy(gray).astype(np.float64)
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
                Syy: "xp.ndarray" = xp.abs(self.obj_F) ** 2 / N
                Snn_psd: float = sigma ** 2
                # Signal PSD estimate: Sxx = (Syy - Snn) / |H|²
                Sxx: "xp.ndarray" = xp.maximum(
                    (Syy - Snn_psd) / xp.maximum(self.psf_F2, 1e-10),
                    1e-10,
                )
                alpha_map: "xp.ndarray" = Snn_psd / Sxx
                # Smooth in log domain to suppress variance.
                log_alpha = xp.log(alpha_map)
                log_alpha = _uniform_filter(log_alpha.real, size=3)
                alpha = xp.exp(log_alpha)

        else:
            # Manual alpha: ensure on correct device for Spectrum mode.
            if self.mode == "Spectrum":
                alpha = xp.asarray(alpha)

        # Cache for diagnostic access.
        self._last_alpha = alpha

        # ── Step 2: Build filter denominator ─────────────────────────────
        if self.mode == "Tikhonov":
            denom: "xp.ndarray" = self.psf_F2 + alpha * self.L2
            # Floor at eps × max(denom) to prevent blow-up at OTF zeros.
            denom = xp.maximum(denom, self.eps * float(xp.max(denom)))
        elif self.mode == "Classical":
            denom = self.psf_F2 + alpha
        else:  # Spectrum
            denom = self.psf_F2 + alpha

        # ── Step 3: Apply filter ──────────────────────────────────────────
        X_F: "xp.ndarray" = self.conj_psf_F * self.obj_F / denom
        x: "xp.ndarray" = irfft2(X_F, s=self.full_shape)

        # ── Step 4: Crop and return ───────────────────────────────────────
        return self._crop_and_return(x)

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
        return _to_numpy(self._last_alpha)

    @property
    def sigma_est(self) -> Optional[float]:
        """
        Noise σ estimated during the last :meth:`deblur` call.

        Returns ``None`` if alpha was supplied manually (estimation is
        skipped in that case).
        """
        return self._sigma_est


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
    init_kw   = {k: v for k, v in kwargs.items() if k in WienerDeconv._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in WienerDeconv._INIT_KEYS}
    return WienerDeconv(image=image, psf=psf, **init_kw).deblur(**deblur_kw)
