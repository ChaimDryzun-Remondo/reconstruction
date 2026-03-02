"""
_Wiener.py  —  Wiener deconvolution for 2-D grayscale images
=============================================================

Mathematical background
-----------------------
Given the degradation model in the frequency domain:

    Y(f) = H(f) · X(f) + N(f)

where
    Y — observed (blurred, noisy) image,
    H — optical transfer function (OTF), i.e. the FFT of the PSF,
    X — unknown true image,
    N — additive noise,

the Wiener filter produces the MMSE estimate:

    X̂(f) = [H*(f) / (|H(f)|² + alpha(f))] · Y(f)

The three supported regularisation modes differ only in how alpha(f) is defined:

    Classical  : alpha is a global scalar  K = σ²_n / σ²_x  (flat spectrum assumption)
    Tikhonov   : alpha · |L(f)|² where L is the Laplacian prior (promotes smoothness)
    Spectrum   : alpha(f) = S_nn(f) / S_xx(f)  per-frequency noise-to-signal ratio

Performance notes
-----------------
All FFTs are performed with rfft2 / irfft2 rather than the full complex fft2.
For a real-valued MxN image this reduces the spectral array size from MxN to
Mx(N//2+1), yielding approximately 2x speedup and 2x memory saving with no
loss of information (Hermitian symmetry of real-signal spectra).

GPU acceleration
----------------
If CuPy ≥ 12 and a CUDA-capable device are detected at import time the module
uses CuPy arrays and cupyx FFTs transparently.  The public API is identical in
both cases; _AsNumpy() converts back to NumPy before returning to the caller.

References
----------
- Wiener deconvolution: https://en.wikipedia.org/wiki/Wiener_deconvolution
- Tikhonov regularisation: Tikhonov & Arsenin (1977), "Solutions of Ill-Posed Problems"
- noise-level estimation: Donoho & Johnstone (1994), "Ideal spatial adaptation via
  wavelet shrinkage", Biometrika 81(3): 425-455
"""

from __future__ import annotations

import importlib
import logging
import numpy as np
import warnings
from typing import Optional, Literal, Union

from skimage.restoration import estimate_sigma

from RemondoPythonCore.Common.General_Utilities     import padding, cropping, odd_crop_around_center
from RemondoPythonCore.Common.PSF_Preprocessing     import psf_preprocess, condition_psf
from RemondoPythonCore.Common.Image_Preprocessing   import image_normalization, validate_image, to_grayscale

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────────────────

# Set to False to force CPU even when a GPU is available (e.g. for unit tests).
_USER_GPU_FLAG: bool = True


def _detect_gpu() -> bool:
    """
    Probe whether a functional CUDA device is reachable via CuPy.

    Three-stage check:
      1. Package presence.
      2. Device count.
      3. Live allocation.

    Returns
    -------
    bool
        True only if all three stages succeed and ``_USER_GPU_FLAG`` is True.
    """
    if not _USER_GPU_FLAG:
        logger.info("GPU disabled by _USER_GPU_FLAG; using CPU.")
        return False

    if importlib.util.find_spec("cupy") is None:
        logger.info("CuPy not found; using CPU.")
        return False

    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            logger.warning("CuPy installed but no CUDA device found; using CPU.")
            return False

        # Definitive test: actually touch GPU memory and run a kernel.
        dummy = cp.array([1.0], dtype=cp.float32)
        _ = dummy + dummy
        del dummy

        logger.info("CUDA device detected — GPU path enabled.")
        return True

    except Exception as e:
        logger.warning(
            "CuPy installed but GPU initialisation failed; "
            "falling back to NumPy/CPU.  Reason: %s", e
        )
        return False


_use_gpu: bool = _detect_gpu()

# ──────────────────────────────────────────────────────────────────────────────
# Backend selection  (xp, _fft, uniform_filter, convolve2d)
# ──────────────────────────────────────────────────────────────────────────────

if _use_gpu:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import cupy as cp
        xp   = cp
        _fft = cp.fft
        from cupyx.scipy.ndimage import uniform_filter   # type: ignore[import]
        from cupyx.scipy.signal  import convolve2d        # type: ignore[import]
        try:
            # CuPy ≥ 12: pre-allocate FFT plan cache to avoid first-call latency.
            cp.fft.config.set_plan_cache_size(64)   # 64 plans ≈ 16–32 MiB
        except AttributeError:
            pass    # Older CuPy; harmless.
else:
    xp   = np
    _fft = np.fft
    from scipy.ndimage import uniform_filter            # type: ignore[import]
    from scipy.signal  import convolve2d                # type: ignore[import]

# ──────────────────────────────────────────────────────────────────────────────
# FFT helpers  (real-valued optimisation)
# ──────────────────────────────────────────────────────────────────────────────
# We use rfft2 / irfft2 throughout.  For a real M×N array rfft2 returns a
# complex M×(N//2+1) array, exploiting Hermitian symmetry to halve both the
# memory footprint and the arithmetic work compared to the full fft2.

def rfft2(a: xp.ndarray, *args, **kwargs) -> xp.ndarray:
    """
    Backend-agnostic real 2-D FFT → shape (H, W//2+1).

    Parameters
    ----------
    a : array

    Returns
    ------- 
    array
        The 2D FFT of a real array.
    """
    return _fft.rfft2(a, *args, **kwargs)


def irfft2(a: xp.ndarray, s: tuple[int, int], *args, **kwargs) -> xp.ndarray:
    """
    Backend-agnostic inverse real 2-D FFT.

    Parameters
    ----------
    a : array, shape (H, W//2+1), complex
    s : (H, W) — the *output* shape.  Required to disambiguate even/odd W.

    Returns
    ------- 
    array
        The inverse of the 2D FFT of a real array.
    """
    return _fft.irfft2(a, s=s, *args, **kwargs)


# For PSF centering we still need the full-complex shift helpers.
ifftshift = _fft.ifftshift


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _freeze(a: xp.ndarray) -> xp.ndarray:
    """Mark array as read-only to prevent accidental in-place modification."""
    try:
        a.flags.writeable = False
    except AttributeError:
        pass    # CuPy arrays may not support this on all versions.
    return a


def _AsNumpy(x: xp.ndarray) -> np.ndarray:
    """Transfer array to host memory if it lives on the GPU."""
    if _use_gpu:
        return xp.asnumpy(x)   # cp.asnumpy performs a DMA copy to host.
    return x                    # Already NumPy; return as-is.


# ──────────────────────────────────────────────────────────────────────────────
# Laplacian regulariser kernel
# ──────────────────────────────────────────────────────────────────────────────
# This is the 3×3 isotropic Laplacian (includes diagonal neighbours with
# weight ½ relative to axis-aligned neighbours).  Defined as:
#
#   L = (1/6) · [[1, 4, 1],
#                 [4,-20, 4],
#                 [1, 4, 1]]
#
# Choosing the diagonal-weighted variant (rather than the simpler
# [[0,1,0],[1,-4,1],[0,1,0]]) gives better rotational isotropy, which reduces
# visible directional artefacts in the deblurred image.
#
# The kernel is kept in float64 to avoid precision loss when computing
# var_Ln = σ² · Σ(L²) in _alpha_from_sigma.

_LAPL: xp.ndarray = xp.array(
    [[1.0, 4.0, 1.0],
     [4.0, -20.0, 4.0],
     [1.0, 4.0, 1.0]],
    dtype=xp.float64
) / 6.0


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

ModeStr    = Literal["Classical", "Spectrum", "Tikhonov"]
PaddingStr = Literal["Reflect", "Symmetric", "Wrap", "Edge", "LinearRamp", "Zero"]


# ──────────────────────────────────────────────────────────────────────────────
# Wiener class
# ──────────────────────────────────────────────────────────────────────────────

class Wiener:
    """
    Non-blind Wiener deconvolution with three regularisation strategies.

    The constructor performs all pre-computation that does not depend on the
    regularisation parameter alpha (padding, FFTs of the PSF and Laplacian).
    The :meth:`deblur` method is therefore cheap to call repeatedly with
    different alpha values, making it convenient for parameter sweeps.

    Parameters
    ----------
    image : np.ndarray
        Input image, either grayscale (HxW) or colour (HxWxC, uint8 or float).
        Colour inputs are converted to grayscale internally; the deblurred
        output is always grayscale.
    psf : np.ndarray
        Point spread function.  Need not be normalised; need not have odd
        dimensions (``psf_preprocess`` enforces odd shape internally).  Must be
        2-D and have positive support after clipping negatives.
    mode : {"Tikhonov", "Classical", "Spectrum"}
        Regularisation strategy.  See module docstring for details.
    paddingMode : str
        Border extension strategy used when padding the *image* before FFT.
        "Reflect" is recommended as it minimises spectral leakage at image
        boundaries.  The PSF is always padded with "Edge" (zero-order hold).
    padding_scale : float ≥ 1.0
        Controls the padded FFT canvas size.  The canvas height/width is
        ``image_size + padding_scale * psf_size`` (rounded up to the next odd
        integer).  Larger values reduce circular-convolution wrap-around
        artefacts at the cost of larger FFTs.  2.0 is a safe default.
    gamma : float > 0
        Scaling factor applied to the auto-estimated alpha in Tikhonov mode.
        Values > 1 increase regularisation (smoother output); values < 1
        decrease it (sharper but noisier output).

    Attributes
    ----------
    gray : xp.ndarray
        Preprocessed, normalised grayscale image on the compute device.
    full_shape : tuple[int, int]
        (H, W) of the padded FFT canvas (always odd in both dimensions).
    conj_psf_F : xp.ndarray, complex
        Conjugate of the OTF, shape (full_shape[0], full_shape[1]//2+1).
    psf_F2 : xp.ndarray, float
        |OTF|², same shape as conj_psf_F.
    L2 : xp.ndarray, float
        |FFT(Laplacian)|², same shape — precomputed for Tikhonov mode.
    obj_F : xp.ndarray, complex
        rfft2 of the padded, normalised image.
    """

    def __init__(
        self,
        image:         np.ndarray,
        psf:           np.ndarray,
        mode:          ModeStr    = "Tikhonov",
        paddingMode:   PaddingStr = "Reflect",
        normalize_image:bool = False,
        padding_scale: float      = 2.0,
        gamma:         float      = 1.0,
    ) -> None:

        # ── Parameter validation ──────────────────────────────────────────────
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if padding_scale < 1.0:
            raise ValueError(f"padding_scale must be ≥ 1.0, got {padding_scale}")
        if mode not in ("Classical", "Spectrum", "Tikhonov"):
            raise ValueError(f"Unknown mode '{mode}'; choose Classical, Spectrum, or Tikhonov.")

        self.mode  = mode
        self.gamma = gamma

        # ── Image preprocessing ───────────────────────────────────────────────
        validate_image(image)

        # Convert to grayscale.
        gray: np.ndarray = to_grayscale(image)

        # Enforce odd spatial dimensions.  Some downstream operations
        # (e.g. odd_crop_around_center) and the fftshift/ifftshift symmetry
        # requirement are simpler with odd sizes.
        H, W = gray.shape
        OH   = H if H % 2 == 1 else H - 1
        OW   = W if W % 2 == 1 else W - 1
        if OH <= 0 or OW <= 0:
            raise ValueError("Image is too small after enforcing odd spatial shape.")
        if (OH, OW) != (H, W):
            gray = odd_crop_around_center(gray, (OH, OW))

        # Normalise to [0, 1] float — keeps FFT magnitudes well-conditioned and
        # makes the noise variance estimate from estimate_sigma directly comparable
        # to the signal variance of self.gray.
        if normalize_image:
            gray = image_normalization(image=gray, bit_depth=1, is_int=False)

        # Move to compute device (GPU array if _use_gpu, else plain NumPy).
        self.gray: xp.ndarray = xp.array(gray, dtype=xp.float32)

        # ── FFT canvas size ───────────────────────────────────────────────────
        # Strategy: canvas = image + padding_scale * PSF.
        # This ensures the periodic extension seen by the circular convolution
        # is separated from the signal by at least one PSF-width of padding on
        # each side, suppressing wrap-around ringing.
        self.h, self.w = gray.shape
        pH, pW = psf.shape

        fH = int(self.h + padding_scale * pH)
        fW = int(self.w + padding_scale * pW)

        # Round up to the next odd integer so ifftshift ↔ fftshift are exact
        # inverses (they differ by 1 pixel for even-sized arrays).
        OH_full = fH if fH % 2 == 1 else fH + 1
        OW_full = fW if fW % 2 == 1 else fW + 1

        self.full_shape: tuple[int, int] = (OH_full, OW_full)

        logger.debug(
            "Image shape %s  →  padded canvas %s", gray.shape, self.full_shape
        )

        if _use_gpu:
            # Warm up the CuPy FFT plan cache with a representative float32
            # array so the first real call incurs no planning latency.
            _dummy = xp.zeros(self.full_shape, dtype=xp.float32)
            _ = rfft2(_dummy)
            del _dummy        

        # ── Image FFT ─────────────────────────────────────────────────────────
        # Pad with the user-selected border extension and a cosine taper to
        # smoothly bring the edges to zero, suppressing Gibbs ringing.
        padded_image: xp.ndarray = xp.array(
            padding(image=gray, full_size=self.full_shape, Type=paddingMode, apply_taper=True),
            dtype=xp.float32,
        )

        # rfft2: exploit real-signal Hermitian symmetry → M×(N//2+1) spectrum.
        self.obj_F: xp.ndarray = rfft2(padded_image)

        # ── PSF FFT ───────────────────────────────────────────────────────────
        psf_np: np.ndarray = psf_preprocess(
            psf=psf,
            center_method="com",       # Centre-of-mass alignment
            remove_negatives="clip",   # Physical PSFs are non-negative
            eps=1e-12,
            enforce_odd_shape=True,    # Required by ifftshift convention
        )

        # PSF tail conditioning inside the PSF patch: background subtraction + outer radial taper
        psf_np = condition_psf(psf=psf_np, bg_ring_frac=0.15, taper_outer_frac=0.90, taper_end_frac=1.0)

        # Pad with Zero extension
        psf_pad: xp.ndarray = xp.array(
            padding(image=psf_np, full_size=self.full_shape, Type="Zero", apply_taper=False),
            dtype=xp.float32,
        )

        # ifftshift moves the PSF centre from the array centre to corner [0,0].
        # This is the standard technique to make FFT-based convolution
        # equivalent to linear (non-circular) convolution without a phase ramp.
        psf_pad = ifftshift(psf_pad)

        # Freeze spectral arrays — they are read-only after construction and
        # should never be mutated in place.
        self.conj_psf_F: xp.ndarray = _freeze(rfft2(psf_pad).conj())
        self.psf_F2:     xp.ndarray = _freeze(xp.abs(self.conj_psf_F) ** 2)

        # ── Laplacian regulariser FFT ─────────────────────────────────────────
        # Pad _LAPL with zeros (the kernel has compact support; zero extension
        # is exact).  ifftshift centres the 3×3 kernel at [0,0].
        lap_pad: xp.ndarray = xp.array(
            padding(image=_AsNumpy(_LAPL), full_size=self.full_shape, Type="Zero", apply_taper=False),
            dtype=xp.float32,
        )
        lap_pad = ifftshift(lap_pad)
        self.L2: xp.ndarray = _freeze(xp.abs(rfft2(lap_pad)) ** 2)

        # ── Regularisation floor ──────────────────────────────────────────────
        # eps is used in Tikhonov mode: denom = max(denom, eps * max(denom)).
        # Prevents numerical blow-up where |H|² ≈ 0 (beyond the diffraction
        # cutoff or in deep noise notches).
        self.eps: float = 1e-8

        # Diagnostic state (populated by deblur())
        self._last_alpha: Optional[xp.ndarray | float] = None
        self._sigma_est:  Optional[float]               = None

    # ──────────────────────────────────────────────────────────────────────────
    # Noise estimation
    # ──────────────────────────────────────────────────────────────────────────

    def _estimate_sigma(self) -> float:
        """Estimate the standard deviation of additive Gaussian noise.

        Uses the median absolute deviation (MAD) of the finest-scale diagonal
        wavelet sub-band — the method of Donoho & Johnstone (1994), as
        implemented in ``skimage.restoration.estimate_sigma``.

        The estimator is robust to structured signal content because diagonal
        high-frequency sub-bands are dominated by noise rather than by image
        edges.

        Returns
        -------
        float
            Estimated per-pixel noise standard deviation (in normalised
            [0, 1] intensity units).
        """
        # self.gray is 2-D; channel_axis=None is correct for grayscale.
        # Passing channel_axis=-1 on a 2-D array would treat the W axis as a
        # channel dimension, producing wrong results.
        gray_np = _AsNumpy(self.gray)
        return float(estimate_sigma(gray_np, channel_axis=None, average_sigmas=True))

    # ──────────────────────────────────────────────────────────────────────────
    # Auto alpha estimation for Tikhonov mode
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_from_sigma(
        gray:       xp.ndarray,
        sigma:      float,
        lap_kernel: xp.ndarray,
        gamma:      float = 1.0,
    ) -> float:
        """
        Estimate the Tikhonov regularisation parameter alpha from noise level σ.

        Derivation
        ----------
        Apply the Laplacian filter L to the observed image y = x + n:

            z = L * y = L * x + L * n

        The variance of the filtered observation decomposes as:

            Var(z) = Var(L*x)  +  Var(L*n)
                   ≈ Var(L*x)  +  σ² · ‖L‖²_F        [noise independence]

        So the signal contribution is:

            Var(L*x) ≈ Var(z) - σ² · ‖L‖²_F

        The optimal Tikhonov alpha under a white-noise / Laplacian-prior model is
        proportional to the noise-to-signal ratio in the Laplacian domain:

            α = γ · σ² / Var(L*x)

        γ is a user-supplied scaling factor; γ > 1 over-regularises (smoother),
        γ < 1 under-regularises (sharper).

        Parameters
        ----------
        gray       : 2-D float array, normalised to [0, 1].
        sigma      : estimated noise standard deviation.
        lap_kernel : Laplacian convolution kernel (same backend as gray).
        gamma      : regularisation scaling factor.

        Returns
        -------
        float  — the estimated alpha value.
        """
        # Apply the Laplacian.  boundary="symm" (mirror extension) is consistent
        # with the Reflect padding used in __init__, minimising boundary variance
        # inflation that would bias the estimate.
        z: xp.ndarray = convolve2d(gray, lap_kernel, mode="same", boundary="symm")

        # Variance of the filtered image (signal + noise contribution).
        var_z: float = float(z.var())

        # Expected noise variance after Laplacian filtering.
        # For white noise with variance σ², applying a linear filter L multiplies
        # the variance by ‖L‖²_F (squared Frobenius norm of the kernel).
        var_Ln: float = float(sigma ** 2 * float((lap_kernel ** 2).sum()))

        # Signal variance in Laplacian domain — clamp to ε to avoid division by zero.
        var_signal: float = max(var_z - var_Ln, float(xp.finfo(xp.float32).eps))

        # Tikhonov parameter: noise power / signal power (scaled by γ).
        alpha: float = gamma * sigma ** 2 / var_signal

        return alpha

    # ──────────────────────────────────────────────────────────────────────────
    # Deblurring core
    # ──────────────────────────────────────────────────────────────────────────

    def deblur(
        self,
        alpha: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Apply the Wiener filter and return the deblurred image.

        The Wiener filter in the frequency domain is:

            X̂(f) = H*(f) · Y(f) / D(f)           

        where the denominator D(f) depends on the chosen mode:

            Classical  :  D = |H|² + K          (K = σ²_n / σ²_x, scalar)
            Tikhonov   :  D = |H|² + α·|L|²     (L = Laplacian)
            Spectrum   :  D = |H|² + α(f)        (α(f) = S_nn / S_xx per frequency)

        All FFTs are real-valued (rfft2/irfft2), exploiting Hermitian symmetry
        for a ≈2x speedup over the full complex FFT.

        Parameters
        ----------
        alpha : float or array, optional
            Regularisation parameter.  If None (default), α is estimated
            automatically from the noise level using ``_estimate_sigma`` and
            the appropriate mode-specific formula.
            For "Spectrum" mode an array alpha(f) of shape matching
            ``(full_shape[0], full_shape[1]//2+1)`` may be supplied directly.

        Returns
        -------
        np.ndarray, float32, shape (H, W)
            Deblurred image in the same spatial dimensions as the input image
            (before any odd-size cropping), normalised to approximately [0, 1].
        """

        # ── Step 1: Determine α ───────────────────────────────────────────────
        if alpha is None:

            sigma = self._estimate_sigma()
            self._sigma_est = sigma

            if self.mode == "Tikhonov":
                # Estimate α from Laplacian-domain noise-to-signal ratio.
                alpha = self._alpha_from_sigma(self.gray, sigma, _LAPL, self.gamma)
                logger.info("Tikhonov: auto α = %.4e  (σ = %.4e)", alpha, sigma)

            elif self.mode == "Classical":
                # Global noise-to-signal ratio: K = σ²_n / σ²_x.
                # σ²_x ≈ Var(y) - σ²_n  (subtract noise contribution from total variance).
                var_y = float(self.gray.var())
                sigma_x2 = max(var_y - sigma ** 2, float(xp.finfo(xp.float32).eps))
                alpha = sigma ** 2 / sigma_x2
                logger.info("Classical: auto α = %.4e  (σ = %.4e)", alpha, sigma)

            else:   # Spectrum
                N = self.full_shape[0] * self.full_shape[1]

                # --- Power spectral density of the observed image ---
                # PSD(f) = |Y(f)|² / N  (with 1/N normalisation, consistent with
                # Parseval's theorem for the unnormalised FFT: Σ|y|² = (1/N)·Σ|Y|²).
                Syy: xp.ndarray = xp.abs(self.obj_F) ** 2 / N

                # --- Noise PSD ---
                # White noise with spatial variance σ² has flat PSD = σ²
                # (after the same 1/N normalisation).
                Snn_psd: float = sigma ** 2

                # --- Signal PSD ---
                # Observed spectrum: E[Syy] = |H|²·Sxx + Snn_psd
                # → Sxx = (Syy - Snn_psd) / |H|²
                # Clamp to a small positive floor to prevent negative estimates.
                Sxx: xp.ndarray = xp.maximum(
                    (Syy - Snn_psd) / xp.maximum(self.psf_F2, 1e-10),
                    1e-10,
                )

                # --- Per-frequency noise-to-signal ratio ---
                alpha_map: xp.ndarray = Snn_psd / Sxx

                # Smooth α(f) in the log domain to suppress erratic pixel-to-pixel
                # jumps caused by spectral estimation variance.  A 3×3 uniform filter
                # is a reasonable trade-off between stability and spectral resolution.
                log_alpha = xp.log(alpha_map)
                log_alpha = uniform_filter(log_alpha.real, size=3)
                alpha = xp.exp(log_alpha)

        else:
            # If a manual α is provided for Spectrum mode, ensure it lives on
            # the correct compute device.
            if self.mode == "Spectrum":
                alpha = xp.array(alpha)

        # Cache for diagnostic access via self.last_alpha.
        self._last_alpha = alpha

        # ── Step 2: Build the filter denominator ──────────────────────────────
        if self.mode == "Tikhonov":
            # D(f) = |H(f)|²  +  α · |L(f)|²
            # The eps floor prevents division blow-up at the OTF zero-crossings
            # (beyond the diffraction cutoff frequency).
            denom: xp.ndarray = self.psf_F2 + alpha * self.L2
            denom = xp.maximum(denom, self.eps * xp.max(denom))

        elif self.mode == "Classical":
            # D(f) = |H(f)|²  +  K      (K is the global SNR-derived scalar)
            denom = self.psf_F2 + alpha

        else:   # Spectrum
            # D(f) = |H(f)|²  +  α(f)   (per-frequency map)
            denom = self.psf_F2 + alpha

        # ── Step 3: Apply the filter ──────────────────────────────────────────
        # X̂(f) = H*(f) · Y(f) / D(f)
        X_F: xp.ndarray = self.conj_psf_F * self.obj_F / denom

        # Inverse rfft2.  Must supply s=full_shape so irfft2 reconstructs the
        # correct number of columns (especially important when full_shape[1] is odd).
        x: xp.ndarray = irfft2(X_F, s=self.full_shape)
        # irfft2 output is guaranteed real; no .real call needed.

        # ── Step 4: Crop back to the original image dimensions ────────────────
        x = cropping(_AsNumpy(x), (self.h, self.w))

        # Transfer to host if necessary and return as a NumPy array.
        return _AsNumpy(x)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def last_alpha(self) -> Optional[Union[float, np.ndarray]]:
        """
        The alpha value used in the most recent :meth:`deblur` call.

        Returns a NumPy scalar or array regardless of compute backend.
        Useful for diagnostic plots and parameter sweep logging.
        """
        if self._last_alpha is None:
            return None
        if isinstance(self._last_alpha, (int, float)):
            return self._last_alpha
        return _AsNumpy(self._last_alpha)

    @property
    def sigma_est(self) -> Optional[float]:
        """Noise standard deviation estimated during the last :meth:`deblur` call.

        None if alpha was supplied manually (estimation is skipped in that case).
        """
        return self._sigma_est


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def wiener_deblur(
    image: np.ndarray,
    psf:   np.ndarray,
    *,
    mode: ModeStr = "Tikhonov",
    paddingMode: PaddingStr = "Reflect",
    normalize_image:bool = False,
    alpha: Optional[Union[float, np.ndarray]] = None,
    **kwargs,
) -> np.ndarray:
    """
    One-shot Wiener deconvolution.

    Constructs a :class:`Wiener` object and immediately calls :meth:`deblur`.
    Convenient for single-use calls; use the class directly for parameter sweeps
    so that the expensive constructor (FFT pre-computation) runs only once.

    Parameters
    ----------
    image  : np.ndarray  — blurred input image (grayscale or colour).
    psf    : np.ndarray  — point spread function.
    mode   : str         — regularisation mode (default "Tikhonov").
    paddingMode : str    — padding mode (default "Reflect").
    **kwargs             — forwarded to :class:`Wiener.__init__`.

    Returns
    -------
    np.ndarray  — deblurred grayscale image, float32, shape (H, W).
    """
    return Wiener(image, psf, mode=mode, paddingMode=paddingMode, normalize_image=normalize_image, **kwargs).deblur(alpha=alpha)