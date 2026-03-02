from __future__ import annotations

import importlib
import logging
import numpy as np
from typing import Optional, Literal

from Shared.Common.General_Utilities   import padding, cropping
from Shared.Common.PSF_Preprocessing  import psf_preprocess, condition_psf
from Shared.Common.Image_Preprocessing import (image_normalization, validate_image,
                                                to_grayscale, odd_crop_around_center)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# GPU Detection
# ══════════════════════════════════════════════════════════════════════════════

# Set to False to force CPU even when a GPU is available
_USER_GPU_FLAG: bool = True


def _detect_gpu() -> bool:
    """
    Probe whether a functional CUDA device is reachable via CuPy.

    Performs a three-stage check to avoid false positives:

    1. **Package presence** — is CuPy installed?
    2. **Device count** — does the CUDA runtime see at least one device?
    3. **Live allocation** — can we actually allocate and compute on the GPU?

    Returns
    -------
    bool
        ``True`` only if all three stages succeed *and* ``_USER_GPU_FLAG``
        is ``True``.
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

        # Definitive test: touch GPU memory and run a kernel.
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


# ══════════════════════════════════════════════════════════════════════════════
# Backend Selection  (xp, _fft)
# ══════════════════════════════════════════════════════════════════════════════
# We maintain a single code path by aliasing either NumPy or CuPy as ``xp``
# and the corresponding FFT module as ``_fft``.  All downstream code uses
# ``xp`` for array creation and ``rfft2`` / ``irfft2`` for transforms, so
# switching between CPU and GPU requires no code changes.

if _use_gpu:
    import cupy as cp
    xp   = cp
    _fft = cp.fft
    try:
        # CuPy ≥ 12: pre-allocate FFT plan cache to avoid first-call latency.
        cp.fft.config.set_plan_cache_size(64)         # 64 plans ≈ 16–32 MiB
    except AttributeError:
        pass  # Older CuPy; harmless.
else:
    xp   = np
    _fft = np.fft

# ══════════════════════════════════════════════════════════════════════════════
# FFT Helpers  (real-valued optimisation)
# ══════════════════════════════════════════════════════════════════════════════
# We use rfft2 / irfft2 throughout.  For a real M×N array, rfft2 returns a
# complex M×(N//2+1) array, exploiting Hermitian symmetry to halve both the
# memory footprint and the arithmetic cost compared to the full fft2/ifft2.

def rfft2(a: xp.ndarray, **kwargs) -> xp.ndarray:
    """
    Backend-agnostic real 2-D FFT.

    Parameters
    ----------
    a : xp.ndarray, shape (H, W), real
        Spatial-domain real-valued array.

    Returns
    -------
    xp.ndarray, shape (H, W//2+1), complex
        Half-spectrum exploiting Hermitian symmetry.
    """
    return _fft.rfft2(a, **kwargs)


def irfft2(a: xp.ndarray, s: tuple[int, int], **kwargs) -> xp.ndarray:
    """
    Backend-agnostic inverse real 2-D FFT.

    Parameters
    ----------
    a : xp.ndarray, shape (H, W//2+1), complex
        Half-spectrum (output of ``rfft2``).
    s : tuple of int
        ``(H, W)`` — the desired *output* spatial shape.  Required because
        ``W`` cannot be uniquely inferred from ``W//2+1`` (even/odd ambiguity).

    Returns
    -------
    xp.ndarray, shape (H, W), real
        Reconstructed spatial-domain array.
    """
    return _fft.irfft2(a, s=s, **kwargs)


# For PSF centering we need the full-complex shift helper.
ifftshift = _fft.ifftshift


# ══════════════════════════════════════════════════════════════════════════════
# Utility Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _freeze(a: xp.ndarray) -> xp.ndarray:
    """Mark array as read-only to prevent accidental in-place modification."""
    try:
        a.flags.writeable = False
    except AttributeError:
        pass  # CuPy arrays may not support this on all versions.
    return a


def _to_numpy(x: xp.ndarray) -> np.ndarray:
    """
    Transfer array to host (CPU) memory if it lives on the GPU.

    If the backend is already NumPy, this is a no-op (returns the same object).
    """
    if _use_gpu:
        return xp.asnumpy(x)  # cp.asnumpy performs a DMA copy to host.
    return x

# ══════════════════════════════════════════════════════════════════════════════
# Dey et al. Multiplicative TV Correction
# ══════════════════════════════════════════════════════════════════════════════

def _tv_multiplicative_correction(
    x: xp.ndarray,
    lambda_tv: float,
    eps_grad: float = 1e-8,
) -> xp.ndarray:
    """
    Compute the Dey et al. multiplicative TV correction factor.

    For a current estimate *x*, this returns the denominator of Eq. (2):

        C(x) = 1  -  λ · div( ∇x / |∇x|_ε )

    so that the regularized RL update becomes:

        x_{k+1} = ( x_k · Hᵀ(y / Hx_k) ) / C(x_k)

    The divergence of the normalized gradient is sometimes called the *mean
    curvature* of the level sets of *x*.  It is large and positive at concave
    edges (bright→dark), large and negative at convex edges (dark→bright), and
    near zero in flat regions.  Dividing the RL update by ``C(x)`` therefore
    selectively attenuates the update at edges — the hallmark of TV
    regularization — while leaving flat regions untouched.

    Discrete Operators
    ------------------
    We use **forward differences** with **Neumann (zero-flux) boundary
    conditions** for the gradient, and **backward differences** for the
    divergence (which is the adjoint of the negative forward-difference
    operator under Neumann BC).

    Gradient (forward differences, Neumann BC):
        (∂x/∂h)[i,j] = x[i+1,j] - x[i,j]   for i = 0, …, H-2
        (∂x/∂h)[H-1,j] = 0                    (Neumann: no flux at boundary)

        (∂x/∂w)[i,j] = x[i,j+1] - x[i,j]   for j = 0, …, W-2
        (∂x/∂w)[i,W-1] = 0

    Divergence (backward differences, adjoint of -∇):
        div(p_h, p_w)[i,j] = bwd_h(p_h)[i,j] + bwd_w(p_w)[i,j]

        where bwd_h(p)[i,j] = p[i,j] - p[i-1,j]   for i = 1, …, H-2
              bwd_h(p)[0,j] = p[0,j]                 (boundary: p[-1,j] = 0)
              bwd_h(p)[H-1,j] = -p[H-2,j]            (boundary: p[H-1,j] = 0
                                                       from forward diff)

    This adjoint pairing ensures that ⟨-∇x, p⟩ = ⟨x, div(p)⟩ for all x, p
    satisfying the boundary conditions — a necessary condition for the
    variational derivation of Eq. (2) to hold.

    Parameters
    ----------
    x : xp.ndarray, shape (H, W), float32
        Current image estimate.  Must be non-negative.
    lambda_tv : float
        TV regularization strength.  Typical range for [0, 1]-normalized
        satellite imagery: 1e-4 to 1e-2.
    eps_grad : float, optional
        Smoothing constant for the gradient magnitude to avoid division by
        zero in flat regions.  Default 1e-8.

    Returns
    -------
    xp.ndarray, shape (H, W), float32
        The correction factor ``C(x) = 1 - λ · div(∇x / |∇x|_ε)``.
        The caller divides the standard RL update by this array.

    Notes
    -----
    - For ``lambda_tv = 0`` this function should not be called (the caller
      skips it), but if called it returns an array of ones.
    - The denominator is clamped to ``[0.5, +∞)`` to prevent sign inversion
      or amplification blow-up when λ is large relative to the local
      curvature.  This clamp is a safety measure; well-chosen λ values
      should rarely trigger it.

    References
    ----------
    [1] Dey et al., Microscopy Research and Technique, 2006, Eq. (5).
    """
    # ── Discrete gradient (forward differences, Neumann BC) ───────────────
    # Allocate the two components of ∇x.  Using sliced assignment avoids a
    # full-array diff + pad, and the last row/column remains zero (Neumann).
    dh = xp.zeros_like(x)
    dw = xp.zeros_like(x)
    dh[:-1, :] = x[1:, :] - x[:-1, :]   # ∂x/∂h; last row = 0
    dw[:, :-1] = x[:, 1:] - x[:, :-1]   # ∂x/∂w; last col = 0

    # ── Smoothed gradient magnitude ───────────────────────────────────────
    # The ε² term prevents division by zero where ∇x ≈ 0 (flat regions).
    # This is standard in TV implementations and corresponds to the Huber
    # approximation of the L1 norm near the origin.
    mag = xp.sqrt(dh * dh + dw * dw + eps_grad * eps_grad)

    # ── Normalized gradient field  n = ∇x / |∇x|_ε ──────────────────────
    # In flat regions, mag ≈ eps_grad and (dh, dw) ≈ 0, so n ≈ 0 —
    # the regularization has no effect there, as desired.
    nh = dh / mag
    nw = dw / mag
    # dh, dw, mag are no longer needed; allow garbage collection.
    del dh, dw, mag

    # ── Divergence (backward differences, adjoint of −∇) ─────────────────
    # We compute div(n_h, n_w) = bwd_h(n_h) + bwd_w(n_w) directly into
    # a single output array to avoid an intermediate allocation.
    div = xp.empty_like(x)

    # Vertical component: backward difference of n_h
    # Interior rows: n_h[i] − n_h[i−1]
    div[1:-1, :] = nh[1:-1, :] - nh[:-2, :]
    # Top boundary: n_h[0] − n_h[−1], but n_h[−1] = 0 (Neumann BC adjoint)
    div[0, :] = nh[0, :]
    # Bottom boundary: forward diff set n_h[H−1] = 0, so backward diff gives −n_h[H−2]
    div[-1, :] = -nh[-2, :]

    # Horizontal component: backward difference of n_w (accumulated into div)
    # Interior columns
    div[:, 1:-1] += nw[:, 1:-1] - nw[:, :-2]
    # Left boundary
    div[:, 0] += nw[:, 0]
    # Right boundary
    div[:, -1] += -nw[:, -2]

    del nh, nw

    # ── Assemble correction factor ────────────────────────────────────────
    # C(x) = 1 − λ · div(∇x/|∇x|_ε)
    #
    # In well-behaved regions, |div| is moderate and C ≈ 1.  Near edges,
    # div can be large, and C deviates from 1, applying the TV correction.
    #
    # Safety clamp: if λ is too large, (1 − λ·div) can become ≤ 0, which
    # would invert or explode the estimate.  We clamp to a minimum of 0.5
    # to guarantee the correction is at most a factor-of-2 amplification.
    # This is a conservative guard; in practice, a well-chosen λ should
    # keep C [~0.8, ~1.2] almost everywhere.
    correction = 1.0 - lambda_tv * div

    xp.clip(correction, a_min=0.5, a_max=None, out=correction)

    return correction

# ══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ══════════════════════════════════════════════════════════════════════════════

PaddingStr = Literal["Reflect", "Symmetric", "Wrap", "Edge", "LinearRamp", "Zero"]

class RLUnknownBoundary:
    """
    Richardson-Lucy with masked likelihood (unknown boundaries) + optional
    Dey et al. multiplicative TV regularization.

    Core idea:
      - reconstruct on padded canvas Ω'
      - compare data only on original support Ω using mask M
      - update:
            x_{k+1} = x_k * (H^T ( M * y/(Hx_k+eps) )) / (H^T M + eps)
    """

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        paddingMode: "PaddingStr" = "Reflect",
        padding_scale: float = 2.0,
        initialEstimate: Optional[np.ndarray] = None,
        apply_taper_on_padding_band: bool = False,
    ) -> None:

        validate_image(image)
        # Convert to grayscale.  Deconvolution is performed on luminance
        # only; per-channel PSF characterisation would be needed for colour
        # deconvolution and is out of scope here.
        gray: np.ndarray = to_grayscale(image)

        # Enforce odd spatial dimensions so that ifftshift ↔ fftshift are
        # exact inverses (they differ by 1 pixel for even-length axes).
        H, W = gray.shape
        OH = H if H % 2 == 1 else H - 1
        OW = W if W % 2 == 1 else W - 1
        if OH <= 0 or OW <= 0:
            raise ValueError("Image is too small after enforcing odd spatial shape.")
        if (OH, OW) != (H, W):
            gray = odd_crop_around_center(gray, (OH, OW))
            
        # Normalise to [0, 1] float.  This keeps FFT magnitudes
        # well-conditioned and makes any noise-variance estimates directly
        # comparable to the signal range.
        gray = image_normalization(image=gray, bit_depth=1, is_int=False)                 
        
        # Keep original size for final crop
        self.h, self.w = int(gray.shape[0]), int(gray.shape[1])

        # ── FFT canvas size ───────────────────────────────────────────────
        # Strategy: canvas = image + padding_scale × PSF.
        # This separates the periodic extension (inherent to circular
        # convolution via FFT) from the true image content by at least one
        # PSF-width of padding on each side, suppressing wrap-around ringing.
        self.h, self.w = gray.shape
        pH, pW = psf.shape

        fH = int(self.h + padding_scale * pH)
        fW = int(self.w + padding_scale * pW)

        # Round up to next odd integer (same ifftshift rationale as above).
        OH_full = fH if fH % 2 == 1 else fH + 1
        OW_full = fW if fW % 2 == 1 else fW + 1
        self.full_shape: tuple[int, int] = (OH_full, OW_full)

        logger.debug("Image shape %s  →  padded canvas %s", gray.shape, self.full_shape)

        if _use_gpu:
            _dummy = xp.zeros(self.full_shape, dtype=xp.float32)
            _ = rfft2(_dummy)  # warm-up for real FFT
            del _dummy

        # ── Observed image on padded canvas ───────────────────────────────
        # Taper is allowed, but MUST be padding-band-only with interior weight=1.
        self.image = xp.array(
            padding(
                image=gray,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )

        # ── Mask M: 1 on original support Ω, 0 outside ────────────────────
        self.mask = xp.zeros(self.full_shape, dtype=xp.float32)
        off_y = (self.full_shape[0] - self.h) // 2
        off_x = (self.full_shape[1] - self.w) // 2
        self.mask[off_y:off_y + self.h, off_x:off_x + self.w] = 1.0

        # ── PSF frequency-domain preparation ──────────────────────────────
        # 1. Centre (centre-of-mass), clip negatives, enforce odd shape.
        psf_np: np.ndarray = psf_preprocess(
            psf=psf,
            center_method="com",
            remove_negatives="clip",
            eps=1e-12,
            enforce_odd_shape=True,
        )

        # 2. Condition PSF tails: subtract residual background, apply outer
        #    radial taper to suppress measurement noise in the wings.
        psf_np = condition_psf(
            psf=psf_np,
            bg_ring_frac=0.15,
            taper_outer_frac=0.20,
            taper_end_frac=0.50,
        )

        # 3. Zero-pad to FFT canvas size.
        #    IMPORTANT: no edge-extension and no taper on the PSF.  The PSF
        #    must satisfy Σh = 1 (energy conservation) for the RL
        #    multiplicative update to preserve photometric consistency.
        #    Any non-zero padding or tapering would violate this.
        psf_pad: xp.ndarray = xp.array(
            padding(image=psf_np, full_size=self.full_shape,
                    Type="Zero", apply_taper=False),
            dtype=xp.float32,
        )

        # 4. ifftshift: move PSF centre from the array centre to the
        #    top-left corner [0, 0].  This is the standard technique that
        #    makes FFT-based convolution equivalent to centred linear
        #    convolution without introducing a phase ramp.
        psf_pad = ifftshift(psf_pad)

        # 5. Compute and freeze the PSF spectrum and its conjugate.
        #    These are read-only for the lifetime of this object.
        #    H(f)  = FFT(h)        — forward model in frequency domain
        #    H*(f) = conj(H(f))    — correlation (adjoint) operator
        self.PF: xp.ndarray = _freeze(rfft2(psf_pad))
        self.conjPF: xp.ndarray = _freeze(self.PF.conj())

        # Precompute H^T M (normalization term), fixed over iterations
        fshape = self.full_shape
        self.HTM = _freeze(irfft2(self.conjPF * rfft2(self.mask), s=fshape).astype(xp.float32))

        # ── Initial estimate on padded canvas ─────────────────────────────
        init_source = initialEstimate if initialEstimate is not None else gray
        self.estimated_image = xp.array(
            padding(
                image=init_source,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )
        
        # Ensure strictly positive start (RL requires x0 > 0)
        eps0 = xp.float32(1e-8)
        xp.maximum(self.estimated_image, eps0, out=self.estimated_image)

    def deblur(
        self,
        num_iter: int = 100,
        lambda_tv: float = 0.0002,
        tol: float = 1e-6,
        min_iter: int = 5,
        check_every: int = 5,        
        epsilon_devision: float = 1e-12,
        epsilon_positivity: float = 1e-8,
        tv_on_full_canvas: bool = True,
    ) -> np.ndarray:
        """
        Run unknown-boundary RL with optional multiplicative TV correction.

        tv_on_full_canvas:
          - True: TV acts on all pixels (Ω' ), i.e. weak prior outside Ω too.
          - False: TV correction is applied only inside Ω (outside remains unconstrained).
        """

        num_iter = int(np.clip(num_iter, 1, 10000))
        eps_dev = xp.float32(epsilon_devision)
        eps_pos = xp.float32(epsilon_positivity)
        use_tv = (lambda_tv is not None) and (float(lambda_tv) > 0.0)
        lam = float(lambda_tv)

        y = self.image
        M = self.mask
        PF = self.PF
        conjPF = self.conjPF
        HTM = self.HTM
        fshape = self.full_shape

        x_k = self.estimated_image.copy()

        for k in range(num_iter):

            # Forward model: H x_k
            Hx_k = irfft2(PF * rfft2(x_k), s=fshape)

            # Ratio only on observed support Ω:
            # ratio = M * y / (Hx_k + eps_dev)
            ratio = (M * y) / ((Hx_k * M) + ((1.0 - M) + eps_dev))

            # Backprojection: H^T ratio
            back = irfft2(conjPF * rfft2(ratio), s=fshape)

            # Mask-normalized RL update:
            # x_{k+1} = x_k * back / (H^T M + eps_dev)
            x_new = x_k * (back / (HTM + eps_dev))

            # Optional multiplicative TV correction (Dey et al.)
            if use_tv:
                if tv_on_full_canvas:
                    correction = _tv_multiplicative_correction(x_k, lam)
                    x_new /= correction
                else:
                    # Apply correction only inside Ω
                    correction = _tv_multiplicative_correction(x_k, lam)
                    x_new = x_new / (1.0 + (correction - 1.0) * M)

            # Positivity
            xp.maximum(x_new, eps_pos, out=x_new)

            # Convergence check
            if k >= min_iter and (k + 1) % check_every == 0:
                den = xp.linalg.norm(x_new)
                den = den if float(den) > 0.0 else eps_pos
                rel_chg = float(xp.linalg.norm(x_new - x_k) / den)
                if rel_chg < tol:
                    break

            x_k = x_new

        self.estimated_image = x_k.copy()

        # Crop back to original FOV and return on CPU
        return _to_numpy(cropping(x_k, (self.h, self.w)))


def rl_deblur_unknown_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 100,
    lambda_tv: float = 0.0002,
    paddingMode: "PaddingStr" = "Reflect",
    padding_scale: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience one-shot wrapper.
    """
    rl = RLUnknownBoundary(
        image=image,
        psf=psf,
        paddingMode=paddingMode,
        padding_scale=padding_scale,
        **kwargs,
    )
    return rl.deblur(num_iter=iters, lambda_tv=lambda_tv)


