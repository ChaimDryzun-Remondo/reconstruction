"""
_base.py — DeconvBase abstract base class for all deconvolution algorithms.

Owns the entire forward-model setup shared by every algorithm:

  - Image validation, grayscale conversion, odd-dimension enforcement
  - Normalization, canvas sizing, padding
  - Binary (or full) mask construction
  - PSF conditioning and frequency-domain precomputation (PF, conjPF)
  - H^T M precomputation with relative floor clamp
  - Lipschitz constant estimation
  - Initial estimate construction

Subclasses implement only ``deblur()``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional

import numpy as np

from ._backend import (
    xp, rfft2, irfft2, ifftshift, _freeze, _to_numpy, _use_gpu, PaddingStr,
)
from Shared.Common.General_Utilities import padding, cropping
from Shared.Common.PSF_Preprocessing import psf_preprocess, condition_psf
from Shared.Common.Image_Preprocessing import (
    image_normalization, validate_image, to_grayscale, odd_crop_around_center,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DeconvBase — abstract base class
# ══════════════════════════════════════════════════════════════════════════════

class DeconvBase(ABC):
    """
    Abstract base class for deconvolution algorithms.

    Handles all shared forward-model setup in ``__init__``.  Subclasses
    implement only :meth:`deblur`.

    The constructor performs 13 steps in order:

    1.  Validate the input image.
    2.  Convert to grayscale.
    3.  Enforce odd spatial dimensions (required for ifftshift exactness).
    4.  Normalise to [0, 1].
    5.  Store ``self.h, self.w`` — **assigned once**, here and nowhere else.
    6.  Compute the FFT canvas size (rounded up to next odd integer).
    7.  GPU warm-up (if GPU backend is active).
    8.  Pad the observed image onto the canvas.
    9.  Build the binary mask M (or all-ones if ``use_mask=False``).
    10. PSF conditioning + zero-padding + ifftshift + rfft2 → PF, conjPF.
    11. Precompute H^T M with a relative floor clamp.
    12. Estimate the Lipschitz constant L = max |H(f)|².
    13. Construct the initial estimate on the padded canvas.

    Class Attributes
    ----------------
    _INIT_KEYS : frozenset[str]
        Parameter names accepted by ``__init__`` (excluding ``image`` and
        ``psf``).  Convenience wrappers use this set to split ``**kwargs``
        between the constructor and ``deblur()``:

            init_kw   = {k: v for k, v in kwargs.items() if k in DeconvBase._INIT_KEYS}
            deblur_kw = {k: v for k, v in kwargs.items() if k not in DeconvBase._INIT_KEYS}

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB.
    psf : np.ndarray
        Point spread function (before conditioning).  Values must be
        non-negative; it will be normalised inside the constructor.
    paddingMode : PaddingStr, optional
        Edge-extension mode for the padded canvas.  Default ``"Reflect"``.
    padding_scale : float, optional
        Padding width as a multiple of the PSF size on each side.
        Default 2.0.  Larger values reduce wrap-around artefacts at the
        cost of a bigger FFT.
    initialEstimate : np.ndarray or None, optional
        Initial guess for the deconvolved image.  If ``None``, the
        normalised observed image is used.
    apply_taper_on_padding_band : bool, optional
        Apply a cosine taper on the padding band to suppress boundary
        discontinuities.  Default ``False``.
    htm_floor_frac : float, optional
        Relative floor fraction for H^T M clamping:
        ``floor = max(htm_floor_frac * max(H^T M), 1e-12)``.
        Default 0.05 (5 % of the maximum).
    use_mask : bool, optional
        If ``True`` (default), build a binary mask M that equals 1 on the
        original image support Ω and 0 outside.  If ``False``, M = 1
        everywhere (no masking — suitable for algorithms like Wiener that
        do not support masked data fidelity).
    """

    _INIT_KEYS: frozenset[str] = frozenset({
        "paddingMode",
        "padding_scale",
        "initialEstimate",
        "apply_taper_on_padding_band",
        "htm_floor_frac",
        "use_mask",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        paddingMode: PaddingStr = "Reflect",
        padding_scale: float = 2.0,
        initialEstimate: Optional[np.ndarray] = None,
        apply_taper_on_padding_band: bool = False,
        htm_floor_frac: float = 0.05,
        use_mask: bool = True,
    ) -> None:

        # ── Step 1: Validate the input image ──────────────────────────────
        validate_image(image)

        # ── Step 2: Convert to grayscale ──────────────────────────────────
        # Deconvolution is performed on luminance only.  Per-channel PSF
        # characterisation would be needed for colour deconvolution.
        gray: np.ndarray = to_grayscale(image)

        # ── Step 3: Enforce odd spatial dimensions ─────────────────────────
        # ifftshift ↔ fftshift are exact inverses only for odd-length axes.
        # For even-length arrays they differ by ±1 sample, introducing a
        # phase error that corrupts the PSF alignment.
        H, W = gray.shape
        OH = H if H % 2 == 1 else H - 1
        OW = W if W % 2 == 1 else W - 1
        if OH <= 0 or OW <= 0:
            raise ValueError(
                "Image is too small after enforcing odd spatial shape "
                f"(result would be {OH}×{OW})."
            )
        if (OH, OW) != (H, W):
            gray = odd_crop_around_center(gray, (OH, OW))

        # ── Step 4: Normalise to [0, 1] float ─────────────────────────────
        # Keeps FFT magnitudes well-conditioned and makes noise-variance
        # estimates directly comparable to the signal range.
        gray = image_normalization(image=gray, bit_depth=1, is_int=False)

        # ── Step 5: Store original size — SINGLE assignment ────────────────
        # Note: the reference RL file accidentally assigns self.h, self.w
        # twice (once before the canvas calculation, once after, yielding
        # identical values).  This single assignment is correct and clear.
        self.h, self.w = gray.shape

        # ── Step 6: Compute FFT canvas size ───────────────────────────────
        # Strategy: canvas = image + padding_scale × PSF on each side.
        # This separates the periodic extension (circular convolution) from
        # the true image content by at least one PSF-width, suppressing
        # wrap-around ringing.
        # Round up to next odd integer (same ifftshift rationale as Step 3).
        pH, pW = psf.shape
        fH = int(self.h + padding_scale * pH)
        fW = int(self.w + padding_scale * pW)
        OH_full = fH if fH % 2 == 1 else fH + 1
        OW_full = fW if fW % 2 == 1 else fW + 1
        self.full_shape: tuple[int, int] = (OH_full, OW_full)

        logger.debug(
            "Image shape %s → padded canvas %s", gray.shape, self.full_shape
        )

        # ── Step 7: GPU warm-up ────────────────────────────────────────────
        # Pre-trigger JIT compilation and FFT plan caching on the first
        # allocation so that subsequent transforms start without latency.
        if _use_gpu:
            _dummy = xp.zeros(self.full_shape, dtype=xp.float32)
            _ = rfft2(_dummy)
            del _dummy

        # ── Step 8: Pad observed image onto the canvas ────────────────────
        # Taper is allowed but MUST be padding-band-only with interior
        # weight = 1 so the observed data inside Ω is not modified.
        self.image: xp.ndarray = xp.array(
            padding(
                image=gray,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )

        # ── Step 9: Build mask M ──────────────────────────────────────────
        # M = 1 on the original image support Ω, 0 outside.
        #
        # CENTERING ASSUMPTION: padding() places the image at
        #   off = (canvas_size - image_size) // 2   (integer division)
        # in each dimension.  The mask uses the same formula.  If padding()
        # uses a different convention (e.g. biased toward top-left), the
        # mask will be mis-registered by ±1 pixel.  Verify the padding()
        # implementation matches this assumption.
        self.use_mask: bool = use_mask
        if use_mask:
            self.mask: xp.ndarray = xp.zeros(self.full_shape, dtype=xp.float32)
            off_y = (self.full_shape[0] - self.h) // 2
            off_x = (self.full_shape[1] - self.w) // 2
            self.mask[off_y:off_y + self.h, off_x:off_x + self.w] = 1.0
        else:
            self.mask = xp.ones(self.full_shape, dtype=xp.float32)

        # ── Step 10: PSF frequency-domain preparation ─────────────────────

        # a) Centre (centre-of-mass), clip negatives, enforce odd shape.
        psf_np: np.ndarray = psf_preprocess(
            psf=psf,
            center_method="com",
            remove_negatives="clip",
            eps=1e-12,
            enforce_odd_shape=True,
        )

        # b) Condition PSF tails: subtract residual background, apply outer
        #    radial taper to suppress measurement noise in the wings.
        psf_np = condition_psf(
            psf=psf_np,
            bg_ring_frac=0.15,
            taper_outer_frac=0.20,
            taper_end_frac=0.50,
        )

        # c) Zero-pad to FFT canvas size.
        #    IMPORTANT: no edge extension and no taper on the PSF.  The PSF
        #    must satisfy Σh = 1 (energy conservation); any non-zero padding
        #    or tapering would violate photometric consistency.
        psf_pad: xp.ndarray = xp.array(
            padding(
                image=psf_np,
                full_size=self.full_shape,
                Type="Zero",
                apply_taper=False,
            ),
            dtype=xp.float32,
        )

        # d) ifftshift: move the PSF centre from the array centre to [0, 0].
        #    This makes FFT-based convolution equivalent to centred linear
        #    convolution without introducing a phase ramp.
        psf_pad = ifftshift(psf_pad)

        # e) Compute and freeze the PSF spectrum and its conjugate.
        #    H(f)  = FFT(h)       — forward model in the frequency domain
        #    H*(f) = conj(H(f))   — correlation (adjoint) operator
        self.PF: xp.ndarray = _freeze(rfft2(psf_pad))
        self.conjPF: xp.ndarray = _freeze(self.PF.conj())

        # ── Step 11: Precompute H^T M with relative floor clamp ───────────
        # H^T M = irfft2(H*(f) · F[M]) measures, at each pixel, how much
        # of the PSF footprint overlaps the observed region Ω.  Near-zero
        # values outside Ω have no data constraint; the floor clamp
        # prevents division blow-up there.
        #
        # Relative floor:  floor = max(htm_floor_frac × max(H^T M), 1e-12)
        # Typical htm_floor_frac = 0.05 (5 % of the peak).
        fshape = self.full_shape
        htm_raw = irfft2(
            self.conjPF * rfft2(self.mask), s=fshape
        ).astype(xp.float32)

        htm_max = float(xp.max(htm_raw))
        htm_floor = max(htm_floor_frac * htm_max, 1e-12)
        xp.clip(htm_raw, a_min=htm_floor, a_max=None, out=htm_raw)
        self.HTM: xp.ndarray = _freeze(htm_raw)

        logger.debug(
            "HTM: max=%.4f, floor=%.4f (%.1f%% of max)",
            htm_max, htm_floor, 100.0 * htm_floor_frac,
        )

        # ── Step 12: Lipschitz constant L = max |H(f)|² ───────────────────
        # L = ||H^T H||_op = max_f |H(f)|².
        # Used by Landweber-type algorithms to set a stable step size.
        self._lipschitz: float = float(xp.max(xp.abs(self.PF) ** 2))
        logger.debug("Lipschitz constant L = %.6f", self._lipschitz)

        # ── Step 13: Initial estimate on the padded canvas ────────────────
        init_source = initialEstimate if initialEstimate is not None else gray
        self.estimated_image: xp.ndarray = xp.array(
            padding(
                image=init_source,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )
        # Ensure strictly positive start (required for RL; beneficial for
        # Landweber when combined with a positivity projection).
        xp.maximum(self.estimated_image, xp.float32(1e-8), out=self.estimated_image)

    # ══════════════════════════════════════════════════════════════════════
    # Abstract interface
    # ══════════════════════════════════════════════════════════════════════

    @abstractmethod
    def deblur(self, **kwargs) -> np.ndarray:
        """
        Run the deconvolution algorithm.

        Parameters
        ----------
        **kwargs
            Algorithm-specific parameters (iteration count, TV weight,
            tolerance, etc.).  See subclass for documentation.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view,
            on the CPU as a NumPy array.
        """
        ...

    # ══════════════════════════════════════════════════════════════════════
    # Shared helper methods
    # ══════════════════════════════════════════════════════════════════════

    def _crop_and_return(self, x_k: xp.ndarray) -> np.ndarray:
        """
        Store the final state, crop to the original FOV, and transfer to CPU.

        Parameters
        ----------
        x_k : xp.ndarray, shape self.full_shape
            The final iterate on the padded canvas.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view.
        """
        self.estimated_image = x_k.copy()
        return _to_numpy(cropping(x_k, (self.h, self.w)))

    def _check_convergence(
        self,
        x_new: xp.ndarray,
        x_old: xp.ndarray,
        k: int,
        num_iter: int,
        tol: float,
        eps: float = 1e-8,
    ) -> tuple[float, bool]:
        """
        Compute the relative iterate change and check against a tolerance.

        The relative change is defined as:

            rel_change = ||x_new - x_old|| / max(||x_new||, ε)

        Parameters
        ----------
        x_new : xp.ndarray
            Current iterate.
        x_old : xp.ndarray
            Previous iterate.
        k : int
            Current iteration index (0-based).  Used for logging only.
        num_iter : int
            Maximum iteration count.  Used for logging only.
        tol : float
            Convergence threshold.  If ``rel_change < tol`` the algorithm
            is declared converged.
        eps : float, optional
            Denominator floor to prevent division by zero when ``x_new``
            is near zero.  Default 1e-8.

        Returns
        -------
        rel_change : float
            Relative iterate change.
        converged : bool
            ``True`` if ``rel_change < tol``.
        """
        den = xp.linalg.norm(x_new)
        den = den if float(den) > 0.0 else xp.float32(eps)
        rel_chg = float(xp.linalg.norm(x_new - x_old) / den)
        converged = rel_chg < tol
        if converged:
            logger.info(
                "Converged at iteration %d/%d (rel_change=%.2e < tol=%.2e)",
                k + 1, num_iter, rel_chg, tol,
            )
        return rel_chg, converged

    def _log_no_convergence(self, num_iter: int, tol: float) -> None:
        """Log that the maximum iteration count was reached without convergence."""
        logger.info(
            "Reached max iterations (%d) without convergence (tol=%.2e).",
            num_iter, tol,
        )
