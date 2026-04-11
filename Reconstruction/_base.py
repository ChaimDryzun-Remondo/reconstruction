"""
_base.py — DeconvBase abstract base class for all deconvolution algorithms.

Owns the entire forward-model setup shared by every algorithm:

  - Image validation, grayscale conversion, odd-dimension enforcement
  - Normalization, canvas sizing, padding
  - Binary (or full) mask construction
  - Destructive PSF preprocessing, conditioning, and frequency-domain
    precomputation (PF, conjPF)
  - H^T M precomputation with relative floor clamp
  - Lipschitz constant estimation
  - Initial estimate construction

Subclasses implement only ``deblur()``.

Important boundary-model note:
``DeconvBase`` defines only the padded-canvas, PSF-placement, and mask layers
shared by the package. That is only part of the full boundary model. Solver
families may still differ in how their TV / gradient operators treat the
canvas boundary (for example Neumann-family TV prox versus periodic
gradient/divergence pairs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Callable, Optional

import numpy as np

from . import _backend as backend
from ._backend import PaddingStr
from ._common import (
    padding,
    cropping,
    odd_crop_around_center,
    psf_preprocess,
    condition_psf,
    image_normalization, validate_image, to_grayscale,
)

logger = logging.getLogger(__name__)

_ITERATIVE_PSF_BG_RING_FRAC: float = 0.15
_ITERATIVE_PSF_TAPER_OUTER_FRAC: float = 0.20
_ITERATIVE_PSF_TAPER_END_FRAC: float = 0.50

_WIENER_PSF_BG_RING_FRAC: float = 0.15
_WIENER_PSF_TAPER_OUTER_FRAC: float = 0.90
_WIENER_PSF_TAPER_END_FRAC: float = 1.0


def _split_init_and_deblur_kwargs(
    init_keys: frozenset[str],
    kwargs: dict,
) -> tuple[dict, dict]:
    """
    Split wrapper kwargs into constructor kwargs and deblur kwargs.

    Parameters
    ----------
    init_keys : frozenset[str]
        Names routed to the constructor.
    kwargs : dict
        Wrapper kwargs supplied by the caller.

    Returns
    -------
    tuple[dict, dict]
        ``(init_kw, deblur_kw)`` where unknown keys remain in ``deblur_kw``.
    """
    init_kw = {k: v for k, v in kwargs.items() if k in init_keys}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in init_keys}
    return init_kw, deblur_kw


def _prepare_psf_fft(
    psf: np.ndarray,
    full_shape: tuple[int, int],
    *,
    bg_ring_frac: float,
    taper_outer_frac: float,
    taper_end_frac: float,
    preprocess_fn: Optional[Callable] = None,
    condition_fn: Optional[Callable] = None,
    padding_fn: Optional[Callable] = None,
) -> tuple["backend.xp.ndarray", "backend.xp.ndarray"]:
    """
    Prepare the PSF spectrum on the padded FFT canvas.

    Applies the current package-default preprocessing pipeline:
    COM centering, negative clipping, odd-shape enforcement, conditioning,
    zero-padding, `ifftshift`, and `rfft2`.
    """
    if preprocess_fn is None:
        preprocess_fn = psf_preprocess
    if condition_fn is None:
        condition_fn = condition_psf
    if padding_fn is None:
        padding_fn = padding

    psf_np: np.ndarray = preprocess_fn(
        psf=psf,
        center_method="com",
        remove_negatives="clip",
        eps=1e-12,
        enforce_odd_shape=True,
    )
    psf_np = condition_fn(
        psf=psf_np,
        bg_ring_frac=bg_ring_frac,
        taper_outer_frac=taper_outer_frac,
        taper_end_frac=taper_end_frac,
    )
    psf_pad: "backend.xp.ndarray" = backend.xp.array(
        padding_fn(
            image=psf_np,
            full_size=full_shape,
            Type="Zero",
            apply_taper=False,
        ),
        dtype=backend.xp.float32,
    )
    psf_pad = backend.ifftshift(psf_pad)
    PF = backend._freeze(backend.rfft2(psf_pad))
    conjPF = backend._freeze(PF.conj())
    return PF, conjPF


# ══════════════════════════════════════════════════════════════════════════════
# DeconvBase — abstract base class
# ══════════════════════════════════════════════════════════════════════════════

class DeconvBase(ABC):
    """
    Abstract base class for deconvolution algorithms.

    Handles all shared forward-model setup in ``__init__``.  Subclasses
    implement only :meth:`deblur`.

    ``DeconvBase`` should therefore be read as a shared forward-model scaffold,
    not as a complete boundary-condition contract. Padding and masking are
    universal here, but the full effective boundary model also depends on the
    solver-specific regularizer / operator layer.

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
    10. Destructive PSF preprocessing policy
        (COM centering, negative clipping, odd-shape enforcement,
        conditioning, zero-padding, ifftshift) + rfft2 → PF, conjPF.
    11. Precompute H^T M with a relative floor clamp.
    12. Estimate the Lipschitz constant L = max |H(f)|².
    13. Construct the initial estimate on the padded canvas.

    Statefulness contract
    ---------------------
    ``self.estimated_image`` is the persistent object-level iterate state.
    Iterative subclasses typically start each new :meth:`deblur` call from
    ``self.estimated_image`` and overwrite it on successful completion. This
    makes repeated object-level calls warm starts by default unless a subclass
    explicitly documents otherwise.

    Convenience wrappers are not stateful: they instantiate a fresh solver,
    call :meth:`deblur`, and discard the object.

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
        Point spread function before the package default preprocessing policy
        is applied. The current default policy is destructive: the PSF is
        re-centred by centre of mass, negative values are clipped, odd shape
        is enforced, the tails are conditioned, the result is zero-padded to
        the FFT canvas, and `ifftshift` is applied before FFT placement. This
        may alter the supplied PSF centroid, support, and wing amplitudes.
    paddingMode : PaddingStr, optional
        Edge-extension mode for the padded canvas.  Default ``"Reflect"``.
        This controls how the observed image is embedded into the FFT canvas,
        but it does not by itself determine the full solver boundary model.
    padding_scale : float, optional
        Padding width as a multiple of the PSF size on each side.
        Default 2.0.  Larger values reduce wrap-around artefacts at the
        cost of a bigger FFT.
    initialEstimate : np.ndarray or None, optional
        Initial guess for the deconvolved image.  If ``None``, the
        normalised observed image is used. The padded, working-domain version
        of this initial guess becomes ``self.estimated_image``, which is also
        the persistent per-object iterate state across repeated calls for the
        iterative solver families.
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
        do not support masked data fidelity).  This mask defines only the
        data-fidelity support; it does not imply that all other operators in
        the solver use the same boundary condition.
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
                f"(result would be {OH}x{OW})."
            )
        if (OH, OW) != (H, W):
            gray = odd_crop_around_center(gray, (OH, OW))

        # ── Step 4: Normalise to [0, 1] float ─────────────────────────────
        # Keeps FFT magnitudes well-conditioned and makes noise-variance
        # estimates directly comparable to the signal range.
        gray_working_raw = gray.astype(np.float64, copy=True)
        gray_min = float(np.min(gray_working_raw))
        gray_max = float(np.max(gray_working_raw))
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
        if backend._use_gpu:
            _dummy = backend.xp.zeros(self.full_shape, dtype=backend.xp.float32)
            _ = backend.rfft2(_dummy)
            del _dummy

        # ── Step 8: Pad observed image onto the canvas ────────────────────
        # Taper is allowed but MUST be padding-band-only with interior
        # weight = 1 so the observed data inside Ω is not modified.
        self.image: "backend.xp.ndarray" = backend.xp.array(
            padding(
                image=gray,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=backend.xp.float32,
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
            self.mask: "backend.xp.ndarray" = backend.xp.zeros(
                self.full_shape, dtype=backend.xp.float32
            )
            off_y = (self.full_shape[0] - self.h) // 2
            off_x = (self.full_shape[1] - self.w) // 2
            self.mask[off_y:off_y + self.h, off_x:off_x + self.w] = 1.0
        else:
            self.mask = backend.xp.ones(self.full_shape, dtype=backend.xp.float32)

        # ── Step 10: PSF frequency-domain preparation ─────────────────────

        # a–e) Centre (centre-of-mass), clip negatives, enforce odd shape,
        #      condition PSF tails, zero-pad, ifftshift, then compute the
        #      frozen PSF spectrum and its conjugate.
        self.PF, self.conjPF = _prepare_psf_fft(
            psf,
            self.full_shape,
            bg_ring_frac=_ITERATIVE_PSF_BG_RING_FRAC,
            taper_outer_frac=_ITERATIVE_PSF_TAPER_OUTER_FRAC,
            taper_end_frac=_ITERATIVE_PSF_TAPER_END_FRAC,
        )

        # ── Step 11: Precompute H^T M with relative floor clamp ───────────
        # H^T M = irfft2(H*(f) · F[M]) measures, at each pixel, how much
        # of the PSF footprint overlaps the observed region Ω.  Near-zero
        # values outside Ω have no data constraint; the floor clamp
        # prevents division blow-up there.
        #
        # Relative floor:  floor = max(htm_floor_frac × max(H^T M), 1e-12)
        # Typical htm_floor_frac = 0.05 (5 % of the peak).
        fshape = self.full_shape
        htm_raw = backend.irfft2(
            self.conjPF * backend.rfft2(self.mask), s=fshape
        ).astype(backend.xp.float32)

        htm_max = float(backend.xp.max(htm_raw))
        htm_floor = max(htm_floor_frac * htm_max, 1e-12)
        backend.xp.clip(htm_raw, a_min=htm_floor, a_max=None, out=htm_raw)
        self.HTM: "backend.xp.ndarray" = backend._freeze(htm_raw)

        logger.debug(
            "HTM: max=%.4f, floor=%.4f (%.1f%% of max)",
            htm_max, htm_floor, 100.0 * htm_floor_frac,
        )

        # ── Step 12: Lipschitz constant L = max |H(f)|² ───────────────────
        # L = ||H^T H||_op = max_f |H(f)|².
        # Used by Landweber-type algorithms to set a stable step size.
        self._lipschitz: float = float(backend.xp.max(backend.xp.abs(self.PF) ** 2))
        logger.debug("Lipschitz constant L = %.6f", self._lipschitz)

        # ── Step 13: Initial estimate on the padded canvas ────────────────
        # If provided, the initial estimate must live in the same working
        # domain as the observation: grayscale, odd-cropped, and scaled
        # with the *image-derived* affine map rather than its own min/max.
        if initialEstimate is not None:
            validate_image(initialEstimate)
            init_source = to_grayscale(initialEstimate)

            init_H, init_W = init_source.shape
            init_OH = init_H if init_H % 2 == 1 else init_H - 1
            init_OW = init_W if init_W % 2 == 1 else init_W - 1
            if init_OH <= 0 or init_OW <= 0:
                raise ValueError(
                    "initialEstimate is too small after enforcing odd spatial shape "
                    f"(result would be {init_OH}x{init_OW})."
                )
            if (init_OH, init_OW) != (init_H, init_W):
                init_source = odd_crop_around_center(init_source, (init_OH, init_OW))

            if init_source.shape != (self.h, self.w):
                raise ValueError(
                    "initialEstimate must match the image working-domain shape "
                    f"after grayscale/odd-shape preprocessing; got {init_source.shape}, "
                    f"expected {(self.h, self.w)}."
                )

            init_source = init_source.astype(np.float64, copy=False)
            if gray_max > gray_min:
                init_source = (init_source - gray_min) / (gray_max - gray_min)
            else:
                init_source = init_source.copy()
        else:
            init_source = gray

        self.estimated_image: "backend.xp.ndarray" = backend.xp.array(
            padding(
                image=init_source,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=backend.xp.float32,
        )
        # Ensure strictly positive start (required for RL; beneficial for
        # Landweber when combined with a positivity projection).
        backend.xp.maximum(
            self.estimated_image,
            backend.xp.float32(1e-8),
            out=self.estimated_image,
        )

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

    def _restore_last_finite_state(self, last_finite: Optional["backend.xp.ndarray"]) -> None:
        """
        Restore ``self.estimated_image`` from the most recent verified-finite iterate.

        If ``last_finite`` is ``None`` or itself non-finite, the existing solver
        state is left untouched. This helper intentionally preserves the most
        recent finite iterate, which may come from the current failed call
        rather than from the exact pre-call state.
        """
        if last_finite is None:
            return
        if not bool(backend.xp.isfinite(last_finite).all()):
            return
        self.estimated_image = last_finite.copy()

    def _fail_on_nonfinite(
        self,
        arr: "backend.xp.ndarray",
        *,
        name: str,
        iteration: Optional[int] = None,
        last_finite: Optional["backend.xp.ndarray"] = None,
    ) -> None:
        """
        Raise ``FloatingPointError`` if ``arr`` contains NaN/Inf values.

        Optionally restores ``self.estimated_image`` from ``last_finite`` before
        raising so solver state remains finite and reusable after failure.
        The guarantee is finite-state preservation, not rollback to the exact
        state that existed before the current :meth:`deblur` call started.
        """
        if bool(backend.xp.isfinite(arr).all()):
            return

        self._restore_last_finite_state(last_finite)

        iter_msg = f" at iteration {iteration}" if iteration is not None else ""
        logger.warning("Non-finite values detected in %s%s.", name, iter_msg)
        raise FloatingPointError(
            f"Non-finite values detected in {name}{iter_msg}."
        )

    def _crop_and_return(
        self,
        x_k: "backend.xp.ndarray",
        *,
        last_finite: Optional["backend.xp.ndarray"] = None,
    ) -> np.ndarray:
        """
        Store the final state, crop to the original FOV, and transfer to CPU.

        Parameters
        ----------
        x_k : xp.ndarray, shape self.full_shape
            The final iterate on the padded canvas.
        last_finite : xp.ndarray or None, optional
            Most recent verified-finite iterate.  Used to preserve solver state
            if ``x_k`` is non-finite.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view.

        Notes
        -----
        On success this method overwrites ``self.estimated_image`` with the
        supplied padded-canvas iterate. That stored iterate is the warm-start
        state used by the iterative solver families on later object-level
        :meth:`deblur` calls.
        """
        self._fail_on_nonfinite(
            x_k, name="final iterate", last_finite=last_finite
        )
        self.estimated_image = x_k.copy()
        result = backend._to_numpy(cropping(x_k, (self.h, self.w)))
        if not bool(np.isfinite(result).all()):
            self._restore_last_finite_state(last_finite)
            logger.warning("Non-finite values detected in cropped return array.")
            raise FloatingPointError(
                "Non-finite values detected in cropped return array."
            )
        return result

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
        self._fail_on_nonfinite(x_new, name="x_new for convergence check")
        self._fail_on_nonfinite(x_old, name="x_old for convergence check")

        den = backend.xp.linalg.norm(x_new)
        if not bool(backend.xp.isfinite(den)):
            logger.warning("Non-finite denominator norm in convergence check.")
            raise FloatingPointError(
                "Non-finite denominator norm in convergence check."
            )
        den = den if float(den) > 0.0 else backend.xp.float32(eps)
        num = backend.xp.linalg.norm(x_new - x_old)
        if not bool(backend.xp.isfinite(num)):
            logger.warning("Non-finite numerator norm in convergence check.")
            raise FloatingPointError(
                "Non-finite numerator norm in convergence check."
            )
        rel_chg = float(num / den)
        if not np.isfinite(rel_chg):
            logger.warning("Non-finite relative change in convergence check.")
            raise FloatingPointError(
                "Non-finite relative change in convergence check."
            )
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
