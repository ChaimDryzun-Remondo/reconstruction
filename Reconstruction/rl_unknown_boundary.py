"""
rl_unknown_boundary.py — Richardson-Lucy deconvolution with unknown-boundary masking.

Algorithm: unknown-boundary RL with optional multiplicative TV regularisation
(Dey et al. 2006).  No new ``__init__`` — all setup is handled by
:class:`~._base.DeconvBase`.

Public API
----------
RLUnknownBoundary : DeconvBase subclass
    Stateful deconvolution object.  Instantiate once, call :meth:`deblur`
    one or more times.

rl_deblur_unknown_boundary : convenience wrapper
    One-shot function.  Creates an ``RLUnknownBoundary``, calls ``deblur``,
    and returns the result.
"""
from __future__ import annotations

import logging

import numpy as np

from ._backend import xp, rfft2, irfft2
from ._base import DeconvBase
from ._tv_operators import tv_multiplicative_correction

logger = logging.getLogger(__name__)


class RLUnknownBoundary(DeconvBase):
    """
    Richardson-Lucy deconvolution with unknown-boundary masking.

    Inherits all constructor logic from :class:`DeconvBase` — no
    ``__init__`` override.  Implements only :meth:`deblur`.

    See :class:`DeconvBase` for constructor parameters.
    """

    def deblur(
        self,
        num_iter: int = 100,
        lambda_tv: float = 0.0002,
        tol: float = 1e-6,
        min_iter: int = 5,
        check_every: int = 5,
        epsilon_division: float = 1e-12,
        epsilon_positivity: float = 1e-8,
        tv_on_full_canvas: bool = True,
    ) -> np.ndarray:
        """
        Run unknown-boundary RL with optional multiplicative TV correction.

        Parameters
        ----------
        num_iter : int
            Maximum number of RL iterations.  Clamped to [1, 10000].
        lambda_tv : float
            TV regularisation strength.  Set to 0 to disable TV.
        tol : float
            Convergence tolerance on the relative iterate change.
        min_iter : int
            Minimum iterations before convergence checks begin.
        check_every : int
            Check convergence every this many iterations.
        epsilon_division : float
            Small constant added to denominators to prevent division by zero.
        epsilon_positivity : float
            Positivity floor applied after each update.
        tv_on_full_canvas : bool
            If ``True``, TV acts on all pixels (full padded canvas).
            If ``False``, TV correction is masked to the observed region Ω.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view.
        """
        num_iter = int(np.clip(num_iter, 1, 10000))
        eps_dev = xp.float32(epsilon_division)
        eps_pos = xp.float32(epsilon_positivity)
        use_tv = (lambda_tv is not None) and (float(lambda_tv) > 0.0)
        lam = float(lambda_tv)

        y      = self.image
        M      = self.mask
        PF     = self.PF
        conjPF = self.conjPF
        HTM    = self.HTM
        fshape = self.full_shape

        x_k = self.estimated_image.copy()

        for k in range(num_iter):

            # ── Step 1: Forward model H x_k ──────────────────────────────
            Hx_k = irfft2(PF * rfft2(x_k), s=fshape)

            # ── Step 2: Ratio on observed support Ω ──────────────────────
            # Outside Ω (M=0): numerator=0, denominator≈1 → ratio≈0.
            ratio = (M * y) / ((Hx_k * M) + ((1.0 - M) + eps_dev))

            # ── Step 3: Back-projection H^T ratio ────────────────────────
            back = irfft2(conjPF * rfft2(ratio), s=fshape)

            # ── Step 4: Mask-normalised RL update ────────────────────────
            x_new = x_k * (back / (HTM + eps_dev))

            # ── Step 5: Optional multiplicative TV correction ─────────────
            if use_tv:
                if tv_on_full_canvas:
                    correction = tv_multiplicative_correction(x_k, lam)
                    x_new /= correction
                else:
                    correction = tv_multiplicative_correction(x_k, lam)
                    x_new = x_new / (1.0 + (correction - 1.0) * M)

            # ── Step 6: Positivity projection ────────────────────────────
            xp.maximum(x_new, eps_pos, out=x_new)

            # ── Step 7: Convergence check ─────────────────────────────────
            if k >= min_iter and (k + 1) % check_every == 0:
                _, converged = self._check_convergence(
                    x_new, x_k, k=k, num_iter=num_iter, tol=tol,
                )
                if converged:
                    break

            x_k = x_new

        else:
            self._log_no_convergence(num_iter, tol)

        # ── Step 8: Store, crop, and return ──────────────────────────────
        return self._crop_and_return(x_k)


def rl_deblur_unknown_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Convenience one-shot wrapper for Richardson-Lucy with unknown boundaries.

    Splits ``**kwargs`` between the :class:`RLUnknownBoundary` constructor
    and :meth:`~RLUnknownBoundary.deblur` using
    :attr:`DeconvBase._INIT_KEYS`.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    **kwargs
        Any parameter accepted by the :class:`RLUnknownBoundary`
        constructor or :meth:`~RLUnknownBoundary.deblur`.

    Returns
    -------
    np.ndarray
        Deconvolved image, shape (H, W) matching the original image
        field of view.
    """
    init_kw   = {k: v for k, v in kwargs.items() if k in DeconvBase._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in DeconvBase._INIT_KEYS}
    obj = RLUnknownBoundary(image=image, psf=psf, **init_kw)
    return obj.deblur(**deblur_kw)
