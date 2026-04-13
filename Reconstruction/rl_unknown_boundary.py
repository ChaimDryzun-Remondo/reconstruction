"""
rl_unknown_boundary.py — Richardson-Lucy deconvolution with unknown-boundary masking.

Algorithm: unknown-boundary RL with optional multiplicative TV regularisation
(Dey et al. 2006).  No new ``__init__`` — all setup is handled by
:class:`~._base.DeconvBase`.

Boundary model
--------------
RL uses masked data fidelity on the original support over the padded FFT
canvas inherited from :class:`~._base.DeconvBase`. When TV is active, the
regularisation step uses the Dey-style multiplicative correction, which
belongs to the Neumann-family TV operators rather than the periodic
Fourier-diagonalized family.

Public API
----------
RLUnknownBoundary : DeconvBase subclass
    Stateful deconvolution object.  Instantiate once, call :meth:`deblur`
    one or more times. Repeated object-level calls warm-start from the stored
    ``estimated_image`` iterate by default.

rl_deblur_unknown_boundary : convenience wrapper
    One-shot function.  Creates an ``RLUnknownBoundary``, calls ``deblur``,
    and returns the result. Each wrapper call is a fresh cold start.
"""
from __future__ import annotations

import logging

import numpy as np

from . import _backend as backend
from ._base import DeconvBase, _run_wrapper_deblur
from ._tv_operators import tv_multiplicative_correction

logger = logging.getLogger(__name__)


class RLUnknownBoundary(DeconvBase):
    """
    Richardson-Lucy deconvolution with unknown-boundary masking.

    Inherits all constructor logic from :class:`DeconvBase` — no
    ``__init__`` override.  Implements only :meth:`deblur`.

    See :class:`DeconvBase` for constructor parameters. The effective
    boundary model is:

    - padded FFT canvas from :class:`DeconvBase`
    - masked data fidelity on the observed support Ω
    - circular convolution on the full padded canvas
    - Neumann-family TV behaviour when ``lambda_tv > 0``

    Repeated object-level :meth:`deblur` calls warm-start from the stored
    ``estimated_image`` state unless a numerical failure interrupts the run.
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
            In both cases the correction comes from the Neumann-family
            multiplicative TV operator rather than a periodic gradient pair.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view.
        """
        num_iter = int(np.clip(num_iter, 1, 10000))
        eps_dev = backend.xp.float32(epsilon_division)
        eps_pos = backend.xp.float32(epsilon_positivity)
        use_tv = (lambda_tv is not None) and (float(lambda_tv) > 0.0)
        lam = float(lambda_tv)

        y      = self.image
        M      = self.mask
        PF     = self.PF
        conjPF = self.conjPF
        HTM    = self.HTM
        fshape = self.full_shape

        x_k = self.estimated_image.copy()
        last_finite = x_k.copy()

        for k in range(num_iter):

            # ── Step 1: Forward model H x_k ──────────────────────────────
            with backend.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Hx_k = backend.irfft2(PF * backend.rfft2(x_k), s=fshape)

            # ── Step 2: Ratio on observed support Ω ──────────────────────
            # Outside Ω (M=0): numerator=0, denominator≈1 → ratio≈0.
            ratio = (M * y) / ((Hx_k * M) + ((1.0 - M) + eps_dev))

            # ── Step 3: Back-projection H^T ratio ────────────────────────
            with backend.errstate(over="ignore", invalid="ignore", divide="ignore"):
                back = backend.irfft2(conjPF * backend.rfft2(ratio), s=fshape)

            # ── Step 4: Mask-normalised RL update ────────────────────────
            with backend.errstate(over="ignore", invalid="ignore", divide="ignore"):
                x_new = x_k * (back / (HTM + eps_dev))

            # ── Step 5: Optional multiplicative TV correction ─────────────
            # F17: hoist the identical correction call out of both branches;
            # only the application rule differs.
            if use_tv:
                correction = tv_multiplicative_correction(x_k, lam)
                if tv_on_full_canvas:
                    x_new /= correction
                else:
                    x_new = x_new / (1.0 + (correction - 1.0) * M)

            # ── Step 6: Positivity projection ────────────────────────────
            backend.xp.maximum(x_new, eps_pos, out=x_new)
            self._fail_on_nonfinite(
                x_new,
                name="RL iterate",
                iteration=k + 1,
                last_finite=last_finite,
            )

            # ── Step 7: Convergence check ─────────────────────────────────
            converged = False
            if k >= min_iter and (k + 1) % check_every == 0:
                _, converged = self._check_convergence(
                    x_new, x_k, k=k, num_iter=num_iter, tol=tol,
                )

            # Advance state *before* breaking so the returned iterate is the
            # improved one that was just validated (fix for F1).
            x_k = x_new
            # F10: refresh the rollback snapshot at the convergence-check
            # cadence rather than every iteration.  Always refresh on
            # convergence so the returned last_finite matches x_k.
            if converged or (k + 1) % check_every == 0:
                last_finite = x_k.copy()

            if converged:
                break

        else:
            self._log_no_convergence(num_iter, tol)

        # ── Step 8: Store, crop, and return ──────────────────────────────
        return self._crop_and_return(x_k, last_finite=last_finite)


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
    return _run_wrapper_deblur(
        RLUnknownBoundary,
        DeconvBase._INIT_KEYS,
        image,
        psf,
        kwargs,
    )
