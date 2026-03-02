"""
landweber_unknown_boundary.py — FISTA-accelerated preconditioned Landweber
with proximal TV as a thin subclass of DeconvBase.

Solves the variational problem:

    min_x  (1/2) ||M(Hx − y)||²  +  λ · TV(x)

where H is convolution with the PSF, M is the binary mask, and TV is
isotropic total variation.

Algorithm
---------
Uses FISTA acceleration (Beck & Teboulle 2009) with optional diagonal
preconditioning by H^T M and adaptive restart (O'Donoghue & Candès 2015).
The TV proximal operator is solved via Chambolle 2004 dual projection.

No ``__init__`` override — all setup is handled by :class:`~._base.DeconvBase`.

Public API
----------
LandweberUnknownBoundary : DeconvBase subclass
    Stateful deconvolution object.  Instantiate once, call :meth:`deblur`
    one or more times.

landweber_deblur_unknown_boundary : convenience wrapper
    One-shot function.  Creates a ``LandweberUnknownBoundary``, calls
    ``deblur``, and returns the result.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._backend import xp, rfft2, irfft2
from ._base import DeconvBase
from ._tv_operators import prox_tv_chambolle

logger = logging.getLogger(__name__)


class LandweberUnknownBoundary(DeconvBase):
    """
    FISTA-accelerated preconditioned Landweber with proximal TV.

    Inherits all constructor logic from :class:`DeconvBase` — no
    ``__init__`` override.  Implements only :meth:`deblur`.

    See :class:`DeconvBase` for constructor parameters.

    References
    ----------
    [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
        Algorithm for Linear Inverse Problems," SIAM J. Imaging Sciences,
        2(1):183–202, 2009.
    [2] A. Chambolle, "An Algorithm for Total Variation Minimization and
        Applications," J. Math. Imaging and Vision, 20(1–2):89–97, 2004.
    [3] B. O'Donoghue and E. Candès, "Adaptive Restart for Accelerated
        Gradient Schemes," Found. Comput. Math., 15:715–732, 2015.
    """

    def deblur(
        self,
        num_iter: int = 200,
        lambda_tv: float = 0.001,
        tol: float = 1e-6,
        min_iter: int = 10,
        check_every: int = 5,
        step_size: Optional[float] = None,
        enforce_positivity: bool = True,
        epsilon_positivity: float = 1e-8,
        precondition: bool = True,
        tv_n_inner: int = 50,
        adaptive_restart: bool = True,
    ) -> np.ndarray:
        """
        Run FISTA-accelerated preconditioned Landweber with proximal TV.

        Parameters
        ----------
        num_iter : int
            Maximum number of outer (FISTA) iterations.  Clamped to
            [1, 10000].
        lambda_tv : float
            TV regularisation weight.  Set to 0 to disable TV (pure
            Landweber).
        tol : float
            Relative-change convergence threshold.
        min_iter : int
            Minimum iterations before convergence checking begins.
        check_every : int
            Check convergence every this many iterations.
        step_size : float or None
            Outer gradient step size τ.  If None, set automatically:
            0.95 (preconditioned) or 0.95/L (unpreconditioned).
        enforce_positivity : bool
            Project onto x ≥ ε after each proximal step.
        epsilon_positivity : float
            Floor value for the positivity projection.
        precondition : bool
            If True (default), divide the gradient by H^T M to equalise
            convergence speed across the canvas.  Uses a scalar TV
            proximal parameter γ = τ · λ / median(H^T M|_Ω).
        tv_n_inner : int
            Number of inner Chambolle iterations for the TV proximal
            operator.
        adaptive_restart : bool
            Apply O'Donoghue–Candès velocity restart: when consecutive
            iterate steps reverse direction, reset momentum.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original field of view.
        """
        num_iter = int(np.clip(num_iter, 1, 10000))
        eps_pos  = xp.float32(epsilon_positivity)
        use_tv   = (lambda_tv is not None) and (float(lambda_tv) > 0.0)
        lam      = float(lambda_tv) if use_tv else 0.0

        y      = self.image
        M      = self.mask
        PF     = self.PF
        conjPF = self.conjPF
        HTM    = self.HTM
        fshape = self.full_shape
        L      = self._lipschitz

        # ── Step size selection ───────────────────────────────────────────
        # Unpreconditioned:  τ < 2/L  for convergence;  we use 0.95/L.
        # Preconditioned:    τ = 0.95 works because HTM ≈ 1 inside Ω
        #                    and the floor clamp limits the worst case.
        if step_size is not None:
            tau = float(step_size)
        elif precondition:
            tau = 0.95
        else:
            tau = 0.95 / L

        logger.debug(
            "Landweber: τ=%.6f, L=%.6f, precondition=%s, λ_tv=%.2e, "
            "FISTA=%s, restart=%s",
            tau, L, precondition, lam, True, adaptive_restart,
        )

        # ── Proximal TV parameter γ ───────────────────────────────────────
        # Unpreconditioned:  γ = τ · λ
        # Preconditioned:    γ = τ · λ / median(HTM|_Ω)
        #   Median is taken over the observed region Ω only.
        #   HTM ≈ 1 inside Ω for a normalised PSF, so γ ≈ τ · λ.
        if use_tv:
            if precondition:
                htm_inside = HTM[M > 0.5]
                htm_med    = float(xp.median(htm_inside))
                gamma_tv   = tau * lam / max(htm_med, 1e-12)
            else:
                gamma_tv = tau * lam

            logger.debug("TV proximal parameter γ = %.6e", gamma_tv)

        # ── FISTA state ───────────────────────────────────────────────────
        x_k   = self.estimated_image.copy()   # current iterate
        x_km1 = x_k.copy()                    # previous iterate (x_{k-1})
        z_k   = x_k.copy()                    # momentum extrapolation point
        t_k   = 1.0                            # FISTA momentum parameter

        for k in range(num_iter):

            # ── Gradient of data fidelity at z_k ──────────────────────────
            # ∇f(z) = H^T [ M · (H z − y) ]
            Hz       = irfft2(PF * rfft2(z_k), s=fshape)
            residual = M * (Hz - y)
            grad     = irfft2(conjPF * rfft2(residual), s=fshape)

            # ── Gradient descent step ─────────────────────────────────────
            if precondition:
                # Diagonal preconditioner D = HTM (already floored)
                x_half = z_k - tau * (grad / HTM)
            else:
                x_half = z_k - tau * grad

            # ── Proximal TV step ──────────────────────────────────────────
            if use_tv:
                x_new = prox_tv_chambolle(x_half, gamma_tv, n_inner=tv_n_inner)
            else:
                x_new = x_half

            # ── Positivity projection ─────────────────────────────────────
            if enforce_positivity:
                xp.maximum(x_new, eps_pos, out=x_new)

            # ── FISTA momentum update ─────────────────────────────────────
            t_new    = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            momentum = (t_k - 1.0) / t_new

            # Adaptive restart (O'Donoghue & Candès 2015, §3.1):
            # If consecutive step directions are opposed, reset momentum.
            if adaptive_restart and k > 0:
                dx_new = x_new - x_k
                dx_old = x_k   - x_km1
                ip     = float(xp.sum(dx_new * dx_old))
                if ip < 0.0:
                    t_new    = 1.0
                    momentum = 0.0
                    logger.debug(
                        "FISTA restart at iteration %d (ip=%.2e)", k + 1, ip
                    )

            z_new = x_new + momentum * (x_new - x_k)

            # ── Convergence check ─────────────────────────────────────────
            if k >= min_iter and (k + 1) % check_every == 0:
                _, converged = self._check_convergence(
                    x_new, x_k, k=k, num_iter=num_iter, tol=tol,
                )
                if converged:
                    break

            # ── Advance state ─────────────────────────────────────────────
            x_km1 = x_k
            x_k   = x_new
            z_k   = z_new
            t_k   = t_new

        else:
            self._log_no_convergence(num_iter, tol)

        # ── Store, crop, and return ───────────────────────────────────────
        return self._crop_and_return(x_k)


def landweber_deblur_unknown_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Convenience one-shot wrapper for Landweber with unknown boundaries.

    Splits ``**kwargs`` between the :class:`LandweberUnknownBoundary`
    constructor and :meth:`~LandweberUnknownBoundary.deblur` using
    :attr:`DeconvBase._INIT_KEYS`.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    **kwargs
        Any parameter accepted by :class:`LandweberUnknownBoundary`
        (constructor) or :meth:`~LandweberUnknownBoundary.deblur`.

    Returns
    -------
    np.ndarray
        Deconvolved image, shape (H, W) matching the original image
        field of view.
    """
    init_kw   = {k: v for k, v in kwargs.items() if k in DeconvBase._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in DeconvBase._INIT_KEYS}
    obj = LandweberUnknownBoundary(image=image, psf=psf, **init_kw)
    return obj.deblur(**deblur_kw)
