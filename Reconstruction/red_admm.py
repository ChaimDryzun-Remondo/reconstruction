"""
red_admm.py — Regularization by Denoising via ADMM (RED-ADMM).

Inherits from :class:`~.admm.ADMMDeconv` and replaces the explicit TV
regularizer with the RED prior (Romano et al. 2017).

RED prior
---------
Romano et al. define an explicit regularizer using a denoiser D_σ:

    ρ(x) = (1/2) x^T (x − D_σ(x))

Under the Jacobian symmetry assumption (∂D/∂x symmetric), its gradient is:

    ∇ρ(x) = x − D_σ(x)

This makes the full objective explicitly differentiable, giving RED
stronger theoretical grounding than PnP-ADMM (which uses an implicit
regularizer defined by the denoiser's fixed-point equation).

RED vs PnP-ADMM
---------------
Both RED and PnP use a denoiser as a prior, but differ in how the
denoiser enters the optimization:

  PnP-ADMM:  z = x split; denoiser ≈ prox of implicit R(x).
             σ adapts each iteration: σ = sigma_scale · √(λ/ρ_z).
             THREE variable splits (v, z, x); TWO dual variables.

  RED-ADMM:  No z-split needed; denoiser provides the gradient ∇ρ.
             σ is FIXED throughout (not adaptive).
             TWO variable splits (v, x); ONE dual variable (d_v).
             Denominator is ρ_v|H|² + λ (scalar λ, no Laplacian).

Variable splitting
------------------
Introduce v = Hx for masked fidelity:

    min_{x,v}  (1/2)||M ⊙ (v − y)||²  +  (λ/2) x^T(x − D_σ(x))
    s.t.  v = Hx

Augmented Lagrangian (scaled dual d_v):

    L = (1/2)||M(v−y)||²  +  (λ/2)x^T(x−D_σ(x))
        + (ρ_v/2)||Hx − v + d_v||²

ADMM blocks per iteration
--------------------------
1. v-update (INHERITED from ADMMDeconv, unchanged):
       v ← (M ⊙ y + ρ_v(Hx + d_v)) / (M + ρ_v)

2. x-update (fixed-point linearization, denoiser lagged at x_prev):
       (ρ_v H^T H + λI) x = ρ_v H^T(v − d_v) + λ D_σ(x_prev)

   In Fourier domain (exact solution):
       denom   = ρ_v |H(f)|² + λ
       prior_rhs = λ D_σ(x_prev)      [spatial; parent FFTs internally]
       x̂ = FFT(ρ_v H^T(v−d_v) + prior_rhs) / (denom + ε)
       x = Re[IFFT(x̂)]

   Note: The denominator has scalar +λ (no Laplacian), unlike TV-ADMM
   which uses ρ_w · lap_fft.  This is the same structure as PnP-ADMM's
   ρ_z scalar, but the prior_rhs is λ D(x) instead of ρ_z(z − d_z).

3. Dual v-update (INHERITED):
       d_v ← d_v + Hx_new − v

4. Adaptive ρ_v (INHERITED, Boyd §3.4.1).

Mapping to ADMMDeconv prior interface
--------------------------------------
Override method     Description
-----------------   -----------------------------------------------
_prior_init         Returns {"denoised": u.copy()}  (no w/z variables)
_prior_update       Calls BM3D; returns λ · D_σ(x_prev)
_prior_dual_update  No-op (RED has no prior dual variable)
_x_update_denom     Returns ρ_v |H|² + λ  (scalar λ, not ρ_w·lap_fft)

The λ for the denominator comes from the ``lambda_reg`` passed to
:meth:`deblur`, not from the constructor.  :meth:`deblur` captures it as
``self._current_lambda`` before calling the parent loop, so
``_x_update_denom`` can access it.

References
----------
[REM17] Romano, Y., Elad, M. & Milanfar, P. "The little engine that
        could: Regularization by Denoising (RED)." SIAM J. Imaging Sci.,
        10(4):1804–1844, 2017.

[RS18]  Reehorst, E.T. & Schniter, P. "Regularization by Denoising:
        Clarifications and New Interpretations." IEEE Trans. Comp.
        Imaging, 5(1):52–67, 2019.

[Boy11] Boyd, S. et al. "Distributed Optimization and Statistical Learning
        via ADMM." Found. Trends ML, 3(1):1–122, 2011.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._backend import xp
from ._denoise_utils import _HAS_BM3D, bm3d_denoise
from .admm import ADMMDeconv

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# REDDeconv
# ══════════════════════════════════════════════════════════════════════════════

class REDDeconv(ADMMDeconv):
    """
    Regularization by Denoising ADMM (RED-ADMM) deconvolution.

    Inherits the full ADMM scaffolding (v-update, x-update FFT solve,
    dual d_v update, adaptive ρ_v, convergence checking, cost tracking)
    from :class:`~.admm.ADMMDeconv` and overrides the four prior-interface
    methods to use the RED explicit gradient ∇ρ(x) = λ(x − D_σ(x)).

    Unlike PnP-ADMM which introduces an extra z = x split, RED-ADMM uses
    only the data-fidelity split v = Hx.  The denoiser σ is **fixed**
    throughout all iterations (set at construction time), giving more
    predictable regularization strength.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB.
    psf : np.ndarray
        Point spread function.
    sigma : float, optional
        Fixed BM3D denoiser noise level σ.  Typical range for satellite
        imagery normalized to [0, 1]: 0.01–0.10.  Default 0.05.
    denoiser_profile : str, optional
        BM3D profile.  ``'np'`` (normal profile, default) gives the best
        quality/speed trade-off; ``'lc'`` is faster but lower quality.
    rho_v : float, optional
        Initial data penalty (adaptive).  Default 1.0.
    nonneg : bool, optional
        Enforce x ≥ ε_pos after each x-update.  Default True.
    **kwargs
        Any parameter accepted by :class:`~.admm.ADMMDeconv`
        (``rho_w``, ``rho_max``, ``rho_min``, ``rho_factor``,
        ``paddingMode``, ``padding_scale``, etc.).

        Note: ``rho_w`` is inherited from ADMMDeconv but is unused by
        RED-ADMM — the x-update denominator uses ``lambda_reg`` (the
        value passed to :meth:`deblur`) rather than ``rho_w``.

    Raises
    ------
    ImportError
        If the ``bm3d`` package is not installed.

    Notes
    -----
    ``use_mask=True`` is always forced (inherited from ADMMDeconv default).

    The fixed σ differs from PnP-ADMM where σ varies each iteration as
    ``sigma_scale · √(λ/ρ_z)``.  RED's fixed σ is simpler to tune: choose
    σ to match the noise level you want the denoiser to suppress.

    ``TVnorm`` inherited from ADMMDeconv has no effect in RED-ADMM (there
    is no TV shrinkage step), but it controls the TV term in the cost
    display.  Since RED's prior state has no ``w_h``/``w_w`` keys,
    ``_compute_admm_cost`` gracefully skips the TV term.
    """

    _INIT_KEYS: frozenset[str] = ADMMDeconv._INIT_KEYS | frozenset({
        "sigma",
        "denoiser_profile",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        sigma: float = 0.05,
        denoiser_profile: str = "np",
        rho_v: float = 1.0,
        nonneg: bool = True,
        **kwargs,
    ) -> None:
        if not _HAS_BM3D:
            raise ImportError(
                "RED-ADMM requires the 'bm3d' package. "
                "Install with:  pip install bm3d"
            )

        super().__init__(
            image, psf,
            rho_v=rho_v,
            nonneg=nonneg,
            **kwargs,
        )

        self._sigma: float = float(sigma)
        self.denoiser_profile: str = str(denoiser_profile)

        # Will be set at the start of each deblur() call to the current
        # lambda_reg value, enabling _x_update_denom to access it.
        self._current_lambda: float = 0.01

        logger.debug(
            "REDDeconv: sigma=%.4f, profile=%s, rho_v=%.3e",
            self._sigma, self.denoiser_profile, rho_v,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def sigma(self) -> float:
        """Fixed BM3D denoiser noise level σ (set at construction time)."""
        return self._sigma

    # ══════════════════════════════════════════════════════════════════════
    # deblur — captures lambda_reg before delegating to parent loop
    # ══════════════════════════════════════════════════════════════════════

    def deblur(
        self,
        num_iter: int = 50,
        lambda_reg: float = 0.01,
        tol: float = 1e-4,
        min_iter: int = 5,
        check_every: int = 1,
        nonneg: Optional[bool] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run RED-ADMM deconvolution.

        Parameters
        ----------
        num_iter : int
            Maximum ADMM iterations.  Default 50 (RED converges faster
            than TV-ADMM due to the stronger BM3D denoiser).
        lambda_reg : float
            Regularization weight λ.  Controls both the weight of the RED
            gradient term (λ D(x) on the RHS) and the denominator shift
            (ρ_v|H|² + λ).  Typical range: 0.001–0.05 for satellite imagery.
        tol : float
            Convergence threshold for the scale-normalised v-constraint
            residual.
        min_iter : int
            Minimum iterations before early termination is allowed.
        check_every : int
            Check convergence every this many iterations.
        nonneg : bool or None
            Override constructor ``nonneg`` flag.
        verbose : bool
            Log per-iteration details at DEBUG level.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original FOV, on CPU.
        """
        # Capture lambda_reg so that _x_update_denom can use it.
        # Must be set BEFORE calling super().deblur() which enters the loop.
        self._current_lambda = float(lambda_reg)

        logger.debug(
            "REDDeconv.deblur: num_iter=%d, lambda_reg=%.3e, sigma=%.4f",
            num_iter, lambda_reg, self._sigma,
        )

        return super().deblur(
            num_iter=num_iter,
            lambda_tv=lambda_reg,
            tol=tol,
            min_iter=min_iter,
            check_every=check_every,
            nonneg=nonneg,
            verbose=verbose,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Prior interface overrides
    # ══════════════════════════════════════════════════════════════════════

    def _prior_init(self, u: "xp.ndarray") -> dict:
        """
        Initialise RED prior state.

        RED has no auxiliary primal variables (no w = ∇x or z = x split).
        Only the last denoised image is stored, for monitoring.

        Parameters
        ----------
        u : xp.ndarray
            Initial image estimate, float64.

        Returns
        -------
        dict
            State dict with key ``"denoised"`` (copy of the initial estimate).
        """
        return {"denoised": u.copy()}

    def _prior_update(
        self,
        u: "xp.ndarray",
        state: dict,
        lambda_tv: float,
        rho_w: float,
        eps: float,
    ) -> "xp.ndarray":
        """
        Compute the RED prior contribution to the x-update RHS.

        Applies the fixed-sigma BM3D denoiser to the current iterate ``u``
        (lagged denoiser evaluation), then returns:

            prior_rhs = λ · D_σ(u)

        so that the x-update RHS assembles to:

            rhs = ρ_v H^T(v − d_v) + λ D_σ(u_prev)

        Combined with the denominator ρ_v|H|² + λ (from _x_update_denom),
        this solves the fixed-point linearized RED x-update:

            (ρ_v H^T H + λI) x = ρ_v H^T(v − d_v) + λ D_σ(u_prev)

        Note: The RHS is **λ D(x)**, NOT **λ(x − D(x))** (the gradient
        ∇ρ).  The x-term cancels into the denominator as the scalar +λ,
        leaving only λ D(x) on the right-hand side.

        Parameters
        ----------
        u : xp.ndarray
            Current (OLD) image estimate, float64.
        state : dict
            Mutable state dict from _prior_init (key: ``"denoised"``).
        lambda_tv : float
            Regularization weight λ (named lambda_tv for parent compat.).
        rho_w : float
            Parent's prior penalty (unused by RED — present for interface
            compatibility only).
        eps : float
            Denominator floor (unused by RED).

        Returns
        -------
        xp.ndarray
            prior_rhs = λ · D_σ(u), spatial domain, same shape as u.
        """
        denoised = self._denoise(u, self._sigma)
        state["denoised"] = denoised
        logger.debug("RED prior_update: sigma=%.4f", self._sigma)
        return lambda_tv * denoised

    def _prior_dual_update(self, u: "xp.ndarray", state: dict) -> None:
        """
        No-op: RED-ADMM has no prior dual variable.

        Unlike PnP-ADMM (which has d_z) and TV-ADMM (which has d_w),
        RED uses only the data-fidelity dual d_v (managed by the parent).
        The state dict is left unchanged.

        Parameters
        ----------
        u : xp.ndarray
            New image estimate (unused).
        state : dict
            Prior state dict (unchanged by this method).
        """

    def _x_update_denom(self, rho_v: float, rho_w: float) -> "xp.ndarray":
        """
        Denominator array for the FFT x-update solve.

        For RED, the denominator is:

            denom[k, l] = ρ_v |H(k, l)|² + λ

        where λ = ``self._current_lambda`` (set at the start of each
        :meth:`deblur` call).  This is a scalar shift (Tikhonov-like
        regularization in the frequency domain) — no Laplacian term,
        unlike TV-ADMM's ``ρ_w · lap_fft``.

        Parameters
        ----------
        rho_v : float
            Current (adaptive) data penalty.
        rho_w : float
            Parent's prior penalty (unused by RED).

        Returns
        -------
        xp.ndarray, shape self.full_shape, dtype float64
        """
        return rho_v * self.H_H_conj + self._current_lambda

    # ══════════════════════════════════════════════════════════════════════
    # BM3D wrapper
    # ══════════════════════════════════════════════════════════════════════

    def _denoise(self, image: "xp.ndarray", sigma: float) -> "xp.ndarray":
        """
        Apply BM3D denoising with automatic GPU↔CPU transfer.

        Delegates to :func:`~._denoise_utils.bm3d_denoise`.

        Parameters
        ----------
        image : xp.ndarray
            Image to denoise (float64, values expected in [0, 1]).
        sigma : float
            BM3D noise standard deviation.

        Returns
        -------
        xp.ndarray
            Denoised image, same dtype and shape as input.
        """
        return bm3d_denoise(image, sigma, self.denoiser_profile)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def red_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 50,
    lambda_reg: float = 0.01,
    sigma: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    One-shot RED-ADMM deconvolution with BM3D.

    Splits ``**kwargs`` between :class:`REDDeconv.__init__` and
    :meth:`REDDeconv.deblur` using :attr:`REDDeconv._INIT_KEYS`.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred) image.
    psf : np.ndarray
        Point spread function.
    iters : int, optional
        Maximum ADMM iterations.  Default 50.
    lambda_reg : float, optional
        Regularization weight λ.  Default 0.01.
    sigma : float, optional
        Fixed BM3D denoiser noise level σ.  Default 0.05.
    **kwargs
        Forwarded to the constructor and/or ``deblur()``.

    Returns
    -------
    np.ndarray
        Deconvolved image.

    Raises
    ------
    ImportError
        If the ``bm3d`` package is not installed.
    """
    init_kw   = {k: v for k, v in kwargs.items() if k in REDDeconv._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in REDDeconv._INIT_KEYS}
    solver = REDDeconv(image, psf, sigma=sigma, **init_kw)
    return solver.deblur(num_iter=iters, lambda_reg=lambda_reg, **deblur_kw)
