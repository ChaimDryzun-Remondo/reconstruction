"""
pnp_admm.py — Plug-and-Play ADMM with BM3D denoiser.

Inherits from :class:`~.admm.ADMMDeconv` and replaces the explicit TV
regularizer with an implicit prior defined by the BM3D denoiser.

Variable splitting
------------------
Instead of  w = ∇x  (TV split), PnP uses a direct copy split:

    v = Hx   (data fidelity, masked — identical to ADMMDeconv)
    z = x    (denoiser prior — replaces w = ∇x)

Augmented Lagrangian (scaled form):

    L = (1/2)||M ⊙ (v − y)||²
        + (ρ_v/2)||Hx − v + d_v||²
        + (ρ_z/2)||x − z + d_z||²

ADMM steps per iteration
------------------------
1. v-update (INHERITED, unchanged):
       v ← (M ⊙ y + ρ_v(Hx + d_v)) / (M + ρ_v)

2. z-update (DENOISER replaces TV shrinkage):
       σ = sigma_scale · √(λ / ρ_z)
       z ← BM3D(x_old + d_z, σ)

3. x-update (exact FFT solve, scalar denominator — no Laplacian):
       rhs  = ρ_v · Re[ℱ⁻¹(H* · ℱ(v − d_v))]  +  ρ_z · (z − d_z)
       denom = ρ_v |H|² + ρ_z
       x ← Re[ℱ⁻¹(ℱ(rhs) / (denom + ε))]

       Note: ρ_z is a scalar added to every frequency bin, acting as
       Tikhonov-like damping.  No lap_fft needed (z=x split, not ∇x=w).

4. Dual updates (INHERITED d_v; new d_z):
       d_v ← d_v + Hx_new − v
       d_z ← d_z + x_new − z

5. Adaptive ρ_v (INHERITED, Boyd §3.4.1).

Plug-and-Play interpretation
------------------------------
If BM3D is the proximal operator of an implicit regularizer R, then the
ADMM iterates converge to min_x (1/2)||M(Hx−y)||² + λR(x).  BM3D is not
exactly a proximal operator, but PnP-ADMM with bounded denoisers is
provably convergent under mild assumptions (Ryu et al., 2019).

BM3D GPU transfer
-----------------
The bm3d package is CPU-only.  Each z-update therefore transfers the
current iterate from GPU to CPU (via _to_numpy), calls BM3D, and
transfers the result back.  For typical satellite image sizes (1000×4000)
and 30–100 outer ADMM iterations the transfer overhead is negligible
compared to BM3D's own compute time.

Typical parameters (satellite imagery)
---------------------------------------
lambda_tv : 0.001 – 0.05    overall regularization strength
rho_v     : 1.0             data penalty (adaptive ρ adjusts)
rho_z     : 0.1 – 10.0     denoiser coupling strength
sigma_scale: 0.5 – 2.0     fine-tune denoiser level
num_iter  : 30 – 100       PnP converges faster than TV-ADMM

The effective BM3D noise level  σ = sigma_scale · √(λ / ρ_z).
Typical well-conditioned range: σ ∈ [0.01, 0.1].
If σ > 0.2 → over-smoothing; if σ < 0.005 → denoiser barely active.

References
----------
[1] S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg,
    "Plug-and-Play Priors for Model Based Reconstruction,"
    IEEE GlobalSIP, Austin TX, 2013.
[2] E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, W. Yin,
    "Plug-and-Play Methods Provably Converge with Properly Trained
    Denoisers," ICML, 2019.
[3] Y. Mäkinen, L. Azzari, A. Foi, "Collaborative Filtering of Correlated
    Noise: Exact Transform-Domain Variance for Improved Shrinkage and
    Patch Matching," IEEE Trans. Image Process., 29:8339–8354, 2020.
    (BM3D-based correlated-noise framework used by bm3d ≥ 4.0)
[4] S. Boyd et al., "Distributed Optimization and Statistical Learning
    via the Alternating Direction Method of Multipliers,"
    Found. Trends ML, 3(1):1–122, 2011.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._backend import xp, _to_numpy
from .admm import ADMMDeconv

logger = logging.getLogger(__name__)

# ── Optional BM3D import ──────────────────────────────────────────────────────
try:
    from bm3d import bm3d as _bm3d_func
    _HAS_BM3D: bool = True
except ImportError:
    _HAS_BM3D = False


# ══════════════════════════════════════════════════════════════════════════════
# PnPADMM
# ══════════════════════════════════════════════════════════════════════════════

class PnPADMM(ADMMDeconv):
    """
    Plug-and-Play ADMM deconvolution with BM3D denoiser.

    Inherits the full ADMM scaffolding (v-update, x-update FFT solve,
    dual d_v update, adaptive ρ_v, convergence check) from
    :class:`~.admm.ADMMDeconv` and overrides the four prior-interface
    methods to use BM3D instead of TV shrinkage.

    The z = x variable split replaces the w = ∇x split used by the TV
    prior.  This changes only the denominator (ρ_z scalar vs ρ_w · lap_fft)
    and the prior RHS term (ρ_z(z − d_z) vs −ρ_w div(w − d_w)).
    See the module docstring for the full derivation.

    **rho_z / rho_w mapping**: The parent class uses ``rho_w`` as the second
    penalty parameter.  PnPADMM maps ``rho_z → rho_w`` when calling
    ``super().__init__()``, so ``self.rho_w`` holds the denoiser coupling
    strength.  The ``rho_z`` property provides a cleaner alias.  The
    parameter passed to the overridden methods (``_prior_update``,
    ``_x_update_denom``) under the name ``rho_w`` should be read as ``ρ_z``.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB.
    psf : np.ndarray
        Point spread function.
    rho_v : float, optional
        Initial augmented Lagrangian penalty for the v = Hx constraint.
        Adapted online by Boyd §3.4.1.  Default 1.0.
    rho_z : float, optional
        Fixed coupling strength between x and the denoiser output z.
        Higher values make the denoiser more influential per iteration.
        Default 1.0.
    rho_max : float, optional
        Maximum allowed ρ_v.  Default 1024.0.
    rho_min : float, optional
        Minimum allowed ρ_v.  Default 0.03125.
    rho_factor : float, optional
        Multiplicative step for adaptive ρ_v.  Default 1.2.
    denoiser_profile : str, optional
        BM3D profile string: ``'np'`` (normal, good quality/speed balance)
        or ``'lc'`` (low complexity, faster but slightly lower quality).
        Default ``'np'``.
    sigma_scale : float, optional
        Multiplicative factor applied to the computed denoiser noise level
        ``σ = sigma_scale · √(λ / ρ_z)``.  Allows fine-tuning without
        changing λ or ρ_z.  Default 1.0.
    nonneg : bool, optional
        Enforce x ≥ ε_pos after each x-update.  Default True.
    **kwargs
        Passed to :class:`~._base.DeconvBase` (paddingMode, padding_scale,
        initialEstimate, apply_taper_on_padding_band, htm_floor_frac,
        use_mask).

    Raises
    ------
    ImportError
        If the ``bm3d`` package is not installed.

    Notes
    -----
    The ``TVnorm`` parameter inherited from :class:`~.admm.ADMMDeconv` is
    unused by PnPADMM (there is no TV shrinkage step).  It remains in
    ``_INIT_KEYS`` for backward compatibility with generic kwargs splitters
    but has no effect.
    """

    _INIT_KEYS: frozenset[str] = ADMMDeconv._INIT_KEYS | frozenset({
        "rho_z", "denoiser_profile", "sigma_scale",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        rho_v: float = 1.0,
        rho_z: float = 1.0,
        rho_max: float = 1024.0,
        rho_min: float = 0.03125,
        rho_factor: float = 1.2,
        denoiser_profile: str = "np",
        sigma_scale: float = 1.0,
        nonneg: bool = True,
        **kwargs,
    ) -> None:
        if not _HAS_BM3D:
            raise ImportError(
                "PnP-ADMM requires the 'bm3d' package. "
                "Install it with:  pip install bm3d"
            )

        # Map rho_z → rho_w so the parent's deblur() loop sees the correct
        # second penalty under its generic 'rho_w' name.
        super().__init__(
            image, psf,
            rho_v=rho_v,
            rho_w=rho_z,
            rho_max=rho_max,
            rho_min=rho_min,
            rho_factor=rho_factor,
            nonneg=nonneg,
            **kwargs,
        )

        self.denoiser_profile: str = str(denoiser_profile)
        self.sigma_scale: float = float(sigma_scale)

        logger.debug(
            "PnPADMM: rho_z=%.3e, sigma_scale=%.3f, profile=%s",
            rho_z, sigma_scale, denoiser_profile,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Prior interface overrides
    # ══════════════════════════════════════════════════════════════════════

    def _prior_init(self, u: xp.ndarray) -> dict:
        """
        Initialise PnP split variables.

        Default TV prior initialises w = ∇x.  PnP instead initialises
        z = x (identity copy split) and d_z = 0.

        Parameters
        ----------
        u : xp.ndarray
            Initial image estimate on the canvas, float64.

        Returns
        -------
        dict
            State dict with keys ``"z"`` (copy of u) and ``"d_z"`` (zeros).
        """
        return {
            "z": u.copy(),
            "d_z": xp.zeros_like(u),
        }

    def _prior_update(
        self,
        u: xp.ndarray,
        state: dict,
        lambda_tv: float,
        rho_w: float,
        eps: float,
    ) -> xp.ndarray:
        """
        Apply BM3D denoiser to (x_old + d_z) in place of TV shrinkage.

        The BM3D noise level is derived from the ADMM parameters:

            σ = sigma_scale · √(λ / ρ_z)

        This corresponds to the PnP interpretation: if BM3D were the
        proximal operator of some regularizer R, the effective regularization
        strength would be  λ_eff = λ  with denoiser strength σ.

        Called BEFORE the x-update with the OLD iterate u.  Updates
        ``state["z"]`` in place.

        Parameters
        ----------
        u : xp.ndarray
            Current (OLD) image estimate, float64.
        state : dict
            Mutable state dict from _prior_init (keys: z, d_z).
            ``rho_w`` in this signature represents ρ_z for PnP.
        lambda_tv : float
            TV regularization weight (used to derive σ).
        rho_w : float
            Denoiser coupling strength ρ_z (mapped from the parent's rho_w).
        eps : float
            Denominator floor (prevents division by zero when rho_w ≈ 0).

        Returns
        -------
        xp.ndarray
            prior_rhs = ρ_z · (z − d_z), the spatial-domain contribution
            to the x-update RHS.  The parent assembles:
                rhs = ρ_v H^T(v − d_v) + prior_rhs
            then solves  x = Re[ℱ⁻¹(ℱ(rhs) / (denom + ε))].
        """
        rho_z = rho_w  # rename for clarity within this method
        sigma = self.sigma_scale * float(xp.sqrt(lambda_tv / (rho_z + eps)))
        sigma = max(sigma, 1e-6)  # floor: prevent BM3D no-op / instability

        # Save previous z for convergence diagnostics
        state["z_old"] = state["z"].copy()

        denoiser_input = u + state["d_z"]
        state["z"] = self._denoise(denoiser_input, sigma)

        logger.debug("PnP z-update: sigma=%.4f", sigma)

        # Spatial-domain prior RHS: ρ_z (z − d_z)
        return rho_z * (state["z"] - state["d_z"])

    def _prior_dual_update(self, u: xp.ndarray, state: dict) -> None:
        """
        Update the z dual variable after the x-update.

        Applies the Boyd scaled-dual convention:  d_z ← d_z + x_new − z.

        Parameters
        ----------
        u : xp.ndarray
            New image estimate (after x-update + positivity), float64.
        state : dict
            Mutable state dict (keys: z, d_z).
        """
        state["d_z"] += u - state["z"]

    def _x_update_denom(self, rho_v: float, rho_w: float) -> xp.ndarray:
        """
        Denominator for the FFT x-update solve.

        For the z = x split the denominator is:

            denom[k,l] = ρ_v |H(k,l)|² + ρ_z

        where ρ_z is a scalar added to every frequency bin.  This acts as
        Tikhonov-like regularisation in the frequency domain — simpler
        than the  ρ_w · lap_fft  term in the TV case (no Laplacian needed
        because the z = x split does not involve spatial differences).

        Parameters
        ----------
        rho_v : float
            Current (adaptive) data penalty.
        rho_w : float
            Denoiser coupling strength ρ_z (mapped from parent's rho_w).

        Returns
        -------
        xp.ndarray, shape self.full_shape, dtype float64
        """
        return rho_v * self.H_H_conj + rho_w  # scalar broadcast

    # ══════════════════════════════════════════════════════════════════════
    # BM3D wrapper
    # ══════════════════════════════════════════════════════════════════════

    def _denoise(self, image: xp.ndarray, sigma: float) -> xp.ndarray:
        """
        Apply BM3D to an image with automatic GPU↔CPU transfer.

        Steps:
        1. Transfer ``image`` to CPU numpy array (no-op if already CPU).
        2. Clip to [0, 1] (BM3D's expected image range).
        3. Call BM3D with the configured profile and σ.
        4. Clip output to [0, 1] (BM3D can slightly overshoot).
        5. Transfer result back to the active backend (no-op if CPU).

        The dtype of the output matches the dtype of the input.

        Parameters
        ----------
        image : xp.ndarray
            Noisy image on either CPU or GPU, any floating dtype.
        sigma : float
            BM3D noise standard deviation (same scale as image values,
            which are in [0, 1] after clipping).

        Returns
        -------
        xp.ndarray
            Denoised image, same dtype and shape as input.
        """
        if sigma < 1e-6:
            # No meaningful denoising; return as-is to avoid BM3D no-op artefacts
            return image

        image_np = _to_numpy(image).astype(np.float64)
        image_np = np.clip(image_np, 0.0, 1.0)

        denoised_np = _bm3d_func(
            image_np,
            sigma_psd=sigma,
            profile=self.denoiser_profile,
        )
        denoised_np = np.clip(denoised_np, 0.0, 1.0)

        return xp.array(denoised_np, dtype=image.dtype)

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def rho_z(self) -> float:
        """
        Denoiser coupling strength ρ_z (alias for ``self.rho_w``).

        The constructor maps ``rho_z → rho_w`` when initialising the parent.
        This property provides the cleaner public name.
        """
        return self.rho_w


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def pnp_admm_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 50,
    lambda_tv: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    One-shot PnP-ADMM deconvolution with BM3D.

    Splits ``**kwargs`` between :class:`PnPADMM.__init__` and
    :meth:`PnPADMM.deblur` using :attr:`PnPADMM._INIT_KEYS`.

    Uses fewer default iterations than TV-ADMM (50 vs 300) because BM3D
    is computationally expensive per iteration but the stronger prior
    means convergence requires fewer ADMM cycles.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred) image.
    psf : np.ndarray
        Point spread function.
    iters : int, optional
        Maximum ADMM iterations.  Default 50.
    lambda_tv : float, optional
        Regularization weight (controls BM3D σ via σ = sigma_scale · √(λ/ρ_z)).
        Default 0.01.
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
    init_kw = {k: v for k, v in kwargs.items() if k in PnPADMM._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in PnPADMM._INIT_KEYS}
    return PnPADMM(image, psf, **init_kw).deblur(
        num_iter=iters, lambda_tv=lambda_tv, **deblur_kw
    )
