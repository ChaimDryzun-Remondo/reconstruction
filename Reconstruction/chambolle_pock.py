"""
chambolle_pock.py — Condat-Vũ primal-dual forward-backward deconvolution.

Solves the masked deconvolution problem:

    min_x  G(x)  +  F(Kx)

where:
    G(x) = (1/2) ||M ⊙ (Hx − y)||²    smooth masked data fidelity
    F(p) = λ ||p||_{2,1}                isotropic Total Variation (TVnorm=2)
          or λ ||p||_1                   anisotropic TV (TVnorm=1)
    K    = ∇                             forward gradient operator (periodic BC)

Why Condat-Vũ instead of standard Chambolle-Pock
-------------------------------------------------
Standard CP requires prox_{τG} to handle the smooth term.  For our G, the
mask M breaks the shift-invariance of the data-fidelity term, so prox_{τG}
has no closed-form FFT diagonalization.  Condat-Vũ replaces the proximal
step on G with an explicit gradient step ∇G, which is cheap via rfft2.

Algorithm (Condat 2013, Algorithm 1)
-------------------------------------
Per iteration n:

  1. Dual update (gradient of x_bar + projection onto λ-ball):
         (dx, dy) = forward_grad_periodic(x_bar)
         p̃_h = p_h + σ dx,   p̃_w = p_w + σ dy
         (p_h, p_w) ← _dual_project(p̃_h, p̃_w, λ)

  2. Primal gradient:
         Hx    = irfft2(PF · rfft2(x))
         resid = M ⊙ (Hx − y)
         ∇G(x) = irfft2(H* · rfft2(resid))

  3. Primal update (gradient step + dual coupling):
         x_new = x − τ ∇G(x) + τ div_per(p)
         Note: K^T = −div_per, so −τ K^T p = τ div_per(p)

  4. Positivity projection (optional):
         x_new ← max(x_new, 0)

  5. Extrapolation:
         x_bar = x_new + θ (x_new − x)
         [θ=1 (default) gives: x_bar = 2 x_new − x]

  6. Convergence check.

  7. Advance: x ← x_new, p ← (p_h, p_w).

Step sizes (Condat 2013, Theorem 3.1)
--------------------------------------
The convergence condition requires:
    τ⁻¹ − σ ||K||² ≥ L_G / 2

For 2-D periodic ∇, the spectral norm satisfies ||K||² = 8 (the maximum
eigenvalue of −∇^T∇ = 4−2cos(2πf_h)−2cos(2πf_w) at f_h=f_w=0.5).

Safe defaults (with a 1% margin):
    σ = 0.99 / √8
    τ = 0.99 / (L_G/2 + 8σ)

Dual projection (TVnorm=2, isotropic):
    (p̃_h, p̃_w) → (p̃_h, p̃_w) / max(1, ||(p̃_h, p̃_w)||_2 / λ)
    i.e. project each per-pixel 2-vector onto the disk of radius λ.

Dual projection (TVnorm=1, anisotropic):
    p̃_h → clip(p̃_h, −λ, λ)
    p̃_w → clip(p̃_w, −λ, λ)

References
----------
[Con13] Condat, L. "A primal-dual splitting method for convex optimization
        involving Lipschitzian, proximable and linear composite terms."
        J. Optim. Theory Appl., 158(2):460–479, 2013.

[CV12]  Chambolle, A. & Pock, T. "A first-order primal-dual algorithm for
        convex problems with applications to imaging." J. Math. Imaging
        Vision, 40(1):120–145, 2011.

[Was20] Wasilewski, P. "Condat-Vu splitting: a unified primal-dual
        framework." (Tutorial notes, 2020.)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ._backend import xp, rfft2, irfft2
from ._base import DeconvBase
from ._tv_operators import forward_grad_periodic, backward_div_periodic

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ChambollePockDeconv
# ══════════════════════════════════════════════════════════════════════════════

class ChambollePockDeconv(DeconvBase):
    """
    Condat-Vũ primal-dual forward-backward deconvolution with TV regularization.

    Implements Condat (2013) Algorithm 1 for the problem:

        min_x  (1/2)||M ⊙ (Hx − y)||²  +  λ TV(x)

    where TV is either isotropic (TVnorm=2, per-pixel vector projection)
    or anisotropic (TVnorm=1, componentwise clamping).

    Unlike standard Chambolle-Pock, the smooth data-fidelity term G is
    handled via an explicit gradient step rather than a proximal operator,
    enabling efficient FFT-based computation even when the mask M is present.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    sigma_dual : float or None, optional
        Dual step size σ.  If ``None`` (default), uses the safe default
        ``0.99 / sqrt(8)`` (one percent below the theoretical maximum for
        2-D periodic ∇).
    theta : float, optional
        Extrapolation parameter θ ∈ [0, 1].  Default 1.0 (full
        extrapolation, standard Condat-Vũ).  θ=0 disables extrapolation
        (plain forward-backward splitting).
    nonneg : bool, optional
        If ``True`` (default), enforce non-negativity after each primal
        update.  Can be overridden at :meth:`deblur` call time.
    TVnorm : {1, 2}, optional
        TV norm variant.  2 = isotropic (per-pixel ‖·‖₂ projection),
        1 = anisotropic (componentwise clamp).  Default 2.
    **base_kwargs
        Forwarded to :class:`~Reconstruction._base.DeconvBase`:
        ``paddingMode``, ``padding_scale``, ``initialEstimate``, etc.

    Attributes
    ----------
    sigma_dual : float
        Dual step size σ.
    tau_primal : float
        Primal step size τ, computed from σ to satisfy the Condat (2013)
        convergence condition: τ⁻¹ − σ · 8 ≥ L_G / 2.
    lipschitz : float
        Lipschitz constant of ∇G, L = max |H(f)|².

    Notes
    -----
    ``use_mask=True`` is always forced: the mask is essential for the
    unknown-boundary formulation.

    The periodic BC gradient/divergence pair (forward_grad_periodic /
    backward_div_periodic) is used here because the dual update is part of
    the global iteration structure, not a standalone proximal sub-problem.
    This ensures the adjoint relation ⟨K^T p, x⟩ = ⟨p, Kx⟩ holds
    consistently throughout — see CLAUDE.md pitfall #8.
    """

    _INIT_KEYS: frozenset[str] = DeconvBase._INIT_KEYS | frozenset({
        "sigma_dual",
        "theta",
        "nonneg",
        "TVnorm",
    })

    # Spectral norm of the 2-D periodic forward-difference gradient:
    #   ||∇||² = max_f { 4 − 2cos(2πf_h) − 2cos(2πf_w) } = 8   (at f_h=f_w=0.5)
    _K_NORM_SQ: float = 8.0

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        sigma_dual: Optional[float] = None,
        theta: float = 1.0,
        nonneg: bool = True,
        TVnorm: int = 2,
        **base_kwargs,
    ) -> None:
        super().__init__(image, psf, use_mask=True, **base_kwargs)

        if TVnorm not in (1, 2):
            raise ValueError(
                f"TVnorm must be 1 or 2, got {TVnorm!r}.  "
                "Use 2 for isotropic TV or 1 for anisotropic TV."
            )

        self._TVnorm: int = int(TVnorm)
        self._theta: float = float(theta)
        self.nonneg: bool = bool(nonneg)

        # ── Step sizes (Condat 2013, Theorem 3.1) ─────────────────────────
        # Convergence condition:  τ⁻¹ − σ ||K||² ≥ L_G / 2
        # Safe defaults with 1 % margin below theoretical boundary:
        #   σ_default = 0.99 / sqrt(8)
        #   τ         = 0.99 / (L_G/2 + 8 σ)
        if sigma_dual is None:
            self._sigma: float = 0.99 / math.sqrt(self._K_NORM_SQ)
        else:
            self._sigma = float(sigma_dual)

        self._tau: float = 0.99 / (
            self._lipschitz / 2.0 + self._K_NORM_SQ * self._sigma
        )

        logger.debug(
            "ChambollePockDeconv: L=%.6f, σ=%.6f, τ=%.6f, "
            "θ=%.2f, TVnorm=%d, nonneg=%s",
            self._lipschitz, self._sigma, self._tau,
            self._theta, self._TVnorm, self.nonneg,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def sigma_dual(self) -> float:
        """Dual step size σ."""
        return self._sigma

    @property
    def tau_primal(self) -> float:
        """Primal step size τ, satisfying τ⁻¹ − 8σ ≥ L_G/2."""
        return self._tau

    @property
    def lipschitz(self) -> float:
        """Lipschitz constant L = max |H(f)|²."""
        return self._lipschitz

    # ══════════════════════════════════════════════════════════════════════
    # Main algorithm
    # ══════════════════════════════════════════════════════════════════════

    def deblur(
        self,
        num_iter: int = 200,
        lambda_tv: float = 0.01,
        tol: float = 1e-5,
        min_iter: int = 10,
        check_every: int = 5,
        nonneg: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Run the Condat-Vũ primal-dual forward-backward algorithm.

        Parameters
        ----------
        num_iter : int
            Maximum outer iterations.  Clamped to [1, 10000].
        lambda_tv : float
            TV regularization weight λ.  Typical range: 1e-3 to 1e-1 for
            satellite imagery normalized to [0, 1].
        tol : float
            Relative-change convergence threshold on the primal variable.
        min_iter : int
            Minimum iterations before convergence checking begins.
        check_every : int
            Check convergence every this many iterations.
        nonneg : bool or None
            Non-negativity enforcement.  ``None`` uses the constructor value.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image on CPU, cropped to the original field of view.
        """
        num_iter    = int(np.clip(num_iter, 1, 10000))
        nonneg_flag = self.nonneg if nonneg is None else bool(nonneg)

        sigma = xp.float32(self._sigma)
        tau   = xp.float32(self._tau)
        theta = xp.float32(self._theta)
        lam   = xp.float32(lambda_tv)
        s     = self.full_shape
        PF    = self.PF
        cPF   = self.conjPF
        M     = self.mask
        y     = self.image

        logger.debug(
            "CP deblur: num_iter=%d, λ=%.2e, σ=%.4e, τ=%.4e, "
            "θ=%.2f, TVnorm=%d, nonneg=%s",
            num_iter, lambda_tv, self._sigma, self._tau,
            self._theta, self._TVnorm, nonneg_flag,
        )

        # ── State initialization ─────────────────────────────────────────
        x     = self.estimated_image.copy()   # primal variable
        x_bar = x.copy()                      # extrapolated point
        p_h   = xp.zeros_like(x)             # dual variable (vertical)
        p_w   = xp.zeros_like(x)             # dual variable (horizontal)

        for k in range(num_iter):
            x_old = x

            # ── 1. Dual update ────────────────────────────────────────────
            # Gradient of the extrapolated point
            dx_bar, dy_bar = forward_grad_periodic(x_bar)
            p_h_tilde = p_h + sigma * dx_bar
            p_w_tilde = p_w + sigma * dy_bar
            del dx_bar, dy_bar

            # Project onto the constraint set for F*
            p_h_new, p_w_new = self._dual_project(p_h_tilde, p_w_tilde, lam)
            del p_h_tilde, p_w_tilde

            # ── 2. Primal gradient ∇G(x) = H^T [M ⊙ (Hx − y)] ──────────
            Hx    = irfft2(PF * rfft2(x), s=s)
            resid = M * (Hx - y)
            grad_G = irfft2(cPF * rfft2(resid), s=s)
            del Hx, resid

            # ── 3. Primal update ─────────────────────────────────────────
            # x_{n+1} = x_n − τ ∇G(x_n) + τ div_per(p_{n+1})
            # (because K^T = −div_per, so −τ K^T p = τ div_per(p))
            div_p  = backward_div_periodic(p_h_new, p_w_new)
            x_new  = x - tau * grad_G + tau * div_p
            del grad_G, div_p

            # ── 4. Positivity projection ─────────────────────────────────
            if nonneg_flag:
                xp.maximum(x_new, xp.float32(0.0), out=x_new)

            # ── 5. Convergence check ─────────────────────────────────────
            if k >= min_iter and (k + 1) % check_every == 0:
                _, converged = self._check_convergence(
                    x_new, x_old, k=k, num_iter=num_iter, tol=tol,
                )
                if converged:
                    x = x_new   # return the converged iterate
                    break

            # ── 6. Extrapolation: x_bar = x_new + θ (x_new − x) ─────────
            x_bar = x_new + theta * (x_new - x)

            # ── 7. Advance state ─────────────────────────────────────────
            x   = x_new
            p_h = p_h_new
            p_w = p_w_new

        else:
            self._log_no_convergence(num_iter, tol)

        return self._crop_and_return(x)

    # ══════════════════════════════════════════════════════════════════════
    # Dual projection
    # ══════════════════════════════════════════════════════════════════════

    def _dual_project(
        self,
        p_h: "xp.ndarray",
        p_w: "xp.ndarray",
        lam: "xp.ndarray",
    ) -> "tuple[xp.ndarray, xp.ndarray]":
        """
        Project the dual variable onto the feasible set for the selected TV norm.

        For TVnorm=2 (isotropic TV), the constraint is ‖p_{i,j}‖₂ ≤ λ
        (per-pixel vector projection onto a disk of radius λ).

        For TVnorm=1 (anisotropic TV), the constraint is |p_{i,j}| ≤ λ
        componentwise (clamp each component to [−λ, λ]).

        Parameters
        ----------
        p_h, p_w : xp.ndarray, shape (H, W)
            Tentative dual variable after the gradient step.
        lam : xp.ndarray (scalar float32)
            TV regularization weight λ.

        Returns
        -------
        p_h_new, p_w_new : xp.ndarray, each shape (H, W)
            Projected dual variable.
        """
        if self._TVnorm == 2:
            # Isotropic: project per-pixel (p_h, p_w) vector onto the λ-disk.
            # scale = λ / max(‖p‖₂, λ) — equals 1 when ‖p‖₂ ≤ λ (no-op).
            mag   = xp.sqrt(p_h * p_h + p_w * p_w)
            scale = lam / xp.maximum(mag, lam)
            return p_h * scale, p_w * scale
        else:
            # Anisotropic: clamp each component to [−λ, λ].
            return xp.clip(p_h, -lam, lam), xp.clip(p_w, -lam, lam)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def chambolle_pock_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 200,
    lambda_tv: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    One-shot Condat-Vũ primal-dual deconvolution.

    Splits ``**kwargs`` between :class:`ChambollePockDeconv` constructor
    parameters (those in :attr:`ChambollePockDeconv._INIT_KEYS`) and
    :meth:`~ChambollePockDeconv.deblur` parameters (everything else).

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    iters : int
        Maximum iterations.  Default 200.
    lambda_tv : float
        TV regularization weight λ.  Default 0.01.
    **kwargs
        Any parameter accepted by :class:`ChambollePockDeconv` or
        :meth:`~ChambollePockDeconv.deblur`
        (e.g. ``padding_scale``, ``TVnorm``, ``tol``).

    Returns
    -------
    np.ndarray
        Deconvolved image, shape (H, W) matching the original image FOV.
    """
    init_kw   = {k: v for k, v in kwargs.items()
                 if k in ChambollePockDeconv._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items()
                 if k not in ChambollePockDeconv._INIT_KEYS}
    solver = ChambollePockDeconv(image, psf, **init_kw)
    return solver.deblur(num_iter=iters, lambda_tv=lambda_tv, **deblur_kw)
