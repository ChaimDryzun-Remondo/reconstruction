"""
tval3.py — TVAL3 deconvolution with v=Hx split three-block ADMM.

Solves the masked deconvolution problem:

    min_x  (1/2) ||M ⊙ (Hx − y)||²  +  λ_tv ||∇x||_{2,1}

via the variable split v = Hx, w = ∇x with augmented Lagrangian (scaled
dual variable convention, Boyd et al. 2011):

    L = (1/2)||M(v−y)||²  +  λ||w||_{TV}
        + (ρ_v/2)||Hx − v + d_v||²
        + (ρ_w/2)||∇x − w + d_w||²

Three ADMM blocks per iteration
---------------------------------
1. v-update (pointwise closed form, masked data fidelity):
       v ← (M ⊙ y + ρ_v(Hx + d_v)) / (M + ρ_v)
   M=1 inside FOV: weighted average of data and prediction.
   M=0 outside FOV: v = Hx + d_v (no data constraint).

2. x-update (exact FFT solve, periodic BC Laplacian eigenvalues):
       rhs = ρ_v · Re[ℱ⁻¹(H* · ℱ(v − d_v))]
             − ρ_w · backward_div_periodic(w − d_w)
       denom = ρ_v · |H|² + ρ_w · lap_fft
       x ← Re[ℱ⁻¹(ℱ(rhs) / (denom + ε))]
   where lap_fft = 4 − 2cos(2πf_y) − 2cos(2πf_x)  (periodic BC eigenvalues
   of −∇^T∇, which equal the eigenvalues of G^TG for forward differences G).

3. w-update (vectorial shrinkage for isotropic or anisotropic TV):
       w ← shrink(∇x + d_w, λ/ρ_w)

4. Dual updates (Boyd convention, d increases by constraint violation):
       d_v ← d_v + Hx − v
       d_w ← d_w + ∇x − w

5. Adaptive ρ_v (Boyd et al. 2011, §3.4.1):
   Increase ρ_v when primal residual dominates, decrease otherwise.

The mask M enters ONLY in the v-update; the x-update is mask-free and
exactly diagonalizable in the Fourier domain.  This is the key innovation
of the v=Hx split over the direct ATb approach.

Precision notes
---------------
Internal computation uses float64 for numerical stability of dual variable
accumulation across many iterations.  Inputs and outputs remain float32/numpy
as per the package convention.

References
----------
[1] C. Li, W. Yin, H. Jiang, Y. Zhang, "An Efficient Augmented Lagrangian
    Method with Applications to Total Variation Minimization,"
    Computational Optimization and Applications, 56(3):507–530, 2013.
[2] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein, "Distributed
    Optimization and Statistical Learning via the Alternating Direction
    Method of Multipliers," Foundations and Trends in ML, 3(1):1–122, 2011.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._backend import (
    xp,
    fft2, ifft2, fftfreq,
    rfft2, irfft2,
    ifftshift,
    _freeze, _to_numpy,
)
from ._base import DeconvBase
from ._tv_operators import forward_grad_periodic, backward_div_periodic

logger = logging.getLogger(__name__)

# Module-level constants (float64 context)
_EPSILON: float = 1e-8
_EPS_GRAD: float = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# TVAL3Deconv
# ══════════════════════════════════════════════════════════════════════════════

class TVAL3Deconv(DeconvBase):
    """
    TVAL3 deconvolution with v=Hx split three-block ADMM.

    See module docstring for the full mathematical formulation.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB.
    psf : np.ndarray
        Point spread function.
    mu : float, optional
        Initial augmented Lagrangian penalty ρ_v (and ρ_w).  Default 32.
    mu_max : float, optional
        Maximum allowed ρ_v.  Default 1024.
    mu_min : float, optional
        Minimum allowed ρ_v.  Default 0.03125 (= 2⁻⁵).
    mu_factor : float, optional
        Multiplicative step for adaptive ρ_v updates.  Default 1.2.
    TVnorm : {1, 2}, optional
        TV norm variant.  1 = anisotropic, 2 = isotropic.  Default 2.
    nonneg : bool, optional
        Enforce x ≥ ε_pos after each x-update.  Default True.
    adaptive_tv : bool, optional
        Use spatially-varying adaptive TV weights based on edge strength.
        Default True.
    burn_in_frac : float, optional
        Fraction of iterations to run with uniform TV before switching to
        adaptive (when adaptive_tv=True).  Default 0.2.
    **kwargs
        Passed to :class:`~._base.DeconvBase` (paddingMode, padding_scale,
        initialEstimate, apply_taper_on_padding_band, htm_floor_frac,
        use_mask).

    Notes
    -----
    use_mask defaults to True (inherited from DeconvBase).  Setting
    use_mask=False makes M = 1 everywhere, disabling the boundary
    abstraction and making the v-update a simple weighted average of y
    and Hx without any masking benefit.
    """

    _INIT_KEYS: frozenset[str] = DeconvBase._INIT_KEYS | frozenset({
        "mu", "mu_max", "mu_min", "mu_factor",
        "TVnorm", "nonneg", "adaptive_tv", "burn_in_frac",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        mu: float = 32.0,
        mu_max: float = 1024.0,
        mu_min: float = 0.03125,
        mu_factor: float = 1.2,
        TVnorm: int = 2,
        nonneg: bool = True,
        adaptive_tv: bool = True,
        burn_in_frac: float = 0.2,
        **kwargs,
    ) -> None:
        # use_mask=True by default (access to self.mask required for v-update).
        super().__init__(image, psf, **kwargs)

        # ── Full complex PSF spectrum in float64 ───────────────────────────
        # Recover the spatial (ifftshifted) PSF from the base-class rfft2
        # precomputation, cast to float64, then compute full M×N complex FFT.
        # This avoids duplicating the PSF preprocessing pipeline.
        psf_spatial: xp.ndarray = irfft2(
            self.PF, s=self.full_shape
        ).astype(xp.float64)

        H_full = fft2(psf_spatial)
        self.H_full: xp.ndarray = _freeze(H_full)
        self.H_conj_full: xp.ndarray = _freeze(H_full.conj().copy())
        self.H_H_conj: xp.ndarray = _freeze(
            xp.real(self.H_conj_full * self.H_full).copy()
        )

        # ── Laplacian eigenvalue tensor (periodic BC) ──────────────────────
        # lap_fft[k,l] = 4 − 2cos(2πk/M) − 2cos(2πl/N)
        # These are the eigenvalues of G^TG for the forward-difference gradient
        # G under periodic BC.  Required for the exact FFT x-solve.
        M_f, N_f = self.full_shape
        fy = fftfreq(M_f).reshape(-1, 1).astype(xp.float64)
        fx = fftfreq(N_f).reshape(1, -1).astype(xp.float64)
        self.lap_fft: xp.ndarray = _freeze(
            (4.0 - 2.0 * xp.cos(2.0 * xp.pi * fy)
             - 2.0 * xp.cos(2.0 * xp.pi * fx)).copy()
        )

        logger.debug(
            "TVAL3 precomputed H_full (%s), lap_fft (%s) on canvas %s.",
            H_full.dtype, self.lap_fft.dtype, self.full_shape,
        )

        # ── Algorithm hyperparameters ──────────────────────────────────────
        self.mu: float = float(mu)
        self.mu_max: float = float(mu_max)
        self.mu_min: float = float(mu_min)
        self.mu_factor: float = float(mu_factor)
        self.TVnorm: int = int(TVnorm)
        self.nonneg: bool = bool(nonneg)
        self.adaptive_tv: bool = bool(adaptive_tv)
        self.burn_in_frac: float = float(burn_in_frac)

        # ── State (populated by deblur) ────────────────────────────────────
        self.costs: list[float] = []
        self._last_mu: float = self.mu

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _shrink(
        self,
        x: xp.ndarray,
        y: xp.ndarray,
        thresh,          # scalar float or xp.ndarray (for adaptive TV)
        eps: float,
        tvnorm: int,
    ) -> tuple[xp.ndarray, xp.ndarray]:
        """
        Shrinkage / vectorial soft-thresholding.

        Parameters
        ----------
        x, y : xp.ndarray
            Horizontal and vertical gradient components.
        thresh : float or xp.ndarray
            Uniform or spatially-varying threshold.
        eps : float
            Denominator floor for TVnorm=2 (isotropic).
        tvnorm : int
            1 for anisotropic componentwise, 2 for isotropic vectorial.

        Returns
        -------
        (x_s, y_s) : tuple[xp.ndarray, xp.ndarray]
        """
        if tvnorm == 1:
            return (
                xp.sign(x) * xp.maximum(xp.abs(x) - thresh, 0.0),
                xp.sign(y) * xp.maximum(xp.abs(y) - thresh, 0.0),
            )
        else:
            mag = xp.sqrt(x * x + y * y)
            scale = xp.maximum(mag - thresh, 0.0) / (mag + eps)
            return scale * x, scale * y

    def _compute_edge_map(
        self, u: xp.ndarray, lambda_tv: float
    ) -> xp.ndarray:
        """
        Spatially-varying TV weight based on local edge strength.

        Uses the periodic-BC gradient so BC matches the outer algorithm.

        Parameters
        ----------
        u : xp.ndarray
            Current image estimate.
        lambda_tv : float
            Global TV regularization weight.

        Returns
        -------
        xp.ndarray, shape self.full_shape
            Adaptive weight ∈ [0.2·λ, λ].
            Strong edges → 0.2·λ (less smoothing to preserve sharpness).
            Flat regions → λ   (full smoothing).
        """
        dx, dy = forward_grad_periodic(u)
        edge = xp.sqrt(dx * dx + dy * dy + 1e-8)
        e_min = float(edge.min())
        e_max = float(edge.max())
        if e_max > e_min:
            edge = (edge - e_min) / (e_max - e_min)
        else:
            edge = xp.zeros_like(edge)
        return lambda_tv * (1.0 - 0.8 * edge)

    def _compute_cost(
        self,
        w_h: xp.ndarray,
        w_w: xp.ndarray,
        Hx: xp.ndarray,
        lambda_tv: float,
        tvnorm: int,
    ) -> float:
        """
        Primal cost: (1/2)||M(Hx−y)||²  +  λ·TV(w).

        Parameters
        ----------
        w_h, w_w : xp.ndarray
            Current TV auxiliary variables (w = ∇x at convergence).
        Hx : xp.ndarray
            Precomputed H·x (avoids a redundant FFT pair).
        lambda_tv : float
            TV regularization weight.
        tvnorm : int
            1 for sum|w|, 2 for sum sqrt(w_h²+w_w²).

        Returns
        -------
        float
        """
        mask_f64 = self.mask.astype(xp.float64)
        y_f64 = self.image.astype(xp.float64)
        data_term = 0.5 * float(xp.sum((mask_f64 * (Hx - y_f64)) ** 2))
        if tvnorm == 1:
            tv_term = float(xp.sum(xp.abs(w_h)) + xp.sum(xp.abs(w_w)))
        else:
            tv_term = float(xp.sum(xp.sqrt(w_h * w_h + w_w * w_w + 1e-12)))
        return data_term + lambda_tv * tv_term

    def _check_tval3_convergence(
        self,
        Hx: xp.ndarray,
        v: xp.ndarray,
        v_old: xp.ndarray,
        dx: xp.ndarray,
        dy: xp.ndarray,
        w_h: xp.ndarray,
        w_w: xp.ndarray,
        w_h_old: xp.ndarray,
        w_w_old: xp.ndarray,
        rho_v: float,
        rho_w: float,
        u: xp.ndarray,
        tol: float,
    ) -> tuple[bool, float, float, float]:
        """
        Check primal and dual ADMM residuals.

        Primal residuals measure constraint satisfaction (Hx≈v, ∇x≈w).
        Dual residuals measure the rate of change of consensus variables.

        Returns
        -------
        converged : bool
        rel_change : float
            Scale-normalised max(r_primal, r_dual).
        r_primal : float
            ||Hx−v||² + ||∇x−w||²  (combined Frobenius norm).
        r_dual : float
            ρ_v||v−v_old|| + ρ_w||w−w_old||  (combined dual residual).
        """
        r_v = float(xp.linalg.norm(Hx - v))
        r_wx = float(xp.linalg.norm(dx - w_h))
        r_wy = float(xp.linalg.norm(dy - w_w))
        r_primal = float(xp.sqrt(r_v ** 2 + r_wx ** 2 + r_wy ** 2))

        r_dv = rho_v * float(xp.linalg.norm(v - v_old))
        r_dwx = rho_w * float(xp.linalg.norm(w_h - w_h_old))
        r_dwy = rho_w * float(xp.linalg.norm(w_w - w_w_old))
        r_dual = float(xp.sqrt(r_dv ** 2 + r_dwx ** 2 + r_dwy ** 2))

        scale = float(xp.linalg.norm(u)) + _EPSILON
        rel_change = max(r_primal, r_dual) / scale
        converged = rel_change < tol
        return converged, rel_change, r_primal, r_dual

    # ══════════════════════════════════════════════════════════════════════
    # Main algorithm
    # ══════════════════════════════════════════════════════════════════════

    def deblur(
        self,
        num_iter: int = 300,
        lambda_tv: float = 0.01,
        tol: float = 1e-4,
        min_iter: int = 5,
        check_every: int = 1,
        nonneg: Optional[bool] = None,
        adaptive_tv: Optional[bool] = None,
        burn_in_frac: Optional[float] = None,
        TVnorm: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run the TVAL3 v=Hx split ADMM algorithm.

        Parameters
        ----------
        num_iter : int
            Maximum ADMM iterations.
        lambda_tv : float
            TV regularization weight.
        tol : float
            Convergence threshold for scale-normalised primal/dual residual.
        min_iter : int
            Minimum iterations before early termination is allowed.
        check_every : int
            Check convergence every this many iterations.
        nonneg : bool or None
            Override constructor ``nonneg`` flag.
        adaptive_tv : bool or None
            Override constructor ``adaptive_tv`` flag.
        burn_in_frac : float or None
            Override constructor ``burn_in_frac``.
        TVnorm : {1, 2} or None
            Override constructor ``TVnorm``.
        verbose : bool
            Log per-iteration details at DEBUG level (use_mask=True by default).

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original FOV, on CPU.
        """
        # ── Resolve per-call overrides ─────────────────────────────────────
        _nonneg = self.nonneg if nonneg is None else bool(nonneg)
        _adaptive = self.adaptive_tv if adaptive_tv is None else bool(adaptive_tv)
        _burn = self.burn_in_frac if burn_in_frac is None else float(burn_in_frac)
        _tvnorm = self.TVnorm if TVnorm is None else int(TVnorm)
        burn_in_iters = max(1, int(_burn * num_iter))

        # ── Precision constants ────────────────────────────────────────────
        eps = _EPSILON
        eps_pos = xp.float64(1e-8)

        # ── Adaptive penalty state (rho_v adapts; rho_w is fixed) ─────────
        rho_v: float = self.mu
        rho_w: float = self.mu

        # ── Initialise state in float64 ────────────────────────────────────
        u: xp.ndarray = self.estimated_image.astype(xp.float64).copy()
        mask_f64: xp.ndarray = self.mask.astype(xp.float64)
        y_f64: xp.ndarray = self.image.astype(xp.float64)

        # Initial forward pass and gradient
        Hx_k: xp.ndarray = xp.real(ifft2(self.H_full * fft2(u)))
        dx, dy = forward_grad_periodic(u)

        # Initialise auxiliary and dual variables
        v: xp.ndarray = Hx_k.copy()
        w_h: xp.ndarray = dx.copy()
        w_w: xp.ndarray = dy.copy()
        d_v: xp.ndarray = xp.zeros_like(u)
        d_w_h: xp.ndarray = xp.zeros_like(u)
        d_w_w: xp.ndarray = xp.zeros_like(u)

        # Initial cost (Bug-fix #4: compute once, not twice)
        prev_cost = self._compute_cost(w_h, w_w, Hx_k, lambda_tv, _tvnorm)
        self.costs = [prev_cost]

        logger.debug(
            "TVAL3 deblur: num_iter=%d, lambda_tv=%.3e, TVnorm=%d, "
            "adaptive_tv=%s, burn_in=%d, rho_v=%.2e",
            num_iter, lambda_tv, _tvnorm, _adaptive, burn_in_iters, rho_v,
        )

        # ── Main ADMM loop ─────────────────────────────────────────────────
        for k in range(1, num_iter + 1):

            # Save old consensus variables for convergence check
            v_old = v.copy()
            w_h_old = w_h.copy()
            w_w_old = w_w.copy()

            # ── Step 1: v-update (pointwise, masked data fidelity) ─────────
            # v = (M⊙y + ρ_v(Hx_k + d_v)) / (M + ρ_v)
            # M=1 → (y + ρ_v(Hx+d_v)) / (1 + ρ_v)  (weighted average)
            # M=0 → Hx + d_v                          (no data constraint)
            v = (mask_f64 * y_f64 + rho_v * (Hx_k + d_v)) / (mask_f64 + rho_v)

            # ── Step 2: x-update (exact FFT solve) ─────────────────────────
            # Solve: (ρ_v H^TH + ρ_w G^TG) x = rhs
            # where G^T = −backward_div_periodic  (from adjointness relation
            #   ⟨−Gx, p⟩ = ⟨x, backward_div_periodic(p)⟩)
            # rhs = ρ_v H^T(v−d_v) − ρ_w backward_div_periodic(w−d_w)
            rhs = (
                rho_v * xp.real(ifft2(self.H_conj_full * fft2(v - d_v)))
                - rho_w * backward_div_periodic(w_h - d_w_h, w_w - d_w_w)
            )
            denom = rho_v * self.H_H_conj + rho_w * self.lap_fft
            u = xp.real(ifft2(fft2(rhs) / (denom + eps)))

            # ── Step 3: NaN/Inf check (Bug-fix #5: before cost/projection) ─
            if not bool(xp.isfinite(u).all()):
                logger.warning(
                    "NaN/Inf in u at iteration %d; stopping early.", k
                )
                raise FloatingPointError(
                    f"NaN/Inf encountered at iteration {k}; "
                    "check lambda_tv or initial mu."
                )

            # Positivity projection
            if _nonneg:
                u = xp.maximum(u, eps_pos)

            # ── Step 4: Recompute Hx and gradients for next iteration ──────
            Hx_k = xp.real(ifft2(self.H_full * fft2(u)))
            dx, dy = forward_grad_periodic(u)

            # ── Step 5: w-update (vectorial shrinkage) ─────────────────────
            if _adaptive and k > burn_in_iters:
                threshold = self._compute_edge_map(u, lambda_tv) / rho_w
            else:
                threshold = lambda_tv / rho_w

            w_h, w_w = self._shrink(
                dx + d_w_h, dy + d_w_w, threshold, _EPS_GRAD, _tvnorm
            )

            # ── Step 6: Dual updates (Boyd convention: d += violation) ─────
            d_v += Hx_k - v
            d_w_h += dx - w_h
            d_w_w += dy - w_w

            # ── Step 7: Cost + logging ──────────────────────────────────────
            cost = self._compute_cost(w_h, w_w, Hx_k, lambda_tv, _tvnorm)
            self.costs.append(cost)

            if verbose:
                logger.debug(
                    "iter %d/%d  cost=%.4e  rho_v=%.2e",
                    k, num_iter, cost, rho_v,
                )

            # Cost explosion / stagnation checks
            if not (np.isfinite(cost) and cost < 1e20):
                logger.warning(
                    "Cost explosion at iter %d (cost=%.3e). Stopping.", k, cost
                )
                break

            cost_change = abs(cost - prev_cost) / (abs(prev_cost) + eps)
            prev_cost = cost
            if cost_change < 1e-8 and k > 10:
                logger.info(
                    "Cost stagnation at iter %d (Δcost=%.2e). Stopping.",
                    k, cost_change,
                )
                break

            # ── Step 7b: Primal/dual convergence check ─────────────────────
            if k >= min_iter and k % check_every == 0:
                converged, rel_change, r_primal, r_dual = (
                    self._check_tval3_convergence(
                        Hx_k, v, v_old, dx, dy,
                        w_h, w_w, w_h_old, w_w_old,
                        rho_v, rho_w, u, tol,
                    )
                )
                if verbose:
                    logger.debug(
                        "  primal=%.3e  dual=%.3e  rel=%.3e",
                        r_primal, r_dual, rel_change,
                    )
                if converged:
                    logger.info(
                        "Converged at iter %d/%d "
                        "(primal=%.2e, dual=%.2e, rel=%.2e < tol=%.2e)",
                        k, num_iter, r_primal, r_dual, rel_change, tol,
                    )
                    break

            # ── Step 8: Adaptive ρ_v (Boyd et al. 2011 §3.4.1) ────────────
            r_pv = float(xp.linalg.norm(Hx_k - v))
            r_dv = rho_v * float(xp.linalg.norm(v - v_old))
            ratio = r_pv / (r_dv + eps)
            if ratio > 10.0:
                rho_v = min(rho_v * self.mu_factor, self.mu_max)
            elif ratio < 0.1:
                rho_v = max(rho_v / self.mu_factor, self.mu_min)

        else:
            self._log_no_convergence(num_iter, tol)

        self._last_mu = rho_v
        del d_v, d_w_h, d_w_w, v, w_h, w_w, dx, dy, Hx_k, rhs, denom
        return self._crop_and_return(u.astype(xp.float32))

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def cost_history(self) -> list[float]:
        """Cost values: initial (iter 0) followed by one entry per iteration."""
        return list(self.costs)

    @property
    def last_mu(self) -> float:
        """Final adaptive penalty ρ_v after the most recent deblur() call."""
        return self._last_mu


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def tval3_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 300,
    lambda_tv: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    One-shot TVAL3 deconvolution convenience wrapper.

    Splits ``**kwargs`` between :class:`TVAL3Deconv.__init__` and
    :meth:`TVAL3Deconv.deblur` using :attr:`TVAL3Deconv._INIT_KEYS`.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred) image.
    psf : np.ndarray
        Point spread function.
    iters : int
        Maximum ADMM iterations.
    lambda_tv : float
        TV regularization weight.
    **kwargs
        Forwarded to constructor and/or ``deblur()``.

    Returns
    -------
    np.ndarray
        Deconvolved image.
    """
    init_kw = {k: v for k, v in kwargs.items() if k in TVAL3Deconv._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in TVAL3Deconv._INIT_KEYS}
    return TVAL3Deconv(image, psf, **init_kw).deblur(
        num_iter=iters, lambda_tv=lambda_tv, **deblur_kw
    )
