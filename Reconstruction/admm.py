"""
admm.py — ADMM deconvolution with v=Hx split and overridable prior interface.

Solves the masked deconvolution problem:

    min_x  (1/2) ||M ⊙ (Hx − y)||²  +  λ R(x)

via the variable split v = Hx with augmented Lagrangian (scaled dual variable
convention, Boyd et al. 2011):

    L = (1/2)||M(v−y)||²  +  λ R(x)
        + (ρ_v/2)||Hx − v + d_v||²
        + prior_penalty(x, state)

Three ADMM blocks per iteration
---------------------------------
1. v-update (pointwise closed form, masked data fidelity) — FIXED:
       v ← (M ⊙ y + ρ_v(Hx + d_v)) / (M + ρ_v)

2. Prior update — OVERRIDABLE via _prior_update():
   Default TV:  w ← shrink(∇x_old + d_w, λ/ρ_w)
                returns  prior_rhs = −ρ_w · backward_div_periodic(w − d_w)
   PnP:  denoiser call on x_old, returns denoiser prior contribution.

3. x-update (exact FFT solve, denominator from _x_update_denom()) — FIXED:
       rhs   = ρ_v · Re[ℱ⁻¹(H* · ℱ(v − d_v))]  +  prior_rhs
       denom = _x_update_denom(ρ_v, ρ_w)
       x ← Re[ℱ⁻¹(ℱ(rhs) / (denom + ε))]

4. Dual v-update — FIXED:
       d_v ← d_v + Hx_new − v

5. Prior dual update — OVERRIDABLE via _prior_dual_update():
   Default TV:  d_w ← d_w + ∇x_new − w
   PnP:  no-op.

6. Adaptive ρ_v (Boyd §3.4.1, v-constraint only) — FIXED:
   Increase ρ_v when primal residual dominates; decrease otherwise.

Design for PnP-ADMM extensibility
----------------------------------
The four overridable methods form the entire "prior interface":

  _prior_init(u)                        → state dict
  _prior_update(u, state, λ, ρ_w, ε)   → prior_rhs array
  _prior_dual_update(u, state)          → None
  _x_update_denom(ρ_v, ρ_w)            → denom array

A PnP subclass overrides _prior_update to call a denoiser and returns 0,
overrides _prior_dual_update as a no-op, and overrides _x_update_denom to
return rho_v * H_H_conj (no TV Laplacian term).  The v-update, x-solve, dual
d_v update, convergence check, and adaptive ρ_v remain completely unchanged.

Differences from TVAL3
-----------------------
- No adaptive TV weights / edge map.
- Separate rho_w parameter (fixed, not adapted).
- w-update uses OLD u's gradient (prior_update called BEFORE x-update).
- Convergence check uses v-constraint only (universal across prior types).
- Overridable method interface for PnP-ADMM extensibility.

Precision notes
---------------
Internal computation uses float64 for numerical stability of dual variable
accumulation.  Inputs and outputs remain float32/numpy per package convention.

Boundary model
--------------
ADMM uses masked data fidelity on the original support over the padded FFT
canvas inherited from :class:`~._base.DeconvBase`. Its TV prior interface is
coupled directly to Fourier-diagonalized x-updates, so the default TV path
uses periodic gradient / divergence operators. Subclasses such as PnP and RED
inherit the same masked-fidelity and padded-FFT structure even when they
replace the TV prior itself.

Statefulness
------------
``ADMMDeconv`` is a stateful iterative solver. Repeated object-level
``deblur()`` calls warm-start from the stored ``estimated_image`` iterate and
replace diagnostic state such as ``cost_history`` and the reported final
penalties. The wrapper ``admm_deblur`` is a cold-start helper that creates a
fresh solver object for each call.

References
----------
[1] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein, "Distributed
    Optimization and Statistical Learning via the Alternating Direction
    Method of Multipliers," Foundations and Trends in ML, 3(1):1–122, 2011.
[2] S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg, "Plug-and-Play Priors
    for Model Based Reconstruction," IEEE GlobalSIP, 2013.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import _backend as backend
from ._base import DeconvBase, _split_init_and_deblur_kwargs
from ._tv_operators import backward_div_periodic, forward_grad_periodic, shrink_tv

logger = logging.getLogger(__name__)

# Module-level constants (float64 context)
_EPSILON: float = 1e-8
_EPS_GRAD: float = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# ADMMDeconv
# ══════════════════════════════════════════════════════════════════════════════

class ADMMDeconv(DeconvBase):
    """
    ADMM deconvolution with v=Hx split and overridable prior interface.

    See module docstring for the full mathematical formulation.

    Repeated object-level :meth:`deblur` calls warm-start from the stored
    ``estimated_image`` iterate by default. On finite early-stop / divergence
    paths, the solver stores and returns the last verified finite iterate.
    On explicit numerical failure, it raises ``FloatingPointError`` while
    preserving finite persistent state.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.  2-D grayscale or 3-D RGB.
    psf : np.ndarray
        Point spread function.
    rho_v : float, optional
        Initial augmented Lagrangian penalty for the v=Hx constraint.
        Adapted online by Boyd §3.4.1 rule.  Default 32.0.
    rho_w : float, optional
        Fixed augmented Lagrangian penalty for the prior (TV gradient)
        constraint.  Not adapted.  Default 32.0.
    rho_max : float, optional
        Maximum allowed rho_v.  Default 1024.0.
    rho_min : float, optional
        Minimum allowed rho_v.  Default 0.03125 (= 2⁻⁵).
    rho_factor : float, optional
        Multiplicative step for adaptive rho_v.  Default 1.2.
    TVnorm : {1, 2}, optional
        TV norm variant.  1 = anisotropic, 2 = isotropic.  Default 2.
    nonneg : bool, optional
        Enforce x ≥ ε_pos after each x-update.  Default True.
    **kwargs
        Passed to :class:`~._base.DeconvBase` (paddingMode, padding_scale,
        initialEstimate, apply_taper_on_padding_band, htm_floor_frac,
        use_mask).

    Notes
    -----
    The four overridable methods (_prior_init, _prior_update,
    _prior_dual_update, _x_update_denom) allow subclasses to swap the default
    TV prior for a PnP denoiser without touching the ADMM scaffolding.

    Effective boundary model for the default TV prior:

    - padded FFT canvas from :class:`DeconvBase`
    - masked data fidelity on the observed support Ω
    - circular convolution on the full padded canvas
    - periodic gradient / divergence operators in the TV split
    """

    _INIT_KEYS: frozenset[str] = DeconvBase._INIT_KEYS | frozenset({
        "rho_v", "rho_w", "rho_max", "rho_min", "rho_factor",
        "TVnorm", "nonneg",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        rho_v: float = 32.0,
        rho_w: float = 32.0,
        rho_max: float = 1024.0,
        rho_min: float = 0.03125,
        rho_factor: float = 1.2,
        TVnorm: int = 2,
        nonneg: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(image, psf, **kwargs)

        if rho_v <= 0.0:
            raise ValueError(f"rho_v must be positive, got {rho_v!r}")
        if rho_w <= 0.0:
            raise ValueError(f"rho_w must be positive, got {rho_w!r}")
        if rho_max <= 0.0:
            raise ValueError(f"rho_max must be positive, got {rho_max!r}")
        if rho_min <= 0.0:
            raise ValueError(f"rho_min must be positive, got {rho_min!r}")
        if rho_min > rho_max:
            raise ValueError(
                f"rho_min must be <= rho_max, got rho_min={rho_min!r}, "
                f"rho_max={rho_max!r}"
            )
        if rho_factor <= 0.0:
            raise ValueError(f"rho_factor must be positive, got {rho_factor!r}")
        if TVnorm not in (1, 2):
            raise ValueError(
                f"TVnorm must be 1 or 2, got {TVnorm!r}. "
                "Use 2 for isotropic TV or 1 for anisotropic TV."
            )

        # ── Full complex PSF spectrum in float64 ───────────────────────────
        # Recover the spatial PSF from the base-class rfft2 precomputation,
        # cast to float64, then compute full M×N complex FFT.
        psf_spatial: "backend.xp.ndarray" = backend.irfft2(
            self.PF, s=self.full_shape
        ).astype(backend.xp.float64)

        H_full = backend.fft2(psf_spatial)
        self.H_full: "backend.xp.ndarray" = backend._freeze(H_full)
        # F8: ``H_full.conj()`` on a complex array already allocates a fresh
        # buffer (verified: ``a.conj() is a`` is False and ``owndata`` is
        # True), so the trailing ``.copy()`` here was redundant.  The
        # ``.copy()`` below is kept: ``xp.real(complex_array)`` is a view
        # with ``owndata=False``, and copying it releases the full complex
        # intermediate instead of holding it alive as ``.base``.
        self.H_conj_full: "backend.xp.ndarray" = backend._freeze(H_full.conj())
        self.H_H_conj: "backend.xp.ndarray" = backend._freeze(
            backend.xp.real(self.H_conj_full * self.H_full).copy()
        )

        # ── F6: cache float64 mask/image views for the data-fidelity term ──
        # self.mask / self.image are frozen after DeconvBase.__init__, so
        # casting them once here avoids one full-canvas float64 allocation
        # per deblur() call (in the setup block below) plus two more per
        # iteration inside _compute_admm_cost.
        self._mask_f64: "backend.xp.ndarray" = backend._freeze(
            self.mask.astype(backend.xp.float64)
        )
        self._image_f64: "backend.xp.ndarray" = backend._freeze(
            self.image.astype(backend.xp.float64)
        )

        # ── Laplacian eigenvalue tensor (periodic BC) ──────────────────────
        # lap_fft[k,l] = 4 − 2cos(2πk/M) − 2cos(2πl/N)
        # Eigenvalues of G^TG for forward-difference gradient G under
        # periodic BC.  Required for the exact FFT x-solve.
        M_f, N_f = self.full_shape
        fy = backend.fftfreq(M_f).reshape(-1, 1).astype(backend.xp.float64)
        fx = backend.fftfreq(N_f).reshape(1, -1).astype(backend.xp.float64)
        self.lap_fft: "backend.xp.ndarray" = backend._freeze(
            (4.0 - 2.0 * backend.xp.cos(2.0 * backend.xp.pi * fy)
             - 2.0 * backend.xp.cos(2.0 * backend.xp.pi * fx)).copy()
        )

        logger.debug(
            "ADMM precomputed H_full (%s), lap_fft (%s) on canvas %s.",
            H_full.dtype, self.lap_fft.dtype, self.full_shape,
        )

        # ── Algorithm hyperparameters ──────────────────────────────────────
        self.rho_v: float = float(rho_v)
        self.rho_w: float = float(rho_w)
        self.rho_max: float = float(rho_max)
        self.rho_min: float = float(rho_min)
        self.rho_factor: float = float(rho_factor)
        self.TVnorm: int = int(TVnorm)
        self.nonneg: bool = bool(nonneg)

        # ── State (populated by deblur) ────────────────────────────────────
        self.costs: list[float] = []
        self._last_rho_v: float = self.rho_v
        self._last_rho_w: float = self.rho_w

    # ══════════════════════════════════════════════════════════════════════
    # Overridable prior interface
    # ══════════════════════════════════════════════════════════════════════

    def _prior_init(self, u: xp.ndarray) -> dict:
        """
        Initialize prior state variables.

        Called once before the ADMM loop.  Must return a state dict that
        will be passed to _prior_update and _prior_dual_update each iteration.

        Default (TV prior): initialise w = ∇x, d_w = 0.

        Parameters
        ----------
        u : xp.ndarray
            Initial image estimate on the canvas, float64.

        Returns
        -------
        dict
            State dict with keys 'w_h', 'w_w', 'd_w_h', 'd_w_w'.
        """
        dx, dy = forward_grad_periodic(u)
        return {
            "w_h": dx.copy(),
            "w_w": dy.copy(),
            "d_w_h": backend.xp.zeros_like(u),
            "d_w_w": backend.xp.zeros_like(u),
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
        Compute the prior's contribution to the x-update RHS and update w.

        Called BEFORE the x-update, using the OLD iterate u.  Updates the
        state dict in place.

        Default (TV prior): apply vectorial shrinkage on (∇x + d_w), then
        return  −ρ_w · backward_div_periodic(w_new − d_w_old).

        The sign convention follows from the adjointness relation:
            ⟨−Gx, p⟩ = ⟨x, backward_div_periodic(p)⟩  →  G^T = −backward_div_periodic.

        Parameters
        ----------
        u : xp.ndarray
            Current (OLD) image estimate, float64.
        state : dict
            Mutable state dict from _prior_init (contains w_h, w_w, d_w_h, d_w_w).
        lambda_tv : float
            TV regularization weight.
        rho_w : float
            Fixed prior penalty parameter.
        eps : float
            Denominator floor for isotropic shrinkage.

        Returns
        -------
        xp.ndarray
            prior_rhs: the prior term to ADD to the x-update RHS.
        """
        dx, dy = forward_grad_periodic(u)
        w_h_new, w_w_new = self._shrink(
            dx + state["d_w_h"],
            dy + state["d_w_w"],
            lambda_tv / rho_w,
            eps,
            self.TVnorm,
        )
        state["w_h"] = w_h_new
        state["w_w"] = w_w_new

        # G^T = −backward_div_periodic  →  prior_rhs = −ρ_w G^T(w − d_w)
        return -rho_w * backward_div_periodic(
            state["w_h"] - state["d_w_h"],
            state["w_w"] - state["d_w_w"],
        )

    def _prior_dual_update(self, u: xp.ndarray, state: dict) -> None:
        """
        Update prior dual variables using the NEW iterate u.

        Called AFTER the x-update.  Updates the state dict in place.

        Default (TV prior): d_w ← d_w + ∇x_new − w.

        Parameters
        ----------
        u : xp.ndarray
            New image estimate (after x-update + positivity), float64.
        state : dict
            Mutable state dict (contains w_h, w_w, d_w_h, d_w_w).
        """
        dx, dy = forward_grad_periodic(u)
        state["d_w_h"] += dx - state["w_h"]
        state["d_w_w"] += dy - state["w_w"]

    def _x_update_denom(self, rho_v: float, rho_w: float) -> xp.ndarray:
        """
        Denominator array for the FFT x-update solve.

        denom[k,l] = ρ_v |H(k,l)|² + ρ_w · lap_fft[k,l]

        Override in PnP subclasses to return  ρ_v · H_H_conj  (no TV term).

        Parameters
        ----------
        rho_v : float
            Current (adaptive) data penalty.
        rho_w : float
            Fixed prior penalty.

        Returns
        -------
        xp.ndarray, shape self.full_shape, dtype float64
        """
        return rho_v * self.H_H_conj + rho_w * self.lap_fft

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers (non-overridable)
    # ══════════════════════════════════════════════════════════════════════

    def _shrink(
        self,
        x: xp.ndarray,
        y: xp.ndarray,
        thresh,
        eps: float,
        tvnorm: int,
    ) -> tuple[xp.ndarray, xp.ndarray]:
        """
        Shrinkage / vectorial soft-thresholding.

        F14: the body lives in ``_tv_operators.shrink_tv`` (shared with
        TVAL3).  This thin wrapper is retained so the numerical-failure
        contract tests can still monkey-patch ``solver._shrink`` as a
        fault-injection seam.
        """
        return shrink_tv(x, y, thresh, eps, tvnorm)

    def _compute_admm_cost(
        self,
        Hx: xp.ndarray,
        lambda_tv: float,
        state: dict,
        tvnorm: int,
    ) -> float:
        """
        Primal cost: (1/2)||M(Hx−y)||²  +  λ·TV(w).

        Uses w from state if present; gracefully skips the TV term when
        the state dict has no 'w_h'/'w_w' keys (PnP subclasses).

        Parameters
        ----------
        Hx : xp.ndarray
            Precomputed H·x (avoids a redundant FFT pair).
        lambda_tv : float
            TV regularization weight.
        state : dict
            Prior state dict; may contain 'w_h', 'w_w'.
        tvnorm : int
            1 for sum|w|, 2 for sum sqrt(w_h²+w_w²).

        Returns
        -------
        float
        """
        data_term = 0.5 * float(
            backend.xp.sum((self._mask_f64 * (Hx - self._image_f64)) ** 2)
        )

        w_h = state.get("w_h")
        w_w = state.get("w_w")
        if w_h is not None and w_w is not None:
            if tvnorm == 1:
                tv_term = float(
                    backend.xp.sum(backend.xp.abs(w_h))
                    + backend.xp.sum(backend.xp.abs(w_w))
                )
            else:
                tv_term = float(
                    backend.xp.sum(backend.xp.sqrt(w_h * w_h + w_w * w_w + 1e-12))
                )
            return data_term + lambda_tv * tv_term
        return data_term

    def _check_prior_convergence(
        self,
        u: xp.ndarray,
        state: dict,
        rho_w: float,
        tol: float,
    ) -> tuple[bool, float, dict]:
        """
        Optional prior-specific convergence hook.

        Default implementation: no extra prior residuals.
        Subclasses (e.g. PnP) can override.

        Returns
        -------
        prior_ok : bool
        prior_rel : float
        prior_stats : dict
        """
        return True, 0.0, {}

    def _check_admm_convergence(
        self,
        Hx: xp.ndarray,
        v: xp.ndarray,
        v_old: xp.ndarray,
        rho_v: float,
        u: xp.ndarray,
        tol: float,
    ) -> tuple[bool, float, float, float]:
        """
        Check v-constraint primal and dual residuals.

        Uses only the v=Hx constraint residuals (universal across prior types).

        Parameters
        ----------
        Hx : xp.ndarray
            Current H·x.
        v : xp.ndarray
            Current v after v-update.
        v_old : xp.ndarray
            v from previous iteration.
        rho_v : float
            Current adaptive data penalty.
        u : xp.ndarray
            Current image estimate (for scale normalisation).
        tol : float
            Convergence threshold.

        Returns
        -------
        converged : bool
        rel_change : float
        r_primal : float
            ||Hx−v||  (v-constraint violation).
        r_dual : float
            ρ_v · ||v−v_old||  (v dual residual).
        """
        r_v = float(backend.xp.linalg.norm(Hx - v))
        r_dv = rho_v * float(backend.xp.linalg.norm(v - v_old))
        scale = float(backend.xp.linalg.norm(u)) + _EPSILON
        rel_change = max(r_v, r_dv) / scale
        converged = rel_change < tol
        return converged, rel_change, r_v, r_dv

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
        TVnorm: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run the ADMM v=Hx split algorithm.

        Parameters
        ----------
        num_iter : int
            Maximum ADMM iterations.
        lambda_tv : float
            TV regularization weight (passed to _prior_update).
        tol : float
            Convergence threshold for scale-normalised v-constraint residual.
        min_iter : int
            Minimum iterations before early termination is allowed.
        check_every : int
            Check convergence every this many iterations.
        nonneg : bool or None
            Override constructor ``nonneg`` flag.
        TVnorm : {1, 2} or None
            Override constructor ``TVnorm``.
        verbose : bool
            Log per-iteration details at DEBUG level.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image cropped to the original FOV, on CPU.
        """
        if num_iter < 1:
            raise ValueError(f"num_iter must be >= 1, got {num_iter!r}")
        if lambda_tv < 0.0:
            raise ValueError(f"lambda_tv must be >= 0, got {lambda_tv!r}")
        if tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {tol!r}")
        if min_iter < 0:
            raise ValueError(f"min_iter must be >= 0, got {min_iter!r}")
        if check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {check_every!r}")

        # ── Resolve per-call overrides ─────────────────────────────────────
        _nonneg = self.nonneg if nonneg is None else bool(nonneg)
        _tvnorm = self.TVnorm if TVnorm is None else int(TVnorm)
        if _tvnorm not in (1, 2):
            raise ValueError(
                f"TVnorm must be 1 or 2, got {_tvnorm!r}. "
                "Use 2 for isotropic TV or 1 for anisotropic TV."
            )

        # ── Precision constants ────────────────────────────────────────────
        eps = _EPSILON
        eps_pos = backend.xp.float64(1e-8)

        # ── Adaptive penalty state (rho_v adapts; rho_w is fixed) ─────────
        rho_v: float = self.rho_v
        rho_w: float = self.rho_w

        # ── Initialise state in float64 ────────────────────────────────────
        u: "backend.xp.ndarray" = self.estimated_image.astype(backend.xp.float64).copy()
        # F6: mask/image float64 casts are cached on self in __init__.
        mask_f64 = self._mask_f64
        y_f64 = self._image_f64

        # Initial forward pass
        Hx_k: "backend.xp.ndarray" = backend.xp.real(
            backend.ifft2(self.H_full * backend.fft2(u))
        )
        last_finite: "backend.xp.ndarray" = u.copy()

        # Initialise v and d_v
        v: "backend.xp.ndarray" = Hx_k.copy()
        d_v: "backend.xp.ndarray" = backend.xp.zeros_like(u)

        # Initialise prior state via overridable method
        state: dict = self._prior_init(u)

        # Initial cost
        prev_cost = self._compute_admm_cost(Hx_k, lambda_tv, state, _tvnorm)
        self.costs = [prev_cost]

        logger.debug(
            "ADMM deblur: num_iter=%d, lambda_tv=%.3e, TVnorm=%d, "
            "rho_v=%.2e, rho_w=%.2e",
            num_iter, lambda_tv, _tvnorm, rho_v, rho_w,
        )

        # ── Main ADMM loop ─────────────────────────────────────────────────
        for k in range(1, num_iter + 1):

            # Save old v for convergence / adaptive rho_v
            v_old = v.copy()

            # ── Step 1: v-update (pointwise, masked data fidelity) ─────────
            # v = (M⊙y + ρ_v(Hx_k + d_v)) / (M + ρ_v)
            v = (mask_f64 * y_f64 + rho_v * (Hx_k + d_v)) / (mask_f64 + rho_v)
            self._fail_on_nonfinite(
                v,
                name="ADMM v-update state",
                iteration=k,
                last_finite=last_finite,
            )

            # ── Step 2: Prior update (uses OLD u) ──────────────────────────
            # Overridable: updates state (w, etc.) and returns prior_rhs.
            prior_rhs: "backend.xp.ndarray" = self._prior_update(
                u, state, lambda_tv, rho_w, _EPS_GRAD
            )
            # F4: the prior_rhs check below is a strict superset of the
            # state-dict sweep that used to sit here — prior_rhs is built
            # from state entries by local/linear operations, so NaN anywhere
            # in state propagates to prior_rhs.  Drop the per-key sweep (4
            # full-canvas reductions per iteration on the default TV path).
            self._fail_on_nonfinite(
                prior_rhs,
                name="ADMM prior RHS",
                iteration=k,
                last_finite=last_finite,
            )

            # ── Step 3: x-update (exact FFT solve) ─────────────────────────
            # rhs = ρ_v H^T(v − d_v) + prior_rhs
            rhs = (
                rho_v * backend.xp.real(
                    backend.ifft2(self.H_conj_full * backend.fft2(v - d_v))
                )
                + prior_rhs
            )
            denom = self._x_update_denom(rho_v, rho_w)
            self._fail_on_nonfinite(
                rhs,
                name="ADMM x-update RHS",
                iteration=k,
                last_finite=last_finite,
            )
            self._fail_on_nonfinite(
                denom,
                name="ADMM x-update denominator",
                iteration=k,
                last_finite=last_finite,
            )
            # F5: keep the x-update spectrum U around so the forward
            # projection below can reuse it when the positivity projection
            # is off (u is a pure real-valued ifft2 of U, so fft2(u) == U).
            U = backend.fft2(rhs) / (denom + eps)
            u = backend.xp.real(backend.ifft2(U))

            # ── Step 4: NaN/Inf guard ──────────────────────────────────────
            self._fail_on_nonfinite(
                u,
                name="ADMM primal iterate",
                iteration=k,
                last_finite=last_finite,
            )

            # Positivity projection (mutates u, invalidating the cached U).
            if _nonneg:
                u = backend.xp.maximum(u, eps_pos)
                self._fail_on_nonfinite(
                    u,
                    name="ADMM primal iterate after positivity projection",
                    iteration=k,
                    last_finite=last_finite,
                )

            # ── Step 5: Recompute Hx with new u ───────────────────────────
            if _nonneg:
                Hx_k = backend.xp.real(
                    backend.ifft2(self.H_full * backend.fft2(u))
                )
            else:
                # F5: reuse the cached spectrum — one fft2 saved per iter.
                Hx_k = backend.xp.real(backend.ifft2(self.H_full * U))
            self._fail_on_nonfinite(
                Hx_k,
                name="ADMM forward projection",
                iteration=k,
                last_finite=last_finite,
            )

            # ── Step 6: Dual v-update ──────────────────────────────────────
            d_v += Hx_k - v
            self._fail_on_nonfinite(
                d_v,
                name="ADMM data dual state",
                iteration=k,
                last_finite=last_finite,
            )

            # ── Step 7: Prior dual update (uses NEW u) ─────────────────────
            self._prior_dual_update(u, state)
            for key, value in state.items():
                self._fail_on_nonfinite(
                    backend.xp.asarray(value),
                    name=f"ADMM prior state ({key})",
                    iteration=k,
                    last_finite=last_finite,
                )
            last_finite = u.copy()

            # ── Step 8: Cost + logging ──────────────────────────────────────
            cost = self._compute_admm_cost(Hx_k, lambda_tv, state, _tvnorm)
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

            # ── Step 9: Primal/dual convergence check ────────────────────────────
            if k >= min_iter and k % check_every == 0:
                converged_v, rel_v, r_primal_v, r_dual_v = self._check_admm_convergence(
                    Hx_k, v, v_old, rho_v, u, tol,
                )

                prior_ok, prior_rel, prior_stats = self._check_prior_convergence(
                    u, state, rho_w, tol,
                )

                rel_change = max(rel_v, prior_rel)
                converged = converged_v and prior_ok

                if verbose:
                    msg = (
                        f"  v_primal={r_primal_v:.3e}  "
                        f"v_dual={r_dual_v:.3e}  "
                        f"rel_v={rel_v:.3e}"
                    )
                    if prior_stats:
                        for key, val in prior_stats.items():
                            msg += f"  {key}={val:.3e}"
                        msg += f"  rel_prior={prior_rel:.3e}"
                    logger.debug(msg)

                if converged:
                    logger.info(
                        "Converged at iter %d/%d (rel=%.2e < tol=%.2e)",
                        k, num_iter, rel_change, tol,
                    )
                    break

            # ── Step 10: Adaptive ρ_v (Boyd §3.4.1) ─────────────────────────────
            r_pv = float(backend.xp.linalg.norm(Hx_k - v))
            r_dv = rho_v * float(backend.xp.linalg.norm(v - v_old))
            ratio = r_pv / (r_dv + eps)

            rho_v_old = rho_v
            rho_v_new = rho_v

            if ratio > 10.0:
                rho_v_new = min(rho_v * self.rho_factor, self.rho_max)
            elif ratio < 0.1:
                rho_v_new = max(rho_v / self.rho_factor, self.rho_min)

            if rho_v_new != rho_v_old:
                # scaled-dual consistency: u_scaled = y / rho
                d_v *= (rho_v_old / rho_v_new)
                rho_v = rho_v_new

        else:
            self._log_no_convergence(num_iter, tol)

        self._last_rho_v = rho_v
        self._last_rho_w = rho_w
        del d_v, v, Hx_k, rhs, denom, prior_rhs
        return self._crop_and_return(
            u.astype(backend.xp.float32),
            last_finite=last_finite.astype(backend.xp.float32),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def cost_history(self) -> list[float]:
        """Cost values: initial (iter 0) followed by one entry per iteration."""
        return list(self.costs)

    @property
    def last_rho_v(self) -> float:
        """Final adaptive data penalty ρ_v after the most recent deblur()."""
        return self._last_rho_v

    @property
    def last_rho_w(self) -> float:
        """Fixed prior penalty ρ_w (unchanged from constructor)."""
        return self._last_rho_w


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def admm_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 300,
    lambda_tv: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    One-shot ADMM deconvolution convenience wrapper.

    Splits ``**kwargs`` between :class:`ADMMDeconv.__init__` and
    :meth:`ADMMDeconv.deblur` using :attr:`ADMMDeconv._INIT_KEYS`.

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
    init_kw, deblur_kw = _split_init_and_deblur_kwargs(
        ADMMDeconv._INIT_KEYS, kwargs
    )
    return ADMMDeconv(image, psf, **init_kw).deblur(
        num_iter=iters, lambda_tv=lambda_tv, **deblur_kw
    )
