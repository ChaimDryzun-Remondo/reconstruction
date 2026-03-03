"""
_tv_operators.py — Total Variation operators for the Reconstruction package.

Single source of truth for all TV-related operations shared across
algorithm modules:

  - :func:`forward_grad` / :func:`backward_div` — discrete gradient and
    divergence forming the adjoint pair under **Neumann BC** (zero-flux).
  - :func:`forward_grad_periodic` / :func:`backward_div_periodic` — the
    same adjoint pair under **periodic BC** (wrap-around), required by
    algorithms that diagonalize ∇^T∇ in the Fourier domain (ADMM x-update,
    TVAL3 x-update).
  - :func:`tv_multiplicative_correction` — Dey et al. correction factor
    used by the RL-TV update.
  - :func:`prox_tv_chambolle` — proximal operator of γ·TV(·) via the
    Chambolle 2004 dual projection algorithm.

**Choosing between Neumann and periodic BC:**

* Use **Neumann** (``forward_grad`` / ``backward_div``) when the TV
  proximal operator is solved as a standalone ROF subproblem (Chambolle
  dual projection) or when using the Dey et al. multiplicative correction.
  The Chambolle solver's self-contained BC need not match the outer
  algorithm's FFT structure.

* Use **periodic** (``forward_grad_periodic`` / ``backward_div_periodic``)
  whenever the gradient/divergence pair appears inside a Fourier-domain
  linear solve (e.g. ADMM or TVAL3 x-updates).  The DFT eigenvalues of
  −∇^T∇ are ``4 − 2cos(2πf_y) − 2cos(2πf_x)``, which are derived under
  periodic BC.  Using Neumann BC in that context introduces a model
  mismatch that silently degrades convergence.

All array operations use ``xp`` imported from ``._backend``.  Never
import numpy or cupy directly in this module.
"""
from __future__ import annotations

import logging

from ._backend import xp

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Discrete Gradient & Divergence  (Neumann BC, adjoint pair)
# ══════════════════════════════════════════════════════════════════════════════
# These operators form an adjoint pair under Neumann (zero-flux) boundary
# conditions:  ⟨−∇x, p⟩ = ⟨x, div(p)⟩  for all x, p satisfying the BCs.
#
# This property is essential for the Chambolle TV proximal solver (which
# relies on adjointness for convergence guarantees) and for any primal-dual
# algorithm applied to the TV-regularised problem.
#
# Convention:
#   ∇  = forward differences,  last row/col = 0    (Neumann)
#   div = backward differences, adjoint of −∇

def forward_grad(x: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Discrete gradient with forward differences and Neumann BC.

    Computes the two-component discrete gradient of a 2-D array using
    forward (causal) finite differences:

        (∂x/∂h)[i, j] = x[i+1, j] − x[i, j]   for i = 0, …, H−2
        (∂x/∂h)[H−1, j] = 0                      (Neumann: no flux at boundary)

        (∂x/∂w)[i, j] = x[i, j+1] − x[i, j]   for j = 0, …, W−2
        (∂x/∂w)[i, W−1] = 0

    The zero boundary condition (Neumann) ensures that
    :func:`backward_div` is the adjoint of −∇ under the standard
    Euclidean inner product:  ⟨−∇x, p⟩ = ⟨x, div(p)⟩.

    Parameters
    ----------
    x : xp.ndarray, shape (H, W)
        Input 2-D array (image or auxiliary variable).

    Returns
    -------
    dh, dw : xp.ndarray, each shape (H, W)
        Vertical (row) and horizontal (column) partial derivatives.
        Last row of ``dh`` and last column of ``dw`` are zero (Neumann BC).
    """
    dh = xp.zeros_like(x)
    dw = xp.zeros_like(x)
    dh[:-1, :] = x[1:, :] - x[:-1, :]    # ∂x/∂h; last row = 0
    dw[:, :-1] = x[:, 1:] - x[:, :-1]    # ∂x/∂w; last col = 0
    return dh, dw


def backward_div(p_h: xp.ndarray, p_w: xp.ndarray) -> xp.ndarray:
    """
    Discrete divergence with backward differences (adjoint of −∇).

    Computes ``div(p) = bwd_h(p_h) + bwd_w(p_w)`` where the backward
    differences are defined as:

        bwd_h(p)[0, :]     =  p_h[0, :]                    (top boundary)
        bwd_h(p)[i, :]     =  p_h[i, :] − p_h[i−1, :]     (interior)
        bwd_h(p)[H−1, :]   = −p_h[H−2, :]                  (bottom: forward
                                                              diff set p[H−1]=0)

    and analogously for ``bwd_w`` along columns.

    This is the adjoint of the negative forward-difference gradient:
    ⟨−∇x, p⟩ = ⟨x, div(p)⟩ for all x, p satisfying the Neumann boundary
    conditions used by :func:`forward_grad`.

    Parameters
    ----------
    p_h, p_w : xp.ndarray, each shape (H, W)
        Vertical and horizontal components of a 2-D vector field.

    Returns
    -------
    div : xp.ndarray, shape (H, W)
        ``div(p) = bwd_h(p_h) + bwd_w(p_w)``.
    """
    div = xp.empty_like(p_h)

    # ── Vertical component: backward difference of p_h ────────────────────
    div[0, :]    = p_h[0, :]
    div[1:-1, :] = p_h[1:-1, :] - p_h[:-2, :]
    div[-1, :]   = -p_h[-2, :]

    # ── Horizontal component: backward difference of p_w (accumulated) ────
    div[:, 0]    += p_w[:, 0]
    div[:, 1:-1] += p_w[:, 1:-1] - p_w[:, :-2]
    div[:, -1]   += -p_w[:, -2]

    return div


# ══════════════════════════════════════════════════════════════════════════════
# Discrete Gradient & Divergence  (Periodic BC, adjoint pair)
# ══════════════════════════════════════════════════════════════════════════════
# These operators form an adjoint pair under periodic (wrap-around) boundary
# conditions:  ⟨−∇_per x, p⟩ = ⟨x, div_per(p)⟩  for all x, p.
#
# **When to use periodic BC vs. Neumann BC:**
#   Use this pair whenever the gradient appears inside a Fourier-domain linear
#   solve.  The DFT eigenvalues of −∇^T∇ are
#       D_lap[k, l] = 4 − 2cos(2πk/H) − 2cos(2πl/W)
#   which are derived under periodic BC.  Using Neumann BC in that context
#   introduces a silent model mismatch.
#
# Convention:
#   ∇_per  = forward differences with wrap-around at the last row/col
#   div_per = backward differences with wrap-around; adjoint of −∇_per

def forward_grad_periodic(x: "xp.ndarray") -> "tuple[xp.ndarray, xp.ndarray]":
    """
    Discrete gradient with forward differences and periodic (wrap-around) BC.

    Computes the two-component discrete gradient of a 2-D array using
    forward finite differences.  Unlike :func:`forward_grad` (Neumann BC),
    the last row and column wrap around to the first:

        (∂x/∂h)[i, j] = x[i+1, j] − x[i, j]   for i = 0, …, H−2
        (∂x/∂h)[H−1, j] = x[0, j] − x[H−1, j]  (periodic wrap)

        (∂x/∂w)[i, j] = x[i, j+1] − x[i, j]   for j = 0, …, W−2
        (∂x/∂w)[i, W−1] = x[i, 0] − x[i, W−1]  (periodic wrap)

    The periodic BC ensures that :func:`backward_div_periodic` is the
    adjoint of −∇_per under the standard Euclidean inner product:

        ⟨−∇_per x, p⟩ = ⟨x, div_per(p)⟩

    This adjoint property is required for algorithms that diagonalize the
    operator −∇^T∇ in the Fourier domain (ADMM and TVAL3 x-updates).

    Parameters
    ----------
    x : xp.ndarray, shape (H, W)
        Input 2-D array.

    Returns
    -------
    dh, dw : xp.ndarray, each shape (H, W)
        Vertical (row) and horizontal (column) partial derivatives with
        periodic wrap-around at the last row/column.

    See Also
    --------
    forward_grad : Neumann BC variant (zero at boundaries).
    backward_div_periodic : Adjoint of −∇_per.
    """
    dh = xp.empty_like(x)
    dw = xp.empty_like(x)

    # Vertical forward differences with periodic wrap
    dh[:-1, :] = x[1:, :] - x[:-1, :]   # interior rows
    dh[-1, :]  = x[0, :]  - x[-1, :]    # last row wraps to first

    # Horizontal forward differences with periodic wrap
    dw[:, :-1] = x[:, 1:] - x[:, :-1]  # interior columns
    dw[:, -1]  = x[:, 0]  - x[:, -1]   # last col wraps to first

    return dh, dw


def backward_div_periodic(
    p_h: "xp.ndarray",
    p_w: "xp.ndarray",
) -> "xp.ndarray":
    """
    Discrete divergence with periodic BC (adjoint of −∇_per).

    Computes ``div_per(p) = bwd_h_per(p_h) + bwd_w_per(p_w)`` where the
    backward differences wrap around at the boundaries:

        bwd_h_per(p)[i, j] = p_h[i, j] − p_h[i−1, j]   (interior: i ≥ 1)
        bwd_h_per(p)[0, j] = p_h[0, j] − p_h[H−1, j]   (periodic: p[−1] = p[H−1])

    and analogously for ``bwd_w_per`` along columns.

    Equivalently, using ``xp.roll``:

        div_h = p_h − roll(p_h, shift=+1, axis=0)
        div_w = p_w − roll(p_w, shift=+1, axis=1)
        return div_h + div_w

    This is the adjoint of the negative periodic forward-difference gradient:

        ⟨−∇_per x, p⟩ = ⟨x, div_per(p)⟩

    for all arrays x, p (no boundary conditions required — periodicity
    is built into both operators).

    Parameters
    ----------
    p_h, p_w : xp.ndarray, each shape (H, W)
        Vertical and horizontal components of a 2-D vector field.

    Returns
    -------
    div : xp.ndarray, shape (H, W)
        ``div_per(p) = bwd_h_per(p_h) + bwd_w_per(p_w)``.

    Notes
    -----
    The roll formulation is used rather than explicit index assignment
    to avoid off-by-one errors at the boundaries and to be compatible
    with both NumPy and CuPy backends.

    See Also
    --------
    backward_div : Neumann BC variant.
    forward_grad_periodic : Forward operator (adjoint of −div_per).
    """
    # Backward difference = p − shifted(p), where shift=+1 pulls p[i-1]
    # to position i (equivalently, shift by +1 ≡ p[i] - p[i-1] with wrap).
    div_h = p_h - xp.roll(p_h, shift=1, axis=0)
    div_w = p_w - xp.roll(p_w, shift=1, axis=1)
    return div_h + div_w


# ══════════════════════════════════════════════════════════════════════════════
# Dey et al. Multiplicative TV Correction
# ══════════════════════════════════════════════════════════════════════════════

def tv_multiplicative_correction(
    x: xp.ndarray,
    lambda_tv: float,
    eps_grad: float = 1e-8,
) -> xp.ndarray:
    """
    Compute the Dey et al. multiplicative TV correction factor.

    For a current estimate *x*, this returns the denominator of Eq. (2):

        C(x) = 1  −  λ · div( ∇x / |∇x|_ε )

    so that the regularized RL update becomes:

        x_{k+1} = ( x_k · Hᵀ(y / Hx_k) ) / C(x_k)

    The divergence of the normalized gradient is sometimes called the *mean
    curvature* of the level sets of *x*.  It is large and positive at concave
    edges (bright→dark), large and negative at convex edges (dark→bright), and
    near zero in flat regions.  Dividing the RL update by ``C(x)`` therefore
    selectively attenuates the update at edges — the hallmark of TV
    regularization — while leaving flat regions untouched.

    Discrete Operators
    ------------------
    Uses :func:`forward_grad` (forward differences, Neumann BC) for the
    gradient and :func:`backward_div` (backward differences, adjoint of −∇)
    for the divergence.

    Gradient (forward differences, Neumann BC):
        (∂x/∂h)[i,j] = x[i+1,j] - x[i,j]   for i = 0, …, H-2
        (∂x/∂h)[H-1,j] = 0                    (Neumann: no flux at boundary)

        (∂x/∂w)[i,j] = x[i,j+1] - x[i,j]   for j = 0, …, W-2
        (∂x/∂w)[i,W-1] = 0

    Divergence (backward differences, adjoint of -∇):
        div(p_h, p_w)[i,j] = bwd_h(p_h)[i,j] + bwd_w(p_w)[i,j]

        where bwd_h(p)[i,j] = p[i,j] - p[i-1,j]   for i = 1, …, H-2
              bwd_h(p)[0,j] = p[0,j]                 (boundary: p[-1,j] = 0)
              bwd_h(p)[H-1,j] = -p[H-2,j]            (boundary: p[H-1,j] = 0
                                                       from forward diff)

    This adjoint pairing ensures that ⟨-∇x, p⟩ = ⟨x, div(p)⟩ for all x, p
    satisfying the boundary conditions — a necessary condition for the
    variational derivation of Eq. (2) to hold.

    Parameters
    ----------
    x : xp.ndarray, shape (H, W), float32
        Current image estimate.  Must be non-negative.
    lambda_tv : float
        TV regularization strength.  Typical range for [0, 1]-normalized
        satellite imagery: 1e-4 to 1e-2.
    eps_grad : float, optional
        Smoothing constant for the gradient magnitude to avoid division by
        zero in flat regions.  Default 1e-8.

    Returns
    -------
    xp.ndarray, shape (H, W), float32
        The correction factor ``C(x) = 1 - λ · div(∇x / |∇x|_ε)``.
        The caller divides the standard RL update by this array.

    Notes
    -----
    - For ``lambda_tv = 0`` this function should not be called (the caller
      skips it), but if called it returns an array of ones.
    - The denominator is clamped to ``[0.5, +∞)`` to prevent sign inversion
      or amplification blow-up when λ is large relative to the local
      curvature.  This clamp is a safety measure; well-chosen λ values
      should rarely trigger it.

    References
    ----------
    [1] Dey et al., Microscopy Research and Technique, 2006, Eq. (5).
    """
    # ── Discrete gradient (forward differences, Neumann BC) ───────────────
    dh, dw = forward_grad(x)

    # ── Smoothed gradient magnitude ───────────────────────────────────────
    # The ε² term prevents division by zero where ∇x ≈ 0 (flat regions).
    # This is standard in TV implementations and corresponds to the Huber
    # approximation of the L1 norm near the origin.
    mag = xp.sqrt(dh * dh + dw * dw + eps_grad * eps_grad)

    # ── Normalized gradient field  n = ∇x / |∇x|_ε ──────────────────────
    # In flat regions, mag ≈ eps_grad and (dh, dw) ≈ 0, so n ≈ 0 —
    # the regularization has no effect there, as desired.
    nh = dh / mag
    nw = dw / mag
    # dh, dw, mag are no longer needed; allow garbage collection.
    del dh, dw, mag

    # ── Divergence (backward differences, adjoint of −∇) ─────────────────
    div = backward_div(nh, nw)
    del nh, nw

    # ── Assemble correction factor ────────────────────────────────────────
    # C(x) = 1 − λ · div(∇x/|∇x|_ε)
    #
    # Safety clamp: if λ is too large, (1 − λ·div) can become ≤ 0, which
    # would invert or explode the estimate.  We clamp to a minimum of 0.5
    # to guarantee the correction is at most a factor-of-2 amplification.
    correction = 1.0 - lambda_tv * div

    xp.clip(correction, a_min=0.5, a_max=None, out=correction)

    return correction


# ══════════════════════════════════════════════════════════════════════════════
# Proximal TV Operator  (Chambolle 2004 Dual Projection)
# ══════════════════════════════════════════════════════════════════════════════

def prox_tv_chambolle(
    v: xp.ndarray,
    gamma: float,
    n_inner: int = 50,
    tau_dual: float = 0.125,
) -> xp.ndarray:
    """
    Compute the proximal operator of γ · TV(·) at point v.

    Solves the ROF denoising subproblem:

        prox_{γ TV}(v) = argmin_u  (1/2)||u − v||² + γ · TV(u)

    using Chambolle's fast dual projection algorithm [1].

    Dual Formulation
    ----------------
    The solution is:

        u* = v − γ · div(p*)

    where p* = (p_h*, p_w*) is the optimal dual variable satisfying the
    pointwise constraint |p_{i,j}| ≤ 1, found by iterating:

        g^n      = ∇(div(p^n) − v/γ)
        p^{n+1}  = (p^n + τ_d · g^n) / max(1, |p^n + τ_d · g^n|)

    The max operation projects each (p_h, p_w) vector onto the unit disk,
    enforcing the dual feasibility constraint.

    Step Size
    ---------
    The dual step τ_d must satisfy τ_d ≤ 1/(4·dim) = 1/8 for 2-D images
    to guarantee convergence (where dim=2 and 4 is the operator norm of
    the discrete gradient).  The default τ_d = 1/8 is the largest stable
    value.

    Parameters
    ----------
    v : xp.ndarray, shape (H, W), float32
        The point at which to evaluate the proximal operator (typically
        the result of a gradient descent step on the data fidelity term).
    gamma : float
        Proximal parameter = (outer step size τ) × (TV weight λ).
        Controls the strength of TV denoising in this subproblem.
        If ``gamma <= 0``, returns ``v.copy()`` immediately (identity).
    n_inner : int
        Number of Chambolle dual iterations.  30–50 is usually sufficient
        for satellite imagery; more may be needed for very strong TV (large γ).
    tau_dual : float
        Dual step size.  Must be ≤ 1/8 for convergence.  Default 1/8.

    Returns
    -------
    xp.ndarray, shape (H, W), float32
        The TV-denoised image prox_{γ TV}(v).

    References
    ----------
    [1] A. Chambolle, "An Algorithm for Total Variation Minimization and
        Applications," J. Math. Imaging and Vision, 20(1–2):89–97, 2004.
    """
    if gamma <= 0.0:
        return v.copy()

    inv_gamma = xp.float32(1.0 / gamma)

    # Dual variables — a 2-component vector field, initialised to zero.
    p_h = xp.zeros_like(v)
    p_w = xp.zeros_like(v)

    for _ in range(n_inner):
        # 1. Compute u_current = v − γ · div(p)   (primal from current dual)
        #    Then form the argument for the gradient:  div(p) − v/γ
        #    Equivalently:  −u_current / γ
        #    But computing div(p) − v/γ directly avoids the extra multiply.
        div_p = backward_div(p_h, p_w)
        arg = div_p - v * inv_gamma         # shape (H, W)

        # 2. Gradient of the argument
        g_h, g_w = forward_grad(arg)

        # 3. Semi-implicit update + pointwise projection onto unit ball
        p_h_new = p_h + tau_dual * g_h
        p_w_new = p_w + tau_dual * g_w

        # Pointwise magnitude
        mag = xp.sqrt(p_h_new * p_h_new + p_w_new * p_w_new)
        # Project: divide by max(1, |p|) to enforce |p| ≤ 1
        mag = xp.maximum(mag, 1.0)

        p_h = p_h_new / mag
        p_w = p_w_new / mag

    # Final primal recovery:  u* = v − γ · div(p*)
    result = v - gamma * backward_div(p_h, p_w)
    return result
