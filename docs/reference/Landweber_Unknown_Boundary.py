from __future__ import annotations

import importlib
import logging
import numpy as np
from typing import Optional, Literal

from Shared.Common.General_Utilities   import padding, cropping
from Shared.Common.PSF_Preprocessing  import psf_preprocess, condition_psf
from Shared.Common.Image_Preprocessing import (image_normalization, validate_image,
                                                to_grayscale, odd_crop_around_center)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# GPU Detection
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: This infrastructure section (GPU detection, backend selection, FFT
# helpers, utility functions) is identical to RL_Unknown_Boundary.py.
# In a production codebase, consider extracting into a shared base module
# (e.g. Shared.Common.Deconv_Backend) to avoid duplication.

_USER_GPU_FLAG: bool = True


def _detect_gpu() -> bool:
    """
    Probe whether a functional CUDA device is reachable via CuPy.

    Three-stage check: package presence → device count → live allocation.
    Returns True only if all stages succeed and _USER_GPU_FLAG is True.
    """
    if not _USER_GPU_FLAG:
        logger.info("GPU disabled by _USER_GPU_FLAG; using CPU.")
        return False

    if importlib.util.find_spec("cupy") is None:
        logger.info("CuPy not found; using CPU.")
        return False

    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            logger.warning("CuPy installed but no CUDA device found; using CPU.")
            return False

        dummy = cp.array([1.0], dtype=cp.float32)
        _ = dummy + dummy
        del dummy

        logger.info("CUDA device detected — GPU path enabled.")
        return True

    except Exception as e:
        logger.warning(
            "CuPy installed but GPU initialisation failed; "
            "falling back to NumPy/CPU.  Reason: %s", e
        )
        return False


_use_gpu: bool = _detect_gpu()


# ══════════════════════════════════════════════════════════════════════════════
# Backend Selection  (xp, _fft)
# ══════════════════════════════════════════════════════════════════════════════

if _use_gpu:
    import cupy as cp
    xp   = cp
    _fft = cp.fft
    try:
        cp.fft.config.set_plan_cache_size(64)
    except AttributeError:
        pass
else:
    xp   = np
    _fft = np.fft


# ══════════════════════════════════════════════════════════════════════════════
# FFT Helpers  (real-valued optimisation)
# ══════════════════════════════════════════════════════════════════════════════

def rfft2(a: xp.ndarray, **kwargs) -> xp.ndarray:
    """Backend-agnostic real 2-D FFT → shape (H, W//2+1) complex."""
    return _fft.rfft2(a, **kwargs)


def irfft2(a: xp.ndarray, s: tuple[int, int], **kwargs) -> xp.ndarray:
    """Backend-agnostic inverse real 2-D FFT → shape (H, W) real."""
    return _fft.irfft2(a, s=s, **kwargs)


ifftshift = _fft.ifftshift


# ══════════════════════════════════════════════════════════════════════════════
# Utility Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _freeze(a: xp.ndarray) -> xp.ndarray:
    """Mark array as read-only to prevent accidental in-place modification."""
    try:
        a.flags.writeable = False
    except AttributeError:
        pass
    return a


def _to_numpy(x: xp.ndarray) -> np.ndarray:
    """Transfer array to host (CPU) memory if on GPU; no-op on CPU."""
    if _use_gpu:
        return xp.asnumpy(x)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Discrete Gradient & Divergence  (Neumann BC)
# ══════════════════════════════════════════════════════════════════════════════
# These operators form an adjoint pair under Neumann (zero-flux) boundary
# conditions:  ⟨−∇x, p⟩ = ⟨x, div(p)⟩  for all x, p satisfying the BCs.
#
# This property is essential for both the Chambolle TV proximal solver
# (which relies on adjointness for convergence guarantees) and for any
# primal-dual algorithm applied to the TV-regularised problem.
#
# Convention:
#   ∇  = forward differences,  last row/col = 0    (Neumann)
#   div = backward differences, adjoint of −∇

def _forward_grad(x: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Discrete gradient with forward differences and Neumann BC.

    Parameters
    ----------
    x : xp.ndarray, shape (H, W)

    Returns
    -------
    dh, dw : xp.ndarray, each shape (H, W)
        Vertical and horizontal partial derivatives.
        Last row of dh and last column of dw are zero (Neumann BC).
    """
    dh = xp.zeros_like(x)
    dw = xp.zeros_like(x)
    dh[:-1, :] = x[1:, :] - x[:-1, :]    # ∂x/∂h; last row = 0
    dw[:, :-1] = x[:, 1:] - x[:, :-1]    # ∂x/∂w; last col = 0
    return dh, dw


def _backward_div(p_h: xp.ndarray, p_w: xp.ndarray) -> xp.ndarray:
    """
    Discrete divergence with backward differences (adjoint of −∇).

    Parameters
    ----------
    p_h, p_w : xp.ndarray, each shape (H, W)
        Components of a vector field.

    Returns
    -------
    div : xp.ndarray, shape (H, W)
        div(p) = bwd_h(p_h) + bwd_w(p_w)

    The backward differences are defined as:
        bwd_h(p)[0, :]     =  p[0, :]                (top boundary)
        bwd_h(p)[i, :]     =  p[i, :] − p[i−1, :]   (interior)
        bwd_h(p)[H−1, :]   = −p[H−2, :]              (bottom: forward set p[H−1]=0)
    and analogously for bwd_w along columns.
    """
    div = xp.empty_like(p_h)

    # Vertical component
    div[0, :]    = p_h[0, :]
    div[1:-1, :] = p_h[1:-1, :] - p_h[:-2, :]
    div[-1, :]   = -p_h[-2, :]

    # Horizontal component (accumulated)
    div[:, 0]    += p_w[:, 0]
    div[:, 1:-1] += p_w[:, 1:-1] - p_w[:, :-2]
    div[:, -1]   += -p_w[:, -2]

    return div


# ══════════════════════════════════════════════════════════════════════════════
# Proximal TV Operator  (Chambolle 2004 Dual Projection)
# ══════════════════════════════════════════════════════════════════════════════

def _prox_tv_chambolle(
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
        div_p = _backward_div(p_h, p_w)
        arg = div_p - v * inv_gamma         # shape (H, W)

        # 2. Gradient of the argument
        g_h, g_w = _forward_grad(arg)

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
    result = v - gamma * _backward_div(p_h, p_w)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ══════════════════════════════════════════════════════════════════════════════

PaddingStr = Literal["Reflect", "Symmetric", "Wrap", "Edge", "LinearRamp", "Zero"]


# ══════════════════════════════════════════════════════════════════════════════
# Landweber  + Proximal TV  + Unknown Boundaries  (FISTA-accelerated)
# ══════════════════════════════════════════════════════════════════════════════

class LandweberUnknownBoundary:
    """
    FISTA-accelerated preconditioned Landweber with proximal TV
    regularization and unknown-boundary (masked data-fidelity) treatment.

    Solves the variational problem:

        min_x  (1/2) ||M(Hx − y)||²  +  λ · TV(x)           … (★)

    where
        H   = convolution with the PSF (linear, shift-invariant blur),
        M   = binary mask  (1 on observed support Ω, 0 outside),
        y   = observed (blurred + noisy) image,
        TV  = isotropic total variation.

    Algorithm
    ---------
    The data-fidelity term f(x) = (1/2)||M(Hx − y)||² has gradient:

        ∇f(x) = H^T [ M · (Hx − y) ]

    which is Lipschitz-continuous with constant L = ||H^T M H||_op.
    We bound L ≤ max(|H(f)|²) (the spectral norm of H^T H) since
    ||M||_op = 1 for a binary mask.

    **Preconditioned variant** (default, ``precondition=True``):
    To equalise convergence speed across the canvas (interior pixels
    where H^T M ≈ 1 vs. boundary pixels where H^T M ≪ 1), we apply
    the diagonal preconditioner D = diag(H^T M + ε):

        gradient step:   z_{k+½} = z_k − (τ/D) · ∇f(z_k)
        proximal step:   x_{k+1} = prox_{(τ/D)·λ·TV}(z_{k+½})

    When ``precondition=False``, the standard (un-preconditioned) FISTA is
    used with a scalar step size τ = 1/L.

    **FISTA acceleration** (Beck & Teboulle, 2009):
    Nesterov momentum is applied to achieve O(1/k²) convergence:

        t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
        z_{k+1} = x_{k+1} + ((t_k − 1) / t_{k+1}) · (x_{k+1} − x_k)

    with adaptive restart (O'Donoghue & Candès, 2015): if consecutive
    iterate steps reverse direction, reset t_k = 1 to quench oscillation.

    Noise Model Assumption
    ----------------------
    The least-squares data fidelity assumes **additive Gaussian noise**.
    This is appropriate for satellite imagery after radiometric calibration
    where read noise and quantisation dominate (moderate-to-high SNR
    regime).  For photon-noise-dominated scenarios (low light, short
    integration), the Poisson-likelihood-based RL method in
    ``RLUnknownBoundary`` is more appropriate.

    Constructor
    -----------
    Identical to ``RLUnknownBoundary``: same padding, mask, PSF
    preparation, and H^T M precomputation.

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

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        paddingMode: PaddingStr = "Reflect",
        padding_scale: float = 2.0,
        initialEstimate: Optional[np.ndarray] = None,
        apply_taper_on_padding_band: bool = False,
        htm_floor_frac: float = 0.05,
    ) -> None:

        validate_image(image)
        gray: np.ndarray = to_grayscale(image)

        # Enforce odd spatial dimensions (ifftshift ↔ fftshift exactness).
        H, W = gray.shape
        OH = H if H % 2 == 1 else H - 1
        OW = W if W % 2 == 1 else W - 1
        if OH <= 0 or OW <= 0:
            raise ValueError("Image is too small after enforcing odd spatial shape.")
        if (OH, OW) != (H, W):
            gray = odd_crop_around_center(gray, (OH, OW))

        # Normalise to [0, 1].
        gray = image_normalization(image=gray, bit_depth=1, is_int=False)

        # Keep original size for final crop (single assignment).
        self.h, self.w = gray.shape

        # ── FFT canvas size ───────────────────────────────────────────────
        pH, pW = psf.shape
        fH = int(self.h + padding_scale * pH)
        fW = int(self.w + padding_scale * pW)
        OH_full = fH if fH % 2 == 1 else fH + 1
        OW_full = fW if fW % 2 == 1 else fW + 1
        self.full_shape: tuple[int, int] = (OH_full, OW_full)

        logger.debug("Image shape %s  →  padded canvas %s", gray.shape, self.full_shape)

        if _use_gpu:
            _dummy = xp.zeros(self.full_shape, dtype=xp.float32)
            _ = rfft2(_dummy)
            del _dummy

        # ── Observed image on padded canvas ───────────────────────────────
        self.image = xp.array(
            padding(
                image=gray,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )

        # ── Mask M: 1 on observed support Ω, 0 outside ───────────────────
        # ASSUMPTION: padding() centres the image via integer division
        # off = (canvas − image) // 2.  Verify your padding() matches.
        self.mask = xp.zeros(self.full_shape, dtype=xp.float32)
        off_y = (self.full_shape[0] - self.h) // 2
        off_x = (self.full_shape[1] - self.w) // 2
        self.mask[off_y:off_y + self.h, off_x:off_x + self.w] = 1.0

        # ── PSF frequency-domain preparation ──────────────────────────────
        psf_np: np.ndarray = psf_preprocess(
            psf=psf,
            center_method="com",
            remove_negatives="clip",
            eps=1e-12,
            enforce_odd_shape=True,
        )
        psf_np = condition_psf(
            psf=psf_np,
            bg_ring_frac=0.15,
            taper_outer_frac=0.20,
            taper_end_frac=0.50,
        )

        psf_pad: xp.ndarray = xp.array(
            padding(image=psf_np, full_size=self.full_shape,
                    Type="Zero", apply_taper=False),
            dtype=xp.float32,
        )
        psf_pad = ifftshift(psf_pad)

        self.PF: xp.ndarray  = _freeze(rfft2(psf_pad))
        self.conjPF: xp.ndarray = _freeze(self.PF.conj())

        # ── H^T M with relative floor clamp ───────────────────────────────
        fshape = self.full_shape
        htm_raw = irfft2(self.conjPF * rfft2(self.mask), s=fshape).astype(xp.float32)

        htm_max = float(xp.max(htm_raw))
        htm_floor = max(htm_floor_frac * htm_max, 1e-12)
        xp.clip(htm_raw, a_min=htm_floor, a_max=None, out=htm_raw)
        self.HTM = _freeze(htm_raw)

        logger.debug(
            "HTM: max=%.4f, applied floor=%.4f (%.1f%% of max)",
            htm_max, htm_floor, 100.0 * htm_floor_frac,
        )

        # ── Lipschitz constant of ∇f ──────────────────────────────────────
        # L = ||H^T M H||_op ≤ ||H^T H||_op = max(|H(f)|²)
        # This bound is tight when M ≈ I (large image, small PSF support).
        self._lipschitz = float(xp.max(xp.abs(self.PF) ** 2))
        logger.debug("Lipschitz constant L = %.6f", self._lipschitz)

        # ── Initial estimate on padded canvas ─────────────────────────────
        init_source = initialEstimate if initialEstimate is not None else gray
        self.estimated_image = xp.array(
            padding(
                image=init_source,
                full_size=self.full_shape,
                Type=paddingMode,
                apply_taper=bool(apply_taper_on_padding_band),
            ),
            dtype=xp.float32,
        )

        # Non-negative start (not strictly required for Landweber, but
        # beneficial when combined with positivity projection or TV).
        eps0 = xp.float32(1e-8)
        xp.maximum(self.estimated_image, eps0, out=self.estimated_image)

    # ──────────────────────────────────────────────────────────────────────
    # Main Deblurring Loop
    # ──────────────────────────────────────────────────────────────────────

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
            Maximum number of outer (FISTA) iterations.  Typical range:
            100–500.  Landweber generally needs more iterations than RL to
            reach comparable reconstruction quality, but FISTA acceleration
            (O(1/k²) vs O(1/k)) compensates substantially.
        lambda_tv : float
            TV regularization weight.  Larger values produce smoother
            reconstructions.  Typical range for [0,1]-normalised satellite
            imagery: 1e-4 to 1e-2.  Set to 0 to disable TV (pure Landweber).
        tol : float
            Relative-change convergence threshold:
            ||x_{k+1} − x_k|| / ||x_{k+1}|| < tol  →  stop.
        min_iter : int
            Minimum iterations before convergence checking begins.
        check_every : int
            Check convergence every this many iterations.
        step_size : float or None
            Outer gradient step size τ.  If None (default), automatically
            set to 0.95/L (un-preconditioned) or 0.95 (preconditioned),
            where L is the Lipschitz constant of ∇f.  The 0.95 safety
            factor provides a small margin below the theoretical stability
            limit.
        enforce_positivity : bool
            If True, project onto x ≥ ε after each proximal step.  This
            is a convex constraint that is compatible with the proximal
            framework (the projection commutes with the TV prox in the
            limit, and in practice interleaving them converges reliably).
        epsilon_positivity : float
            Floor value for the positivity constraint.
        precondition : bool
            If True (default), divide the gradient by H^T M + ε to
            equalise convergence across the canvas.  This produces a
            variable-metric (diagonal preconditioned) proximal gradient
            method.

            When preconditioned, the proximal TV step uses a spatially
            varying γ = τ · λ / HTM[i,j].  For efficiency, we approximate
            this with a scalar γ = τ · λ / median(HTM) which is exact
            inside Ω (where HTM ≈ 1 for a normalised PSF) and slightly
            over-regularises the exterior.  This is acceptable because the
            exterior is unconstrained by data anyway.
        tv_n_inner : int
            Number of inner Chambolle iterations for the TV proximal
            operator.  30–50 is adequate for most cases.
        adaptive_restart : bool
            If True, apply the O'Donoghue–Candès velocity restart: when
            consecutive iterate steps reverse direction (negative inner
            product), reset momentum to prevent oscillation.

        Returns
        -------
        np.ndarray, shape (self.h, self.w), float32
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
        # Preconditioned:    the effective Lipschitz constant after dividing
        #   by HTM is ≈ L / min(HTM).  But since HTM is clamped to a floor,
        #   and the floor is typically ~5% of max, the effective L_prec can
        #   be large.  In practice τ = 0.95 works because HTM ≈ 1 inside Ω
        #   and the floor clamp limits the worst case.
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

        # ── Proximal parameter γ for TV ───────────────────────────────────
        # For the un-preconditioned case:   γ = τ · λ
        # For the preconditioned case:      γ = τ · λ / median(HTM|_Ω)
        #   where the median is taken over the observed region only
        #   (where HTM ≈ 1.0 for a normalised PSF, so γ ≈ τ · λ).
        if use_tv:
            if precondition:
                # Median of HTM inside Ω (where M = 1)
                htm_inside = HTM[M > 0.5]
                htm_med = float(xp.median(htm_inside))
                gamma_tv = tau * lam / max(htm_med, 1e-12)
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
            Hz  = irfft2(PF * rfft2(z_k), s=fshape)
            residual = M * (Hz - y)
            grad = irfft2(conjPF * rfft2(residual), s=fshape)

            # ── Gradient descent step ─────────────────────────────────────
            if precondition:
                # Preconditioned: divide gradient by HTM (already floored)
                x_half = z_k - tau * (grad / HTM)
            else:
                x_half = z_k - tau * grad

            # ── Proximal TV step ──────────────────────────────────────────
            if use_tv:
                x_new = _prox_tv_chambolle(x_half, gamma_tv, n_inner=tv_n_inner)
            else:
                x_new = x_half

            # ── Positivity projection ─────────────────────────────────────
            if enforce_positivity:
                xp.maximum(x_new, eps_pos, out=x_new)

            # ── FISTA momentum update ─────────────────────────────────────
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            momentum = (t_k - 1.0) / t_new

            # Adaptive restart (O'Donoghue & Candès 2015, §3.1):
            # Velocity restart — if the direction of progress reverses
            # between consecutive iterates, the momentum is oscillating
            # and should be reset to prevent divergence.
            #   Test:  ⟨x_{k+1} − x_k, x_k − x_{k−1}⟩ < 0  →  restart
            if adaptive_restart and k > 0:
                dx_new = x_new - x_k      # current step direction
                dx_old = x_k - x_km1      # previous step direction
                ip = float(xp.sum(dx_new * dx_old))
                if ip < 0.0:
                    t_new = 1.0
                    momentum = 0.0     # z_new = x_new + 0*(...) = x_new
                    logger.debug("FISTA restart at iteration %d (ip=%.2e)", k + 1, ip)

            z_new = x_new + momentum * (x_new - x_k)

            # ── Convergence check ─────────────────────────────────────────
            if k >= min_iter and (k + 1) % check_every == 0:
                den = xp.linalg.norm(x_new)
                den = den if float(den) > 0.0 else eps_pos
                rel_chg = float(xp.linalg.norm(x_new - x_k) / den)
                if rel_chg < tol:
                    logger.info(
                        "Converged at iteration %d/%d  (rel_change=%.2e < tol=%.2e)",
                        k + 1, num_iter, rel_chg, tol,
                    )
                    break

            # ── Advance state ─────────────────────────────────────────────
            x_km1 = x_k
            x_k   = x_new
            z_k   = z_new
            t_k   = t_new

        else:
            # Loop completed without convergence.
            logger.info(
                "Reached max iterations (%d) without convergence (tol=%.2e).",
                num_iter, tol,
            )

        self.estimated_image = x_k.copy()

        # Crop back to original FOV and return on CPU.
        return _to_numpy(cropping(x_k, (self.h, self.w)))


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Wrapper
# ══════════════════════════════════════════════════════════════════════════════

def landweber_deblur_unknown_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 200,
    lambda_tv: float = 0.001,
    paddingMode: PaddingStr = "Reflect",
    padding_scale: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience one-shot wrapper for LandweberUnknownBoundary.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    iters : int
        Maximum outer iterations.
    lambda_tv : float
        TV regularization weight.
    paddingMode : str
        Edge-extension mode for the padded canvas.
    padding_scale : float
        Padding width as a multiple of the PSF size.
    **kwargs
        Forwarded to LandweberUnknownBoundary.__init__ and .deblur().

    Returns
    -------
    np.ndarray
        Deconvolved image.
    """
    # Split kwargs into constructor vs. deblur arguments.
    init_keys = {"initialEstimate", "apply_taper_on_padding_band", "htm_floor_frac"}
    init_kw   = {k: v for k, v in kwargs.items() if k in init_keys}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in init_keys}

    lw = LandweberUnknownBoundary(
        image=image,
        psf=psf,
        paddingMode=paddingMode,
        padding_scale=padding_scale,
        **init_kw,
    )
    return lw.deblur(num_iter=iters, lambda_tv=lambda_tv, **deblur_kw)
