"""
fista.py — FISTA deconvolution with three regularization modes.

Solves the composite minimization problem:

    min_x  f(x) + λ g(x)

where:
    f(x) = (1/2) ||M ⊙ (Hx − y)||²    (smooth, masked data fidelity)
    g(x) = regularizer                   (convex, possibly non-smooth)

Three choices for g(x):
    reg_mode="TV"         g(x) = ||∇x||_{2,1}   isotropic Total Variation
    reg_mode="L1"         g(x) = ||x||_1         image-domain sparsity
    reg_mode="L1_wavelet" g(x) = ||Wx||_1        wavelet-domain sparsity

Algorithm: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm,
Beck & Teboulle 2009) with O'Donoghue-Candès adaptive restart.

Gradient of f:
    ∇f(x) = H^T [ M ⊙ (Hx − y) ]
    step size τ = 1/L where L = max |H(f)|² (Lipschitz constant of ∇f)

Key differences from LandweberUnknownBoundary
----------------------------------------------
1. No diagonal preconditioning.  Landweber divides the gradient by HTM.
   FISTA uses the standard unpreconditioned step τ = 1/L (exact Beck-Teboulle).

2. Three regularization modes.  Landweber only has TV.

3. Overridable _prox_step method.  Enables future PnP-FISTA by subclassing.

References
----------
[BT09]  Beck, A. & Teboulle, M. "A fast iterative shrinkage-thresholding
        algorithm for linear inverse problems." SIAM J. Imaging Sci.,
        2(1):183–202, 2009.

[BT09b] Beck, A. & Teboulle, M. "Fast gradient-based algorithms for constrained
        total variation image denoising and deblurring problems." IEEE Trans.
        Image Process., 18(11):2419–2434, 2009.

[OC15]  O'Donoghue, B. & Candès, E. "Adaptive restart for accelerated gradient
        schemes." Found. Comput. Math., 15(3):715–732, 2015.

[Cha04] Chambolle, A. "An algorithm for total variation minimisation and
        applications." J. Math. Imaging Vision, 20:89–97, 2004.

[CDL98] Chambolle, A., De Vore, R., Lee, N., Lucier, B. "Nonlinear wavelet
        image processing: variational problems, compression, and noise removal
        through wavelet shrinkage." IEEE Trans. Image Process., 7(3):319–335,
        1998.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ._backend import xp, rfft2, irfft2, _to_numpy
from ._base import DeconvBase

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Optional PyWavelets dependency
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pywt as _pywt
    _HAS_PYWT: bool = True
except ImportError:
    _HAS_PYWT = False


# ══════════════════════════════════════════════════════════════════════════════
# FISTADeconv
# ══════════════════════════════════════════════════════════════════════════════

class FISTADeconv(DeconvBase):
    """
    FISTA deconvolution with TV, L1, or wavelet-domain sparsity regularization.

    Implements Beck & Teboulle (2009) Algorithm 1 with O'Donoghue-Candès
    adaptive restart and three interchangeable proximal operators.

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    wavelet : str, optional
        Wavelet name for L1_wavelet mode (PyWavelets naming convention).
        Default ``"db4"`` (Daubechies-4 — orthogonal, good smoothness).
    wavelet_levels : int, optional
        DWT decomposition depth.  Clamped to [1, 10].  Default 3.
    nonneg : bool, optional
        If ``True`` (default), enforce non-negativity after each proximal
        step.  Can be overridden at :meth:`deblur` call time.
    **base_kwargs
        Forwarded to :class:`~Reconstruction._base.DeconvBase`:
        ``paddingMode``, ``padding_scale``, ``initialEstimate``, etc.

    Attributes
    ----------
    step_size : float
        τ = 1 / L, the FISTA step size.
    lipschitz : float
        L = max |H(f)|², Lipschitz constant of the smooth objective gradient.

    Notes
    -----
    ``use_mask=True`` is forced: the masked gradient is essential for the
    unknown-boundary formulation (CLAUDE.md §Solver Architecture).

    The proximal operator for L1_wavelet mode is exact for orthogonal wavelets
    (Haar, Daubechies dbN, Symlets symN) and a good approximation for
    biorthogonal families.  Approximation sub-coefficients at the coarsest
    DWT level are NOT thresholded to preserve the image mean [CDL98].
    """

    _INIT_KEYS: frozenset[str] = DeconvBase._INIT_KEYS | frozenset({
        "wavelet",
        "wavelet_levels",
        "nonneg",
    })

    def __init__(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        wavelet: str = "db4",
        wavelet_levels: int = 3,
        nonneg: bool = True,
        **base_kwargs,
    ) -> None:
        super().__init__(image, psf, use_mask=True, **base_kwargs)

        self._wavelet: str = str(wavelet)
        self._levels: int = int(np.clip(int(wavelet_levels), 1, 10))
        self.nonneg: bool = bool(nonneg)
        self._step: float = 1.0 / self._lipschitz

        # Validate wavelet name at construction time (only when pywt available)
        if _HAS_PYWT:
            try:
                _pywt.Wavelet(self._wavelet)
            except Exception as exc:
                raise ValueError(
                    f"Invalid wavelet name {self._wavelet!r}: {exc}"
                ) from exc

        logger.debug(
            "FISTADeconv: L=%.6f, τ=%.6f, wavelet=%s, levels=%d, nonneg=%s",
            self._lipschitz, self._step, self._wavelet, self._levels, self.nonneg,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def step_size(self) -> float:
        """FISTA step size τ = 1 / L."""
        return self._step

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
        lambda_reg: float = 0.001,
        reg_mode: str = "TV",
        tol: float = 1e-5,
        min_iter: int = 10,
        check_every: int = 5,
        nonneg: Optional[bool] = None,
        tv_inner: int = 30,
    ) -> np.ndarray:
        """
        Run FISTA with the selected regularization mode.

        Parameters
        ----------
        num_iter : int
            Maximum outer FISTA iterations.  Clamped to [1, 10000].
        lambda_reg : float
            Regularization weight λ (TV, L1, or wavelet coefficient threshold).
        reg_mode : {"TV", "L1", "L1_wavelet"}
            Regularizer choice.  ``"L1_wavelet"`` requires PyWavelets.
        tol : float
            Relative-change convergence threshold.
        min_iter : int
            Minimum iterations before convergence checking begins.
        check_every : int
            Check convergence every this many iterations.
        nonneg : bool or None
            Non-negativity enforcement.  ``None`` uses the constructor value.
        tv_inner : int
            Inner Chambolle iterations for the TV proximal operator.
            Ignored for ``reg_mode != "TV"``.

        Returns
        -------
        np.ndarray, shape (self.h, self.w)
            Deconvolved image on CPU, cropped to the original field of view.

        Raises
        ------
        ValueError
            If ``reg_mode`` is not one of the three accepted values.
        ImportError
            If ``reg_mode="L1_wavelet"`` and PyWavelets is not installed.
        """
        num_iter    = int(np.clip(num_iter, 1, 10000))
        nonneg_flag = self.nonneg if nonneg is None else bool(nonneg)

        # ── Validate reg_mode early ──────────────────────────────────────
        _VALID_MODES = ("TV", "L1", "L1_wavelet")
        if reg_mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown reg_mode: {reg_mode!r}. "
                f"Expected one of {_VALID_MODES}."
            )
        if reg_mode == "L1_wavelet" and not _HAS_PYWT:
            raise ImportError(
                "reg_mode='L1_wavelet' requires PyWavelets. "
                "Install with: pip install PyWavelets"
            )

        tau = self._step
        s   = self.full_shape
        PF  = self.PF
        cPF = self.conjPF
        M   = self.mask
        y   = self.image

        logger.debug(
            "FISTA deblur: num_iter=%d, λ=%.2e, mode=%s, τ=%.4e, nonneg=%s",
            num_iter, lambda_reg, reg_mode, tau, nonneg_flag,
        )

        # ── FISTA state ─────────────────────────────────────────────────
        x_k   = self.estimated_image.copy()  # current iterate x_k
        x_km1 = x_k.copy()                   # previous iterate x_{k-1}
        y_k   = x_k.copy()                   # Nesterov extrapolated point
        t_k   = 1.0                           # momentum parameter

        for k in range(num_iter):

            # ── 1. Gradient ∇f(y_k) = H^T [ M ⊙ (H y_k − y) ] ─────────
            # Uses rfft2/irfft2 (real FFT, half-spectrum) throughout.
            # Always pass s=full_shape to irfft2 (even/odd ambiguity fix).
            Hy_k  = irfft2(PF * rfft2(y_k), s=s)
            resid = M * (Hy_k - y)
            grad  = irfft2(cPF * rfft2(resid), s=s)

            # ── 2. Gradient step: z = y_k − τ ∇f(y_k) ──────────────────
            z = y_k - tau * grad

            # ── 3. Proximal step: x_{k+1} = prox_{τλ,g}(z) ─────────────
            x_new = self._prox_step(z, tau * lambda_reg, reg_mode, tv_inner)

            # ── 4. Positivity projection ─────────────────────────────────
            if nonneg_flag:
                xp.maximum(x_new, xp.float32(0.0), out=x_new)

            # ── 5. Convergence check (before momentum update) ────────────
            if k >= min_iter and (k + 1) % check_every == 0:
                _, converged = self._check_convergence(
                    x_new, x_k, k=k, num_iter=num_iter, tol=tol,
                )
                if converged:
                    x_k = x_new   # keep the best iterate
                    break

            # ── 6. FISTA momentum update (Beck-Teboulle Algorithm 1) ─────
            # t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
            # y_{k+1} = x_{k+1} + ((t_k − 1) / t_{k+1}) (x_{k+1} − x_k)
            t_new    = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
            momentum = (t_k - 1.0) / t_new
            y_new    = x_new + xp.float32(momentum) * (x_new - x_k)

            # ── 7. O'Donoghue-Candès gradient restart [OC15 §3.1] ───────
            # Restart when consecutive iterate steps are opposed (angle > 90°):
            #   ⟨x_{k+1} − x_k, x_k − x_{k-1}⟩ < 0
            # This detects when the Nesterov extrapolation overshoots and
            # momentum is pointing against the descent direction.
            # Guard on k>0: at k=0, x_km1 == x_k (zero previous step).
            if k > 0:
                ip = float(xp.sum((x_new - x_k) * (x_k - x_km1)))
                if ip < 0.0:
                    t_new = 1.0
                    y_new = x_new.copy()
                    logger.debug(
                        "FISTA restart at iteration %d (ip=%.2e)", k + 1, ip
                    )

            # ── 8. Advance state ─────────────────────────────────────────
            x_km1 = x_k
            x_k   = x_new
            y_k   = y_new
            t_k   = t_new

        else:
            self._log_no_convergence(num_iter, tol)

        return self._crop_and_return(x_k)

    # ══════════════════════════════════════════════════════════════════════
    # Proximal interface (overridable for PnP-FISTA extension)
    # ══════════════════════════════════════════════════════════════════════

    def _prox_step(
        self,
        z: xp.ndarray,
        threshold: float,
        reg_mode: str,
        tv_inner: int,
    ) -> xp.ndarray:
        """
        Evaluate the proximal operator prox_{threshold, g}(z).

        Override this method to plug in a custom regularizer (e.g. a learned
        denoiser for PnP-FISTA).

        Parameters
        ----------
        z : xp.ndarray
            Gradient-step result y_k − τ ∇f(y_k).
        threshold : float
            τ × λ, the combined step-size × regularization weight.
        reg_mode : str
            ``"TV"``, ``"L1"``, or ``"L1_wavelet"``.
        tv_inner : int
            Inner Chambolle iterations (used only when ``reg_mode="TV"``).

        Returns
        -------
        xp.ndarray
            Proximal evaluation, same shape and dtype as ``z``.
        """
        if reg_mode == "TV":
            return self._prox_tv(z, threshold, tv_inner)
        elif reg_mode == "L1":
            return self._prox_l1(z, threshold)
        elif reg_mode == "L1_wavelet":
            return self._prox_l1_wavelet(z, threshold)
        else:
            raise ValueError(f"Unknown reg_mode: {reg_mode!r}")

    def _prox_tv(
        self,
        z: xp.ndarray,
        gamma: float,
        n_inner: int,
    ) -> xp.ndarray:
        """
        TV proximal operator via Chambolle dual projection [Cha04].

        Solves:  prox_{γ TV}(z) = argmin_u (1/2)||u − z||² + γ TV(u)

        Uses Neumann BC (correct for a standalone proximal sub-problem;
        see CLAUDE.md pitfall #8 for the BC discussion).

        Parameters
        ----------
        z : xp.ndarray
            Input array.
        gamma : float
            τ × λ.
        n_inner : int
            Number of Chambolle dual iterations.

        Returns
        -------
        xp.ndarray
        """
        from ._tv_operators import prox_tv_chambolle
        return prox_tv_chambolle(z, gamma=gamma, n_inner=n_inner)

    def _prox_l1(
        self,
        z: xp.ndarray,
        threshold: float,
    ) -> xp.ndarray:
        """
        Image-domain soft-thresholding (ℓ1 proximal).

        prox_{τλ, ||·||_1}(z) = sign(z) ⊙ max(|z| − τλ, 0)

        Parameters
        ----------
        z : xp.ndarray
            Input array.
        threshold : float
            τ × λ.

        Returns
        -------
        xp.ndarray, same dtype as z.
        """
        th = xp.float32(threshold)
        return xp.sign(z) * xp.maximum(xp.abs(z) - th, xp.float32(0.0))

    def _prox_l1_wavelet(
        self,
        z: xp.ndarray,
        threshold: float,
    ) -> xp.ndarray:
        """
        Wavelet-domain ℓ1 proximal via DWT shrinkage [CDL98].

        For orthogonal wavelets (db, sym, haar), this is the EXACT proximal
        of λ ||W·||_1.  For biorthogonal families it is an approximation.

        Approximation coefficients at the coarsest DWT level are NOT
        thresholded, preserving the image mean and large-scale structure.

        pywt is CPU-only; GPU↔CPU transfer is handled transparently.

        Parameters
        ----------
        z : xp.ndarray
            Input array.
        threshold : float
            τ × λ.

        Returns
        -------
        xp.ndarray, same shape and dtype as z.
        """
        if not _HAS_PYWT:
            raise ImportError(
                "L1_wavelet mode requires PyWavelets. "
                "Install with: pip install PyWavelets"
            )

        # GPU → CPU (no-op on CPU)
        z_np = _to_numpy(z).astype(np.float64)

        # Forward DWT
        coeffs = _pywt.wavedec2(z_np, self._wavelet, level=self._levels)

        # Soft-threshold DETAIL coefficients only (indices 1..levels).
        # coeffs[0] = approximation at coarsest level — left untouched.
        th = float(threshold)
        for i in range(1, len(coeffs)):
            coeffs[i] = tuple(
                np.sign(c) * np.maximum(np.abs(c) - th, 0.0)
                for c in coeffs[i]
            )

        # Inverse DWT
        result = _pywt.waverec2(coeffs, self._wavelet)

        # pywt.waverec2 may return shape (H+1, W) or (H, W+1) for odd dims.
        if result.shape != z_np.shape:
            result = result[: z_np.shape[0], : z_np.shape[1]]

        # CPU → GPU (no-op on CPU); match input dtype
        return xp.array(result.astype(np.float32), dtype=z.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def fista_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = 200,
    lambda_reg: float = 0.001,
    reg_mode: str = "TV",
    **kwargs,
) -> np.ndarray:
    """
    One-shot FISTA deconvolution.

    Splits ``**kwargs`` between :class:`FISTADeconv` constructor parameters
    (those in :attr:`FISTADeconv._INIT_KEYS`) and :meth:`~FISTADeconv.deblur`
    parameters (everything else).

    Parameters
    ----------
    image : np.ndarray
        Observed (blurred + noisy) image.
    psf : np.ndarray
        Point spread function.
    iters : int
        Maximum FISTA iterations.
    lambda_reg : float
        Regularization weight λ.
    reg_mode : {"TV", "L1", "L1_wavelet"}
        Regularizer choice.
    **kwargs
        Any parameter accepted by :class:`FISTADeconv` or
        :meth:`~FISTADeconv.deblur` (e.g. ``padding_scale``, ``tv_inner``).

    Returns
    -------
    np.ndarray
        Deconvolved image, shape (H, W) matching the original image field of
        view.
    """
    init_kw   = {k: v for k, v in kwargs.items() if k in FISTADeconv._INIT_KEYS}
    deblur_kw = {k: v for k, v in kwargs.items() if k not in FISTADeconv._INIT_KEYS}
    solver = FISTADeconv(image, psf, **init_kw)
    return solver.deblur(
        num_iter=iters,
        lambda_reg=lambda_reg,
        reg_mode=reg_mode,
        **deblur_kw,
    )
