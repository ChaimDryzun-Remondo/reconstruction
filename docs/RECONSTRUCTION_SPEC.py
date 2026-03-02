# Reconstruction Package — Architecture Specification & Implementation Plan
# =========================================================================
#
# This document is a complete specification for refactoring the existing
# standalone deconvolution modules (RL_Unknown_Boundary.py and
# Landweber_Unknown_Boundary.py) into a clean, extensible package.
#
# TARGET AUDIENCE: Claude Code (or any implementer).
# Follow the phases in order.  Each phase ends with a verification step.
# Do NOT skip verifications — they catch integration errors early.
#
# EXISTING CODE LOCATIONS (read these first to understand current state):
#   - RL_Unknown_Boundary.py          (567 lines, the corrected version)
#   - Landweber_Unknown_Boundary.py   (757 lines)
#   - Shared/Common/General_Utilities.py   (provides: padding, cropping)
#   - Shared/Common/PSF_Preprocessing.py   (provides: psf_preprocess, condition_psf)
#   - Shared/Common/Image_Preprocessing.py (provides: image_normalization,
#         validate_image, to_grayscale, odd_crop_around_center)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1:  DIRECTORY STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════
#
# Create the following directory tree.  All files listed below will be
# created during the implementation phases.
#
#   Reconstruction/
#   ├── __init__.py                         Phase 6
#   ├── _backend.py                         Phase 1
#   ├── _tv_operators.py                    Phase 2
#   ├── _base.py                            Phase 3
#   ├── rl_unknown_boundary.py              Phase 4a
#   ├── landweber_unknown_boundary.py       Phase 4b
#   ├── wiener.py                           Phase 5a
#   ├── rl_standard.py                      Phase 5b  (classical RL, no mask)
#   ├── admm.py                             Phase 5c
#   └── tval3.py                            Phase 5d
#
# NOTES:
# - Leading-underscore modules (_backend, _tv_operators, _base) are internal.
#   Users import from the package root or from algorithm modules directly.
# - Phases 5a–5d are independent of each other and can be done in any order.
# - The existing standalone files remain untouched until all phases pass.
#   They are replaced only at the end (Phase 7).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2:  SHARED CONVENTIONS
# ═════════════════════════════════════════════════════════════════════════════
#
# These conventions apply to ALL files in the package.
#
# 2.1  IMPORTS
#   - Every file starts with `from __future__ import annotations`.
#   - Standard library, then third-party, then local — separated by blanks.
#   - NumPy is always imported as `np`.
#   - The backend array module is always accessed as `xp` (imported from
#     _backend).  Never import numpy or cupy directly for array operations
#     in algorithm files.
#
# 2.2  TYPE HINTS
#   - Use lowercase generics: `tuple[int, int]`, not `Tuple[int, int]`.
#   - Use `Optional[X]` from typing (not `X | None`) for 3.9 compatibility.
#   - The PSF/image arrays in public signatures are `np.ndarray` (CPU).
#     Internal arrays are `xp.ndarray` (may be GPU).
#
# 2.3  LOGGING
#   - Each module: `logger = logging.getLogger(__name__)`.
#   - Use logger.debug for per-iteration or setup details.
#   - Use logger.info for convergence events and summary statistics.
#   - Use logger.warning for fallbacks and potential issues.
#   - NEVER use print().
#
# 2.4  FLOAT PRECISION
#   - All internal computation in float32 (xp.float32).
#   - Epsilon constants stored as xp.float32 scalars.
#
# 2.5  DOCSTRINGS
#   - NumPy-style docstrings on all public classes and functions.
#   - Include Parameters, Returns, Notes, References sections as needed.
#   - Mathematical equations use ASCII/Unicode where practical.
#   - Reference the relevant papers with author, journal, year, equation #.
#
# 2.6  SECTION HEADERS
#   - Use the box-drawing style from the existing code:
#     # ══════════════════════════════════════════════════════════════════
#     # Section Title
#     # ══════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3:  PHASE 1 — _backend.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Single source of truth for GPU detection, backend selection, FFT
#   wrappers, and small utility functions.  Imported by every other module
#   in the package.
#
# WHAT TO EXTRACT:
#   Copy the following sections from RL_Unknown_Boundary.py verbatim (they
#   are identical in both existing files), then add the set_backend()
#   function described below.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/_backend.py
# ─────────────────────────────────────────────────────────────────────────
#
# CONTENTS (in order):
#
#   1. Module docstring explaining purpose.
#
#   2. Imports: __future__ annotations, importlib, logging, numpy as np,
#      typing (Literal).
#
#   3. PaddingStr type alias:
#        PaddingStr = Literal["Reflect","Symmetric","Wrap","Edge","LinearRamp","Zero"]
#      (Moved here from RL file, since it is used by the base class.)
#
#   4. _USER_GPU_FLAG: bool = True
#      (Module-level default.)
#
#   5. _detect_gpu() -> bool
#      Exact copy of existing function.  Three-stage check:
#      a) _USER_GPU_FLAG
#      b) importlib.util.find_spec("cupy")
#      c) device count + live allocation test
#
#   6. Module-level state variables (mutable module globals):
#        _use_gpu: bool = _detect_gpu()
#        xp = ...   (cupy or numpy)
#        _fft = ... (cupy.fft or numpy.fft)
#
#   7. set_backend(mode: Literal["auto", "cpu", "gpu"]) -> None
#      NEW FUNCTION.  Allows runtime backend switching BEFORE any
#      algorithm object is constructed.
#
#      Specification:
#        - "auto": re-run _detect_gpu(), set _use_gpu/xp/_fft accordingly.
#        - "cpu":  force _use_gpu=False, xp=np, _fft=np.fft.
#        - "gpu":  attempt GPU; raise RuntimeError if unavailable.
#        - Must declare `global _use_gpu, xp, _fft` and reassign them.
#        - Log the resulting backend at INFO level.
#
#      IMPORTANT: This function modifies module-level globals.  It MUST be
#      called before constructing any DeconvBase subclass.  Calling it
#      after construction produces undefined behavior (existing objects
#      still hold references to the old xp).  Document this clearly.
#
#   8. FFT helpers (exact copies from existing code):
#        rfft2(a, **kwargs) -> xp.ndarray
#        irfft2(a, s, **kwargs) -> xp.ndarray
#        ifftshift  (alias of _fft.ifftshift)
#
#   9. Utility helpers (exact copies):
#        _freeze(a) -> xp.ndarray
#        _to_numpy(x) -> np.ndarray
#
# EXPORTS (all public):
#   PaddingStr, xp, _fft, _use_gpu, rfft2, irfft2, ifftshift,
#   _freeze, _to_numpy, set_backend
#
# ─── VERIFICATION (Phase 1) ────────────────────────────────────────────
#   Write a test script that:
#   1. `from Reconstruction._backend import xp, rfft2, irfft2, _to_numpy`
#   2. Creates a random 64×64 float32 array with xp.
#   3. Verifies rfft2 → irfft2 round-trip recovers the original (allclose).
#   4. Verifies _to_numpy returns a numpy array.
#   5. Calls set_backend("cpu"), verifies xp is numpy.
#   Print PASS/FAIL for each check.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4:  PHASE 2 — _tv_operators.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   All Total Variation related operators in one place.  Used by RL
#   (multiplicative correction), Landweber (Chambolle prox), ADMM
#   (shrinkage step), and TVAL3.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/_tv_operators.py
# ─────────────────────────────────────────────────────────────────────────
#
# IMPORTS:
#   from ._backend import xp
#
# CONTENTS (in order):
#
#   1. forward_grad(x) -> tuple[xp.ndarray, xp.ndarray]
#      Discrete gradient with forward differences, Neumann BC.
#      - dh[:-1, :] = x[1:, :] - x[:-1, :]   (last row = 0)
#      - dw[:, :-1] = x[:, 1:] - x[:, :-1]   (last col = 0)
#      Returns (dh, dw).
#
#      EXTRACTED FROM: Both existing files have this inline.
#      The Landweber file already has it as _forward_grad().
#      CHANGE: Remove leading underscore — this is a package-internal
#      public function (other modules in the package import it).
#
#   2. backward_div(p_h, p_w) -> xp.ndarray
#      Discrete divergence, backward differences, adjoint of −∇.
#      Implementation:
#        div[0, :]    =  p_h[0, :]
#        div[1:-1, :] =  p_h[1:-1, :] - p_h[:-2, :]
#        div[-1, :]   = -p_h[-2, :]
#        div[:, 0]    += p_w[:, 0]
#        div[:, 1:-1] += p_w[:, 1:-1] - p_w[:, :-2]
#        div[:, -1]   += -p_w[:, -2]
#
#      EXTRACTED FROM: Landweber _backward_div().
#      Same change: remove leading underscore.
#
#   3. tv_multiplicative_correction(x, lambda_tv, eps_grad=1e-8) -> xp.ndarray
#      The Dey et al. correction factor: C(x) = 1 − λ · div(∇x / |∇x|_ε).
#      Clamped to [0.5, +∞).
#
#      IMPLEMENTATION: Must now use forward_grad() and backward_div() from
#      this same module instead of inline code.  The logic is:
#        dh, dw = forward_grad(x)
#        mag = sqrt(dh² + dw² + eps²)
#        nh, nw = dh/mag, dw/mag
#        div = backward_div(nh, nw)
#        correction = clip(1 - lambda_tv * div, min=0.5)
#
#      Preserve the full docstring from the existing RL file (Dey et al.
#      reference, operator description, parameter docs, notes on clamping).
#
#   4. prox_tv_chambolle(v, gamma, n_inner=50, tau_dual=0.125) -> xp.ndarray
#      Proximal operator of γ·TV(·) via Chambolle 2004 dual projection.
#
#      Solves:  argmin_u  (1/2)||u − v||² + γ · TV(u)
#
#      Algorithm (iterate n_inner times):
#        div_p = backward_div(p_h, p_w)
#        arg = div_p - v / γ
#        g_h, g_w = forward_grad(arg)
#        p_h_new = p_h + τ_d · g_h
#        p_w_new = p_w + τ_d · g_w
#        mag = sqrt(p_h_new² + p_w_new²)
#        mag = maximum(mag, 1.0)
#        p_h, p_w = p_h_new / mag,  p_w_new / mag
#
#      Final recovery:  u* = v − γ · backward_div(p_h, p_w)
#
#      If gamma <= 0: return v.copy() immediately.
#
#      tau_dual MUST be ≤ 1/8 for convergence guarantee (||∇||²_op = 8
#      for 2-D discrete forward differences).
#
#      EXTRACTED FROM: Landweber file's _prox_tv_chambolle().
#      Preserve the full docstring (dual formulation, step size analysis,
#      Chambolle 2004 reference).
#
# EXPORTS:
#   forward_grad, backward_div, tv_multiplicative_correction,
#   prox_tv_chambolle
#
# ─── VERIFICATION (Phase 2) ────────────────────────────────────────────
#   Write a test script that:
#   1. Adjointness test:
#      Generate random x (64×64) and random p_h, p_w (64×64).
#      Compute: lhs = sum(-forward_grad(x)[0]*p_h + -forward_grad(x)[1]*p_w)
#               rhs = sum(x * backward_div(p_h, p_w))
#      Verify |lhs - rhs| < 1e-5.
#      (This confirms the adjoint pairing ⟨−∇x, p⟩ = ⟨x, div(p)⟩.)
#
#   2. TV prox identity test:
#      For gamma=0, verify prox_tv_chambolle(v, 0) == v.
#
#   3. TV prox denoising test:
#      Create a 64×64 constant image (value=0.5) + Gaussian noise σ=0.05.
#      Apply prox_tv_chambolle(noisy, gamma=0.05, n_inner=100).
#      Verify the result's standard deviation is significantly less than
#      the input's (at least 50% reduction).
#
#   4. Multiplicative correction shape test:
#      Create a 64×64 random image.
#      Verify tv_multiplicative_correction(x, 0.001).shape == x.shape.
#      Verify all values >= 0.5 (the clamp).
#
#   Print PASS/FAIL for each check.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5:  PHASE 3 — _base.py  (DeconvBase abstract class)
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Abstract base class that owns the entire forward-model setup:
#   image preprocessing, canvas construction, padding, mask, PSF
#   conditioning, FFT precomputation, HTM.  Subclasses only implement
#   the deblur() iteration loop.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/_base.py
# ─────────────────────────────────────────────────────────────────────────
#
# IMPORTS:
#   from abc import ABC, abstractmethod
#   import logging, numpy as np
#   from typing import Optional
#   from ._backend import (xp, rfft2, irfft2, ifftshift, _freeze,
#                           _to_numpy, _use_gpu, PaddingStr)
#   from Shared.Common.General_Utilities   import padding, cropping
#   from Shared.Common.PSF_Preprocessing  import psf_preprocess, condition_psf
#   from Shared.Common.Image_Preprocessing import (image_normalization,
#       validate_image, to_grayscale, odd_crop_around_center)
#
# CLASS: DeconvBase(ABC)
#
# ── CONSTRUCTOR SIGNATURE ──────────────────────────────────────────────
#
#   def __init__(
#       self,
#       image: np.ndarray,
#       psf: np.ndarray,
#       paddingMode: PaddingStr = "Reflect",
#       padding_scale: float = 2.0,
#       initialEstimate: Optional[np.ndarray] = None,
#       apply_taper_on_padding_band: bool = False,
#       htm_floor_frac: float = 0.05,
#       use_mask: bool = True,
#   ) -> None:
#
# ── CONSTRUCTOR BODY (step by step) ────────────────────────────────────
#
#   Step 1: validate_image(image)
#
#   Step 2: gray = to_grayscale(image)
#
#   Step 3: Enforce odd spatial dimensions.
#     H, W = gray.shape
#     OH = H if H % 2 == 1 else H - 1
#     OW = W if W % 2 == 1 else W - 1
#     Raise ValueError if OH <= 0 or OW <= 0.
#     If (OH, OW) != (H, W): gray = odd_crop_around_center(gray, (OH, OW))
#
#   Step 4: Normalize to [0,1].
#     gray = image_normalization(image=gray, bit_depth=1, is_int=False)
#
#   Step 5: Store original size (SINGLE assignment).
#     self.h, self.w = gray.shape
#
#   Step 6: Compute FFT canvas size.
#     pH, pW = psf.shape
#     fH = int(self.h + padding_scale * pH)
#     fW = int(self.w + padding_scale * pW)
#     OH_full = fH if fH % 2 == 1 else fH + 1
#     OW_full = fW if fW % 2 == 1 else fW + 1
#     self.full_shape: tuple[int, int] = (OH_full, OW_full)
#     logger.debug("Image shape %s → padded canvas %s", gray.shape, self.full_shape)
#
#   Step 7: GPU warm-up (if _use_gpu).
#     _dummy = xp.zeros(self.full_shape, dtype=xp.float32)
#     _ = rfft2(_dummy); del _dummy
#
#   Step 8: Pad observed image onto canvas.
#     self.image = xp.array(
#         padding(image=gray, full_size=self.full_shape,
#                 Type=paddingMode, apply_taper=bool(apply_taper_on_padding_band)),
#         dtype=xp.float32)
#
#   Step 9: Build mask M.
#     IMPORTANT: Document centering assumption (padding() centres via
#     integer division off = (canvas - image) // 2).
#
#     if use_mask:
#         self.mask = xp.zeros(self.full_shape, dtype=xp.float32)
#         off_y = (self.full_shape[0] - self.h) // 2
#         off_x = (self.full_shape[1] - self.w) // 2
#         self.mask[off_y:off_y+self.h, off_x:off_x+self.w] = 1.0
#     else:
#         self.mask = xp.ones(self.full_shape, dtype=xp.float32)
#
#     Store self.use_mask = use_mask for subclass inspection.
#
#   Step 10: PSF frequency-domain preparation.
#     a) psf_np = psf_preprocess(psf, center_method="com",
#            remove_negatives="clip", eps=1e-12, enforce_odd_shape=True)
#     b) psf_np = condition_psf(psf_np, bg_ring_frac=0.15,
#            taper_outer_frac=0.20, taper_end_frac=0.50)
#     c) Zero-pad to full_shape: padding(..., Type="Zero", apply_taper=False)
#     d) ifftshift (move centre to [0,0]).
#     e) self.PF = _freeze(rfft2(psf_pad))
#        self.conjPF = _freeze(self.PF.conj())
#
#   Step 11: Precompute H^T M with relative floor clamp.
#     htm_raw = irfft2(self.conjPF * rfft2(self.mask), s=fshape).astype(xp.float32)
#     htm_max = float(xp.max(htm_raw))
#     htm_floor = max(htm_floor_frac * htm_max, 1e-12)
#     xp.clip(htm_raw, a_min=htm_floor, a_max=None, out=htm_raw)
#     self.HTM = _freeze(htm_raw)
#     logger.debug("HTM: max=%.4f, floor=%.4f (%.1f%%)", ...)
#
#   Step 12: Lipschitz constant L = max(|H(f)|²).
#     self._lipschitz = float(xp.max(xp.abs(self.PF) ** 2))
#     logger.debug("Lipschitz constant L = %.6f", self._lipschitz)
#
#   Step 13: Initial estimate on padded canvas.
#     init_source = initialEstimate if initialEstimate is not None else gray
#     self.estimated_image = xp.array(
#         padding(init_source, self.full_shape, Type=paddingMode,
#                 apply_taper=bool(apply_taper_on_padding_band)),
#         dtype=xp.float32)
#     xp.maximum(self.estimated_image, xp.float32(1e-8), out=self.estimated_image)
#
# ── ABSTRACT METHOD ────────────────────────────────────────────────────
#
#   @abstractmethod
#   def deblur(self, **kwargs) -> np.ndarray:
#       """Run the algorithm.  Returns cropped CPU array."""
#       ...
#
# ── SHARED HELPER METHODS ──────────────────────────────────────────────
#
#   def _crop_and_return(self, x_k: xp.ndarray) -> np.ndarray:
#       """Store final state, crop to (self.h, self.w), move to CPU."""
#       self.estimated_image = x_k.copy()
#       return _to_numpy(cropping(x_k, (self.h, self.w)))
#
#   def _check_convergence(
#       self,
#       x_new: xp.ndarray,
#       x_old: xp.ndarray,
#       k: int,
#       num_iter: int,
#       tol: float,
#       eps: float = 1e-8,
#   ) -> tuple[float, bool]:
#       """
#       Compute relative change and check against tolerance.
#       Returns (rel_change, converged).
#       Logs info on convergence.
#       """
#       den = xp.linalg.norm(x_new)
#       den = den if float(den) > 0.0 else xp.float32(eps)
#       rel_chg = float(xp.linalg.norm(x_new - x_old) / den)
#       converged = rel_chg < tol
#       if converged:
#           logger.info("Converged at iteration %d/%d (rel_change=%.2e < tol=%.2e)",
#                       k + 1, num_iter, rel_chg, tol)
#       return rel_chg, converged
#
#   def _log_no_convergence(self, num_iter: int, tol: float) -> None:
#       """Log that max iterations reached without convergence."""
#       logger.info("Reached max iterations (%d) without convergence (tol=%.2e).",
#                   num_iter, tol)
#
# ── CONVENIENCE WRAPPER FACTORY (staticmethod) ─────────────────────────
#
#   There is no factory in the base class.  Each subclass module defines
#   its own wrapper function.  However, the base class documents the
#   standard init_keys pattern for splitting **kwargs:
#
#   INIT_PARAM_NAMES: The set of parameter names accepted by __init__
#   (excluding 'self', 'image', 'psf').  Subclass wrappers use this
#   to split kwargs:
#     _INIT_KEYS = {"paddingMode", "padding_scale", "initialEstimate",
#                   "apply_taper_on_padding_band", "htm_floor_frac", "use_mask"}
#   Define this as a class attribute on DeconvBase.
#
# EXPORTS:
#   DeconvBase
#
# ─── VERIFICATION (Phase 3) ────────────────────────────────────────────
#   Write a minimal concrete subclass for testing:
#
#     class _TestDeconv(DeconvBase):
#         def deblur(self, **kwargs):
#             return self._crop_and_return(self.estimated_image)
#
#   Test script:
#   1. Create a synthetic 64×64 image and 11×11 Gaussian PSF.
#   2. Construct _TestDeconv(image, psf).
#   3. Verify self.full_shape is odd in both dims.
#   4. Verify self.mask has the correct number of ones (h × w).
#   5. Verify self.PF.shape == (full_shape[0], full_shape[1]//2+1).
#   6. Verify self.HTM.min() >= 0.05 * self.HTM.max() (floor clamp).
#   7. Verify deblur() returns a numpy array of shape (h, w).
#   8. Construct with use_mask=False, verify mask is all ones.
#   Print PASS/FAIL.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6:  PHASE 4a — rl_unknown_boundary.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Thin subclass that contains ONLY the RL iteration loop.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/rl_unknown_boundary.py
# ─────────────────────────────────────────────────────────────────────────
#
# IMPORTS:
#   from ._backend import xp, rfft2, irfft2
#   from ._base import DeconvBase
#   from ._tv_operators import tv_multiplicative_correction
#   import numpy as np, logging
#
# CLASS: RLUnknownBoundary(DeconvBase)
#
#   Class docstring: Copy from existing file (masked RL + Dey TV).
#
#   __init__: NO OVERRIDE.  Inherits from DeconvBase unchanged.
#   (If in the future RL needs extra constructor logic, it can call
#   super().__init__(...) and then add its own setup.)
#
#   def deblur(
#       self,
#       num_iter: int = 100,
#       lambda_tv: float = 0.0002,
#       tol: float = 1e-6,
#       min_iter: int = 5,
#       check_every: int = 5,
#       epsilon_division: float = 1e-12,       # FIXED typo from original
#       epsilon_positivity: float = 1e-8,
#       tv_on_full_canvas: bool = True,
#   ) -> np.ndarray:
#
#   ALGORITHM (per iteration k):
#
#     1. Forward model:   Hx_k = irfft2(PF * rfft2(x_k), s=fshape)
#
#     2. Masked ratio:    ratio = (M * y) / ((Hx_k * M) + (1 - M) + eps_div)
#
#        NOTE on the ratio formula:
#        Inside Ω (M=1):  ratio = y / (Hx_k + eps_div)    ← standard RL ratio
#        Outside Ω (M=0): ratio = 0 / (1 + eps_div) ≈ 0   ← no data contribution
#        This cleanly zeroes the residual outside the observed region.
#
#     3. Backprojection:  back = irfft2(conjPF * rfft2(ratio), s=fshape)
#
#     4. RL update:       x_new = x_k * (back / (HTM + eps_div))
#
#     5. TV correction (if lambda_tv > 0):
#        if tv_on_full_canvas:
#            correction = tv_multiplicative_correction(x_k, lam)
#            x_new /= correction
#        else:
#            correction = tv_multiplicative_correction(x_k, lam)
#            x_new = x_new / (1 + (correction - 1) * M)
#
#        Document the boundary artifact warning for tv_on_full_canvas=False
#        in the docstring (same warning text as the corrected version).
#
#     6. Positivity:      xp.maximum(x_new, eps_pos, out=x_new)
#
#     7. Convergence:     Use self._check_convergence(x_new, x_k, k, ...)
#
#     8. Advance:         x_k = x_new
#
#   After loop: use for/else to call self._log_no_convergence if no break.
#   Return self._crop_and_return(x_k).
#
# WRAPPER FUNCTION:
#
#   def rl_deblur_unknown_boundary(
#       image: np.ndarray,
#       psf: np.ndarray,
#       iters: int = 100,
#       lambda_tv: float = 0.0002,
#       **kwargs,
#   ) -> np.ndarray:
#       init_kw = {k: v for k, v in kwargs.items() if k in DeconvBase._INIT_KEYS}
#       deblur_kw = {k: v for k, v in kwargs.items() if k not in DeconvBase._INIT_KEYS}
#       obj = RLUnknownBoundary(image=image, psf=psf, **init_kw)
#       return obj.deblur(num_iter=iters, lambda_tv=lambda_tv, **deblur_kw)
#
# EXPORTS:
#   RLUnknownBoundary, rl_deblur_unknown_boundary
#
# ─── VERIFICATION (Phase 4a) ───────────────────────────────────────────
#   Numerical regression test:
#   1. Create a 64×64 test image (e.g., a Shepp-Logan phantom or a
#      simple pattern: central 32×32 block at value 0.8, background 0.2).
#   2. Create an 11×11 Gaussian PSF (σ=2.0), normalised to sum=1.
#   3. Blur the image: blurred = convolve(image, psf) using scipy or fft.
#   4. Run the OLD standalone RL: old_result = old_rl.deblur(num_iter=50)
#   5. Run the NEW package RL:    new_result = new_rl.deblur(num_iter=50)
#   6. Verify np.allclose(old_result, new_result, atol=1e-5).
#      If not allclose, print the max absolute difference.
#   This confirms the refactor is behaviour-preserving.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7:  PHASE 4b — landweber_unknown_boundary.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Thin subclass with the FISTA + proximal TV loop.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/landweber_unknown_boundary.py
# ─────────────────────────────────────────────────────────────────────────
#
# IMPORTS:
#   from ._backend import xp, rfft2, irfft2
#   from ._base import DeconvBase
#   from ._tv_operators import prox_tv_chambolle
#   import numpy as np, logging
#
# CLASS: LandweberUnknownBoundary(DeconvBase)
#
#   __init__: NO OVERRIDE.
#
#   def deblur(
#       self,
#       num_iter: int = 200,
#       lambda_tv: float = 0.001,
#       tol: float = 1e-6,
#       min_iter: int = 10,
#       check_every: int = 5,
#       step_size: Optional[float] = None,
#       enforce_positivity: bool = True,
#       epsilon_positivity: float = 1e-8,
#       precondition: bool = True,
#       tv_n_inner: int = 50,
#       adaptive_restart: bool = True,
#   ) -> np.ndarray:
#
#   ALGORITHM (per iteration k):
#
#     Step size selection:
#       if step_size given: tau = step_size
#       elif precondition:  tau = 0.95
#       else:               tau = 0.95 / L
#
#     Proximal TV parameter:
#       if precondition: gamma = tau * lam / median(HTM[M > 0.5])
#       else:            gamma = tau * lam
#
#     FISTA state: x_k, x_km1, z_k, t_k=1.0
#
#     Per iteration:
#       1. Gradient at z_k:  grad = H^T[M(Hz_k − y)]
#            Hz = irfft2(PF * rfft2(z_k), s=fshape)
#            residual = M * (Hz - y)
#            grad = irfft2(conjPF * rfft2(residual), s=fshape)
#
#       2. Gradient step:
#            if precondition: x_half = z_k - tau * (grad / HTM)
#            else:            x_half = z_k - tau * grad
#
#       3. Proximal TV:
#            x_new = prox_tv_chambolle(x_half, gamma, n_inner=tv_n_inner)
#            (skip if lambda_tv <= 0)
#
#       4. Positivity projection (if enforce_positivity).
#
#       5. FISTA momentum:
#            t_new = (1 + sqrt(1 + 4*t_k²)) / 2
#            momentum = (t_k - 1) / t_new
#
#       6. Adaptive restart (O'Donoghue & Candès 2015, velocity restart):
#            if k > 0:
#              ip = sum((x_new - x_k) * (x_k - x_km1))
#              if ip < 0:  t_new=1, momentum=0
#
#       7. Extrapolation:  z_new = x_new + momentum * (x_new - x_k)
#
#       8. Convergence: self._check_convergence(...)
#
#       9. Advance: x_km1=x_k, x_k=x_new, z_k=z_new, t_k=t_new
#
#     After loop: for/else with self._log_no_convergence.
#     Return self._crop_and_return(x_k).
#
# WRAPPER: landweber_deblur_unknown_boundary(image, psf, iters=200, ...)
#   Same kwargs-splitting pattern as RL.
#
# ─── VERIFICATION (Phase 4b) ───────────────────────────────────────────
#   Same regression test structure as Phase 4a:
#   1. Use same blurred test image.
#   2. Compare old standalone Landweber output vs new package output.
#   3. Verify np.allclose(old, new, atol=1e-5).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8:  PHASE 5a — wiener.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Classical Wiener deconvolution (non-iterative, frequency-domain).
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/wiener.py
# ─────────────────────────────────────────────────────────────────────────
#
# CLASS: WienerDeconv(DeconvBase)
#
#   DEFAULT OVERRIDES in __init__:
#     Calls super().__init__(..., use_mask=False) — Wiener does not
#     support masked data fidelity in its classical formulation.
#
#   def deblur(
#       self,
#       snr: Optional[float] = None,
#       noise_power: Optional[float] = None,
#       regularization: float = 0.01,
#   ) -> np.ndarray:
#
#   ALGORITHM:
#     The Wiener filter in the frequency domain is:
#
#       X̂(f) = [ H*(f) / (|H(f)|² + 1/SNR) ] · Y(f)
#
#     Or equivalently with a regularization parameter λ:
#
#       X̂(f) = [ H*(f) / (|H(f)|² + λ) ] · Y(f)
#
#     Parameter precedence:
#       - If snr is given:         λ = 1 / snr
#       - If noise_power is given: λ = noise_power / signal_power_estimate
#       - Otherwise:               λ = regularization (direct)
#
#     Implementation:
#       Y_f = rfft2(self.image)
#       H_sq = xp.abs(self.PF) ** 2          # |H(f)|²
#       wiener = self.conjPF / (H_sq + lam)  # H*(f) / (|H(f)|² + λ)
#       result = irfft2(wiener * Y_f, s=self.full_shape)
#       return self._crop_and_return(result)
#
#     NOTE: No iteration loop.  No positivity enforcement (Wiener can
#     produce negative values; this is normal and expected).
#     No TV regularization (not applicable to non-iterative Wiener).
#
#   DOCSTRING should note:
#     - Assumes stationary Gaussian noise.
#     - λ controls the noise-resolution trade-off: larger λ → smoother
#       result, lower noise amplification, more blur residual.
#     - For satellite imagery, typical SNR range: 10–1000 (λ = 0.1–0.001).
#     - This is a fast baseline — useful for initial inspection and as
#       an initial estimate for iterative methods.
#
# WRAPPER: wiener_deblur(image, psf, snr=None, ...)
#
# ─── VERIFICATION (Phase 5a) ───────────────────────────────────────────
#   1. Blur a test image with known PSF, no noise.
#   2. Apply Wiener with very small λ (e.g. 1e-10).
#   3. Verify reconstruction is close to original (PSNR > 40 dB).
#   4. Add Gaussian noise (σ=0.01), apply Wiener with snr=100.
#   5. Verify reconstruction PSNR > 25 dB (reasonable for SNR=100).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9:  PHASE 5b — rl_standard.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Classical Richardson-Lucy WITHOUT unknown-boundary (mask) treatment.
#   This is the simplest RL: operates on the full padded canvas with
#   M = 1 everywhere.  Useful as a baseline and for cases where the
#   image boundaries are known/trusted.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/rl_standard.py
# ─────────────────────────────────────────────────────────────────────────
#
# CLASS: RLStandard(DeconvBase)
#
#   DEFAULT OVERRIDES in __init__:
#     super().__init__(..., use_mask=False)
#
#   def deblur(
#       self,
#       num_iter: int = 100,
#       lambda_tv: float = 0.0,
#       tol: float = 1e-6,
#       min_iter: int = 5,
#       check_every: int = 5,
#       epsilon_division: float = 1e-12,
#       epsilon_positivity: float = 1e-8,
#   ) -> np.ndarray:
#
#   ALGORITHM (standard RL, no mask complexity):
#     Per iteration:
#       1. Hx_k = irfft2(PF * rfft2(x_k), s=fshape)
#       2. ratio = y / (Hx_k + eps_div)
#       3. back = irfft2(conjPF * rfft2(ratio), s=fshape)
#       4. x_new = x_k * back
#          (No HTM division needed since M=1 → H^T M = H^T 1 ≈ 1.)
#          Actually: for normalized PSF, H^T 1 = 1 exactly, so no division.
#       5. TV correction (optional, same Dey et al. as masked version).
#       6. Positivity.
#       7. Convergence check.
#
#   NOTE: This is simpler than the masked version.  Step 2 has no mask
#   multiplication, and step 4 has no HTM normalization.
#
# WRAPPER: rl_deblur_standard(image, psf, iters=100, ...)
#
# ─── VERIFICATION (Phase 5b) ───────────────────────────────────────────
#   1. Blur test image, add Poisson-like noise.
#   2. Run RLStandard for 50 iterations.
#   3. Verify output PSNR > Wiener baseline PSNR (RL should improve
#      over Wiener for Poisson noise at moderate SNR).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10:  PHASE 5c — admm.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   ADMM (Alternating Direction Method of Multipliers) for TV-regularized
#   deconvolution with unknown boundaries.
#
# MATHEMATICAL FORMULATION:
#   We solve:
#     min_x  (1/2) ||M(Hx − y)||² + λ TV(x)
#
#   Introduce auxiliary variable z = ∇x and write as:
#     min_{x,z}  (1/2) ||M(Hx − y)||² + λ ||z||_{2,1}
#     subject to  z = ∇x
#
#   Where ||z||_{2,1} = Σ_{i,j} sqrt(z_h² + z_w²)  (isotropic TV).
#
#   Augmented Lagrangian with penalty ρ:
#     L(x, z, u) = (1/2)||M(Hx-y)||² + λ||z||_{2,1}
#                  + (ρ/2)||∇x - z + u||² - (ρ/2)||u||²
#
#   ADMM iterates:
#     x-update:  x_{k+1} = argmin_x (1/2)||M(Hx-y)||² + (ρ/2)||∇x - z_k + u_k||²
#     z-update:  z_{k+1} = prox_{(λ/ρ) || · ||_{2,1}}(∇x_{k+1} + u_k)
#     u-update:  u_{k+1} = u_k + ∇x_{k+1} - z_{k+1}
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/admm.py
# ─────────────────────────────────────────────────────────────────────────
#
# IMPORTS:
#   from ._backend import xp, rfft2, irfft2
#   from ._base import DeconvBase
#   from ._tv_operators import forward_grad, backward_div
#
# CLASS: ADMMDeconv(DeconvBase)
#
#   __init__: NO OVERRIDE (inherits mask support from base).
#
#   def deblur(
#       self,
#       num_iter: int = 100,
#       lambda_tv: float = 0.001,
#       rho: float = 1.0,
#       tol: float = 1e-6,
#       min_iter: int = 5,
#       check_every: int = 5,
#       enforce_positivity: bool = True,
#       epsilon_positivity: float = 1e-8,
#   ) -> np.ndarray:
#
#   ALGORITHM:
#
#   PRECOMPUTATION:
#     The x-update requires solving a linear system.  In the frequency
#     domain (using the masked formulation), this becomes:
#
#       x_{k+1} = F^{-1} [ (H* · F[M · (M·Hx_{rhs})] + ρ · F[div(z_k - u_k)])
#                           / (|H|² · F[M] * F[M] + ρ · D_lap) ]
#
#     HOWEVER, the mask M makes this non-diagonal in the frequency domain
#     (M is not shift-invariant).  The standard workaround is to use the
#     LINEARIZED ADMM (also called split Bregman) approach:
#
#     Replace the exact x-minimization with a single gradient step:
#       gradient of data fidelity: H^T[M(Hx_k - y)]
#       gradient of augmented term: -ρ · div(∇x_k - z_k + u_k)
#       x_{k+1} = x_k - τ [H^T[M(Hx_k - y)] - ρ · div(∇x_k - z_k + u_k)]
#
#     Where τ is a step size (use τ = 1 / (L + ρ · 8), where
#     L = Lipschitz of data fidelity, 8 = ||∇||²_op for 2D).
#
#   z-UPDATE (vectorial soft-thresholding / shrinkage):
#     v_h, v_w = forward_grad(x_{k+1}) + u_h_k, u_w_k
#     mag = sqrt(v_h² + v_w²)
#     shrink = max(mag - λ/ρ, 0) / max(mag, eps)
#     z_h_{k+1} = shrink * v_h
#     z_w_{k+1} = shrink * v_w
#
#     This is the proximal operator of (λ/ρ)||·||_{2,1}, applied
#     pointwise to the 2-vector (v_h, v_w) at each pixel.
#
#   u-UPDATE (dual variable / scaled residual):
#     dx_h, dx_w = forward_grad(x_{k+1})
#     u_h_{k+1} = u_h_k + dx_h - z_h_{k+1}
#     u_w_{k+1} = u_w_k + dx_w - z_w_{k+1}
#
#   CONVERGENCE: Same self._check_convergence on x_new vs x_k.
#
#   IMPORTANT NOTES for implementation:
#   - z and u are 2-component vector fields: store as (z_h, z_w) and
#     (u_h, u_w), each shape (H, W).  Initialize all to zero.
#   - Positivity can be enforced after the x-update (it's a projection
#     onto a convex set, compatible with ADMM).
#   - ρ controls convergence speed vs. accuracy of the splitting:
#     too small → slow, too large → noisy.  ρ = 1.0 is a reasonable
#     default; consider adaptive ρ (Boyd et al. 2011) for production.
#
#   REFERENCES:
#   [1] S. Boyd et al., "Distributed Optimization and Statistical Learning
#       via the Alternating Direction Method of Multipliers," Found. and
#       Trends in Machine Learning, 3(1):1–122, 2011.
#
# WRAPPER: admm_deblur(image, psf, iters=100, ...)
#
# ─── VERIFICATION (Phase 5c) ───────────────────────────────────────────
#   1. Same blurred test image.
#   2. Run ADMM for 100 iterations with lambda_tv=0.001.
#   3. Verify output PSNR is reasonable (> 20 dB).
#   4. Verify PSNR improves over Wiener baseline.
#   5. Verify z and u residuals decrease (primal/dual residual check).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11:  PHASE 5d — tval3.py
# ═════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   TVAL3 — TV minimization by Augmented Lagrangian and Alternating
#   Direction Algorithm.  Based on C. Li, W. Yin, H. Jiang, Y. Zhang,
#   "An efficient augmented Lagrangian method with applications to total
#   variation minimization," Computational Optimization and Applications,
#   56(3):507–530, 2013.
#
# MATHEMATICAL FORMULATION:
#   Same problem as ADMM:
#     min_x  (1/2) ||M(Hx − y)||² + λ ||∇x||_{2,1}
#
#   TVAL3 uses an augmented Lagrangian formulation with continuation
#   (increasing penalty parameter β over outer iterations) and an
#   alternating minimization inner loop.
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/tval3.py
# ─────────────────────────────────────────────────────────────────────────
#
# CLASS: TVAL3Deconv(DeconvBase)
#
#   def deblur(
#       self,
#       num_iter: int = 200,
#       lambda_tv: float = 0.001,
#       beta_init: float = 1.0,
#       beta_max: float = 1e5,
#       beta_rate: float = 2.0,
#       inner_iter: int = 5,
#       tol: float = 1e-6,
#       min_iter: int = 10,
#       check_every: int = 5,
#       enforce_positivity: bool = True,
#       epsilon_positivity: float = 1e-8,
#   ) -> np.ndarray:
#
#   ALGORITHM:
#
#   The TVAL3 algorithm alternates between:
#
#   OUTER LOOP (continuation on β):
#     β starts at beta_init and is multiplied by beta_rate each outer
#     iteration until β > beta_max.  Continuation improves convergence
#     by solving a sequence of easier subproblems (small β → loose
#     constraint → fast convergence; large β → tight constraint → accuracy).
#
#   INNER LOOP (for each β, run inner_iter alternating steps):
#
#     w-subproblem (TV shrinkage, same as ADMM z-update):
#       v = ∇x + λ_dual / β
#       w = shrink(v, λ/β)
#       (Pointwise vectorial soft-thresholding.)
#
#     x-subproblem (gradient step on augmented Lagrangian):
#       The x-minimization of:
#         (1/2)||M(Hx-y)||² + (β/2)||∇x - w + λ_dual/β||²
#       is solved by a single linearized gradient step (same approach
#       as ADMM linearized x-update):
#         grad_data = H^T[M(Hx - y)]
#         grad_aug  = -β · div(∇x - w + λ_dual/β)
#         x = x - τ · (grad_data + grad_aug)
#       Step size: τ = 1 / (L + β · 8)
#
#     Dual update:
#       λ_dual = λ_dual + β · (∇x - w)
#
#   NOTE: λ_dual here is the Lagrange multiplier (a 2-component vector
#   field like z, u in ADMM), NOT the TV weight lambda_tv.  Use a
#   distinct variable name (e.g., mu_h, mu_w) to avoid confusion.
#
#   CONVERGENCE: Check on x between successive outer iterations.
#
#   REFERENCES:
#   [1] C. Li et al., "An efficient augmented Lagrangian method with
#       applications to total variation minimization," Comput. Optim.
#       Appl., 56(3):507–530, 2013.
#
# WRAPPER: tval3_deblur(image, psf, iters=200, ...)
#
# ─── VERIFICATION (Phase 5d) ───────────────────────────────────────────
#   1. Same blurred test image.
#   2. Run TVAL3 for 200 outer iterations.
#   3. Verify output PSNR > 20 dB.
#   4. Verify that β increases monotonically (log the continuation).
#   5. Verify PSNR comparable to ADMM (±2 dB).


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12:  PHASE 6 — __init__.py  (Public API)
# ═════════════════════════════════════════════════════════════════════════════
#
# ─────────────────────────────────────────────────────────────────────────
# FILE: Reconstruction/__init__.py
# ─────────────────────────────────────────────────────────────────────────
#
# """
# Reconstruction — modular deconvolution algorithms for satellite imagery.
#
# All algorithms share a common base class (DeconvBase) that handles
# image preprocessing, PSF conditioning, FFT setup, and unknown-boundary
# mask construction.  Individual algorithms differ only in their
# iteration strategy.
#
# Quick start (one-shot wrappers):
#     from Reconstruction import rl_deblur_unknown_boundary
#     result = rl_deblur_unknown_boundary(image, psf, iters=100)
#
# Class-based (fine-grained control):
#     from Reconstruction import RLUnknownBoundary
#     rl = RLUnknownBoundary(image, psf)
#     result_50  = rl.deblur(num_iter=50)
#     result_100 = rl.deblur(num_iter=50)   # continues from iteration 50
#
# Backend control:
#     from Reconstruction import set_backend
#     set_backend("cpu")   # Force CPU even if GPU available
# """
#
# from ._backend import set_backend
#
# from .wiener import WienerDeconv, wiener_deblur
# from .rl_standard import RLStandard, rl_deblur_standard
# from .rl_unknown_boundary import RLUnknownBoundary, rl_deblur_unknown_boundary
# from .landweber_unknown_boundary import (
#     LandweberUnknownBoundary, landweber_deblur_unknown_boundary,
# )
# from .admm import ADMMDeconv, admm_deblur
# from .tval3 import TVAL3Deconv, tval3_deblur
#
# __all__ = [
#     "set_backend",
#     "WienerDeconv", "wiener_deblur",
#     "RLStandard", "rl_deblur_standard",
#     "RLUnknownBoundary", "rl_deblur_unknown_boundary",
#     "LandweberUnknownBoundary", "landweber_deblur_unknown_boundary",
#     "ADMMDeconv", "admm_deblur",
#     "TVAL3Deconv", "tval3_deblur",
# ]
#
# ─── VERIFICATION (Phase 6) ────────────────────────────────────────────
#   1. `from Reconstruction import *` — no import errors.
#   2. Verify each algorithm class is accessible.
#   3. Verify each wrapper function is accessible.
#   4. Verify set_backend("cpu") works.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13:  PHASE 7 — INTEGRATION & CLEANUP
# ═════════════════════════════════════════════════════════════════════════════
#
# This phase replaces the old standalone files and runs the full test suite.
#
# STEPS:
#
#   1. Run ALL verification scripts from Phases 1–6 in sequence.
#      ALL must pass before proceeding.
#
#   2. Run the regression tests (Phases 4a, 4b) one more time with
#      the final package structure.
#
#   3. Update any code that imports from the old standalone files
#      to import from the Reconstruction package instead.  Search for:
#        - "from RL_Unknown_Boundary import"
#        - "from Landweber_Unknown_Boundary import"
#        - "import RL_Unknown_Boundary"
#        - "import Landweber_Unknown_Boundary"
#
#   4. Move the old standalone files to a backup location (do NOT delete):
#        RL_Unknown_Boundary.py          → _archive/RL_Unknown_Boundary_standalone.py
#        Landweber_Unknown_Boundary.py   → _archive/Landweber_Unknown_Boundary_standalone.py
#
#   5. Final smoke test: construct each algorithm class, run 10 iterations,
#      verify no exceptions and output shape is correct.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14:  FUTURE EXTENSIONS  (not implemented now, but design for them)
# ═════════════════════════════════════════════════════════════════════════════
#
# The architecture should support these future additions without modifying
# any existing code:
#
# 14.1  NEW ALGORITHMS
#   Adding a new algorithm requires:
#     1. Create new_algorithm.py in Reconstruction/.
#     2. Define NewAlgorithm(DeconvBase) with deblur().
#     3. Add wrapper function.
#     4. Add imports to __init__.py.
#   No changes to _backend, _tv_operators, or _base needed.
#
# 14.2  NEW TV SOLVERS
#   If a faster TV solver is implemented (e.g., Chambolle-Pock primal-dual):
#     1. Add prox_tv_chambolle_pock() to _tv_operators.py.
#     2. Algorithms that use prox_tv can accept a callable parameter
#        `tv_solver` to switch between implementations.
#
# 14.3  MULTI-CHANNEL DECONVOLUTION
#   The current base class assumes grayscale.  For RGB/multispectral:
#     - Override grayscale conversion in a MultichannelDeconvBase subclass.
#     - Process channels independently (same PSF) or with per-channel PSFs.
#     - This is a base class extension, not an algorithm change.
#
# 14.4  NON-BLIND → BLIND UPGRADE
#   For blind deconvolution (joint estimation of image + PSF):
#     - Create a BlindDeconvBase that alternates between image update
#       (calling the existing deblur methods) and PSF update.
#     - Each existing algorithm becomes the "image step" in the blind loop.
#
# 14.5  CONVERGENCE MONITORING CALLBACK
#   Consider adding an optional callback parameter to deblur():
#     callback: Optional[Callable[[int, float, xp.ndarray], None]] = None
#   Called as callback(iteration, rel_change, x_k) at each check_every.
#   Enables external logging, plotting, or early stopping logic.
#   This can be added to DeconvBase._check_convergence without touching
#   any subclass code.


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15:  IMPLEMENTATION SUMMARY — QUICK REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
#
# PHASE  FILE                              DEPENDS ON       LINES (est.)
# ─────  ────────────────────────────────  ───────────────  ────────────
# 1      _backend.py                       (none)           ~140
# 2      _tv_operators.py                  _backend         ~200
# 3      _base.py                          _backend         ~200
# 4a     rl_unknown_boundary.py            _base, _tv_ops   ~130
# 4b     landweber_unknown_boundary.py     _base, _tv_ops   ~180
# 5a     wiener.py                         _base            ~80
# 5b     rl_standard.py                    _base, _tv_ops   ~100
# 5c     admm.py                           _base, _tv_ops   ~180
# 5d     tval3.py                          _base, _tv_ops   ~200
# 6      __init__.py                       all above        ~30
# ─────                                                     ──────
# TOTAL                                                     ~1440
#
# Compared to current state: ~1320 lines across 2 files with massive
# duplication.  The new structure is ~1440 lines across 10 files with
# ZERO duplication and 4 additional algorithms (Wiener, RL standard,
# ADMM, TVAL3).
#
# IMPLEMENTATION ORDER:
#   Phases 1 → 2 → 3 → 4a → 4b → (5a,5b,5c,5d in any order) → 6 → 7
#   Run verification at the end of EACH phase before proceeding.
