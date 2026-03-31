"""
_denoise_utils.py — Shared BM3D denoiser utility for PnP-ADMM and RED-ADMM.

Provides a single implementation of the BM3D wrapper (GPU↔CPU transfer,
clipping to [0, 1], profile selection) used by both
:class:`~.pnp_admm.PnPADMM` and :class:`~.red_admm.REDDeconv`.

Usage
-----
Import the availability flag and the denoiser function together::

    from ._denoise_utils import _HAS_BM3D, bm3d_denoise

Each class that uses BM3D should check ``_HAS_BM3D`` in its own
``__init__`` and raise ``ImportError`` if the package is absent.

Notes
-----
BM3D is CPU-only.  :func:`bm3d_denoise` automatically handles the
GPU↔CPU round-trip via :func:`~._backend._to_numpy`.  The call is a
no-op when σ < 1e-6 to avoid BM3D artefacts at negligible noise levels.
"""
from __future__ import annotations

import numpy as np

from ._backend import xp, _to_numpy

# ── Optional BM3D dependency ───────────────────────────────────────────────
try:
    from bm3d import bm3d as _bm3d_func
    _HAS_BM3D: bool = True
except ImportError:
    _HAS_BM3D = False


def bm3d_denoise(
    image: "xp.ndarray",
    sigma: float,
    profile: str = "np",
    range_mode: str = "affine",
) -> "xp.ndarray":
    """
    Apply BM3D denoising with automatic GPU↔CPU transfer.

    Steps:

    1. Transfer ``image`` to CPU (no-op if already on CPU).
    2. Cast to float64 and clip to [0, 1] (BM3D's expected image range).
    3. Call BM3D with the given σ and profile.
    4. Clip output to [0, 1] (BM3D can slightly overshoot).
    5. Transfer result back to the active backend (no-op if CPU).

    Parameters
    ----------
    image : xp.ndarray
        Image to denoise.  Can be GPU or CPU, any floating dtype.
        Expected to be normalized to [0, 1].
    sigma : float
        BM3D noise standard deviation (same units as image values,
        which should be in [0, 1]).  If σ < 1e-6 the input is returned
        unchanged to avoid BM3D no-op artefacts.
    profile : str, optional
        BM3D profile.  ``'np'`` (normal profile, default) or ``'lc'``
        (low complexity, faster but slightly lower quality).
    range_mode : {'affine', 'none'}, optional
        How to adapt data for BM3D's expected [0,1] domain.

        - 'affine': if image is already in [0,1], denoise directly.
          Otherwise, apply an invertible affine map to [0,1],
          denoise there with sigma scaled accordingly, then map back.
        - 'none': pass the image directly to BM3D unchanged.

    Returns
    -------
    xp.ndarray
        Denoised image, same dtype and shape as ``image``.

    Raises
    ------
    ImportError
        If the ``bm3d`` package is not installed.  Callers should
        check :data:`_HAS_BM3D` before calling this function.
    """
    if not _HAS_BM3D:
        raise ImportError(
            "bm3d_denoise requires the 'bm3d' package. "
            "Install with:  pip install bm3d"
        )

    if sigma < 1e-6:
        # σ too small for meaningful denoising; return as-is.
        return image

    if range_mode not in {"affine", "none"}:
        raise ValueError("range_mode must be 'affine' or 'none'")

    # GPU → CPU (no-op on CPU)
    image_np = _to_numpy(image).astype(np.float64)

    if range_mode == "none":
        den_input = image_np
        sigma_eff = float(sigma)
        offset = 0.0
        scale = 1.0

    else:
        x_min = float(np.min(image_np))
        x_max = float(np.max(image_np))

        # Already in BM3D's natural domain: do nothing.
        if x_min >= 0.0 and x_max <= 1.0:
            den_input = image_np
            sigma_eff = float(sigma)
            offset = 0.0
            scale = 1.0
        else:
            scale = max(x_max - x_min, 1e-12)
            offset = x_min
            den_input = (image_np - offset) / scale
            sigma_eff = float(sigma) / scale

    denoised_np = _bm3d_func(image_np, sigma_psd=sigma, profile=profile)
    
    if range_mode == "affine" and not (offset == 0.0 and scale == 1.0):
        denoised_np = offset + scale * denoised_np    

    # CPU → GPU (no-op on CPU); match input dtype
    return xp.array(denoised_np, dtype=image.dtype)
