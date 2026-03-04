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

    # GPU → CPU (no-op on CPU)
    image_np = _to_numpy(image).astype(np.float64)
    image_np = np.clip(image_np, 0.0, 1.0)

    denoised_np = _bm3d_func(image_np, sigma_psd=sigma, profile=profile)
    denoised_np = np.clip(denoised_np, 0.0, 1.0)

    # CPU → GPU (no-op on CPU); match input dtype
    return xp.array(denoised_np, dtype=image.dtype)
