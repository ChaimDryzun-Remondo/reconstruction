"""
_backend.py — GPU detection, backend selection, FFT helpers, and utilities.

Single source of truth for the Reconstruction package's compute backend.
All other modules in this package import ``xp``, ``_fft``, and the
FFT/utility helpers from here.

Usage
-----
Import symbols directly::

    from ._backend import xp, rfft2, irfft2, ifftshift, _freeze, _to_numpy

To switch backends before constructing any algorithm object::

    from ._backend import set_backend
    set_backend("cpu")   # force CPU
    set_backend("gpu")   # force GPU (raises if unavailable)
    set_backend("auto")  # auto-detect (default)

.. warning::
    ``set_backend()`` must be called **before** constructing any
    :class:`~Reconstruction._base.DeconvBase` subclass.  Calling it after
    construction produces undefined behaviour: existing objects still hold
    references to the old ``xp`` and will **not** be migrated.
"""
from __future__ import annotations

import importlib
import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ══════════════════════════════════════════════════════════════════════════════

PaddingStr = Literal["Reflect", "Symmetric", "Wrap", "Edge", "LinearRamp", "Zero"]

# ══════════════════════════════════════════════════════════════════════════════
# GPU Detection
# ══════════════════════════════════════════════════════════════════════════════

# Set to False to force CPU even when a GPU is available.
_USER_GPU_FLAG: bool = True


def _detect_gpu() -> bool:
    """
    Probe whether a functional CUDA device is reachable via CuPy.

    Performs a three-stage check to avoid false positives:

    1. **User flag** — ``_USER_GPU_FLAG`` must be ``True``.
    2. **Package presence** — is CuPy installed?
    3. **Device count + live allocation** — does the CUDA runtime see at
       least one device, and can we actually allocate and compute on it?

    Returns
    -------
    bool
        ``True`` only if all three stages succeed *and* ``_USER_GPU_FLAG``
        is ``True``.
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

        # Definitive test: touch GPU memory and run a kernel.
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


# ══════════════════════════════════════════════════════════════════════════════
# Backend Selection  (xp, _fft)
# ══════════════════════════════════════════════════════════════════════════════
# We maintain a single code path by aliasing either NumPy or CuPy as ``xp``
# and the corresponding FFT module as ``_fft``.  All downstream code uses
# ``xp`` for array creation and ``rfft2`` / ``irfft2`` for transforms, so
# switching between CPU and GPU requires no code changes.

_use_gpu: bool = _detect_gpu()

if _use_gpu:
    import cupy as cp
    xp = cp
    _fft = cp.fft
    try:
        # CuPy ≥ 12: pre-allocate FFT plan cache to avoid first-call latency.
        cp.fft.config.set_plan_cache_size(64)  # 64 plans ≈ 16–32 MiB
    except AttributeError:
        pass  # Older CuPy; harmless.
else:
    xp = np
    _fft = np.fft


def set_backend(mode: Literal["auto", "cpu", "gpu"]) -> None:
    """
    Switch the compute backend at runtime.

    Reassigns the module-level globals ``_use_gpu``, ``xp``, ``_fft``, and
    ``ifftshift``.

    .. warning::
        Must be called **before** constructing any
        :class:`~Reconstruction._base.DeconvBase` subclass.  Calling it
        after construction produces undefined behaviour: existing objects
        still hold references to the old ``xp`` and will not be migrated.

    Parameters
    ----------
    mode : {"auto", "cpu", "gpu"}
        - ``"auto"`` : re-run GPU detection and select accordingly.
        - ``"cpu"``  : force CPU (NumPy) regardless of hardware.
        - ``"gpu"``  : force GPU (CuPy); raises ``RuntimeError`` if
          unavailable.

    Raises
    ------
    RuntimeError
        If ``mode="gpu"`` and no functional CUDA device is available.
    ValueError
        If ``mode`` is not one of ``"auto"``, ``"cpu"``, ``"gpu"``.
    """
    global _use_gpu, xp, _fft, ifftshift

    if mode == "auto":
        _use_gpu = _detect_gpu()
    elif mode == "cpu":
        _use_gpu = False
    elif mode == "gpu":
        if not _detect_gpu():
            raise RuntimeError(
                "set_backend('gpu') requested but no functional CUDA device "
                "was found.  Install CuPy and ensure a CUDA-capable GPU is "
                "present, or use set_backend('cpu') or set_backend('auto')."
            )
        _use_gpu = True
    else:
        raise ValueError(
            f"Unknown backend mode {mode!r}.  Use 'auto', 'cpu', or 'gpu'."
        )

    if _use_gpu:
        import cupy as cp
        xp = cp
        _fft = cp.fft
        try:
            cp.fft.config.set_plan_cache_size(64)
        except AttributeError:
            pass
    else:
        xp = np
        _fft = np.fft

    # Keep ifftshift alias in sync with the new _fft.
    ifftshift = _fft.ifftshift

    logger.info(
        "Reconstruction backend set to %s (xp=%s).",
        "GPU/CuPy" if _use_gpu else "CPU/NumPy",
        xp.__name__,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FFT Helpers  (real-valued optimisation)
# ══════════════════════════════════════════════════════════════════════════════
# We use rfft2 / irfft2 throughout.  For a real M×N array, rfft2 returns a
# complex M×(N//2+1) array, exploiting Hermitian symmetry to halve both the
# memory footprint and the arithmetic cost compared to the full fft2/ifft2.

def rfft2(a: xp.ndarray, **kwargs) -> xp.ndarray:
    """
    Backend-agnostic real 2-D FFT.

    Parameters
    ----------
    a : xp.ndarray, shape (H, W), real
        Spatial-domain real-valued array.

    Returns
    -------
    xp.ndarray, shape (H, W//2+1), complex
        Half-spectrum exploiting Hermitian symmetry.
    """
    return _fft.rfft2(a, **kwargs)


def irfft2(a: xp.ndarray, s: tuple[int, int], **kwargs) -> xp.ndarray:
    """
    Backend-agnostic inverse real 2-D FFT.

    Parameters
    ----------
    a : xp.ndarray, shape (H, W//2+1), complex
        Half-spectrum (output of ``rfft2``).
    s : tuple of int
        ``(H, W)`` — the desired *output* spatial shape.  Required because
        ``W`` cannot be uniquely inferred from ``W//2+1`` (even/odd ambiguity).

    Returns
    -------
    xp.ndarray, shape (H, W), real
        Reconstructed spatial-domain array.
    """
    return _fft.irfft2(a, s=s, **kwargs)


# For PSF centering we need the full-complex shift helper.
# NOTE: This module-level alias is kept in sync with _fft by set_backend().
ifftshift = _fft.ifftshift


# ══════════════════════════════════════════════════════════════════════════════
# Utility Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _freeze(a: xp.ndarray) -> xp.ndarray:
    """Mark array as read-only to prevent accidental in-place modification."""
    try:
        a.flags.writeable = False
    except AttributeError:
        pass  # CuPy arrays may not support this on all versions.
    return a


def _to_numpy(x: xp.ndarray) -> np.ndarray:
    """
    Transfer array to host (CPU) memory if it lives on the GPU.

    If the backend is already NumPy, this is a no-op (returns the same object).
    """
    if _use_gpu:
        return xp.asnumpy(x)  # cp.asnumpy performs a DMA copy to host.
    return x
