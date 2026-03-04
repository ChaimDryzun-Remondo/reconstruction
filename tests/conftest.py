"""
Shared fixtures for Reconstruction package tests.

Provides:
  - Synthetic test images and PSFs.
  - Mock/stub implementations of RemondoPythonCore.Common utilities so tests
    can run without the broader project installed.  Shared.Common stubs are
    also installed for compatibility with the docs/reference/*.py regression
    tests (which still import from that namespace).
"""

from __future__ import annotations

import sys
import types
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Mock RemondoPythonCore.Common / Shared.Common Utilities
# ═══════════════════════════════════════════════════════════════════════════
# The Reconstruction package imports from RemondoPythonCore.Common, which
# lives outside this repo.  These stubs replicate the minimal behaviour
# needed for tests.  They are injected into sys.modules BEFORE any
# Reconstruction import, so the real packages are never required.

def _mock_padding(image: np.ndarray, full_size: tuple[int, int],
                  Type: str = "Reflect", apply_taper: bool = False) -> np.ndarray:
    """Centre-pad image to full_size using numpy.pad."""
    h, w = image.shape[:2]
    fh, fw = full_size

    pad_h = fh - h
    pad_w = fw - w
    top = pad_h // 2
    bot = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    mode_map = {
        "Reflect": "reflect",
        "Symmetric": "symmetric",
        "Wrap": "wrap",
        "Edge": "edge",
        "Zero": "constant",
        "LinearRamp": "linear_ramp",
    }
    np_mode = mode_map.get(Type, "reflect")

    kwargs = {}
    if np_mode == "constant":
        kwargs["constant_values"] = 0

    return np.pad(image, ((top, bot), (left, right)), mode=np_mode, **kwargs)


def _mock_cropping(image: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Centre-crop image to crop_size."""
    h, w = image.shape[:2]
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[top:top + ch, left:left + cw].copy()


def _mock_psf_preprocess(psf: np.ndarray, **kwargs) -> np.ndarray:
    """Minimal PSF preprocessing: clip negatives, normalize to sum=1, ensure odd shape."""
    out = psf.copy().astype(np.float64)
    out = np.clip(out, 0, None)
    h, w = out.shape
    if h % 2 == 0:
        out = out[:-1, :]
    if out.shape[1] % 2 == 0:
        out = out[:, :-1]
    total = out.sum()
    if total > 0:
        out /= total
    return out


def _mock_condition_psf(psf: np.ndarray, **kwargs) -> np.ndarray:
    """No-op conditioning for tests (PSF already clean)."""
    out = psf.copy()
    total = out.sum()
    if total > 0:
        out /= total
    return out


def _mock_image_normalization(image: np.ndarray, bit_depth: int = 1,
                               is_int: bool = False) -> np.ndarray:
    """Scale image to [0, 1]."""
    out = image.astype(np.float64)
    vmin, vmax = out.min(), out.max()
    if vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
    return out


def _mock_validate_image(image: np.ndarray) -> None:
    """Basic validation: must be 2D or 3D numpy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D.")


def _mock_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 3:
        return (0.2989 * image[:, :, 0] +
                0.5870 * image[:, :, 1] +
                0.1140 * image[:, :, 2]).astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].astype(np.float64)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _mock_odd_crop_around_center(image: np.ndarray,
                                  target_shape: tuple[int, int]) -> np.ndarray:
    """Crop to target_shape, centred."""
    return _mock_cropping(image, target_shape)


def _install_mocks() -> None:
    """Inject mock modules into sys.modules before any Reconstruction imports.

    Each namespace is guarded independently so that whichever is already
    provided by the real project is left untouched while the other is mocked.
    """
    # ── RemondoPythonCore.Common.* (primary import path in _base.py) ──────
    if "RemondoPythonCore.Common.General_Utilities" not in sys.modules:
        rpc = types.ModuleType("RemondoPythonCore")
        rpc.__path__ = []
        rpc_common = types.ModuleType("RemondoPythonCore.Common")
        rpc_common.__path__ = []

        rpc_gen_utils = types.ModuleType("RemondoPythonCore.Common.General_Utilities")
        rpc_gen_utils.padding = _mock_padding
        rpc_gen_utils.cropping = _mock_cropping
        rpc_gen_utils.odd_crop_around_center = _mock_odd_crop_around_center

        rpc_psf_pre = types.ModuleType("RemondoPythonCore.Common.PSF_Preprocessing")
        rpc_psf_pre.psf_preprocess = _mock_psf_preprocess
        rpc_psf_pre.condition_psf = _mock_condition_psf

        rpc_img_pre = types.ModuleType("RemondoPythonCore.Common.Image_Preprocessing")
        rpc_img_pre.image_normalization = _mock_image_normalization
        rpc_img_pre.validate_image = _mock_validate_image
        rpc_img_pre.to_grayscale = _mock_to_grayscale
        rpc_img_pre.odd_crop_around_center = _mock_odd_crop_around_center

        sys.modules["RemondoPythonCore"] = rpc
        sys.modules["RemondoPythonCore.Common"] = rpc_common
        sys.modules["RemondoPythonCore.Common.General_Utilities"] = rpc_gen_utils
        sys.modules["RemondoPythonCore.Common.PSF_Preprocessing"] = rpc_psf_pre
        sys.modules["RemondoPythonCore.Common.Image_Preprocessing"] = rpc_img_pre

    # ── Shared.Common.* (needed by docs/reference/*.py regression tests) ──
    if "Shared.Common.General_Utilities" not in sys.modules:
        shared = types.ModuleType("Shared")
        shared.__path__ = []
        s_common = types.ModuleType("Shared.Common")
        s_common.__path__ = []

        s_gen = types.ModuleType("Shared.Common.General_Utilities")
        s_gen.padding = _mock_padding
        s_gen.cropping = _mock_cropping

        s_psf = types.ModuleType("Shared.Common.PSF_Preprocessing")
        s_psf.psf_preprocess = _mock_psf_preprocess
        s_psf.condition_psf = _mock_condition_psf

        s_img = types.ModuleType("Shared.Common.Image_Preprocessing")
        s_img.image_normalization = _mock_image_normalization
        s_img.validate_image = _mock_validate_image
        s_img.to_grayscale = _mock_to_grayscale
        s_img.odd_crop_around_center = _mock_odd_crop_around_center

        sys.modules["Shared"] = shared
        sys.modules["Shared.Common"] = s_common
        sys.modules["Shared.Common.General_Utilities"] = s_gen
        sys.modules["Shared.Common.PSF_Preprocessing"] = s_psf
        sys.modules["Shared.Common.Image_Preprocessing"] = s_img


# Install mocks at import time (before any Reconstruction imports).
_install_mocks()


# ═══════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_gaussian_psf(size: int = 11, sigma: float = 2.0) -> np.ndarray:
    """Create a normalized Gaussian PSF."""
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf.astype(np.float64)


def _make_test_image(h: int = 65, w: int = 65) -> np.ndarray:
    """
    Create a synthetic test image: central bright block on dark background.

    Shape is odd by default (65×65) to match the odd-enforcement logic
    in DeconvBase.
    """
    img = np.full((h, w), 0.1, dtype=np.float64)
    # Central bright block
    ch, cw = h // 4, w // 4
    img[ch:3 * ch, cw:3 * cw] = 0.8
    # Diagonal gradient in one quadrant (tests edge/gradient behaviour)
    for i in range(ch):
        for j in range(cw):
            img[3 * ch + i, 3 * cw + j] = 0.1 + 0.6 * (i + j) / (ch + cw)
    return img


def _blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Blur an image with a PSF using FFT convolution (ground truth)."""
    from scipy.signal import fftconvolve
    blurred = fftconvolve(image, psf, mode="same")
    return np.clip(blurred, 0, None)


@pytest.fixture
def gaussian_psf() -> np.ndarray:
    """11×11 Gaussian PSF with σ=2.0, sum=1."""
    return _make_gaussian_psf(size=11, sigma=2.0)


@pytest.fixture
def test_image() -> np.ndarray:
    """65×65 synthetic test image in [0, 1]."""
    return _make_test_image(h=65, w=65)


@pytest.fixture
def blurred_image(test_image, gaussian_psf) -> np.ndarray:
    """Test image blurred with the Gaussian PSF."""
    return _blur_image(test_image, gaussian_psf)


@pytest.fixture
def noisy_blurred_image(blurred_image) -> np.ndarray:
    """Blurred test image with additive Gaussian noise (σ=0.01)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, blurred_image.shape)
    return np.clip(blurred_image + noise, 0, None)
