"""
Phase 1 verification tests for Reconstruction._backend.

Checks:
  1. rfft2 → irfft2 round-trip recovers the original array (allclose).
  2. _to_numpy returns a numpy.ndarray.
  3. set_backend("cpu") forces xp=numpy and _use_gpu=False.
  4. set_backend("auto") completes without error.
  5. set_backend("gpu") raises RuntimeError when no GPU is available.
  6. ifftshift is callable and matches numpy.fft.ifftshift.
  7. _freeze makes numpy arrays read-only.
  8. PaddingStr is exported with the correct Literal args.
"""
from __future__ import annotations

import typing

import numpy as np
import pytest

import Reconstruction._backend as backend


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """
    Force CPU backend before and after every test for isolation.

    Tests that explicitly call set_backend() are still isolated because
    this fixture restores CPU state in the teardown (after yield).
    """
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# FFT Round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestFFTRoundtrip:
    """Verify rfft2 → irfft2 round-trip correctness."""

    def test_roundtrip_64x64(self):
        """rfft2 → irfft2 on a 64×64 float32 array returns the original."""
        rng = np.random.default_rng(42)
        arr_np = rng.random((64, 64)).astype(np.float32)
        arr = backend.xp.array(arr_np)

        spectrum = backend.rfft2(arr)
        recovered = backend.irfft2(spectrum, s=(64, 64))
        recovered_np = backend._to_numpy(recovered)

        np.testing.assert_allclose(
            arr_np, recovered_np, atol=1e-5,
            err_msg="rfft2 → irfft2 round-trip did not recover original array",
        )

    def test_rfft2_output_shape(self):
        """rfft2 of (H, W) real array returns (H, W//2+1) half-spectrum."""
        arr = backend.xp.zeros((64, 64), dtype=backend.xp.float32)
        spectrum = backend.rfft2(arr)
        assert spectrum.shape == (64, 33), (
            f"Expected half-spectrum shape (64, 33), got {spectrum.shape}"
        )

    def test_irfft2_output_shape(self):
        """irfft2 with explicit s returns the requested spatial shape."""
        arr = backend.xp.zeros((64, 64), dtype=backend.xp.float32)
        spectrum = backend.rfft2(arr)
        recovered = backend.irfft2(spectrum, s=(64, 64))
        assert recovered.shape == (64, 64)

    def test_roundtrip_odd_shape(self):
        """Round-trip works correctly for odd spatial dimensions."""
        rng = np.random.default_rng(0)
        arr_np = rng.random((65, 65)).astype(np.float32)
        arr = backend.xp.array(arr_np)
        recovered_np = backend._to_numpy(
            backend.irfft2(backend.rfft2(arr), s=(65, 65))
        )
        np.testing.assert_allclose(arr_np, recovered_np, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# _to_numpy
# ══════════════════════════════════════════════════════════════════════════════

class TestToNumpy:
    """Verify _to_numpy always returns a CPU numpy array."""

    def test_returns_numpy_ndarray(self):
        """_to_numpy must return numpy.ndarray."""
        arr = backend.xp.array(np.ones((4, 4), dtype=np.float32))
        result = backend._to_numpy(arr)
        assert isinstance(result, np.ndarray), (
            f"Expected numpy.ndarray, got {type(result).__name__}"
        )

    def test_values_preserved(self):
        """_to_numpy preserves all element values."""
        arr_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr = backend.xp.array(arr_np)
        result = backend._to_numpy(arr)
        np.testing.assert_array_equal(result, arr_np)

    def test_shape_preserved(self):
        """_to_numpy preserves array shape."""
        arr_np = np.zeros((7, 13), dtype=np.float32)
        result = backend._to_numpy(backend.xp.array(arr_np))
        assert result.shape == (7, 13)


# ══════════════════════════════════════════════════════════════════════════════
# set_backend
# ══════════════════════════════════════════════════════════════════════════════

class TestSetBackend:
    """Verify set_backend() runtime backend switching."""

    def test_set_backend_cpu_forces_numpy(self):
        """set_backend('cpu') sets xp=numpy and _use_gpu=False."""
        backend.set_backend("cpu")
        assert backend.xp is np, "xp should be numpy after set_backend('cpu')"
        assert backend._use_gpu is False
        assert backend._fft is np.fft

    def test_set_backend_cpu_fft_roundtrip(self):
        """FFT round-trip still works correctly after set_backend('cpu')."""
        backend.set_backend("cpu")
        rng = np.random.default_rng(7)
        arr_np = rng.random((32, 32)).astype(np.float32)
        arr = backend.xp.array(arr_np)
        recovered = backend._to_numpy(
            backend.irfft2(backend.rfft2(arr), s=(32, 32))
        )
        np.testing.assert_allclose(arr_np, recovered, atol=1e-5)

    def test_set_backend_auto_no_error(self):
        """set_backend('auto') completes without raising."""
        backend.set_backend("auto")
        assert hasattr(backend.xp, "zeros")
        assert hasattr(backend.xp, "array")

    def test_set_backend_auto_xp_is_numpy_when_no_gpu(self):
        """set_backend('auto') selects numpy when no GPU is available."""
        if backend._detect_gpu():
            pytest.skip("GPU present; CPU-only auto test not applicable.")
        backend.set_backend("auto")
        assert backend.xp is np

    def test_set_backend_gpu_raises_without_gpu(self):
        """set_backend('gpu') raises RuntimeError when no GPU is available."""
        if backend._detect_gpu():
            pytest.skip("GPU is available; skipping no-GPU error-path test.")
        with pytest.raises(RuntimeError, match="no functional CUDA device"):
            backend.set_backend("gpu")

    def test_set_backend_invalid_mode_raises(self):
        """set_backend() with an unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend mode"):
            backend.set_backend("invalid")  # type: ignore[arg-type]

    def test_set_backend_cpu_to_numpy_returns_numpy(self):
        """After set_backend('cpu'), _to_numpy returns numpy array."""
        backend.set_backend("cpu")
        arr = backend.xp.array(np.ones((3, 3), dtype=np.float32))
        result = backend._to_numpy(arr)
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# ifftshift
# ══════════════════════════════════════════════════════════════════════════════

class TestIfftshift:
    """Verify the ifftshift alias."""

    def test_ifftshift_is_callable(self):
        """ifftshift must be callable."""
        assert callable(backend.ifftshift)

    def test_ifftshift_matches_numpy(self):
        """ifftshift output matches numpy.fft.ifftshift on CPU."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = backend.ifftshift(arr)
        expected = np.fft.ifftshift(arr)
        np.testing.assert_array_equal(result, expected)

    def test_ifftshift_preserves_shape(self):
        """ifftshift does not change array shape."""
        arr = np.ones((11, 11), dtype=np.float32)
        result = backend.ifftshift(arr)
        assert result.shape == arr.shape


# ══════════════════════════════════════════════════════════════════════════════
# _freeze
# ══════════════════════════════════════════════════════════════════════════════

class TestFreeze:
    """Verify _freeze makes arrays read-only."""

    def test_freeze_returns_same_object(self):
        """_freeze returns the exact same array object."""
        arr = np.ones((4, 4), dtype=np.float32)
        result = backend._freeze(arr)
        assert result is arr

    def test_freeze_sets_not_writeable(self):
        """_freeze marks the numpy array as not writeable."""
        arr = np.ones((4, 4), dtype=np.float32)
        assert arr.flags.writeable  # sanity check: starts writeable
        backend._freeze(arr)
        assert not arr.flags.writeable

    def test_frozen_array_raises_on_write(self):
        """Writing to a frozen array raises ValueError."""
        arr = np.ones((4, 4), dtype=np.float32)
        backend._freeze(arr)
        with pytest.raises((ValueError, TypeError)):
            arr[0, 0] = 999.0


# ══════════════════════════════════════════════════════════════════════════════
# PaddingStr
# ══════════════════════════════════════════════════════════════════════════════

class TestPaddingStr:
    """Verify PaddingStr type alias is exported with the correct Literal args."""

    def test_padding_str_is_exported(self):
        """PaddingStr is importable from Reconstruction._backend."""
        from Reconstruction._backend import PaddingStr
        assert PaddingStr is not None

    def test_padding_str_literal_args(self):
        """PaddingStr contains exactly the six expected mode strings."""
        from Reconstruction._backend import PaddingStr
        args = typing.get_args(PaddingStr)
        expected = {"Reflect", "Symmetric", "Wrap", "Edge", "LinearRamp", "Zero"}
        assert set(args) == expected, (
            f"PaddingStr args mismatch.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Got:      {sorted(args)}"
        )
