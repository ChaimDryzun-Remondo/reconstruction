from __future__ import annotations

import numpy as np
import pytest

import Reconstruction._backend as backend
import Reconstruction._denoise_utils as denoise_utils


@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


class TestBM3DDenoiseRangeMode:

    def test_affine_in_range_uses_original_scale_and_clips_output(self, monkeypatch):
        calls: dict[str, object] = {}

        def fake_bm3d(image, sigma_psd, profile):
            calls["image"] = image.copy()
            calls["sigma_psd"] = sigma_psd
            calls["profile"] = profile
            return image + 0.25

        monkeypatch.setattr(denoise_utils, "_HAS_BM3D", True)
        monkeypatch.setattr(denoise_utils, "_bm3d_func", fake_bm3d)

        arr = backend.xp.array([[0.0, 0.5], [1.0, 0.75]], dtype=backend.xp.float64)
        result = denoise_utils.bm3d_denoise(arr, sigma=0.1, profile="lc", range_mode="affine")

        np.testing.assert_allclose(
            calls["image"],
            np.array([[0.0, 0.5], [1.0, 0.75]], dtype=np.float64),
            atol=1e-12,
        )
        assert calls["sigma_psd"] == 0.1
        assert calls["profile"] == "lc"
        np.testing.assert_allclose(
            backend._to_numpy(result),
            np.array([[0.25, 0.75], [1.0, 1.0]], dtype=np.float64),
            atol=1e-12,
        )

    def test_affine_out_of_range_rescales_sigma_and_inverts_mapping(self, monkeypatch):
        calls: dict[str, object] = {}

        def fake_bm3d(image, sigma_psd, profile):
            calls["image"] = image.copy()
            calls["sigma_psd"] = sigma_psd
            calls["profile"] = profile
            return image * 0.5

        monkeypatch.setattr(denoise_utils, "_HAS_BM3D", True)
        monkeypatch.setattr(denoise_utils, "_bm3d_func", fake_bm3d)

        arr = backend.xp.array([[10.0, 12.0], [14.0, 20.0]], dtype=backend.xp.float64)
        result = denoise_utils.bm3d_denoise(arr, sigma=2.0, range_mode="affine")

        expected_input = np.array([[0.0, 0.2], [0.4, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(calls["image"], expected_input, atol=1e-12)
        assert abs(float(calls["sigma_psd"]) - 0.2) < 1e-12
        assert calls["profile"] == "np"
        np.testing.assert_allclose(
            backend._to_numpy(result),
            np.array([[10.0, 11.0], [12.0, 15.0]], dtype=np.float64),
            atol=1e-12,
        )

    def test_none_passes_through_without_affine_rescaling(self, monkeypatch):
        calls: dict[str, object] = {}

        def fake_bm3d(image, sigma_psd, profile):
            calls["image"] = image.copy()
            calls["sigma_psd"] = sigma_psd
            calls["profile"] = profile
            return image + 0.5

        monkeypatch.setattr(denoise_utils, "_HAS_BM3D", True)
        monkeypatch.setattr(denoise_utils, "_bm3d_func", fake_bm3d)

        arr = backend.xp.array([[-2.0, 0.0], [3.0, 5.0]], dtype=backend.xp.float64)
        result = denoise_utils.bm3d_denoise(arr, sigma=0.3, range_mode="none")

        np.testing.assert_allclose(
            calls["image"],
            np.array([[-2.0, 0.0], [3.0, 5.0]], dtype=np.float64),
            atol=1e-12,
        )
        assert calls["sigma_psd"] == 0.3
        assert calls["profile"] == "np"
        np.testing.assert_allclose(
            backend._to_numpy(result),
            np.array([[-1.5, 0.5], [3.5, 5.5]], dtype=np.float64),
            atol=1e-12,
        )

    def test_invalid_range_mode_raises(self, monkeypatch):
        monkeypatch.setattr(denoise_utils, "_HAS_BM3D", True)

        arr = backend.xp.array([[0.0, 1.0]], dtype=backend.xp.float64)
        with pytest.raises(ValueError, match="range_mode"):
            denoise_utils.bm3d_denoise(arr, sigma=0.1, range_mode="bad")
