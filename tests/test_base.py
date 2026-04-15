"""
Phase 3 verification tests for Reconstruction._base.DeconvBase.

Uses a minimal concrete subclass _TestDeconv that returns the initial
estimate unchanged.  Verifies:

  1. full_shape is odd in both dimensions.
  2. mask has exactly h*w ones (use_mask=True).
  3. PF.shape == (full_shape[0], full_shape[1]//2+1).
  4. HTM.min() >= htm_floor_frac * HTM.max()  (floor clamp).
  5. _lipschitz > 0.
  6. h, w match the original odd-enforced image dimensions.
  7. deblur() returns a numpy array of shape (h, w).
  8. use_mask=False produces an all-ones mask.
  9. _check_convergence: correct (float, bool) for converged / not-converged.
 10. _crop_and_return: correct shape, dtype, and state update.
 11. _INIT_KEYS class attribute: correct contents.
"""
from __future__ import annotations

import numpy as np
import pytest

import Reconstruction._backend as backend
import Reconstruction._base as base_module
from Reconstruction._base import DeconvBase


# ══════════════════════════════════════════════════════════════════════════════
# Minimal concrete subclass for testing
# ══════════════════════════════════════════════════════════════════════════════

class _TestDeconv(DeconvBase):
    """Trivial subclass: deblur() returns the initial estimate unchanged."""

    def deblur(self, **kwargs) -> np.ndarray:
        return self._crop_and_return(self.estimated_image, **kwargs)


def _to_expected_working_domain(image: np.ndarray) -> np.ndarray:
    """Mirror the intended working-domain contract in test space."""
    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        ).astype(np.float64)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0].astype(np.float64)
    else:
        raise ValueError(f"Unsupported image shape for test helper: {image.shape}")

    H, W = gray.shape
    OH = H if H % 2 == 1 else H - 1
    OW = W if W % 2 == 1 else W - 1
    off_y = (H - OH) // 2
    off_x = (W - OW) // 2
    gray = gray[off_y:off_y + OH, off_x:off_x + OW]

    vmin = float(gray.min())
    vmax = float(gray.max())
    if vmax > vmin:
        gray = (gray - vmin) / (vmax - vmin)
    return gray


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def ensure_cpu_backend():
    """Force CPU backend before and after every test."""
    backend.set_backend("cpu")
    yield
    backend.set_backend("cpu")


@pytest.fixture
def deconv(test_image, gaussian_psf) -> _TestDeconv:
    """_TestDeconv constructed with use_mask=True (default)."""
    return _TestDeconv(test_image, gaussian_psf)


@pytest.fixture
def deconv_no_mask(test_image, gaussian_psf) -> _TestDeconv:
    """_TestDeconv constructed with use_mask=False."""
    return _TestDeconv(test_image, gaussian_psf, use_mask=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Constructor — canvas shape and original dimensions
# ══════════════════════════════════════════════════════════════════════════════

class TestConstructorShape:
    """Verify padded canvas shape and image dimension storage."""

    def test_full_shape_is_odd_height(self, deconv):
        """full_shape[0] must be an odd integer."""
        assert deconv.full_shape[0] % 2 == 1, (
            f"full_shape[0] = {deconv.full_shape[0]} is not odd"
        )

    def test_full_shape_is_odd_width(self, deconv):
        """full_shape[1] must be an odd integer."""
        assert deconv.full_shape[1] % 2 == 1, (
            f"full_shape[1] = {deconv.full_shape[1]} is not odd"
        )

    def test_full_shape_larger_than_image(self, deconv):
        """Padded canvas must be strictly larger than the original image."""
        assert deconv.full_shape[0] > deconv.h
        assert deconv.full_shape[1] > deconv.w

    def test_h_w_are_odd(self, deconv):
        """Stored h and w must be odd (enforced by the constructor)."""
        assert deconv.h % 2 == 1, f"h = {deconv.h} is not odd"
        assert deconv.w % 2 == 1, f"w = {deconv.w} is not odd"

    def test_h_w_match_image_shape(self, deconv, test_image):
        """h and w must match the spatial dimensions of the test image."""
        # The test_image is 65×65 (already odd) so no crop happens.
        assert deconv.h == test_image.shape[0]
        assert deconv.w == test_image.shape[1]

    def test_image_array_shape(self, deconv):
        """The padded image attribute must have shape full_shape."""
        assert deconv.image.shape == deconv.full_shape


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mask construction
# ══════════════════════════════════════════════════════════════════════════════

class TestMask:
    """Verify mask shape, content, and the use_mask=False path."""

    def test_mask_shape(self, deconv):
        """Mask must have the same shape as the padded canvas."""
        assert deconv.mask.shape == deconv.full_shape

    def test_mask_sum_equals_h_times_w(self, deconv):
        """use_mask=True: mask must contain exactly h*w ones."""
        expected = float(deconv.h * deconv.w)
        actual = float(np.sum(deconv.mask))
        assert actual == expected, (
            f"Mask sum {actual} ≠ h*w = {expected} "
            f"(h={deconv.h}, w={deconv.w})"
        )

    def test_mask_values_binary(self, deconv):
        """use_mask=True: mask values must be 0.0 or 1.0 only."""
        mask_np = np.asarray(deconv.mask)
        unique = set(np.unique(mask_np).tolist())
        assert unique <= {0.0, 1.0}, f"Non-binary mask values: {unique}"

    def test_use_mask_false_all_ones(self, deconv_no_mask):
        """use_mask=False: mask must be entirely 1.0."""
        mask_np = np.asarray(deconv_no_mask.mask)
        assert np.all(mask_np == 1.0), (
            f"use_mask=False produced non-unit mask "
            f"(min={mask_np.min():.6f}, max={mask_np.max():.6f})"
        )

    def test_use_mask_false_shape(self, deconv_no_mask):
        """use_mask=False: mask shape must still equal full_shape."""
        assert deconv_no_mask.mask.shape == deconv_no_mask.full_shape

    def test_use_mask_attribute_stored(self, deconv, deconv_no_mask):
        """use_mask attribute must match the constructor argument."""
        assert deconv.use_mask is True
        assert deconv_no_mask.use_mask is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. PSF / frequency-domain precomputation
# ══════════════════════════════════════════════════════════════════════════════

class TestPSFPrecomputation:
    """Verify PF, conjPF, HTM, and Lipschitz constant."""

    def test_PF_shape(self, deconv):
        """PF must be the half-spectrum: shape (full_H, full_W//2+1)."""
        fH, fW = deconv.full_shape
        expected = (fH, fW // 2 + 1)
        assert deconv.PF.shape == expected, (
            f"PF.shape = {deconv.PF.shape}, expected {expected}"
        )

    def test_conjPF_shape_matches_PF(self, deconv):
        """conjPF must have the same shape as PF."""
        assert deconv.conjPF.shape == deconv.PF.shape

    def test_conjPF_is_conjugate_of_PF(self, deconv):
        """conjPF must equal PF.conj() element-wise."""
        np.testing.assert_allclose(
            np.asarray(deconv.conjPF),
            np.asarray(deconv.PF).conj(),
            atol=1e-6,
            err_msg="conjPF is not the conjugate of PF",
        )

    def test_PF_is_read_only(self, deconv):
        """PF must be frozen (read-only)."""
        pf_np = np.asarray(deconv.PF)
        assert not pf_np.flags.writeable, "PF should be read-only after _freeze"

    def test_HTM_shape(self, deconv):
        """HTM must have shape full_shape."""
        assert deconv.HTM.shape == deconv.full_shape

    def test_HTM_floor_clamp(self, deconv):
        """HTM.min() must be >= htm_floor_frac * HTM.max() (floor clamp)."""
        htm_np = np.asarray(deconv.HTM)
        htm_min = float(htm_np.min())
        htm_max = float(htm_np.max())
        # Default htm_floor_frac = 0.01
        floor = 0.01 * htm_max
        assert htm_min >= floor - 1e-7, (
            f"HTM floor clamp violated: min={htm_min:.6f} < "
            f"0.01 × max={htm_max:.6f} = {floor:.6f}"
        )

    def test_HTM_all_positive(self, deconv):
        """All HTM values must be strictly positive after the floor clamp."""
        htm_np = np.asarray(deconv.HTM)
        assert float(htm_np.min()) > 0.0, "HTM contains non-positive values"

    def test_HTM_is_read_only(self, deconv):
        """HTM must be frozen (read-only)."""
        htm_np = np.asarray(deconv.HTM)
        assert not htm_np.flags.writeable, "HTM should be read-only after _freeze"

    def test_lipschitz_positive(self, deconv):
        """_lipschitz must be strictly positive."""
        assert deconv._lipschitz > 0.0, (
            f"Lipschitz constant is non-positive: {deconv._lipschitz}"
        )

    def test_lipschitz_equals_max_abs_pf_sq(self, deconv):
        """_lipschitz must equal max |PF|²."""
        pf_np = np.asarray(deconv.PF)
        expected = float(np.max(np.abs(pf_np) ** 2))
        assert abs(deconv._lipschitz - expected) < 1e-6, (
            f"_lipschitz {deconv._lipschitz} ≠ max|PF|² {expected}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3b. PSF pipeline contract
# ══════════════════════════════════════════════════════════════════════════════

class TestPSFPipelineContract:
    """Lock the current iterative-family PSF preprocessing pipeline."""

    def test_iterative_family_psf_preprocess_defaults(self, test_image, gaussian_psf, monkeypatch):
        captured: dict[str, dict] = {}

        def fake_psf_preprocess(psf, **kwargs):
            captured["preprocess"] = kwargs
            return np.asarray(psf, dtype=np.float64)

        def fake_condition_psf(psf, **kwargs):
            captured["condition"] = kwargs
            return np.asarray(psf, dtype=np.float64)

        monkeypatch.setattr(base_module, "psf_preprocess", fake_psf_preprocess)
        monkeypatch.setattr(base_module, "condition_psf", fake_condition_psf)

        _TestDeconv(test_image, gaussian_psf)

        assert captured["preprocess"] == {
            "center_method": "com",
            "remove_negatives": "clip",
            "eps": 1e-12,
            "enforce_odd_shape": True,
        }
        assert captured["condition"] == {
            "bg_ring_frac": 0.15,
            "taper_outer_frac": 0.20,
            "taper_end_frac": 0.50,
        }

    def test_iterative_family_psf_zero_padding_and_ifftshift(self, test_image, gaussian_psf, monkeypatch):
        original_padding = base_module.padding
        captured: dict[str, object] = {}

        def fake_padding(image, full_size, Type, apply_taper=False):
            if Type == "Zero" and image.shape == gaussian_psf.shape:
                captured["padding_call"] = {
                    "full_size": full_size,
                    "Type": Type,
                    "apply_taper": apply_taper,
                }
                return np.full(full_size, 7.0, dtype=np.float32)
            return original_padding(
                image=image,
                full_size=full_size,
                Type=Type,
                apply_taper=apply_taper,
            )

        def fake_ifftshift(arr):
            if captured.get("padding_call") and np.all(arr == 7.0):
                captured["ifftshift_seen"] = True
            return np.fft.ifftshift(arr)

        monkeypatch.setattr(base_module, "padding", fake_padding)
        monkeypatch.setattr(base_module.backend, "ifftshift", fake_ifftshift)

        obj = _TestDeconv(test_image, gaussian_psf)

        assert captured["padding_call"] == {
            "full_size": obj.full_shape,
            "Type": "Zero",
            "apply_taper": False,
        }
        assert captured.get("ifftshift_seen") is True


# ══════════════════════════════════════════════════════════════════════════════
# 4. Initial estimate
# ══════════════════════════════════════════════════════════════════════════════

class TestInitialEstimate:
    """Verify the initial estimate attributes."""

    def test_estimated_image_shape(self, deconv):
        """estimated_image must have shape full_shape."""
        assert deconv.estimated_image.shape == deconv.full_shape

    def test_estimated_image_positive(self, deconv):
        """estimated_image must be strictly positive (≥ floor applied in constructor).

        The floor is np.float32(1e-8), which rounds to ~9.999e-09 in float32.
        We compare against that same float32 value to avoid false failures
        from float32 → float64 precision differences.
        """
        floor = float(np.float32(1e-8))
        est_np = np.asarray(deconv.estimated_image)
        assert float(est_np.min()) >= floor, (
            f"estimated_image has values below float32(1e-8): min={est_np.min():.2e}"
        )

    def test_estimated_image_is_mutable(self, deconv):
        """estimated_image must NOT be frozen (algorithms modify it)."""
        est_np = np.asarray(deconv.estimated_image)
        assert est_np.flags.writeable, "estimated_image should be writable"


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Working-domain contract
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkingDomainContract:
    """Lock the chosen grayscale/odd/normalised working-domain contract."""

    def test_image_preprocessing_uses_grayscale_odd_normalized_domain(self, gaussian_psf):
        image = np.arange(6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
        deconv = _TestDeconv(image, gaussian_psf)

        expected = _to_expected_working_domain(image)
        result = deconv.deblur()

        assert deconv.h == expected.shape[0]
        assert deconv.w == expected.shape[1]
        assert result.ndim == 2
        assert result.shape == expected.shape
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_default_initial_estimate_matches_image_working_domain(self, gaussian_psf):
        image = np.linspace(10.0, 30.0, 6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
        deconv = _TestDeconv(image, gaussian_psf)

        expected = _to_expected_working_domain(image)
        result = deconv.deblur()

        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(
            result,
            np.asarray(deconv.image)[
                (deconv.full_shape[0] - deconv.h) // 2:(deconv.full_shape[0] + deconv.h) // 2,
                (deconv.full_shape[1] - deconv.w) // 2:(deconv.full_shape[1] + deconv.w) // 2,
            ],
            atol=1e-6,
        )

    def test_initial_estimate_must_be_transformed_into_same_working_domain(
        self, gaussian_psf
    ):
        image = np.linspace(0.0, 255.0, 6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
        initial = np.linspace(100.0, 400.0, 6 * 8, dtype=np.float64).reshape(6, 8)

        deconv = _TestDeconv(image, gaussian_psf, initialEstimate=initial)
        result = deconv.deblur()
        image_working = _to_expected_working_domain(image)
        initial_gray = initial[0:5, 0:7].astype(np.float64)
        image_gray = (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        ).astype(np.float64)
        image_gray = image_gray[0:5, 0:7]
        image_min = float(image_gray.min())
        image_max = float(image_gray.max())
        expected = (initial_gray - image_min) / (image_max - image_min)

        assert image_working.shape == expected.shape

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_constant_image_uses_explicit_identity_fallback_working_domain(
        self, gaussian_psf
    ):
        """
        Constant observed images fall back to the odd-cropped grayscale values.

        The non-degenerate affine normalization is undefined when the image
        range is zero, so the degenerate fallback keeps the grayscale working
        domain in raw units rather than forcing a synthetic [0, 1] remap.
        """
        image = np.full((6, 8, 3), 5.0, dtype=np.float64)
        deconv = _TestDeconv(image, gaussian_psf)

        expected = _to_expected_working_domain(image)
        result = deconv.deblur()

        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(
            result,
            np.asarray(deconv.image)[
                (deconv.full_shape[0] - deconv.h) // 2:(deconv.full_shape[0] + deconv.h) // 2,
                (deconv.full_shape[1] - deconv.w) // 2:(deconv.full_shape[1] + deconv.w) // 2,
            ],
            atol=1e-6,
        )

    def test_constant_image_initial_estimate_uses_same_identity_fallback(
        self, gaussian_psf
    ):
        """
        initialEstimate follows the same degenerate identity map as the image.

        For a constant observed image, both the observed image and the initial
        estimate stay in the grayscale odd-cropped raw-unit working domain.
        """
        image = np.full((6, 8, 3), 5.0, dtype=np.float64)
        initial = np.linspace(100.0, 400.0, 6 * 8, dtype=np.float64).reshape(6, 8)

        deconv = _TestDeconv(image, gaussian_psf, initialEstimate=initial)
        result = deconv.deblur()

        expected = initial[0:5, 0:7].astype(np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_constant_image_initial_estimate_shape_must_match_working_domain(
        self, gaussian_psf
    ):
        image = np.full((6, 8, 3), 5.0, dtype=np.float64)
        bad_initial = np.ones((6, 6), dtype=np.float64)

        with pytest.raises(ValueError, match="initialEstimate must match the image working-domain shape"):
            _TestDeconv(image, gaussian_psf, initialEstimate=bad_initial)

    def test_nondegenerate_path_remains_unit_interval_affine_map(
        self, gaussian_psf
    ):
        """The constant-image fallback must not change the non-degenerate path."""
        image = np.linspace(10.0, 30.0, 6 * 8, dtype=np.float64).reshape(6, 8)
        deconv = _TestDeconv(image, gaussian_psf)

        result = deconv.deblur()
        expected = _to_expected_working_domain(image)

        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert float(result.min()) == pytest.approx(0.0, abs=1e-7)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-7)

    def test_output_is_returned_in_working_domain_grayscale_scale(self, gaussian_psf):
        image = np.linspace(5.0, 55.0, 6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
        result = _TestDeconv(image, gaussian_psf).deblur()

        assert result.ndim == 2
        assert result.shape == (5, 7)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_inverse_normalize_restores_grayscale_raw_units_for_nondegenerate_input(
        self, gaussian_psf
    ):
        image = np.linspace(5.0, 55.0, 6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
        deconv = _TestDeconv(image, gaussian_psf)
        raw_gray = base_module.odd_crop_around_center(
            base_module.to_grayscale(image), (deconv.h, deconv.w)
        ).astype(np.float64)

        result = deconv.deblur(inverse_normalize=True)

        np.testing.assert_allclose(result, raw_gray.astype(result.dtype), atol=1e-6)

    def test_inverse_normalize_is_noop_for_constant_image_fallback(self, gaussian_psf):
        image = np.full((6, 8, 3), 5.0, dtype=np.float64)
        deconv = _TestDeconv(image, gaussian_psf)

        result_default = deconv.deblur()
        result_inverse = deconv.deblur(inverse_normalize=True)

        np.testing.assert_allclose(result_inverse, result_default, atol=1e-6)

    def test_inverse_normalize_does_not_change_internal_estimated_image_scale(
        self, gaussian_psf
    ):
        image = np.linspace(10.0, 30.0, 6 * 8, dtype=np.float64).reshape(6, 8)
        deconv = _TestDeconv(image, gaussian_psf)

        deconv.deblur(inverse_normalize=True)
        stored = np.asarray(deconv.estimated_image)[
            (deconv.full_shape[0] - deconv.h) // 2:(deconv.full_shape[0] + deconv.h) // 2,
            (deconv.full_shape[1] - deconv.w) // 2:(deconv.full_shape[1] + deconv.w) // 2,
        ]
        expected = _to_expected_working_domain(image)

        np.testing.assert_allclose(stored, expected, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 5. deblur() — abstract method and _TestDeconv implementation
# ══════════════════════════════════════════════════════════════════════════════

class TestDeblur:
    """Verify deblur() output type and shape."""

    def test_deblur_returns_numpy_array(self, deconv):
        """deblur() must return a numpy.ndarray."""
        result = deconv.deblur()
        assert isinstance(result, np.ndarray), (
            f"deblur() returned {type(result).__name__}, expected np.ndarray"
        )

    def test_deblur_output_shape(self, deconv):
        """deblur() output shape must be (h, w)."""
        result = deconv.deblur()
        assert result.shape == (deconv.h, deconv.w), (
            f"deblur() shape {result.shape} ≠ ({deconv.h}, {deconv.w})"
        )

    def test_deblur_output_shape_no_mask(self, deconv_no_mask):
        """deblur() shape is (h, w) when use_mask=False too."""
        result = deconv_no_mask.deblur()
        assert result.shape == (deconv_no_mask.h, deconv_no_mask.w)

    def test_deblur_cannot_instantiate_abstract(self, test_image, gaussian_psf):
        """DeconvBase cannot be instantiated directly (abstract method)."""
        with pytest.raises(TypeError):
            DeconvBase(test_image, gaussian_psf)  # type: ignore[abstract]


# ══════════════════════════════════════════════════════════════════════════════
# 6. _check_convergence helper
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckConvergence:
    """Verify _check_convergence return types and logic."""

    def test_converged_when_identical(self, deconv):
        """Identical x_new and x_old → rel_change=0, converged=True."""
        x = np.ones(deconv.full_shape, dtype=np.float32) * 0.5
        rel_chg, converged = deconv._check_convergence(
            x, x.copy(), k=10, num_iter=100, tol=1e-6
        )
        assert isinstance(rel_chg, float)
        assert isinstance(converged, bool)
        assert rel_chg == pytest.approx(0.0, abs=1e-10)
        assert converged is True

    def test_not_converged_when_very_different(self, deconv):
        """x_new = ones, x_old = zeros → rel_change = 1.0, converged=False."""
        x_new = np.ones(deconv.full_shape, dtype=np.float32)
        x_old = np.zeros(deconv.full_shape, dtype=np.float32)
        rel_chg, converged = deconv._check_convergence(
            x_new, x_old, k=0, num_iter=100, tol=1e-6
        )
        assert isinstance(rel_chg, float)
        assert isinstance(converged, bool)
        assert rel_chg == pytest.approx(1.0, rel=1e-5)
        assert converged is False

    def test_borderline_converged(self, deconv):
        """rel_change < tol must return converged=True."""
        x_new = np.ones(deconv.full_shape, dtype=np.float32)
        # Perturbation small enough to be below tol
        tiny = np.float32(1e-9)
        x_old = x_new + tiny
        rel_chg, converged = deconv._check_convergence(
            x_new, x_old, k=5, num_iter=100, tol=1e-6
        )
        assert converged is True, f"Expected converged but rel_chg={rel_chg:.2e}"

    def test_borderline_not_converged(self, deconv):
        """rel_change > tol must return converged=False."""
        x_new = np.ones(deconv.full_shape, dtype=np.float32)
        x_old = np.zeros(deconv.full_shape, dtype=np.float32)
        _, converged = deconv._check_convergence(
            x_new, x_old, k=0, num_iter=100, tol=0.5
        )
        assert converged is False

    def test_return_is_tuple_float_bool(self, deconv):
        """Return value must be (float, bool)."""
        x = np.ones(deconv.full_shape, dtype=np.float32)
        result = deconv._check_convergence(x, x.copy(), k=0, num_iter=10, tol=1e-4)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], bool)

    def test_zero_x_new_uses_eps_floor(self, deconv):
        """x_new = 0 must not raise (eps denominator floor guards division)."""
        x_new = np.zeros(deconv.full_shape, dtype=np.float32)
        x_old = np.ones(deconv.full_shape, dtype=np.float32)
        rel_chg, _ = deconv._check_convergence(
            x_new, x_old, k=0, num_iter=10, tol=1e-6
        )
        assert np.isfinite(rel_chg), "rel_change should be finite even when x_new=0"

    def test_nan_iterate_raises_floating_point_error(self, deconv):
        """Non-finite convergence norms must be treated as numerical failure."""
        x_new = np.ones(deconv.full_shape, dtype=np.float32)
        x_new[0, 0] = np.nan
        x_old = np.ones(deconv.full_shape, dtype=np.float32)
        with pytest.raises(FloatingPointError):
            deconv._check_convergence(x_new, x_old, k=0, num_iter=10, tol=1e-6)

    def test_inf_iterate_raises_floating_point_error(self, deconv):
        """Infinite iterates must not be interpreted as a valid residual."""
        x_new = np.ones(deconv.full_shape, dtype=np.float32)
        x_old = np.ones(deconv.full_shape, dtype=np.float32)
        x_old[0, 0] = np.inf
        with pytest.raises(FloatingPointError):
            deconv._check_convergence(x_new, x_old, k=0, num_iter=10, tol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 7. _crop_and_return helper
# ══════════════════════════════════════════════════════════════════════════════

class TestCropAndReturn:
    """Verify _crop_and_return output type, shape, and state update."""

    def test_returns_numpy_array(self, deconv):
        """_crop_and_return must return numpy.ndarray."""
        x_k = np.ones(deconv.full_shape, dtype=np.float32)
        result = deconv._crop_and_return(x_k)
        assert isinstance(result, np.ndarray)

    def test_output_shape_is_h_by_w(self, deconv):
        """_crop_and_return must crop to (h, w)."""
        x_k = np.ones(deconv.full_shape, dtype=np.float32)
        result = deconv._crop_and_return(x_k)
        assert result.shape == (deconv.h, deconv.w)

    def test_updates_estimated_image(self, deconv):
        """_crop_and_return must store x_k as estimated_image."""
        sentinel_val = np.float32(7.77)
        x_k = np.full(deconv.full_shape, sentinel_val, dtype=np.float32)
        deconv._crop_and_return(x_k)
        np.testing.assert_array_equal(
            np.asarray(deconv.estimated_image),
            np.full(deconv.full_shape, sentinel_val, dtype=np.float32),
            err_msg="_crop_and_return did not update estimated_image",
        )

    def test_stored_copy_is_independent(self, deconv):
        """The stored estimated_image must be a copy, not a view of x_k."""
        x_k = np.ones(deconv.full_shape, dtype=np.float32)
        deconv._crop_and_return(x_k)
        # Mutate x_k after calling _crop_and_return
        x_k[:] = 0.0
        # estimated_image should still hold the old values
        est_min = float(np.asarray(deconv.estimated_image).min())
        assert est_min > 0.0, (
            "estimated_image was mutated when x_k changed — not a copy"
        )

    def test_inverse_normalize_maps_cropped_output_only(self, gaussian_psf):
        image = np.linspace(10.0, 30.0, 6 * 8, dtype=np.float64).reshape(6, 8)
        deconv = _TestDeconv(image, gaussian_psf)
        x_k = np.full(deconv.full_shape, np.float32(0.25), dtype=np.float32)

        result = deconv._crop_and_return(x_k, inverse_normalize=True)

        expected_val = np.float32(
            deconv._gray_min + 0.25 * (deconv._gray_max - deconv._gray_min)
        )
        np.testing.assert_allclose(
            result,
            np.full((deconv.h, deconv.w), expected_val, dtype=result.dtype),
            atol=1e-6,
        )
        np.testing.assert_array_equal(np.asarray(deconv.estimated_image), x_k)


# ══════════════════════════════════════════════════════════════════════════════
# 8. _INIT_KEYS class attribute
# ══════════════════════════════════════════════════════════════════════════════

class TestInitKeys:
    """Verify _INIT_KEYS contains the correct constructor parameter names."""

    def test_init_keys_is_iterable(self):
        """_INIT_KEYS must be iterable (set-like)."""
        assert hasattr(DeconvBase._INIT_KEYS, "__contains__")

    def test_init_keys_contains_expected(self):
        """_INIT_KEYS must contain all six constructor keyword parameters."""
        expected = {
            "paddingMode",
            "padding_scale",
            "initialEstimate",
            "apply_taper_on_padding_band",
            "htm_floor_frac",
            "use_mask",
        }
        missing = expected - set(DeconvBase._INIT_KEYS)
        assert not missing, f"_INIT_KEYS is missing: {missing}"

    def test_init_keys_excludes_image_and_psf(self):
        """image and psf must NOT appear in _INIT_KEYS."""
        assert "image" not in DeconvBase._INIT_KEYS
        assert "psf" not in DeconvBase._INIT_KEYS

    def test_init_keys_excludes_self(self):
        """'self' must not appear in _INIT_KEYS."""
        assert "self" not in DeconvBase._INIT_KEYS

    def test_init_keys_kwarg_split_works(self, test_image, gaussian_psf):
        """
        Kwargs split via _INIT_KEYS must produce a valid constructor call
        and leave the remainder for deblur().
        """
        all_kwargs = {
            "paddingMode": "Zero",
            "padding_scale": 1.5,
            "use_mask": False,
            "some_deblur_param": 42,
        }
        init_kw = {k: v for k, v in all_kwargs.items()
                   if k in DeconvBase._INIT_KEYS}
        deblur_kw = {k: v for k, v in all_kwargs.items()
                     if k not in DeconvBase._INIT_KEYS}

        # init_kw should contain the three constructor params
        assert set(init_kw.keys()) == {"paddingMode", "padding_scale", "use_mask"}
        # deblur_kw should contain only the non-init param
        assert set(deblur_kw.keys()) == {"some_deblur_param"}

        # And it must be possible to actually construct the object
        obj = _TestDeconv(test_image, gaussian_psf, **init_kw)
        assert obj.use_mask is False
