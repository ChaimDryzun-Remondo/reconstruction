"""
_common.py — Compatibility imports for shared preprocessing utilities.

The reconstruction package is developed inside the larger RemondoPythonCore
codebase, where these utilities live under ``RemondoPythonCore.Common``.
Some historical scripts still expose the same helpers under ``Shared.Common``.

This module provides a single consistent import surface for reconstruction's
internal use and raises a clear error when neither namespace is available.
"""
from __future__ import annotations

def _is_missing_namespace(exc: ImportError, root: str) -> bool:
    """
    Return True only when the namespace itself is absent.

    Nested/transitive import failures inside an available namespace must not be
    treated as a fallback trigger.
    """
    if not isinstance(exc, ModuleNotFoundError):
        return False
    missing = getattr(exc, "name", "") or ""
    return missing == root or missing.startswith(f"{root}.")


def _import_remondo_common():
    from RemondoPythonCore.Common.General_Utilities import (
        padding,
        cropping,
        odd_crop_around_center,
    )
    from RemondoPythonCore.Common.PSF_Preprocessing import (
        psf_preprocess,
        condition_psf,
    )
    from RemondoPythonCore.Common.Image_Preprocessing import (
        image_normalization,
        validate_image,
        to_grayscale,
    )
    return (
        padding,
        cropping,
        odd_crop_around_center,
        psf_preprocess,
        condition_psf,
        image_normalization,
        validate_image,
        to_grayscale,
    )


def _import_shared_common():
    from Shared.Common.General_Utilities import padding, cropping
    from Shared.Common.PSF_Preprocessing import psf_preprocess, condition_psf
    from Shared.Common.Image_Preprocessing import (
        image_normalization,
        validate_image,
        to_grayscale,
        odd_crop_around_center,
    )
    return (
        padding,
        cropping,
        odd_crop_around_center,
        psf_preprocess,
        condition_psf,
        image_normalization,
        validate_image,
        to_grayscale,
    )


try:
    (
        padding,
        cropping,
        odd_crop_around_center,
        psf_preprocess,
        condition_psf,
        image_normalization,
        validate_image,
        to_grayscale,
    ) = _import_remondo_common()
except ImportError as remondo_exc:
    if not _is_missing_namespace(remondo_exc, "RemondoPythonCore"):
        raise
    try:
        (
            padding,
            cropping,
            odd_crop_around_center,
            psf_preprocess,
            condition_psf,
            image_normalization,
            validate_image,
            to_grayscale,
        ) = _import_shared_common()
    except ImportError as shared_exc:
        if not _is_missing_namespace(shared_exc, "Shared"):
            raise
        raise ImportError(
            "Reconstruction solver modules require the shared preprocessing "
            "utilities from 'RemondoPythonCore.Common' (preferred) or "
            "'Shared.Common' (legacy). Install/use the full "
            "RemondoPythonCore package layout before importing solver "
            "modules."
        ) from shared_exc
