"""
_common.py — Compatibility imports for shared preprocessing utilities.

The reconstruction package is developed inside the larger RemondoPythonCore
codebase, where these utilities live under ``RemondoPythonCore.Common``.
Some historical scripts still expose the same helpers under ``Shared.Common``.

This module provides a single consistent import surface for reconstruction's
internal use and raises a clear error when neither namespace is available.
"""
from __future__ import annotations

try:
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
except ImportError as remondo_exc:
    try:
        from Shared.Common.General_Utilities import padding, cropping
        from Shared.Common.PSF_Preprocessing import psf_preprocess, condition_psf
        from Shared.Common.Image_Preprocessing import (
            image_normalization,
            validate_image,
            to_grayscale,
            odd_crop_around_center,
        )
    except ImportError as shared_exc:
        raise ImportError(
            "Reconstruction solver modules require the shared preprocessing "
            "utilities from 'RemondoPythonCore.Common' (preferred) or "
            "'Shared.Common' (legacy). Install/use the full "
            "RemondoPythonCore package layout before importing solver "
            "modules."
        ) from shared_exc
