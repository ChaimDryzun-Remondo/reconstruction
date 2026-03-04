"""
Reconstruction — Modular deconvolution algorithms for satellite imagery.

All algorithms share a common forward-model base class (:class:`~._base.DeconvBase`)
that handles image preprocessing, padded-canvas construction, binary mask M for
unknown-boundary masking, PSF conditioning and frequency-domain precomputation,
and GPU/CPU backend selection.  Algorithm subclasses implement only the
iteration loop.

Quick start
-----------
One-shot wrapper (simplest interface)::

    from Reconstruction import rl_deblur_unknown_boundary
    result = rl_deblur_unknown_boundary(image, psf, iters=100)

Class-based (fine-grained control over hyperparameters)::

    from Reconstruction import RLUnknownBoundary
    rl = RLUnknownBoundary(image, psf, padding_scale=3.0)
    result = rl.deblur(num_iter=100, lambda_tv=0.0002)

Backend selection::

    from Reconstruction import set_backend
    set_backend("gpu")   # use CuPy (requires cupy-cuda12x)
    set_backend("cpu")   # force NumPy (default)
    set_backend("auto")  # GPU if available, else CPU

Algorithm summary
-----------------
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| Algorithm                 | Class                     | Wrapper                            | Use case                                      |
+===========================+===========================+====================================+===============================================+
| Wiener filter             | WienerDeconv              | wiener_deblur                      | Fast single-pass baseline, linear             |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| Richardson-Lucy (masked)  | RLUnknownBoundary         | rl_deblur_unknown_boundary         | Poisson noise, unknown boundaries             |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| Landweber / FISTA         | LandweberUnknownBoundary  | landweber_deblur_unknown_boundary  | Gaussian noise, TV regularization, unknown    |
|                           |                           |                                    | boundaries                                    |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| ADMM-TV                   | ADMMDeconv                | admm_deblur                        | TV regularization, masked ADMM, extensible    |
|                           |                           |                                    | prior interface                               |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| TVAL3                     | TVAL3Deconv               | tval3_deblur                       | TV, exact FFT solve, adaptive spatially-      |
|                           |                           |                                    | varying weights                               |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| PnP-ADMM                  | PnPADMM                   | pnp_admm_deblur                    | BM3D denoiser prior; requires bm3d package    |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| FISTA                     | FISTADeconv               | fista_deblur                       | TV / L1 / wavelet sparsity; textbook          |
|                           |                           |                                    | Beck-Teboulle 2009; overridable prox step     |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| Chambolle-Pock            | ChambollePockDeconv       | chambolle_pock_deblur              | TV; Condat-Vũ primal-dual forward-backward;   |
| (Condat-Vũ)               |                           |                                    | isotropic / anisotropic TV; no proxG needed   |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+
| RED-ADMM                  | REDDeconv                 | red_deblur                         | Regularization by Denoising [REM17]; BM3D     |
|                           |                           |                                    | prior; fixed σ; requires bm3d package         |
+---------------------------+---------------------------+------------------------------------+-----------------------------------------------+

PnP-ADMM and RED-ADMM are conditionally available — they require
``pip install bm3d``.  The rest of the package imports and runs without it.
"""
from __future__ import annotations

# ── Backend control ────────────────────────────────────────────────────────
from ._backend import set_backend

# ── Core algorithms ────────────────────────────────────────────────────────
from .wiener import WienerDeconv, wiener_deblur
from .rl_unknown_boundary import RLUnknownBoundary, rl_deblur_unknown_boundary
from .landweber_unknown_boundary import (
    LandweberUnknownBoundary,
    landweber_deblur_unknown_boundary,
)
from .admm import ADMMDeconv, admm_deblur
from .tval3 import TVAL3Deconv, tval3_deblur
from .fista import FISTADeconv, fista_deblur
from .chambolle_pock import ChambollePockDeconv, chambolle_pock_deblur

# ── Optional BM3D-based algorithms (require bm3d) ─────────────────────────
# The bm3d package is an optional dependency (pip install bm3d).
# If it is not installed we silently skip both imports so that the package
# is usable without BM3D.
try:
    from .pnp_admm import PnPADMM, pnp_admm_deblur
    _HAS_PNP: bool = True
except ImportError:
    _HAS_PNP = False

try:
    from .red_admm import REDDeconv, red_deblur
    _HAS_RED: bool = True
except ImportError:
    _HAS_RED = False

# ── Public API ─────────────────────────────────────────────────────────────
__version__: str = "0.1.0"

__all__ = [
    # backend control
    "set_backend",
    # core classes
    "WienerDeconv",
    "RLUnknownBoundary",
    "LandweberUnknownBoundary",
    "ADMMDeconv",
    "TVAL3Deconv",
    "FISTADeconv",
    "ChambollePockDeconv",
    # core wrappers
    "wiener_deblur",
    "rl_deblur_unknown_boundary",
    "landweber_deblur_unknown_boundary",
    "admm_deblur",
    "tval3_deblur",
    "fista_deblur",
    "chambolle_pock_deblur",
    # package metadata
    "__version__",
]

if _HAS_PNP:
    __all__ += ["PnPADMM", "pnp_admm_deblur"]

if _HAS_RED:
    __all__ += ["REDDeconv", "red_deblur"]
