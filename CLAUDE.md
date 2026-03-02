# Reconstruction — Deconvolution Algorithms for Satellite Imagery

## Project Overview

This package implements modular deconvolution (image reconstruction) algorithms for Earth observation satellite imagery. All algorithms share a common infrastructure (GPU/CPU backend, FFT, PSF conditioning, padded canvas, unknown-boundary mask) and differ only in their iteration strategy.

The package is being built from two existing standalone files (`RL_Unknown_Boundary.py` and `Landweber_Unknown_Boundary.py`) following a phased refactoring plan defined in `docs/RECONSTRUCTION_SPEC.py`. **Always read the spec before implementing any phase.**

## Architecture

```
Reconstruction/
├── __init__.py                         # Public API re-exports
├── _backend.py                         # GPU detection, xp/fft backend, FFT helpers, utilities
├── _tv_operators.py                    # Gradient, divergence, Chambolle prox_TV, Dey multiplicative TV
├── _base.py                            # DeconvBase abstract class (shared constructor + interface)
├── rl_unknown_boundary.py              # RLUnknownBoundary(DeconvBase)
├── landweber_unknown_boundary.py       # LandweberUnknownBoundary(DeconvBase)
├── wiener.py                           # WienerDeconv(DeconvBase)
├── rl_standard.py                      # RLStandard(DeconvBase)
├── admm.py                             # ADMMDeconv(DeconvBase)
└── tval3.py                            # TVAL3Deconv(DeconvBase)
```

**Dependency flow** (strict — no circular imports):
- `_backend.py` → depends on nothing in this package
- `_tv_operators.py` → imports from `_backend`
- `_base.py` → imports from `_backend`
- Algorithm files → import from `_backend`, `_base`, and `_tv_operators`

**Key design principle**: All algorithms inherit from `DeconvBase`. The base class owns the entire forward-model setup (image preprocessing, padding, mask, PSF FFT, H^T M precomputation). Algorithm subclasses implement only `deblur()`.

## Implementation Plan

The full specification is in `docs/RECONSTRUCTION_SPEC.py`. Implementation follows 7 phases in strict order:

1. `_backend.py` — extract shared GPU/FFT infrastructure
2. `_tv_operators.py` — extract TV operators (gradient, divergence, Chambolle prox, Dey correction)
3. `_base.py` — build abstract base class with shared constructor
4. `rl_unknown_boundary.py` + `landweber_unknown_boundary.py` — refactor existing algorithms as thin subclasses
5. `wiener.py`, `rl_standard.py`, `admm.py`, `tval3.py` — new algorithms (independent, any order)
6. `__init__.py` — public API
7. Integration testing + cleanup

**Each phase ends with a verification step defined in the spec. Run verifications before proceeding to the next phase.**

## Reference Files

These are the existing implementations being refactored. They live in `docs/reference/` and must NOT be modified:

- `docs/reference/RL_Unknown_Boundary.py` — corrected RL with 7 bug fixes applied (567 lines)
- `docs/reference/Landweber_Unknown_Boundary.py` — FISTA + proximal TV (757 lines)

When implementing, **match the behavior of these files exactly** for Phases 4a/4b. The regression tests must show bit-identical output (within float32 tolerance).

## External Dependencies (from the broader project)

The package imports from `RemondoPythonCore.Common`, which is outside this repository:

```python
from RemondoPythonCore.Common.General_Utilities   import padding, cropping, odd_crop_around_center
from RemondoPythonCore.Common.PSF_Preprocessing  import psf_preprocess, condition_psf
from RemondoPythonCore.Common.Image_Preprocessing import image_normalization, validate_image, to_grayscale
```

These are assumed to exist and work correctly. Do NOT reimplement them. If writing tests that need to run standalone (without `RemondoPythonCore.Common`), create mock/stub versions in `tests/conftest.py`.

## Coding Conventions

### Imports
- Every file starts with `from __future__ import annotations`.
- Order: standard library → third-party → local, separated by blank lines.
- NumPy is always `import numpy as np`.
- For array operations in algorithm files, always use `xp` (imported from `_backend`). Never import numpy or cupy directly for array math.

### Type Hints
- Use lowercase generics: `tuple[int, int]`, not `Tuple[int, int]`.
- Use `Optional[X]` from typing (not `X | None`) for Python 3.9 forward compatibility.
- Public API signatures use `np.ndarray` (CPU). Internal arrays use `xp.ndarray` (may be GPU).

### Logging
- Each module: `logger = logging.getLogger(__name__)`.
- `logger.debug` for per-iteration details, setup info, internal state.
- `logger.info` for convergence events, summary statistics, backend selection.
- `logger.warning` for fallbacks, potential issues, deprecation notices.
- **Never use `print()`.**

### Numerical Precision
- All internal computation in `float32` (`xp.float32`).
- Epsilon constants stored as `xp.float32` scalars for backend consistency.

### Docstrings
- NumPy-style docstrings on all public classes and functions.
- Include `Parameters`, `Returns`, `Notes`, `References` sections as appropriate.
- Mathematical formulae in ASCII/Unicode (e.g., `x_{k+1} = x_k * (H^T ratio) / (H^T M + eps)`).
- Reference papers with: Author(s), Journal, Year, Equation number.

### Section Headers
Use the box-drawing style consistent with the existing codebase:
```python
# ══════════════════════════════════════════════════════════════════════════════
# Section Title
# ══════════════════════════════════════════════════════════════════════════════
```

### Code Style
- No trailing whitespace.
- One blank line between methods within a class.
- Two blank lines between top-level definitions (functions, classes).
- Maximum line length: 88 characters (Black default), with exceptions for long comments.
- Use `del` to free large temporary arrays explicitly (aids GPU memory management).

## Testing

Tests live in `tests/` and run with `pytest`:

```bash
conda activate reconstruction
pytest tests/ -v
```

### Test structure:
- `tests/conftest.py` — shared fixtures (synthetic images, Gaussian PSFs, mock RemondoPythonCore.Common utilities)
- `tests/test_backend.py` — Phase 1 verification
- `tests/test_tv_operators.py` — Phase 2 verification (adjointness, prox identity, denoising)
- `tests/test_base.py` — Phase 3 verification (constructor, shapes, mask, HTM floor)
- `tests/test_rl.py` — Phase 4a regression test
- `tests/test_landweber.py` — Phase 4b regression test
- `tests/test_wiener.py` — Phase 5a
- `tests/test_rl_standard.py` — Phase 5b
- `tests/test_admm.py` — Phase 5c
- `tests/test_tval3.py` — Phase 5d

### Writing tests:
- Each test file corresponds to one phase in the spec.
- Use `pytest.approx` or `np.testing.assert_allclose` for numerical comparisons.
- Keep tolerances at `atol=1e-5` for float32 computations.
- For regression tests (Phases 4a, 4b): compare new package output against stored reference output.

## Domain Context

This is a satellite imaging project. Key concepts:

- **PSF** (Point Spread Function): the blur kernel of the optical system. Always normalized to sum=1.
- **Deconvolution**: inverse problem — recovering the sharp image from the blurred observation.
- **Unknown boundaries**: the observed image is a crop of a larger scene. Pixels outside the observed region are unknown and must be extrapolated. The mask M = 1 on observed pixels, 0 outside.
- **H^T M**: the adjoint PSF applied to the mask. It measures how much of the PSF footprint overlaps the observed region at each pixel. Near-zero values outside the mask indicate pixels with no data constraint — the HTM floor clamp prevents division blow-up there.
- **TV (Total Variation)**: edge-preserving regularization. Penalizes the L1 norm of the image gradient, which promotes piecewise-smooth solutions.
- **FISTA**: Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle, 2009). Nesterov-accelerated proximal gradient with O(1/k²) convergence.
- **Proximal operator**: `prox_{γf}(v) = argmin_u (1/2)||u-v||² + γf(u)`. For TV, this is the ROF denoising problem, solved via Chambolle's dual projection.

## Common Pitfalls

1. **Never import cupy directly in algorithm files.** Always use `xp` from `_backend`. The backend may be numpy on CPU-only systems.
2. **The mask centering assumes `padding()` centres via `(canvas - image) // 2`.** If this assumption is wrong, the mask will be mis-registered by ±1 pixel. This is documented but must be verified.
3. **`irfft2` requires the `s` parameter** (output spatial shape) because `W//2+1` is ambiguous between even/odd `W`.
4. **`ifftshift` (not `fftshift`)** is used to move the PSF centre to `[0,0]` for FFT-based convolution.
5. **Float32 precision**: accumulation errors grow over hundreds of iterations. Epsilon values must be `xp.float32` scalars, not Python floats, to avoid silent promotion to float64.
6. **The `_freeze` function** marks arrays as read-only. Precomputed arrays (PF, conjPF, HTM, mask) should always be frozen to catch accidental mutation.
7. **CuPy memory**: explicitly `del` large temporaries and call `cp.get_default_memory_pool().free_all_blocks()` if memory is tight.
