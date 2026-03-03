# Reconstruction Package Documentation

## Package Overview

Modular deconvolution algorithms for Earth observation satellite imagery.
All algorithms share a common base class (`DeconvBase`) that handles image
preprocessing, padded-canvas construction, binary mask M for unknown-boundary
masking, PSF conditioning, frequency-domain precomputation, and GPU/CPU backend
selection.

## Implemented Algorithms

| Algorithm | Class | Wrapper | Status |
|-----------|-------|---------|--------|
| Wiener filter | `WienerDeconv` | `wiener_deblur` | ✓ Complete |
| Richardson-Lucy (masked) | `RLUnknownBoundary` | `rl_deblur_unknown_boundary` | ✓ Complete |
| Landweber / FISTA | `LandweberUnknownBoundary` | `landweber_deblur_unknown_boundary` | ✓ Complete |
| ADMM-TV | `ADMMDeconv` | `admm_deblur` | ✓ Complete |
| TVAL3 | `TVAL3Deconv` | `tval3_deblur` | ✓ Complete |
| PnP-ADMM (BM3D) | `PnPADMM` | `pnp_admm_deblur` | ✓ Complete (requires `bm3d`) |

## Package Structure

```
Reconstruction/
├── __init__.py                    Phase 6 — public API re-exports
├── _backend.py                    Phase 1 — GPU detection, xp/fft backend, utilities
├── _tv_operators.py               Phase 2 — gradient, divergence, TV prox, periodic BC ops
├── _base.py                       Phase 3 — DeconvBase abstract class
├── rl_unknown_boundary.py         Phase 4a — RL with unknown boundaries
├── landweber_unknown_boundary.py  Phase 4b — FISTA/Landweber with TV
├── wiener.py                      Phase 5a — Wiener deconvolution
├── admm.py                        Phase 5c — ADMM-TV with overridable prior interface
├── tval3.py                       Phase 5d — TVAL3 with adaptive TV weights
└── pnp_admm.py                    Phase 5e — Plug-and-Play ADMM with BM3D

tests/
├── conftest.py                    Shared fixtures + Shared.Common mock stubs
├── test_backend.py                Phase 1 verification (31 tests)
├── test_tv_operators.py           Phase 2 verification (27 tests)
├── test_base.py                   Phase 3 verification (44 tests)
├── test_rl.py                     Phase 4a regression tests
├── test_landweber.py              Phase 4b regression tests
├── test_wiener.py                 Phase 5a verification (196 tests)
├── test_admm.py                   Phase 5c verification (44 tests)
├── test_tval3.py                  Phase 5d verification (38 tests)
├── test_pnp_admm.py               Phase 5e verification (50 tests, skipped if bm3d absent)
├── test_package_api.py            Phase 6/7 public API verification
└── test_integration.py            Phase 7 cross-algorithm smoke tests
```

## Reference Files

- **RECONSTRUCTION_SPEC.py** — Complete architecture specification and phased
  implementation plan.  Read before modifying any phase.

- **reference/** — Original standalone implementations for regression testing:
  - `RL_Unknown_Boundary.py` — Corrected RL with 7 bug fixes (567 lines).
  - `Landweber_Unknown_Boundary.py` — FISTA + proximal TV (757 lines).

## Quick Start

```python
# One-shot wrapper
from Reconstruction import admm_deblur
result = admm_deblur(image, psf, iters=100, lambda_tv=0.01)

# Class-based (fine-grained control)
from Reconstruction import ADMMDeconv
solver = ADMMDeconv(image, psf, rho_v=32.0, rho_w=32.0)
result = solver.deblur(num_iter=100, lambda_tv=0.01, TVnorm=2)
print(solver.cost_history[-1])   # final cost
print(solver.last_rho_v)         # final adaptive penalty

# PnP-ADMM (requires: pip install bm3d)
from Reconstruction import pnp_admm_deblur
result = pnp_admm_deblur(image, psf, iters=50, lambda_tv=0.01,
                          rho_z=2.0, sigma_scale=1.0)

# GPU backend
from Reconstruction import set_backend
set_backend("gpu")
```

## Installation

```bash
# Core (CPU only)
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With PnP-ADMM (BM3D denoiser)
pip install -e ".[pnp]"

# Everything
pip install -e ".[all]"
```

## Running Tests

```bash
conda activate env_py311
pytest tests/ -v
```
