# Reconstruction Package Documentation

## Package Overview

Modular deconvolution algorithms for Earth observation satellite imagery.
All algorithms share a common base class (`DeconvBase`) that handles image
preprocessing, padded-canvas construction, binary mask M for unknown-boundary
masking, PSF conditioning, frequency-domain precomputation, and GPU/CPU backend
selection.

## Boundary Handling

The package does not currently expose a single public `boundary_policy` API.
Instead, the effective boundary model is solver-family specific and is built
from four layers:

1. image padding onto an extended FFT canvas
2. optional masking of the observed field of view
3. circular FFT convolution on the padded canvas
4. solver-specific TV / gradient operator assumptions

### Common padded-canvas structure

All solvers work on an extended canvas constructed by `DeconvBase`.
The observed image is embedded into that canvas using the selected
`paddingMode`. The PSF is zero-padded to the same canvas and shifted with
`ifftshift` before FFT placement, so the forward operator `H` is always a
circular convolution on the padded canvas.

For the iterative solvers, the binary mask `M` is then used to restrict the
data-fidelity term to the original observed support. This means the package's
"unknown-boundary" abstraction is primarily a masked-fidelity abstraction, not
a claim that every internal operator uses the same non-periodic boundary
condition.

### Solver-family contract

- `WienerDeconv`:
  padded / tapered circular deconvolution with no masked unknown-boundary
  fidelity. It hard-forces `use_mask=False` and
  `apply_taper_on_padding_band=True`.

- `RLUnknownBoundary`, `LandweberUnknownBoundary`, `FISTADeconv`:
  masked fidelity on the original support over a padded FFT canvas.
  When TV is active, the TV step belongs to the Neumann family:
  RL uses the Dey-style multiplicative TV correction, and
  Landweber / FISTA use Chambolle's TV proximal solver.

- `ChambollePockDeconv`, `ADMMDeconv`, `TVAL3Deconv`:
  masked fidelity on the original support over a padded FFT canvas, with
  periodic gradient / divergence operators. The periodic operator choice is
  required because these solvers place `∇` and `∇^T∇` directly inside
  Fourier-diagonalized updates.

- `PnPADMM`, `REDDeconv`:
  inherit ADMM's masked-fidelity and padded-FFT structure. They do not
  currently define a separate public TV-boundary contract because their prior
  step is denoiser-based rather than TV-prox based.

### Practical implication

Two solvers may both be "unknown-boundary" solvers in the package sense while
still using different internal regularizer boundary assumptions. Cross-solver
comparisons should therefore be interpreted as comparisons between related, but
not identical, boundary models.

## PSF Handling

The current package default is a PSF preprocessing policy, not a pass-through
contract for the user-supplied PSF array.

For the iterative-family solvers (`RLUnknownBoundary`,
`LandweberUnknownBoundary`, `FISTADeconv`, `ChambollePockDeconv`,
`ADMMDeconv`, `TVAL3Deconv`, `PnPADMM`, `REDDeconv`), the default PSF pipeline
is:

1. Centre by centre of mass (`center_method="com"`).
2. Clip negative values (`remove_negatives="clip"`).
3. Enforce odd spatial shape (`enforce_odd_shape=True`).
4. Condition the PSF by residual-background subtraction and outer radial taper
   with `bg_ring_frac=0.15`, `taper_outer_frac=0.20`,
   `taper_end_frac=0.50`.
5. Zero-pad to the FFT canvas (`Type="Zero"`, `apply_taper=False`).
6. Apply `ifftshift` before FFT placement.

This policy is scientifically destructive in the sense that it may alter the
submitted PSF centroid, support, negativity pattern, and wing amplitudes before
the forward model is built.

`WienerDeconv` shares the same centering, clipping, odd-shape enforcement,
zero-padding, and `ifftshift` placement steps, but it currently uses a
solver-specific conditioning override:

- iterative-family default:
  `bg_ring_frac=0.15`, `taper_outer_frac=0.20`, `taper_end_frac=0.50`
- Wiener override:
  `bg_ring_frac=0.15`, `taper_outer_frac=0.90`, `taper_end_frac=1.0`

So Wiener and the iterative-family solvers do not, by default, operate on
identically conditioned PSFs.

## Statefulness and Repeated Calls

The package currently has two distinct calling contracts:

- Class instances for the iterative solvers
  (`RLUnknownBoundary`, `LandweberUnknownBoundary`, `FISTADeconv`,
  `ChambollePockDeconv`, `ADMMDeconv`, `TVAL3Deconv`, `PnPADMM`,
  `REDDeconv`) are stateful warm-start solvers.
  After a successful `deblur()` call, the final padded iterate is stored in
  `estimated_image`, and the next object-level `deblur()` call starts from
  that stored iterate by default.

- Wrapper functions are stateless cold-start helpers.
  Each wrapper call constructs a fresh solver object, runs `deblur()`, and
  discards the internal state afterwards.

- `WienerDeconv` is stateful for setup and diagnostics, but not for iterate
  warm starts.
  Repeated calls reuse constructor-time FFT setup and update diagnostic
  fields such as `last_alpha` / `sigma_est`, but they do not read the prior
  `estimated_image` as an initial iterate because Wiener is not iterative.

- Numerical failure preserves finite state, but not necessarily the exact
  pre-call state.
  If an iterative run fails after making finite progress, the solver may keep
  the last verified finite iterate reached during that failed call rather than
  rolling all the way back to the pre-call iterate. The contract is
  "state remains finite and reusable", not "state is unchanged on failure".

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
