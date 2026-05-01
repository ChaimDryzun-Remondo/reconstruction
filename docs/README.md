# Reconstruction Package Documentation

## Package Overview

Modular deconvolution algorithms for Earth observation satellite imagery.
All algorithms share a common base class (`DeconvBase`) that handles image
preprocessing, padded-canvas construction, binary mask M for unknown-boundary
masking, PSF conditioning, frequency-domain precomputation, and GPU/CPU backend
selection.

## Working Domain

The package working domain is currently:

- grayscale
- odd-cropped in each spatial dimension if needed
- affine-normalized to `[0, 1]` for non-degenerate images

For degenerate constant images, the affine normalization map is undefined.
The explicit fallback is therefore:

- use the grayscale, odd-cropped image in raw units

`initialEstimate`, if provided, is transformed into that same working domain:

- non-degenerate image: apply the image-derived affine map
- constant image: apply the same identity fallback as the observed image

This means a non-degenerate image can still produce an `initialEstimate`
outside `[0, 1]` when the supplied initial estimate lies outside the observed
image range. For constant images, both the image working domain and the
`initialEstimate` remain in grayscale raw units.

Returned outputs are, by default, left in that same internal working domain.
The public `deblur()` methods also expose `inverse_normalize=False` as an
output-only flag:

- non-degenerate image + `inverse_normalize=False`: return working-domain
  grayscale values
- non-degenerate image + `inverse_normalize=True`: return the odd-cropped
  grayscale output mapped back to the observed image's raw-unit grayscale scale
- constant image: the working domain is already raw grayscale units, so
  `inverse_normalize=True` is a no-op

This flag changes only the returned cropped array. Persistent solver state
remains in the package working domain.

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
`ADMMDeconv`, `TVAL3Deconv`, `PnPADMM`, `REDDeconv`), the active PSF pipeline
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

`WienerDeconv` shares the same preprocessing and placement steps:

1. Centre by centre of mass (`center_method="com"`).
2. Clip negative values (`remove_negatives="clip"`).
3. Enforce odd spatial shape (`enforce_odd_shape=True`).
4. Zero-pad to the FFT canvas (`Type="Zero"`, `apply_taper=False`).
5. Apply `ifftshift` before FFT placement.

The scientific difference is the conditioning preset used for the final active
PSF spectrum:

- iterative-family default:
  `bg_ring_frac=0.15`, `taper_outer_frac=0.20`, `taper_end_frac=0.50`
- Wiener active PF:
  `bg_ring_frac=0.15`, `taper_outer_frac=0.90`, `taper_end_frac=1.0`

So Wiener and the iterative-family solvers share centering, clipping,
odd-shape enforcement, zero-padding, and `ifftshift` placement, but do not, by
default, operate on identically conditioned PSFs.

Current constructor-time implementation detail:
`WienerDeconv` first executes `DeconvBase.__init__()`, which constructs the
iterative-family PSF spectrum with the `0.15 / 0.20 / 0.50` conditioning
preset, and then immediately rebuilds and overwrites `PF` / `conjPF` from the
original user PSF with the Wiener-specific `0.15 / 0.90 / 1.0` conditioning
preset. The final active scientific contract is the overwritten Wiener
conditioned PF, not the intermediate base-class PF.

## Statefulness and Repeated Calls

The package uses one shared repeated-call contract with three cases:

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

- `WienerDeconv` is stateful for setup and diagnostics, but not for
  iterate warm starts.
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

## Recent Refactor (bug fixes + code optimization)

The items below landed as one atomic commit per finding in a single
refactor pass.  Each commit ran the full pytest suite with no
regression — the count stayed at **652 passed** on every step, with
4 pre-existing facade test failures that were resolved in a subsequent
dedicated pass; see `tests/test_import_smoke.py` and the
*Import entry point* section below.

### Correctness bugs (P0)

| ID  | File(s)                            | Summary |
|-----|------------------------------------|---------|
| F1  | `rl_unknown_boundary.py`           | Return the improved iterate on convergence — the old code broke out of the loop *before* advancing state, so `_crop_and_return` returned the stale previous iterate even though the convergence test had just validated the new one. |
| F2  | `landweber_unknown_boundary.py`    | Same pattern — advance the full FISTA state (`x_km1`, `x_k`, `z_k`, `t_k`, `last_finite`) before the `break` on convergence. |
| F3  | `wiener.py`                        | Recompute `HTM` and `_lipschitz` from the active Wiener PF.  The base class set both from the iterative-family PF before Wiener's PF/conjPF override, leaving stale values bound to the wrong spectrum.  Wiener itself never reads these fields, but any FISTA-style subclass of `WienerDeconv` would inherit numerically wrong values — this closes the latent trap. |

### Performance / silent-correctness risks (P1)

| ID  | File(s)                                                       | Summary |
|-----|---------------------------------------------------------------|---------|
| F4  | `admm.py`                                                     | Drop the redundant post-`_prior_update` per-key state sweep.  `prior_rhs` is built from state entries by linear/local operations, so the `prior_rhs` check is a strict superset of the sweep.  Default TV path: 8 full-canvas `isfinite().all()` reductions → 4 per iteration. |
| F5  | `admm.py`, `tval3.py`                                         | Reuse the cached x-update spectrum `U = fft2(rhs) / (denom+eps)` for the forward projection when `nonneg=False` — `u = real(ifft2(U))` means `fft2(u) == U` exactly (rhs and denom are both real).  Saves one `fft2`/iter on the `nonneg=False` path; byte-identical on the default `nonneg=True` path. |
| F6  | `admm.py`, `tval3.py`                                         | Cache `self._mask_f64` and `self._image_f64` once in `__init__` (the mask/image are already frozen by `DeconvBase`).  Removes two float64 allocations per `_compute_cost` call and two more per `deblur()` call. |
| F8  | `admm.py`, `tval3.py`                                         | Drop the redundant `.copy()` after `H_full.conj()` — verified empirically that `.conj()` on complex arrays already allocates a fresh buffer.  Kept the `.copy()` after `xp.real(...)` because `xp.real` on a complex array returns a view that would otherwise hold the complex intermediate alive. |
| F10 | `rl_unknown_boundary.py`, `landweber_unknown_boundary.py`, `fista.py`, `chambolle_pock.py` | Refresh the `last_finite` rollback snapshot at the existing `check_every` cadence rather than every iteration.  On default `check_every=5` this is a 5× reduction in bookkeeping copies.  Convergence detection itself is already gated on the same cadence, so the returned `last_finite` is always consistent with the returned iterate on the success path. |
| F11 | `chambolle_pock.py`                                           | Rename `x_old` → `x_prev` and document the aliasing invariant (the name is an alias of `x`, not a copy, and nothing in the iteration body mutates `x` in place).  No numerical change; removes a review trap. |
| F12 | `fista.py`                                                    | Document the FISTA alias-rotation invariant (`x_km1 = x_k; x_k = x_new`) that the O'Donoghue-Candès restart test depends on.  Comment-only. |

### Cleanup (P2)

| ID  | File(s)                                 | Summary |
|-----|-----------------------------------------|---------|
| F14 | `_tv_operators.py`, `admm.py`, `tval3.py` | Extract the byte-identical `_shrink` method bodies into a single `_tv_operators.shrink_tv` function.  Each solver retains `_shrink` as a thin wrapper because `TestNumericalFailureContract` monkey-patches `solver._shrink` as a fault-injection seam. |
| F15 | `chambolle_pock.py`                     | Return two independent `zeros_like` buffers from `_dual_project` on the `lam <= 0` branch (was `zero, zero.copy()`, half-aliased).  No numerical change. |
| F17 | `rl_unknown_boundary.py`                | Hoist the duplicated `tv_multiplicative_correction(x_k, lam)` call out of the `tv_on_full_canvas` branch — only the *application* of the correction differs. |
| F18 | `fista.py`                              | At FISTA cold-start, `x_km1` and `y_k` can alias `x_k` (the F12 invariant guarantees no in-place mutation before the state advance rebinds them).  Saves 2 full-canvas copies per `deblur()` call.  `last_finite` stays a true copy. |

### Skipped findings (with rationale)

| ID  | Why not |
|-----|---------|
| F7  | Two tests in `TestWienerPSFContract` explicitly encode the double-PSF-pipeline (base pipeline + Wiener override) as an architectural contract — they assert that `condition_psf` is called **twice**, once with iterative-family constants (0.20/0.50) and once with Wiener constants (0.90/1.00).  F3 already closes the correctness trap; the remaining work is a constructor-only cost. |
| F9  | Finding premise incorrect.  All four callers of `_check_convergence` (`RL`, `Landweber`, `FISTA`, `Chambolle-Pock`) already guard with `if k >= min_iter and (k + 1) % check_every == 0:`; ADMM and TVAL3 don't call this function at all.  The proposed inner guard would be dead code. |
| F16 | `np.sqrt` → `math.sqrt` is not byte-equivalent under NumPy 2.x NEP-50 weak promotion.  The Landweber momentum step `z_new = x_new + momentum * (x_new - x_k)` propagates `t_new`'s scalar type into an op with a float32 array; `np.float64` (strong) upcasts the intermediate to float64, while Python `float` (weak) keeps it at float32.  The compounded rounding drift breaks the regression test (max diff 0.021 against the reference).  A "correct" F16 would require wrapping `momentum` in `backend.xp.float64(...)` — saving ~5 ns/iter, invisible against per-iter FFT cost. |
| F20 | Finding premise incorrect.  `Syy = |rfft2(y)|² / N` is the correct per-frequency PSD at each rfft2 bin — verified empirically that it is bit-identical to `|fft2(y)|² / N` at the same bins (max diff 3e-15, pure roundoff).  The factor-of-2 weighting the finding proposed is only needed for Parseval *total-energy sums*, not for the per-frequency α(f) computation the Wiener Spectrum mode performs. |

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
├── conftest.py                    Shared fixtures
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
result = admm_deblur(image, psf, iters=100, lambda_tv=0.001)

# Class-based (fine-grained control)
from Reconstruction import ADMMDeconv
solver = ADMMDeconv(image, psf, rho_v=32.0, rho_w=32.0)
result = solver.deblur(num_iter=100, lambda_tv=0.001, TVnorm=2)
print(solver.cost_history[-1])   # final cost
print(solver.last_rho_v)         # final adaptive penalty

# PnP-ADMM (requires: pip install bm3d)
from Reconstruction import pnp_admm_deblur
result = pnp_admm_deblur(image, psf, iters=50, lambda_tv=0.001,
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

From the submodule root, with the reconstruction conda env active:

```bash
conda activate reconstruction
cd external_reconstruction
python -m pip install -e .
python -m pytest -q tests
```

The default test command now runs the `core` profile only.  Tests marked
`imaging`, `pnp`, or `monorepo` are excluded by default so a standalone core
install does not fail on optional dependencies.

### Supported test profiles

```bash
# Core standalone profile (default)
cd external_reconstruction
python -m pip install -e .
python -m pytest -q tests

# Imaging profile (requires scikit-image and related imaging extras)
cd external_reconstruction
python -m pip install -e ".[imaging]"
python -m pytest -q tests --override-ini="addopts=-v --tb=short" -m imaging

# PnP / RED profile (requires bm3d)
cd external_reconstruction
python -m pip install -e ".[pnp]"
python -m pytest -q tests --override-ini="addopts=-v --tb=short" -m pnp

# Full local standalone sweep (core + imaging + pnp)
cd external_reconstruction
python -m pip install -e ".[imaging,pnp]"
python -m pytest -q tests --override-ini="addopts=-v --tb=short" -m "core or imaging or pnp"

# Monorepo-only tests (reserved for tests that intentionally use the real
# RemondoPythonCore.Common namespace instead of the test mocks)
cd ~/git/RemondoPythonCore
python -m pytest -q external_reconstruction/tests --override-ini="addopts=-v --tb=short" -m monorepo
```

### Validated numerical environments

The package metadata currently allows broad lower-bound compatibility:

- Python `>=3.10`
- NumPy `>=1.24`
- SciPy `>=1.11`

That metadata should be treated as an installation compatibility floor, not as
an already matrix-validated numerical contract.

The core profile has been rechecked in these concrete environments:

- Python `3.10.13`, NumPy `1.26.4`, SciPy `1.12.0`
- Python `3.11.15`, NumPy `2.4.3`, SciPy `1.17.1`

For numerically sensitive regression work, prefer one of those validated
stacks or use a project constraints file. Wider NumPy/SciPy ranges should be
treated as compatibility targets until a broader version matrix is exercised
in CI.

### Import entry point

`import Reconstruction` is the sole supported top-level entry point.
Bare `import reconstruction` (lowercase) is not supported.

Current import contract:

- `import Reconstruction` is lazy and standalone-friendly at the package-root
  level. It imports the package entry point and export table, but does not
  eagerly import solver modules.
- Accessing optional symbols rewrites missing optional dependencies into
  explicit user-facing errors:
  `PnPADMM` / `REDDeconv` report missing `bm3d`, and Wiener automatic noise
  estimation reports missing `scikit-image`.
- Accessing solver symbols such as `WienerDeconv` or `RLUnknownBoundary` still
  requires the shared preprocessing utilities at runtime from
  `RemondoPythonCore.Common`. A plain `import Reconstruction` does not
  guarantee that those solver modules are importable in a standalone
  environment.
- Nested or transitive import failures inside an available shared-preprocessing
  namespace are preserved as their original `ImportError`; they are not
  relabeled as if the namespace were simply missing.
- Most tests do not exercise the real runtime namespace layout directly:
  `tests/conftest.py` installs mock `RemondoPythonCore.Common` modules,
  so the default test environment and a bare standalone runtime are
  intentionally not identical.

The commit `0ead10c` renamed the submodule directory
`reconstruction/` → `external_reconstruction/`, retiring the
lowercase-facade dual-import contract that earlier tests encoded.
Those four tests were rewritten in a subsequent dedicated pass
(see `tests/test_import_smoke.py`) to assert real properties of the
current package layout.  The `external_reconstruction/__init__.py`
facade continues to serve the dotted path
`RemondoPythonCore.reconstruction` for parent-package consumers.
