# external_reconstruction Submodule — Working Agreement

## Project context
This is the `external_reconstruction` submodule of RemondoPythonCore, the Python codebase
for Remondo's electro-optical satellite imaging pipeline. It implements image
deconvolution and reconstruction algorithms (Wiener, Chambolle-Pock, ADMM-TV,
Richardson-Lucy, TVAL3, FISTA, RED-ADMM, PnP-ADMM, Landweber, BM3D-based
variants) operating on sensor-simulated and real satellite imagery.

Upstream context Claude should keep in mind:
- Inputs come from actucal cameras or a simulation pipeline.
- PSFs come from actucal cameras or TMA optical simulation (Optiland-based).
- Downstream consumers use MTF/MSGMSD metrics and NIIRS/GIQE scoring.

## My preferences
- Professional, neutral tone. No filler, no cheerleading.
- Detailed, technical, mathematical explanations. Derive things when relevant.
- Languages: Python (primary), C++ (secondary). Type hints required; prefer
  `mypy --strict` clean code.
- Be innovative — suggest cutting-edge approaches when justified, but always
  cite a paper, doc, or source. **Never fabricate references or APIs.**
  If you don't know something, say so explicitly.
- Ask clarifying questions when intent is ambiguous. Do not guess.

## Environment
- Windows workstation.
- Python via conda. Use the reconstruction environment. Confirm the active env before running anything; do not
  assume. Ask me if unclear.
- Test runner: pytest. Run tests for the touched module after every change.
- Lint/type: ruff + mypy --strict where already configured.

## How I want you to work — phased, gated, cautious

Operate in explicit phases. **Stop and wait for my explicit approval between
phases.** Do not chain phases.

Phase 1 — Structural inventory (read-only)
  - Package/module tree, key entry points, main classes/functions
  - Data flow through the reconstruction pipeline
  - External dependencies (with versions where pinned)
  - Test coverage map and missing-test inventory
  - Initial list of architectural risks

Phase 2 — Structured report covering:
  1. High-level architecture
  2. Module-by-module responsibilities
  3. Likely bugs / fragile areas
  4. Numerical and stability concerns (conditioning, regularization, FFT
     edge handling, dtype/precision, NaN/Inf propagation, convergence)
  5. API design issues
  6. Performance bottlenecks (with evidence — profile or cite the line)
  7. Refactoring opportunities
  8. Prioritized recommendations (impact × effort × risk)

Phase 3 — Deep technical review of reconstruction internals
  - Per-algorithm correctness check against the canonical formulation
  - Boundary conditions, padding, FFT shifts, normalization conventions
  - Step-size / regularization parameter handling
  - GPU/CPU code paths (CuPy/NumPy parity)

Phase 4 — Step-by-step fix and improvement plan
  - Ordered, atomic, each item independently revertible
  - Each item: rationale, files touched, risk, test strategy

Phase 5 — Implementation
  - Minimal, cautious, surgical edits. One logical change per commit.
  - **Show the diff and wait for my approval before writing any file.**
  - After each change: run the relevant tests and report results verbatim.
  - If a change exceeds ~50 lines or touches >2 files, stop and re-confirm scope.
  - Never reformat untouched code. Never bundle unrelated fixes.
  - If a test fails, stop. Do not "fix" by loosening the test.

## Hard rules
- No fabrication. Cite file:line for claims about this code; cite paper/URL
  for claims about algorithms.
- Read before editing. If a file is large, read it in full or state explicitly
  which ranges you read.
- Never modify `tests/` to make failing code pass unless I explicitly ask.
- Never touch git history, submodule pointers, or CI config without asking.
- Preserve public API signatures unless Phase 4 plan explicitly approves a break.
- Numerical code: preserve dtype unless change is justified and called out.

## Communication
- Explain *what* you're doing and *why* before doing it.
- When uncertain, present options with tradeoffs rather than picking silently.
- Summaries at end of each phase: what was found, what's proposed, what you
  need from me to proceed.

## Test environment note

Run the test suite with `--import-mode=importlib`.  Pytest's default
`prepend` mode walks up from `tests/test_*.py` past `tests/__init__.py`
and `external_reconstruction/__init__.py` to the parent repo root,
which shadows the submodule's inner `Reconstruction/` package with any
`Reconstruction/` directory that happens to exist at the parent level:

```bash
conda activate reconstruction
pytest tests/ --import-mode=importlib -v
```

## Resolved: lowercase-facade import contract (commit `0ead10c` follow-up)

The four formerly-failing tests in `tests/test_import_smoke.py` have
been resolved.  The lowercase-facade dual-import contract
(`import reconstruction` as a bare top-level name) was retired in a
dedicated fix pass.  `import Reconstruction` is now the sole supported
top-level entry point.

The four tests were rewritten (not deleted) to assert real properties
of the current package layout.  See `tests/test_import_smoke.py` and
`docs/README.md` (*Import entry point* section) for the record.

Baseline (core-profile cold checkout, 2026-04-23, Pre-Sprint-0):
**532 passed, 6 failed, 3 skipped, 24 deselected of 563 total tests**.

The default invocation filters to the `core` profile via `pyproject.toml:38`'s
`-m 'not imaging and not pnp and not monorepo'` addopts.  The 24 deselected
tests are split across three optional profiles: `imaging` (20 tests — needs
`scikit-image` + `matplotlib`), `pnp` (4 tests — needs `bm3d`), and `monorepo`
(marker defined for tests requiring the full `RemondoPythonCore.Common`
environment; currently 0 tests carry it).  See `pyproject.toml:38–44` for the
marker declarations and `docs/README.md:347–362` for the alternate invocations.

An earlier statement here claimed "656 passed, 0 failed"; that figure could
not be reproduced against any invocation of the current test tree.  See
`docs/refactoring-audit/NOTES.md` §"Sprint 0 investigation outcomes" for the
analysis.  The 6 failures and 3 skipped are the Sprint 0 pre-existing
failures tracked in `NOTES.md` findings F2 and F3.
