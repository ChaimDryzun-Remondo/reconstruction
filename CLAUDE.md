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

Baseline (core-profile cold checkout on `audit-2025-q2`, Post-Sprint-5):
**548 passed, 1 xfailed, 3 skipped, 27 deselected of 577 total tests, in ~11 s**.

The default invocation filters to the `core` profile via `pyproject.toml:38`'s
`-m 'not imaging and not pnp and not monorepo'` addopts.  The 27 deselected
tests are split across three optional profiles: `imaging` (20 tests — needs
`scikit-image` + `matplotlib`), `pnp` (4 tests — needs `bm3d`), and `monorepo`
(3 tests — require the full `RemondoPythonCore.Common` environment, added in
Sprint 4 thread 3).  See `pyproject.toml:38–44` for the marker declarations
and `docs/README.md:347–362` for the alternate invocations.

The 1 xfailed is the T3.1 NaN-metric robustness test in
`tests/test_optimize_alpha.py`.  The three Sprint 0 W6 `Shared.Common`
fallback xfail tests were deleted in Sprint 5 item 2 alongside the fallback
mechanism itself; the redundant `test_common_nested_import_failure_in_preferred_namespace_is_not_mislabeled`
test was deleted in Sprint 5 item 3 alongside the `Shared.Common.*` mock
removal in `tests/conftest.py`.

Monorepo-profile baseline (run with `-m monorepo` from the parent monorepo
environment): **3 passed, 2 skipped, 575 deselected, in ~3 s**.  The three
passing tests are `test_optimize_alpha::test_optimize_alpha_default_metric_runs_to_completion`
(Sprint 4 T3.1), `test_example_flow_smoke::test_example_flow_module_loads_cleanly`
(Sprint 4 T3.2), and `test_example_smoke::test_example_module_loads_cleanly`
(Sprint 4 T3.3).

History:
- An earlier statement in this file claimed "656 passed, 0 failed"; that
  figure could not be reproduced against any invocation of the current test
  tree and was superseded by the Pre-Sprint-0 core-profile baseline during
  Sprint 0 commit W1 (parent `ff2d5bb`, submodule `776d178`).  Full
  investigation at `../docs/refactoring-audit/NOTES.md` §"Sprint 0
  investigation outcomes" 2026-04-23 W1 entry.
- Pre-Sprint-0 baseline (2026-04-23): 532 passed, 6 failed, 3 skipped,
  24 deselected of 563 total tests, in 11.26 s.  The 6 failures were
  3 `_bm3d_func` drift + 3 `Shared.Common` fallback-contract failures.
- Sprint 0 commit W5 (submodule `804fdea`) cleared the 3 `_bm3d_func`
  failures into passes by making `monkeypatch.setattr` non-raising.
- Sprint 0 commit W6 (submodule `ab354c7`) converted the 3 `Shared.Common`
  failures to xfailed pending Sprint 5 Q2 removal.
- Sprint 0 commit W7a (submodule `e43f9c0`, parent commit recorded in the
  parent-repo `CLAUDE.md`) updated this baseline line to the
  Post-Sprint-0 state and aligned its wording to the canonical format used
  in the repo-root `CLAUDE.md` §"Running the test suites" section and the
  `NOTES.md` §"Consolidated baseline for gate comparison" table.
- Sprint 1 (three submodule commits): `a021d17` deleted the unused
  `external_reconstruction/OptimizeTemp.py` scratch script (audit M-finding
  closed; pre-destruction grep confirmed zero importers); `f7e4d46` added
  a four-line note to `Reconstruction/__init__.py`'s module docstring
  recording §8 Q4 (the PascalCase `Reconstruction/` name is deliberate
  and not to be renamed); `05fb923` relocated `example.py` and
  `example_flow.py` from the package root to a new `examples/`
  subdirectory (Sprint 1 commits C7 + C9 bundled), with `example_flow.py`
  receiving an `if __name__ == "__main__":` wrap to make its imperative
  code safe to import.  The Sprint 1 work did not change the test gate;
  Post-Sprint-1 = Post-Sprint-0 (535 passed, 3 xfailed, 3 skipped,
  24 deselected of 563 total tests).
- Sprint 4 thread 3 (three submodule commits) advanced the gate to its
  Post-Sprint-4 state by adding 18 tests (16 from T3.1, 1 each from T3.2
  and T3.3) and 3 monorepo-marked deselects (1 from T3.1, 1 each from
  T3.2 and T3.3):
  - `2106e7e` (T3.1) promoted `optimize_wiener_alpha` from
    `examples/example_flow.py` into a public `optimize_alpha` function +
    `OptimizeAlphaResult` frozen dataclass at `Reconstruction.wiener`,
    co-locating four private helpers (`_resolve_metrics`, `_coarse_search`,
    `_brent_refine`, `_normalize_image_for_metric`) in the same module
    and exporting the two public symbols via `Reconstruction/__init__.py`'s
    lazy `_EXPORTS`.  Behavioral comparison harness (run before commit;
    not committed) confirmed bit-exact agreement on the search trajectory
    and within-FFT-noise-floor (~1e-6) on final-evaluation values.  Added
    16 tests at `tests/test_optimize_alpha.py` (5 convergence,
    5 robustness including 1 NaN-metric flexible xfail, 3 contract,
    2 search-parameter dispatch, 1 metric-substitution sanity); 14 in
    the core profile, 1 xfailed, 1 monorepo-deselected.  Updated
    `examples/example_flow.py` to import the promoted API.
  - `717ffc2` (T3.2) rewrote `examples/example_flow.py`'s `__main__`
    block to use synthetic input (replacing the hardcoded user paths
    flagged as the C9 finding), consolidated the eleven copy-paste
    algorithm blocks into a declarative `_AlgoSpec` dataclass + executor
    function, fixed the deprecated `RemondoPythonCore.reconstruction`
    import path, gated PnP-ADMM and RED-ADMM on `bm3d` availability,
    dropped per-algorithm TIFF saves and per-algorithm `plt.show()`
    calls in favour of a single end-of-demo tiled figure plus an
    α-trajectory subplot.  Added one Level-1 module-load smoke test at
    `tests/test_example_flow_smoke.py`, marked `@pytest.mark.monorepo`.
  - `b6d976e` (T3.3) rewrote `examples/example.py` per three divergence
    dispositions: kept local `blur_image` (honouring T1.2's deferred
    M1 consolidation), kept local `add_awgn` (pedagogical clarity),
    replaced local `psnr`/`ssim` with `Common.PSNR`/`Common.SSIM`
    (six call sites; argument order swapped from `(reference, estimate)`
    to `(image, ref_image)`).  Extracted the smoke-test sys.modules /
    sys.path bootstrap into a shared helper at
    `tests/_remondopythoncore_bootstrap.py` and refactored T3.2's
    smoke test to use it; added one new Level-1 module-load smoke
    test at `tests/test_example_smoke.py`.
- Submodule batch closeout commit `e812d0c` updated this baseline line
  to the Post-Sprint-4 state and appended the Sprint 1 and Sprint 4
  entries above.  No code changes; documentation alignment only.
- Sprint 5 (two submodule commits) advanced the gate to its
  Post-Sprint-5 state by closing the F3 cleanup thread.  See parent
  `docs/refactoring-audit/NOTES.md` Sprint 5 closeout entry for the
  full sprint summary.  Submodule commits in order:
  - `1c82b66` (Sprint 5 item 2) collapsed
    `Reconstruction/_common.py` from a try/except fallback selector
    to a direct import block from
    `RemondoPythonCore.Common.{General_Utilities,PSF_Preprocessing,Image_Preprocessing}`,
    and deleted the three xfail tests in
    `tests/test_import_smoke.py` that Sprint 0 W6 had marked as
    signposts for the Sprint 5 Q2 removal
    (`test_common_falls_back_to_shared_namespace_when_remondo_absent`,
    `test_common_missing_both_namespaces_raises_clear_error`,
    `test_root_solver_symbol_requires_shared_preprocessing_namespace`).
    Core profile gate moved
    549 passed + 4 xfailed + 3 skipped + 27 deselected of 581 total
    → 549 passed + 1 xfailed + 3 skipped + 27 deselected of 578
    total.
  - `a2fa88d` (Sprint 5 item 3) severed the submodule's remaining
    dependence on the parallel `Shared` codebase.  Seven pieces:
    `tests/conftest.py` deleted the `Shared.Common.*` mock stub
    block (lines 162-187 of the pre-commit file);
    `docs/reference/RL_Unknown_Boundary.py` and
    `docs/reference/Landweber_Unknown_Boundary.py` rewrote three
    `from Shared.Common.* import …` lines each to
    `from RemondoPythonCore.Common.* import …`;
    `docs/reference/TVAL3.py` (dead reference, no test loads it)
    replaced its `Shared.algo.Utilities` and `Shared` imports with
    local `NotImplementedError`-raising stubs;
    `Reconstruction/__init__.py` simplified the
    `_translate_import_error` message to drop the
    `'(legacy) or Shared.Common'` qualifier;
    `tests/_remondopythoncore_bootstrap.py` rewrote its docstring
    to reframe the cache-pollution motivation accurately
    (the `RemondoPythonCore.*` mocks remain; the helper logic
    itself stays);
    `tests/test_import_smoke.py` renamed
    `test_common_prefers_remondo_namespace_when_available` →
    `test_common_resolves_to_remondo_namespace` (dropping the
    redundant `Shared` sentinel installation) and deleted
    `test_common_nested_import_failure_in_preferred_namespace_is_not_mislabeled`
    outright (Python's own ImportError propagation made the
    assertion redundant; the third `TestCommonImportContract` test
    covers the same scenario at the user entry point).  Core
    profile gate moved
    549 passed + 1 xfailed + 3 skipped + 27 deselected of 578 total
    → 548 passed + 1 xfailed + 3 skipped + 27 deselected of 577 total.
    The most significant finding from this commit (recorded in
    parent `WORKPLAN.md` §8 Q2's Revised 2026-04-29 footnote and
    parent `NOTES.md` Sprint 5 closeout entry) is that §8 Q2's
    foundational premise — that `Shared.Common` was a symbolic
    alias for `RemondoPythonCore.Common` — was incorrect.  `Shared`
    is an independent sibling git repository at `c:/git/Shared/`
    with parallel structure and drifted implementations; the
    parent's `Tools/MC_image_analysis.py:57` migration was
    therefore deferred to a dedicated cleanup pass.
- Sprint 5 closeout (this commit) updates this baseline line to the
  Post-Sprint-5 state and appends the Sprint 5 entries above.  No
  code changes; documentation alignment only.  Future readers should
  consult parent `docs/refactoring-audit/NOTES.md` Sprint 5 closeout
  entry for the full sprint summary, including the
  verification-refines-workplan pattern that emerged across items
  1, 3, and 6.
