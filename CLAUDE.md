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

## Known pre-existing failures (not caused by solver refactors)

Four tests in `tests/test_import_smoke.py` fail on every commit in
recent history and are unrelated to any solver change:

- `TestImportSmoke::test_import_lowercase_facade_smoke`
- `TestImportSmoke::test_import_uppercase_backend_smoke`
- `TestImportSmoke::test_import_uppercase_base_aliases_lowercase_module`
- `TestImportSmoke::test_uppercase_and_lowercase_exports_share_implementation_objects`

**Root cause.**  All four exercise a lowercase-facade dual-import
contract — they require both `import reconstruction` *and*
`import Reconstruction` to succeed and resolve to the same underlying
module objects.  The facade was designed around a submodule physically
named `reconstruction/`.  Commit `0ead10c` (*"Rename submodule
reconstruction → external_reconstruction to avoid case collision with
Reconstruction/ package on Windows"*) moved the physical directory out
from under the lowercase name, leaving the facade stranded: after the
rename, `import reconstruction` has no target on `sys.path` because no
package by that name is installed or importable.  The facade file at
`external_reconstruction/__init__.py` was intended to provide
`RemondoPythonCore.reconstruction` (a dotted path under a parent
package), not the bare `reconstruction` top-level name the tests use.

**What was also part of this.**  The parent repo used to hold a
stale `Reconstruction/` sibling with an `_alias.py` forwarder as a
remnant of the old layout; pytest's default `prepend` import mode
walked up to the parent rootpath and picked up that sibling instead
of the submodule.  That sibling was deleted in the same refactor pass
(see the accompanying parent-repo commit).  Its removal **did not
affect these four failures** — they failed before the deletion too.

**Do not** try to fix these by editing `tests/`.  The hard rule above
forbids it, and the right fix is architectural.

**Options for a dedicated fix pass** (pick one — ask me before
implementing):

- **A.**  `pip install -e ./external_reconstruction` from the parent
  with the submodule's distribution name set to `reconstruction` (it
  already is — see `pyproject.toml`).  Additional setuptools config
  needed to make the dotted path
  `reconstruction.Reconstruction._base` resolve as the tests use it.
- **B.**  Create a physical `reconstruction/` shim package at the
  parent repo root that forwards to
  `external_reconstruction.Reconstruction`.  Conflicts with Windows
  filesystem case-insensitivity if any `Reconstruction/` ever reappears
  at the same level.
- **C.**  Inject `sys.modules` aliases in `tests/conftest.py`.  Works
  for the tests specifically; arguably a test-harness fix rather than
  a real package fix.
- **D.**  Update the four tests to match the post-rename reality
  (only `Reconstruction` is a valid top-level name; the lowercase
  facade is retired).  Requires explicit approval because it changes
  an architectural contract the tests encode.

## Active session scope: facade test architectural fix

This session is authorized to modify `tests/` — specifically the 4 pre-existing
facade failures in `test_import_smoke.py` and any adjacent test infrastructure
needed to implement the chosen fix. All other hard rules remain in force.

Additional scope for this session:
- May add new tests, remove obsolete tests, or restructure test files when
  required by the chosen architectural option.
- May modify `Reconstruction/__init__.py`, package-level re-exports, and
  facade-layer code if the chosen option requires it.
- May NOT modify algorithm implementations (Wiener, RL, ADMM, CP, FISTA,
  TVAL3, RED, PnP, Landweber) — those are out of scope for this pass.
- May NOT modify any test that currently passes without first showing me
  the diff, the rationale, and waiting for explicit approval.

Process for this session:
- Phase 1: Re-read the facade failure documentation in docs/README.md and
  CLAUDE.md (options A–D). Summarize each option with its tradeoffs.
  Recommend one, but do NOT proceed.
- Phase 2: After I pick an option, produce an atomic step-by-step plan.
  Each step must be independently revertible.
- Phase 3: Implement one step at a time. Show diff, wait for approval,
  apply, run full test suite, report results verbatim.
- Target baseline: 656 passed (current 652 + 4 recovered), 0 failed.
  Any regression from 652 is a stop condition.
- Commit per step with conventional-commit message (fix: / refactor: / test:).

This session scope is removed at session end. Revert to the base working
agreement for any subsequent session.