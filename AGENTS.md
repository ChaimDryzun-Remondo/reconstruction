# AGENTS.md

## Purpose

This repository contains a reconstruction/deconvolution submodule with scientifically meaningful behavior.  
Changes must preserve documented contracts unless a task explicitly asks to change them.

This file defines how agents should work in this repo:
- how to plan work
- how to classify changes
- how to run tests
- what is considered safe vs risky
- what contracts must not be changed casually
- how to report results

The default expectation is:
- make small, reviewable changes
- preserve behavior unless the task explicitly asks for a semantic change
- use test-first workflows whenever behavior, contracts, or numerics are involved
- prefer explicit failure to silent numerical corruption

---

## Repository focus

The highest-sensitivity code is the external_reconstruction submodule.

Primary target:
- `external_reconstruction/`

Important implementation package:
- `external_reconstruction/Reconstruction/`

Tests:
- `external_reconstruction/tests/`

Docs/spec:
- `external_reconstruction/docs/`
- `external_reconstruction/docs/README.md`
- `external_reconstruction/docs/RECONSTRUCTION_SPEC.py`

---

## Working principles

### 1. Contracts come first
This project now has explicit contracts in tests and docs.  
Do not casually change them.

Important contracts already established:
- numerical-failure contract
- working-domain / normalization contract
- PSF preprocessing contract
- boundary-handling contract
- statefulness / warm-start contract

If a task may change one of these, do **not** implement directly.  
First:
1. identify the contract being touched
2. explain the impact
3. add or update tests
4. then implement

### 2. Behavior-preserving refactors must remain behavior-preserving
If the task is a refactor:
- do not change solver math
- do not change public APIs
- do not change statefulness semantics
- do not change preprocessing semantics
- do not change error-routing semantics
- do not change wrapper cold-start behavior
- do not change class warm-start behavior

If exact preservation seems impossible, stop and report why.

### 3. Small steps only
Prefer:
- one narrow refactor
- one bug fix cluster
- one contract clarification
- one family of tests

Avoid:
- broad rewrites
- multi-theme edits
- simultaneous API + math + docs changes
- "cleanup" that touches unrelated files

### 4. Tests are mandatory for meaningful changes
For anything beyond trivial comments/doc wording:
- run targeted tests first
- then run full suite if targeted tests pass

### 5. Scientific code should fail clearly
The package should not silently return non-finite results.
Prefer:
- `FloatingPointError`
- preserved finite solver state
- explicit documentation of divergence / early stop behavior

### 6. Docs must track real behavior
If code changes any documented contract:
- update tests first or in the same change
- update docs/spec in the same change or immediately afterward

---

## Current near-term contract (must preserve unless explicitly changing it)

### Working domain
Default working domain is:
- grayscale

`initialEstimate`, if provided, must be transformed into the same image-derived working domain.

Outputs are returned as grayscale working-domain arrays unless a future contract explicitly changes this.

`WienerDeconv.normalize_image` currently exists for compatibility only and should not be given new semantics casually.

### PSF handling
Current default PSF policy is destructive preprocessing:
- center-of-mass centering
- negative clipping
- odd-shape enforcement
- conditioning
- zero-padding
- `ifftshift`

Wiener uses the same general PSF pipeline but a distinct conditioning preset from the iterative-family solvers.

Do not unify Wiener and iterative PSF conditioning unless the task explicitly asks for a scientific contract change.

### Boundary handling
Current boundary model is hybrid by solver family:
- `WienerDeconv`: padded / tapered circular deconvolution, no masked unknown-boundary fidelity
- RL / Landweber / FISTA: masked fidelity on original support over padded FFT canvas; Neumann-family TV/prox behavior when TV is active
- Chambolle-Pock / ADMM / TVAL3: masked fidelity on original support over padded FFT canvas; periodic gradient/divergence operators
- PnP / RED: inherit ADMM masked-fidelity padded-FFT structure, no explicit TV boundary contract

Do not flatten this into a fake single model in code or docs.

### Statefulness
- Iterative solver class instances are stateful warm-start solvers.
- Persistent iterate is `estimated_image`.
- Repeated object-level `deblur()` calls warm-start from that state by default.
- Wrapper functions are stateless cold-start helpers.
- `WienerDeconv` is stateful for setup reuse and diagnostics, but not iterative-warm-started.
- On numerical failure, solver state remains finite and reusable, but may preserve the last verified finite iterate reached during the failed call rather than exact pre-call state.

Do not change these semantics without explicit task approval and tests.

### Numerical-failure policy
For supported regimes:
- return finite output

For non-finite internal solver state:
- raise `FloatingPointError`

For finite-state divergence / cost explosion in ADMM-family behavior:
- safe early stop may return last verified finite iterate, according to current tested contract

Failed runs must not poison persistent solver state with `NaN` or `Inf`.

---

## Change classification

Always classify your task before editing.

### A. Behavior-preserving refactor
Examples:
- helper extraction
- deduplicating wrapper kwarg routing
- extracting PSF preset constants
- factoring internal repeated logic

Requirements:
- no semantic changes
- narrow diff
- targeted tests + full suite

### B. Bug fix
Examples:
- wrong routing
- broken backend switching
- broken range handling
- invalid numerical failure handling
- stale import behavior

Requirements:
- reproduce or identify exact failure
- write or update regression tests
- patch minimally
- run targeted tests + full suite

### C. Contract clarification
Examples:
- docs/spec alignment
- explicit tests for existing semantics
- rename misleading wording in docs

Requirements:
- no code changes unless needed
- keep docs aligned with current actual behavior
- do not “improve” behavior accidentally

### D. Scientific / numerical contract change
Examples:
- changing normalization policy
- changing PSF contract
- changing boundary model
- changing warm-start semantics
- changing divergence behavior

Requirements:
- do not do this casually
- first produce impact analysis
- add/modify tests to define new intended behavior
- only then implement

### E. Performance / optimization
Examples:
- reducing repeated FFT setup
- dtype optimizations
- memory reductions
- faster wrappers

Requirements:
- preserve contracts
- benchmark or justify
- verify no semantic drift
- full suite required

---

## Required workflow

For any nontrivial task, follow this sequence.

### Step 1: Read before editing
Read only the files needed for the task.
Also read relevant tests and doc/spec sections.

At minimum, consult:
- implementation file(s)
- corresponding test file(s)
- relevant part of `docs/README.md`
- relevant part of `docs/RECONSTRUCTION_SPEC.py` if contract-sensitive

### Step 2: State the task class
Before editing, determine whether the task is:
- refactor
- bug fix
- contract clarification
- scientific contract change
- performance optimization

### Step 3: Edit narrowly
Prefer:
- smallest viable patch
- one helper at a time
- no opportunistic cleanups

### Step 4: Run targeted tests
Always run the smallest relevant subset first.

### Step 5: If targeted tests pass, run full suite
Use:
```bash
python -m pytest -q external_reconstruction/tests
```

### Step 6: Report clearly
Your report must include:
- what changed
- files changed
- whether behavior is intended to remain identical
- what tests were run
- whether full suite passed
- any remaining ambiguity or deferred work


## conda enviorment
```bash
conda activate reconstruction
```

For reconstruction development in this repo, use:

cd external_reconstruction
pip install -e .
python -m pytest -q tests

Do not run `pip install -e .` from the monorepo root unless the task is specifically about monorepo packaging.
