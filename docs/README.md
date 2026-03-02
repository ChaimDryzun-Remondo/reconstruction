# Documentation

## Files

- **RECONSTRUCTION_SPEC.py** — Complete architecture specification and phased implementation plan. This is the primary reference document for building the package. Read it before implementing any phase.

- **reference/** — Original standalone implementations that the refactored package must be behaviour-compatible with:
  - `RL_Unknown_Boundary.py` — Richardson-Lucy with unknown boundaries and Dey et al. multiplicative TV (corrected version, 567 lines).
  - `Landweber_Unknown_Boundary.py` — FISTA-accelerated preconditioned Landweber with proximal TV via Chambolle dual projection (757 lines).

## Implementation Workflow

1. Read the spec's section for the current phase.
2. Implement the code.
3. Run the verification step defined at the end of each spec section.
4. Only proceed to the next phase after verification passes.
