# Phase B Results — Chip Constraint Solver

Branch: `feat/effbb-precision`. Measured 2026-06-12 over
`data/effbb_cache/cache.jsonl` (hero-active emitted population).

## What shipped

- `scripts/ocr/chip_solver.py` — pure `check_chips()` returning a `ChipCheck`
  (consistent / residuals / single-field repair). Fully unit-tested (3 tests).
- `scripts/ocr/n8_parser.py` — three calibration features surfaced per hand in
  `_LAST_EFFBB_FEATURES`: `chip_consistent`, `chip_repair_found`,
  `chip_residual`. **Feature-only — zero behavior change.** The check runs only
  on preflop-RESOLVED hands (the conservation equation
  `Σcontributions + antes == preflop_pot` is invalid once the engine
  accumulates postflop contributions; postflop hands leave the verdict `None`).

## B3 marginal-precision audit (`scripts/_tmp_chipgate.py`)

Hero-active emitted hands (n=1270, overall precision 78.2%) bucketed by
chip-solver verdict:

| Bucket | n | precision |
|---|---|---|
| consistent | 0 | — |
| inconsistent + repair found | 0 | — |
| inconsistent + no repair | **4** | 75.0% (3/4) |
| not applicable (postflop / no usable header) | 1266 | 78.2% |

**Locked decision (B3/D3a):** the abstain clause ships only if
`inconsistent & no-repair` measures < 60% precision on **≥ 20** hands; the
confidence nudge ships only if `consistent` beats overall precision. Neither
condition is met (4 < 20; 0 consistent). **No gate clause and no nudge ship.**
The scalar effbb frontier is therefore unchanged at **78.19% @ 70.4%**.

## Why the conservation gate does not fire on this corpus

The panel pot headers exposed by `_engine_streets` (`_ep["preflop"]`) are not
reliable totals — they are frequently a small/partial value (e.g. `1.0` against
~52bb of reconstructed contributions). This is the same data limitation that
gives the pre-existing `pot_residual` feature its "poor discrimination at 46.7%
precision" (Phase-6 finding). A working chip-conservation gate needs a reliable
*final*-pot read; the cleaner per-seat reads from Phase C (avatar-anchored OCR)
are the natural prerequisite, at which point this module — already built and
tested — can be re-audited and enabled.

## Net

Phase B lands reusable, tested infrastructure (the `chip_solver` module + its
features) with **no behavior change and no regression**. This is the
plan-anticipated outcome of a gate that the corpus does not (yet) license —
exactly the discipline that killed 3 of 6 levers in Phase 6.
