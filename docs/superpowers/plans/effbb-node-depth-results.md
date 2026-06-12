# Phase A Results — Per-Node Depth Analysis

Branch: `feat/effbb-precision` (single-branch execution of the full roadmap).
Measured 2026-06-12 against `data/effbb_cache/cache.jsonl` (7183 rows) and the
rebuilt `data/pokercraft_corpus/ground_truth/ground_truth.jsonl` (now carrying
`node_effective_bb`).

## Per-node depth accuracy (`python scripts/effbb_eval.py --per-node`)

Hero-active hands with GT `node_effective_bb` (1837 hands scored):

| Node type | Single hand-wide depth (legacy) | Per-node resolver (D1) | Δ |
|---|---|---|---|
| **OPEN**   | 63.7% (1171/1837) | **98.6% (1812/1837)** | **+34.9pp** |
| **FACING** | 70.7% (188/266)   | **98.9% (263/266)**   | **+28.2pp** |

The "single hand-wide depth" column is the legacy behavior: one depth
(`nearest_depth(effective_bb)`, which the old all-in override forced to the jam
stack) applied to every hero decision node. The per-node resolver assigns each
node its own D1 depth — the open node plays the deepest live cover, the facing
node plays the aggressor's commitment. Residual ~1.4% is rounding (GT uses exact
HH chips; the resolver consumes 1-decimal `stacks_bb`).

This is the user's exact complaint, measured: *"CO 30bb open vs SB 17bb jam"* —
the open node now queries the 30bb tree (not the 17bb jam tree), the facing node
the 17bb tree, with a range-mismatch caveat surfaced to the user.

## Scalar effbb gate (D5) — unchanged

`python scripts/effbb_eval.py` hero-active: **78.19% @ 70.4% (993/1270)** —
identical to the 78.19% @ 70.4% baseline. Phase A changes *what depth we ask the
solver per node*; it does not touch `_compute_effective_bb`, so the scalar
metric is provably unaffected (D5 gate ≥ 77.9% / coverage ≥ 69.4% holds).

## What shipped

- `scripts/node_depth.py` — pure D1 resolver (`resolve_preflop_nodes`).
- `scripts/hh_parser.py` — `node_effectives()` (exact per-node GT from HH) +
  `node_effective_bb` emitted in every parsed hand → ground truth.
- `scripts/analyze_hand.py` — `_build_hero_spot_depths()` adapter; the open node
  keeps its deep depth under an all-in override (no longer dragged to jam depth);
  facing spots carry a `depth_caveat`.
- `scripts/gto_formatter.py` rendering path (`analyze_hand` compact) emits the
  caveat line under the facing section header.
- `scripts/effbb_eval.py --per-node` — the metric above.

Caveat text (Chinese-canonical):
`⚠ 此節點以 {node}bb 樹查詢（你前一個決策是在 {prev}bb 樹做的）；solver 假設你帶著 {node}bb 的範圍到達此節點，範圍銜接會有偏差，數據供參考。`
