# Design: skip per-hand detail fetch for 0-loss hands

**Date:** 2026-07-15
**Branch:** `feat/ingest-zeroloss-skip`
**Status:** design approved (threshold + fidelity), spec under review

## Problem

The GTOW Analyze ingest is slow because it fetches **one detail GET per hand**
(`GET /v4/hand-history/hands/{id}/`), throttled to ~2.5 rps to avoid the
"Too many sessions" FORCED_LOGOUT that this whole pipeline exists to dodge. A
one-month catch-up (526 hands) took ~15 minutes; the one-click extension button
looked frozen.

The list API is already batched (100/page) and returns rich per-hand fields
including `total_ev_loss` and per-street `actions_with_correctness_*`. Across the
34,120 online hands the EV-loss distribution is:

| bucket | hands | % |
|---|---|---|
| exactly 0 | 31,629 | **92.7%** |
| (0, 0.05] | 940 | 2.8% |
| (0.05, 0.1] | 465 | 1.4% |
| > 0.1 | 1,086 | 3.2% |
| null | 0 | — |

**92.7% of hands were played perfectly (total_ev_loss == 0).** Those hands have
no lossy decision worth drilling; the only reason we fetch their detail is to
record the decision nodes. If we can reconstruct those nodes from the list data
we already have, we skip ~93% of the detail GETs — turning a multi-minute sweep
into seconds — without losing anything the North Star metric needs.

## Goal / non-goals

**Goal:** eliminate the detail GET for solved, exactly-0-loss hands while keeping
the ledger's decision records complete and honest — global AND per-spot.

**Non-goals:**
- Do not widen the threshold beyond exactly 0 (any loss ⇒ full detail as today).
- Do not parallelise or speed up the detail GET itself (reintroduces FORCED_LOGOUT).
- No change to how lossy hands are ingested — that path is untouched.

## Invariants preserved (North Star)

- **§2 North Star metric = EV loss / 100 decisions.** The *denominator* (decision
  count) must stay complete. Skipped hands still contribute every hero decision,
  labelled `ev_loss = 0`.
- **§7.3 rankings are EV-weighted.** Reconstructed rows are all `ev_loss = 0`, so
  they never change any avg/total-EV-loss ranking; they only keep the per-spot
  *denominator* honest (avg EV-loss per spot = Σloss / count stays correct).
- **§5.2 source isolation.** Unchanged — these are `source='online'` hands.

## Threshold (decided)

Skip the detail GET **iff** `solution_status` indicates solved **and**
`total_ev_loss == 0` (exact). Everything else — any positive loss, or an
unsolved/no-solution status — takes the existing full-detail path. Rationale:
exactly-0 already captures 92.7%; widening to ≤0.05bb adds only 2.8% while
starting to drop real (if tiny) leaks. Not configurable in v1 (YAGNI); the
threshold lives in one named constant so it is trivial to revisit.

Correctness note: `ev_loss ≥ 0` always, so `total_ev_loss == 0` ⟹ **every** hero
decision in the hand had `ev_loss == 0`. Setting each reconstructed decision's
`ev_loss_bb = 0` is exact, not an approximation.

## Reconstruction (decided: full decisions + taxonomy)

Reuse the existing detail-free classifier `spot_taxonomy.walk_spots_from_parsed`
(already used by the live flow). New adapter converts a list row into the
parsed-hand shape that walker consumes.

### Data available in the list row

`actions_with_correctness_<street>` is an ordered array of **every** action on
that street (hero + villains), each `{action_code, correctness}`. `correctness`
is non-null **only on hero's actions** — this is how we identify hero decisions
without positions. Example (hero = HJ, a 0-loss preflop fold):

```
preflop: [F, F, F, F(BEST_MOVE←hero), F, RAI, F, F]
```

### Adapter: list row → parsed hand

`walk_spots_from_parsed(hand)` needs:
`hero_position`, `players_at_table`, `preflop_actions` (dash-joined token
string), `effective_bb`, `streets[] = {board, actions:[{position, action}]}`.

The list arrays carry ordered **codes** but **not positions**. The adapter must
reconstruct positions:

- **Preflop:** `preflop_actions` = the code sequence joined by `-`; the walker's
  own `_preflop_seat_tokens(tokens, npl)` already assigns seat positions by
  button-relative order (same logic the live flow relies on). Depth =
  `preflop_game_depth` from the list. Hero decisions = tokens at the hero seat.
- **Postflop:** positions are reconstructed by simulating the street among the
  players still active after preflop (fold tracking), OOP-first ordering. Hero's
  actions are cross-checked against the non-null-`correctness` entries as an
  invariant (the reconstructed hero-action indices MUST line up with the graded
  entries; a mismatch ⇒ do not skip, fall back to full detail — see below).
- **Board:** `boards[0]` from the list, split into flop/turn/river.

### Decision rows written for a skipped hand

Per hero decision (from the walker), an `ledger_decisions` row with:

- from walker: `street`, `decision_idx`, `spot_category`, `spot_leaf`,
  `spot_parent`, `spot_keys`, `facing`, `pot_type`, `position`, `depth_band`,
  `played_depth_bb`, `gtow_texture` (from list `board_flop_*`).
- from list array: `taken_code` (= `action_code`), `correctness`.
- fixed/derived: `ev_loss_bb = 0`, `ev_loss_pct_pot = 0`, `best_code = taken_code`,
  `source='online'`, `grader='gtow_list'` (new provenance value, distinct from
  `gtow_analyzer`), `approx_flags = ['list_only']`, `excluded` per existing rules
  (e.g. unsolved never reaches here).
- null (detail-only, never feed rankings/metric): `taken_freq`, `freq_diff`,
  `gto_score`, `hand_eq`, `solver_depth_bb`, `pot_bb`.

### Hand row + state

`ledger_hands` row is upserted from the list exactly as today (so `verify`'s
count matches), with:
- `detail_fetched = false`
- **new column** `detail_status text` — one of `'fetched'`, `'skipped_zeroloss'`,
  `'pending'` (default). Skipped hands get `'skipped_zeroloss'`.

`sweep_detail`'s candidate query changes from `WHERE NOT detail_fetched` to
`WHERE detail_status = 'pending'`, so skipped hands are not re-attempted every run.

## Fallback / safety

The reconstruction must be conservative: if the adapter cannot faithfully rebuild
a hand (position simulation ambiguous, hero-action indices don't line up with the
graded entries, multiway postflop it can't resolve, missing/blank action arrays),
it **does not skip** — the hand is left `detail_status='pending'` and fetched via
the normal detail path. Correctness beats speed; a hand we can't rebuild honestly
just costs one detail GET (it did before). Count and log these fallbacks.

## Reversibility

Skipped hands are identifiable by `detail_status='skipped_zeroloss'`. A new
`ledger_ingest.py --backfill-skipped` mode re-fetches their detail and re-distills
(flipping them to `'fetched'`) — used only if we ever want the secondary solver
stats for perfectly-played hands. Not run by default.

## Acceptance gate (the load-bearing test)

Reconstruction fidelity is the whole risk. Before this ships, an offline harness
must prove the list-only path yields the **same decision structure and taxonomy**
as the detail path on hands we already have both for:

1. Take N (≥200) archived 0-loss hands that currently have BOTH a detail archive
   AND detail-distilled `ledger_decisions` rows.
2. Run the new list-only reconstruction on each.
3. Assert, per hand: same number of hero decisions; identical
   `(street, decision_idx, taken_code, correctness, spot_leaf, spot_category,
   facing, pot_type, position)` tuples as the detail-distilled rows.
4. Report exact-match rate. Gate: **100% on the structural tuple** (spot_leaf,
   facing, pot_type, codes). Any hand that diverges is exactly a hand the runtime
   adapter must send to the fallback path — the harness doubles as the fallback's
   detection logic. Secondary solver-only fields are not compared (absent by design).

This reuses the spirit of the existing `walk_spots ≡ walk_spots_from_parsed`
leaf-equivalence gate.

## Testing

- **Regression:** new unit tests for the adapter (preflop fold, preflop
  3bet-line, HU postflop, multiway→fallback) in `scripts/regression_tests/`.
- **Acceptance harness** above, runnable as a script, checked in.
- Full `python scripts/regression_test.py` must stay green.
- Migration for `detail_status` applied via `supabase db push`; add the column
  handling to `ledger_ingest`/`ledger_distill` and backfill existing rows
  (`detail_fetched=true → detail_status='fetched'`).

## Rollout

1. Migration adds `detail_status` (backfill existing rows).
2. Ship reconstruction + fallback + acceptance harness (harness must pass).
3. Next ingest run: 0-loss hands take the list-only path; lossy unchanged.
4. Observe the daily/button sweep drop to seconds; watch fallback counts in logs.

## Files touched (anticipated)

- `supabase/migrations/<ts>_ledger_detail_status.sql` — new column + backfill.
- `scripts/ledger_ingest.py` — `sweep_detail` gating on `detail_status`; 0-loss
  branch calls the list-only reconstruction; `--backfill-skipped` mode; summary
  counts (`skipped_zeroloss`, `reconstruct_fallback`).
- `scripts/ledger_distill.py` — `distill_hand_from_list(list_row)` adapter +
  parsed-hand builder; shared decision-row assembly with `distill_hand`.
- `scripts/spot_taxonomy.py` — only if the adapter needs a small helper; prefer
  reusing `walk_spots_from_parsed` unchanged.
- `scripts/regression_tests/` — adapter unit tests.
- `scripts/ledger_reconstruct_acceptance.py` (new) — the acceptance harness.
- `src/ingest_runner.py` — surface `skipped_zeroloss` / fallback counts in the
  `_finish` result string so the Telegram summary shows the speedup.
