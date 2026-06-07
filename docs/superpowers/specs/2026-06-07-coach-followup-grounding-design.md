# Coach Follow-up Grounding (P0) — Design Spec

**Date:** 2026-06-07
**Branch:** `feat/coach-followup-grounding` (design) → `feat/coach-followup-grounding-impl` (implementation)
**Status:** P0 + P1 implemented (`scripts/coach_facts.py`)

> **Implemented (2026-06-07):** `scripts/coach_facts.py` ships the deterministic
> classify → fetch → narrate → verify pipeline. Live registry types:
> **P0** `why_action` (B), `fold_equity` (C), `villain_range` (D), `hand_strength` (E);
> **P1** `range_shift` (F), `hypothetical` (G, with off-tree size rejection), `sizing` (H),
> `node_url` (I), plus the numeric-claim audit in the hard verifier. `other`/`range_lookup`
> intents fall back to the existing tool-calling path (no regression). Implementation plan:
> `docs/superpowers/plans/2026-06-07-coach-followup-grounding.md`.
> One design refinement: `allowed_claims` is stored on `Facts` (built inside each `fetch`)
> rather than as a separate `QuestionType.allowed_claims` callable — same guarantee, simpler.

## Problem

The Telegram poker coach hallucinates on "why / strategic" follow-up questions. It
has hero's solver frequencies in context but **no data layer for villain's range,
fold-equity, or equity**, so it fabricates fluent-but-wrong specifics about
villain's holdings.

Evidence (real logs, week of 2026-06-01):
- Q "為什麼 KTo 在 9h8s2s 下注" → only tool called was `evaluate_hand("KTo")`
  (returns hand type), then answer asserted *"BB 範圍裡有大量 AJo/AQo/ATo 會棄牌"* —
  villain range was never queried; invented.
- Q "為什麼 A3s check 但 Q9s bet" → asserted *"對手 KJs/KTs/QJs/JTs 有 6 outs"* — invented.
- Q "為什麼 JJ call 66 all-in" → asserted *"對手大量 QJs/QTs/JTs/AQs/AJs 詐唬"* — invented.

Root cause: the grounding gate forces *a* tool call, but `evaluate_hand` satisfies it
without supplying any range facts. Even Gemini 2.5 Pro hallucinates here — the problem
is missing data + freedom to reason about ranges, not model weakness.

## Key enabling fact

The GTO Wizard **postflop spot-solution** response already contains everything needed,
per node, per player (`players_info[2]`):
- `range[1326]` (combo weights), `hand_eqs[1326]` (equity), `hand_evs[1326]`,
  `hand_eqrs[1326]` (equity realization), `eq_percentile[1326]`
- `hand_categories[17]`, `draw_categories[8]`, `equity_buckets[4]/[7]` — **each already
  broken down per action** (`actions_total_frequencies`)
- `simple_hand_counters` (per 169-class: eq/ev/eqr/freq + per-action combos)
- `blocker_rate[1326]`, `blockers_frequencies[48]`
- `action_solutions[9]` with per-action `strategy`/`evs`/categories

So this is a **data-surfacing** problem, not a compute problem. We let GTO Wizard do
the math (most accurate) and surface the right slice to the model. No range/equity
engine to build.

Note: the **acting** player's `actions_total_frequencies` are populated; a non-acting
player shows only range composition + equity. So "what does my bet fold out" requires
querying the **villain's response node** (after hero's bet), where villain is the actor.

## Scope

**P0 (this spec):** question types B, C, D, E.

| Type | Question | Fact source |
|------|----------|-------------|
| B | Why does this hand take this action? | hero node: combo eq/eqr/percentile/category/draw/blocker + per-action EV/freq |
| C | Fold equity — what does my bet fold out / call? | villain **response** node: `hand_categories`+`equity_buckets` w/ fold/call/raise freqs |
| D | What's in villain's action (bet/raise/shove) range? | villain action node: sub-range by category + hero equity vs that range |
| E | Hand-strength magnitude | subset of B: eq_percentile + bucket + ahead/behind vs villain range |

**P1 (deferred, registry-ready):** F range-shift (scary cards), G hypotheticals
(+ off-tree size rejection), H sizing, I pasted-URL node explain, full numeric-claim audit.

**Out of scope:** type A (range lookups — already grounded via `query_gto`),
type J (ICM — reuses existing path).

## Architecture

New module `scripts/coach_facts.py` between the follow-up handler and the narrator.
**Deterministic routing** (not model-driven tool loop) for P0 intents:

```
follow-up question (+ cached hand context)
  → classify_intent()          # small gemini-2.5-flash call, thinking=0 → B/C/D/E/A/other
  → registry[intent].fetch(ctx) # pulls the right spot-solution node(s) → compact fact card
  → narrate(facts)             # gemini-2.5-flash thinking=0, prose from ONLY the card
  → verify_claims(prose, facts)# hard combo check; regen once → deterministic template
  → send
```

- No multi-round tool loop for P0 intents → removes the wrong-tool failure and most of
  the 55s latency.
- Unknown/`other` intent → falls back to today's existing tool-calling path (no regression).
- Intent classification is a tiny Flash call (not keyword matching) → robust to Chinese
  phrasing variety.

### Extensible registry

```python
@dataclass
class QuestionType:
    id: str                                       # "fold_equity"
    matches: list[str]                            # intent labels the classifier may emit
    fetch: Callable[[Ctx], Facts]                 # builds the fact card from node(s)
    allowed_claims: Callable[[Facts], set[str]]   # combos/classes the narrator may name

REGISTRY = [B_why, C_fold_equity, D_villain_range, E_hand_strength]
```

Adding a P1 type = append one `QuestionType` (matcher + fetch + allowed_claims). Router,
narrator, and verifier are type-agnostic.

## Digest design (categories + grounded examples)

Each `fetch` returns a compact fact card (~10-20 lines) plus `allowed_claims`.

Grain: **hand categories + equity buckets** (with fold/call/raise freqs) PLUS, per shown
category, the **top 1-2 representative 169-classes** from `simple_hand_counters` with
their actual freqs. Example C card slice:

```
villain BTN response to hero bet 33% (board 9h8s2s):
  ace_high    29% of range — fold 80% | call 20%   e.g. AJo fold 84%, AQo call 31%
  top_pair    21%          — fold 4%  | call 88% | raise 8%   e.g. T9s, 98s
  second_pair  5%          — fold 35% | call 65%
  equity buckets: best 25% (mostly continue), trash 16% (mostly fold)
hero hand KTo: eq vs villain range 37%, eq_percentile 0.41, K-high / "trash" bucket
```

The example classes make answers natural ("A高如 AJo 棄牌") AND keep them grounded
(AJo is in facts with a real number).

## Anti-hallucination: hard verifier

`Facts.allowed_claims` = combos/classes literally present in the card + hero's hand.

Verifier scans the narrator draft:
1. **Extract** poker-combo tokens via regex: `AK`, `AKs`, `AJo`, `66`, `KTo`,
   suited-combos `AhKh`. (Latin tokens; works inside Chinese prose.)
2. **Whitelist:** hero's hand (all forms), board cards, any combo/class in the card.
3. **Verdict:** any extracted combo not whitelisted → violation.
4. **On violation:** regenerate once with the allowed vocabulary spelled out; if it still
   trips → **deterministic template** built from the card (no model). User never sees an
   ungrounded combo claim.

Not policed (grounded backbone, passes freely): category-level claims (頂對/A高 + freq),
generic concepts (詐唬/價值/equity/阻斷牌). Numbers: light tolerance check (gross-mismatch
regen); full numeric audit is P1.

Metrics logged: `verifier_block_rate`, `fallback_rate`, intent distribution.

## Narrator model

`gemini-2.5-flash`, `thinking_budget=0`, temperature 0-0.2, small max output. Safe here
because it only narrates a grounded card and the hard verifier catches drift. This also
gives the latency win.

## Latency target

intent-classify (~1s) + 1-2 cached solver fetches + 1 Flash narration (~3-5s) ≈ **5-8s**,
vs ~55s avg today. Cache hits make solver fetches ~free.

## Testing

- **Golden cases** seeded from the 3 real failures (KTo-bet, A3-vs-Q9, JJ-vs-66-shove):
  assert (a) correct digest fetched, (b) no combo named outside facts, (c) verifier passes.
- **Deterministic fetch tests:** each registry type's `fetch` against a cached
  spot-solution fixture (no network) → stable digest.
- **Verifier unit tests:** combo extraction, whitelist (hero hand / board / facts),
  violation → regen → template path.
- Run `python scripts/regression_test.py` (all existing must stay green).

## Files

- `scripts/coach_facts.py` (new) — registry, digests, verifier, intent classifier.
- `src/gemini_session.py` — route P0 follow-up intents through `coach_facts`; keep the
  existing tool path as the `other` fallback.
- `scripts/regression_test.py` — golden + fetch + verifier tests.

## Risks / open items

- Intent misclassification → mitigated by `other` fallback to the existing path.
- Node resolution for C/D (finding villain's response/action node) must reuse the hand
  tree the analyzer already walks; needs care for multiway and overridden sizes.
- Off-tree / low-frequency nodes: digest should flag fragility (full handling is P1).
