# Handoff: Poker-Rules Structural Validator for Parsed Hands

**Date:** 2026-06-08
**Status:** Design approved, ready to implement (next session)
**Author:** prior session (with @Harry)
**Goal:** A single, game-logic-based validator that replays every parsed hand as a real
Texas Hold'em betting game and rejects anything the rules forbid — so a broken parse can
**never silently** reach the solver and surface as `（無 solver 數據）` or wrong advice.

---

## 1. Why this exists (motivation)

Recent bugs (H3514, H3511, H3517, H3518, and the orphan-call class below) share one root:
**the parse→analyze pipeline guesses with brittle heuristics, and when it guesses wrong it
produces a plausible-but-impossible hand that ships silently.** `analyze_hand_full` never
raises on a malformed hand — it just emits `（無 solver 數據）`
(`scripts/analyze_hand.py:2603`), so a *parse bug* and a *genuinely off-tree spot* look
identical to the user.

There is **no single rules-based validator** today. Validation is scattered, OCR-only, and
confidence-shaped rather than rules-shaped (see §6). The Gemini image/text and HH parse
paths have almost none.

### Evidence (scan of all 515 stored snapshots)
A prototype of just the core invariants flagged **18/515** hands. Triage of 10:
- **Real silent bugs (7):** H3517, H2565, H2500, H2838, H2502, H2549, H3485
  - Shape A — **orphan call**: opponent's bet was dropped, leaving `hero Call` with nothing
    to call (H3517 flop `[C UTG+1]`; H2565 flop `[BB X, BB C]`; H2500 flop `[BB C]`;
    H3485 river `[SB X, HJ C]`).
  - Shape B — **act-after-fold**: a position mislabel resurrects a folded seat
    (H2838 `SB` acts though folded; H2502 `LJ` raises though folded; H2549 hero `BTN`
    "folded" preflop yet acts + empty street arrays).
- **False positives (must be designed out):**
  - H2740 — prototype didn't treat `AI14` (sized all-in) as aggression → fake orphan call.
    **Lesson:** validator must recognize ALL aggression: `R<size>`, `AI`/`AI<size>`/`RAI`,
    and the `allin: true` flag.
  - H3511 — multiway hand where hero folded; postflop is rebuilt by
    `_reconcile_preflop_with_streets`, so "folded set from the raw `preflop_actions` string"
    is the wrong participant model. **Lesson:** derive participants from the
    reconciled/analyzed structure, not the raw preflop string.

Prototype scanner (reference only, delete after): was at `scripts/_tmp.py` during design.

---

## 2. Decisions already made (do not re-litigate)

| # | Decision |
|---|---|
| Detection vs repair | **Detection-only.** The validator never mutates the hand. Repairs stay in parsers, *triggered* by the validator's precise error codes. |
| Rollout | **Ship validating directly** (no shadow mode — single user). But the TG reply MUST surface when a parse problem was detected/repaired (see §5). |
| Stack/size checks | **SOFT (warn), not HARD.** OCR stacks are noisy. Only pure-rule violations are HARD. |
| Before/after double-check | **Yes.** Run the validator before *and* after post-processing (`_fix_folded_players`, panel collapses). Valid-before + invalid-after ⇒ our own post-processing corrupted it ⇒ log loudly + prefer the pre-processing version (this is exactly the H3517 cause). |
| Re-parse retry budget | **1 attempt.** |
| User messaging on repair/low-confidence | Add a note like `⚠️ 這手牌解析信心度較低，請再次核對動作順序`. Harry will debug specifics later. |

---

## 3. Module design — `scripts/hand_validator.py` (NEW, pure, no I/O)

```python
@dataclass
class Issue:
    code: str          # ORPHAN_CALL | ACT_AFTER_FOLD | ACTION_AFTER_ALLIN_CALLED |
                       # NON_MONOTONIC_RAISE | OUT_OF_TURN | DUP_CARD | BAD_CARD |
                       # BOARD_COUNT | HERO_POS_INVALID | PREFLOP_LEN | EFFECTIVE_BB |
                       # (SOFT:) ICM_UNCONFIRMED | SIZE_EXCEEDS_STACK | STACKS_LEN
    severity: str      # "hard" | "soft"
    street: str | None # board/card string, or "preflop"
    action_index: int | None
    positions: list[str]
    message: str       # zh-TW, human-readable, localized to the spot
    repair_hint: str   # what the parser most likely got wrong (drives feedback)

@dataclass
class Report:
    ok: bool                 # False iff any hard issue
    hard: list[Issue]
    soft: list[Issue]

def validate_hand(hand: dict, *, participants: dict | None = None) -> Report: ...
def to_parser_feedback(report: Report) -> str:   # render hard issues → correction prompt
```

### 3a. The core: a per-street betting-round state machine
For **each** street (preflop, then each postflop street present), replay the round:

```
participants = players live entering the street
              preflop: all seated (blinds posted)
              postflop: derive from the ANALYZED structure (see §3c), NOT raw preflop string
order        = action order (preflop: first-round = seat order from POSITION_ORDERS;
              postflop: SB, BB, then remaining seats in POSITION_ORDERS order)
state: in_hand set, all_in set, current_bet (bool: has anyone wagered this street),
       committed[pos], (stacks if available — SOFT only)

for each action a in street.actions:
    code, pos = a.action, a.position
    is_aggr  = a.allin OR code starts "R" OR code in {AI, AI<n>, RAI, B, ALLIN}
    is_call  = code starts "C"
    is_check = code in {X, K}

    HARD checks:
      OUT_OF_TURN            : pos not the expected next actor (relaxed/optional — see §7)
      ACT_AFTER_FOLD         : pos in folded set
      ACTION_AFTER_ALLIN_CALLED : the round already closed by a called all-in
      ORPHAN_CALL            : is_call and current_bet == False (nothing to call)
      illegal CHECK          : is_check and current_bet == True (can't check facing a bet)
      NON_MONOTONIC_RAISE    : is_aggr and new amount <= current bet amount

    update: is_aggr→current_bet=True (track amount + all_in); F→folded.add(pos);
            a called all-in closes the round
```
Street-existence rule: street N+1 may exist only if street N's round legally closed with
≥2 players live — **except** a post-all-in runout, which must carry **no decisions** (matches
the existing runout-trim in `n8_parser._build_streets` ~line 2236).

### 3b. Card / structure invariants (HARD)
- No duplicate card across hero_hand + all board cards (absorb
  `n8_parser._duplicate_known_cards:120`).
- Every card is valid rank∈`23456789TJQKA` suit∈`cdhs`; hero_hand = exactly 2 cards.
- Board count matches street: flop `board`=3 cards, turn `card`=1 (cum 4), river `card`=1 (cum 5).
- `hero_position ∈ POSITION_ORDERS[players_at_table]`.
- `preflop_actions` first-round token count == `players_at_table` (continuation tokens after).
- `effective_bb` > 0.

### 3c. Participant derivation (critical to avoid FPs — H3511 lesson)
Do **not** compute the folded set from `preflop_actions[:N]` naively. The validator must use
the same participant model the analyzer uses. Options (pick during impl):
- Preferred: have the caller pass `participants` (flop entrants + per-street folds) derived
  from the already-built `ctx`/streets, OR
- Reconstruct using the same logic as `analyze_hand` / `_reconcile_preflop_with_streets`
  (`scripts/analyze_hand.py`, see [[multiway-preflop-reconcile]]).
Also handle 8-max padding ([[analyze-hand-ctx-double-pad]]): validate against the *physical*
table the actions use, not the padded tree.

### 3d. SOFT invariants (warn, don't block)
- `ICM_UNCONFIRMED`: tournament_type/phase set without an explicit user/stack signal
  (complements the H3518 `possible_ft` work — PR #60).
- `SIZE_EXCEEDS_STACK`: a bet/raise/all-in > effective stack (+tolerance). SOFT — OCR stacks noisy.
- `STACKS_LEN`: `len(player_stacks) != players_at_table`.

---

## 4. Integration points (exact locations)

### 4a. Parse boundary — `src/gemini_session.py`
**Image:** `_parse_hand_from_image` (def at `:2351`). The OCR result is finalized and
post-processed at `:2515-2516` (`_normalize_cards`, `_fix_folded_players`) before the
FAST/MEDIUM returns (`:2563`, `:2573`).
- Run the **before/after double-check** around `_fix_folded_players` (and conceptually around
  the panel collapses in `n8_parser`). Valid-before + invalid-after ⇒ internal corruption ⇒
  log loud, prefer pre-version.
- If the finalized OCR hand is **hard-invalid** ⇒ demote (reuse the existing
  `hand_ok = False` / Gemini-fallback path, e.g. `:2561`) and pass `to_parser_feedback(report)`
  as a **hint** into the Gemini fallback prompt (Channel A, §5).
**Text:** `_parse_hand` (def at `:2889` region) → validate the returned hand; on hard-invalid,
one Gemini re-parse with the feedback appended (Channel B).

### 4b. Analysis boundary — `scripts/analyze_hand.py`
`analyze_hand_full` entry (`:2885` region). Last line of defense for the HH/text paths and
anything that slips through. On hard-invalid: **do not** silently fall through to
`（無 solver 數据）`. Instead return a structured error the bot turns into the §5 message, and
log loudly. This finally separates "parse bug" (caught) from "genuinely off-tree" (rare, valid).

### 4c. OCR confidence feed — `scripts/ocr/n8_parser.py`
Optionally feed hard-issue presence into `confidence_parts` / `diagnostics`
(`_build_diagnostics:1302`, demotions `_apply_structural_confidence_demotions:408`) so the
existing tier gate also reacts. Absorb/My retire the overlapping ad-hoc checks:
`_validate_preflop_bet_physics:2551` (preflop physics), `_check_player_tracking:2314`
(ghost players), `_duplicate_known_cards:120` (dup cards) — keep one source of truth.

---

## 5. Feedback-to-parser + user messaging

### Channel A — OCR path produced the invalid hand
OCR is deterministic, so re-running it is useless. Reaction: **fall back to Gemini full parse,
injecting the issue as a targeted hint** (the stronger parser, aimed at the defect):
> 截圖的 turn 欄位:hero 是 Call,請確認在 hero call 之前是誰下注,不要只記錄 hero 的動作。

### Channel B — Gemini path (image or text) produced the invalid hand
Feed the precise error back and ask Gemini to re-read (**1 retry**):
> 你上一次的解析違反撲克規則:在 turn (board Ad7h3cQc4d),BB 的第一個動作是 Call,
> 但它之前沒有任何下注 — Call 一定要有對象。請重新檢視該街的動作順序(很可能漏掉了對手的
> 下注,或把下注誤判成 check),重新輸出完整 JSON。

`to_parser_feedback(report)` renders hard issues into this correction text from
`message` + `repair_hint` + street/index.

### User-facing TG message (decision #3 — keep it light, Harry debugs later)
- **Repaired after re-parse, or any SOFT/low-confidence flag:** prefix the `📋 H####` card with
  `⚠️ 這手牌解析信心度較低，請再次核對動作順序。`
- **Still hard-invalid after the 1 retry:** do NOT emit bogus GTO. Reply:
  `⚠️ 這手牌的動作解析有矛盾（例如出現沒有對象的 call），可能辨識有誤。請重傳清楚一點的截圖，或用文字描述這手牌。`
- Implementation note: the image flow already has a `possible_ft` message-append pattern at
  `src/gemini_session.py:2325` — mirror it for the validator note.

---

## 6. What exists today (inventory — integrate, don't duplicate)

| Check | Location | Notes |
|---|---|---|
| Preflop bet physics (8 checks) | `n8_parser._validate_preflop_bet_physics:2551` | preflop-only, OCR-only; **absorb** |
| Pot non-decreasing | `n8_parser._check_pot_consistency:2296` | feeds confidence, not pass/fail |
| Folded-player-reappears (postflop) | `n8_parser._check_player_tracking:2314` | overlaps ACT_AFTER_FOLD; **absorb** |
| Duplicate cards | `n8_parser._duplicate_known_cards:120` | hard-zeroes confidence; **absorb** |
| Structural confidence demotions | `n8_parser._apply_structural_confidence_demotions:408` | collapse-loss heuristics |
| All-in fragment repair | `panel_parser._resolve_allin_attribution:316`, `_collapse_dup_allin_badge:537` | keep (repair layer) |
| Tier/confidence gate | `gemini_session._parse_hand_from_image:2351` | FAST≥0.95 / MEDIUM≥0.80 / MIN_CARD_CONF 0.70 / POSTFLOP_COLLAPSE_LOSS≤4 |
| `analyze_hand_full` on bad hand | `analyze_hand.py:2885`, sink `:2603` | never raises → silent `（無 solver 數據）` |
| Declarative schema (jsonschema/pydantic) | — | **none exists** |

Hand JSON contract (the data the validator operates on): see `scripts/analyze_hand.py:7-19`
docstring; `POSITION_ORDERS` at `scripts/analyze_hand.py:68-77`; action codes via
`_street_action_code` (`n8_parser.py:2272`) and `_normalize_preflop_actions`
(`analyze_hand.py:184`). Producers: OCR `n8_parser.parse_n8_screenshot:315`
(`_assemble_hand:1338`, `_build_streets:2118`); Gemini image `IMAGE_PARSE_PROMPT:185`;
Gemini text `PARSE_PROMPT:52`; HH `hh_parser.parse_hand:54`.

---

## 7. Testing strategy (do this — it's how we trust the validator)

1. **Per-invariant unit tests** in `scripts/regression_test.py` (`@test`): one fixture per code
   (orphan call, act-after-fold, action-after-allin-called, non-monotonic raise, illegal check,
   dup card, board count, hero pos invalid, preflop len, effective_bb≤0). Include **negative**
   cases (legal 3-bet pot, legal check-bet-call, legal all-in-called) that must pass.
2. **Aggression-code coverage:** explicit tests that `AI`, `AI14`, `RAI`, and `allin:true` all
   count as aggression (the H2740 FP).
3. **Corpus gate (false-positive guard):** a test that runs `validate_hand` over every stored
   snapshot's parse and asserts the flagged set ⊆ a known triaged list. The ~7 real-bug hands
   (§1) should become fixtures (their *corrected* parse must validate clean). Any *new* FP fails
   the test → forces participant-model/aggression fixes before merge.
4. **Multiway/reconcile FPs:** H3511 must validate clean once participant derivation (§3c) is
   correct. Add it as an explicit regression.
5. Run full `python scripts/regression_test.py` + `python scripts/snapshot_test.py` (baseline
   has 4 env failures + occasional H2494 `.gto_cache` drift — see [[worktree-gto-cache]]).

---

## 8. Suggested build order
1. `scripts/hand_validator.py` + the state machine + Issue/Report + `to_parser_feedback`.
2. Unit tests (§7.1–7.2) — TDD the rules engine.
3. Participant derivation (§3c) wired to the analyzed structure; corpus gate (§7.3–7.4).
4. Integrate at `analyze_hand_full` boundary (loud error instead of silent sink).
5. Integrate at parse boundary + before/after double-check + Channel A/B feedback + retry=1.
6. User-facing TG note (§5).
7. Full regression + snapshot baseline-diff; ship.

## 9. Known edge cases / watch-outs
- 8-max padding vs physical table ([[analyze-hand-ctx-double-pad]]) — validate physical.
- Multiway hero-fold reconcile ([[multiway-preflop-reconcile]]) — participant model.
- ICM/FT soft check overlaps the H3518 `possible_ft` flow (PR #60) — don't double-ask.
- OUT_OF_TURN is the riskiest HARD check (action ordering across mixed OCR labels); consider
  shipping it as SOFT first, promote to HARD after the corpus gate is clean.
- `（無 solver 數據）` is still legitimate for genuinely off-tree but structurally-valid hands —
  only reframe it when the hand is *invalid*.

## 10. Related memory / prior fixes
[[heads-up-villain-position-strip]] (H3517), [[purple-felt-ask-not-assume-ft]] (H3518),
[[multiway-preflop-reconcile]] (H3511), [[analyze-hand-ctx-double-pad]],
[[allin-call-equals-commit]], [[dup-allin-badge-on-call]], [[snapshot-expected-can-be-stale]].
