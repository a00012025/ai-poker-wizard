# GTO Wizard Custom-Spot Practice URLs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the coarse `fh_start_spot=<street>&fh_actions=<pot_type>` shortcut URLs in the weekly leak report with precise `fh_start_spot=custom_spot` deep-links that encode the representative hand's exact action sequence, hero/villain positions, and board-texture filters — so clicking the link drops the user onto the same GTOW practice spot they actually faced.

**Architecture:** For each weekly-report cluster, pick the top-EV-loss hand as representative, load its full `hand_data` from `hand_histories`, replay the action tree through `get_next_actions` to resolve raw bb sizes into GTOW raise codes (`R2.1`, `R1.9`, `R5.2`, ...), classify board texture into GTOW's per-street filter vocab (`flop_suits` / `flop_paired` / `flop_connectedness` / `turn_suit` / `turn_paired` / `river_suit` / `river_paired` — spec extracted from GTOW frontend JS and click-verified), and assemble the `custom_spot` URL. Gracefully fall back to the existing `gtow_trainer_url.build_trainer_url` bucket URL when the custom build fails (RFI, multiway postflop, cash-game unsupported structures, missing `hand_data`).

**Tech Stack:** Python 3.13, asyncpg, existing `scripts/gto_api.py` client, existing `scripts/gtow_trainer_url.py` (kept as fallback). Also renames four UI terms (`對板→公對面`, `濕板→潮濕面`, `同花板→同花面`, `乾板→乾燥面`) in `weekly_report._BOARD_TEXTURE_ZH`.

---

## Context

### The current (wrong) output

Weekly report line today:

```
**1. SRP 轉牌, 對板 偏離（太 passive）**（SRP 轉牌, 對板, n=4, -0.83bb）
   最貴決策：H2665 · H2671 · H2672
   → [到 GTO Wizard 練這個 spot](https://app.gtowizard.com/practice/trainer?...&fh_start_spot=turn&fh_actions=SRP&dialogs=)
```

The URL lands on a generic "SRP turn" dropdown — user still has to hand-pick the position pair, depth, action sequence, and board filters.

### The desired output (click-verified working custom_spot URL, H2665 preflop/flop/turn action path)

```
https://app.gtowizard.com/practice/trainer
  ?solution_type=gwiz
  &gmfs_solution_tab=ai_sols
  &gametype=MTTGeneral
  &depth=30.125
  &depth_list=30.125
  &…TRAINER_UI_DEFAULTS (same flags as the bucket URL builder)…
  &preflop_actions=F-F-F-F-F-R2.1-F-C
  &flop_actions=R1.9-C
  &turn_actions=R5.2
  &history_spot=11
  &fh_start_spot=custom_spot
  &fh_hero=BTN
  &fh_opponent=BB
  &fh_actions=SRP
  &flop_paired=not_paired
  &flop_suits=rainbow
  &flop_connectedness=disconnected
  &turn_paired=paired
  &turn_suit=flush
  &dialogs=trainer-advanced-filter-dialog_namespace-tra/alpha_tmpNamespace-tmp/primary
```

Reference: working URL Harry supplied (2026-04-18) confirms GTOW accepts `gmfs_solution_tab=ai_sols`, `dialogs=trainer-advanced-filter-dialog_...`, and the full board-filter set (`flop_paired`, `flop_suits`, `flop_connectedness`, `turn_paired`, `turn_suit`, `river_paired`, `river_suit`). Param ordering doesn't matter to GTOW; builder emits in a stable, deterministic order for testability.

H2665-specific values above (rainbow/disconnected/flush/paired) are computed from the hand's actual board (`4c6h8h` flop → rainbow + disconnected; `4h` turn → paired + flush-possible). River flags are **omitted** because H2665 didn't reach river.

### What was verified live (before plan authoring)

Called GTOW `next_actions` with Harry's token and confirmed:

| input (our parsed JSON) | GTOW code at 8-max, 30bb | matches example? |
|---|---|---|
| BTN open 2.2bb (5-max) → pad to 8-max F-F-F-F-F-R?- | `R2.1` | ✓ |
| BB donk 2.7bb on 4c6h8h | `R1.9` (only R option) | ✓ |
| BB donk 5.4bb on 4c6h8h4h | `R5.2` (closest of R1.9/R3.15/R5.2/R7.9/R11.9) | ✓ |
| hero fold vs R5.2 | `F` | ✓ (decision is what we link TO) |
| `history_spot` | 8 preflop + 2 flop + 1 turn = 11 | ✓ |

Conclusion: the raise-code resolver only needs to replay the hand through `get_next_actions` street-by-street and snap each raw bb size to the closest available `R*` code — infrastructure we already own in `gto_api.py`.

### Board-filter spec (authoritative — extracted from GTOW frontend bundle + verified via working deep-link)

Harry pulled the exact filter vocabulary from GTOW's JS **and** click-verified a full custom_spot URL loads the correct page. These are the valid values the trainer accepts:

```js
flop_suits:         ["rainbow", "flush_draw", "monotone"]
flop_paired:        ["not_paired", "paired", "tripled"]
flop_connectedness: ["disconnected", "oesd_possible", "connected"]
flop_subset:        ["25", "49", "85", "184"]                        // not used by us
turn_suit:          ["rainbow", "backdoor", "flush"]
turn_paired:        ["not_paired", "paired"]
river_suit:         ["rainbow", "backdoor", "flush"]                 // rainbow unreachable (pigeonhole)
river_paired:       ["not_paired", "paired"]
```

Confirmed behaviors (from the working deep-link):
- **Comma-separated values = OR** (e.g. `turn_suit=backdoor,rainbow`). We emit single values always; multi-value is GTOW's built-in filter widening.
- **Empty string or omitted param = "any"**. We omit the flag if the street wasn't played.
- **`fh_hero` / `fh_opponent` also accept comma-separated multi-position lists.**
- **`gmfs_solution_tab=ai_sols`** is required for the custom-spot page to show the AI-solutions tab by default.
- **`dialogs=trainer-advanced-filter-dialog_namespace-tra/alpha_tmpNamespace-tmp/primary`** is included verbatim from the working URL; drops the user directly onto the filter-confirmation dialog.

**Naming rule**: flop uses `flop_suits` (plural, 3-card suit distribution); turn/river use `turn_suit` / `river_suit` (singular). Values differ — flop uses `flush_draw` vs. turn/river using `backdoor` for the "two of a suit" case.

**Connectedness** (for 3-card flop only; GTOW doesn't expose `turn_connectedness`/`river_connectedness`):
- `connected` = three consecutive ranks (e.g., 789, 89T)
- `oesd_possible` = at least two adjacent ranks, e.g., 78T or 7JT (one gap of 1, one larger)
- `disconnected` = no adjacencies, e.g., 468 or 259

### What was NOT verified (fallback-guarded, not blocking)

1. **RFI spots (hero is opener, no villain yet on preflop decision)**: `fh_opponent` is ambiguous. Plan falls back to bucket URL when hero has no identifiable single villain.
2. **Multiway postflop (>2 active players)**: GTOW trees are HU postflop. Plan falls back to bucket URL when >2 players remain at decision time.
3. **Cash game depth format**: differs from MTT (no `.125` suffix) — plan uses `nearest_cash_depth`. No live click-test for cash-game trainer URL shape.
4. **A-low straight connectedness** (e.g., A23): first-pass implementation treats A as high (rank index 12); A-2-3 wheel boards may be classified as `oesd_possible` instead of `connected`. Minor edge case; misclassified URL still loads, just a slightly different practice filter.

### Files

- **New**: `scripts/gtow_custom_url.py` — per-hand `custom_spot` URL builder + board classifier.
- **New**: `scripts/gtow_action_resolver.py` — replays a hand through `next_actions` to emit street-by-street GTOW action strings + hero/villain positions + history_spot.
- **Modify**: `scripts/weekly_report.py` — switch cluster URL generation to new path, fall back to existing bucket URL on any failure; rename four board-texture strings.
- **Modify**: `scripts/leak_miner.py` — return top hand's `hand_history_id` in a queryable form (already in `top_hand_ids`; confirm shape suffices).
- **Modify**: `scripts/regression_test.py` — add URL-builder tests for H2665, board-texture classifiers (flop_suits / flop_paired / turn_suit / turn_paired / river_suit / river_paired per GTOW JS spec), and the fallback path. Update one existing assertion that checks `"乾板"` to `"乾燥面"`.
- **Keep as-is (fallback)**: `scripts/gtow_trainer_url.py`.

---

## Task 1: Board-texture classifier

**Files:**
- Create: `scripts/gtow_custom_url.py` (first chunk — just the classifier)
- Test: `scripts/regression_test.py` (append tests)

**Purpose:** Given a board string like `"4c6h8h4h"`, return the GTOW per-street texture filter flags (`flop_paired` / `flop_suits` / `turn_paired` / `turn_suit` / ...) independently for each street present on the board (flop = first 3 cards, turn = 4 cards, river = 5 cards). Values and keys match the GTOW frontend JS spec verbatim.

- [ ] **Step 1: Write the failing tests**

Append to `scripts/regression_test.py` (below the last existing `@test`):

```python
@test
def test_classify_board_rainbow_unpaired():
    """gtow_custom_url: 4c6h8h — rainbow flop, not paired, disconnected (H2665 flop)."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "rainbow")
    assert_eq(r["flop_connectedness"], "disconnected")
    assert_eq(r.get("turn_paired"), None)  # no turn card


@test
def test_classify_board_connected_flop():
    """gtow_custom_url: 7h8d9s — 3 consecutive ranks → connected."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8d9s")
    assert_eq(r["flop_connectedness"], "connected")


@test
def test_classify_board_oesd_possible_flop():
    """gtow_custom_url: 7h8dJc — two adjacent + one gap → oesd_possible."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8dJc")
    assert_eq(r["flop_connectedness"], "oesd_possible")


@test
def test_classify_board_turn_pairs_flop():
    """gtow_custom_url: 4c6h8h4h — flop rainbow, turn pairs the 4, 3 hearts on board."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["flop_suits"], "rainbow")
    # 4 cards suit counts: c=1, h=3 → max 3 → flush
    assert_eq(r["turn_suit"], "flush")


@test
def test_classify_board_turn_backdoor():
    """gtow_custom_url: 4c6h8s2h — flop rainbow, turn brings 2nd heart → backdoor."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8s2h")
    assert_eq(r["flop_suits"], "rainbow")
    # c=1, h=2, s=1 → max 2 → backdoor
    assert_eq(r["turn_suit"], "backdoor")


@test
def test_classify_board_flush_draw_flop():
    """gtow_custom_url: AhKh2c — 2-tone flop → flush_draw."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKh2c")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "flush_draw")


@test
def test_classify_board_monotone_flop():
    """gtow_custom_url: AhKhQh — all hearts → monotone."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKhQh")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "monotone")


@test
def test_classify_board_paired_flop():
    """gtow_custom_url: 7h7d2c — paired flop."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d2c")
    assert_eq(r["flop_paired"], "paired")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_river():
    """gtow_custom_url: 4c6h8h4hKh — river present, turn paired, river not-paired."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4hKh")
    # 5 cards: 4c 6h 8h 4h Kh → c=1, h=4 → max 4 → flush
    assert_eq(r["flop_suits"], "rainbow")
    assert_eq(r["turn_suit"], "flush")
    assert_eq(r["river_suit"], "flush")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["river_paired"], "paired")


@test
def test_classify_board_empty():
    """gtow_custom_url: empty board → empty dict (no keys, not an error)."""
    from gtow_custom_url import classify_board
    assert_eq(classify_board(""), {})
    assert_eq(classify_board(None), {})


@test
def test_classify_board_tripled_flop():
    """gtow_custom_url: 7h7d7s — tripled flop (NOT 'paired')."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d7s")
    assert_eq(r["flop_paired"], "tripled")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_odd_length_raises():
    """gtow_custom_url: odd-length board string → ValueError (caller falls back)."""
    from gtow_custom_url import classify_board
    try:
        classify_board("4c6h8")  # 5 chars — malformed
        assert_true(False, "expected ValueError")
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```
python scripts/regression_test.py
```

Expected: the four new tests fail with `ModuleNotFoundError: No module named 'gtow_custom_url'`.

- [ ] **Step 3: Implement `classify_board`**

Create `scripts/gtow_custom_url.py`:

```python
"""Build GTO Wizard custom-spot practice URLs from parsed hand data.

This module produces the precise `fh_start_spot=custom_spot` deep-links used
by the weekly leak report, replacing the coarse bucket shortcuts emitted by
gtow_trainer_url.build_trainer_url.

Fallback contract: any exception raised from build_custom_spot_url is the
caller's cue to fall back to the bucket URL. The builder itself never catches;
callers that want soft failure wrap in try/except.
"""
from __future__ import annotations

from typing import Literal


def _split_board(board: str) -> list[tuple[str, str]]:
    """'4c6h8h' → [('4','c'),('6','h'),('8','h')]. Empty → []."""
    board = (board or "").strip()
    if not board:
        return []
    if len(board) % 2 != 0:
        raise ValueError(f"board length must be even, got {board!r}")
    return [(board[i], board[i + 1]) for i in range(0, len(board), 2)]


def _paired_flag(cards: list[tuple[str, str]]) -> str:
    """'not_paired' | 'paired' | 'tripled' — matches GTOW flop_paired vocab.

    Turn/river only get 'not_paired' | 'paired' (no 'tripled' there per spec),
    but the same trips detection on a 4/5-card board that contains a three-of-
    a-kind subset is still treated as 'paired' by GTOW's turn_paired/river_paired.
    Caller decides which street key to emit — we just do the detection.
    """
    ranks = [r for r, _ in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    max_count = max(counts.values()) if counts else 0
    if max_count >= 3:
        return "tripled"
    if max_count == 2:
        return "paired"
    return "not_paired"


def _suit_flag_flop(cards: list[tuple[str, str]]) -> str:
    """flop_suits: 'rainbow' | 'flush_draw' | 'monotone'."""
    suits = [s for _, s in cards]
    max_count = max(suits.count(s) for s in set(suits)) if suits else 0
    if max_count >= 3:
        return "monotone"
    if max_count == 2:
        return "flush_draw"
    return "rainbow"


def _suit_flag_turn_river(cards: list[tuple[str, str]]) -> str:
    """turn_suit / river_suit: 'rainbow' | 'backdoor' | 'flush'.

    Different vocab from flop. 'backdoor' = max suit count is exactly 2.
    'flush' = max suit count >= 3 (flush possible on this board state).
    On river, 'rainbow' is unreachable (5 cards into 4 suits, pigeonhole
    guarantees max >= 2) — but we still emit it for correctness.
    """
    suits = [s for _, s in cards]
    max_count = max(suits.count(s) for s in set(suits)) if suits else 0
    if max_count >= 3:
        return "flush"
    if max_count == 2:
        return "backdoor"
    return "rainbow"


_RANK_ORDER = "23456789TJQKA"


def _connectedness_flag(cards: list[tuple[str, str]]) -> str:
    """flop_connectedness: 'connected' | 'oesd_possible' | 'disconnected'.

    Only meaningful on the 3-card flop (GTOW exposes no turn/river
    connectedness filter). Classification by gaps between sorted ranks:
        [1,1] → connected (e.g., 789)
        any gap of 1 but not [1,1] → oesd_possible (e.g., 78T, 67T, 89J with gap)
        no gap of 1 → disconnected (e.g., 468, 259)

    A is treated as high (rank index 12). A-low wheel boards (A-2-3) may be
    classified as oesd_possible instead of connected — documented edge case,
    low impact, URL still loads.
    """
    if len(cards) < 3:
        return ""
    ranks = sorted(_RANK_ORDER.index(r) for r, _ in cards[:3])
    gaps = [ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)]
    if gaps == [1, 1]:
        return "connected"
    if 1 in gaps:
        return "oesd_possible"
    return "disconnected"


def classify_board(board: str) -> dict[str, str]:
    """Classify a board string into GTOW custom-spot board-texture flags.

    Returns keys for whatever streets are present on the board:
      - flop_paired  (not_paired|paired|tripled) / flop_suits  (rainbow|flush_draw|monotone)   — if ≥3 cards
      - turn_paired  (not_paired|paired)         / turn_suit   (rainbow|backdoor|flush)        — if ≥4 cards
      - river_paired (not_paired|paired)         / river_suit  (rainbow|backdoor|flush)        — if ≥5 cards

    Flag names and value vocab are authoritative per GTOW frontend JS:
        flop_suits:   ["rainbow", "flush_draw", "monotone"]
        flop_paired:  ["not_paired", "paired", "tripled"]
        turn_suit:    ["rainbow", "backdoor", "flush"]
        turn_paired:  ["not_paired", "paired"]
        river_suit:   ["rainbow", "backdoor", "flush"]
        river_paired: ["not_paired", "paired"]

    The turn/river paired vocab does not include "tripled" per the spec, so
    even on a board like 7h7d7s2c the turn_paired value is "paired". (The
    flop_paired for that same board IS "tripled" because that's flop state.)
    """
    cards = _split_board(board)
    out: dict[str, str] = {}
    if len(cards) >= 3:
        flop = cards[:3]
        out["flop_paired"]        = _paired_flag(flop)
        out["flop_suits"]         = _suit_flag_flop(flop)
        out["flop_connectedness"] = _connectedness_flag(flop)
    if len(cards) >= 4:
        turn = cards[:4]
        # Collapse 'tripled' → 'paired' for turn/river per GTOW vocab
        p = _paired_flag(turn)
        out["turn_paired"] = "paired" if p == "tripled" else p
        out["turn_suit"]   = _suit_flag_turn_river(turn)
    if len(cards) >= 5:
        river = cards[:5]
        p = _paired_flag(river)
        out["river_paired"] = "paired" if p == "tripled" else p
        out["river_suit"]   = _suit_flag_turn_river(river)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```
python scripts/regression_test.py
```

Expected: all four new tests pass. Whole suite still green.

- [ ] **Step 5: Commit**

```bash
git add scripts/gtow_custom_url.py scripts/regression_test.py
git commit -m "feat(weekly-report): add board-texture classifier for custom-spot URLs"
```

---

## Task 2: GTOW action-code resolver

**Files:**
- Create: `scripts/gtow_action_resolver.py`
- Test: `scripts/regression_test.py`

**Purpose:** Given a `hand_data` dict (same shape as `hand_histories.hand_data`) and the deviation's `(street, action_index)`, replay the hand through `get_next_actions` and emit:

- `preflop_actions: "F-F-F-F-F-R2.1-F-C"` (GTOW codes, padded to 8-max for MTT)
- `flop_actions: "R1.9-C"` (only actions BEFORE hero's decision point)
- `turn_actions: "R5.2"`
- `river_actions: ""` (empty if no river in play or decision on turn)
- `hero_pos: "BTN"`, `villain_pos: "BB"` (8-max labels, post-padding)
- `history_spot: 11` (sum of action slots consumed across all streets before hero's decision)
- `depth: 30.125`
- `gametype: "MTTGeneral"` (or cash type)

- [ ] **Step 1: Write the failing H2665 integration test**

Append to `scripts/regression_test.py`:

```python
@test
def test_resolve_h2665_turn_decision():
    """gtow_action_resolver: H2665 turn fold resolves to R2.1 / R1.9-C / R5.2."""
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 36.7,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {
                "board": "4c6h8h",
                "actions": [
                    {"position": "BB",  "action": "R2.7", "size": 2.7},
                    {"position": "BTN", "action": "C"},
                ],
            },
            {
                "card": "4h",
                "actions": [
                    {"position": "BB",  "action": "R5.4", "size": 5.4},
                    {"position": "BTN", "action": "F"},
                ],
            },
        ],
    }

    result = resolve_actions_for_deviation(
        hand_data, street="turn", action_index=0,
    )

    assert_eq(result["preflop_actions"], "F-F-F-F-F-R2.1-F-C")
    assert_eq(result["flop_actions"], "R1.9-C")
    assert_eq(result["turn_actions"], "R5.2")
    assert_eq(result["river_actions"], "")
    assert_eq(result["hero_pos"], "BTN")
    assert_eq(result["villain_pos"], "BB")
    assert_eq(result["history_spot"], 11)
    assert_eq(result["depth"], 30.125)
    assert_eq(result["gametype"], "MTTGeneral")


@test
def test_resolve_3bet_pot_preflop():
    """gtow_action_resolver: 6-max 40bb CO open, BTN 3bet, CO call, flop decision.

    Ensures multi-raise preflop lines resolve correctly (each R token gets a
    new next_actions lookup that sees the previously-resolved prefix).
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 40.0,
        "hero_position": "CO",
        "players_at_table": 6,
        # 6-max: UTG, HJ, CO, BTN, SB, BB. Here: UTG F, HJ F, CO R2.3, BTN R6.5, SB F, BB F, CO C
        "preflop_actions": "F-F-R2.3-R6.5-F-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "CO",  "action": "X"},
                {"position": "BTN", "action": "X"},
            ]},
        ],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="flop", action_index=0,
    )

    # Padded to 8-max: 2 extra folds at front.
    # Expected shape: F-F-F-F-<COcode>-<BTNcode>-F-F-C (9 tokens)
    pf = result["preflop_actions"].split("-")
    assert_eq(len(pf), 9)
    assert_eq(pf[0:4], ["F", "F", "F", "F"])
    assert_true(pf[4].startswith("R"), f"CO open must be R*, got {pf[4]}")
    assert_true(pf[5].startswith("R"), f"BTN 3bet must be R*, got {pf[5]}")
    assert_eq(pf[6:9], ["F", "F", "C"])
    assert_eq(result["hero_pos"], "CO")
    assert_eq(result["villain_pos"], "BTN")  # last non-hero aggressor


@test
def test_resolve_cash_game_depth_has_no_125():
    """gtow_action_resolver: cash games use nearest_cash_depth without .125 suffix."""
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "Cash6m100",
        "effective_bb": 100.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-R2.5-F-C",
        "streets": [],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="preflop", action_index=0,
    )
    # Cash depth is a plain float, no .125 suffix
    assert_true(
        not str(result["depth"]).endswith(".125"),
        f"cash depth should not have .125 suffix, got {result['depth']}",
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
set -a && source .env && set +a && python scripts/regression_test.py
```

Expected: fails with `ModuleNotFoundError: No module named 'gtow_action_resolver'`.

(This is a network-touching test — it will actually call GTOW `next_actions`. Requires a valid token.)

- [ ] **Step 3: Implement the resolver — padding helper first**

Create `scripts/gtow_action_resolver.py`:

```python
"""Replay a parsed hand through GTOW `next_actions` to resolve raw bb sizes
into GTOW raise codes (R2.1, R1.9, R5.2, ...), for custom-spot URL assembly.

Key decisions:
  - MTTGeneral preflop trees are 8-max. Hands with fewer players are padded
    with extra leading folds so hero's physical position maps onto the 8-max
    seat order. (5-max BTN → 8-max BTN; 6-max CO → 8-max CO.)
  - Cash games use the raw players_at_table preflop tree (no padding).
  - Each action is resolved independently via a next_actions call up to that
    decision point, snapping raw bb to the closest R* code (absolute distance
    match; same heuristic as find_closest_action).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_api import (
    get_next_actions,
    nearest_depth,
    nearest_cash_depth,
    find_closest_action,
)

# Position orders (same source-of-truth as analyze_hand.py POSITION_ORDERS).
# Duplicated here intentionally so this module can be imported without touching
# analyze_hand.py's heavier dependencies.
POSITION_ORDERS: dict[int, list[str]] = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}

MTT_TREE_SIZE = 8  # MTTGeneral preflop tree


def _is_cash(gametype: str) -> bool:
    return (gametype or "").startswith("Cash")


def _pad_preflop_to_mtt_tree(
    preflop_actions_raw: str,
    players_at_table: int,
    hero_position: str,
) -> tuple[str, str, list[str]]:
    """Pad a shorter-table preflop line to the 8-max MTT tree.

    Returns (padded_action_string_with_original_codes, hero_position_8max,
             ordered_positions_list_8max).

    The action string still contains the ORIGINAL raw codes (R2.2 etc.) at
    this stage — raise-code resolution happens per-action in _resolve_raises.
    """
    if players_at_table >= MTT_TREE_SIZE:
        return preflop_actions_raw, hero_position, POSITION_ORDERS[players_at_table][:MTT_TREE_SIZE]

    pad = MTT_TREE_SIZE - players_at_table
    prefix = "F-" * pad
    padded = prefix + (preflop_actions_raw or "")
    # Strip trailing dash if preflop_actions_raw was empty
    padded = padded.rstrip("-")
    return padded, hero_position, POSITION_ORDERS[MTT_TREE_SIZE]


def _resolve_one_raise(
    gametype: str,
    depth: float,
    preflop_actions: str,
    board: str,
    flop_actions: str,
    turn_actions: str,
    river_actions: str,
    target_size: float,
) -> str:
    """Call next_actions at the current node and snap target_size to R* code."""
    resp = get_next_actions(
        gametype=gametype, depth=depth, stacks="",
        preflop_actions=preflop_actions, board=board,
        flop_actions=flop_actions, turn_actions=turn_actions,
        river_actions=river_actions,
    )
    available = resp.get("next_actions", {}).get("available_actions", []) or []
    # find_closest_action expects items shaped {"action": {...}}; the API
    # returns that shape, pass through directly.
    code = find_closest_action(available, target_size)

    # Safety check: if target is clearly a bet (>0) but we got back "X" (check),
    # it means next_actions returned no R* options at this node — the spot is
    # off-tree or the prefix is malformed. Raise so the caller falls back to
    # the bucket URL instead of silently emitting a wrong link.
    if target_size > 0 and code == "X":
        raise ValueError(
            f"no raise options at this node (target={target_size}) — off-tree"
        )
    return code
```

- [ ] **Step 4: Add the core resolver function**

Append to `scripts/gtow_action_resolver.py`:

```python
def _resolve_preflop_codes(
    gametype: str,
    depth: float,
    raw_preflop: str,
    hero_pad: int,
) -> str:
    """Walk preflop action-by-action, replacing each R-size with the GTOW code.

    `raw_preflop` is the padded preflop string still containing raw R2.2-style
    tokens. We replay it left-to-right, calling next_actions at each R* node
    to resolve that position's raise to a GTOW code, building the final
    action string as we go.

    F/C/X tokens pass through unchanged.
    """
    if not raw_preflop:
        return ""

    tokens = raw_preflop.split("-")
    out_tokens: list[str] = []
    prefix_history = ""  # what we've built so far, used in next_actions call

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("R"):
            try:
                target = float(tok[1:])
            except ValueError:
                # e.g. "RAI" (all-in) — pass through
                out_tokens.append(tok)
                prefix_history = "-".join(out_tokens)
                continue
            code = _resolve_one_raise(
                gametype=gametype, depth=depth,
                preflop_actions=prefix_history,
                board="", flop_actions="", turn_actions="", river_actions="",
                target_size=target,
            )
            out_tokens.append(code)
        else:
            out_tokens.append(tok)
        prefix_history = "-".join(out_tokens)

    return "-".join(out_tokens)


def _resolve_street_codes(
    gametype: str,
    depth: float,
    preflop_actions: str,
    board_so_far: str,
    prior_streets: dict[str, str],
    street_key: str,  # "flop" | "turn" | "river"
    raw_actions: list[dict],
    stop_after_n: int,
) -> str:
    """Resolve actions for one postflop street, stopping after N actions.

    Returns the street's action string (e.g. "R1.9-C") containing only the
    first `stop_after_n` actions (hero's decision is at index stop_after_n
    and is excluded).
    """
    out_tokens: list[str] = []
    for i, act in enumerate(raw_actions):
        if i >= stop_after_n:
            break
        action = act.get("action", "")
        if action.startswith("R"):
            target = float(act.get("size") or action[1:] or 0)
            code = _resolve_one_raise(
                gametype=gametype, depth=depth,
                preflop_actions=preflop_actions,
                board=board_so_far,
                flop_actions=prior_streets.get("flop", "") if street_key != "flop" else "-".join(out_tokens),
                turn_actions=prior_streets.get("turn", "") if street_key != "turn" else "-".join(out_tokens),
                river_actions=prior_streets.get("river", "") if street_key != "river" else "-".join(out_tokens),
                target_size=target,
            )
            out_tokens.append(code)
        else:
            out_tokens.append(action)
    return "-".join(out_tokens)


def _identify_villain(
    hand_data: dict,
    hero_pos_8max: str,
    preflop_codes: str,
    street: str,
) -> str | None:
    """Identify the HU postflop opponent.

    Strategy: walk preflop actions, find the LAST non-hero position with a
    non-fold action. That's the preflop caller/aggressor who went to the flop
    with hero. For postflop decisions, also sanity-check streets[] has ≤2
    distinct actors.
    """
    # Positions list is always the 8-max MTT tree — shorter tables are padded
    # with leading folds (see _pad_preflop_to_mtt_tree), so the tokens list
    # aligns 1:1 with the 8-max position sequence regardless of physical size.
    positions = POSITION_ORDERS[MTT_TREE_SIZE]

    tokens = preflop_codes.split("-") if preflop_codes else []
    last_villain: str | None = None
    for pos, tok in zip(positions, tokens):
        if pos == hero_pos_8max:
            continue
        if tok in ("F", ""):
            continue
        last_villain = pos

    if street == "preflop":
        return last_villain

    # Postflop: sanity-check streets[] has ≤2 distinct actors and one is hero.
    for s in hand_data.get("streets", []) or []:
        actors = {a.get("position") for a in (s.get("actions") or [])}
        actors.discard(None)
        if len(actors) > 2 or (last_villain and last_villain not in actors and hero_pos_8max not in actors):
            # Multiway or renaming mismatch → None (caller will fall back).
            return None
    return last_villain


STREET_ORDER = ("preflop", "flop", "turn", "river")


def resolve_actions_for_deviation(
    hand_data: dict[str, Any],
    street: str,
    action_index: int,
) -> dict[str, Any]:
    """Replay a hand and return GTOW-formatted action fields for a deviation.

    Args:
        hand_data: parsed hand (shape matches hand_histories.hand_data /
            analyze_hand_full input).
        street: "preflop" | "flop" | "turn" | "river" — hero's decision street.
        action_index: hero's action index WITHIN that street (0-based).

    Returns a dict with keys:
        preflop_actions, flop_actions, turn_actions, river_actions,
        hero_pos, villain_pos, history_spot, depth, gametype

    Raises on malformed hand_data. Caller catches to fall back to bucket URL.
    """
    if street not in STREET_ORDER:
        raise ValueError(f"street must be one of {STREET_ORDER}, got {street!r}")

    gametype = hand_data.get("gametype") or "MTTGeneral"
    effective_bb = float(hand_data.get("effective_bb") or 30.0)
    hero_pos_raw = hand_data.get("hero_position") or ""
    players = int(hand_data.get("players_at_table") or 8)
    raw_preflop = hand_data.get("preflop_actions") or ""

    depth = nearest_cash_depth(effective_bb) if _is_cash(gametype) else nearest_depth(effective_bb)

    # Pad preflop to 8-max for MTT; cash uses raw
    if _is_cash(gametype):
        padded_preflop = raw_preflop
        hero_pos_8 = hero_pos_raw
    else:
        padded_preflop, hero_pos_8, _ = _pad_preflop_to_mtt_tree(
            raw_preflop, players, hero_pos_raw,
        )

    # Preflop codes — for preflop decisions, truncate to actions BEFORE hero's
    if street == "preflop":
        tokens = padded_preflop.split("-") if padded_preflop else []
        # Find hero's physical slot in the padded preflop sequence:
        # hero is the (pad + hero_idx_in_original)-th token.
        positions = POSITION_ORDERS[MTT_TREE_SIZE] if not _is_cash(gametype) else POSITION_ORDERS[players]
        hero_slot = positions.index(hero_pos_8)
        truncated = "-".join(tokens[:hero_slot + action_index]) if action_index else "-".join(tokens[:hero_slot])
        preflop_codes = _resolve_preflop_codes(gametype, depth, truncated, 0)
        flop_codes = turn_codes = river_codes = ""
        board_so_far = ""
    else:
        preflop_codes = _resolve_preflop_codes(gametype, depth, padded_preflop, 0)
        flop_codes = turn_codes = river_codes = ""
        # Walk streets until (and into) hero's decision street
        streets = hand_data.get("streets") or []
        board_parts: list[str] = []
        prior: dict[str, str] = {}
        target_idx = STREET_ORDER.index(street)
        for i, s in enumerate(streets):
            s_name = STREET_ORDER[i + 1]  # streets[0]=flop, [1]=turn, [2]=river
            board_so_far = "".join(board_parts)
            # Update board for this street BEFORE resolving its actions:
            if i == 0:
                board_parts.append(s.get("board") or "")
            else:
                board_parts.append(s.get("card") or "")
            board_now = "".join(board_parts)
            actions = s.get("actions") or []
            if i + 1 < target_idx:
                # Resolve all actions on this prior street
                resolved = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=len(actions),
                )
                prior[s_name] = resolved
                if s_name == "flop":
                    flop_codes = resolved
                elif s_name == "turn":
                    turn_codes = resolved
                elif s_name == "river":
                    river_codes = resolved
            elif i + 1 == target_idx:
                # Hero's decision street — resolve only actions BEFORE hero's
                resolved = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=action_index,
                )
                if s_name == "flop":
                    flop_codes = resolved
                elif s_name == "turn":
                    turn_codes = resolved
                elif s_name == "river":
                    river_codes = resolved
                break
        board_so_far = "".join(board_parts)

    # Count history_spot: total tokens across preflop + flop + turn + river codes
    def _count(s: str) -> int:
        return len([t for t in s.split("-") if t]) if s else 0

    history_spot = _count(preflop_codes) + _count(flop_codes) + _count(turn_codes) + _count(river_codes)

    # Villain
    villain_pos = _identify_villain(hand_data, hero_pos_8, preflop_codes, street)

    return {
        "preflop_actions": preflop_codes,
        "flop_actions":    flop_codes,
        "turn_actions":    turn_codes,
        "river_actions":   river_codes,
        "hero_pos":        hero_pos_8,
        "villain_pos":     villain_pos,
        "history_spot":    history_spot,
        "depth":           depth,
        "gametype":        gametype,
    }
```

- [ ] **Step 5: Run the H2665 test to verify it passes**

```
set -a && source .env && set +a && python scripts/regression_test.py
```

Expected: `test_resolve_h2665_turn_decision` passes. Full suite still green. Requires valid GTO token.

- [ ] **Step 6: Commit**

```bash
git add scripts/gtow_action_resolver.py scripts/regression_test.py
git commit -m "feat(weekly-report): add GTOW action-code resolver (replay hand via next_actions)"
```

---

## Task 3: Custom-spot URL builder

**Files:**
- Modify: `scripts/gtow_custom_url.py` (add `build_custom_spot_url`)
- Test: `scripts/regression_test.py`

**Purpose:** Assemble the final URL from `classify_board` + `resolve_actions_for_deviation` + the trainer UI defaults. Output must string-equal the user's verified H2665 URL (up to url-encoding and trailing `&` ordering).

- [ ] **Step 1: Add the failing URL-shape test**

Append to `scripts/regression_test.py`:

```python
@test
def test_build_custom_spot_url_h2665():
    """gtow_custom_url: H2665 turn fold → URL with all expected params."""
    from gtow_custom_url import build_custom_spot_url

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 36.7,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {"board": "4c6h8h", "actions": [
                {"position": "BB", "action": "R2.7", "size": 2.7},
                {"position": "BTN", "action": "C"},
            ]},
            {"card": "4h", "actions": [
                {"position": "BB", "action": "R5.4", "size": 5.4},
                {"position": "BTN", "action": "F"},
            ]},
        ],
    }

    url = build_custom_spot_url(
        hand_data, street="turn", action_index=0, pot_type="SRP",
    )

    # Must contain each expected param (url-encoded or raw).
    # Values below are computed from H2665's actual board (4c6h8h + 4h turn):
    #   flop: rainbow suits, not paired, disconnected (ranks 4-6-8, no adjacencies)
    #   turn: completes 3 hearts → flush possible; pairs the 4 → paired
    #   river: hero folded turn, no river flags emitted
    assert_in("fh_start_spot=custom_spot", url)
    assert_in("gmfs_solution_tab=ai_sols", url)
    assert_in("preflop_actions=F-F-F-F-F-R2.1-F-C", url)
    assert_in("flop_actions=R1.9-C", url)
    assert_in("turn_actions=R5.2", url)
    assert_in("history_spot=11", url)
    assert_in("fh_hero=BTN", url)
    assert_in("fh_opponent=BB", url)
    assert_in("fh_actions=SRP", url)
    assert_in("flop_paired=not_paired", url)
    assert_in("flop_suits=rainbow", url)
    assert_in("flop_connectedness=disconnected", url)
    assert_in("turn_paired=paired", url)
    assert_in("turn_suit=flush", url)
    # No river flags since hero folded turn
    assert_true("river_paired" not in url, "river flags should be omitted when hand ended on turn")
    assert_true("river_suit"   not in url, "river flags should be omitted when hand ended on turn")
    assert_in("depth=30.125", url)
    assert_in("depth_list=30.125", url)
    assert_in("gametype=MTTGeneral", url)
    # Dialogs string (special chars will be url-encoded; check a stable prefix)
    assert_in("dialogs=trainer-advanced-filter-dialog", url)


@test
def test_build_custom_spot_url_raises_on_multiway_postflop():
    """gtow_custom_url: >2 distinct postflop actors → CustomSpotBuildError.

    This is the fallback trigger — caller catches and falls back to the
    bucket URL. Without this, we'd emit a link pointing at an HU trainer
    spot that doesn't match the user's actual 3-way hand.
    """
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        # 3-way to flop: CO open, BTN call, BB call
        "preflop_actions": "F-F-R2.5-C-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "BB",  "action": "X"},
                {"position": "CO",  "action": "R1.8", "size": 1.8},
                {"position": "BTN", "action": "C"},
                {"position": "BB",  "action": "C"},
            ]},
        ],
    }
    try:
        build_custom_spot_url(hand_data, street="flop", action_index=3, pot_type="SRP")
        assert_true(False, "expected CustomSpotBuildError for multiway")
    except CustomSpotBuildError:
        pass


@test
def test_build_custom_spot_url_raises_on_unmapped_pot_type():
    """gtow_custom_url: unknown pot_type → CustomSpotBuildError (bucket fallback)."""
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {"gametype": "MTTGeneral", "effective_bb": 30.0,
                 "hero_position": "BTN", "players_at_table": 5,
                 "preflop_actions": "F-F-R2.2-F-C", "streets": []}
    try:
        build_custom_spot_url(
            hand_data, street="flop", action_index=0, pot_type="straddled",
        )
        assert_true(False, "expected CustomSpotBuildError")
    except CustomSpotBuildError:
        pass
```

- [ ] **Step 2: Run the test to verify it fails**

```
set -a && source .env && set +a && python scripts/regression_test.py
```

Expected: `ImportError: cannot import name 'build_custom_spot_url'`.

- [ ] **Step 3: Implement the URL builder**

Append to `scripts/gtow_custom_url.py`:

```python
from urllib.parse import quote, urlencode

# Reuse UI defaults from the bucket URL builder — same trainer UI flags,
# single source of truth. (DRY: don't maintain two copies.)
from gtow_trainer_url import _TRAINER_UI_DEFAULTS, _BASE_URL

_CUSTOM_DIALOGS = "trainer-advanced-filter-dialog_namespace-tra/alpha_tmpNamespace-tmp/primary"

# Pot-type → GTOW fh_actions shortcut (same mapping as gtow_trainer_url)
_POT_TYPE_TO_FH_ACTIONS: dict[str, str] = {
    "SRP":      "SRP",
    "3bet":     "3bet",
    "4bet":     "3bet",   # no dedicated 4bet shortcut; closest is 3bet pot
    "squeezed": "Squeeze",
    "limp":     "limp",
    "iso":      "iso",
}


class CustomSpotBuildError(ValueError):
    """Raised when the custom-spot URL can't be built — caller should fall back."""


def build_custom_spot_url(
    hand_data: dict,
    street: str,
    action_index: int,
    pot_type: str,
) -> str:
    """Build a GTOW custom-spot practice URL for a specific hand decision.

    Raises CustomSpotBuildError when:
      - villain can't be identified (e.g. multiway postflop)
      - pot_type has no fh_actions mapping
      - GTOW next_actions call fails (bubbles up from resolver)
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    fh_actions = _POT_TYPE_TO_FH_ACTIONS.get(pot_type or "")
    if not fh_actions:
        raise CustomSpotBuildError(f"pot_type {pot_type!r} has no fh_actions mapping")

    resolved = resolve_actions_for_deviation(hand_data, street, action_index)
    if not resolved.get("villain_pos"):
        raise CustomSpotBuildError("could not identify HU villain (multiway or RFI)")

    # Board classification: full board across all played streets
    streets = hand_data.get("streets") or []
    board_parts = []
    for i, s in enumerate(streets):
        if i == 0:
            board_parts.append(s.get("board") or "")
        else:
            board_parts.append(s.get("card") or "")
    full_board = "".join(board_parts)
    texture = classify_board(full_board)

    depth_str = f"{resolved['depth']:g}"  # e.g. 30.125 → "30.125", 100.0 → "100"
    if not depth_str.endswith(".125") and resolved["gametype"] == "MTTGeneral":
        depth_str = f"{int(resolved['depth'])}.125"

    params: list[tuple[str, str]] = []
    params.append(("solution_type", _TRAINER_UI_DEFAULTS["solution_type"]))
    params.append(("gametype", resolved["gametype"]))
    params.append(("depth", depth_str))
    params.append(("depth_list", depth_str))
    for k, v in _TRAINER_UI_DEFAULTS.items():
        if k == "solution_type":
            continue
        params.append((k, v))
    params.append(("fh_start_spot", "custom_spot"))
    params.append(("gmfs_solution_tab", "ai_sols"))
    params.append(("preflop_actions", resolved["preflop_actions"]))
    params.append(("history_spot", str(resolved["history_spot"])))
    params.append(("fh_actions", fh_actions))
    params.append(("dialogs", _CUSTOM_DIALOGS))
    # Emit flags in GTOW's expected naming.
    # Order: per-street paired, per-street suit(s), then flop_connectedness.
    # flop uses `flop_suits` (plural, 3-card distribution); turn/river use
    # singular `turn_suit` / `river_suit` with different vocab (backdoor vs flush_draw).
    # flop_connectedness has no turn/river counterpart in GTOW's filter spec.
    for k in ("flop_paired", "turn_paired", "river_paired",
              "flop_suits",  "turn_suit",   "river_suit",
              "flop_connectedness"):
        if k in texture and texture[k]:
            params.append((k, texture[k]))
    params.append(("fh_hero", resolved["hero_pos"]))
    params.append(("fh_opponent", resolved["villain_pos"]))
    if resolved["flop_actions"]:
        params.append(("flop_actions", resolved["flop_actions"]))
    if resolved["turn_actions"]:
        params.append(("turn_actions", resolved["turn_actions"]))
    if resolved["river_actions"]:
        params.append(("river_actions", resolved["river_actions"]))

    return f"{_BASE_URL}?{urlencode(params, quote_via=quote)}"
```

- [ ] **Step 4: Run tests to verify they pass**

```
set -a && source .env && set +a && python scripts/regression_test.py
```

Expected: `test_build_custom_spot_url_h2665` passes. Full suite still green.

- [ ] **Step 5: Click-verify the generated URL**

Run a one-off script to print the URL, paste into browser while logged into GTOW, confirm the trainer lands on: BTN vs BB, 30bb MTT, 4c6h8h rainbow → 4h turn (paired), BTN facing BB turn donk ~5.2bb after BB donked ~1.9bb flop.

```bash
python -c "
import sys, json, os
sys.path.insert(0, 'scripts')
from dotenv import load_dotenv; load_dotenv()
import asyncpg, asyncio
from gtow_custom_url import build_custom_spot_url

async def main():
    pool = await asyncpg.create_pool(os.getenv('SUPABASE_CONN'), statement_cache_size=0)
    async with pool.acquire() as c:
        row = await c.fetchrow(\"SELECT hand_data FROM hand_histories WHERE hand_id='H2665'\")
    hd = row['hand_data']
    if isinstance(hd, str): hd = json.loads(hd)
    url = build_custom_spot_url(hd, street='turn', action_index=0, pot_type='SRP')
    print(url)
    await pool.close()
asyncio.run(main())
"
```

(Yes this violates the CLAUDE.md python-c rule — write it to `scripts/_tmp.py` instead.)

Expected: paste URL into browser → lands on a BTN vs BB turn spot with the right structure. If monotone flag name is wrong, the trainer will load but with a slightly different board filter — note that down and patch the constants in `classify_board`.

- [ ] **Step 6: Commit**

```bash
git add scripts/gtow_custom_url.py scripts/regression_test.py
git commit -m "feat(weekly-report): add custom-spot URL builder matching H2665 URL shape"
```

---

## Task 4: Terminology rename + regression-test update

**Files:**
- Modify: `scripts/weekly_report.py`
- Modify: `scripts/regression_test.py`

**Purpose:** Rename board-texture labels from `對板/濕板/同花板/乾板` to `公對面/潮濕面/同花面/乾燥面`. (Already done in this session — this task documents and verifies.)

- [ ] **Step 1: Confirm rename is already in place**

```
grep -n "乾燥面\|潮濕面\|同花面\|公對面" scripts/weekly_report.py
grep -n "乾燥面\|對板\|濕板\|同花板\|乾板" scripts/regression_test.py
```

Expected: `weekly_report.py` shows all four new strings in `_BOARD_TEXTURE_ZH`; `regression_test.py` shows `乾燥面` (renamed) and no remaining old strings.

- [ ] **Step 2: Run the full suite**

```
python scripts/regression_test.py
```

Expected: all tests pass, including the renamed `test_render_cluster_line_postflop_dry`.

- [ ] **Step 3: Commit if not already committed**

```bash
git status scripts/weekly_report.py scripts/regression_test.py
git add scripts/weekly_report.py scripts/regression_test.py
git commit -m "feat(weekly-report): rename board-texture labels to 公對面/潮濕面/同花面/乾燥面"
```

(If already committed as part of this session's earlier work, skip.)

---

## Task 5: Wire into weekly_report with fallback

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `scripts/regression_test.py`

**Purpose:** Replace `_build_url_for_cluster` body with:
1. Fetch the top hand's `hand_data` from `hand_histories`.
2. Load the deviation row for that hand to get `(street, action_index)`.
3. Call `build_custom_spot_url`; on any exception, fall back to the existing `gtow_trainer_url.build_trainer_url`.

- [ ] **Step 1: Add `top_deviation_id` field to the Cluster dataclass**

Modify `scripts/leak_miner.py:42-67` — add one field to `Cluster` so the URL builder has a direct DB pointer instead of re-querying:

```python
@dataclass
class Cluster:
    key:                 ClusterKey
    sample_count:        int
    total_ev_loss_bb:    float
    avg_ev_loss_bb:      float
    aggression_label:    str
    passive_ratio:       float
    aggressive_ratio:    float
    top_hand_ids:        list[int]
    top_deviation_ids:   list[int]  # NEW: parallel to top_hand_ids, for URL builder
    effective_bb_median: float
    gtow_type:           str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key":                 self.key.to_dict(),
            "sample_count":        self.sample_count,
            "total_ev_loss_bb":    round(self.total_ev_loss_bb, 2),
            "avg_ev_loss_bb":      round(self.avg_ev_loss_bb, 3),
            "aggression_label":    self.aggression_label,
            "passive_ratio":       round(self.passive_ratio, 2),
            "aggressive_ratio":    round(self.aggressive_ratio, 2),
            "top_hand_ids":        list(self.top_hand_ids),
            "top_deviation_ids":   list(self.top_deviation_ids),
            "effective_bb_median": round(self.effective_bb_median, 1),
            "gtow_type":           self.gtow_type,
        }
```

- [ ] **Step 2: Populate `top_deviation_ids` in the SQL**

Modify `scripts/leak_miner.py` — in the `grouped` CTE (lines ~109-127), add another `array_agg` aligned to `top_hand_ids`:

```sql
grouped AS (
  SELECT
    pot_type, street, hero_role, villain_pos, hero_pos, spot_category,
    board_texture,
    COUNT(*)                                                 AS n,
    SUM(ev_loss_estimate)                                    AS total_loss,
    AVG(ev_loss_estimate)                                    AS avg_loss,
    COUNT(*) FILTER (WHERE agg_dir = 'too_passive')          AS n_passive,
    COUNT(*) FILTER (WHERE agg_dir = 'too_aggressive')       AS n_aggressive,
    COUNT(*) FILTER (WHERE agg_dir = 'aligned')              AS n_aligned,
    COUNT(*) FILTER (WHERE agg_dir = 'mixed')                AS n_mixed,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY effective_bb) AS eff_bb_median,
    (array_agg(hand_history_id ORDER BY ev_loss_estimate DESC))[1:3]
                                                             AS top_hands,
    (array_agg(deviation_id      ORDER BY ev_loss_estimate DESC))[1:3]
                                                             AS top_dev_ids,
    (array_agg(gtow_type         ORDER BY ev_loss_estimate DESC))[1] AS dom_gtow_type
  FROM cluster_rows
  GROUP BY 1,2,3,4,5,6,7
  HAVING COUNT(*) >= $4
)
```

And in `cluster_rows`, add `id AS deviation_id` to the SELECT list.

Update the Python unpacking to include `top_deviation_ids=list(r["top_dev_ids"] or [])`.

- [ ] **Step 3: Write the failing _build_url_for_cluster fallback test**

Append to `scripts/regression_test.py`:

```python
@test
def test_build_url_for_cluster_falls_back_on_build_error():
    """weekly_report: if custom builder raises, returns bucket URL."""
    from weekly_report import _build_url_for_cluster

    # Mock cluster with empty top_deviation_ids forces the custom path to fail
    class _FakeKey:
        spot_category = "cbet_ip"
        street = "turn"
        pot_type = "SRP"
        hero_pos = "BTN"
        villain_pos = "BB"
        board_texture = "paired"
    class _FakeCluster:
        key = _FakeKey()
        effective_bb_median = 30.0
        top_hand_ids = []
        top_deviation_ids = []

    url = _build_url_for_cluster(_FakeCluster(), pool=None)
    assert_true(url is not None, "fallback should return bucket URL")
    assert_in("fh_start_spot=turn", url)  # bucket URL marker
    assert_in("fh_actions=SRP", url)
```

- [ ] **Step 4: Run test to see it fails**

```
python scripts/regression_test.py
```

Expected: fails — `_build_url_for_cluster` currently doesn't accept `pool`.

- [ ] **Step 5: Update `_build_url_for_cluster` and its caller**

Modify `scripts/weekly_report.py:412-431`:

```python
async def _build_url_for_cluster(cluster, pool) -> str | None:
    """Custom-spot URL from the top deviation; fall back to bucket URL."""
    from gtow_trainer_url import build_trainer_url, SpotNotSupportedError

    # Try custom spot first
    try:
        from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError
        dev_id = (cluster.top_deviation_ids or [None])[0]
        if dev_id and pool is not None:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT d.street, d.action_index, h.hand_data
                    FROM deviations d
                    JOIN hand_histories h ON h.id = d.hand_history_id
                    WHERE d.id = $1
                    """,
                    dev_id,
                )
            if row and row["hand_data"]:
                import json as _json
                hd = row["hand_data"]
                if isinstance(hd, str):
                    hd = _json.loads(hd)
                return build_custom_spot_url(
                    hd, row["street"], row["action_index"],
                    pot_type=cluster.key.pot_type or "",
                )
    except Exception as e:
        logger.info(f"weekly_report: custom URL failed, fallback to bucket: {e}")

    # Fallback: existing bucket URL
    try:
        return build_trainer_url(
            spot_category=cluster.key.spot_category,
            street=cluster.key.street,
            effective_bb=cluster.effective_bb_median or 30.0,
            pot_type=cluster.key.pot_type,
        )
    except SpotNotSupportedError as e:
        logger.info(f"weekly_report: bucket URL also unsupported ({e})")
        return None
    except Exception as e:
        logger.warning(f"weekly_report: bucket URL builder failed: {e}")
        return None
```

- [ ] **Step 6: Update the call site (now async)**

Modify `scripts/weekly_report.py:_render_report` — _render_report becomes async, and its caller (`generate_weekly_report`) must `await` the per-cluster URL builds. Cleanest approach: compute URLs in `generate_weekly_report` before calling `_render_report`, pass them as a list parallel to `clusters`.

In `generate_weekly_report` (around line 515):

```python
narratives = await generate_cluster_narratives(
    clusters=clusters,
    model_client=model_client,
    max_retries=1,
)

urls = []
for c in clusters:
    urls.append(await _build_url_for_cluster(c, pool))

return _render_report(
    clusters=clusters,
    narratives=narratives,
    urls=urls,
    period_start=period_start,
    period_end=period_end,
    total_hands=totals.get("total_hands") if totals else None,
    total_decisions=totals.get("total_decisions") if totals else None,
)
```

And `_render_report` changes signature:

```python
def _render_report(
    clusters: list,
    narratives: list[ClusterNarrative],
    urls: list[str | None],
    period_start: datetime,
    period_end: datetime,
    total_hands: int | None = None,
    total_decisions: int | None = None,
) -> str:
    ...
    for i, (cluster, narrative, url) in enumerate(zip(clusters, narratives, urls), start=1):
        lines.append(_render_cluster_line(cluster, narrative, url, i))
        lines.append("")
        total_loss += cluster.total_ev_loss_bb
    ...
```

Delete the old sync `_build_url_for_cluster` body defined at 412-431 (it's now an `async def` above).

- [ ] **Step 7: Update the fallback test to be async**

The fallback test in Step 3 used a sync call. Rewrite:

```python
@test
def test_build_url_for_cluster_falls_back_on_build_error():
    """weekly_report: if custom builder fails, returns bucket URL."""
    import asyncio
    from weekly_report import _build_url_for_cluster

    class _FakeKey:
        spot_category = "cbet_ip"
        street = "turn"
        pot_type = "SRP"
        hero_pos = "BTN"
        villain_pos = "BB"
        board_texture = "paired"
    class _FakeCluster:
        key = _FakeKey()
        effective_bb_median = 30.0
        top_hand_ids = []
        top_deviation_ids = []

    url = asyncio.run(_build_url_for_cluster(_FakeCluster(), pool=None))
    assert_true(url is not None, "fallback should return bucket URL")
    assert_in("fh_start_spot=turn", url)
    assert_in("fh_actions=SRP", url)
```

- [ ] **Step 8: Run full regression suite**

```
set -a && source .env && set +a && python scripts/regression_test.py
```

Expected: all tests pass. Any pre-existing weekly_report tests that asserted the OLD URL shape must be updated to check the BUCKET fallback URL (since custom path requires DB + token).

- [ ] **Step 9: Commit**

```bash
git add scripts/leak_miner.py scripts/weekly_report.py scripts/regression_test.py
git commit -m "feat(weekly-report): switch cluster URLs to custom-spot with bucket fallback"
```

---

## Task 6: End-to-end smoke test — send report to Harry's Telegram

**Files:** none (runtime-only verification).

**Purpose:** Trigger a weekly report for `chat_id=556028753` against the live DB + GTOW API + Telegram bot, confirm all 5 cluster URLs now render as custom-spot links, and visually click-verify at least one.

- [ ] **Step 1: Write the trigger script**

Write to `scripts/_tmp.py`:

```python
"""Smoke-test: trigger weekly report for Harry with min_sample=2."""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncpg
from telegram import Bot
from telegram.constants import ParseMode

from leak_miner import mine_clusters
from weekly_report import (
    generate_cluster_narratives,
    _render_report,
    _fetch_period_totals,
    _build_url_for_cluster,
)

CHAT_ID = 556028753
MIN_SAMPLE = 2


async def main():
    dsn = os.getenv("SUPABASE_CONN")
    token = os.getenv("BOT_TOKEN")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, statement_cache_size=0)
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=7)

        clusters = await mine_clusters(
            pool, CHAT_ID, start, end,
            min_sample=MIN_SAMPLE, top_k=5,
        )
        print(f"Clusters: {len(clusters)}")
        totals = await _fetch_period_totals(pool, CHAT_ID, start, end)
        narratives = await generate_cluster_narratives(
            clusters=clusters, model_client=None, max_retries=1,
        )
        urls = [await _build_url_for_cluster(c, pool) for c in clusters]
        for c, u in zip(clusters, urls):
            print(f"  {c.key.spot_category} {c.key.street}: {u[:120] if u else 'None'}...")

        report = _render_report(
            clusters=clusters, narratives=narratives, urls=urls,
            period_start=start, period_end=end,
            total_hands=totals.get("total_hands") if totals else None,
            total_decisions=totals.get("total_decisions") if totals else None,
        )
        print("=" * 60)
        print(report)
        print("=" * 60)

        bot = Bot(token=token)
        async with bot:
            await bot.send_message(
                chat_id=CHAT_ID, text=report, parse_mode=ParseMode.MARKDOWN,
            )
        print(f"Sent to chat_id={CHAT_ID}")
    finally:
        await pool.close()


asyncio.run(main())
```

- [ ] **Step 2: Run it**

```
python scripts/_tmp.py
```

Expected:
- 5 clusters printed, each with a URL containing `fh_start_spot=custom_spot` (not `fh_start_spot=turn`).
- Telegram message arrives in Harry's chat.

- [ ] **Step 3: Click-verify at least one URL**

Open one of the sent URLs in a browser (logged into GTOW). Confirm the trainer lands on the hero/villain pair + depth + street + action sequence the cluster describes (e.g., BTN vs BB, 30bb MTT, SRP turn paired flop).

If the trainer loads but with a wrong board filter, patch `classify_board` constants (step below).

- [ ] **Step 4: Commit smoke-test learnings (if any)**

Board-flag vocab is authoritative per the GTOW JS bundle, so no post-deploy patch is expected. If the smoke-test reveals an edge case (a hand's action path produces a URL that GTOW rejects), fix it in-plan and commit as a follow-up. Otherwise no commit — `scripts/_tmp.py` is gitignored.

---

## Self-Review

**Spec coverage**:
- Terminology rename → Task 4 ✓
- Custom-spot URL with action sequence → Tasks 1-3 ✓
- Per-bucket URL (using top hand as representative) → Task 5 ✓
- Browser verification as needed → Not required. GTOW frontend JS supplied authoritative board-flag vocab (post-review). Task 6 click-verify still validates the full URL shape end-to-end but should pass on first try.

**No placeholders**: Every step has concrete code. All flag names and value vocabulary are authoritative per the GTOW frontend JS bundle; no click-verify contingency remains.

**Type consistency**: `resolve_actions_for_deviation` returns `preflop_actions`, `flop_actions`, etc. (keys match `build_custom_spot_url`'s consumption). `Cluster.top_deviation_ids: list[int]` matches the SQL output shape and `_build_url_for_cluster`'s `top_deviation_ids[0]` access. `_render_report` now takes `urls: list[str | None]` — consistent with the new `generate_weekly_report` flow that precomputes URLs before rendering.

**Known risks** (called out inline, not hidden):
- Board-flag vocab is authoritative: extracted from GTOW frontend JS and click-verified via a working deep-link (Harry, 2026-04-18). All flag names and value enums match the UI filter dialog.
- RFI preflop decisions have no villain → Task 2's `_identify_villain` returns None → Task 3 raises `CustomSpotBuildError` → Task 5 falls back to bucket URL. No silent break.
- Multiway postflop → same fallback chain.
- Cash games → `_is_cash` branch uses raw preflop (no 8-max padding) + `nearest_cash_depth`. Not click-verified in this plan; follow-up if cash users report broken URLs.
- A-low straight boards (e.g., A-2-3 wheel) are classified as `oesd_possible` instead of `connected` because `_connectedness_flag` treats A as high-only. Minor; URL still loads.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-gtow-custom-spot-urls.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

---

## NOT in scope (explicit defers)

- **RFI custom-spot URLs.** Hero opens and villain isn't decided yet preflop — `_identify_villain` returns None → bucket fallback. Rationale: GTOW's `custom_spot` shape for RFI needs extra browser reverse-engineering and isn't in the user's example URL.
- **Multiway postflop custom URLs.** GTOW practice trees are HU postflop. Plan falls back to bucket URL. Rationale: different URL schema, out of scope for this diff.
- **Cash game click-verification.** Cash branch is implemented (`_is_cash`, `nearest_cash_depth`) and tested for depth format, but no browser click-test against live GTOW cash trainer. Rationale: Harry's deviations are ~all MTT; cash users haven't complained yet.
- **Consolidating tree-walk logic** across `analyze_hand.py`, `backfill_ev_loss.py`, and the new `gtow_action_resolver.py`. Rationale: three-way refactor is its own PR; doing it here bloats the diff.
- **Pre-computing resolved action paths** at analysis time (storing in `deviations.meta.gtow_action_path`) so weekly-report URL builds become DB-only, no API calls. Rationale: premature optimization — `gto_api_cache` already absorbs the repeat cost.

## What already exists (reused, not rebuilt)

| Sub-problem | Existing code (reused) | Notes |
|---|---|---|
| Snap raw bb → GTOW raise code | `gto_api.find_closest_action` | Plan uses absolute-size matching; postflop variant exists but overkill here |
| Depth snapping for MTT / cash | `gto_api.nearest_depth`, `nearest_cash_depth` | Reused verbatim |
| `next_actions` HTTP + caching | `gto_api.get_next_actions` + `gto_cache` | Resolver just calls it; no new HTTP layer |
| Bucket URL (fallback) | `gtow_trainer_url.build_trainer_url` | Kept as-is; now also exports `_TRAINER_UI_DEFAULTS` + `_BASE_URL` for DRY |
| Cluster top-hand aggregation | `leak_miner.mine_clusters` | Extended to also return `top_deviation_ids` (Task 5 Step 2) |
| Hand data retrieval | `hand_histories.hand_data` JSONB | Already populated; no schema change needed |
| Parsed-hand shape | `scripts/hh_parser.py` / `analyze_hand_full()` input | Resolver's `hand_data` dict matches this shape exactly |

## Worktree parallelization strategy

| Step | Modules touched | Depends on |
|------|----------------|------------|
| T1 (board classifier) | `scripts/gtow_custom_url.py` (new), `scripts/regression_test.py` | — |
| T2 (action resolver) | `scripts/gtow_action_resolver.py` (new), `scripts/regression_test.py` | — |
| T3 (URL builder) | `scripts/gtow_custom_url.py`, `scripts/regression_test.py` | T1, T2 |
| T4 (terminology) | `scripts/weekly_report.py`, `scripts/regression_test.py` | — (already done in this session) |
| T5 (wire into report) | `scripts/weekly_report.py`, `scripts/leak_miner.py`, `scripts/regression_test.py` | T3 |
| T6 (smoke + click-verify) | none (runtime only) | T5 |

**Lanes:**
- Lane A: T1 → T3 → T5 → T6 (custom URL code path)
- Lane B: T2 (resolver, independent of T1)

**Execution order:** Launch T1 + T2 in parallel worktrees. Merge both into the branch driving T3. Then T3 → T5 → T6 sequentially.

**Conflict flag:** T1, T3, T5 all touch `scripts/regression_test.py` (append-only, per existing `@test` convention) and T1+T3 touch `scripts/gtow_custom_url.py`. Sequential execution within Lane A avoids merge conflicts; the parallelism win is only T1+T2 overlap.

## Failure modes (per new codepath)

| Codepath | Realistic failure | Test? | Error handling? | User-visible? |
|---|---|---|---|---|
| `classify_board` malformed board | odd-length string → ValueError | ✓ added in review | No — raises up | Caught by `build_custom_spot_url` caller → bucket fallback |
| `resolve_actions_for_deviation` → `next_actions` timeout | 5s GTOW timeout | No (network-dependent) | Yes — bubbles up | Bucket URL shown, no user error |
| `_resolve_one_raise` → all-X result | off-tree bet size, API has no R options | No (hard to synthesize) | Yes — added ValueError in review | Bucket fallback |
| `_identify_villain` → multiway | >2 postflop actors | ✓ added in review | Returns None → `CustomSpotBuildError` | Bucket fallback |
| `build_custom_spot_url` → unknown pot_type | e.g. "straddled" | ✓ added in review | Raises `CustomSpotBuildError` | Bucket fallback |
| `_build_url_for_cluster` → hand_data NULL in DB | data integrity glitch | No | Try/except in caller | Bucket fallback |
| Hand with exotic A-low wheel flop (A-2-3) | `_connectedness_flag` misclassifies as `oesd_possible` | No | N/A — GTOW accepts any value | URL loads; practice filter slightly off for wheel boards |

**Critical gaps:** none. All new codepaths either have a test, a try/except in the caller, or an explicit manual click-verify gate.

## Review-cycle patches applied to this plan (summary)

Plan was edited inline based on /plan-eng-review findings before handoff:

1. **DRY**: `_TRAINER_UI_DEFAULTS` + `_BASE_URL` now imported from `gtow_trainer_url.py` instead of duplicated in `gtow_custom_url.py`.
2. **Code hygiene**: removed dead `pad` variable from `_identify_villain`.
3. **Robustness**: `_resolve_one_raise` now raises `ValueError("off-tree")` when target > 0 but GTOW returns only X/F — forces bucket fallback instead of emitting a wrong link.
4. **Test coverage** (7 gaps closed):
   - `test_classify_board_empty` (board="" or None)
   - `test_classify_board_tripled_flop` (777 → tripled, not paired)
   - `test_classify_board_odd_length_raises`
   - `test_resolve_3bet_pot_preflop` (multi-raise preflop line)
   - `test_resolve_cash_game_depth_has_no_125`
   - `test_build_custom_spot_url_raises_on_multiway_postflop`
   - `test_build_custom_spot_url_raises_on_unmapped_pot_type`
5. **Authoritative board-flag spec** (post-review, user-supplied): Harry pulled the GTOW frontend JS and gave us the exact filter vocabulary. Corrected three flag names I'd guessed:
   - `flop_monotone` → **`flop_suits`** (values `rainbow | flush_draw | monotone`)
   - `turn_monotone` → **`turn_suit`** (values `rainbow | backdoor | flush`)
   - `river_monotone` → **`river_suit`** (values `rainbow | backdoor | flush`)
   - Added `tripled` as a third value for `flop_paired`.
   - Updated all tests + `classify_board` body + URL builder loop + documentation to match.
   - Removed the "monotone flag unverified → click-verify" caveat from NOT-in-scope and Task 6.
6. **Click-verified working deep-link** (post-review, user-supplied): Harry provided a working `custom_spot` URL that loads the GTOW trainer directly. Confirmed:
   - `gmfs_solution_tab=ai_sols` and `dialogs=trainer-advanced-filter-dialog_namespace-tra/alpha_tmpNamespace-tmp/primary` are both accepted (verbatim inclusion).
   - `flop_connectedness` is also an exposed filter (values `connected | oesd_possible | disconnected`); added `_connectedness_flag` helper + wired into `classify_board` + added to URL-builder loop.
   - Added two new tests: `test_classify_board_connected_flop` (789), `test_classify_board_oesd_possible_flop` (78J).
   - Updated H2665 URL-assertion test to check `flop_suits=rainbow`, `flop_connectedness=disconnected`, `turn_suit=flush`, and absence of river flags (hero folded turn).
   - Param ordering in URL is irrelevant to GTOW; builder emits deterministic order for test stability.

## Unresolved decisions

None — everything was either fixed inline or explicitly deferred in "NOT in scope".

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 0 arch blockers, 3 code-quality fixes applied inline, 7 test gaps closed inline, 0 perf issues, 0 critical failure-mode gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |

**UNRESOLVED:** 0

**VERDICT:** ENG CLEARED — final. Six bite-sized tasks, TDD-first, with concrete fallback paths at every external-system boundary. Board-flag vocabulary is authoritative per GTOW frontend JS **and** click-verified via a working deep-link (Harry, 2026-04-18). Task 6's click-verify is now a sanity check, not a spec gate. Skip CEO Review — this is a straightforward follow-through on the April 13 design doc's explicitly-deferred "Option Z."

