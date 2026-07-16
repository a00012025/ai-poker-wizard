---
name: ai-poker-wizard
description: Use when analyzing poker hands, tournament decisions, or GTO strategy questions. Triggers on poker terminology (bb, SB, UTG, all-in, ranges, equity) or requests for hand analysis, strategy advice, or tournament coaching.
---

# AI Poker Wizard

## Overview

Professional poker tournament analysis using GTO Wizard API data and Gemini LLM coaching. Uses direct API calls for fast, precise solver data extraction with combo-level suit frequency analysis.

## When to Use

**Triggers:**
- Poker hand descriptions with positions, stack sizes, actions
- Questions about GTO strategy, ranges, or frequencies
- Tournament decision analysis and ICM considerations
- Hand history review and coaching requests
- Stack depth strategy questions (bb terminology)
- Terms: SB, BB, UTG, BTN, raise, call, fold, all-in, shove

## Project Layout

```
scripts/
  analyze_hand.py    — Multi-street GTO analysis orchestration
  gto_api.py         — GTO Wizard API client (next-actions, spot-solution)
  gto_formatter.py   — Solver JSON → natural language + combo-level breakdown
  gto_token.py       — Per-user JWT access minting + in-memory cache
  gto_owner_token.py — Owner DB token bootstrap for CLI/regression
  e2e_test.py        — CLI E2E test (no Telegram needed)
src/
  main_gemini.py     — Telegram bot entry point
  gemini_session.py  — Session manager (parse → GTO → coaching with tools)
  telegram_bot/bot.py — Telegram message handler
```

## E2E Testing

```bash
# Single query
python scripts/e2e_test.py "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb, co call. flop KsKhQd, sb bet 1/4 co call, turn 3h, sb bet 60% co fold. 打得合理嗎"

# Interactive mode (multi-turn follow-ups)
python scripts/e2e_test.py -i "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb ..."
```

## Hand Analysis via Script

```bash
python scripts/analyze_hand.py --json '<hand_json>'
```

### Hand JSON Format

```json
{
    "gametype": "MTTGeneral",
    "effective_bb": 32,
    "hero_position": "HJ",
    "hero_hand": "66",
    "preflop_actions": "F-F-F-R2.1-F-F-F-C",
    "streets": [
        {
            "board": "Js6h5s",
            "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R2", "size": 2.0},
                {"position": "BB", "action": "C"}
            ]
        },
        {
            "card": "Kc",
            "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R6.6", "size": 6.6},
                {"position": "BB", "action": "C"}
            ]
        }
    ]
}
```

### Key Rules

**Positions (MTT 8-max):** UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5), SB(6), BB(7)

**Preflop actions:** Dash-separated, one per position in order.
- `F` = Fold, `C` = Call, `RX` = Raise to X (e.g., `R2.1`), `AI` = All-in
- Example: CO open, BB call → `F-F-F-F-R2.1-F-F-C`

**Postflop actions:** Each street lists ALL actions including calls.
- `X` = Check, `C` = Call, `F` = Fold, `R` + size = Bet/Raise

**Board notation:** Rank + suit (c/d/h/s). Flop uses `board`, turn/river uses `card`.

**Available depths:** 100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8 (auto-selects nearest)

## GTO Wizard API

### Authentication
- Refresh tokens stored per-user in `users.gto_refresh_token`
- Owner CLI tools resolve `OWNER_CHAT_ID` from DB automatically
- Access token auto-refreshed via `POST /v1/token/refresh/`
- If refresh fails, re-login and sync through the extension or private `/settoken`

### Endpoints

**Next Actions:** `GET /v1/poker/next-actions/`
- Returns available actions at a decision point

**Spot Solution:** `GET /v4/solutions/spot-solution/`
- Returns full strategy data: action frequencies, per-hand strategies, EVs, equity
- Key response fields:
  - `action_solutions[].strategy` — 1326-length array of per-combo frequencies
  - `action_solutions[].total_frequency` — overall frequency
  - `players_info[].range` — 1326-length array of range weights
  - `players_info[].simple_hand_counters[hand]` — per-hand aggregated data

**Common params:** `gametype`, `depth`, `preflop_actions`, `board`, `flop_actions`, `turn_actions`, `river_actions`

**Headers:** `Authorization: Bearer <token>`, `Origin: https://app.gtowizard.com`

### Combo-Level Strategy (1326 Array)

The `strategy` and `range` arrays map to all C(52,2) = 1326 two-card combos:
- Cards: `2c, 2d, 2h, 2s, 3c, ..., Ac, Ad, Ah, As` (ranks ascending, suits `cdhs`)
- Index: outer loop `j=1..51`, inner loop `i=0..j-1`, combo = `(cards[j], cards[i])`
- Last combo (index 1325) = `AsAh`

`gto_formatter.py` automatically shows suit-specific breakdowns when combo strategies differ significantly (different dominant actions + >35pp frequency spread).

## Analysis Framework (繁體中文 output)

1. **近似說明** — Which solver scenario was used and differences
2. **GTO 策略對比** — Per-street solver data vs actual play
3. **花色差異** — Combo-level suit breakdowns when significant
4. **關鍵錯誤** — Biggest deviations from GTO
5. **改進建議** — Actionable recommendations

## Common Mistakes

- Missing calls in postflop actions (must list ALL actions per street)
- Wrong position mapping (count carefully: UTG=0 through BB=7)
- Guessing raise sizes (use auto-discovery via `find_closest_action`)
