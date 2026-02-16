---
name: ai-poker-wizard
description: Use when analyzing poker hands, tournament decisions, or GTO strategy questions. Triggers on poker terminology (bb, SB, UTG, all-in, ranges, equity) or requests for hand analysis, strategy advice, or tournament coaching.
---

# AI Poker Wizard

## Overview

Professional poker tournament analysis using GTO Wizard API data and expert coaching. Uses direct API calls (no browser) for fast, precise solver data extraction.

## When to Use

**Triggers:**
- Poker hand descriptions with positions, stack sizes, actions
- Questions about GTO strategy, ranges, or frequencies
- Tournament decision analysis and ICM considerations
- Hand history review and coaching requests
- Stack depth strategy questions (bb terminology)
- Terms: SB, BB, UTG, BTN, raise, call, fold, all-in, shove

## Hand Analysis via Script

Run the analysis script with a JSON hand description:

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
        },
        {
            "card": "2s",
            "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "X"}
            ]
        }
    ]
}
```

### Key Rules

**Positions (MTT 8-max):** UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5), SB(6), BB(7)

**Preflop actions:** Dash-separated, one per position in order.
- `F` = Fold, `C` = Call, `RX` = Raise to X (e.g., `R2.1`), `AI` = All-in
- Example: CO open, BB call → `F-F-F-F-R2.1-F-F-C` (4 folds, CO raise, BTN fold, SB fold, BB call)

**Postflop actions:** Each street lists ALL actions including calls.
- `X` = Check, `C` = Call, `F` = Fold, `R` + size = Bet/Raise

**Board notation:** Rank + suit (c/d/h/s). Flop uses `board`, turn/river uses `card`.

**Available depths:** 100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8 (script auto-selects nearest)

### Script Output

The script outputs per-street analysis including:
- Active position's available actions with frequencies and combo counts
- Hero hand's specific strategy (per-action frequencies, EV, equity)
- Actual action taken vs solver recommendation

### Individual Scripts

```bash
python scripts/gto_token.py          # Print valid access token
python scripts/gto_api.py            # API client (imported by other scripts)
python scripts/gto_formatter.py      # JSON → natural language (imported by other scripts)
```

## GTO Wizard API (Direct Access)

### Authentication
- Refresh token stored in `.tokens.json` (auto-managed by scripts)
- Access token auto-refreshed via `POST /v1/token/refresh/`
- If refresh fails, opens browser for manual login

### API Endpoints

**Next Actions:** `GET /v1/poker/next-actions/`
- Returns available actions at a decision point (bet sizes, action types)
- Useful for discovering valid bet sizes

**Spot Solution:** `GET /v4/solutions/spot-solution/`
- Returns full strategy data: action frequencies, per-hand strategies, EVs, equity
- Returns 204 when preflop actions are complete and no board specified
- Key fields:
  - `action_solutions[].action.code` — action code (X, R1.9, RAI, etc.)
  - `action_solutions[].total_frequency` — overall frequency
  - `action_solutions[].total_combos` — combo count
  - `players_info[].simple_hand_counters[hand]` — per-hand aggregated data
    - `actions_total_frequencies` — per-action frequency for this hand
    - `hand_ev`, `hand_eq` — EV and equity

**Common params:** `gametype`, `depth`, `preflop_actions`, `board`, `flop_actions`, `turn_actions`, `river_actions`

**Headers required:** `Authorization: Bearer <token>`, `Origin: https://app.gtowizard.com`

## Scenario Approximation

When the user's exact scenario doesn't match GTO Wizard:
1. **Depth**: Use nearest available depth
2. **Bet sizes**: Script auto-maps to closest solver bet size
3. **Always explain** approximations to the user

## Analysis Framework (繁體中文)

1. **手牌概況** — Situation summary
2. **場景近似說明** — Which solver scenario was used and differences
3. **GTO 策略對比** — Per-street solver data vs actual play
4. **關鍵錯誤** — Biggest deviations from GTO (with specific frequencies)
5. **改進建議** — Actionable recommendations

## Common Mistakes

**❌ Wrong:** Including incomplete street actions (missing calls)
**✅ Right:** List ALL actions per street, including calls after bets

**❌ Wrong:** Using wrong position — e.g., "CO" when the solver maps it to "HJ"
**✅ Right:** Count positions carefully. MTT 8-max: UTG(0)→UTG+1(1)→LJ(2)→HJ(3)→CO(4)→BTN(5)→SB(6)→BB(7)

**❌ Wrong:** Guessing raise sizes in preflop_actions
**✅ Right:** Use the script's auto-discovery (it finds the closest solver size)
