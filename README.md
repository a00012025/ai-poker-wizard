# AI Poker Wizard

A poker tournament analysis tool that combines GTO Wizard MTT solver data with Gemini LLM coaching for comprehensive hand review and strategy learning.

## Features
- Natural language hand parsing (Gemini Flash)
- GTO Wizard API integration (direct HTTP, no browser automation)
- Combo-level suit frequency analysis (1326 combos decoded from solver)
- Professional AI poker coach with multi-turn follow-up (Gemini Pro + tool use)
- Telegram bot interface

## Architecture

```
User (Telegram / CLI)
  ↓
GeminiSessionManager
  ├── Parse hand (Gemini Flash)
  ├── GTO analysis (analyze_hand.py → GTO Wizard API)
  │     ├── gto_token.py — JWT auth & refresh
  │     ├── gto_api.py — API client (next-actions, spot-solution)
  │     └── gto_formatter.py — JSON → natural language + combo breakdown
  └── Coaching response (Gemini Pro + query_gto tool for follow-ups)
```

## Getting Started

### Prerequisites
- Python 3.11+
- `GEMINI_API_KEY` — Google Gemini API key
- `BOT_TOKEN` — Telegram bot token (for bot mode)
- Valid GTO Wizard account (refresh token stored in `.tokens.json`)

### Install
```bash
pip install -r requirements.txt
```

### Run Telegram Bot
```bash
python -m src.main_gemini
```

### E2E Test (no Telegram)
```bash
# Single query
python scripts/e2e_test.py "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb, co call. flop KsKhQd, sb bet 1/4 co call, turn 3h, sb bet 60% co fold. 打得合理嗎"

# Interactive mode (multi-turn follow-ups)
python scripts/e2e_test.py -i "有效 50bb, co open 2bb, hero sb AcTh raise 7.5bb ..."
```

### Regression Tests
```bash
python scripts/regression_test.py          # Run all tests
python scripts/regression_test.py -v       # Verbose (with timing)
python scripts/regression_test.py -k icm   # Run only tests matching "icm"
```

28 tests covering: chip EV analysis, position orders, range compression, GTO API calls, formatter output, ICM mode resolution, and ICM analysis.

Requires a valid GTO Wizard token (`.tokens.json`) and network access. Does NOT require `GEMINI_API_KEY`.

### GTO Token Management
```bash
python scripts/gto_token.py   # Print valid access token (auto-refreshes)
```

## Project Structure
```
ai-poker-wizard/
├── src/
│   ├── main_gemini.py         # Entry point (Telegram bot)
│   ├── gemini_session.py      # Session manager (parse → GTO → coaching)
│   ├── telegram_bot/
│   │   └── bot.py             # Telegram bot handler
│   └── claude_session.py      # Legacy Claude CLI session
├── scripts/
│   ├── analyze_hand.py        # Multi-street GTO analysis orchestration
│   ├── gto_api.py             # GTO Wizard API client
│   ├── gto_formatter.py       # Solver JSON → natural language formatter
│   ├── gto_token.py           # Token management (JWT refresh)
│   └── e2e_test.py            # CLI E2E test script
│   ├── icm_modes.py           # ICM game mode discovery & stack matching
│   └── regression_test.py     # Regression test suite (28 tests)
├── .tokens.json               # GTO Wizard auth tokens (git-ignored)
└── .claude/skills/            # Claude Code skill definition
```

## Key Technical Details

### GTO Wizard API
- **Spot Solution** (`GET /v4/solutions/spot-solution/`) — full strategy data
- **Next Actions** (`GET /v1/poker/next-actions/`) — available actions at a decision point
- Auth: Bearer JWT token, auto-refreshed from `.tokens.json`

### Combo-Level Strategy (1326 Array)
The `strategy` array in spot-solution responses contains per-combo frequencies for all C(52,2) = 1326 two-card combinations.

**Index mapping:**
- Cards ordered: `2c, 2d, 2h, 2s, 3c, ..., Ac, Ad, Ah, As` (ranks ascending, suits `cdhs`)
- Combos generated: outer loop `j=1..51`, inner loop `i=0..j-1`, combo = `(cards[j], cards[i])`
- Last combo (index 1325) = `AsAh`

The formatter automatically shows suit-specific breakdowns when different combos of the same hand have significantly different strategies (e.g., flush draw blockers).

### Positions (MTT 8-max)
UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5), SB(6), BB(7)

### Available Solver Depths
100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8 (auto-selects nearest)
