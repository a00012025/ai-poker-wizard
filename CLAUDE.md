# AI Poker Wizard - Development Guidelines

## Project Structure

```
src/
  gemini_session.py    — Gemini LLM session manager (parse hand → analyze → coach)
  main_gemini.py       — Telegram bot entry point
  telegram_bot/bot.py  — Telegram message handler (.txt/.zip uploads, follow-up by hand_id)
scripts/
  analyze_hand.py      — Multi-street GTO analysis orchestration
  gto_api.py           — GTO Wizard API client (next-actions, spot-solution)
  gto_formatter.py     — Solver JSON → natural language + combo-level breakdown
  gto_token.py         — JWT auth & token refresh (.tokens.json)
  icm_modes.py         — ICM game mode discovery and stack matching
  hand_eval.py         — Deterministic hand type evaluation
  hh_parser.py         — Parse GGPoker HH files → analyze_hand_full() input JSON
  hh_deviation_check.py — Direct GTO API deviation checking per hand
  hh_deviation_report.py — analyze_hands() + format_deviation_report()
  e2e_test.py          — CLI E2E test (no Telegram needed)
  regression_test.py   — Regression test suite
```

## Key Architecture

- **Flow**: User message → Gemini Flash (parse hand JSON) → `analyze_hand_full()` → Gemini Pro (coaching)
- **Follow-ups**: use `query_gto`/`query_next_actions` tools for LLM to query solver on demand
- **ICM modes** are `preflop_only` — postflop falls back to chip EV (`chipev_gametype = "MTTGeneral"`)
- **Position orders** vary by table size (2-9 players), defined in `POSITION_ORDERS` dict

## GTO Wizard API Details

- **Depth format**: `bb + 0.125` (e.g., 30bb → `"30.125"`)
- **ICM stacks**: dash-separated with .125 suffix (e.g., `"50.125-30.125-..."`)
- **Raise codes** are position-dependent in ICM (UTG: R2, CO: R2.1) — must discover via next-actions API
- **API returns** 204 for no solution, 403 for forbidden config — both return `None`
- **1326 Combo Index** (postflop): cards `23456789TJQKA` × suits `cdhs`, outer `j=1..51`, inner `i=0..j-1`
- **169 Hand Index** (preflop): all 169 hand names sorted by ASCII string comparison
  - Hand names: higher rank first, pair=`AA`, suited=`AKs`, offsuit=`AKo`
  - Build: `sorted(all_169_names)` (digits before letters in ASCII: `2,3,...,9,A,J,K,Q,T`)

## Formatter Details

- `_compress_range()`: "+" only when range reaches top kicker, dash notation for partial ranges
- Suit diff triggers when: dominant action differs between combos AND some action spread > 35pp
- Specific combo queries (e.g., "Ah8h") show that combo's strategy prominently, not aggregated "A8s"

## ICM Support

- Triggered when `hand["tournament_type"] == "icm"`
- **Phases**: START/EARLY, PCT75, PCT50, PCT25, PCT10, PCT5, BUBBLEEARLY, BUBBLEMID, BUBBLELATE, FT, T2, T3
- `find_icm_params()` is the high-level entry: returns gametype, depth, stacks, approximation_note
- GGPoker HH files do NOT contain total entries or players remaining — must infer or ask user

## Docker & Deployment

- `.tokens.json` is bind-mounted — do NOT use Write tool (creates new inode, breaks mount)
- To update tokens: write directly into container via `docker compose exec bot sh -c 'cat > /app/.tokens.json ...'`
- Also update host file with in-place write (`open('...', 'r+')` + `truncate()`)
- Deploy: `bash scripts/deploy.sh` (git pull → supabase db push → docker compose build+up)

## Database (Supabase)

- **users**: `user_id` (bigint PK), `username`, `name`, `is_active`, `created_at`, `gto_refresh_token` (text)
  - Token column is `gto_refresh_token`, NOT `refresh_token`
- Migrations: `supabase/migrations/` — always use `supabase db push`, never raw psql

## Ad-hoc Python Scripts

When running ad-hoc Python snippets for debugging/testing, write them to `scripts/_tmp.py` (gitignored) instead of inline `python -c`. This avoids repeated permission prompts.

```bash
python scripts/_tmp.py
```

## E2E Testing

```bash
set -a && source .env && set +a && python scripts/e2e_test.py "有效 50bb, co open ..."
python scripts/e2e_test.py -i "..."  # Interactive mode (multi-turn)
```

## Regression Tests (REQUIRED)

When modifying any of these core analysis files, you MUST run the regression test suite before committing:

- `scripts/analyze_hand.py` — Hand analysis orchestration
- `scripts/gto_api.py` — GTO Wizard API client
- `scripts/gto_formatter.py` — Solver data formatter
- `scripts/icm_modes.py` — ICM game mode discovery
- `src/gemini_session.py` — Gemini session manager (tool execution, query building)

### How to run

```bash
python scripts/regression_test.py
```

All tests must pass. If a test fails, fix the issue before committing.

### What the tests cover

- **Chip EV**: basic preflop, multi-street, re-raise detection, depth mapping
- **Positions**: position orders for 2-9 player tables
- **Range compression**: pair notation (22+), all kickers (AXs), plus notation (K3o+), dash notation (Q2s-Q4s), mixed frequencies (K2o(28%))
- **GTO API**: next_actions, spot_solution, action matching, stacks param, 204/403 handling
- **Formatter**: action summary, hand detail, range by action, hand name normalization
- **ICM**: gametype lookup, stack matching, full preflop analysis, symmetric stacks, 6-max FT, postflop chip EV fallback

### Adding new tests

When adding new features to core analysis logic, add corresponding regression tests to `scripts/regression_test.py`. Use the `@test` decorator and assertion helpers (`assert_eq`, `assert_in`, `assert_true`).

**IMPORTANT: Every bug fix MUST include a regression test.** If it broke once, add a test so it can't break again. This is non-negotiable for all bug reports and fixes.
