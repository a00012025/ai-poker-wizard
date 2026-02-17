# AI Poker Wizard - Development Guidelines

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

All 28 tests must pass. If a test fails, fix the issue before committing.

### What the tests cover

- **Chip EV**: basic preflop, multi-street, re-raise detection, depth mapping
- **Positions**: position orders for 2-9 player tables
- **Range compression**: pair notation (22+), all kickers (AXs), plus notation (K3o+), dash notation (Q2s-Q4s), mixed frequencies (K2o(28%))
- **GTO API**: next_actions, spot_solution, action matching, stacks param, 204/403 handling
- **Formatter**: action summary, hand detail, range by action, hand name normalization
- **ICM**: gametype lookup, stack matching, full preflop analysis, symmetric stacks, 6-max FT, postflop chip EV fallback

### Adding new tests

When adding new features to core analysis logic, add corresponding regression tests to `scripts/regression_test.py`. Use the `@test` decorator and assertion helpers (`assert_eq`, `assert_in`, `assert_true`).
