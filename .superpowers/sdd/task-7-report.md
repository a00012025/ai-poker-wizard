# Task 7 Report — lvadd live hand add-to-queue menu

## Outcome
- Replaced the temporary `lvadd:` disabled guard with real owner-only callback routing in `handle_live_button`.
- Added `_live_add_menu(...)` to load graded `source='live'` decisions for the selected live hand and render one ➕ button per action line.
- Reused `queue_feed.qex_submenu(..., queue_id=0)`, so selected rows continue through the existing stable `qad2:<queue_id>:<hand>:<street>:<idx>` manual-add path.

## Regression Coverage
- `test_live_add_menu_filters_live_decisions_and_emits_qad2_buttons`
  - verifies `ledger_decisions` query filters `source='live'`, `NOT excluded`, and `NOT discarded`.
  - verifies emitted callback data is `qad2:0:<hand>:<street>:<decision_idx>`.
- `test_lvadd_callback_loads_owner_session_and_opens_live_add_menu`
  - verifies `lvadd:<sid>:<hand_idx>` loads the persisted live session and calls `_live_add_menu` with the requested hand index.

## Validation
- `python -m py_compile src/telegram_bot/bot.py scripts/regression_tests/test_live_flow.py` — passed.
- `python scripts/regression_test.py -k lvadd` — 2 passed, 0 failed.
- `python scripts/regression_test.py -k qex_submenu` — 3 passed, 0 failed.
- `grep -n "qad2\|def qex_submenu" scripts/queue_feed.py` — confirmed `qex_submenu` emits `qad2` callbacks.

## Notes
- `lvr:` remains guarded as previously disabled; this task only enables `lvadd:`.
- Full regression suite was not run; targeted tests cover the changed callback/menu path.
