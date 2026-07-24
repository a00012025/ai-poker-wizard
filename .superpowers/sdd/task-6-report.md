# Task 6 Report: Bot wiring — live session persistence + pagination

Status: DONE_WITH_CONCERNS

## Changes
- `src/telegram_bot/bot.py`
  - `/live` now persists the full live batch result with `save_session`, sends page 0 via `render_session_page` + `session_page_buttons`, and records the Telegram `message_id` with `set_session_message`.
  - Added `_send_or_edit_session_page(...)` to re-render persisted live session pages and edit the report message in place while storing the current page via `update_session_result`.
  - Added `lvpg:<session_id>:<page>` callback handling using `load_session`.
  - Extended the existing live/queue callback pattern to include `lvpg|lvadd|lvr`.
  - Added a small guard for `lvadd:`/`lvr:` so those newly-routed callbacks do not accidentally fall through into the old `lvd:` deep-dive path before their dedicated behavior exists.
- `scripts/live_flow.py`
  - `--json-out` now writes the full JSON/session payload instead of stripping `dec_rows`.
  - Added `result_for_json_out(...)` to normalize the full result through JSON with `default=str`, preserving `hand_row` and `dec_rows` while serializing datetimes safely.
- `scripts/regression_tests/test_live_flow.py`
  - Added regression coverage proving JSON/session payloads retain `dec_rows`, retain `hand_row`, serialize datetimes to strings, and still render with `render_session_page`.
  - Updated the existing `/live` subprocess token regression so its fake `live_flow` and DB pool exercise the new success path imports: `save_session`, final `reply_text`, and `set_session_message`, while preserving the original `GTOW_REFRESH_TOKEN` / `POKER_BOT_PROCESS` assertions and asserting no failure edit occurred.

## Validation
- `python -m py_compile scripts/live_flow.py scripts/regression_tests/test_live_flow.py src/telegram_bot/bot.py` → PASS
- `python scripts/regression_test.py -k live_json_out_retains_dec_rows_and_still_renders` → PASS (`1 passed, 0 failed`)
- `python scripts/regression_test.py -k test_live_batch_subprocess_receives_owner_db_token` → PASS (`1 passed, 0 failed`)

## Concerns / Follow-ups
- Manual Telegram integration was not run in this environment.
- `lvadd:` and `lvr:` are routed but only guarded with a user-facing “not enabled yet” alert in this task; dedicated add/resend behavior remains a follow-up if required by the redesign plan.
