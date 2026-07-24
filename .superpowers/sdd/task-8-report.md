# Task 8 Report — single-hand resend + in-place overwrite

## Outcome
- Enabled `lvr:<session>:<hand_idx>` owner-only resend flow: tapping 🔁 stores pending `(session_id, hand_idx)`, echoes the current parsed line and full repair explanation, and intercepts the next text message as a corrected single-hand block.
- Added `_apply_live_resend(...)` to reparse/regrade one hand, overwrite the persisted ledger footprint, update the session JSON/page, and edit the original live-session report in place when possible.
- Added `live_flow.splice_hand(...)` and `_recompute_totals(...)` so session replacement preserves display hand number while recomputing totals and queue from retained `dec_rows`.
- Added `live_flow.overwrite_hand(...)` to delete old `ledger_decisions`/`ledger_hands`, remove old queue source contributions, write the new hand if valid, enqueue the recomputed queue, and attach canonical `queue_id`s inside a DB transaction.
- Added `queue_feed.remove_source_hand(...)` to strip a replaced hand from open queue rows, recompute `source_hands` / `n_sources` / `total_ev_loss_bb`, and mark empty auto/live drill rows `cleared` with `clear_reason='resend'`.
- Added migration `20260724010000_drill_queue_resend_clear_reason.sql` because the existing real schema had `clear_reason` but its check constraint only allowed `completed|mistake|skipped`.

## Regression Coverage
- `splice_recompute` — verifies display idx is preserved and totals recompute after replacing one hand.
- `remove_source_hand_recomputes_or_clears_open_rows` — fake-DB coverage for recompute, empty auto/live clear, and empty manual row preservation.
- `lvr_callback_prompts_for_single_hand_and_records_pending_state` — verifies owner callback loads session, sends audit prompt, and records pending resend state.
- `resend_pending_message_intercepts_and_applies_once` — verifies the next text is intercepted before normal `/live`/chat handling and pending state is consumed.
- `apply_live_resend_overwrites_session_and_edits_original_message` — fake integration coverage for apply flow, session update, page render, original message edit, and confirmation.

## Validation
- Schema validation: `grep -R "CREATE TABLE.*drill_queue\|ALTER TABLE.*drill_queue\|clear_reason\|source_hands\|n_sources\|total_ev_loss_bb" -n supabase scripts src` confirmed columns exist; migration widens `drill_queue_clear_reason_check` for `resend`.
- `python -m py_compile scripts/live_flow.py scripts/queue_feed.py src/telegram_bot/bot.py scripts/regression_tests/test_live_flow.py` — passed.
- `python scripts/regression_test.py -k splice` — 1 passed, 0 failed.
- `python scripts/regression_test.py -k remove_source_hand` — 1 passed, 0 failed.
- `python scripts/regression_test.py -k resend` — 3 passed, 0 failed.
- `python scripts/regression_test.py -k lvr` — 1 passed, 0 failed.

## Notes
- No manual Telegram `/live` smoke was run in this coding environment.
- `overwrite_hand` intentionally keeps parse/solver work inline per the task brief; it can be moved behind `asyncio.to_thread` later if latency becomes an issue.
