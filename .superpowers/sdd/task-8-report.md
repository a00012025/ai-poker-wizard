# Task 8 Report — single-hand resend + in-place overwrite

## Outcome
- Enabled `lvr:<session>:<hand_idx>` owner-only resend flow: tapping 🔁 stores pending `(owner_user_id, session_id, hand_idx)`, echoes the current parsed line and full repair explanation, and intercepts only that owner’s next text message as a corrected single-hand block.
- Removed the reentrant `_user_lock` acquisition from the resend branch; `handle_message` already owns the per-chat lock.
- Added `_apply_live_resend(...)` to briefly load session metadata, fetch the requesting user's GTO refresh token, run synchronous parse/solver grading via `asyncio.to_thread` with the per-user thread-local GTO access token set/cleared in the worker, then edit the original live-session report in place when possible. If editing the original message fails, the fallback report message is persisted via `set_session_message`.
- Added `live_flow.splice_hand(...)` and `_recompute_totals(...)` so session replacement preserves display hand number while recomputing totals and queue from retained `dec_rows`.
- Added `live_flow.overwrite_hand(...)` as the single atomic write API: it locks `live_sessions ... FOR UPDATE`, deletes/replaces ledger rows, removes old queue source contributions, writes the new hand, enqueues the recomputed queue, attaches canonical `queue_id`s, and updates `live_sessions.result_json/page` in one transaction.
- Failed or fully ungraded replacements are non-destructive: they report a failure to the owner and leave old ledger/session/queue state unchanged.
- Added `queue_feed.remove_source_hand(...)` to strip a replaced hand from open queue rows, recompute `source_hands` / `n_sources` / `total_ev_loss_bb`, rebuild `drill_url` from remaining sources, clear stale GTOW binding fields, and mark empty auto/live drill rows `cleared` with `clear_reason='resend'`.
- Added migration `20260724010000_drill_queue_resend_clear_reason.sql` because the existing real schema had `clear_reason` but its check constraint only allowed `completed|mistake|skipped`.

## Regression Coverage
- `splice_recompute` — verifies display idx is preserved and totals recompute after replacing one hand.
- `remove_source_hand_recomputes_or_clears_open_rows` — fake-DB coverage for recompute, drill URL rebuild, GTOW binding reset, empty auto/live clear, and empty manual row preservation.
- `lvr_callback_prompts_for_single_hand_and_records_pending_state` — verifies owner callback loads session, sends audit prompt, and records pending resend state bound to owner user id.
- `resend_pending_message_intercepts_and_applies_once` — verifies the owner’s next text is intercepted before normal `/live`/chat handling and pending state is consumed.
- `resend_pending_handle_message_no_reentrant_lock_deadlock` — exercises `handle_message` with a pending resend under `asyncio.wait_for`, proving no nested same-chat lock deadlock.
- `resend_pending_ignores_non_owner_in_shared_chat` — verifies a non-owner/shared-chat message does not pop or apply the owner’s pending resend.
- `overwrite_hand_locks_session_and_updates_ledger_queue_session_atomically` — fake-DB coverage for `FOR UPDATE`, single transaction, ledger mutation, and session result/page update.
- `overwrite_hand_failed_replacement_is_non_destructive` — verifies failed and fully ungraded replacements do not open a transaction or mutate DB state.
- `apply_live_resend_overwrites_session_and_edits_original_message` — fake integration coverage under `POKER_BOT_PROCESS=1` for to-thread pre-parse, no write connection held during parsing, per-user GTO token visible during `process_resend_block`, token cleared afterward, apply flow, page render, original message edit, and confirmation.
- `apply_live_resend_failed_replacement_is_non_destructive` — verifies failed parse/grade reports to the owner and never calls overwrite.
- `apply_live_resend_fallback_persists_new_message_id` — verifies fallback report sends are saved back to `live_sessions.message_id`.

## Validation
- Schema validation: `grep -R "CREATE TABLE.*drill_queue\|ALTER TABLE.*drill_queue\|clear_reason\|source_hands\|n_sources\|total_ev_loss_bb" -n supabase scripts src` confirmed columns exist; migration widens `drill_queue_clear_reason_check` for `resend`.
- `python -m py_compile scripts/live_flow.py scripts/queue_feed.py src/telegram_bot/bot.py scripts/regression_tests/test_live_flow.py` — passed.
- `python scripts/regression_test.py -k splice` — 1 passed, 0 failed.
- `python scripts/regression_test.py -k remove_source_hand` — 1 passed, 0 failed.
- `python scripts/regression_test.py -k overwrite_hand` — 2 passed, 0 failed.
- `python scripts/regression_test.py -k apply_live_resend` — 3 passed, 0 failed.
- `python scripts/regression_test.py -k resend_pending` — 3 passed, 0 failed.
- `python scripts/regression_test.py -k lvr_callback` — 1 passed, 0 failed.
- `python scripts/regression_test.py -k live_json_out` — 1 passed, 0 failed.

## Notes
- No manual Telegram `/live` smoke was run in this coding environment.
- Parse/solver work for resend now runs via `asyncio.to_thread` with bot per-user GTO token setup/cleanup; the DB transaction covers only the atomic write section.
