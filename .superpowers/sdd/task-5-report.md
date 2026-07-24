# Task 5 Report: Per-hand 復盤 URL + button builder

Status: DONE

## Implemented
- `process_batch` computes `entry["review_url"]` for each successfully graded hand via `gtow_solution_url.build_last_hero_hand_url(hand, non_excluded_decisions)`, with a safe `None` fallback.
- Review URL generation failures now emit a debug-level module log with `exc_info=True` and do not interrupt grading or rendering.
- `main()` slim JSON preserves `hand_row` and only removes `dec_rows`, so Task 4 session rendering metadata survives bot handoff.
- Added `session_page_buttons(result, session_id, page, per_page=PER_PAGE)` with per-hand 復盤/教練/加練/重傳 rows, failed-hand resend-only rows, and `lvpg` pagination nav.
- `session_page_buttons` rejects `per_page <= 0` with `ValueError("per_page must be positive")`.
- Kept legacy `report_buttons` behavior intact for existing callers/tests while the new session-specific interface is available.

## Verification
- Red step observed in original Task 5 pass: importing `session_page_buttons` failed before implementation.
- `python scripts/regression_test.py -k per_hand_buttons` → 1 passed, 0 failed.
- `python scripts/regression_test.py -k session_page_buttons_rejects_non_positive_per_page` → 1 passed, 0 failed.
- `python scripts/regression_test.py -k test_live_queue_selection_and_report` → 1 passed, 0 failed.
- `python -m py_compile scripts/live_flow.py scripts/regression_tests/test_live_flow.py` → passed.

## Notes
- The failed-hand button regression now asserts the exact resend-only row: `[{"text": "🔁 重傳", "callback_data": "lvr:<sid>:<idx0>"}]`.
- No snapshot/cache drift is included in the commit.
