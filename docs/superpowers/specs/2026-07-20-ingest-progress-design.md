# Live ingest progress bar — design

**Date:** 2026-07-20
**Branch:** `feat/ingest-progress`
**Scope:** UX only. No multi-tenant, no schema change, no ingest-pipeline logic change.

## Problem

Triggering a GTOW hand sync (`/ingest` command **or** the extension ♠ button) shows
`⏳ 已排入同步佇列，完成後會通知你` and then goes silent — often for minutes — until a
single final result message. The runner already computes a `progress` string at each
stage and writes it to `gtow_ingest_requests.progress`, but that value is **only a
liveness heartbeat; it is never shown to the user.** A first-time / large import (e.g.
413 new hands, 241 detail fetches) can take ~9 minutes with no feedback.

## Goal

Surface the progress the runner already produces as a **live-updating Telegram
message**, with a real progress bar + ETA during the long detail-fetch phase, and
per-stage timing so we can later see where the time actually goes.

## What the pipeline emits (bounds what we can honestly show)

From `scripts/ledger_ingest.py`:
- `  list sweep: {new} new...` — streaming paginator, **no denominator** → running count only.
- `  detail sweep: {fetched}/{len(rows)}` (every 100) — **has denominator** → bar + ETA.
- `  list-only sweep: {n}/{total}` (every 500) — denominator → bar.
- Stage labels from `ingest_runner._pass`: `攝取中`, `窗外手牌全量補齊中`, `補 spot 分類`,
  `重建 sessions`, `對數中`.

## Design

All changes in `src/ingest_runner.py` plus a few lines in `src/telegram_bot/bot.py`.
**No DB migration** — single-process bot, so a module-level map bridges the
enqueue→claim handoff. The one-open-request-per-user unique index guarantees at most
one live message per user.

### Components

1. **`_LiveStatus` (new, in ingest_runner)** — owns one Telegram message for a run.
   - `chat_id`, `message_id`, `bot`, `started_at`, `detail_started_at`, `last_edit_at`,
     `last_text`, per-stage timings dict.
   - `async def update(stage_label, raw_line=None)`: format text, debounce (≥4s between
     edits unless the stage changed), skip no-op edits, swallow `BadRequest`
     ("message is not modified") and flood-control errors.
   - `async def settle(final_text)`: best-effort final edit (e.g. `✅ 完成` / `❌`).
   - Pure formatting helpers are module-level functions so they're unit-testable
     without a bot:
     - `parse_progress(line) -> (done, total) | None` — parse `x/total` out of a
       sweep line.
     - `render_bar(done, total, width=10) -> str` — `▓▓▓▓░░░░░░`.
     - `format_eta(done, total, elapsed_s) -> str` — `剩約 3 分` / `剩約 40 秒`
       (rate = done/elapsed; None until we have ≥1 datapoint and elapsed > 0).
     - `render_status(stage_label, parsed, elapsed_s, stage_times) -> str` — assembles
       the full message.

2. **Message ownership / handoff**
   - Module-level `_PENDING_STATUS: dict[int, tuple[chat_id, message_id]]` keyed by
     `user_id`.
   - `ingest_command` (bot.py): keep the immediate `⏳ 已排入同步佇列…` reply; register
     its `(chat_id, message_id)` via a new `ingest_runner.register_status_message(user_id, chat_id, message_id)`.
   - `process_next`: after claiming a row, build `_LiveStatus`:
     - if `_PENDING_STATUS` has this user → reuse that message (edit in place);
     - else (extension path) → `bot.send_message(user_id, "⏳ 開始同步…")` and own it.
   - Pop the map entry on claim so a stale id can't leak into a later run.

3. **Wire `progress()`**
   - `progress(text)` still writes DB `progress`/`heartbeat_at` (unchanged — liveness).
   - It **also** calls `live.update(stage_label, raw_line)`. The `heartbeat` inner
     callback in `_pass` already receives the raw `detail sweep:`/`list sweep:` lines;
     pass them through so the bar advances within a stage, not just at stage boundaries.
   - Record `stage_times[label]` when a stage starts; stamp `detail_started_at` when the
     first `detail sweep:` line appears (ETA anchor).

4. **Terminal state**
   - `_finish` unchanged: still sends a **fresh** final message so the phone pings.
   - Additionally call `live.settle(...)` so the bar doesn't sit frozen at ~98%.
   - Error / stale-expiry paths also settle the bar to `❌` (best-effort).

### Message shape (examples)

```
⏳ GTOW 手牌同步
攝取中 · 已抓 320 筆新手牌…
```
```
⏳ GTOW 手牌同步
攝取中 · ▓▓▓▓▓░░░░░ 52% · 剩約 2 分（126/241）
```
```
⏳ GTOW 手牌同步
重建 sessions…
（攝取 3m10s · 補分類 22s）
```

## Testing

Unit tests (no bot, no network) in `scripts/regression_tests/` or a bot-surface test
file:
- `parse_progress`: `"  detail sweep: 126/241"` → `(126, 241)`; non-matching → `None`.
- `render_bar`: boundaries 0%, 50%, 100%; width honored.
- `format_eta`: rate math, `秒` vs `分` thresholds, `None` when no data.
- `render_status`: denominator-less stage → running count, no bar; detail stage → bar+ETA.
- Debounce: two `update` calls <4s apart with same stage → one edit (inject a fake
  clock + a fake bot recording `edit_message_text` calls).
- No-op / `BadRequest` swallowed without raising.

## Non-goals (explicit)

- Multi-tenant / per-user ledger (dropped per North Star §383).
- Any change to `ledger_ingest` / `ledger_sessions` / detail-fetch concurrency —
  that's the **follow-up perf task**, informed by the per-stage timings this adds.
