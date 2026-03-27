# E2E Snapshot Regression Test System

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-capture analysis snapshots to Supabase DB on every hand analysis, then replay them as deterministic regression tests — so bug fixes are verified automatically without manual re-testing in Telegram.

**Architecture:** Bot saves snapshot (input + parsed JSON + GTO output + coaching text) to `analysis_snapshots` DB table after every analysis. Image bytes stored as `bytea` for portability. A standalone test runner (`scripts/snapshot_test.py`) fetches regression-flagged snapshots from DB and verifies: (1) OCR/Gemini parsing produces correct JSON, (2) `analyze_hand_full()` produces identical GTO output. Everything lives in Supabase — no local-only state.

**Tech Stack:** asyncpg (bot capture), asyncio.run + asyncpg (test runner), existing regression_test.py framework (`@test` decorator)

---

## File Structure

| File | Role |
|------|------|
| `supabase/migrations/YYYYMMDD_add_analysis_snapshots.sql` | New DB table |
| `src/database.py` | Add `save_snapshot()` and `update_snapshot_coaching()` methods |
| `src/gemini_session.py` | Hook snapshot capture after GTO analysis + coaching |
| `scripts/snapshot_test.py` | Test runner + CLI for managing regression snapshots |

---

### Task 1: DB Migration — `analysis_snapshots` Table

**Files:**
- Create: `supabase/migrations/20260326_add_analysis_snapshots.sql`

- [ ] **Step 1: Create migration file**

```sql
-- Analysis snapshots for E2E regression testing.
-- Captures full pipeline state: input → parse → GTO output → coaching.
CREATE TABLE analysis_snapshots (
    id BIGSERIAL PRIMARY KEY,
    hand_id TEXT NOT NULL UNIQUE,
    chat_id BIGINT,
    source_type TEXT NOT NULL,          -- 'text' | 'image'
    user_input TEXT,                     -- original user message / caption
    image_data BYTEA,                   -- raw screenshot bytes (NULL for text)
    parsed_json JSONB NOT NULL,         -- what Gemini/OCR parsed
    expected_json JSONB,                -- corrected parse (set during bug fix, NULL until then)
    gto_text TEXT NOT NULL,             -- analyze_hand_full()["text"]
    gto_compact TEXT,                   -- analyze_hand_full()["text_compact"]
    coaching_text TEXT,                  -- final LLM coaching response (reference only)
    is_regression BOOLEAN DEFAULT FALSE,-- flagged for regression testing
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_snapshots_hand_id ON analysis_snapshots (hand_id);
CREATE INDEX idx_snapshots_regression ON analysis_snapshots (is_regression) WHERE is_regression = TRUE;
```

- [ ] **Step 2: Apply migration**

```bash
set -a && source .env && set +a && supabase db push
```

Expected: "Remote database is up to date" or migration applied.

- [ ] **Step 3: Add table to startup check**

In `src/database.py` line 11, add `"analysis_snapshots"` to the `_REQUIRED_TABLES` list:

```python
_REQUIRED_TABLES = ["users", "hand_histories", "gto_api_cache", "message_logs", "token_usage", "analysis_snapshots"]
```

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/20260326_add_analysis_snapshots.sql src/database.py
git commit -m "feat: add analysis_snapshots table for E2E regression testing"
```

---

### Task 2: Database Methods for Snapshot CRUD

**Files:**
- Modify: `src/database.py` (append new methods at end of class)

- [ ] **Step 1: Add `save_snapshot()` method**

Append to the `Database` class in `src/database.py`:

```python
    async def save_snapshot(self, hand_id: str, chat_id: int,
                            source_type: str, user_input: str | None,
                            image_data: bytes | None,
                            parsed_json: dict, gto_text: str,
                            gto_compact: str | None = None):
        """Save analysis snapshot. Upsert by hand_id (idempotent)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_snapshots
                    (hand_id, chat_id, source_type, user_input, image_data,
                     parsed_json, gto_text, gto_compact)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (hand_id) DO UPDATE SET
                    parsed_json = $6, gto_text = $7, gto_compact = $8
                """,
                hand_id, chat_id, source_type, user_input, image_data,
                json.dumps(parsed_json), gto_text, gto_compact,
            )

    async def update_snapshot_coaching(self, hand_id: str, coaching_text: str):
        """Update coaching text for an existing snapshot."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE analysis_snapshots SET coaching_text = $1 WHERE hand_id = $2",
                coaching_text, hand_id,
            )
```

- [ ] **Step 2: Verify methods compile**

```bash
python -c "from src.database import Database; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/database.py
git commit -m "feat: add save_snapshot/update_snapshot_coaching to Database"
```

---

### Task 3: Auto-Capture Snapshots in Bot

**Files:**
- Modify: `src/gemini_session.py` — two locations: text analysis path (~line 790) and image analysis path (~line 960)

The snapshot capture happens in two phases:
1. After `analyze_hand_full()` — save everything except coaching
2. After coaching response — update coaching text

Both must be fire-and-forget (non-blocking) using `asyncio.create_task()`.

- [ ] **Step 1: Add helper method to GeminiSession**

Add this method to the `GeminiSession` class (near other helper methods):

```python
    async def _save_snapshot(self, hand_id: str, chat_id: int,
                              source_type: str, user_input: str | None,
                              image_data: bytes | None,
                              parsed_json: dict, context: dict):
        """Fire-and-forget: save analysis snapshot to DB."""
        if not self.db or not hand_id:
            return
        try:
            await self.db.save_snapshot(
                hand_id=hand_id, chat_id=chat_id,
                source_type=source_type,
                user_input=user_input[:2000] if user_input else None,
                image_data=image_data,
                parsed_json=parsed_json,
                gto_text=context.get("text", ""),
                gto_compact=context.get("text_compact"),
            )
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Failed to save snapshot: {e}")

    async def _update_snapshot_coaching(self, hand_id: str, chat_id: int,
                                         coaching_text: str):
        """Fire-and-forget: update coaching text in snapshot."""
        if not self.db or not hand_id:
            return
        try:
            await self.db.update_snapshot_coaching(hand_id, coaching_text)
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Failed to update snapshot coaching: {e}")
```

- [ ] **Step 2: Hook into TEXT analysis path**

In `send_message()`, after `context = analyze_hand_full(hand_json)` and after the coaching result is ready (~line 790-792), add snapshot saves.

After `self.hand_contexts[chat_id] = context` (around line 763):

```python
                # Save snapshot (fire-and-forget)
                import asyncio as _aio
                _aio.create_task(self._save_snapshot(
                    hand_id, chat_id, "text", user_text,
                    None, hand_json, context))
```

After `result = await self._chat_with_tools(...)` and before the timing log (around line 790):

```python
                # Update snapshot with coaching text
                _coaching_only = result.removeprefix(f"📋 `{hand_id}`\n\n") if hand_id else result
                _aio.create_task(self._update_snapshot_coaching(
                    hand_id, chat_id, _coaching_only))
```

- [ ] **Step 3: Hook into IMAGE analysis path**

In `send_image_message()`, after `context = analyze_hand_full(hand_json)` (~line 907):

```python
                # Save snapshot with image bytes (fire-and-forget)
                import asyncio as _aio
                _aio.create_task(self._save_snapshot(
                    hand_id, chat_id, "image", user_text or "[screenshot]",
                    image_bytes, hand_json, context))
```

After the coaching result is built (~line 965 after `result = await self._chat_with_tools(...)`):

```python
                # Update snapshot with coaching text
                _coaching_only = result.removeprefix(f"📋 `{hand_id}`\n\n") if hand_id else result
                _aio.create_task(self._update_snapshot_coaching(
                    hand_id, chat_id, _coaching_only))
```

- [ ] **Step 4: Test by sending a hand to the bot**

Send a test hand via Telegram (or E2E test), then verify the snapshot was saved:

```bash
set -a && source .env && set +a
psql "$SUPABASE_CONN" -c "SELECT hand_id, source_type, length(gto_text) as gto_len, coaching_text IS NOT NULL as has_coaching FROM analysis_snapshots ORDER BY created_at DESC LIMIT 5;"
```

Expected: Recent hand_id appears with gto_len > 0.

- [ ] **Step 5: Commit**

```bash
git add src/gemini_session.py
git commit -m "feat: auto-capture analysis snapshots to DB after every hand analysis"
```

---

### Task 4: Snapshot Test Runner + CLI

**Files:**
- Create: `scripts/snapshot_test.py`

This is the main deliverable. It connects to Supabase, fetches regression snapshots, and runs two test layers.

- [ ] **Step 1: Create `scripts/snapshot_test.py`**

```python
#!/usr/bin/env python3
"""E2E snapshot regression tests.

Fetches regression-flagged snapshots from Supabase and verifies:
  Layer 1 (parse): OCR/Gemini re-parse produces correct hand JSON
  Layer 2 (GTO):   analyze_hand_full() output matches stored snapshot exactly

Usage:
    python scripts/snapshot_test.py                  # Run all regression tests
    python scripts/snapshot_test.py H2489            # Run specific hand
    python scripts/snapshot_test.py --list            # List regression snapshots
    python scripts/snapshot_test.py --add H2489       # Flag snapshot for regression + store expected output
    python scripts/snapshot_test.py --update H2489    # Re-run analysis and update stored GTO output
    python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s",...}'
                                                      # Set corrected expected_json

Requires: SUPABASE_CONN env var, valid GTO Wizard token (.tokens.json).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg


# ── DB helpers (sync wrappers) ──

def _run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def _connect():
    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        print("ERROR: SUPABASE_CONN not set. Run: set -a && source .env && set +a")
        sys.exit(1)
    return await asyncpg.connect(dsn, statement_cache_size=0)


async def _fetch_regression_snapshots(hand_id_filter: str | None = None):
    conn = await _connect()
    try:
        if hand_id_filter:
            rows = await conn.fetch(
                "SELECT * FROM analysis_snapshots WHERE hand_id = $1",
                hand_id_filter,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM analysis_snapshots WHERE is_regression = TRUE ORDER BY hand_id",
            )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _update_snapshot(hand_id: str, **fields):
    conn = await _connect()
    try:
        sets = []
        vals = [hand_id]
        for i, (k, v) in enumerate(fields.items(), start=2):
            sets.append(f"{k} = ${i}")
            vals.append(v)
        sql = f"UPDATE analysis_snapshots SET {', '.join(sets)} WHERE hand_id = $1"
        await conn.execute(sql, *vals)
    finally:
        await conn.close()


# ── Test logic ──

def _compare_parse_fields(parsed: dict, expected: dict) -> list[str]:
    """Compare key fields between parsed and expected hand JSON. Returns list of errors."""
    errors = []
    for key in ["hero_hand", "hero_position", "preflop_actions", "effective_bb",
                 "players_at_table", "tournament_type"]:
        p_val = parsed.get(key)
        e_val = expected.get(key)
        if p_val != e_val and e_val is not None:
            errors.append(f"  {key}: got {p_val!r}, expected {e_val!r}")
    # Compare streets structure
    p_streets = parsed.get("streets") or parsed.get("postflop_actions") or []
    e_streets = expected.get("streets") or expected.get("postflop_actions") or []
    if len(p_streets) != len(e_streets):
        errors.append(f"  streets count: got {len(p_streets)}, expected {len(e_streets)}")
    else:
        for i, (ps, es) in enumerate(zip(p_streets, e_streets)):
            p_board = ps.get("board", ps.get("card", ""))
            e_board = es.get("board", es.get("card", ""))
            if p_board != e_board:
                errors.append(f"  street[{i}] board: got {p_board!r}, expected {e_board!r}")
    return errors


def run_layer1_ocr(snapshot: dict) -> tuple[bool, str]:
    """Layer 1: Re-parse image with OCR and compare to expected_json."""
    image_data = snapshot.get("image_data")
    if not image_data:
        return True, "SKIP (no image data)"

    expected = json.loads(snapshot["expected_json"]) if snapshot.get("expected_json") else json.loads(snapshot["parsed_json"])

    try:
        from ocr.n8_parser import parse_n8_screenshot
        result = parse_n8_screenshot(bytes(image_data))
    except Exception as e:
        return False, f"OCR error: {e}"

    if not result.get("hand"):
        return False, f"OCR returned no hand (confidence={result.get('confidence', 0):.2f})"

    errors = _compare_parse_fields(result["hand"], expected)
    if errors:
        return False, f"Parse mismatch:\n" + "\n".join(errors)

    return True, f"OK (confidence={result.get('confidence', 0):.2f})"


def run_layer1_text(snapshot: dict) -> tuple[bool, str]:
    """Layer 1: Re-parse text with Gemini and compare to expected_json."""
    user_input = snapshot.get("user_input")
    if not user_input or user_input == "[screenshot]":
        return True, "SKIP (no text input)"

    expected = json.loads(snapshot["expected_json"]) if snapshot.get("expected_json") else json.loads(snapshot["parsed_json"])

    try:
        # Gemini parse is async — use sync wrapper
        from gemini_session_parse import parse_hand_sync
        parsed = parse_hand_sync(user_input)
    except ImportError:
        # Fallback: use the session directly
        return True, "SKIP (parse_hand_sync not available)"
    except Exception as e:
        return False, f"Parse error: {e}"

    if not parsed:
        return False, "Parse returned None"

    errors = _compare_parse_fields(parsed, expected)
    if errors:
        return False, f"Parse mismatch:\n" + "\n".join(errors)

    return True, "OK"


def run_layer2_gto(snapshot: dict) -> tuple[bool, str]:
    """Layer 2: Run analyze_hand_full() and compare GTO output exactly."""
    expected_json = snapshot.get("expected_json") or snapshot.get("parsed_json")
    hand_json = json.loads(expected_json) if isinstance(expected_json, str) else expected_json

    expected_text = snapshot["gto_text"]
    expected_compact = snapshot.get("gto_compact")

    try:
        from analyze_hand import analyze_hand_full
        result = analyze_hand_full(hand_json)
    except Exception as e:
        return False, f"Analysis error: {e}"

    actual_text = result["text"]
    actual_compact = result.get("text_compact")

    # Strip timing lines (they vary between runs)
    import re
    strip_timing = lambda s: re.sub(r"⏱ Discovery:.*$", "", s, flags=re.MULTILINE).rstrip()

    expected_stripped = strip_timing(expected_text)
    actual_stripped = strip_timing(actual_text)

    if actual_stripped != expected_stripped:
        # Find first difference for helpful error
        exp_lines = expected_stripped.split("\n")
        act_lines = actual_stripped.split("\n")
        for i, (el, al) in enumerate(zip(exp_lines, act_lines)):
            if el != al:
                return False, (
                    f"GTO text mismatch at line {i+1}:\n"
                    f"  expected: {el[:120]}\n"
                    f"  actual:   {al[:120]}"
                )
        if len(exp_lines) != len(act_lines):
            return False, f"GTO text line count: expected {len(exp_lines)}, got {len(act_lines)}"
        return False, "GTO text differs (unknown location)"

    if expected_compact and actual_compact:
        expected_c = strip_timing(expected_compact)
        actual_c = strip_timing(actual_compact)
        if actual_c != expected_c:
            return False, "GTO compact text mismatch"

    return True, "OK"


# ── CLI commands ──

def cmd_list():
    """List all regression snapshots."""
    snapshots = _run(_fetch_regression_snapshots())
    if not snapshots:
        print("No regression snapshots found.")
        return
    print(f"{'Hand ID':<10} {'Type':<6} {'Expected?':<10} {'Input':<50}")
    print("-" * 80)
    for s in snapshots:
        has_expected = "YES" if s.get("expected_json") else "no"
        user_input = (s.get("user_input") or "")[:48]
        print(f"{s['hand_id']:<10} {s['source_type']:<6} {has_expected:<10} {user_input}")


def cmd_add(hand_id: str):
    """Flag a snapshot for regression testing and store current GTO output as expected."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    hand_json = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    # Re-run analysis to get fresh GTO output
    print(f"Re-running analyze_hand_full() for {hand_id}...")
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full(hand_json)

    _run(_update_snapshot(hand_id,
        gto_text=result["text"],
        gto_compact=result.get("text_compact"),
        is_regression=True,
    ))
    print(f"✅ {hand_id} flagged as regression test. GTO output updated ({len(result['text'])} chars).")


def cmd_update(hand_id: str):
    """Re-run analysis and update stored GTO output (after code fix)."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    hand_json = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    print(f"Re-running analyze_hand_full() for {hand_id}...")
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full(hand_json)

    _run(_update_snapshot(hand_id,
        gto_text=result["text"],
        gto_compact=result.get("text_compact"),
    ))
    print(f"✅ {hand_id} GTO output updated ({len(result['text'])} chars).")


def cmd_set_expected(hand_id: str, json_patch: str):
    """Set corrected expected_json for a snapshot (merge patch into parsed_json)."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    base = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    patch = json.loads(json_patch)
    base.update(patch)

    _run(_update_snapshot(hand_id, expected_json=json.dumps(base)))
    print(f"✅ {hand_id} expected_json updated. Changed fields: {list(patch.keys())}")


def cmd_test(hand_id_filter: str | None = None):
    """Run regression tests."""
    snapshots = _run(_fetch_regression_snapshots(hand_id_filter))
    if hand_id_filter and not snapshots:
        print(f"ERROR: Snapshot {hand_id_filter} not found.")
        sys.exit(1)
    if not snapshots:
        print("No regression snapshots found. Use --add H{id} to flag snapshots.")
        return True

    passed = 0
    failed = 0
    t0 = time.time()

    for s in snapshots:
        hid = s["hand_id"]
        source = s["source_type"]
        print(f"\n{'='*60}")
        print(f"Testing {hid} ({source})")
        print(f"{'='*60}")

        # Layer 1: Parse test
        if source == "image":
            ok, msg = run_layer1_ocr(s)
            label = "L1-OCR"
        else:
            ok, msg = run_layer1_text(s)
            label = "L1-Parse"

        if ok:
            print(f"  \033[32mPASS\033[0m {label}: {msg}")
        else:
            print(f"  \033[31mFAIL\033[0m {label}: {msg}")
            failed += 1

        # Layer 2: GTO output test
        ok2, msg2 = run_layer2_gto(s)
        if ok2:
            print(f"  \033[32mPASS\033[0m L2-GTO: {msg2}")
            passed += 1
        else:
            print(f"  \033[31mFAIL\033[0m L2-GTO: {msg2}")
            failed += 1

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Snapshot tests: {passed} passed, {failed} failed ({total:.1f}s)")
    print(f"{'='*60}")
    return failed == 0


# ── Main ──

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        success = cmd_test()
        sys.exit(0 if success else 1)

    if args[0] == "--list":
        cmd_list()
    elif args[0] == "--add" and len(args) >= 2:
        cmd_add(args[1])
    elif args[0] == "--update" and len(args) >= 2:
        cmd_update(args[1])
    elif args[0] == "--set-expected" and len(args) >= 3:
        cmd_set_expected(args[1], args[2])
    elif args[0].startswith("H"):
        success = cmd_test(args[0])
        sys.exit(0 if success else 1)
    else:
        print(__doc__)
        sys.exit(1)
```

- [ ] **Step 2: Verify it runs (no regression snapshots yet)**

```bash
set -a && source .env && set +a && python scripts/snapshot_test.py
```

Expected: "No regression snapshots found. Use --add H{id} to flag snapshots."

- [ ] **Step 3: Commit**

```bash
git add scripts/snapshot_test.py
git commit -m "feat: snapshot regression test runner with CLI for managing test cases"
```

---

### Task 5: Deploy + Create First Regression Snapshot

**Files:** None (operational task)

This task validates the full pipeline end-to-end.

- [ ] **Step 1: Deploy the bot**

```bash
bash scripts/deploy.sh
```

- [ ] **Step 2: Send a test hand via Telegram**

Send a known hand to the bot (text or screenshot). Note the hand_id from the response (e.g., `H2501`).

- [ ] **Step 3: Verify snapshot was captured**

```bash
set -a && source .env && set +a
psql "$SUPABASE_CONN" -c "SELECT hand_id, source_type, length(gto_text) as gto_len, coaching_text IS NOT NULL as has_coaching, length(image_data) as img_bytes FROM analysis_snapshots ORDER BY created_at DESC LIMIT 5;"
```

Expected: The hand_id appears with gto_len > 0. For image hands, img_bytes should be > 0.

- [ ] **Step 4: Flag it as a regression test**

```bash
set -a && source .env && set +a && python scripts/snapshot_test.py --add H2501
```

Expected: "✅ H2501 flagged as regression test."

- [ ] **Step 5: Run the regression test**

```bash
python scripts/snapshot_test.py H2501
```

Expected: Both L1 and L2 pass.

- [ ] **Step 6: Commit any fixes needed**

If anything fails, fix and re-test before moving on.

---

### Task 6: Bug Fix Workflow — Retrofill H2489

**Files:** None (operational task demonstrating the workflow)

This task demonstrates the intended bug fix workflow using the H2489 example (T9s misidentified as T9o).

- [ ] **Step 1: Check if H2489 snapshot exists**

```bash
set -a && source .env && set +a
python scripts/snapshot_test.py --list
```

If H2489 is not in the list (it was created before the snapshot system), manually check hand_histories:

```bash
psql "$SUPABASE_CONN" -c "SELECT hand_id, hand_data::text FROM hand_histories WHERE hand_id = 'H2489' LIMIT 1;"
```

If the snapshot doesn't exist in `analysis_snapshots`, you'll need to send the same screenshot again after Task 5 deploy so it gets auto-captured.

- [ ] **Step 2: Set the corrected expected_json**

Once the snapshot exists, fix the hero_hand field:

```bash
python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand": "T9s"}'
```

This merges `{"hero_hand": "T9s"}` into the stored parsed_json and saves as expected_json.

- [ ] **Step 3: Fix the OCR/parse bug in code**

(Actual code fix — separate from this plan)

- [ ] **Step 4: Update the expected GTO output**

After fixing the parse code, the GTO analysis output will change (because the input JSON changed). Update it:

```bash
python scripts/snapshot_test.py --update H2489
```

- [ ] **Step 5: Flag as regression test**

```bash
python scripts/snapshot_test.py --add H2489
```

- [ ] **Step 6: Verify**

```bash
python scripts/snapshot_test.py H2489
```

Expected: Both L1 and L2 pass.

---

### Task 7: Update CLAUDE.md with Snapshot Workflow

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add snapshot testing section to CLAUDE.md**

Add after the "Regression Tests" section:

```markdown
## Snapshot Regression Tests (E2E)

Snapshots auto-capture every hand analysis to `analysis_snapshots` DB table (input + parsed JSON + GTO output + coaching text). Image bytes stored as bytea for portability.

### Bug fix workflow

1. User reports: "H2489 has problem, T9s not T9o"
2. Check snapshot: `python scripts/snapshot_test.py --list`
3. Set corrected parse: `python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s"}'`
4. Fix the code (OCR/parse/analysis)
5. Update expected GTO output: `python scripts/snapshot_test.py --update H2489`
6. Flag for regression: `python scripts/snapshot_test.py --add H2489`
7. Verify: `python scripts/snapshot_test.py H2489`
8. Run full suite: `python scripts/snapshot_test.py`

### CLI commands

```bash
python scripts/snapshot_test.py                    # Run all regression tests
python scripts/snapshot_test.py H2489              # Run specific hand
python scripts/snapshot_test.py --list             # List regression snapshots
python scripts/snapshot_test.py --add H2489        # Flag + store expected output
python scripts/snapshot_test.py --update H2489     # Re-run analysis, update expected
python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s"}'
```

### Test layers

- **Layer 1 (Parse)**: Image → OCR re-parse → compare key fields with expected_json. Text → Gemini re-parse → compare.
- **Layer 2 (GTO)**: `analyze_hand_full(expected_json)` → exact match with stored gto_text. Fully deterministic.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add snapshot regression test workflow to CLAUDE.md"
```
