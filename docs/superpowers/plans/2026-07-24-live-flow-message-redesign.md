# Live Flow Message + Interaction Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `/live` report into a per-hand paginated list with per-hand 復盤/教練/加練/重傳 buttons, rename「主線」→「建議」, auto-escalate off-range nodes one depth bracket, and fix the `b4`-bet parse bug.

**Architecture:** `scripts/live_flow.py` keeps parse+grade+ledger but gains (a) a depth-escalation wrapper around `grade_hand`, (b) a per-hand `review_url`, and (c) a new paginated renderer + button builder. A new `live_sessions` DB table persists the full result JSON so `src/telegram_bot/bot.py` can paginate, add-to-queue, and overwrite single hands in place across bot restarts. Add-to-queue reuses the existing `qad2`/`manual_drill_item` machinery.

**Tech Stack:** Python 3, asyncpg, Supabase migrations, python-telegram-bot v20, Gemini parse, GTO Wizard solver API.

## Global Constraints

- Every bug fix MUST include a regression test in `scripts/regression_tests/` (entry via `scripts/regression_test.py`). Run: `python scripts/regression_test.py` (no `set -a && source .env`).
- All ledger stats queries filter `source='online'`; live rows stay `source='live'` and only surface via live/queue sections (North Star §5.2).
- Queue ranking stays EV-weighted; no frequency-count ordering (§7.3).
- Repairs must remain auditable ([[live-flow-refuse-over-repair]]): the 🔧 per-hand marker + full text on 🔁 resend replaces the removed bulk section — never remove visibility entirely.
- Ad-hoc debug snippets go in `scripts/_tmp.py` (gitignored), never inline `python -c`.
- Depth escalation only ever goes **one** AVAILABLE_DEPTHS bracket up; still-offrange stays ❓.
- Migrations via `supabase db push`, never raw psql.
- Terminology: user-facing Chinese, no 中英夾雜; use「建議」not「主線」.
- All new `/live` callbacks are owner-only (guard with `self._is_owner`).
- Development happens in a worktree `~/ai-poker-wizard-live-redesign` on branch `feat/live-message-redesign`; symlink `.env` and `.gto_cache` in (see [[worktree-gto-cache]]).

---

### Task 0: Worktree setup

**Files:** none (environment only)

- [ ] **Step 1: Create the worktree and branch**

```bash
cd ~/ai-poker-wizard
git fetch origin main && git pull origin main
git worktree add ~/ai-poker-wizard-live-redesign -b feat/live-message-redesign
cd ~/ai-poker-wizard-live-redesign
ln -sf ~/ai-poker-wizard/.env .env
ln -sf ~/ai-poker-wizard/.gto_cache .gto_cache
```

- [ ] **Step 2: Verify baseline tests pass**

Run: `python scripts/regression_test.py`
Expected: all pass (establishes a clean baseline before changes).

---

### Task 1: Terminology 主線 → 建議 (coach copy)

**Files:**
- Modify: `scripts/coach_facts.py:496`
- Modify: `scripts/coach_prompts.py:408`

Note: the `scripts/live_flow.py` render terminology is NOT touched here — Task 4
rewrites that renderer wholesale and writes 建議 from the start (with an explicit
terminology assertion). This task only covers the two coach-copy constants Task 4
does not touch. Verification is a grep (a prompt/string constant has no
meaningful unit test).

**Interfaces:**
- Produces: no signature change; only user-facing copy.

- [ ] **Step 1: Apply the terminology edits**

In `scripts/coach_facts.py:496`:
```python
        facts.note = "你的實際打法偏離 solver 建議，此節點數據僅供參考"
```

In `scripts/coach_prompts.py:408`:
```python
  • ⚪ = 此手牌 0% 到達此節點（off-tree），代表通常前面某街已偏離 GTO 建議、這條街沒有 solver 對照。
```

- [ ] **Step 2: Verify no 主線 remains in these two files**

Run: `grep -rn "主線" scripts/coach_facts.py scripts/coach_prompts.py`
Expected: no output.

- [ ] **Step 3: Verify imports still clean**

Run: `cd scripts && python -c "import coach_facts, coach_prompts" && cd ..`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add scripts/coach_facts.py scripts/coach_prompts.py
git commit -m "refactor: rename 主線 to 建議 in coach copy"
```

---

### Task 2: `live_sessions` DB table + persistence helpers

**Files:**
- Create: `supabase/migrations/20260724000000_live_sessions.sql`
- Modify: `scripts/live_flow.py` (add async helpers `save_session`, `load_session`, `update_session_json`)
- Test: covered by Task 6/8 integration (schema-only task; commit after migration applies)

**Interfaces:**
- Produces:
  - table `live_sessions(id bigserial pk, session_key text unique, chat_id bigint, message_id bigint, page int default 0, result_json jsonb, created_at timestamptz default now())`
  - `async def save_session(conn, session_key: str, chat_id: int, result: dict) -> int` → returns `id`
  - `async def set_session_message(conn, session_id: int, message_id: int) -> None`
  - `async def load_session(conn, session_id: int) -> dict | None` → `{"id","session_key","chat_id","message_id","page","result"}`
  - `async def update_session_result(conn, session_id: int, result: dict, page: int) -> None`

- [ ] **Step 1: Write the migration**

Create `supabase/migrations/20260724000000_live_sessions.sql`:

```sql
CREATE TABLE IF NOT EXISTS live_sessions (
    id           bigserial PRIMARY KEY,
    session_key  text UNIQUE NOT NULL,
    chat_id      bigint NOT NULL,
    message_id   bigint,
    page         int NOT NULL DEFAULT 0,
    result_json  jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS live_sessions_created_at_idx
    ON live_sessions (created_at DESC);
```

- [ ] **Step 2: Apply the migration**

Run: `supabase db push`
Expected: migration applies; `live_sessions` created.

- [ ] **Step 3: Add persistence helpers to `scripts/live_flow.py`**

Append near the other DB functions (after `persist`):

```python
async def save_session(conn, session_key: str, chat_id: int,
                       result: dict) -> int:
    """Insert/replace a live session; returns its id. Idempotent on key."""
    return await conn.fetchval(
        "INSERT INTO live_sessions (session_key, chat_id, result_json) "
        "VALUES ($1, $2, $3) "
        "ON CONFLICT (session_key) DO UPDATE SET "
        "result_json = EXCLUDED.result_json, chat_id = EXCLUDED.chat_id "
        "RETURNING id",
        session_key, chat_id, json.dumps(result, ensure_ascii=False, default=str))


async def set_session_message(conn, session_id: int, message_id: int) -> None:
    await conn.execute(
        "UPDATE live_sessions SET message_id=$2 WHERE id=$1",
        session_id, message_id)


async def load_session(conn, session_id: int) -> dict | None:
    row = await conn.fetchrow(
        "SELECT id, session_key, chat_id, message_id, page, result_json "
        "FROM live_sessions WHERE id=$1", session_id)
    if not row:
        return None
    return {"id": row["id"], "session_key": row["session_key"],
            "chat_id": row["chat_id"], "message_id": row["message_id"],
            "page": row["page"], "result": json.loads(row["result_json"])}


async def update_session_result(conn, session_id: int, result: dict,
                                page: int) -> None:
    await conn.execute(
        "UPDATE live_sessions SET result_json=$2, page=$3 WHERE id=$1",
        session_id, json.dumps(result, ensure_ascii=False, default=str), page)
```

- [ ] **Step 4: Smoke-test the helpers**

Write `scripts/_tmp.py`:

```python
import asyncio, os, asyncpg
from dotenv import load_dotenv
load_dotenv()
from live_flow import save_session, load_session, update_session_result

async def main():
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    sid = await save_session(conn, "live:test:deadbeef", 123, {"hands": [], "page": 0})
    got = await load_session(conn, sid)
    assert got["chat_id"] == 123 and got["result"] == {"hands": [], "page": 0}
    await update_session_result(conn, sid, {"hands": [1], "page": 2}, 2)
    got2 = await load_session(conn, sid)
    assert got2["result"]["hands"] == [1]
    await conn.execute("DELETE FROM live_sessions WHERE id=$1", sid)
    await conn.close()
    print("OK")

asyncio.run(main())
```

Run: `python scripts/_tmp.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add supabase/migrations/20260724000000_live_sessions.sql scripts/live_flow.py
git commit -m "feat: live_sessions table + persistence helpers"
```

---

### Task 3: Auto-escalate off-range nodes one depth bracket

**Files:**
- Modify: `scripts/live_flow.py` (add `_next_depth_up`, `grade_hand_with_escalation`; call it from `process_batch`; thread a `depth_escalated` flag into `build_hand_rows` + display)
- Test: `scripts/regression_tests/test_live_flow.py`

**Interfaces:**
- Consumes: `grade_hand(hand)` (existing), `gto_api.AVAILABLE_DEPTHS`, `gto_api.nearest_depth`.
- Produces:
  - `def _next_depth_up(effective_bb: float) -> float | None` — next AVAILABLE_DEPTHS integer strictly greater than the base bracket, else None.
  - `def grade_hand_with_escalation(hand: dict) -> tuple[dict, set]` — `(devmap, escalated_keys)` where `escalated_keys` is a set of `(street, idx)` that were rescued at the higher depth.
  - `build_hand_rows(hand, hand_id, played_at, raw_text, devmap, escalated_keys=frozenset())` — new optional arg; adds `depth_escalated:{d}` to `approx_flags` for those keys and a `_depth_escalated` display value.

- [ ] **Step 1: Write the failing test**

Add to `scripts/regression_tests/test_live_flow.py`:

```python
from live_flow import _next_depth_up


@test
def next_depth_up_15():
    assert_eq(_next_depth_up(15.0), 17.0)


@test
def next_depth_up_top():
    assert_true(_next_depth_up(100.0) is None, "no bracket above 100")
```

(import `assert_eq` from harness at top of file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/regression_test.py 2>&1 | grep -i "depth bracket"`
Expected: FAIL (`_next_depth_up` not defined).

- [ ] **Step 3: Implement `_next_depth_up` and escalation**

Add to `scripts/live_flow.py` near `grade_hand`:

```python
def _next_depth_up(effective_bb: float) -> float | None:
    """Next AVAILABLE_DEPTHS integer strictly above the base bracket, else None."""
    from gto_api import AVAILABLE_DEPTHS, nearest_depth
    base = int(nearest_depth(effective_bb))          # e.g. 15 -> 14
    higher = [d for d in AVAILABLE_DEPTHS if d > base]
    return float(min(higher)) if higher else None


def grade_hand_with_escalation(hand: dict) -> tuple[dict, set]:
    """Grade at the hand's depth; for any node the solver returns offrange,
    re-grade once at the next depth bracket up and adopt only those nodes.

    Returns (devmap, escalated_keys). escalated_keys are (street, idx) tuples
    rescued at the higher depth — the caller flags them depth_escalated (§5.2).
    """
    base = grade_hand(hand)
    offrange = {k for k, d in base.items()
                if d.get("ungraded") and d.get("reason") == "offrange"}
    if not offrange:
        return base, set()
    up = _next_depth_up(float(hand.get("effective_bb") or 0))
    if up is None:
        return base, set()
    h2 = {**hand, "effective_bb": up}
    try:
        esc = grade_hand(h2)
    except Exception:
        return base, set()
    rescued: set = set()
    for k in offrange:
        d2 = esc.get(k)
        if d2 is not None and not d2.get("ungraded"):
            base[k] = d2
            rescued.add(k)
    return base, rescued
```

- [ ] **Step 4: Thread escalation into build_hand_rows + display**

In `scripts/live_flow.py`:

Change `build_hand_rows` signature:
```python
def build_hand_rows(hand: dict, hand_id: str, played_at: datetime,
                    raw_text: str, devmap: dict,
                    escalated_keys=frozenset()) -> tuple[dict, list[dict]]:
```
Inside its `for spot in walk_spots_from_parsed(hand):` loop, after `flags = [...]`, add:
```python
        if (spot["street"], spot["decision_idx"]) in escalated_keys:
            flags.append(f"depth_escalated:{int(_next_depth_up(float(hand.get('effective_bb') or 0)) or 0)}")
```

In `process_batch`, replace `devmap = grade_hand(hand)` with:
```python
            devmap, escalated_keys = grade_hand_with_escalation(hand)
```
and pass `escalated_keys` to `build_hand_rows(...)`.

In the per-decision `disp` dict built in `process_batch`, add:
```python
                    "depth_escalated": next(
                        (int(f.split(":", 1)[1]) for f in d["approx_flags"]
                         if f.startswith("depth_escalated:")), None),
```
(where `d` is the dec_row; compute before `d.pop` of display extras — the flag lives in `d["approx_flags"]`).

- [ ] **Step 5: Run tests**

Run: `python scripts/regression_test.py 2>&1 | grep -i "depth"`
Expected: PASS for both bracket tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/live_flow.py scripts/regression_tests/test_live_flow.py
git commit -m "feat: escalate offrange live nodes one depth bracket, flag depth_escalated"
```

---

### Task 4: Paginated renderer + per-hand description + 🔧 marker

**Files:**
- Modify: `scripts/live_flow.py` (add `PER_PAGE`, `_pot_type_zh`, `_hand_desc_line`, `render_session_page`; keep `render_tg_html` as a thin `render_session_page(result, 0)[0]` shim for back-compat callers/tests)
- Test: `scripts/regression_tests/test_live_flow.py`

**Interfaces:**
- Consumes: `result` dict from `process_batch`; each hand entry has `idx, ok, echo, repairs, review_url, decisions[], hand_row{hero_hand,position,preflop_depth_bb,pot_type}, error, refusal, validation_hard`.
- Produces:
  - `PER_PAGE = 10`
  - `def render_session_page(result: dict, page: int = 0, per_page: int = PER_PAGE) -> tuple[str, bool, bool]` → `(html, has_prev, has_next)`
  - `def _hand_desc_line(h: dict) -> str` (one-line description incl. severity + 🔧)

- [ ] **Step 1: Write the failing tests**

Add to `scripts/regression_tests/test_live_flow.py`:

```python
from live_flow import render_session_page, PER_PAGE


def _mk_hand(idx, sev="✅", repaired=False, failed=False):
    if failed:
        return {"idx": idx, "ok": False, "error": "validation_failed",
                "refusal": [], "validation_hard": ["這條線不能重播成合法牌局"],
                "raw": "Eff 35bb ...", "decisions": [], "repairs": []}
    ev = {"✅": None, "⚠️": 0.15, "❌": 0.5}[sev]
    decs = ([] if ev is None else [{
        "street": "flop", "idx": 0, "leaf": "l", "ev_loss": ev,
        "severity": sev, "taken": "C", "best": "F", "taken_label": "Call",
        "best_label": "Fold", "gto_freq": 1.0, "ungraded_reason": None,
        "discarded": False, "limp_origin": False, "depth_escalated": None}])
    return {"idx": idx, "ok": True, "hand_id": f"live:x:{idx}",
            "echo": "CO A7s 30bb · ...", "repairs": (["x"] if repaired else []),
            "review_url": "https://app.gtowizard.com/solutions?x", "decisions": decs,
            "hand_row": {"hero_hand": "A7s", "position": "CO",
                         "preflop_depth_bb": 30.0, "pot_type": "single_raised"}}


def _mk_result(n):
    hands = [_mk_hand(i + 1) for i in range(n)]
    return {"totals": {"hands": n, "decisions": n, "graded": n, "mistakes": 0,
                       "parse_failed": 0}, "queue": [], "hands": hands}


@test
def page_split():
    result = _mk_result(23)
    html0, prev0, next0 = render_session_page(result, 0)
    assert_true(not prev0 and next0, "page0 has next, no prev")
    assert_in("(第 1/3 頁)", html0)
    _h1, prev1, next1 = render_session_page(result, 1)
    assert_true(prev1 and next1, "middle page has both")
    _h2, prev2, next2 = render_session_page(result, 2)
    assert_true(prev2 and not next2, "last page no next")


@test
def no_rollup_no_bulk():
    result = _mk_result(2)
    result["hands"][0]["repairs"] = ["HU pot 動作歸屬修補"]
    html, _p, _n = render_session_page(result, 0)
    assert_true("無明顯偏差：" not in html, "roll-up list removed")
    assert_true("已自動校正後送 solver" not in html, "bulk repair section removed")
    assert_in("🔧", html)  # per-hand marker present instead


@test
def clean_hand_line():
    html, _p, _n = render_session_page(_mk_result(1), 0)
    assert_in("Hand 1", html)
    assert_in("✅", html)


@test
def live_render_terminology():
    result = _mk_result(1)
    result["hands"][0] = _mk_hand(1, sev="❌")
    result["totals"]["mistakes"] = 1
    html, _p, _n = render_session_page(result, 0)
    assert_in("建議", html)
    assert_true("主線" not in html, "must not contain 主線")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python scripts/regression_test.py 2>&1 | grep -i "page\|rollup\|clean hand\|terminology"`
Expected: FAIL (`render_session_page` not defined).

- [ ] **Step 3: Implement the renderer**

In `scripts/live_flow.py`, add after `severity` / before `report_buttons`:

```python
PER_PAGE = 10

_POT_TYPE_ZH = {
    "single_raised": "單加注池", "srp": "單加注池", "limped": "跛入池",
    "3bet": "3bet 池", "4bet": "4bet 池", "5bet": "5bet 池",
    "squeezed": "擠壓池", "cold4bet": "cold 4bet 池",
}


def _pot_type_zh(pot_type: str | None) -> str:
    return _POT_TYPE_ZH.get(str(pot_type or "").lower(), str(pot_type or ""))


def _hand_severity(h: dict) -> str:
    sevs = [d["severity"] for d in h.get("decisions") or []
            if not d.get("discarded")]
    if "❌" in sevs:
        return "❌"
    if "⚠️" in sevs:
        return "⚠️"
    if any(d.get("ungraded_reason") for d in h.get("decisions") or []):
        return "❓"
    return "✅"


def _hand_desc_line(h: dict) -> str:
    if not h.get("ok"):
        title, _help = _failure_help(h)
        return f"<b>Hand {h['idx']}</b> · ❗ 無法評分：{escape(title)}"
    row = h.get("hand_row") or {}
    hand = cards_to_emoji(row.get("hero_hand") or "")
    pos = row.get("position") or ""
    depth = row.get("preflop_depth_bb")
    depth_s = f"{depth:g}bb" if depth else ""
    pot = _pot_type_zh(row.get("pot_type"))
    sev = _hand_severity(h)
    wrench = " 🔧" if h.get("repairs") else ""
    bits = [f"<b>Hand {h['idx']}</b>", f"{pos} {hand}".strip(), depth_s, pot, sev]
    return " · ".join(b for b in bits if b) + wrench


def render_session_page(result: dict, page: int = 0,
                        per_page: int = PER_PAGE) -> tuple[str, bool, bool]:
    t = result["totals"]
    hands = result["hands"]
    pages = max(1, (len(hands) + per_page - 1) // per_page)
    page = max(0, min(page, pages - 1))
    lo, hi = page * per_page, page * per_page + per_page
    n_offrange = sum(
        1 for h in hands if h.get("ok")
        and any(d.get("ungraded_reason") for d in h["decisions"])
        and not any(d["ev_loss"] is not None and d["ev_loss"] >= QUEUE_EV_MIN
                    and not d["discarded"] for d in h["decisions"]))
    L = [f"🃏 <b>線下入帳：{t['hands']} 手 / {t['decisions']} 決策</b>　(第 {page+1}/{pages} 頁)"]
    L.append(f"⚠️❌ {t['mistakes']} 偏差 · ❓ {n_offrange} 待深挖 · ✅ 其餘無明顯偏差")
    L.append("")

    for h in hands[lo:hi]:
        L.append(_hand_desc_line(h))
        if not h.get("ok"):
            _title, help_text = _failure_help(h)
            L.append(f"　{escape(help_text)}")
            L.append("")
            continue
        for d in h["decisions"]:
            if d["ev_loss"] is None or d["ev_loss"] < QUEUE_EV_MIN or d["discarded"]:
                continue
            best = d["best_label"] or d["best"] or "?"
            freq = f"（{d['gto_freq']*100:.0f}%）" if d.get("gto_freq") else ""
            approx = f"（於 {d['depth_escalated']}bb 近似）" if d.get("depth_escalated") else ""
            L.append(f"　{d['severity']} {d['street']} "
                     f"{escape(d['taken_label'] or d['taken'] or '?')} → "
                     f"建議 {escape(str(best))}{freq} · 損失 {d['ev_loss']:.2f}bb{approx}")
        offrange = [d for d in h["decisions"] if d.get("ungraded_reason") == "offrange"]
        if offrange:
            first = offrange[0]
            L.append(f"　❓ {first['street']} 起未評分：偏離 GTO 建議後，"
                     f"你的牌已在該線範圍外")
            if _next_depth_up(float((h.get('hand_row') or {}).get('preflop_depth_bb') or 0)):
                L.append("　（已嘗試升一格近似，仍無範圍）")
        L.append("")

    if result.get("queue"):
        L.append(f"📥 已加入練習佇列 {len(result['queue'])} 條行動線（/queue 查看）")
    L.append("⚠️ chipEV 近似（現場賽段未知）；limp 節點不評分。要更正某手：點該手的 🔁 重傳。")
    return "\n".join(L), page > 0, page < pages - 1


def render_tg_html(result: dict) -> str:
    """Back-compat shim: first page only."""
    return render_session_page(result, 0)[0]
```

Delete the old `render_tg_html` body (lines ~1514-1585) — the shim above replaces it. Keep `_failure_help` and `_repair_explanation` (still used).

- [ ] **Step 4: Run tests**

Run: `python scripts/regression_test.py 2>&1 | grep -i "page\|rollup\|clean hand\|建議"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/live_flow.py scripts/regression_tests/test_live_flow.py
git commit -m "feat: paginated per-hand live report, drop rollup + bulk repair section"
```

---

### Task 5: Per-hand 復盤 URL + button builder

**Files:**
- Modify: `scripts/live_flow.py` (compute `entry["review_url"]` in `process_batch`; rewrite `report_buttons` → `session_page_buttons(result, session_id, page, per_page)`)
- Test: `scripts/regression_tests/test_live_flow.py`

**Interfaces:**
- Consumes: `gtow_solution_url.build_last_hero_hand_url(hand, decisions)` (returns `str | None`).
- Produces:
  - each ok hand entry gains `review_url: str | None`
  - `def session_page_buttons(result: dict, session_id: int, page: int, per_page: int = PER_PAGE) -> list[list[dict]]` — per-hand row `[復盤(url)?, 💬教練(lvd), ➕加練(lvadd), 🔁重傳(lvr)]`; failed hands `[🔁重傳]`; last row nav `[◀/▶]` as `lvpg:<sid>:<page±1>`.

- [ ] **Step 1: Write the failing test**

Add to `scripts/regression_tests/test_live_flow.py`:

```python
from live_flow import session_page_buttons


@test
def per_hand_buttons():
    result = _mk_result(12)                       # 2 pages
    result["hands"][1]["ok"] = False
    result["hands"][1]["error"] = "validation_failed"
    rows = session_page_buttons(result, session_id=7, page=0)
    flat = [b for row in rows for b in row]
    texts = [b["text"] for b in flat]
    assert_true(any("復盤" in x for x in texts), "復盤 present")
    assert_true(any("加練" in x for x in texts), "加練 present")
    assert_true(any(b.get("callback_data", "").startswith("lvr:7:")
                    for b in flat), "resend callback present")
    assert_true(any(b.get("callback_data", "").startswith("lvpg:7:1")
                    for b in flat), "next-page nav present")
    # failed hand (idx 2) exposes only a resend button, no 復盤/教練/加練
    assert_true(any(b.get("callback_data") == "lvr:7:1" for b in flat),
                "failed hand resend by index")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/regression_test.py 2>&1 | grep -i "per_hand_buttons\|復盤"`
Expected: FAIL (`session_page_buttons` not defined).

- [ ] **Step 3: Compute review_url in process_batch**

In `scripts/live_flow.py`, inside `process_batch`, after `all_dec_rows.extend(d_rows)` (hand graded ok), add:

```python
        try:
            from gtow_solution_url import build_last_hero_hand_url
            entry["review_url"] = build_last_hero_hand_url(
                hand, [d for d in d_rows if not d.get("excluded")])
        except Exception:
            entry["review_url"] = None
```

Ensure the slim JSON in `main()` keeps `review_url` (it already keeps all keys except `dec_rows`/`hand_row`; `hand_row` is popped — but `_hand_desc_line`/render needs `hand_row`). **Change `main()`'s slim loop to NOT pop `hand_row`** (render needs `hero_hand/position/preflop_depth_bb/pot_type`); keep popping only `dec_rows`:

```python
        for h in slim["hands"]:
            h.pop("dec_rows", None)
```

- [ ] **Step 4: Implement session_page_buttons**

Replace `report_buttons` in `scripts/live_flow.py` with:

```python
def session_page_buttons(result: dict, session_id: int, page: int,
                         per_page: int = PER_PAGE) -> list[list[dict]]:
    """Per-hand button rows for one page + prev/next nav."""
    hands = result["hands"]
    pages = max(1, (len(hands) + per_page - 1) // per_page)
    page = max(0, min(page, pages - 1))
    rows: list[list[dict]] = []
    for h in hands[page * per_page: page * per_page + per_page]:
        idx0 = h["idx"] - 1
        if not h.get("ok"):
            rows.append([{"text": "🔁 重傳",
                          "callback_data": f"lvr:{session_id}:{idx0}"}])
            continue
        row: list[dict] = []
        if h.get("review_url"):
            row.append({"text": f"復盤 H{h['idx']}", "url": h["review_url"]})
        row.append({"text": "💬 教練", "callback_data": f"lvd:{h['hand_id']}"})
        row.append({"text": "➕ 加練", "callback_data": f"lvadd:{session_id}:{idx0}"})
        row.append({"text": "🔁 重傳", "callback_data": f"lvr:{session_id}:{idx0}"})
        rows.append(row)
    nav: list[dict] = []
    if page > 0:
        nav.append({"text": "◀ 上一頁", "callback_data": f"lvpg:{session_id}:{page-1}"})
    if page < pages - 1:
        nav.append({"text": "下一頁 ▶", "callback_data": f"lvpg:{session_id}:{page+1}"})
    if nav:
        rows.append(nav)
    return rows
```

- [ ] **Step 5: Run tests**

Run: `python scripts/regression_test.py 2>&1 | grep -i "per_hand_buttons"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/live_flow.py scripts/regression_tests/test_live_flow.py
git commit -m "feat: per-hand 復盤 URL + paginated session button rows"
```

---

### Task 6: Bot wiring — persist session, send page, lvpg pagination

**Files:**
- Modify: `src/telegram_bot/bot.py` (`_process_live_batch`; add `_send_or_edit_session_page`, `lvpg:` branch in `handle_live_button`; register nothing new in the pattern — `lvpg|lvadd|lvr` add to the existing CallbackQueryHandler pattern at ~line 2554)
- Test: manual (integration); no unit harness for the TG layer

**Interfaces:**
- Consumes: `live_flow.render_session_page`, `session_page_buttons`, `save_session`, `set_session_message`, `load_session`, `hand_id_for`, `split_batch`.
- Produces: `async def _send_or_edit_session_page(self, context, chat_id, session, page, *, edit_message_id=None)`.

- [ ] **Step 1: Extend the callback pattern**

In `src/telegram_bot/bot.py` at the CallbackQueryHandler registration (~line 2554), add `lvpg|lvadd|lvr` to the alternation:

```python
                pattern=r"^(lvd|lvpg|lvadd|lvr|qcl|qpg|qex|qad|qad2|qsrc|qraw|qdet|qdst|srd|srv|srd2|srv2):"))
```

- [ ] **Step 2: Persist the session + send page 0 in `_process_live_batch`**

Replace the result-send block (currently `html = render_tg_html(result)` … `await update.message.reply_text(html …)`) with:

```python
            result = _json.loads(Path(tmp_out).read_text())
            from live_flow import (render_session_page, session_page_buttons,
                                    save_session, set_session_message,
                                    hand_id_for)
            date_str = result.get("date")
            session_key = f"live:{date_str}:{hashlib.sha1(text.strip().encode()).hexdigest()[:10]}"
            async with self.db.pool.acquire() as conn:
                session_id = await save_session(conn, session_key,
                                                update.effective_chat.id, result)
            html, _prev, _next = render_session_page(result, 0)
            markup = self._rows_to_markup(session_page_buttons(result, session_id, 0))
            try:
                await msg.delete()
            except Exception:
                pass
            sent = await update.message.reply_text(
                html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
            async with self.db.pool.acquire() as conn:
                await set_session_message(conn, session_id, sent.message_id)
            self.log.info(f"[{label}] /live done: {result['totals']}")
```

Add `import hashlib` at the top of the method (or module) if not present.

- [ ] **Step 3: Add the shared page render/edit helper**

Add to the bot class near `_fetch_queue_page`:

```python
    async def _send_or_edit_session_page(self, query, session: dict, page: int):
        """Re-render one session page and edit the report message in place."""
        from live_flow import (render_session_page, session_page_buttons,
                               update_session_result)
        result = session["result"]
        html, _prev, _next = render_session_page(result, page)
        markup = self._rows_to_markup(
            session_page_buttons(result, session["id"], page))
        async with self.db.pool.acquire() as conn:
            await update_session_result(conn, session["id"], result, page)
        try:
            await query.edit_message_text(
                html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
        except telegram.error.BadRequest as exc:
            if "Message is not modified" not in str(exc):
                raise
```

- [ ] **Step 4: Add the lvpg branch in handle_live_button**

In `handle_live_button`, before the `lvd:` fallthrough, add:

```python
        if data.startswith("lvpg:"):
            from live_flow import load_session
            _, sid, page = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            await self._send_or_edit_session_page(query, session, int(page))
            return
```

- [ ] **Step 5: Manual integration test**

Deploy locally / run the bot; `/live` a batch of >10 hands; verify: page header `(第 1/2 頁)`, per-hand lines, four buttons per hand, `下一頁 ▶` flips in place. Tap 復盤 → GTOW strategy page. Tap 💬 教練 → existing deep-dive fires.

- [ ] **Step 6: Commit**

```bash
git add src/telegram_bot/bot.py
git commit -m "feat: persist live session + paginated report with in-place page flips"
```

---

### Task 7: ➕ 加練 — add a live decision to the queue (lvadd)

**Files:**
- Modify: `src/telegram_bot/bot.py` (add `lvadd:` branch → `_live_add_menu`)
- Test: manual (reuses covered `qad2` path)

**Interfaces:**
- Consumes: session `result` (hand_id per idx), `ledger_decisions` rows (source='live'), `queue_feed.qex_submenu`, existing `qad2` handler.
- Produces: `async def _live_add_menu(self, context, chat_id, session, hand_idx)`.

- [ ] **Step 1: Add the lvadd branch**

In `handle_live_button`, before the `lvd:` fallthrough:

```python
        if data.startswith("lvadd:"):
            from live_flow import load_session
            _, sid, hand_idx = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            await self._live_add_menu(context, chat_id, session, int(hand_idx))
            return
```

- [ ] **Step 2: Implement _live_add_menu**

Add to the bot class (mirrors `_queue_expand_review`, sentinel queue_id=0):

```python
    async def _live_add_menu(self, context, chat_id, session, hand_idx):
        """Expand a live hand's graded decisions as ➕ manual-add buttons."""
        from html import escape as _esc
        from queue_feed import qex_submenu
        hands = session["result"]["hands"]
        if hand_idx >= len(hands) or not hands[hand_idx].get("ok"):
            await context.bot.send_message(chat_id, "這手沒有可加練的決策。")
            return
        hand_id = hands[hand_idx]["hand_id"]
        rows = await self.db.pool.fetch(
            "SELECT id, gtow_hand_id, street, decision_idx, spot_category, "
            "spot_leaf, hero_cat, villain_cat, ip_oop, position, ev_loss_bb "
            "FROM ledger_decisions "
            "WHERE gtow_hand_id=$1 AND source='live' AND NOT excluded "
            "AND NOT discarded "
            "ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 "
            "WHEN 'turn' THEN 2 WHEN 'river' THEN 3 ELSE 9 END, decision_idx",
            hand_id)
        if not rows:
            await context.bot.send_message(
                chat_id, "這手沒有可加練的已評分決策。")
            return
        btn_rows = qex_submenu([dict(r) for r in rows], queue_id=0)
        await context.bot.send_message(
            chat_id,
            f"➕ <b>選一條 action line 加入練習</b>\nHand {hand_idx + 1}",
            parse_mode="HTML",
            reply_markup=self._rows_to_markup([[b] for b in btn_rows]))
```

- [ ] **Step 3: Verify qex_submenu emits qad2 callbacks**

Run: `grep -n "qad2\|def qex_submenu" scripts/queue_feed.py`
Expected: `qex_submenu` builds `qad2:<queue_id>:<hand>:<street>:<idx>` — confirms the existing `qad2` handler (already wired) will add the picked decision. No new add code needed.

- [ ] **Step 4: Manual test**

`/live` a batch; tap ➕ 加練 on a clean hand; pick a decision; confirm `➕ 已加入練習：…` and it shows in `/queue`.

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/bot.py
git commit -m "feat: add-to-queue from a live hand via existing qad2 path"
```

---

### Task 8: 🔁 重傳 — single-hand resend + in-place overwrite

**Files:**
- Modify: `src/telegram_bot/bot.py` (`lvr:` branch, resend-pending state, message interception, `_apply_live_resend`)
- Modify: `scripts/live_flow.py` (add `async def overwrite_hand`, reuse `process_batch`, `write_hand`, `select_queue_items`, `enqueue`)
- Modify: `scripts/queue_feed.py` (add `async def remove_source_hand`)
- Test: `scripts/regression_tests/test_live_flow.py` (unit for remove_source_hand recompute is DB-bound → smoke via `_tmp.py`; unit-test the result-splice helper)

**Interfaces:**
- Produces:
  - `queue_feed.remove_source_hand(conn, hand_id: str) -> None` — strip `hand_id` entries from every open `drill_queue.source_hands`, recompute `total_ev_loss_bb`, and set rows that become empty (auto/live drill) to `status='cleared'`.
  - `live_flow.splice_hand(result: dict, hand_idx: int, new_entry: dict) -> dict` — replace hand at idx, keep display idx, recompute `totals` + `queue`.
  - bot `_apply_live_resend(self, update, context, session_id, hand_idx, block)`.

- [ ] **Step 1: Write the failing unit test for splice_hand**

Add to `scripts/regression_tests/test_live_flow.py`:

```python
from live_flow import splice_hand


@test
def splice_recompute():
    result = _mk_result(3)
    new_entry = _mk_hand(2, sev="❌")
    new_entry["dec_rows"] = []          # display path only in this unit
    out = splice_hand(result, 1, new_entry)
    assert_eq(out["hands"][1]["idx"], 2)                 # display idx preserved
    assert_eq(out["hands"][1]["decisions"][0]["severity"], "❌")
    assert_eq(out["totals"]["mistakes"], 1)              # the new ❌ counted
```

- [ ] **Step 2: Run to verify fail**

Run: `python scripts/regression_test.py 2>&1 | grep -i splice`
Expected: FAIL (`splice_hand` not defined).

- [ ] **Step 3: Implement splice_hand**

In `scripts/live_flow.py`:

```python
def _recompute_totals(hands: list[dict]) -> dict:
    decisions = graded = mistakes = parse_failed = 0
    for h in hands:
        if not h.get("ok"):
            parse_failed += 1
            continue
        for d in h["decisions"]:
            decisions += 1
            if d["ev_loss"] is not None:
                graded += 1
            if (d["ev_loss"] is not None and d["ev_loss"] >= QUEUE_EV_MIN
                    and not d["discarded"]):
                mistakes += 1
    return {"hands": len(hands), "decisions": decisions, "graded": graded,
            "mistakes": mistakes, "parse_failed": parse_failed}


def splice_hand(result: dict, hand_idx: int, new_entry: dict) -> dict:
    """Replace hands[hand_idx] with new_entry (idx preserved) and recompute
    totals + queue from all current dec_rows."""
    new_entry = dict(new_entry)
    new_entry["idx"] = result["hands"][hand_idx]["idx"]
    result["hands"][hand_idx] = new_entry
    result["totals"] = _recompute_totals(result["hands"])
    all_dec_rows: list[dict] = []
    for h in result["hands"]:
        if h.get("ok"):
            all_dec_rows.extend(h.get("dec_rows") or [])
    result["queue"] = select_queue_items(all_dec_rows)
    return result
```

Note: for `splice_hand`/queue recompute to work post-persist, the session `result_json` MUST retain `dec_rows` per ok hand. In `_process_live_batch` (Task 6) `save_session` stores the **full** `result` read from `tmp_out` — but `main()`'s slim JSON drops `dec_rows`. Fix: in `live_flow.main()`, write TWO artifacts is overkill; instead **keep `dec_rows` in the json-out** and let the bot store it. Change `main()`'s slim loop to keep `dec_rows` (drop only `hand_row` which duplicates data the render needs — but render needs hand_row too). Simplest: keep both. Replace the slim loop with:

```python
        slim = json.loads(json.dumps(result, default=str))
        Path(a.json_out).write_text(json.dumps(slim, ensure_ascii=False, default=str))
```

(dec_rows carry `played_at` datetimes → `default=str` handles them.)

- [ ] **Step 4: Run splice test**

Run: `python scripts/regression_test.py 2>&1 | grep -i splice`
Expected: PASS.

- [ ] **Step 5: Implement remove_source_hand in queue_feed**

Read `scripts/queue_feed.py` around `enqueue_one` to match the `source_hands`/`total_ev_loss_bb` column shape, then add:

```python
async def remove_source_hand(conn, hand_id: str) -> None:
    """Strip a hand's contributions from every open drill queue row.

    For each pending/prescribed row whose source_hands includes hand_id:
    drop those entries, recompute total_ev_loss_bb, and if an auto/live drill
    row is left with no sources, mark it cleared (clear_reason='resend')."""
    rows = await conn.fetch(
        "SELECT id, source_hands, added_by, kind FROM drill_queue "
        "WHERE status IN ('pending','prescribed') "
        "AND source_hands::text LIKE '%' || $1 || '%'", hand_id)
    for r in rows:
        srcs = json.loads(r["source_hands"]) if isinstance(r["source_hands"], str) \
            else (r["source_hands"] or [])
        kept = [s for s in srcs if s.get("hand_id") != hand_id]
        if len(kept) == len(srcs):
            continue
        if not kept and r["kind"] == "drill" and r["added_by"] in ("auto", "live"):
            await conn.execute(
                "UPDATE drill_queue SET status='cleared', cleared_at=NOW(), "
                "clear_reason='resend' WHERE id=$1", r["id"])
            continue
        total = round(sum(float(s.get("ev_loss_bb") or 0) for s in kept), 4)
        await conn.execute(
            "UPDATE drill_queue SET source_hands=$2, total_ev_loss_bb=$3, "
            "n_sources=$4 WHERE id=$1",
            r["id"], json.dumps(kept), total, len(kept))
```

(Confirm column names `source_hands`, `total_ev_loss_bb`, `n_sources`, `clear_reason` exist — they are used elsewhere in queue_feed/bot; adjust if the real schema differs.)

- [ ] **Step 6: Implement overwrite_hand orchestration in live_flow**

```python
async def overwrite_hand(conn, session: dict, hand_idx: int,
                         block: str) -> dict:
    """Reparse+grade a single corrected hand, overwrite its ledger rows,
    recompute the session queue, and return the updated result dict."""
    result = session["result"]
    old = result["hands"][hand_idx]
    old_hand_id = old.get("hand_id")
    date_str = result.get("date")
    single = process_batch(block, date_str)          # 1-hand batch
    new_entry = single["hands"][0] if single["hands"] else {
        "idx": old["idx"], "ok": False, "error": "parse_failed",
        "refusal": ["空白或無法辨識"], "decisions": [], "repairs": [],
        "raw": block}
    # tear down the old hand's ledger + queue footprint
    if old_hand_id:
        await conn.execute("DELETE FROM ledger_decisions WHERE gtow_hand_id=$1",
                           old_hand_id)
        await conn.execute("DELETE FROM ledger_hands WHERE gtow_hand_id=$1",
                           old_hand_id)
        await remove_source_hand(conn, old_hand_id)
    # persist the new hand
    if new_entry.get("ok"):
        await write_hand(conn, new_entry["hand_row"], new_entry["dec_rows"])
    result = splice_hand(result, hand_idx, new_entry)
    await enqueue(conn, result["queue"])
    for item in result["queue"]:
        item["queue_id"] = await conn.fetchval(
            "SELECT id FROM drill_queue WHERE spot_leaf=$1 AND kind='drill' "
            "AND status IN ('pending','prescribed') "
            "ORDER BY (status='pending') DESC, last_added DESC LIMIT 1",
            item["spot_leaf"])
    return result
```

- [ ] **Step 7: Bot — lvr branch + pending state + interception + apply**

In `handle_live_button`, before the `lvd:` fallthrough:

```python
        if data.startswith("lvr:"):
            from live_flow import load_session
            _, sid, hand_idx = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            h = session["result"]["hands"][int(hand_idx)]
            self._live_resend_pending[chat_id] = (int(sid), int(hand_idx))
            reps = "；".join(_repair_explanation(str(r)) for r in (h.get("repairs") or [])) or "無"
            from html import escape as _esc
            await context.bot.send_message(
                chat_id,
                f"請貼上 <b>Hand {int(hand_idx)+1}</b> 的單手更正版本"
                f"（Header / Flop / Turn / River 各一行）。\n"
                f"目前 echo：{_esc(h.get('echo') or '（無法評分）')}\n"
                f"目前校正：{_esc(reps)}",
                parse_mode="HTML")
            return
```

Import `_repair_explanation` from live_flow at the top of the branch or module.

Initialize `self._live_resend_pending = {}` where `self._live_pending` is created.

In the message handler where `_live_pending` is checked (~line 698), add a prior check:

```python
        if update.effective_chat.id in self._live_resend_pending:
            sid, hand_idx = self._live_resend_pending.pop(update.effective_chat.id)
            async with self._user_lock(update.effective_chat.id):
                await self._apply_live_resend(update, context, sid, hand_idx,
                                              update.message.text or "")
            return
```

Add the apply method:

```python
    async def _apply_live_resend(self, update, context, session_id, hand_idx,
                                 block):
        from live_flow import (load_session, overwrite_hand, render_session_page,
                               session_page_buttons, update_session_result)
        msg = await update.message.reply_text("🔁 重新解析並覆蓋中…")
        # run parse+grade off the event loop (blocking solver calls)
        async with self.db.pool.acquire() as conn:
            session = await load_session(conn, session_id)
            if not session:
                await msg.edit_text("這個線下 session 已過期，請重跑 /live。")
                return
            result = await overwrite_hand(conn, session, hand_idx, block)
            page = hand_idx // __import__("live_flow").PER_PAGE
            await update_session_result(conn, session_id, result, page)
        html, _p, _n = render_session_page(result, page)
        markup = self._rows_to_markup(
            session_page_buttons(result, session_id, page))
        await msg.delete()
        if session.get("message_id"):
            try:
                await context.bot.edit_message_text(
                    html, chat_id=session["chat_id"],
                    message_id=session["message_id"], parse_mode="HTML",
                    disable_web_page_preview=True, reply_markup=markup)
                await update.message.reply_text(f"✅ Hand {hand_idx+1} 已更新。")
                return
            except Exception:
                pass
        await update.message.reply_text(
            html, parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=markup)
```

Note: `overwrite_hand` calls `process_batch` (blocking Gemini + solver). For a single hand this is a few seconds; acceptable inline. If latency is a problem, wrap in `asyncio.to_thread` in a follow-up — not required for v1.

- [ ] **Step 8: Smoke-test remove_source_hand + overwrite via _tmp.py**

Write `scripts/_tmp.py` to: `save_session` a fake 2-hand result whose dec_rows include a queued deviation, `enqueue` it, run `overwrite_hand` replacing the deviating hand with a clean one, assert the old hand_id no longer appears in `drill_queue.source_hands`. Run: `python scripts/_tmp.py` → `OK`.

- [ ] **Step 9: Manual test**

`/live` a batch including a wrong hand; tap 🔁 重傳; paste the corrected single hand; confirm the report edits in place with the corrected line, and `/queue` no longer double-counts.

- [ ] **Step 10: Commit**

```bash
git add scripts/live_flow.py scripts/queue_feed.py src/telegram_bot/bot.py \
        scripts/regression_tests/test_live_flow.py
git commit -m "feat: single-hand resend overwrites ledger + queue in place"
```

---

### Task 9: Fix Hand 2 `b4` = small-blind bet parse bug

**Files:**
- Modify: whichever layer drops the bet (determined by repro — likely `scripts/live_flow.py` action-hint/`repair_hu_pot`, or the Gemini `LIVE_HINT`)
- Test: `scripts/regression_tests/test_live_flow.py`

**Interfaces:** no signature change; parse correctness only.

- [ ] **Step 1: Reproduce with the real block (systematic-debugging)**

Write `scripts/_tmp.py`:

```python
from live_flow import parse_block, repair_hu_pot, find_ghost
from hand_validator import validate_hand

block = ("Eff 35bb Co raise hero btn call 7s8s sb raise 7bb co fold hero call\n"
         "Ac5c6d b4 call\n"
         "4s x b8 fold")
h = parse_block(block)
print("PARSED:", h)
h = repair_hu_pot(h)
print("GHOST:", find_ghost(h))
print("STREETS:", h.get("streets"))
rep = validate_hand(h)
print("VALID:", rep.ok, [i.message for i in rep.hard])
```

Run: `python scripts/_tmp.py`
Expected: reproduces the "SB Call with no bettor" (orphan call) — confirm which street/action shows `SB Call` where `SB bet 4` belongs.

- [ ] **Step 2: Identify root cause**

Inspect the printed `streets`: determine whether (a) Gemini emitted `SB call` for the flop (parse/structure), (b) `repair_hu_pot`'s HU alternation reassigned the bettor into a caller, or (c) the SB (3bettor/last aggressor) was mis-seated so the flop bettor became a folded seat and got stripped. Cross-check `_extract_street_action_hints(block)` → the flop hint should be `["R4","C"]`. If hints are correct but parsed streets are not, the fix belongs in `repair_street_actions_from_block` (it only inserts a leading check today; extend to reconcile a dropped **bet** owner in a HU street when hint classes are `[R, C]` but parsed is `[C]`/`[X,?]`). Do NOT loosen anything that would mis-seat correct hands — gate the fix on HU + exact hint/parse class mismatch.

- [ ] **Step 3: Write the failing regression test**

```python
from live_flow import parse_block, repair_hu_pot
from hand_validator import validate_hand


@test
def flop_b4_is_bet():
    block = ("Eff 35bb Co raise hero btn call 7s8s sb raise 7bb co fold hero call\n"
             "Ac5c6d b4 call\n"
             "4s x b8 fold")
    h = parse_block(block)
    assert_true(h is not None and not h.get("_refused"), "parses")
    h = repair_hu_pot(h)
    rep = validate_hand(h)
    assert_true(rep.ok, f"must be legal: {[i.message for i in rep.hard]}")
    flop = next(s for s in h["streets"] if (s.get("board") or "").startswith("Ac5c6"))
    actions = [(a.get("position"), a.get("action")) for a in flop["actions"]]
    # first flop action is a bet (R…/B…), not a Call
    assert_true(actions and str(actions[0][1]).upper().startswith(("R", "B")),
                f"flop opens with a bet, got {actions}")
```

- [ ] **Step 4: Run to verify it fails**

Run: `python scripts/regression_test.py 2>&1 | grep -i "b4"`
Expected: FAIL (orphan call / illegal line).

- [ ] **Step 5: Implement the minimal fix**

Apply the gated repair identified in Step 2 (extend `repair_street_actions_from_block` for the `[R,C]`-hint vs dropped-bettor HU case, or fix the mis-seat feeding `repair_hu_pot`). Keep it HU-only + exact-mismatch-gated per [[live-flow-refuse-over-repair]].

- [ ] **Step 6: Run the regression test + full suite**

Run: `python scripts/regression_test.py`
Expected: the `b4` test PASSES and all others still pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/live_flow.py scripts/regression_tests/test_live_flow.py
git commit -m "fix: live flop 'b4' parsed as SB bet not orphan call (Hand 2)"
```

---

### Task 10: Full suite + PR

- [ ] **Step 1: Run the full regression suite**

Run: `python scripts/regression_test.py`
Expected: all pass (core-analysis files touched → mandatory per CLAUDE.md).

- [ ] **Step 2: Push + PR**

```bash
git push -u origin feat/live-message-redesign
gh pr create --title "feat: live flow paginated report + 復盤/加練/重傳 + depth escalation" \
  --body "$(cat <<'EOF'
Redesigns the /live report per docs/superpowers/specs/2026-07-24-live-flow-message-redesign-design.md:
- Per-hand paginated list (10/page) with 復盤 / 💬教練 / ➕加練 / 🔁重傳 buttons
- 主線 → 建議 terminology
- Off-range nodes auto-escalate one depth bracket (flagged depth_escalated)
- Removed the 無明顯偏差 roll-up + bulk 🔧 section (per-hand 🔧 marker + resend echo)
- live_sessions table persists the report for pagination/resend across restarts
- Single-hand 🔁 resend overwrites ledger + queue in place
- Fix: flop 'b4' parsed as SB bet not orphan call (Hand 2)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §1 版面 → Tasks 4,5,6. §2 術語 → Task 1. §3 升格 → Task 3. §4 🔧 → Tasks 4 (marker) + 8 (resend echo). §5 live_sessions → Task 2. §6 按鈕 → Task 5. §7 加練 → Task 7. §8 重傳 → Task 8. §9 Hand2 → Task 9. §10 測試 → per-task + Task 10. ✅ all covered.
- ❓ summary count (`n 待深挖`) — implemented in `render_session_page` (Task 4). ✅

**Placeholder scan:** Task 2/9 root-cause steps require reading real schema/repro (inherent to a debug task) but give the exact repro command, the decision criteria, and the gated fix location — not "handle edge cases". Column-name confirmations are flagged explicitly. No TBD/TODO copy.

**Type consistency:** `render_session_page(result, page, per_page) -> (str,bool,bool)` used identically in Tasks 4/6/8. `session_page_buttons(result, session_id, page, per_page)` consistent in 5/6/8. `load_session`→`{"id","chat_id","message_id","page","result"}` consistent across 6/7/8. `splice_hand`/`overwrite_hand`/`remove_source_hand` signatures match their call sites. `depth_escalated` display key set in Task 3, read in Task 4. ✅
