# Phase 1 Ledger（GTOW Analyze 全量帳本 + 最小教練迴圈）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把選手 2026-03-01 至今上傳 GTO Wizard Analyzer 的全部 MTT 手牌（backfill 當下 33,608 手）攝取為誠實、保真、可重算的 Decision Ledger，並讓最小教練迴圈轉起來：自動入帳 → EV 加權診斷 → 週記分卡 + 焦點處方（TG 推播 + HTML）→ 隔週回讀。

**Architecture:** 獨立攝取層（GTOW Analyze API client → 本地 raw 檔案庫 → 純函數蒸餾 → Supabase `ledger_*` 表）；診斷/記分卡是帳本上的純查詢層；TG bot 只做觸發與推播。`deviations` 表與現有 bot 流程零接觸。上游 spec：`docs/superpowers/specs/2026-07-07-phase1-ledger-design.md`（先讀完再動工）；憲法：`docs/NORTH_STAR.md`。

**Tech Stack:** Python 3.11+、requests（API）、asyncpg（Supabase）、python-telegram-bot v20+（現有 bot）、標準庫 HTML/SVG 渲染（零新依賴）。

## Global Constraints

- **開發流程**：全程在 worktree `~/ai-poker-wizard-phase1-ledger`（branch `feat/phase1-ledger`）工作；每個 task 至少一個 commit；最後發 PR。絕不直接改 main。
- **零新依賴**：只用 `requirements.txt` 已有的套件。
- **Migrations**：只用 `supabase db push`，永不 raw psql 改 schema。
- **`.tokens.json` 是 bind-mount**：只能 in-place 讀寫（`open(path, "r+")` + `truncate()`），永不整檔替換 inode。
- **API 禮貌**：對 `api.gtowizard.com` 全域節流 ~2.5 rps + jitter；429/5xx 指數退避；headers 模仿 SPA（`origin: https://app.gtowizard.com`）。
- **北極星不變量落地**：每筆決策帶 `approx_flags` 與 `excluded`（不變量 2）；一切排序用 EV 不用頻率（不變量 3）；raw JSON 永久落地本地（不變量 9）；所有統計輸出帶 n（§14.3）；降級必須可觀測（§14.2 大聲失敗）。
- **時區**：`played_at` 是 UTC；所有使用者可見的日/週分桶用 `Asia/Taipei`。
- **N=1 設計**：`ledger_*` 表不帶 chat_id（單一選手系統，北極星 §0）；TG 推播對象 = `OWNER_CHAT_ID` env（fallback：`users` 表唯一 `is_active` 使用者）。
- **API 數字是字串**（如 `"ev_loss": "22.66..."`）：入帳前一律 `float()`。
- **Ad-hoc 探測**寫 `scripts/_tmp.py`（gitignored），不用 inline `python -c`。
- **測試**：新測試加進 `scripts/regression_test.py`（`@test` 裝飾器 + `assert_eq`/`assert_in`/`assert_true`），fixture-based、離線可跑。修 bug 必附 regression test。
- **硬性 STOP gate**：Task 10（首份診斷預覽）產出後必須停下等選手驗收，批准前不得進行 Task 11+。
- **語言**：程式碼/註解/commit 英文；使用者可見輸出（記分卡、TG 訊息）繁體中文，術語沿用現有 terminology 規範。

## 已驗證的 API 事實（執行時直接引用，不需重新發現）

- Base `https://api.gtowizard.com`；headers：`authorization: Bearer <access>`（`scripts/gto_token.get_access_token()` 取得，自動 refresh）+ `gwclientid: <uuid>` + `origin: https://app.gtowizard.com`。
- `POST /v4/hand-history/hands/`：body `{"filters": {"played_at__range": ["2026-02-28T16:00:00.000Z", null], "analyzer_game_format": "TOURNAMENT"}, "pagination": {"limit": 100, "offset": 0, "ordering": ["played_at"]}, "response_fields": [...]}` → `{"items": [...], "total": N, ...}`。`ordering` 支援 `played_at`、`-total_ev_loss`；**filter 不支援 `total_ev_loss__range`（422）**。
- `GET /v4/hand-history/hands/{uuid}/` → 完整 detail：`game_analysis.game_points[]` 每個動作一節點；hero 決策節點有 `analysis_solved.available_actions[]`（每個動作的 `frequency/frequency_difference/correctness/ev/ev_loss/ev_loss_as_pot/gto_score/selected`）、`hand_eq`；節點另有 `real_game`（`pot`、`pot_odds`、`board`、`current_street.type` ∈ PREFLOP/FLOP/TURN/RIVER、players+stacks）、`gametype`、`depth`、`solved_game_action`（tree-snapped）；`game_analysis` 頂層有 `warning_status`、`approximation_reason`、`live_solved_from_street`、`live_solved_depth`。
- 已知樣本（用於 fixtures 與 pinned tests）：
  - `eef0b07b-23b6-4fe0-bcc6-41d83629583c`：SB Qh8c，board `Kh6h4hQs8s`，SRP，7 人桌，depth 35.125，`total_ev_loss = 22.6626541335449`，`hand_correctness = BLUNDER`；hero 6 個決策，全部 0 損失除了 river 第 2 決策（check 後面對 R16.65 棄牌：`ev_loss = 22.6626541335449`、`correctness = BLUNDER`、best = C `BEST_MOVE`、taken_freq = 0.0、`hand_eq = 0.7069345116615295`）。15 個 game_points。
  - `bed8860a-442b-4478-a9b4-8acfd52b6143`：SB 5c3d 4 人桌 preflop fold，depth 22.222，total_ev_loss 0.0，hero 1 個決策（F，`BEST_MOVE`）。
- 規模：since 3/1 共 33,608 手；僅 2,448 手（7.3%）total_ev_loss > 0。
- Token 壞掉（FORCED_LOGOUT）的修復程序：repo skill `.claude/skills/gtow-cdp-session/SKILL.md`。

---

### Task 0: Worktree 與腳手架

**Files:**
- Create: worktree `~/ai-poker-wizard-phase1-ledger`（branch `feat/phase1-ledger`）
- Modify: `.gitignore`

**Interfaces:**
- Produces: 之後所有 task 的工作目錄。

- [ ] **Step 1: 建 worktree 與 symlinks**

```bash
cd ~/ai-poker-wizard && git fetch origin main && git pull origin main
git worktree add ~/ai-poker-wizard-phase1-ledger -b feat/phase1-ledger
cd ~/ai-poker-wizard-phase1-ledger
ln -s ~/ai-poker-wizard/.env .env
ln -s ~/ai-poker-wizard/.tokens.json .tokens.json
ln -s ~/ai-poker-wizard/.gto_cache .gto_cache
mkdir -p data/gtow_raw/list data/gtow_raw/detail data/scorecards \
         scripts/fixtures/gtow docs/superpowers/specs docs/superpowers/plans
```

- [ ] **Step 2: 把 spec 與本 plan 搬進 worktree（它們在 main repo 是 untracked，不搬 worktree 裡不會有）**

```bash
mv ~/ai-poker-wizard/docs/superpowers/specs/2026-07-07-phase1-ledger-design.md docs/superpowers/specs/
mv ~/ai-poker-wizard/docs/superpowers/plans/2026-07-07-phase1-ledger.md docs/superpowers/plans/
```

之後一律讀 worktree 內的這兩份（勾選框也勾在 worktree 的 plan 上）。

- [ ] **Step 3: .gitignore 補條目**

在 `.gitignore` 加：

```
data/gtow_raw/
data/scorecards/
.gtow_client_id
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore docs/superpowers
git commit -m "docs: phase1 ledger spec + implementation plan; chore: scaffold data dirs"
```

---

### Task 1: Ledger DB migration

**Files:**
- Create: `supabase/migrations/20260707000000_add_ledger_tables.sql`
- Modify: `src/database.py`（`_REQUIRED_TABLES` 加 5 個表名）

**Interfaces:**
- Produces: 表 `ledger_hands`、`ledger_decisions`、`ledger_sessions`、`coach_focus`、`scorecards`（欄位如下，之後所有 task 依賴這些欄名）。

- [ ] **Step 1: 寫 migration SQL**

```sql
-- Ledger tables: grader-agnostic decision ledger (North Star §5.2/§6).
-- N=1 system: no chat_id on ledger tables (single-player, NORTH_STAR §0).

CREATE TABLE ledger_hands (
  id BIGSERIAL PRIMARY KEY,
  gtow_hand_id TEXT NOT NULL UNIQUE,
  played_at TIMESTAMPTZ NOT NULL,
  tournament_id TEXT,
  tournament_name TEXT,
  tournament_buyin NUMERIC,
  file_name TEXT,
  site TEXT,
  position TEXT,
  hero_hand TEXT,
  boards TEXT,
  pot_type TEXT,                    -- GTOW pot type (Preflop/SRP/3bet/...)
  total_players INT,
  preflop_depth_bb REAL,
  total_ev_loss_bb REAL,
  total_ev_loss_pct_pot REAL,
  avg_gto_score REAL,
  winloss_bb REAL,
  hand_correctness TEXT,
  solution_status TEXT,
  session_id BIGINT,                -- FK filled by session rebuild
  raw_path TEXT,                    -- local raw archive path (detail JSON)
  detail_fetched BOOLEAN NOT NULL DEFAULT FALSE,
  ingested_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_ledger_hands_played ON ledger_hands(played_at);
CREATE INDEX idx_ledger_hands_detail ON ledger_hands(detail_fetched) WHERE NOT detail_fetched;
CREATE INDEX idx_ledger_hands_tourney ON ledger_hands(tournament_id);

CREATE TABLE ledger_decisions (
  id BIGSERIAL PRIMARY KEY,
  gtow_hand_id TEXT NOT NULL REFERENCES ledger_hands(gtow_hand_id) ON DELETE CASCADE,
  street TEXT NOT NULL,             -- preflop/flop/turn/river
  decision_idx INT NOT NULL,        -- 0-based hero decision counter within street
  source TEXT NOT NULL DEFAULT 'online',
  grader TEXT NOT NULL DEFAULT 'gtow_analyzer',
  family TEXT NOT NULL,             -- our spot_categorizer taxonomy
  texture TEXT,                     -- our classify_board_texture
  gtow_texture TEXT,                -- GTOW connectedness/pairedness "oesd_possible/not_paired"
  depth_band TEXT NOT NULL,         -- le15 / 15_25 / 25_40 / 40plus
  position TEXT,
  pot_type TEXT,
  facing TEXT,                      -- e.g. "vs_R16.65", "unopened", "checked_to"
  taken_code TEXT,
  best_code TEXT,
  correctness TEXT,                 -- GTOW grade on taken action
  ev_loss_bb REAL,
  ev_loss_pct_pot REAL,
  taken_freq REAL,
  freq_diff REAL,
  gto_score REAL,
  hand_eq REAL,
  pot_bb REAL,
  gametype TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  approx_flags JSONB NOT NULL DEFAULT '[]',
  excluded BOOLEAN NOT NULL DEFAULT FALSE,
  played_at TIMESTAMPTZ,            -- denormalized for time queries
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(gtow_hand_id, street, decision_idx)
);
CREATE INDEX idx_ledger_decisions_family ON ledger_decisions(family, depth_band);
CREATE INDEX idx_ledger_decisions_played ON ledger_decisions(played_at);
CREATE INDEX idx_ledger_decisions_loss ON ledger_decisions(ev_loss_bb) WHERE ev_loss_bb > 0;

CREATE TABLE ledger_sessions (
  id BIGSERIAL PRIMARY KEY,
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ NOT NULL,
  duration_min REAL,
  tournaments JSONB NOT NULL DEFAULT '[]',
  max_concurrent_tables INT,
  hands_count INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE coach_focus (
  id BIGSERIAL PRIMARY KEY,
  week TEXT NOT NULL UNIQUE,        -- ISO week label e.g. "2026-W28"
  families JSONB NOT NULL,          -- [{family, depth_band, rationale...}]
  rationale JSONB,
  prescriptions JSONB,              -- [{label, url}]
  readback JSONB,                   -- next-week delta, filled by following scorecard
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scorecards (
  id BIGSERIAL PRIMARY KEY,
  week TEXT NOT NULL UNIQUE,
  html TEXT,
  data_json JSONB,
  pushed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE ledger_hands ENABLE ROW LEVEL SECURITY;
ALTER TABLE ledger_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ledger_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE coach_focus ENABLE ROW LEVEL SECURITY;
ALTER TABLE scorecards ENABLE ROW LEVEL SECURITY;
```

- [ ] **Step 2: Push 並驗證**

```bash
cd ~/ai-poker-wizard-phase1-ledger && supabase db push
set -a && source .env && set +a
psql "$SUPABASE_CONN" -c "\dt ledger_*" -c "\dt coach_focus" -c "\dt scorecards"
```

Expected: 5 個表列出。

- [ ] **Step 3: `src/database.py` 的 `_REQUIRED_TABLES` 加入** `"ledger_hands", "ledger_decisions", "ledger_sessions", "coach_focus", "scorecards"`。

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/20260707000000_add_ledger_tables.sql src/database.py
git commit -m "feat(ledger): ledger_* tables migration (hands/decisions/sessions/coach_focus/scorecards)"
```

---

### Task 2: GTOW Analyze API client

**Files:**
- Create: `scripts/gtow_analyze_api.py`
- Test: `scripts/regression_test.py`（新增 3 個測試）

**Interfaces:**
- Consumes: `scripts/gto_token.get_access_token() -> str`（已存在）。
- Produces:
  - `LIST_FIELDS: list[str]`（下方完整列出）
  - `list_hands(since_iso: str, until_iso: str | None = None, offset: int = 0, limit: int = 100, ordering: list[str] | None = None, request_fn=None) -> dict`（回傳 API 原始 dict：`{items,total,...}`）
  - `iter_all_hands(since_iso, until_iso=None, page_size=100, request_fn=None) -> Iterator[dict]`（自動翻頁 yield 每個 list row）
  - `hand_detail(gtow_hand_id: str, request_fn=None) -> dict`
  - `get_client_id() -> str`（`.gtow_client_id` 持久化 uuid）

- [ ] **Step 1: 寫失敗測試（加進 `scripts/regression_test.py`，跟現有測試同格式）**

```python
@test
def test_analyze_api_pagination():
    """iter_all_hands pages until offset >= total using injected transport."""
    import gtow_analyze_api as gapi
    pages = [
        {"items": [{"hand_id": "a"}, {"hand_id": "b"}], "total": 3, "limit": 2, "offset": 0},
        {"items": [{"hand_id": "c"}], "total": 3, "limit": 2, "offset": 2},
    ]
    calls = []
    def fake_request(method, url, **kw):
        calls.append(kw["json"]["pagination"]["offset"])
        class R:
            status_code = 200
            def json(self): return pages[len(calls) - 1]
            content = b"{}"
        return R()
    rows = list(gapi.iter_all_hands("2026-02-28T16:00:00.000Z", page_size=2,
                                    request_fn=fake_request))
    assert_eq([r["hand_id"] for r in rows], ["a", "b", "c"])
    assert_eq(calls, [0, 2])


@test
def test_analyze_api_backoff_then_success():
    """429 twice then 200 -> returns parsed json; delays follow _backoff_delay."""
    import gtow_analyze_api as gapi
    assert_eq(gapi._backoff_delay(0), 2)
    assert_eq(gapi._backoff_delay(3), 16)
    seq = [429, 429, 200]
    def fake_request(method, url, **kw):
        class R:
            status_code = seq.pop(0)
            def json(self): return {"items": [], "total": 0}
            content = b"{}"
        return R()
    out = gapi.list_hands("2026-02-28T16:00:00.000Z", request_fn=fake_request,
                          _sleep=lambda s: None)
    assert_eq(out["total"], 0)


@test
def test_analyze_api_client_id_persisted(tmp_path=None):
    import gtow_analyze_api as gapi, os, uuid as _uuid
    p = "/tmp/_test_gtow_client_id"
    if os.path.exists(p): os.remove(p)
    a = gapi.get_client_id(path=p)
    b = gapi.get_client_id(path=p)
    assert_eq(a, b)
    _uuid.UUID(a)  # raises if not a uuid
    os.remove(p)
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
python scripts/regression_test.py -k analyze_api
```

Expected: FAIL（module not found）。

- [ ] **Step 3: 實作 `scripts/gtow_analyze_api.py`**

```python
#!/usr/bin/env python3
"""GTO Wizard Analyze API client (hand-history list + detail).

Auth = Bearer access token (gto_token) + GWCLIENTID header. Global
throttle ~2.5 rps with jitter; exponential backoff on 429/5xx; one token
re-mint retry on 401. All probing that discovered this contract lives in
docs/superpowers/specs/2026-07-07-phase1-ledger-design.md §3.
"""
from __future__ import annotations

import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_token import get_access_token

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"
_CLIENT_ID_PATH = Path(__file__).resolve().parent.parent / ".gtow_client_id"
_MIN_INTERVAL = 0.4          # ~2.5 rps
_MAX_RETRIES = 5
_TIMEOUT = 30
_last_request_ts = 0.0

LIST_FIELDS = [
    "played_at", "total_ev_loss", "total_ev_loss_as_pot", "avg_gto_score",
    "avg_frequency_difference", "player_winloss", "player_position",
    "pot_type", "hero_hand", "boards", "hand_correctness",
    "preflop_game_depth", "blinds", "game_format", "file_original_name",
    "site", "solution_status", "total_players",
    "tournament_id", "tournament_name", "tournament_buyin", "total_pot",
    "board_flop_connectedness", "board_flop_pairedness",
    "actions_with_correctness_preflop", "actions_with_correctness_flop",
    "actions_with_correctness_turn", "actions_with_correctness_river",
]


def get_client_id(path: Path | str = _CLIENT_ID_PATH) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    cid = str(uuid.uuid4())
    p.write_text(cid)
    return cid


def _headers() -> dict:
    return {
        "authorization": f"Bearer {get_access_token()}",
        "gwclientid": get_client_id(),
        "origin": ORIGIN,
        "content-type": "application/json",
    }


def _backoff_delay(attempt: int) -> int:
    return min(2 ** (attempt + 1), 60)


def _throttle(_sleep=time.sleep):
    global _last_request_ts
    wait = _MIN_INTERVAL + random.uniform(0, 0.2) - (time.monotonic() - _last_request_ts)
    if wait > 0:
        _sleep(wait)
    _last_request_ts = time.monotonic()


def _request(method: str, url: str, request_fn=None, _sleep=time.sleep, **kw):
    """Central request with throttle/backoff/401-retry. request_fn injectable for tests."""
    fn = request_fn or requests.request
    reminted = False
    for attempt in range(_MAX_RETRIES + 1):
        if request_fn is None:
            _throttle(_sleep)
            kw["headers"] = _headers()
            kw["timeout"] = _TIMEOUT
        r = fn(method, url, **kw)
        if r.status_code == 401 and not reminted:
            reminted = True          # token may have just expired; re-mint once
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
            _sleep(_backoff_delay(attempt))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"GTOW Analyze API {r.status_code} for {url}: {r.content[:300]!r}")
        return r.json()
    raise RuntimeError(f"GTOW Analyze API retries exhausted for {url}")


def list_hands(since_iso: str, until_iso: str | None = None, offset: int = 0,
               limit: int = 100, ordering: list[str] | None = None,
               request_fn=None, _sleep=time.sleep) -> dict:
    body = {
        "filters": {
            "played_at__range": [since_iso, until_iso],
            "analyzer_game_format": "TOURNAMENT",
        },
        "pagination": {"limit": limit, "offset": offset,
                       "ordering": ordering or ["played_at"]},
        "response_fields": LIST_FIELDS,
    }
    return _request("POST", f"{API_BASE}/v4/hand-history/hands/",
                    request_fn=request_fn, _sleep=_sleep, json=body)


def iter_all_hands(since_iso: str, until_iso: str | None = None,
                   page_size: int = 100, request_fn=None) -> Iterator[dict]:
    offset = 0
    while True:
        page = list_hands(since_iso, until_iso, offset=offset, limit=page_size,
                          request_fn=request_fn)
        items = page.get("items", [])
        yield from items
        offset += len(items)
        if offset >= page.get("total", 0) or not items:
            return


def hand_detail(gtow_hand_id: str, request_fn=None) -> dict:
    return _request("GET", f"{API_BASE}/v4/hand-history/hands/{gtow_hand_id}/",
                    request_fn=request_fn)
```

注意 `_request` 在注入 `request_fn` 時不做 throttle/不加 headers（純測試路徑）；`json=body` kwarg 對 fake 也可見（測試靠它讀 offset）。

- [ ] **Step 4: 跑測試通過**

```bash
python scripts/regression_test.py -k analyze_api
```

Expected: 3 PASS。

- [ ] **Step 5: Live smoke（1 個請求）**

寫進 `scripts/_tmp.py` 後執行：`from gtow_analyze_api import list_hands; print(list_hands("2026-02-28T16:00:00.000Z", limit=1)["total"])` → 期望 ≥ 33608。

- [ ] **Step 6: Commit**

```bash
git add scripts/gtow_analyze_api.py scripts/regression_test.py
git commit -m "feat(ledger): GTOW Analyze API client (throttle/backoff/pagination)"
```

---

### Task 3: 捕捉 fixtures（live，一次性）

**Files:**
- Create: `scripts/capture_gtow_fixtures.py`、`scripts/fixtures/gtow/{list_rows.json, detail_eef0b07b.json, detail_bed8860a.json}`

**Interfaces:**
- Produces: fixtures 供 Task 4/5 測試；`list_rows.json` = 兩手的 list row（用 `played_at__range` 夾出該手當日 + 比對 hand_id 撈到）。

- [ ] **Step 1: 寫 `scripts/capture_gtow_fixtures.py`**

```python
#!/usr/bin/env python3
"""Capture frozen GTOW Analyze fixtures for ledger regression tests."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gtow_analyze_api import iter_all_hands, hand_detail

FIX = Path(__file__).resolve().parent / "fixtures" / "gtow"
TARGETS = {
    "eef0b07b-23b6-4fe0-bcc6-41d83629583c": ["2026-05-30T00:00:00.000Z", "2026-05-31T00:00:00.000Z"],
    "bed8860a-442b-4478-a9b4-8acfd52b6143": ["2026-03-01T00:00:00.000Z", "2026-03-02T00:00:00.000Z"],
}

rows = {}
for hid, (a, b) in TARGETS.items():
    for row in iter_all_hands(a, b):
        if row["hand_id"] == hid:
            rows[hid] = row
            break
    assert hid in rows, f"list row not found for {hid}"
    det = hand_detail(hid)
    (FIX / f"detail_{hid[:8]}.json").write_text(json.dumps(det, indent=1))
    print(hid[:8], "detail game_points:",
          len(det["game_analysis"]["game_points"]))

(FIX / "list_rows.json").write_text(json.dumps(rows, indent=1))
print("fixtures written to", FIX)
```

- [ ] **Step 2: 執行並檢視**

```bash
python scripts/capture_gtow_fixtures.py
python - <<'EOF'
import json; d=json.load(open('scripts/fixtures/gtow/detail_eef0b07b.json'))
print(sorted(d.keys())); print(len(d['game_analysis']['game_points']))
EOF
```

Expected: `game_points` = 15（eef0b07b）。順便人工檢視 `real_game.current_street.type` 的 preflop 值寫法（預期 `"PREFLOP"`，若不同，記下真值供 Task 4 使用）。

- [ ] **Step 3: Commit**（fixtures 是選手自己的手牌資料，私有 repo，允許入庫）

```bash
git add scripts/capture_gtow_fixtures.py scripts/fixtures/gtow/
git commit -m "test(ledger): frozen GTOW Analyze fixtures (river blunder + preflop fold)"
```

---

### Task 4: 蒸餾器 `ledger_distill.py`

**Files:**
- Create: `scripts/ledger_distill.py`
- Test: `scripts/regression_test.py`

**Interfaces:**
- Consumes: `spot_categorizer.categorize_spot(hand, street, action_index, street_actions_before_hero) -> (family, texture)`、`spot_categorizer.compute_pot_type_from_preflop`、fixtures。
- Produces:
  - `distill_hand(list_row: dict, detail: dict) -> tuple[dict, list[dict]]` — `(hand_row, decision_rows)`，欄名 = Task 1 表欄名。
  - `depth_band(bb: float) -> str`
  - `CHIPEV_FLAG = "chipev_grading"` 等 flag 常數。

- [ ] **Step 1: 寫失敗測試（pinned 值來自本 plan 開頭「已驗證的 API 事實」）**

```python
def _load_fix(name):
    import json
    from pathlib import Path
    return json.loads((Path("scripts/fixtures/gtow") / name).read_text())


@test
def test_distill_river_blunder_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"]
    det = _load_fix("detail_eef0b07b.json")
    hand, decs = distill_hand(lr, det)

    assert_eq(hand["gtow_hand_id"], "eef0b07b-23b6-4fe0-bcc6-41d83629583c")
    assert_eq(hand["position"], "SB")
    assert_eq(round(hand["total_ev_loss_bb"], 4), 22.6627)
    assert_eq(hand["hand_correctness"], "BLUNDER")

    assert_eq(len(decs), 6)
    assert_eq([d["street"] for d in decs],
              ["preflop", "flop", "turn", "turn", "river", "river"])
    assert_eq([d["decision_idx"] for d in decs], [0, 0, 0, 1, 0, 1])

    pre = decs[0]
    assert_eq(pre["family"], "open_raise")
    assert_eq(pre["correctness"], "BEST_MOVE")
    assert_eq(pre["ev_loss_bb"], 0.0)
    assert_eq(pre["depth_band"], "25_40")

    flop = decs[1]
    assert_eq(flop["family"], "cbet_oop")
    assert_eq(flop["texture"], "monotone")     # Kh6h4h
    assert_eq(flop["correctness"], "CORRECT_MOVE")

    riv = decs[5]
    assert_eq(riv["family"], "check_raise")    # checked, now facing a bet
    assert_eq(riv["taken_code"], "F")
    assert_eq(riv["best_code"], "C")
    assert_eq(riv["correctness"], "BLUNDER")
    assert_eq(round(riv["ev_loss_bb"], 4), 22.6627)
    assert_eq(round(riv["hand_eq"], 4), 0.7069)
    assert_true(riv["facing"].startswith("vs_R"), riv["facing"])
    assert_true("chipev_grading" in riv["approx_flags"])
    assert_eq(riv["excluded"], False)

    # fidelity property: per-decision losses sum to hand total
    assert_eq(round(sum(d["ev_loss_bb"] for d in decs), 4),
              round(hand["total_ev_loss_bb"], 4))


@test
def test_distill_preflop_fold_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["bed8860a-442b-4478-a9b4-8acfd52b6143"]
    det = _load_fix("detail_bed8860a.json")
    hand, decs = distill_hand(lr, det)
    assert_eq(len(decs), 1)
    assert_eq(decs[0]["street"], "preflop")
    assert_eq(decs[0]["taken_code"], "F")
    assert_eq(decs[0]["correctness"], "BEST_MOVE")
    assert_eq(decs[0]["family"], "open_raise")
    assert_eq(decs[0]["depth_band"], "15_25")
    assert_eq(hand["total_ev_loss_bb"], 0.0)


@test
def test_distill_honesty_rules():
    """Synthetic mutations of the fixture exercise every honesty rule (pure fn)."""
    import copy
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = copy.deepcopy(rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"])
    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))

    det["game_analysis"]["warning_status"] = "SOMETHING_ODD"
    _, decs = distill_hand(lr, det)
    assert_true(all(d["excluded"] for d in decs))
    assert_true(all(any(f.startswith("warning:") for f in d["approx_flags"]) for d in decs))

    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))
    det["game_analysis"]["approximation_reason"] = "NEAREST_DEPTH"
    _, decs = distill_hand(lr, det)
    assert_true(all(any(f.startswith("approx:") for f in d["approx_flags"]) for d in decs))
    assert_true(not any(d["excluded"] for d in decs))  # approx flags don't exclude

    lr2 = copy.deepcopy(lr); lr2["solution_status"] = "NO_SOLUTION"
    _, decs = distill_hand(lr2, _load_fix("detail_eef0b07b.json"))
    assert_true(all(d["excluded"] for d in decs))


@test
def test_depth_band_boundaries():
    from ledger_distill import depth_band
    assert_eq(depth_band(9.9), "le15")
    assert_eq(depth_band(15.0), "15_25")
    assert_eq(depth_band(24.99), "15_25")
    assert_eq(depth_band(25.0), "25_40")
    assert_eq(depth_band(40.0), "40plus")
```

- [ ] **Step 2: 跑測試確認失敗** — `python scripts/regression_test.py -k distill` → FAIL。

- [ ] **Step 3: 實作 `scripts/ledger_distill.py`**

```python
#!/usr/bin/env python3
"""Distill raw GTOW Analyze JSON into ledger rows. Pure functions, re-runnable.

Input = (list_row, detail) as returned by gtow_analyze_api. Output rows use
exactly the ledger_hands / ledger_decisions column names. Raw stays on disk;
this module can always be re-run over the archive when taxonomy evolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_categorizer import categorize_spot, compute_pot_type_from_preflop

STREET_ORDER = ["preflop", "flop", "turn", "river"]
CHIPEV_FLAG = "chipev_grading"
DEPTH_GAP_BB = 3.0
SIZING_SNAP_REL = 0.25


def depth_band(bb: float) -> str:
    if bb < 15: return "le15"
    if bb < 25: return "15_25"
    if bb < 40: return "25_40"
    return "40plus"


def _street_of(gp: dict) -> str:
    t = (gp.get("real_game") or {}).get("current_street", {}).get("type", "")
    t = (t or "").lower()
    return t if t in STREET_ORDER else "preflop"


def _norm_code(code: str) -> str:
    """Normalize GTOW action code to spot_categorizer token vocabulary."""
    if not code: return ""
    if code == "RAI": return "AI"
    if code.startswith("B"):          # defensive: bets appear as R{n} in solved codes
        return "R" + code[1:] if len(code) > 1 else "R2"
    return code                        # F / C / X / R{n} pass through


def _board_for_street(boards: str, street: str) -> str | None:
    if not boards: return None
    n = {"flop": 6, "turn": 8, "river": 10}.get(street)
    return boards[:n] if n else None


def _mk_hand_ctx(list_row: dict, preflop_tokens: list[str], boards: str) -> dict:
    streets = []
    if boards and len(boards) >= 6:
        streets.append({"board": boards[:6]})
        if len(boards) >= 8: streets.append({"card": boards[6:8]})
        if len(boards) >= 10: streets.append({"card": boards[8:10]})
    return {
        "hero_position": list_row.get("player_position", ""),
        "preflop_actions": "-".join(preflop_tokens),
        "players_at_table": list_row.get("total_players") or 8,
        "streets": streets,
    }


def _facing(street_actions: list[dict]) -> str:
    last_aggr = None
    saw_check = False
    for a in street_actions:
        c = a["action"]
        if c.startswith("R") or c.startswith("AI"): last_aggr = c
        elif c == "X": saw_check = True
    if last_aggr: return f"vs_{last_aggr}"
    return "checked_to" if saw_check else "unopened"


def _hand_flags(list_row: dict, ga: dict) -> tuple[list[str], bool]:
    """Hand-level honesty flags + hand-level exclusion."""
    flags, excluded = [], False
    ss = list_row.get("solution_status")
    if ss and ss != "OK":
        flags.append(f"solution:{ss}"); excluded = True
    ws = ga.get("warning_status")
    if ws and ws != "OK":
        flags.append(f"warning:{ws}"); excluded = True
    ar = ga.get("approximation_reason")
    if ar:
        flags.append(f"approx:{ar}")
    return flags, excluded


def distill_hand(list_row: dict, detail: dict) -> tuple[dict, list[dict]]:
    ga = detail.get("game_analysis") or {}
    gps = ga.get("game_points") or []
    hero_pos = list_row.get("player_position", "")
    boards = (list_row.get("boards") or [""])[0]
    real_depth = float(list_row.get("preflop_game_depth") or 0)

    hand_row = {
        "gtow_hand_id": list_row["hand_id"],
        "played_at": list_row["played_at"],
        "tournament_id": list_row.get("tournament_id"),
        "tournament_name": list_row.get("tournament_name"),
        "tournament_buyin": list_row.get("tournament_buyin"),
        "file_name": list_row.get("file_original_name"),
        "site": list_row.get("site"),
        "position": hero_pos,
        "hero_hand": list_row.get("hero_hand"),
        "boards": boards,
        "pot_type": list_row.get("pot_type"),
        "total_players": list_row.get("total_players"),
        "preflop_depth_bb": real_depth,
        "total_ev_loss_bb": float(list_row.get("total_ev_loss") or 0),
        "total_ev_loss_pct_pot": float(list_row.get("total_ev_loss_as_pot") or 0),
        "avg_gto_score": float(list_row["avg_gto_score"]) if list_row.get("avg_gto_score") is not None else None,
        "winloss_bb": float(list_row["player_winloss"]) if list_row.get("player_winloss") is not None else None,
        "hand_correctness": list_row.get("hand_correctness"),
        "solution_status": list_row.get("solution_status"),
    }

    hand_flags, hand_excluded = _hand_flags(list_row, ga)
    gtow_texture = "/".join(x for x in (list_row.get("board_flop_connectedness"),
                                        list_row.get("board_flop_pairedness")) if x) or None

    decisions: list[dict] = []
    preflop_tokens: list[str] = []               # action-order tokens (== seat order round 1)
    street_actions: dict[str, list[dict]] = {s: [] for s in STREET_ORDER}
    hero_count: dict[str, int] = {s: 0 for s in STREET_ORDER}

    for gp in gps:
        rga = gp.get("real_game_action") or {}
        sga = gp.get("solved_game_action") or rga
        pos = rga.get("position", "")
        street = _street_of(gp)
        code = _norm_code(sga.get("code") or rga.get("code") or "")

        sol = gp.get("analysis_solved") or {}
        avail = sol.get("available_actions") or []
        is_hero_decision = pos == hero_pos and any(a.get("selected") for a in avail)

        if is_hero_decision:
            sel = next(a for a in avail if a.get("selected"))
            best = next((a for a in avail if a.get("correctness") == "BEST_MOVE"), None)
            if best is None:
                best = max(avail, key=lambda a: float(a.get("ev") or 0))
            idx = hero_count[street]
            if street == "preflop":
                fam, tex = categorize_spot(
                    _mk_hand_ctx(list_row, preflop_tokens + [code], boards),
                    "preflop", action_index=idx)
            else:
                fam, tex = categorize_spot(
                    _mk_hand_ctx(list_row, preflop_tokens, boards),
                    street, action_index=idx,
                    street_actions_before_hero=list(street_actions[street]))

            flags = list(hand_flags)
            excluded = hand_excluded
            corr = sel.get("correctness")
            if corr in (None, "UNSOLVED"):
                flags.append("unsolved"); excluded = True
            gp_depth = float(gp.get("depth") or 0)
            if real_depth and gp_depth and abs(real_depth - gp_depth) > DEPTH_GAP_BB:
                flags.append("depth_snap_gap")
            gametype = gp.get("gametype") or ""
            if "ICM" not in gametype.upper():
                flags.append(CHIPEV_FLAG)
            rb, sb_ = rga.get("betsize"), sga.get("betsize")
            try:
                rbf, sbf = float(rb or 0), float(sb_ or 0)
                if rbf > 0 and sbf > 0 and abs(rbf - sbf) / rbf > SIZING_SNAP_REL:
                    flags.append("sizing_snap")
            except (TypeError, ValueError):
                pass

            rg = gp.get("real_game") or {}
            decisions.append({
                "gtow_hand_id": list_row["hand_id"],
                "street": street, "decision_idx": idx,
                "source": "online", "grader": "gtow_analyzer",
                "family": fam, "texture": tex, "gtow_texture": gtow_texture,
                "depth_band": depth_band(real_depth),
                "position": hero_pos,
                "pot_type": compute_pot_type_from_preflop(
                    "-".join(preflop_tokens), list_row.get("total_players") or 8),
                "facing": _facing(street_actions[street]),
                "taken_code": _norm_code((sel.get("action") or {}).get("code", "")),
                "best_code": _norm_code((best.get("action") or {}).get("code", "")),
                "correctness": corr,
                "ev_loss_bb": float(sel.get("ev_loss") or 0),
                "ev_loss_pct_pot": float(sel.get("ev_loss_as_pot") or 0),
                "taken_freq": float(sel.get("frequency") or 0),
                "freq_diff": float(sel.get("frequency_difference") or 0),
                "gto_score": float(sel.get("gto_score") or 0),
                "hand_eq": float(sol.get("hand_eq") or 0) or None,
                "pot_bb": float(rg.get("pot") or 0) or None,
                "gametype": gametype,
                "confidence": 1.0,
                "approx_flags": flags,
                "excluded": excluded,
                "played_at": list_row["played_at"],
            })
            hero_count[street] = idx + 1

        # record the action AFTER the decision so "before hero" lists are correct
        if street == "preflop":
            preflop_tokens.append(code)
        else:
            street_actions[street].append({"position": pos, "action": code})

    return hand_row, decisions
```

- [ ] **Step 4: 跑測試** — `python scripts/regression_test.py -k distill` → 期望 PASS。若 fixture 揭露的欄位形狀與上述推定不同（例如 street type 字串、preflop family/board 邊界），**修 code 對齊 fixture 真值**（不是改斷言遷就 code）；pinned 數值（22.6627/0.7069/BLUNDER/6 decisions/idx 序列）來自 live 探測，是 ground truth。

- [ ] **Step 5: Commit**

```bash
git add scripts/ledger_distill.py scripts/regression_test.py
git commit -m "feat(ledger): pure distiller raw->ledger rows with honesty flags"
```

---

### Task 5: 攝取 CLI `ledger_ingest.py`

**Files:**
- Create: `scripts/ledger_ingest.py`
- Test: `scripts/regression_test.py`（1 個純函數測試）+ live smoke

**Interfaces:**
- Consumes: Task 2 client、Task 4 distiller、`SUPABASE_CONN` env。
- Produces:
  - CLI：`--backfill --since 2026-03-01` / `--incremental` / `--verify` / `--limit N`（dev 用）
  - `raw_paths(hand_id, played_at) -> (list_dir, detail_path)`；raw 落地 `data/gtow_raw/detail/YYYY-MM/{hand_id}.json.gz` 與 `data/gtow_raw/list/YYYY-MM.jsonl.gz`（append）
  - stdout 摘要行：`INGEST list=<n_new_hands> detail=<n_fetched> decisions=<n> skipped=<n_known>`（`/ingest` 指令與測試都 parse 這行）
  - `--verify` 不符時 exit code 2 並印 `VERIFY MISMATCH api=<N> db=<M>`

- [ ] **Step 1: 失敗測試（upsert SQL 冪等性由 ON CONFLICT 保證；測 raw path 與月分桶純函數）**

```python
@test
def test_ingest_raw_paths():
    from ledger_ingest import raw_paths
    ld, dp = raw_paths("abc-123", "2026-05-30T21:03:23Z")
    assert_true(str(dp).endswith("data/gtow_raw/detail/2026-05/abc-123.json.gz"))
    assert_true(str(ld).endswith("data/gtow_raw/list/2026-05.jsonl.gz"))
```

- [ ] **Step 2: 跑測試失敗**、**Step 3: 實作**

```python
#!/usr/bin/env python3
"""Ingest GTOW Analyze hands into the ledger. Idempotent, resumable, loud.

Modes:
  --backfill --since 2026-03-01   full list sweep + full detail sweep
  --incremental                   re-sweep trailing 30 days (late-upload safe)
  --verify                        API total vs DB count since 3/1; exit 2 on mismatch
  --limit N                       cap detail fetches this run (dev/smoke)
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))

import gtow_analyze_api as gapi
from ledger_distill import distill_hand

RAW = ROOT / "data" / "gtow_raw"
EPOCH_SINCE = "2026-02-28T16:00:00.000Z"     # 2026-03-01 Taipei
HAND_COLS = [
    "gtow_hand_id", "played_at", "tournament_id", "tournament_name",
    "tournament_buyin", "file_name", "site", "position", "hero_hand",
    "boards", "pot_type", "total_players", "preflop_depth_bb",
    "total_ev_loss_bb", "total_ev_loss_pct_pot", "avg_gto_score",
    "winloss_bb", "hand_correctness", "solution_status",
]
DEC_COLS = [
    "gtow_hand_id", "street", "decision_idx", "source", "grader", "family",
    "texture", "gtow_texture", "depth_band", "position", "pot_type", "facing",
    "taken_code", "best_code", "correctness", "ev_loss_bb", "ev_loss_pct_pot",
    "taken_freq", "freq_diff", "gto_score", "hand_eq", "pot_bb", "gametype",
    "confidence", "approx_flags", "excluded", "played_at",
]


def raw_paths(hand_id: str, played_at: str):
    ym = played_at[:7]
    return (RAW / "list" / f"{ym}.jsonl.gz",
            RAW / "detail" / ym / f"{hand_id}.json.gz")


def _ts(v):  # ISO str -> aware datetime for asyncpg
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else v


async def upsert_hand(conn, h: dict):
    vals = [h.get(c) for c in HAND_COLS]
    vals[1] = _ts(vals[1])
    cols = ", ".join(HAND_COLS)
    ph = ", ".join(f"${i+1}" for i in range(len(HAND_COLS)))
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in HAND_COLS if c != "gtow_hand_id")
    await conn.execute(
        f"INSERT INTO ledger_hands ({cols}) VALUES ({ph}) "
        f"ON CONFLICT (gtow_hand_id) DO UPDATE SET {upd}", *vals)


async def upsert_decisions(conn, decs: list[dict]):
    for d in decs:
        vals = [d.get(c) for c in DEC_COLS]
        vals[DEC_COLS.index("played_at")] = _ts(d["played_at"])
        vals[DEC_COLS.index("approx_flags")] = json.dumps(d["approx_flags"])
        cols = ", ".join(DEC_COLS)
        ph = ", ".join(f"${i+1}" for i in range(len(DEC_COLS)))
        upd = ", ".join(f"{c}=EXCLUDED.{c}"
                        for c in DEC_COLS if c not in ("gtow_hand_id", "street", "decision_idx"))
        await conn.execute(
            f"INSERT INTO ledger_decisions ({cols}) VALUES ({ph}) "
            f"ON CONFLICT (gtow_hand_id, street, decision_idx) DO UPDATE SET {upd}", *vals)


async def sweep_list(conn, since_iso: str) -> tuple[int, int]:
    new = known = 0
    for row in gapi.iter_all_hands(since_iso):
        lp, _ = raw_paths(row["hand_id"], row["played_at"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        exists = await conn.fetchval(
            "SELECT 1 FROM ledger_hands WHERE gtow_hand_id=$1", row["hand_id"])
        if exists:
            known += 1
            continue
        with gzip.open(lp, "at") as f:
            f.write(json.dumps(row) + "\n")
        hand_row, _ = distill_hand(row, {"game_analysis": {"game_points": []}})
        await upsert_hand(conn, hand_row)
        new += 1
        if new % 500 == 0:
            print(f"  list sweep: {new} new...", flush=True)
    return new, known


async def sweep_detail(conn, limit: int | None) -> tuple[int, int]:
    rows = await conn.fetch(
        "SELECT gtow_hand_id, played_at FROM ledger_hands "
        "WHERE NOT detail_fetched ORDER BY played_at")
    fetched = ndec = 0
    for r in rows:
        if limit and fetched >= limit:
            break
        hid = r["gtow_hand_id"]
        played = r["played_at"].isoformat()
        _, dp = raw_paths(hid, played)
        dp.parent.mkdir(parents=True, exist_ok=True)
        det = gapi.hand_detail(hid)
        with gzip.open(dp, "wt") as f:
            json.dump(det, f)
        lp, _ = raw_paths(hid, played)
        list_row = _find_list_row(lp, hid)
        hand_row, decs = distill_hand(list_row, det)
        async with conn.transaction():
            await upsert_hand(conn, hand_row)
            await upsert_decisions(conn, decs)
            await conn.execute(
                "UPDATE ledger_hands SET detail_fetched=true, raw_path=$2 "
                "WHERE gtow_hand_id=$1", hid, str(dp.relative_to(ROOT)))
        fetched += 1
        ndec += len(decs)
        if fetched % 100 == 0:
            print(f"  detail sweep: {fetched}/{len(rows)}", flush=True)
    return fetched, ndec


def _find_list_row(list_path: Path, hand_id: str) -> dict:
    with gzip.open(list_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["hand_id"] == hand_id:
                return row
    raise RuntimeError(f"list row for {hand_id} not in {list_path}")


async def verify(conn) -> int:
    api_total = gapi.list_hands(EPOCH_SINCE, limit=1)["total"]
    db_total = await conn.fetchval("SELECT count(*) FROM ledger_hands")
    if api_total == db_total:
        print(f"VERIFY OK api={api_total} db={db_total}")
        return 0
    print(f"VERIFY MISMATCH api={api_total} db={db_total}")
    return 2


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import os
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        if a.verify:
            return await verify(conn)
        if a.incremental:
            since_dt = datetime.now(timezone.utc) - timedelta(days=30)
            since = since_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            since = f"{a.since}T00:00:00.000Z" if "T" not in a.since else a.since
        n_new, n_known = await sweep_list(conn, since)
        n_det, n_dec = await sweep_detail(conn, a.limit)
        print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} skipped={n_known}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
```

注意：list sweep 先用空 detail 建 hand row（`distill_hand` 帶空 game_points 只產 hand_row），detail sweep 補齊。`--incremental` since = now−30d，同一條路徑天然冪等。

- [ ] **Step 4: 跑純函數測試 PASS** — `python scripts/regression_test.py -k ingest_raw`

- [ ] **Step 5: Live smoke（5 手）+ 冪等驗證**

```bash
python scripts/ledger_ingest.py --backfill --since 2026-03-01 --limit 5
python scripts/ledger_ingest.py --backfill --since 2026-03-01 --limit 5
```

Expected: 第一次 `detail=5`；第二次 list `skipped` 全部、`detail=5`（接續下 5 手，不重複前 5 手）。查 DB：`psql "$SUPABASE_CONN" -c "SELECT count(*) FROM ledger_decisions"` > 0。

- [ ] **Step 6: Commit** — `git add scripts/ledger_ingest.py scripts/regression_test.py && git commit -m "feat(ledger): idempotent resumable ingest CLI (backfill/incremental/verify)"`

---

### Task 6: Session 重建 `ledger_sessions.py`

**Files:**
- Create: `scripts/ledger_sessions.py`
- Test: `scripts/regression_test.py`

**Interfaces:**
- Produces: `cluster_sessions(hands: list[dict]) -> list[dict]`（hands = `{gtow_hand_id, played_at(datetime), tournament_id}` 按 played_at 升冪；回傳 `{started_at, ended_at, duration_min, tournaments, max_concurrent_tables, hands_count, hand_ids}`）；CLI `--rebuild` 清空重建 `ledger_sessions` 並回填 `ledger_hands.session_id`。

- [ ] **Step 1: 失敗測試**

```python
@test
def test_session_clustering():
    from datetime import datetime, timedelta, timezone
    from ledger_sessions import cluster_sessions
    t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    mk = lambda i, mins, t: {"gtow_hand_id": f"h{i}",
                             "played_at": t0 + timedelta(minutes=mins),
                             "tournament_id": t}
    hands = [mk(1, 0, "A"), mk(2, 5, "B"), mk(3, 10, "A"),   # session 1: A+B overlap
             mk(4, 200, "C"), mk(5, 210, "C")]               # gap 190min -> session 2
    ss = cluster_sessions(hands)
    assert_eq(len(ss), 2)
    assert_eq(ss[0]["hands_count"], 3)
    assert_eq(ss[0]["max_concurrent_tables"], 2)
    assert_eq(sorted(ss[0]["tournaments"]), ["A", "B"])
    assert_eq(ss[1]["max_concurrent_tables"], 1)
```

- [ ] **Step 2: 失敗**、**Step 3: 實作**（gap > 60 分鐘切段；併發 = 以 10 分鐘滑動窗看不同 tournament_id 數的最大值）

```python
#!/usr/bin/env python3
"""Rebuild ledger_sessions from ledger_hands timestamps (gap>60min clustering)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

GAP = timedelta(minutes=60)
WINDOW = timedelta(minutes=10)


def cluster_sessions(hands: list[dict]) -> list[dict]:
    sessions, cur = [], []
    for h in hands:
        if cur and h["played_at"] - cur[-1]["played_at"] > GAP:
            sessions.append(_finish(cur)); cur = []
        cur.append(h)
    if cur:
        sessions.append(_finish(cur))
    return sessions


def _finish(hands: list[dict]) -> dict:
    tourneys = sorted({h["tournament_id"] for h in hands if h["tournament_id"]})
    max_cc = 1
    for h in hands:
        cc = {g["tournament_id"] for g in hands
              if g["tournament_id"] and abs((g["played_at"] - h["played_at"]).total_seconds())
              <= WINDOW.total_seconds()}
        max_cc = max(max_cc, len(cc) or 1)
    start, end = hands[0]["played_at"], hands[-1]["played_at"]
    return {"started_at": start, "ended_at": end,
            "duration_min": (end - start).total_seconds() / 60,
            "tournaments": tourneys, "max_concurrent_tables": max_cc,
            "hands_count": len(hands),
            "hand_ids": [h["gtow_hand_id"] for h in hands]}


async def rebuild():
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        rows = await conn.fetch(
            "SELECT gtow_hand_id, played_at, tournament_id FROM ledger_hands ORDER BY played_at")
        sessions = cluster_sessions([dict(r) for r in rows])
        async with conn.transaction():
            await conn.execute("UPDATE ledger_hands SET session_id=NULL")
            await conn.execute("DELETE FROM ledger_sessions")
            for s in sessions:
                sid = await conn.fetchval(
                    "INSERT INTO ledger_sessions (started_at, ended_at, duration_min, "
                    "tournaments, max_concurrent_tables, hands_count) "
                    "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    s["started_at"], s["ended_at"], s["duration_min"],
                    json.dumps(s["tournaments"]), s["max_concurrent_tables"], s["hands_count"])
                await conn.execute(
                    "UPDATE ledger_hands SET session_id=$1 WHERE gtow_hand_id = ANY($2)",
                    sid, s["hand_ids"])
        print(f"SESSIONS rebuilt: {len(sessions)}")
    finally:
        await conn.close()


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        asyncio.run(rebuild())
```

- [ ] **Step 4: 測試 PASS**、**Step 5: Commit** — `git commit -m "feat(ledger): session reconstruction (gap clustering + concurrency)"`

---

### Task 7: 啟動全量 backfill（live，背景）

**Files:** 無新檔（操作性 task）。

- [ ] **Step 1: 背景啟動**

```bash
cd ~/ai-poker-wizard-phase1-ledger
nohup python scripts/ledger_ingest.py --backfill --since 2026-03-01 > /tmp/ledger_backfill.log 2>&1 &
echo $! > /tmp/ledger_backfill.pid
```

- [ ] **Step 2: 確認在跑** — `sleep 120 && tail -5 /tmp/ledger_backfill.log`：應見 list sweep 進度。預期總時 ~4-5 小時（33.6k × ~0.45s）。**不等它跑完** — 繼續 Task 8/9（純 code + fixtures），Task 10 開頭再回來驗收。若中斷：直接重跑同指令（冪等續傳）。

---

### Task 8: 診斷引擎 `ledger_diagnostics.py`

**Files:**
- Create: `scripts/ledger_diagnostics.py`
- Test: `scripts/regression_test.py`

**Interfaces:**
- Consumes: `ledger_decisions`/`ledger_hands`/`ledger_sessions` 列（以 dict list 傳入 — 純函數；thin async fetchers 附在模組尾）。
- Produces（全部帶 `n` 與 `excluded_n`；輸入一律先濾 `excluded=False`）：
  - `weekly_series(decisions, tz="Asia/Taipei") -> list[{week, ev_loss_per_100, total_bb, n}]`
  - `leak_board(decisions, min_n=25, weeks_window=4) -> list[{family, depth_band, total_bb, n, per100, trend, leak_type, slice_desc}]`（EV 排序；`trend` = 近 4 週 per100 − 前 4 週 per100；不足 n 的 cell 收進回傳 dict 的 `insufficient` list）
  - `classify_leak(family_decisions) -> (leak_type, slice_desc)`（`boundary` 若單一 depth_band 或 texture 子切片佔該 family 損失 ≥70% 且子切片 n≥10，否則 `knowledge`）
  - `session_correlations(decisions, hands, sessions) -> {by_hour: [...], by_tables: [...], post_bad_beat: {...}}`（bad beat = `winloss_bb < -20` 的手，其後 15 分鐘內決策的 per100 vs 全體）
  - `most_expensive_hands(hands, k=3) -> list[hand dict]`
  - `pick_focus(board, k=2) -> list[cell]`（過 n 門檻的前 k）
  - async fetchers：`fetch_decisions(conn, since=None)`、`fetch_hands(conn, since=None)`、`fetch_sessions(conn)`

- [ ] **Step 1: 失敗測試（合成資料，deterministic）**

```python
def _dec(family="facing_cbet_oop", band="15_25", loss=0.0, week_day="2026-06-01",
         texture="wet", excluded=False):
    from datetime import datetime, timezone
    return {"family": family, "depth_band": band, "ev_loss_bb": loss,
            "texture": texture, "excluded": excluded,
            "played_at": datetime.fromisoformat(week_day + "T12:00:00+00:00")}


@test
def test_leak_board_ev_ranking_and_min_n():
    from ledger_diagnostics import leak_board
    decs = ([_dec(loss=1.0)] * 30                                   # 30bb over n=30
            + [_dec(family="open_raise", band="40plus", loss=5.0)] * 3   # big but n<25
            + [_dec(family="probe", loss=0.0)] * 40)
    out = leak_board(decs, min_n=25)
    ranked = out["cells"]
    assert_eq(ranked[0]["family"], "facing_cbet_oop")
    assert_eq(ranked[0]["n"], 30)
    assert_eq(round(ranked[0]["per100"], 2), round(30 / 30 * 100, 2))
    assert_true(all(c["family"] != "open_raise" for c in ranked))
    assert_true(any(c["family"] == "open_raise" for c in out["insufficient"]))


@test
def test_classify_leak_boundary_vs_knowledge():
    from ledger_diagnostics import classify_leak
    conc = [_dec(band="le15", loss=1.0)] * 12 + [_dec(band="40plus", loss=0.1)] * 12
    t, desc = classify_leak(conc)
    assert_eq(t, "boundary")
    assert_in("le15", desc)
    spread = ([_dec(band="le15", loss=0.5)] * 12 + [_dec(band="15_25", loss=0.5)] * 12
              + [_dec(band="40plus", loss=0.5)] * 12)
    t2, _ = classify_leak(spread)
    assert_eq(t2, "knowledge")


@test
def test_weekly_series_tz_bucketing():
    from ledger_diagnostics import weekly_series
    # 2026-06-07 15:59 UTC = 06-07 23:59 Taipei (Sunday, W23); 16:01 UTC = 06-08 Taipei (Monday, W24)
    from datetime import datetime, timezone
    d1 = dict(_dec(loss=2.0), played_at=datetime(2026, 6, 7, 15, 59, tzinfo=timezone.utc))
    d2 = dict(_dec(loss=0.0), played_at=datetime(2026, 6, 7, 16, 1, tzinfo=timezone.utc))
    out = weekly_series([d1, d2])
    assert_eq([w["week"] for w in out], ["2026-W23", "2026-W24"])
    assert_eq(out[0]["n"], 1)
```

- [ ] **Step 2: 失敗** → **Step 3: 實作**。核心程式碼：

```python
#!/usr/bin/env python3
"""EV-weighted diagnostics over ledger rows. Pure functions + thin fetchers."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

TPE = ZoneInfo("Asia/Taipei")


def _week_label(dt: datetime) -> str:
    y, w, _ = dt.astimezone(TPE).isocalendar()
    return f"{y}-W{w:02d}"


def _included(decisions):
    return [d for d in decisions if not d.get("excluded")]


def weekly_series(decisions: list[dict]) -> list[dict]:
    by_week: dict[str, list[float]] = defaultdict(list)
    for d in _included(decisions):
        by_week[_week_label(d["played_at"])].append(d["ev_loss_bb"] or 0.0)
    out = []
    for wk in sorted(by_week):
        losses = by_week[wk]
        out.append({"week": wk, "n": len(losses), "total_bb": sum(losses),
                    "ev_loss_per_100": sum(losses) / len(losses) * 100})
    return out


def classify_leak(family_decisions: list[dict]) -> tuple[str, str]:
    total = sum(d["ev_loss_bb"] or 0 for d in family_decisions) or 1e-9
    for dim in ("depth_band", "texture"):
        slices: dict[str, list[dict]] = defaultdict(list)
        for d in family_decisions:
            if d.get(dim):
                slices[d[dim]].append(d)
        for key, ds in slices.items():
            share = sum(x["ev_loss_bb"] or 0 for x in ds) / total
            if share >= 0.7 and len(ds) >= 10:
                return "boundary", f"{dim}={key} ({share:.0%} of loss)"
    return "knowledge", "loss spread across slices"


def leak_board(decisions: list[dict], min_n: int = 25,
               weeks_window: int = 4) -> dict:
    inc = _included(decisions)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for d in inc:
        cells[(d["family"], d["depth_band"])].append(d)
        by_family[d["family"]].append(d)

    latest = max((d["played_at"] for d in inc), default=datetime.now(timezone.utc))
    cur_lo = latest - timedelta(weeks=weeks_window)
    prev_lo = latest - timedelta(weeks=2 * weeks_window)

    def per100(ds):
        return (sum(d["ev_loss_bb"] or 0 for d in ds) / len(ds) * 100) if ds else 0.0

    ranked, insufficient = [], []
    for (fam, band), ds in cells.items():
        total = sum(d["ev_loss_bb"] or 0 for d in ds)
        cur = [d for d in ds if d["played_at"] >= cur_lo]
        prev = [d for d in ds if prev_lo <= d["played_at"] < cur_lo]
        ltype, sdesc = classify_leak(by_family[fam])
        row = {"family": fam, "depth_band": band, "total_bb": total,
               "n": len(ds), "per100": per100(ds),
               "trend": per100(cur) - per100(prev),
               "trend_n": (len(cur), len(prev)),
               "leak_type": ltype, "slice_desc": sdesc}
        (ranked if len(ds) >= min_n else insufficient).append(row)
    ranked.sort(key=lambda r: -r["total_bb"])
    insufficient.sort(key=lambda r: -r["total_bb"])
    return {"cells": ranked, "insufficient": insufficient,
            "excluded_n": len(decisions) - len(inc)}


def most_expensive_hands(hands: list[dict], k: int = 3) -> list[dict]:
    return sorted(hands, key=lambda h: -(h.get("total_ev_loss_bb") or 0))[:k]


def pick_focus(board: dict, k: int = 2) -> list[dict]:
    return board["cells"][:k]


def session_correlations(decisions, hands, sessions) -> dict:
    inc = _included(decisions)
    hand_by_id = {h["gtow_hand_id"]: h for h in hands}
    sess_start = {s["id"]: s["started_at"] for s in sessions}

    def bucket(key_fn):
        b: dict = defaultdict(list)
        for d in inc:
            h = hand_by_id.get(d["gtow_hand_id"])
            if not h:
                continue
            key = key_fn(d, h)
            if key is not None:
                b[key].append(d["ev_loss_bb"] or 0)
        return [{"key": key, "n": len(v), "per100": sum(v) / len(v) * 100,
                 "low_n": len(v) < 20}
                for key, v in sorted(b.items())]

    by_hour = bucket(lambda d, h: (
        int((h["played_at"] - sess_start[h["session_id"]]).total_seconds() // 3600)
        if h.get("session_id") in sess_start else None))
    sess_tables = {s["id"]: s.get("max_concurrent_tables") for s in sessions}
    by_tables = bucket(lambda d, h: sess_tables.get(h.get("session_id")))

    bad_beats = sorted(h["played_at"] for h in hands if (h.get("winloss_bb") or 0) < -20)
    def in_window(t):
        import bisect
        i = bisect.bisect_left(bad_beats, t) - 1
        return i >= 0 and (t - bad_beats[i]).total_seconds() <= 900 and t != bad_beats[i]
    win = [d["ev_loss_bb"] or 0 for d in inc
           if in_window(hand_by_id.get(d["gtow_hand_id"], {}).get("played_at", d["played_at"]))]
    base = [d["ev_loss_bb"] or 0 for d in inc]
    post_bb = {"n": len(win),
               "per100": (sum(win) / len(win) * 100) if win else 0.0,
               "baseline_per100": (sum(base) / len(base) * 100) if base else 0.0,
               "low_n": len(win) < 20}
    return {"by_hour": by_hour, "by_tables": by_tables, "post_bad_beat": post_bb}


async def fetch_decisions(conn, since=None):
    q = "SELECT * FROM ledger_decisions" + (" WHERE played_at >= $1" if since else "")
    rows = await (conn.fetch(q, since) if since else conn.fetch(q))
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("approx_flags"), str):
            import json as _json
            d["approx_flags"] = _json.loads(d["approx_flags"])
        out.append(d)
    return out


async def fetch_hands(conn, since=None):
    q = "SELECT * FROM ledger_hands" + (" WHERE played_at >= $1" if since else "")
    return [dict(r) for r in await (conn.fetch(q, since) if since else conn.fetch(q))]


async def fetch_sessions(conn):
    return [dict(r) for r in await conn.fetch("SELECT * FROM ledger_sessions")]
```

→ **Step 4: PASS** → **Step 5: Commit** `feat(ledger): EV-weighted diagnostics (leak board / leak typing / sessions / weekly series)`

---

### Task 9: 記分卡 `scorecard.py`（HTML + 處方 + 回讀）

**Files:**
- Create: `scripts/scorecard.py`
- Test: `scripts/regression_test.py`

**Interfaces:**
- Consumes: Task 8 診斷、`gtow_trainer_url.build_trainer_url(spot_category, street, effective_bb, pot_type=None, gametype="MTTGeneral")`（**已存在**，`SpotNotSupportedError` 要接）、`coach_focus`/`scorecards` 表。
- Produces:
  - `compute_scorecard_data(decisions, hands, sessions, prev_focus: dict | None, window_label: str) -> dict`（純函數 → `data_json`；含 `headline`（中文一句話）、`weekly_series`、`leak_board`、`top_hands`、`focus`（含 `prescriptions` 連結清單與 `readback`）、`honesty`（excluded_n / unsolved_share / chipev_share / verify 狀態）、`session_obs`）
  - `render_scorecard_html(data) -> str`（self-contained，inline CSS + SVG sparkline）
  - `analyze_table_url(day_start_taipei: str, day_end_taipei: str) -> str`（保底復盤連結：`https://app.gtowizard.com/analyze/v4/hands/table?filters=<urlencoded json {"played_at__range":[utc_start,utc_end]}>&preselectGamemode=TOURNAMENT` — 格式取自選手實際使用的 URL）
  - CLI：`--preview`（全歷史窗、**不寫** coach_focus/scorecards、輸出 `data/scorecards/preview.html` + `preview_data.json` + `preview_summary.md`）；`--weekly`（本週窗、寫表、輸出 `data/scorecards/<week>.html`）
- 處方規則：`pick_focus` 的 cell → `build_trainer_url(family, street=("preflop" if family 屬 preflop 桶 else "flop"), effective_bb=band 中值 {le15:12, 15_25:20, 25_40:32, 40plus:50}, pot_type=cell 眾數 pot_type)`；`SpotNotSupportedError` → fallback `analyze_table_url`。preflop 桶集合 = `{open_raise, facing_open, possible_squeeze, hero_3bet, facing_3bet, vs_squeeze, squeeze, facing_4bet, limp_pot}`（`spot_categorizer` docstring 的九桶）。
- readback：`prev_focus` 的每個 family → 本窗 per100 vs 處方當週 per100，附「單週讀數僅供參考，連續 4 週才算數」。

- [ ] **Step 1: 失敗測試**

```python
@test
def test_scorecard_data_and_html():
    from ledger_diagnostics import leak_board
    from scorecard import compute_scorecard_data, render_scorecard_html
    decs = [_dec(loss=1.0)] * 30 + [_dec(family="probe", loss=0.0)] * 40
    hands = [{"gtow_hand_id": "h1", "played_at": decs[0]["played_at"],
              "total_ev_loss_bb": 22.7, "hero_hand": "Qh8c", "position": "SB",
              "boards": "Kh6h4hQs8s", "tournament_name": "Test $5",
              "winloss_bb": -10.8, "session_id": 1}]
    data = compute_scorecard_data(decs, hands, [], prev_focus=None,
                                  window_label="2026-W28")
    assert_true(data["headline"])
    assert_eq(data["leak_board"]["cells"][0]["family"], "facing_cbet_oop")
    assert_true(data["focus"]["families"])
    assert_true(data["focus"]["families"][0]["prescriptions"][0]["url"]
                .startswith("https://app.gtowizard.com/"))
    html = render_scorecard_html(data)
    assert_in("facing_cbet_oop", html)
    assert_in("<svg", html)
    assert_in("誠實層", html)
    assert_true("<script src" not in html)   # self-contained


@test
def test_analyze_table_url_shape():
    from scorecard import analyze_table_url
    url = analyze_table_url("2026-05-30", "2026-05-30")
    assert_in("app.gtowizard.com/analyze/v4/hands/table?filters=", url)
    assert_in("preselectGamemode=TOURNAMENT", url)
```

- [ ] **Step 2: 失敗** → **Step 3: 實作**。HTML 模板骨架（完整寫出，CSS 精簡；`_svg_sparkline(points)` 用 `<polyline>`）：

```python
def _svg_sparkline(values: list[float], w=560, h=80) -> str:
    if not values:
        return "<svg/>"
    mx, mn = max(values) or 1, min(values)
    span = (mx - mn) or 1
    pts = " ".join(f"{i * w / max(len(values) - 1, 1):.1f},"
                   f"{h - (v - mn) / span * (h - 10) - 5:.1f}"
                   for i, v in enumerate(values))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{pts}"/></svg>')
```

頁面區塊順序（`render_scorecard_html`）：標題+窗標籤 → 主指標卡（本窗 per100、n、對上窗 delta）+ sparkline（`weekly_series` 的 per100 序列）→ leak 榜 table（family/depth_band/total_bb/n/per100/trend/型別）→ 最貴 3 手卡（牌、位置、損失、`analyze_table_url` 連結）→ 焦點處方（family、理由、trainer 連結按鈕、readback 區）→ session 觀察 → 誠實層附註。中文文案；`headline` 生成規則：`f"本窗 EV loss {per100:.2f}bb/100（n={n}），{'較上窗改善' if delta<0 else '較上窗惡化' if delta>0 else '持平'} {abs(delta):.2f}"`。

核心程式碼（`compute_scorecard_data` / `analyze_table_url` / CLI；render 依上述區塊以 f-string 組裝，樣式 inline `<style>` 一段即可）：

```python
#!/usr/bin/env python3
"""Weekly scorecard: data compute (pure) + self-contained HTML + CLI."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gtow_trainer_url import build_trainer_url, SpotNotSupportedError
import ledger_diagnostics as diag

TPE = ZoneInfo("Asia/Taipei")
PREFLOP_FAMILIES = {"open_raise", "facing_open", "possible_squeeze", "hero_3bet",
                    "facing_3bet", "vs_squeeze", "squeeze", "facing_4bet", "limp_pot"}
BAND_MID = {"le15": 12, "15_25": 20, "25_40": 32, "40plus": 50}


def analyze_table_url(day_start_taipei: str, day_end_taipei: str) -> str:
    start = datetime.fromisoformat(day_start_taipei).replace(tzinfo=TPE)
    end = datetime.fromisoformat(day_end_taipei).replace(tzinfo=TPE) + timedelta(days=1)
    fmt = lambda d: d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filters = json.dumps({"played_at__range": [fmt(start), fmt(end)]})
    return (f"https://app.gtowizard.com/analyze/v4/hands/table"
            f"?filters={quote(filters)}&preselectGamemode=TOURNAMENT")


def _hand_review_url(h: dict) -> str:
    day = h["played_at"].astimezone(TPE).strftime("%Y-%m-%d")
    return analyze_table_url(day, day)


def _prescribe(cell: dict, decisions: list[dict]) -> list[dict]:
    fam, band = cell["family"], cell["depth_band"]
    street = "preflop" if fam in PREFLOP_FAMILIES else "flop"
    pot_types = [d.get("pot_type") for d in decisions
                 if d["family"] == fam and d.get("pot_type")]
    pot_type = max(set(pot_types), key=pot_types.count) if pot_types else "SRP"
    out = []
    try:
        out.append({"label": f"GTOW Trainer drill：{fam} @{band}",
                    "url": build_trainer_url(fam, street, BAND_MID[band],
                                             pot_type=None if street == "preflop" else pot_type)})
    except SpotNotSupportedError:
        # fallback: link to the Analyze table so the spot is still one click away
        today = datetime.now(TPE).strftime("%Y-%m-%d")
        out.append({"label": f"復盤 {fam} 手牌（Analyze）",
                    "url": analyze_table_url(today, today)})
    return out


def compute_scorecard_data(decisions, hands, sessions, prev_focus, window_label) -> dict:
    series = diag.weekly_series(decisions)
    board = diag.leak_board(decisions)
    inc = [d for d in decisions if not d.get("excluded")]
    n = len(inc)
    per100 = (sum(d["ev_loss_bb"] or 0 for d in inc) / n * 100) if n else 0.0
    delta = (series[-1]["ev_loss_per_100"] - series[-2]["ev_loss_per_100"]
             if len(series) >= 2 else 0.0)
    word = "較上窗改善" if delta < 0 else ("較上窗惡化" if delta > 0 else "持平")

    focus_cells = diag.pick_focus(board)
    focus = {"families": [dict(c, prescriptions=_prescribe(c, inc),
                               top_hands=[dict(h, review_url=_hand_review_url(h))
                                          for h in _family_top_hands(c, inc, hands)])
                          for c in focus_cells],
             "readback": _readback(prev_focus, decisions)}

    unsolved = sum(1 for d in decisions
                   if any(f == "unsolved" for f in d.get("approx_flags", [])))
    chipev = sum(1 for d in inc
                 if any(f == "chipev_grading" for f in d.get("approx_flags", [])))
    return {
        "window": window_label,
        "headline": f"本窗 EV loss {per100:.2f}bb/100（n={n}），{word} {abs(delta):.2f}",
        "per100": per100, "n": n, "delta": delta,
        "weekly_series": series, "leak_board": board,
        "top_hands": [dict(h, review_url=_hand_review_url(h))
                      for h in diag.most_expensive_hands(hands)],
        "focus": focus,
        "session_obs": diag.session_correlations(decisions, hands, sessions),
        "honesty": {"excluded_n": board["excluded_n"], "unsolved_n": unsolved,
                    "chipev_share": (chipev / n) if n else 0.0,
                    "note": "chipEV 評分：後期/泡沫手的判定含 ICM 近似誤差（Phase 3 處理）；"
                            "單週讀數僅供參考，連續 4 週才算數"},
    }


def _family_top_hands(cell, decisions, hands, k=3):
    ids = {d["gtow_hand_id"] for d in decisions
           if d["family"] == cell["family"] and (d["ev_loss_bb"] or 0) > 0}
    fam_hands = [h for h in hands if h["gtow_hand_id"] in ids]
    return diag.most_expensive_hands(fam_hands, k)


def _readback(prev_focus, decisions):
    if not prev_focus:
        return None
    out = []
    for f in prev_focus.get("families", []):
        fam = f["family"]
        fam_ds = [d for d in decisions
                  if d["family"] == fam and not d.get("excluded")]
        cur = (sum(d["ev_loss_bb"] or 0 for d in fam_ds) / len(fam_ds) * 100) if fam_ds else 0.0
        out.append({"family": fam, "prescribed_per100": f.get("per100"),
                    "current_per100": cur, "n": len(fam_ds),
                    "note": "單週讀數僅供參考，連續 4 週才算數"})
    return out
```

CLI（同檔案尾）：`--preview` = fetch 全期 → compute（`prev_focus=None`、label `"preview-all-history"`）→ 寫 `data/scorecards/preview.html` + `preview_data.json`（`json.dumps(default=str)`）+ `preview_summary.md`（Task 10 Step 2 的 7 個 section，由 data 逐項 render 成 markdown）；`--weekly` = fetch 本週窗（本週一 00:00 Taipei 起）+ 讀 `coach_focus` 上一列作 `prev_focus` → compute → 寫 html 檔 + `INSERT INTO scorecards` + `INSERT INTO coach_focus`（本週 focus，`per100` 記進 families 各項）+ 回填上週列的 `readback`。

- [ ] **Step 4: PASS** → **Step 5: Commit** `feat(ledger): scorecard html + prescription + readback`

---

### Task 10: 保真對數 + 首份診斷預覽 ⛔ STOP GATE

**Files:**
- Create: `scripts/ledger_fidelity_check.py`
- 產物: `data/scorecards/preview.html`、`preview_summary.md`、`fidelity_report.md`

**前置**：backfill 完成 — `tail /tmp/ledger_backfill.log` 見 `INGEST list=... detail=...` 結尾行；`python scripts/ledger_ingest.py --verify` 印 `VERIFY OK`（api == db）。然後 `python scripts/ledger_sessions.py --rebuild`。

- [ ] **Step 1: 寫並跑 `scripts/ledger_fidelity_check.py`**

```python
#!/usr/bin/env python3
"""Fidelity check: 20 random lossy hands — ledger vs re-fetched API detail."""
import asyncio
import os
import random
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
from gtow_analyze_api import hand_detail


async def main():
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    hands = await conn.fetch(
        "SELECT gtow_hand_id, played_at, total_ev_loss_bb FROM ledger_hands "
        "WHERE total_ev_loss_bb > 0.1 ORDER BY random() LIMIT 20")
    lines, mismatches = [], 0
    for h in hands:
        det = hand_detail(h["gtow_hand_id"])
        api_loss = sum(
            float(a["ev_loss"]) for gp in det["game_analysis"]["game_points"]
            for a in (gp.get("analysis_solved") or {}).get("available_actions", [])
            if a.get("selected"))
        db_loss = await conn.fetchval(
            "SELECT COALESCE(sum(ev_loss_bb),0) FROM ledger_decisions WHERE gtow_hand_id=$1",
            h["gtow_hand_id"])
        ok = abs(api_loss - db_loss) < 1e-4
        mismatches += (not ok)
        lines.append(f"| {h['gtow_hand_id'][:8]} | {h['played_at']:%m-%d} "
                     f"| {db_loss:.3f} | {api_loss:.3f} | {'✅' if ok else '❌'} |")
    report = ("# Fidelity check (20 random lossy hands)\n\n"
              "| hand | date | ledger bb | api bb | match |\n|---|---|---|---|---|\n"
              + "\n".join(lines) + f"\n\nmismatches: {mismatches}/20\n")
    (ROOT / "data/scorecards/fidelity_report.md").write_text(report)
    print(report)
    await conn.close()
    return 1 if mismatches else 0


raise SystemExit(asyncio.run(main()))
```

Expected: `mismatches: 0/20`。非 0 → 修 distill/ingest 到 0 才能繼續（見 Global：修 bug 附 regression test）。

- [ ] **Step 2: 產首份預覽**

```bash
python scripts/scorecard.py --preview
```

產出 `preview.html` + `preview_data.json` + `preview_summary.md`。`preview_summary.md` 必含（由 data 生成，非手寫）：
1. 全期主指標：EV loss/100 + 週趨勢方向 + n
2. **Top 5 leak lines**（family × depth_band、total bb、n、per100、知識/邊界型 + slice_desc）— 這就是選手要的「你有沒有觀察到我在某個 line 比較常損失 EV」
3. 每個 top-3 leak line 的最貴 2 手（日期、牌、損失、`analyze_table_url` 連結供 GTOW UI 抽查）
4. 全期最貴 5 手
5. Session 觀察（第幾小時 / 桌數 / bad-beat 窗口的 per100 對比，帶 n）
6. 誠實層統計（excluded 決策數與占比、unsolved 占比、chipev 占比、VERIFY 狀態）
7. 假設本週開處方會開什麼（focus + trainer 連結，dry-run 不寫表）

- [ ] **Step 3: ⛔ 停下，交付選手驗收**

把 `preview_summary.md`、`preview.html`、`fidelity_report.md` 交給選手（若在 subagent session 內：以最終訊息完整呈現 summary 內容 + 檔案路徑，結束本輪等待指示）。**選手明確批准前，不得執行 Task 11+。**選手驗收方式：對照自己的對局感 + 抽 2-3 手在 GTOW UI 對數字。

- [ ] **Step 4: Commit**（fidelity script）`git add scripts/ledger_fidelity_check.py && git commit -m "test(ledger): fidelity spot-check script (20 lossy hands vs live API)"`

---

### Task 11: TG 佈線（/ingest、每日攝取 job、週日記分卡 job）

**Files:**
- Modify: `src/telegram_bot/bot.py`（~line 1316 handler 註冊區 + 新 command 方法）
- Modify: `src/main_gemini.py`（post_init 加兩個 job）
- Create: `scripts/ledger_service.py`（owner 解析 + 之後 Task 12 的查詢函數也放這）

**Interfaces:**
- Produces:
  - `ledger_service.resolve_owner_chat_id(pool) -> int | None`（env `OWNER_CHAT_ID` 優先；否則 `SELECT user_id FROM users WHERE is_active` 恰一列時用之）
  - bot `/ingest`：僅 owner 可用；`asyncio.create_subprocess_exec(sys.executable, "scripts/ledger_ingest.py", "--incremental")`，完成後回覆 stdout 的 `INGEST ...` 摘要行
  - `_daily_ingest_job(context)`：同上 subprocess + `--verify`；verify exit 2 → `bot.send_message(owner, "⚠️ Ledger 對數不符：...")`
  - `_weekly_scorecard_job(context)`：`scorecard.py --weekly` subprocess → 讀 `data/scorecards/<week>.html` → `bot.send_message(owner, 摘要)` + `bot.send_document(owner, document=open(html_path,'rb'), filename=f"scorecard-{week}.html")`
  - jobs 註冊（`main_gemini.post_init`，沿用現有 `run_daily` 模式；PTB days：0=Sun）：daily 05:00 Taipei `days=tuple(range(7))`；Sunday 21:00 Taipei `days=(0,)`

- [ ] **Step 1: 實作**（貼齊 `_weekly_report_job` 既有寫法與 logging；subprocess cwd = repo root）。核心程式碼：

`scripts/ledger_service.py`（owner 解析）：

```python
async def resolve_owner_chat_id(pool) -> int | None:
    import os
    env = os.getenv("OWNER_CHAT_ID")
    if env:
        return int(env)
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT user_id FROM users WHERE is_active")
    return rows[0]["user_id"] if len(rows) == 1 else None
```

`src/main_gemini.py`（post_init 內，沿用現有 run_daily 區塊之後；`_run_script` 為共用 helper）：

```python
async def _run_script(*args) -> tuple[int, str]:
    import asyncio, sys
    proc = await asyncio.create_subprocess_exec(
        sys.executable, *args,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode()[-3000:]


async def _daily_ingest_job(context):
    from ledger_service import resolve_owner_chat_id
    try:
        rc, out = await _run_script("scripts/ledger_ingest.py", "--incremental")
        logger.info(f"Daily ingest rc={rc}: {out.splitlines()[-1] if out else ''}")
        await _run_script("scripts/ledger_sessions.py", "--rebuild")
        rc_v, out_v = await _run_script("scripts/ledger_ingest.py", "--verify")
        if rc_v == 2:
            owner = await resolve_owner_chat_id(db.pool)
            if owner:
                await context.bot.send_message(owner, f"⚠️ Ledger 對數不符\n{out_v.strip()}")
    except Exception as e:
        logger.error(f"Daily ingest job failed: {e}")


async def _weekly_scorecard_job(context):
    from ledger_service import resolve_owner_chat_id
    try:
        rc, out = await _run_script("scripts/scorecard.py", "--weekly")
        if rc != 0:
            logger.error(f"Scorecard failed: {out}"); return
        owner = await resolve_owner_chat_id(db.pool)
        if not owner:
            logger.warning("Scorecard: no owner chat id"); return
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT week, html, data_json FROM scorecards ORDER BY created_at DESC LIMIT 1")
        data = json.loads(row["data_json"])
        cells = data["leak_board"]["cells"][:3]
        leaks = "\n".join(f"{i+1}. {c['family']} @{c['depth_band']}: "
                          f"-{c['total_bb']:.1f}bb (n={c['n']})"
                          for i, c in enumerate(cells))
        focus = data["focus"]["families"]
        links = "\n".join(p["url"] for f in focus for p in f.get("prescriptions", []))
        await context.bot.send_message(
            owner, f"📊 週記分卡 {row['week']}\n{data['headline']}\n\nTop leaks:\n{leaks}"
                   f"\n\n本週焦點處方:\n{links}")
        path = Path(__file__).resolve().parent.parent / "data/scorecards" / f"{row['week']}.html"
        with open(path, "rb") as fh:
            await context.bot.send_document(owner, document=fh,
                                            filename=f"scorecard-{row['week']}.html")
        async with db.pool.acquire() as conn:
            await conn.execute("UPDATE scorecards SET pushed_at=NOW() WHERE week=$1",
                               row["week"])
    except Exception as e:
        logger.error(f"Weekly scorecard job failed: {e}")

# post_init 註冊（現有 weekly_leak_report 之後）：
        application.job_queue.run_daily(
            _daily_ingest_job,
            time=dt_time(hour=5, minute=0, tzinfo=TZ_TAIPEI),
            days=tuple(range(7)), name="daily_ledger_ingest")
        application.job_queue.run_daily(
            _weekly_scorecard_job,
            time=dt_time(hour=21, minute=0, tzinfo=TZ_TAIPEI),
            days=(0,), name="weekly_scorecard")
```

`src/telegram_bot/bot.py`（`report_command` 附近加方法 + 1316 區註冊 `CommandHandler("ingest", self.ingest_command)`）：

```python
async def ingest_command(self, update, context):
    """Owner-only: pull newly uploaded GTOW Analyze hands into the ledger."""
    from ledger_service import resolve_owner_chat_id
    owner = await resolve_owner_chat_id(self.db.pool)
    if not owner or update.effective_chat.id != owner:
        return
    msg = await update.message.reply_text("⏳ 攝取中…")
    import asyncio, sys
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "scripts/ledger_ingest.py", "--incremental",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    summary = next((l for l in out.decode().splitlines() if l.startswith("INGEST")),
                   "INGEST（無輸出）")
    await msg.edit_text(f"✅ {summary}")
```
- [ ] **Step 2: 手動驗證**

```bash
python scripts/ledger_ingest.py --incremental   # 直跑一次，確認 INGEST 摘要行
python scripts/scorecard.py --weekly            # 直跑一次，確認 html 落地 + 表有列
```

再以 `scripts/_tmp.py` 用 `telegram.Bot(token)` 對 owner 發一次 send_document 冒煙（訊息可見即刪）。
- [ ] **Step 3: Commit** `feat(ledger): TG wiring (/ingest, daily ingest job, sunday scorecard push)`

---

### Task 12: LLM ledger 追問工具 ×2

**Files:**
- Modify: `scripts/ledger_service.py`（查詢函數）
- Modify: `src/gemini_session.py`（declarations 加在 `GET_PROGRESS_DECLARATION`（~line 820）之後；dispatch 加進 ~line 3374 的 elif tuple；execution 加在 ~line 3589 區塊，鏡射 `query_my_leaks` 的寫法與回傳格式）
- Test: `scripts/regression_test.py`

**Interfaces:**
- Produces（`ledger_service.py`）：
  - `async query_ledger_summary(pool, family=None, depth_band=None, days=None) -> dict`：`{"per100": float, "total_bb": float, "n": int, "excluded_n": int, "by_family": [{family, total_bb, n, per100} top10 EV 排序], "window_days": days}`（SQL 對 `ledger_decisions WHERE NOT excluded [AND family=$..][AND depth_band=$..][AND played_at > now()-days]` 聚合）
  - `async query_ledger_hands(pool, family=None, min_ev_loss=0.5, days=90, limit=5) -> list[dict]`：join hands，回 `{played_at, hero_hand, position, family, ev_loss_bb, tournament_name, review_url}`（`review_url` = `scorecard.analyze_table_url` 該日）
- Declarations（完整程式碼寫出，並以 `QUERY_MY_LEAKS_DECLARATION`（gemini_session.py:749）的實際風格為準對齊 types 用法）：

```python
QUERY_LEDGER_SUMMARY_DECLARATION = types.FunctionDeclaration(
    name="query_ledger_summary",
    description=(
        "查詢全量帳本（GTOW Analyzer 評分的線上 MTT 決策帳）的 EV loss 聚合。"
        "支援按 spot family、深度帶、天數過濾。回傳 EV loss/100 決策、總損失、"
        "樣本數 n、excluded 數與 top families。使用者問『我哪裡漏 EV / 某類 spot "
        "表現如何』時用這個。"),
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "family": types.Schema(type=types.Type.STRING,
                               description="spot family，如 facing_cbet_oop、open_raise"),
        "depth_band": types.Schema(type=types.Type.STRING,
                                   description="le15 / 15_25 / 25_40 / 40plus"),
        "days": types.Schema(type=types.Type.INTEGER, description="回看天數，省略=全期"),
    }),
)

QUERY_LEDGER_HANDS_DECLARATION = types.FunctionDeclaration(
    name="query_ledger_hands",
    description=("列出帳本中符合條件的具體手牌（EV loss 排序），附復盤連結。"
                 "使用者要看『哪幾手 / 最貴的手』時用這個。"),
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "family": types.Schema(type=types.Type.STRING),
        "min_ev_loss": types.Schema(type=types.Type.NUMBER, description="bb 門檻，預設 0.5"),
        "days": types.Schema(type=types.Type.INTEGER, description="預設 90"),
        "limit": types.Schema(type=types.Type.INTEGER, description="預設 5，最大 10"),
    }),
)
```

- [ ] **Step 1: 失敗測試**（declarations 可 import、名字唯一、dispatch tuple 含新名 — 用 `inspect.getsource` 檢查 dispatch 分支存在；`query_ledger_summary` 的 SQL 建構以 stub pool 驗證 WHERE 組裝 — 把 SQL 組裝抽成純函數 `_summary_sql(family, depth_band, days) -> (sql, args)` 測它）。
- [ ] **Step 2: 失敗** → **Step 3: 實作** → **Step 4: PASS**。
- [ ] **Step 5: E2E 冒煙**：`set -a && source .env && set +a && python scripts/e2e_test.py "這三個月我在哪些 spot 漏最多 EV？"` → 回答引用工具數字（帶 n）。
- [ ] **Step 6: Commit** `feat(ledger): LLM follow-up tools query_ledger_summary/hands`

---

### Task 13: 收尾（迴圈演練 runbook、文件、全套測試、PR）

**Files:**
- Create: `docs/handoffs/2026-07-XX-phase1-ledger.md`（交接：做了什麼、驗收狀態、已知限制、下一步=Phase 2）
- Modify: `AGENTS.md`（Project Structure 加 6 個新 scripts 一行說明；CLAUDE.md 指向它，不用另改）
- Create: `docs/phase1-loop-runbook.md`（選手手動驗收清單：上傳→/ingest→追問→週日記分卡→點連結→隔週回讀，每步預期結果）

- [ ] **Step 1: 全套 regression** — `python scripts/regression_test.py` 全綠（含既有測試——`gemini_session` 被改過，此為 CLAUDE.md 硬性要求）。
- [ ] **Step 2: 文件三件** 寫齊 commit。
- [ ] **Step 3: 發 PR**

```bash
git push -u origin feat/phase1-ledger
gh pr create --title "feat: Phase 1 ledger — GTOW Analyze full ingestion + minimal coach loop" \
  --body "$(cat <<'EOF'
## Summary
- Full-history fidelity ledger: 33.6k MTT hands ingested from GTOW Analyze (list+detail, raw archived, honesty-flagged, idempotent/resumable)
- EV-weighted diagnostics (leak board / knowledge-boundary typing / session correlations) + weekly HTML scorecard + focus prescription with trainer links + next-week readback
- TG: /ingest, daily ingest job, Sunday scorecard push, 2 grounded follow-up tools

## Spec / Plan
- docs/superpowers/specs/2026-07-07-phase1-ledger-design.md
- docs/superpowers/plans/2026-07-07-phase1-ledger.md

## 驗收狀態（spec §11）
- [ ] 1 全量入帳三方對數
- [ ] 2 誠實層抽查
- [ ] 3 20 手保真對數 0 mismatch
- [ ] 4 記分卡（第 1 份已出，第 2 份待下週）
- [ ] 5 處方 + coach_focus
- [ ] 6 每日增量 7 天（進行中）
- [ ] 7 TG 追問 3 問
- [ ] 8 預覽 gate 已由選手驗收

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4:** 驗收 4/6 需要真實時間（第 2 份記分卡、7 天增量）— PR 掛著等訊號累積，或選手同意先 merge 由 cron 繼續累積。由選手決定。

---

## 執行順序與依賴

```
T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7(背景 backfill 啟動)
                                    ├─ T8 → T9（fixtures 驅動，與 backfill 並行）
T7 完成 + T8/T9 完成 → T10 ⛔ 預覽驗收 gate → T11 → T12 → T13
```

## Executor 注意事項

1. **先讀 spec 全文與 NORTH_STAR §7/§13/§14**，再動工。
2. Fixture 揭露的形狀與 plan 推定不符時：**以 fixture 為準修 code**；pinned 數值是 live ground truth，不可改斷言遷就。
3. 任何 GTOW API 意外（403/schema 變化）：停下記錄到 handoff，不要硬繞。token 壞掉照 `gtow-cdp-session` skill 修。
4. Backfill 期間別重複起第二個 process（檢查 `/tmp/ledger_backfill.pid`）。
5. ⛔ Task 10 Step 3 是硬性 STOP：預覽交付選手後結束當輪，等待明確批准。
