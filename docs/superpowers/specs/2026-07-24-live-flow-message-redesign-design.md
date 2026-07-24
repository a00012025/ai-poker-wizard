# 線下流 /live 訊息與互動重新設計 (design)

- Date: 2026-07-24
- Scope: `scripts/live_flow.py`（render + grade）、`src/telegram_bot/bot.py`（/live handler + callbacks）、
  新增 `live_sessions` DB 表、`scripts/hh_deviation_check.py`（僅讀取，不改）、
  `scripts/coach_facts.py` / `scripts/coach_prompts.py`（術語）。
- 背景：`/live` 目前把整批手牌解析→評分→入 ledger + drill_queue，然後用
  `render_tg_html()` 拼出「分類報告」一次送出，report 跑完即丟、**session 不留狀態**。
  owner 反饋要把它改成「逐手分頁清單」，並補上單手重傳、每手復盤/教練/加練按鈕。

## 北極星對齊（§13 gate）

- §5.2 誠實層：depth 升格評分掛 `depth_escalated` 旗、chipEV 近似旗保留 → 不假裝精確。
- §7.3 排序不變量：queue 仍 EV 加權，加練沿用 `manual_drill_item` 既有口徑，不新增頻率排序。
- §5.1 stream 3：線下流仍是選擇性記錄的偏誤樣本，來源隔離（source='live'）不變。
- **修補必可見不變量**（[[live-flow-refuse-over-repair]]）：bulk `🔧` 區塊移除，改成
  **每手行尾「已自動校正」+ 點 🔁 重傳時顯示完整校正**，可稽核性保留（見 §4）。

---

## 1. 目標訊息版面（逐手分頁清單）

每頁 **10 手**（owner 指定）。每手：一行描述 +（可選）偏差子行 + 一整排按鈕。

```
🃏 線下入帳：24 手 / 49 決策　(第 1/3 頁)
⚠️❌ 3 偏差 · ❓ 5 待深挖 · ✅ 其餘無明顯偏差

Hand 1 · BB K8♠o · 22bb · 3B Pot · 已自動校正
　❌ preflop Call → 建議 Fold（100%）· 損失 0.40bb
　[復盤]　[💬 教練]　[➕ 加練]　[🔁 重傳]

Hand 2 · ❗ 無法評分：動作線不合法
　b4 沒被讀成小盲下注 → 修好即可評（重傳）
　[🔁 重傳]

Hand 3 · CO A♦7♦s · 30bb · 單加注池 · ✅
　[復盤]　[💬 教練]　[➕ 加練]　[🔁 重傳]

Hand 15 · LJ 88 · 15bb · 3bet 池 · ❓
　❓ preflop 起未評分：偏離 GTO 建議後，你的牌已在該線範圍外
　（已嘗試 17bb 近似，仍無範圍）
　[復盤]　[💬 教練]　[➕ 加練]　[🔁 重傳]
…
[◀ 上一頁]　　[下一頁 ▶]
```

Footer（整份一次，最後一頁或每頁尾）：
```
⚠️ chipEV 近似（現場賽段未知）；limp 節點不評分。要更正某手：點該手的 🔁 重傳。
```

### 每手描述行格式
- OK 手：`Hand N · {位置} {手牌emoji} · {depth}bb · {pot_type}{ · ✅|⚠️|❌}{ · 已自動校正}`
  - severity 取該手最嚴重節點：有 ❌(≥0.30bb) → ❌；有 ⚠️(≥0.10bb) → ⚠️；否則 ✅。
  - `已自動校正` 只在該手有 `repairs` 時出現。
- 失敗手：`Hand N · ❗ 無法評分：{標題}` + 一行精簡原因（沿用 `_failure_help` 的
  標題/help，但壓成單行）。

### 偏差子行（沿用現有，術語改「建議」）
- `　{sev} {street} {taken} → 建議 {best}（{freq}）· 損失 {ev:.2f}bb`
- 升格評分節點：子行尾註 `（於 {d}bb 近似）`。
- 升格後仍 offrange：`　❓ {street} 起未評分：偏離 GTO 建議後，你的牌已在該線範圍外`
  + 若嘗試過升格：`　（已嘗試 {d}bb 近似，仍無範圍）`。

### 移除
- `✅ 無明顯偏差：Hand 3, Hand 4…` 整行（乾淨手改為在清單中顯示自己那行 + ✅）。
- `🔧 N 手已自動校正後送 solver…` 整段 bulk 區塊（改成每手 🔧 標記，見 §4）。

---

## 2. 術語：主線 → 建議

改動點（全部 user-facing）：
- `scripts/live_flow.py`：`→ 主線` → `→ 建議`；`偏離主線` → `偏離 GTO 建議`。
- `scripts/coach_facts.py:496`：note `偏離 solver 主線` → `偏離 solver 建議`。
- `scripts/coach_prompts.py:408`：`偏離主線` → `偏離 GTO 建議`（教練深挖系統提示）。

（術語只換字，語意不變；coach 深挖路徑一併改是 owner 明確要求。）

---

## 3. 自動升格評分（Hand 15 類）

**問題**：hero 前面偏離 GTO 建議線後，到達節點時 hero 的具體牌在該線 solver 範圍
外（0 combo）→ 節點 `reason='offrange'` → 排除、不評分（❓）。

**解法**：節點 offrange 時，自動用「上一格 effective bb」重解一次，撿回那些節點的近似評分。

- depth 檔位來自 `gto_api.AVAILABLE_DEPTHS`（含 …17,14,12,10…）。
  「上一格」= AVAILABLE_DEPTHS 中「嚴格大於目前 `nearest_depth(effective_bb)` 對應
  整數檔」的最小值。例：15bb → base 檔 14 → 升格檔 17。
- 只升 **一格**；升格後該節點若仍 offrange，維持 ❓。
- 只採用「base 評分為 offrange，而升格後可評」的節點；其餘節點一律用 base 評分
  （不因升格而動）。
- 掛旗：該 decision 的 `approx_flags` 追加 `depth_escalated:{d}`（進 ledger，§5.2 誠實層）。
- 頂格（已是最高檔 100/…）無可升 → 不動。
- 位置：改在 `live_flow.grade_hand()` 包一層 `grade_hand_with_escalation(hand)`：
  1. `base = check_hand(h, emit_ungraded=True)`；建 `devmap`。
  2. 若 `devmap` 內有 `reason=='offrange'` 的節點且非頂格：
     `h2 = {**h, 'effective_bb': next_up}`；`esc = check_hand(h2, emit_ungraded=True)`。
     對每個 base 為 offrange 且 esc 可評的 `(street, idx)`，用 esc 的 dev 取代，
     並標記待 `build_hand_rows` 加 `depth_escalated` 旗。
  3. 傳回合併後的 devmap（含一個 side-channel 標記哪些節點被升格）。
- 成本：只有「含 offrange 節點的手」多跑一次 solver pass；可接受（線下批量、非即時）。

---

## 4. 修補顯示（決策：壓成每手小標記 + 點開才看）

- 移除 bulk `🔧` 區塊。
- 有 `repairs` 的手，描述行尾加「已自動校正」。
- 完整校正文字在 **點 🔁 重傳** 時顯示：重傳提示訊息帶
  `目前校正：{'；'.join(_repair_explanation(r))}`（無則「無」）。
  → 修補仍可稽核（不變量守住），但不再洗版。
- （💬 教練深挖本來就會回貼 raw 原文，額外多一層可讀性。）

---

## 5. Session 持久化（新 DB 表，決策：存 DB 可跨重啟）

### 新表 `live_sessions`
```sql
CREATE TABLE live_sessions (
  id           bigserial PRIMARY KEY,
  session_key  text UNIQUE NOT NULL,     -- live:{date}:{sha1(batch)[:10]}，冪等
  chat_id      bigint NOT NULL,
  message_id   bigint,                   -- 已送出的報告訊息（就地編輯用）
  page         int NOT NULL DEFAULT 0,
  result_json  jsonb NOT NULL,           -- 完整 result（含每手 dec_rows，供重算 queue）
  created_at   timestamptz NOT NULL DEFAULT now()
);
```
- `result_json` 存「完整」result（比送 TG 的 slim 版多留 `dec_rows`），
  重傳重算 queue 需要它。
- migration：`supabase/migrations/2026072400xxxx_live_sessions.sql`。

### callback_data（皆 owner-only，`< 64 bytes`）
- 分頁：`lvpg:<sid>:<page>`
- 教練：`lvd:<hand_id>`（**不變**，hand_id 從 result 取；失敗手無此鈕）
- 加練：`lvadd:<sid>:<hand_idx>`（展開該手決策 → 既有 `qad2` 加練）
- 重傳：`lvr:<sid>:<hand_idx>`
- 復盤：URL 按鈕（render 時建，見 §6）

`<sid>` = `live_sessions.id`（短整數，避開 session_key 內的冒號與 64B 限制）。

---

## 6. 每手按鈕

一整排（owner 指定）：`[復盤] [💬 教練] [➕ 加練] [🔁 重傳]`。失敗手僅 `[🔁 重傳]`。

- **復盤**（URL）：GTOW `/solutions?...&soltab=strategy` 策略頁，落在 hero 該決策節點看
  範圍組成。用 `gtow_solution_url.build_hand_solution_url`（線下手需先把 live parsed
  hand 轉成 resolver 可吃的格式；沿用 queue_feed 既有 `_parsed_hand_from_analyze`/
  custom-spot fallback 的做法）。建不出來就省略此鈕（不亂連別的 spot）。
- **💬 教練**：`lvd:<hand_id>`，即現行深入分析路徑（改成每手都有，非只有偏差手）。
- **➕ 加練**：`lvadd:<sid>:<hand_idx>`（見 §7）。
- **🔁 重傳**：`lvr:<sid>:<hand_idx>`（見 §8）。

Telegram：10 手 × 4 鈕 + 2 nav = 42 鈕（< 100 上限）。單列 4 顆用短標籤。
（若手機顯示過擠，退而每手 2×2；此為 render 細節，不影響資料流。）

---

## 7. 加練整合（➕ 從 live 紀錄直接排入訓練）

沿用既有 manual-add 機制，**幾乎零新程式**：
- `/live` 已把每手決策寫入 `ledger_decisions`（source='live'）。
- `lvadd:<sid>:<hand_idx>`：
  1. 從 session `result_json` 取該手 `hand_id`。
  2. 查 `ledger_decisions WHERE gtow_hand_id=$1 AND NOT excluded AND NOT discarded`
     （同 `_queue_expand_review` 的查詢）。
  3. 用 `queue_feed.qex_submenu(rows, queue_id=0)` 產生 ➕ 子選單，每列
     `qad2:0:<hand_id>:<street>:<decision_idx>`。
  4. 送出「選一條 action line 加入練習」子選單。
- 點 ➕ → 既有 `_queue_add_manual`（qad2 分支）→ `manual_drill_item` + `enqueue`
  （kind='drill', added_by='manual', source='manual'）。`queue_id=0` 只是 sentinel，
  `_queue_add_manual` 不使用它做 enqueue。
- 與現行「偏差線自動入列」共存：自動列是 ≥0.1bb 偏差；➕ 加練可排任何節點
  （含 clean / 次門檻），供 owner 主動練。

---

## 8. 單手重傳（🔁）

回答 owner 的疑問「要怎麼重送」：**點該手 🔁 重傳 → 貼單手更正 → 就地覆蓋**。

流程：
1. `lvr:<sid>:<hand_idx>`：owner 驗證後，設 in-memory pending
   `self._live_resend_pending[chat_id] = (sid, hand_idx, message_id)`；回覆
   「請貼上 Hand N 的單手更正版本（Header / Flop / Turn / River 各一行）。
   目前 echo：{echo}；目前校正：{repairs or 無}」。
2. 下一則文字訊息（在既有 `_live_pending` 攔截點旁多一個攔截）→ 當單手 batch：
   `process_batch(block, date=session_date)` 取 `hands[0]`。
3. 覆蓋：
   - 讀 `live_sessions.result_json`；以新 entry 取代 `hands[hand_idx]`（保留顯示用
     `idx=hand_idx+1` 的編號）。
   - 新 `hand_id`（內容雜湊）≠ 舊。刪舊 `hand_id` 的 `ledger_hands` +
     `ledger_decisions`；若新手 ok，寫入新 ledger 列。
   - **queue 重算**：以 session 全部現有 `dec_rows` 重跑 `select_queue_items`；
     先移除舊 `hand_id` 的 queue 貢獻（`drill_queue.source_hands` 內該 hand_id 的
     entry — 新增 `queue_feed.remove_source_hand(conn, hand_id)` helper，剝掉條目、
     重算 `total_ev_loss_bb`，剝到空的 auto/live drill row 轉 cleared），再
     `enqueue` 新結果（merge）。
   - 更新 `result_json`、`page`（維持目前頁）。
4. 重繪含 `hand_idx` 的那頁，就地 `edit_message_text` 報告訊息。
5. pending 狀態為 in-memory：bot 若在等待中重啟，owner 重點 🔁 即可（可接受）。

---

## 9. Hand 2 解析 bug（`b4` = 小盲下注）

獨立 bugfix（含 regression test，CLAUDE.md 強制）。
- 症狀：原文
  ```
  Eff 35bb Co raise hero btn call 7s8s sb raise 7bb co fold hero call
  Ac5c6d b4 call
  4s x b8 fold
  ```
  翻牌 `Ac5c6d b4 call`＝3bettor(SB) 下注 4bb、hero 跟；卻被判「SB Call 但前面沒下注」
  （orphan call）→ validation_failed。
- 待重現定位：`b4` 下注被丟/歸錯，疑點在 Gemini 結構化或 `repair_hu_pot` 動作重排。
  用 `scripts/_tmp.py` 跑真實 block 重現，找根因後修，加 regression test。
- 走 systematic-debugging，不先猜。此項可與訊息重繪並行、獨立 PR。

---

## 10. 元件邊界與測試

| 單元 | 職責 | 依賴 |
|------|------|------|
| `grade_hand_with_escalation` (live_flow) | offrange→升一格近似評分 + 掛旗 | check_hand, AVAILABLE_DEPTHS |
| `render_session_page(result, page, per_page=10)` | 逐手清單 + 分頁 HTML | — |
| `session_page_buttons(result, sid, page)` | 每手按鈕列 + nav + 復盤 URL | build_hand_solution_url |
| `live_sessions` 表 + upsert helpers | session 持久化 | asyncpg |
| `queue_feed.remove_source_hand` | 重傳時剝離舊手 queue 貢獻 | drill_queue |
| bot `lvpg/lvadd/lvr` handlers | 分頁 / 加練 / 重傳互動 | 上述 + 既有 qad2 |

測試：
- `regression_test.py` 網域模組：升格評分（offrange→graded@next depth）、
  render 分頁切頁、severity 聚合、術語「建議」、Hand 2 解析。
- Snapshot：不涉（live 非 snapshot 路徑）。
- 手動：/live 一批 >10 手驗分頁、每手四鈕、重傳覆蓋 + queue 不重複膨脹。

## 11. 不做（YAGNI）
- 不做多格連續升格（只升一格）。
- 不做重傳 pending 的 DB 持久化（in-memory，重點即可）。
- 不改 online 掃描 / 週記分卡 / /queue 既有版面。
- 不動 grading 引擎 `check_hand` 本身（只在 live 包一層）。
