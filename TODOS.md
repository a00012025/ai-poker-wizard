# TODOS

## P1 — Grounded Coach causal evidence follow-ups

### Opponent response child-node facts
After Hero chooses a solver-supported bet/raise size, query the exact Villain response
node and extract fold/call/raise frequencies, category/equity composition, and the
equity carried by the folding range. Promote explanations such as equity denial,
value targeting, raise vulnerability, or inducing only when this response node exists.

- **Why:** The current Hero node can prove what Hero should do, but not what Villain
  folds, calls, raises, or later bluffs. This is the missing evidence behind AA check
  stories, thin-value sizing, and rigorous equity-denial claims.
- **Guardrail:** Node path, size, depth, board, positions, and reach probability must
  match exactly. No response-node match means no opponent-response causal claim.
- **Depends on:** Reuse the response-node traversal already designed in
  `docs/superpowers/specs/2026-06-07-coach-followup-grounding-design.md`.

### Turn-card and stack-depth counterfactual comparisons
Query matched sibling solutions where exactly one variable changes: either the turn
card with the prior line fixed, or the effective stack/depth bucket with the spot
fixed. Compare range equity, 90–100% top-equity mass, category shares, action
frequency, size construction, and exact-combo EV/frequency.

- **Why:** This upgrades “the turn card/low SPR caused the strategy change” from a
  theory-informed correlation to counterfactual evidence.
- **Guardrail:** Only narrate the changed variable when every other node dimension is
  verified equal; otherwise retain the current exact-node explanation.
- **Cost control:** Cache sibling nodes and request them only for material EV
  deviations where the current-node evidence cannot supply a useful explanation.

## P1 — Weekly Report v2 follow-ups

### GTOW trainer URL schema reverse-engineering (Option Z)
Weekly Report v2 ships with `fh_actions=<type>` + `depth` URLs that land on the right pot-type + depth but don't pin the exact position pair (e.g. LJ-vs-HJ). User manually picks the seat on arrival. To make links position-accurate: open Chrome devtools Network tab on app.gtowizard.com, click through the practice UI for different position pairs, capture the URL/POST params (likely a hidden `shortcut_id=<int>` or `positions=...` field), then rewrite `scripts/gtow_trainer_url.py:build_trainer_url` to emit position-accurate URLs.
- **Why:** Closes the loop from "leak found" to "drilling the exact spot" in one tap
- **Effort:** CC ~30 min once access is set up
- **Risk:** Low — isolated URL builder change
- **Depends on:** Weekly Report v2 shipped
- **Gate:** Do this once URL imprecision becomes the most-complained-about issue in reports

### `raw_solver_snippet` JSONB column on deviations
Add a column to `deviations` table that stores the slice of `next_actions` response needed for future re-processing (per-action EVs + frequencies for hero's hand). Means future analytics iterations never need to re-hit GTOW API or the cache.
- **Why:** Future-proof — today's backfill works because gto_api_cache hits, but a schema/solver change could invalidate cache. Embedding the data in deviations makes it permanent.
- **Effort:** CC ~15 min — one migration + one write at insertion point
- **Risk:** Low — additive column
- **Depends on:** Weekly Report v2 shipped (can be a follow-up PR)

### Cross-week cluster recurrence detection
When the same cluster key appears in consecutive weekly reports with similar total_ev_loss, surface it as "recurring leak, not improving" in the report header. Also track "leak resolved" when a cluster drops out.
- **Why:** Motivational + accountability signal — "you've had this leak for 3 weeks now"
- **Effort:** CC ~20 min
- **Risk:** Low — additive query on leak_reports history
- **Depends on:** Weekly Report v2 shipped + 2-3 weeks of data

### Tilt detection v2 on ev_loss_rate
Current tilt detection uses deviation_rate moving window. Rebase on ev_loss_rate (cumulative -bb over last N decisions) to match the rest of v2's philosophy.
- **Why:** Ranking-consistency with main report, truer signal of "bleeding chips now"
- **Effort:** CC ~15 min
- **Risk:** Low — swap the metric in existing logic
- **Depends on:** Weekly Report v2 shipped + 1 week of ev_loss data

## P2 — Phase 2 (核心 pipeline 完成後)

### Meta-leak detection (問題模式分析)
追蹤使用者問問題的模式，識別概念性盲點 vs 執行錯誤。例如「你這個月問了 8 次 c-bet 決策 — 這可能是概念性的差距」。
- **Why:** 結合「怎麼打」和「怎麼問」的模式識別是最強大的長期功能
- **Effort:** CC ~30 min
- **Risk:** Medium — 依賴 LLM 問題分類可靠性
- **Depends on:** 核心 deviation pipeline
- **Files:** 新增 question_logs table 或在 message_logs 加 spot_category tag

### Spot challenge / 即時測驗
當檢測到 leak 時提供即時測驗，把被動報告變成主動學習。「你面對 c-bet OOP 常折疊。快速測試：Kh9h on Jc8c4d，對手 c-bet 66%。你怎麼做？」
- **Why:** 主動學習效果遠優於被動閱讀
- **Effort:** CC ~30 min
- **Risk:** Medium — 需要測驗生成邏輯 + 多步對話處理
- **Depends on:** 核心 deviation pipeline + leak detection

### Spot category 細化到 ~45 buckets
等有足夠資料後，將 ~15 core buckets 細化為含 board texture 的 ~45 buckets。
- **Why:** 更細粒度的分類讓 leak detection 更精確
- **Effort:** CC ~20 min
- **Risk:** Low
- **Depends on:** 核心 pipeline 運行 + 足夠資料量（每個 bucket ~20+ samples）

## P3 — Nice to Have

### Gamification (連勝追蹤 + 里程碑)
連勝追蹤、里程碑慶祝、「第一個洞察」特殊訊息。
- **Why:** 情感鉤子讓人持續使用
- **Effort:** CC ~15 min
- **Risk:** Low — 計數器邏輯在 deviations table 上
- **Depends on:** 核心 deviation pipeline
