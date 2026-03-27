# TODOS

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
