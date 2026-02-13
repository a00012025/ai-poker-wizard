# AI Poker Wizard - 設計文檔

## 概述
AI Poker Wizard 是一個結合 GTO Wizard MTT 數據和 LLM 分析的撲克錦標賽學習工具，主要用於手牌復盤和策略分析。

## 核心功能
- **手牌解析**：支援自然語言描述、結構化格式、Natural8 歷史檔案
- **GTO 數據擷取**：透過瀏覽器自動化與 GTO Wizard 互動，已驗證技術可行性
- **AI 教練分析**：Claude 作為專業撲克教練提供全面分析
- **Telegram 整合**：便利的對話式介面

## 技術架構
**方案A - Claude Code 中心化** ✅
- Python 腳本處理解析和自動化
- Playwright 進行瀏覽器控制
- Telegram Bot 作為前端介面
- Claude Code 作為分析引擎

## 已驗證的技術發現

### GTO Wizard 數據提取
✅ **瀏覽器自動化**：使用 agent-browser 成功導航和數據捕獲
✅ **網路數據結構**：確認 JSON API 格式和策略矩陣結構
✅ **手牌映射邏輯**：169 位置策略陣列，已確認映射規律
✅ **策略頻率解析**：能準確提取 FOLD/CALL/ALLIN 頻率

### 核心數據格式
```json
{
  "action_solutions": [
    {"action": {"code": "F", "display_name": "FOLD"}, "total_frequency": 0.819, "strategy": [...]},
    {"action": {"code": "C", "display_name": "CALL"}, "total_frequency": 0.091, "strategy": [...]},
    {"action": {"code": "RAI", "display_name": "ALLIN"}, "total_frequency": 0.090, "strategy": [...]}
  ]
}
```

### 手牌映射規律
確認的映射點（ASCII 排序）：
- Index 80 → AA，Index 81 → AJo，Index 82 → AJs
- Index 83 → AKo，Index 84 → AKs，Index 85 → AQo
- Index 86 → AQs，Index 87 → ATo，Index 88 → A9o，Index 89 → A9s

### 策略分析能力
✅ **混合策略**：準確解析混合頻率（如 97% all-in, 3% fold）
✅ **多動作分析**：同時處理 FOLD/CALL/RAISE/ALLIN 策略
✅ **場景匹配**：驗證不同籌碼深度和動作序列的準確性

## 目標用戶體驗
1. **輸入手牌**：用戶透過 Telegram 輸入自然語言描述或上傳 Natural8 檔案
2. **自動查詢**：系統自動解析手牌並查詢 GTO Wizard 獲取策略數據
3. **專業分析**：AI 教練提供職業級別的復盤分析，包含：
   - 手牌決策點分析
   - GTO 策略對比
   - 範圍和 equity 考量
   - ICM 影響評估
   - 具體改進建議
4. **對話式學習**：支援對話式問答釐清問題並深入討論

## 實作策略

### 第一階段：核心功能
- 手牌數據模型和解析器
- GTO Wizard 瀏覽器自動化
- LLM 分析引擎
- Telegram Bot 介面

### 第二階段：增強功能
- 策略緩存優化
- 批量手牌分析
- 檔案上傳處理
- 錯誤處理和重試機制

### 第三階段：進階功能
- 視覺化圖表和範圍顯示
- AI 自我增強能力（搜尋策略文章、學習手牌、追蹤進步）
- 手牌歷史追蹤和進步分析

## 技術風險與解決方案

### 瀏覽器自動化穩定性
- **風險**：GTO Wizard 網站變更可能影響數據擷取
- **解決**：模組化設計，易於更新；添加錯誤處理和重試邏輯

### 手牌映射準確性
- **風險**：手牌到策略矩陣位置映射錯誤
- **解決**：已驗證映射邏輯；添加測試案例確保準確性

### API 速率限制
- **風險**：過度查詢可能被限制訪問
- **解決**：實作緩存機制；添加請求間隔控制

## 成功指標
- ✅ 技術可行性驗證完成
- 🎯 手牌解析準確率 >95%
- 🎯 GTO 數據擷取成功率 >90%
- 🎯 用戶滿意度分析質量