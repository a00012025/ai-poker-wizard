# AI Poker Wizard - 設計文檔

## 概述
AI Poker Wizard 是一個結合 GTO Wizard MTT 數據和 LLM 分析的撲克錦標賽學習工具，主要用於手牌復盤和策略分析。

## 核心功能
- 手牌解析：支援自然語言描述、結構化格式、Natural8 歷史檔案
- GTO 數據擷取：透過瀏覽器自動化與 GTO Wizard 互動
- AI 教練分析：Claude 作為專業撲克教練提供全面分析
- Telegram 整合：便利的對話式介面

## 技術架構
**方案A - Claude Code 中心化**
- Python 腳本處理解析和自動化
- Playwright 進行瀏覽器控制
- Telegram Bot 作為前端介面
- Claude Code 作為分析引擎

## 目標用戶體驗
1. 用戶透過 Telegram 輸入手牌或上傳 N8 檔案
2. 系統自動查詢 GTO Wizard 獲取策略數據
3. AI 提供職業級別的復盤分析和建議
4. 支援對話式問答釐清問題

## 未來擴展
- 視覺化圖表和範圍顯示
- AI 自我增強能力（搜尋策略文章、學習手牌、追蹤進步）