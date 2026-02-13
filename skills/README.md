# AI Poker Wizard Skill

這個 skill 讓 Claude 能夠進行專業的撲克錦標賽分析，結合 GTO Wizard 求解器數據和專家級指導。

## 安裝

### 方法 1: 複製到 Claude Code Skills 目錄
```bash
cp -r ./skills/ai-poker-wizard ~/.claude/skills/
```

### 方法 2: 在對話中直接引用
在 Claude Code 中使用：
```
使用 ai-poker-wizard skill 分析這手牌：
"Hero 17bb effective, UTG raise 2bb..."
```

## 功能特色

✅ **精確的 GTO 求解器數據** - 來自 GTO Wizard MTT Premium
✅ **專業中文教練分析** - 適合華語撲克學習者
✅ **錦標賽 ICM 考量** - 包含 chip EV 和位置分析
✅ **169 手牌映射驗證** - 確認的手牌策略矩陣
✅ **瀏覽器自動化** - 自動查詢和數據提取

## 使用範例

### 手牌分析
```
分析這手牌：Hero 42bb effective, UTG+1 raise 2bb, hero SB all-in A9s
```

### GTO 策略查詢
```
17bb 深度，SB 面對 UTG raise + BTN call，哪些牌可以 all-in？
```

### 錦標賽決策
```
分析這個錦標賽決策點，需要考慮 ICM...
```

## 技術架構

Skill 整合了完整的 ai-poker-wizard 項目：
- **解析器**: LLM 驅動的手牌描述解析
- **GTO 控制器**: agent-browser 自動化查詢
- **專業教練**: 中文策略指導和 ICM 分析
- **數據驗證**: 已驗證的求解器數據提取

## 開發者資訊

完整項目源碼在 `/src/` 目錄，包含：
- 手牌數據模型 (Pydantic)
- GTO Wizard 瀏覽器控制器
- LLM 撲克教練分析引擎
- Telegram Bot 介面
- 核心整合模組

適合撲克學習者、錦標賽選手和策略研究。