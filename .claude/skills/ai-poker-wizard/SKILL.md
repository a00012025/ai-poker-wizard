---
name: ai-poker-wizard
description: Use when analyzing poker hands, tournament decisions, or GTO strategy questions. Triggers on poker terminology (bb, SB, UTG, all-in, ranges, equity) or requests for hand analysis, strategy advice, or tournament coaching.
---

# AI Poker Wizard

## Overview

Professional poker tournament analysis using GTO Wizard solver data and expert coaching. Combines browser automation to extract precise GTO strategies with LLM-powered analysis for comprehensive hand reviews and strategic guidance.

## When to Use

**Triggers:**
- Poker hand descriptions with positions, stack sizes, actions
- Questions about GTO strategy, ranges, or frequencies
- Tournament decision analysis and ICM considerations
- Hand history review and coaching requests
- Stack depth strategy questions (bb terminology)
- Terms: SB, BB, UTG, BTN, raise, call, fold, all-in, shove

**Perfect for:**
- "Hero 17bb effective, UTG raise 2bb, should I shove A9s?"
- "What's the GTO frequency for this spot?"
- "Analyze this hand from my tournament"
- Natural8 hand history file analysis

## Core Pattern

```python
# Complete analysis pipeline
async def analyze_poker_hand(hand_description: str) -> str:
    # 1. Parse hand using LLM
    # 2. Query GTO Wizard with agent-browser
    # 3. Extract precise strategy data
    # 4. Generate expert coaching analysis
    # 5. Return comprehensive Chinese report
```

## Implementation

Use the complete ai-poker-wizard codebase at `/Users/cbd/opensource/me/ai-poker-wizard/`:

### Quick Analysis
```python
from src.wizard_core import PokerWizardCore

wizard = PokerWizardCore()
result = await wizard.analyze_hand_text(hand_description)
print(result['summary'])
```

### Manual GTO Query
```bash
# Use agent-browser for direct queries
agent-browser open "https://app.gtowizard.com/solutions"
agent-browser snapshot -i
# Navigate to scenario and extract JSON data
```

### Key Components
- **Parser**: LLM-based hand description parsing
- **GTO Controller**: agent-browser automation with cookie persistence
- **Coach**: Professional analysis in Chinese with ICM considerations
- **Hand Mapping**: 169-position strategy matrix interpretation

## Verified Data Extraction

**GTO Wizard JSON Structure:**
```json
{
  "action_solutions": [
    {"action": {"code": "F", "display_name": "FOLD"}, "total_frequency": 0.819, "strategy": [...]},
    {"action": {"code": "C", "display_name": "CALL"}, "total_frequency": 0.091, "strategy": [...]},
    {"action": {"code": "RAI", "display_name": "ALLIN"}, "total_frequency": 0.090, "strategy": [...]}
  ]
}
```

**Confirmed Hand Mappings:**
- Index 80 → AA, Index 81 → AJo, Index 82 → AJs
- Index 89 → A9s (0% all-in frequency in SB vs UTG+BTN)
- 169 total positions covering all poker hand combinations

## Analysis Framework

**Professional Coaching Report:**
1. **手牌概況** - Situation summary with stack/position context
2. **GTO 策略對比** - Exact frequencies from solver data
3. **範圍分析** - Range vs range equity calculations
4. **ICM 考量** - Tournament context and chip EV implications
5. **改進建議** - Specific actionable recommendations

**Example Output:**
```
🎯 **GTO 分析結果**

**場景**: SB vs UTG raise + BTN call, 17bb effective
**手牌**: A9s

**GTO 頻率**:
- FOLD: 82%
- ALL-IN: 0%
- 建議動作: FOLD

**分析**: A9s 在此深度面對 UTG+BTN 範圍沒有足夠 equity 支撐 all-in...
```

## Common Mistakes

**❌ Wrong:** Using general poker knowledge without solver verification
**✅ Right:** Extract actual GTO Wizard data for precise frequencies

**❌ Wrong:** English analysis for Chinese-speaking users
**✅ Right:** Professional Chinese coaching terminology and explanations

**❌ Wrong:** Ignoring stack depth and position dynamics
**✅ Right:** Context-aware analysis with ICM and tournament considerations

## Real-World Impact

- Precise GTO frequencies vs general estimates
- Tournament-specific coaching in native language
- Verified solver data for strategic decisions
- Professional improvement through accurate analysis