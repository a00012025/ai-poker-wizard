# src/analysis/poker_coach.py
import json
from typing import Dict, Any
from src.models.hand_models import Hand

class PokerCoach:
    def __init__(self):
        self.coaching_prompt = """你是一位專業的撲克錦標賽教練。
分析這個手牌並提供專業建議：

1. 手牌概況和關鍵決策點
2. GTO 策略對比分析
3. 範圍分析和 equity 考量
4. ICM 影響評估（如適用）
5. 具體改進建議

請用中文回答，提供實用的策略指導。"""

    def analyze_hand_with_llm(self, hand_text: str, gto_data: Dict[str, Any]) -> str:
        """Use Claude Code's LLM capability to analyze hand"""
        analysis_prompt = f"""
{self.coaching_prompt}

手牌描述：
{hand_text}

GTO Wizard 數據：
{json.dumps(gto_data, indent=2, ensure_ascii=False)}

請提供詳細的專業分析：
"""
        # In real implementation, this would call Claude API
        # For now, return structured mock analysis
        return f"""
📊 **手牌分析報告**

**概況**
- 場景：{gto_data.get('scenario', {}).get('position', 'Unknown')} 位置
- 有效籌碼：{gto_data.get('scenario', {}).get('stack_bb', 0)}bb
- 翻牌：{gto_data.get('scenario', {}).get('flop', 'N/A')}

**GTO 策略對比**
根據 GTO Wizard 數據：
- FOLD 頻率：{gto_data.get('action_solutions', [{}])[0].get('total_frequency', 0):.1%}
- CALL 頻率：{gto_data.get('action_solutions', [{}])[1].get('total_frequency', 0):.1%}
- ALL-IN 頻率：{gto_data.get('action_solutions', [{}])[2].get('total_frequency', 0):.1%}

**範圍分析**
基於當前籌碼深度和位置，建議的策略範圍...

**ICM 考量**
在錦標賽環境中，需要考慮...

**改進建議**
1. 位置意識：注意你在 {gto_data.get('scenario', {}).get('position', '')} 位置的優勢
2. 籌碼管理：{gto_data.get('scenario', {}).get('stack_bb', 0)}bb 的深度需要...
3. 對手讀牌：基於對手的行動序列...

**總結**
這個決策點的關鍵是平衡...
"""

    def analyze_hand(self, hand: Hand, gto_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze hand with professional coaching insights"""
        hand_text = f"""
Hero 位置：{hand.hero_position}
有效籌碼：{hand.effective_stack}bb
翻牌：{hand.flop or 'N/A'}
底池：{hand.pot_size or 0}bb
"""

        analysis = self.analyze_hand_with_llm(hand_text, gto_data)

        return {
            'summary': f'分析 {hand.hero_position} 位置手牌，有效籌碼 {hand.effective_stack}bb',
            'gto_comparison': '與 GTO 策略對比分析',
            'recommendations': analysis,
            'hand_data': hand.model_dump(),
            'gto_reference': gto_data
        }