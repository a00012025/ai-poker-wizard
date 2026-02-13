# src/wizard_core.py
import asyncio
from typing import Dict, Any
from src.parsers.natural_parser import NaturalLanguageParser
from src.parsers.natural8_parser import Natural8Parser
from src.gto_automation.browser_controller import GTOWizardController
from src.analysis.poker_coach import PokerCoach

class PokerWizardCore:
    def __init__(self):
        self.natural_parser = NaturalLanguageParser()
        self.n8_parser = Natural8Parser()
        self.gto_controller = GTOWizardController()
        self.poker_coach = PokerCoach()

    async def analyze_hand_text(self, hand_text: str) -> Dict[str, Any]:
        """Complete analysis pipeline for text input"""
        try:
            print("🔍 正在解析手牌描述...")

            # Parse hand using LLM approach
            hand = self.natural_parser.parse(hand_text)

            print("🌐 正在查詢 GTO Wizard 策略...")

            # Query GTO Wizard
            gto_data = await self.gto_controller.query_scenario(
                position=hand.hero_position,
                stack_bb=hand.effective_stack,
                action_sequence=hand_text,
                flop=hand.flop or ""
            )

            print("🤖 正在生成專業分析...")

            # Get coaching analysis
            coaching_insights = self.poker_coach.analyze_hand(hand, gto_data)

            return {
                'hand_analysis': hand.model_dump(),
                'gto_data': gto_data,
                'coaching_insights': coaching_insights,
                'success': True,
                'summary': f"""
🎯 **分析完成！**

**手牌概況：**
• 位置：{hand.hero_position}
• 籌碼：{hand.effective_stack}bb
• 翻牌：{hand.flop or 'N/A'}

**GTO 建議：**
{coaching_insights.get('recommendations', '策略分析中...')[:500]}...

💡 **詳細分析請查看完整報告**
"""
            }
        except Exception as e:
            return {
                'error': f"分析失敗：{str(e)}",
                'success': False
            }

    async def analyze_n8_file(self, file_path: str, hand_id: str) -> Dict[str, Any]:
        """Analyze specific hand from Natural8 file"""
        try:
            hand_data = self.n8_parser.find_hand(file_path, hand_id)
            if not hand_data:
                return {'error': '找不到指定手牌', 'success': False}

            # Analyze the raw text
            return await self.analyze_hand_text(hand_data['raw_text'])
        except Exception as e:
            return {'error': f"檔案分析失敗：{str(e)}", 'success': False}