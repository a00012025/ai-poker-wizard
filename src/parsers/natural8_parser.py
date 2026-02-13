# src/parsers/natural8_parser.py
from typing import List, Optional
from src.models.hand_models import Hand, Action

class Natural8Parser:
    def parse_file(self, file_path: str) -> List[dict]:
        """Parse Natural8 tournament file and return list of hands"""
        hands = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by hand boundaries
                hand_sections = content.split('Hand #')
                for section in hand_sections[1:]:  # Skip empty first split
                    if section.strip():
                        hands.append({
                            'hand_id': f"#{section.split()[0]}",
                            'raw_text': f"Hand #{section}"
                        })
        except FileNotFoundError:
            pass
        return hands

    def find_hand(self, file_path: str, hand_id: str) -> Optional[dict]:
        """Find specific hand by ID in tournament file"""
        hands = self.parse_file(file_path)
        for hand in hands:
            if hand['hand_id'] == hand_id:
                return hand
        return None