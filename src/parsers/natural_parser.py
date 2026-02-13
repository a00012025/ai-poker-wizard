# src/parsers/natural_parser.py
import re
from typing import List
from src.models.hand_models import Hand, Action

class NaturalLanguageParser:
    def parse(self, text: str) -> Hand:
        # Extract effective stack
        stack_match = re.search(r'(\d+)bb effective', text)
        effective_stack = float(stack_match.group(1)) if stack_match else 0.0

        # Extract hero position
        hero_pos_match = re.search(r'hero (\w+) raise', text, re.IGNORECASE)
        hero_position = hero_pos_match.group(1).upper() if hero_pos_match else "UNKNOWN"

        # Extract flop
        flop_match = re.search(r'Flop ([2-9AKQJT][hdsc][2-9AKQJT][hdsc][2-9AKQJT][hdsc])', text)
        flop = flop_match.group(1) if flop_match else None

        # Extract pot size
        pot_match = re.search(r'pot (\d+)bb', text)
        pot_size = float(pot_match.group(1)) if pot_match else None

        # For now, create minimal actions list
        actions = []

        return Hand(
            hero_position=hero_position,
            effective_stack=effective_stack,
            actions=actions,
            flop=flop,
            pot_size=pot_size
        )
