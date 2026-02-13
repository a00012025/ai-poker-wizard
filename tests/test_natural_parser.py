# tests/test_natural_parser.py
import pytest
from src.parsers.natural_parser import NaturalLanguageParser
from src.models.hand_models import Hand

def test_parse_hero_hand():
    parser = NaturalLanguageParser()
    text = """Hero 42bb effective
Utg +1 raise 2bb, Lj call, co call, hero sb raise 10bb Ad9d, +1 fold, Lj fold, co call
Flop AcJc7h, pot 26bb, hero has 32bb behind
Hero check co bet 8bb hero all in 32bb co fold"""

    hand = parser.parse(text)
    assert hand.hero_position == "SB"
    assert hand.effective_stack == 42.0
    assert hand.flop == "AcJc7h"
    assert hand.pot_size == 26.0
