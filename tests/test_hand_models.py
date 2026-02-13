# tests/test_hand_models.py
import pytest
from src.models.hand_models import Hand, Action

def test_hand_model_creation():
    hand = Hand(
        hero_position="SB",
        effective_stack=42,
        actions=[
            Action(position="UTG+1", action="raise", amount=2),
            Action(position="LJ", action="call"),
            Action(position="CO", action="call"),
            Action(position="SB", action="raise", amount=10, cards="Ad9d")
        ],
        flop="AcJc7h",
        pot_size=26
    )
    assert hand.hero_position == "SB"
    assert hand.effective_stack == 42
    assert len(hand.actions) == 4
    assert hand.flop == "AcJc7h"