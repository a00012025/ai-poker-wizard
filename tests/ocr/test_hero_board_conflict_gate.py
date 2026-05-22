"""Preflop false board detections must not rewrite correct hero cards."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_preflop_only_false_board_conflict_keeps_hero_cards():
    # Board detector sees a false 4h in the table region on this preflop
    # screenshot, while CardCNN correctly reads hero as 5h4h. The conflict
    # resolver used to swap the 4h to its top-2 rank (2h), corrupting a
    # correct hand even though no postflop street exists.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5846885329.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_hand"] == "5h4h"


def _hero_card_set(hero_hand: str) -> set[str]:
    return {hero_hand[i:i + 2] for i in range(0, len(hero_hand), 2)}


def test_false_undealt_turn_river_cards_do_not_rewrite_hero_cards():
    # These postflop hands have only flop/turn action evidence, but the raw
    # table card detector also sees false later-street cards that duplicate
    # the hero cards. Conflict resolution must ignore undealt slots instead
    # of rewriting the correct high-confidence CardCNN hero prediction.
    cases = {
        "TM5875583428": {"Kc", "4h"},  # false river 4h used to become 4d
        "TM5896602760": {"7d", "5s"},  # false turn 7d used to become 2d
        "TM5919864282": {"5h", "4d"},  # false river 4d used to become 4h
        "TM5920466966": {"Ts", "4h"},  # false river 4h used to become 2h
    }

    for hand_id, expected_cards in cases.items():
        result = parse_n8_screenshot(
            Path(f"data/hand_images/img/{hand_id}.png").read_bytes()
        )
        assert result["hand"] is not None, hand_id
        assert _hero_card_set(result["hand"]["hero_hand"]) == expected_cards


def test_masked_suit_pass_does_not_override_stronger_raw_suit():
    # A false/over-aggressive WIN mask can reduce suit confidence and flip a
    # correct raw spade to a club. Keep the masked pass only when it is not
    # materially weaker than the raw suit prediction.
    cases = {
        "TM5913549379": {"Js", "4c"},
        "TM5947067895": {"7s", "4s"},
    }

    for hand_id, expected_cards in cases.items():
        result = parse_n8_screenshot(
            Path(f"data/hand_images/img/{hand_id}.png").read_bytes()
        )
        assert result["hand"] is not None, hand_id
        assert _hero_card_set(result["hand"]["hero_hand"]) == expected_cards
