"""Phase 11.D-c — the solver-relevant board stops at the all-in decision
street. After a postflop all-in (called or not) there are no more decisions,
so the physical turn/river runout shown in the screenshot must NOT contribute
board cards. This was the dominant board_wrong cause (27/28 on the test set):
the parser read the dealt flop perfectly, then appended the visible runout.
"""
from __future__ import annotations



from ocr.n8_parser import _build_streets

POS = ["LJ", "HJ", "CO", "BTN", "SB", "BB"]


def _col(name, entries):
    return {"name": name, "entries": entries}


def test_flop_allin_drops_runout_turn_and_river():
    # Flop all-in + call; turn/river ran out (board detector saw 5 cards) but
    # there are no turn/river decisions → board must be flop only.
    street_cols = [
        _col("flop", [
            {"type": "hero", "action": "all-in", "size": 7.3},
            {"type": "opponent", "action": "call", "position": "BB"},
        ]),
        _col("turn", []),
        _col("river", []),
    ]
    board = ["5s", "Q h".replace(" ", ""), "Th", "7h", "2d"]  # flop + runout
    streets = _build_streets(street_cols, board, POS, hero_position="HJ",
                             active_positions=["HJ", "BB"])
    assert len(streets) == 1
    assert streets[0].get("board") == "5sQhTh"
    assert "card" not in streets[0]


def test_runout_dropped_even_if_panel_has_phantom_turn_entries():
    # Sometimes OCR finds spurious turn rows; an all-in on the flop still closes
    # the tree, so the turn street must be dropped regardless.
    street_cols = [
        _col("flop", [
            {"type": "opponent", "action": "all-in", "size": 10.0,
             "position": "SB"},
            {"type": "hero", "action": "call"},
        ]),
        _col("turn", [
            {"type": "hero", "action": "check"},
        ]),
    ]
    board = ["9s", "6d", "7s", "Kc"]
    streets = _build_streets(street_cols, board, POS, hero_position="BTN",
                             active_positions=["SB", "BTN"])
    assert len(streets) == 1
    assert streets[0]["board"] == "9s6d7s"


def test_normal_multiway_streets_are_kept():
    # No all-in: a legitimately-played turn with action must survive.
    street_cols = [
        _col("flop", [
            {"type": "hero", "action": "check"},
            {"type": "opponent", "action": "bet", "size": 2.0, "position": "BB"},
            {"type": "hero", "action": "call"},
        ]),
        _col("turn", [
            {"type": "hero", "action": "check"},
            {"type": "opponent", "action": "check", "position": "BB"},
        ]),
    ]
    board = ["5s", "Qh", "Th", "7h"]
    streets = _build_streets(street_cols, board, POS, hero_position="HJ",
                             active_positions=["HJ", "BB"])
    assert len(streets) == 2
    assert streets[0]["board"] == "5sQhTh"
    assert streets[1]["card"] == "7h"
