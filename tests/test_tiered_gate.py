"""Tests for the tiered confidence gate in gemini_session + audit helpers."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import pytest


# ── _cards_disagreement ──

def test_disagreement_returns_none_when_hero_and_boards_match():
    from gemini_session import GeminiSessionManager
    ocr = {
        "hero_hand": "AcKd",
        "streets": [{"board": "2h3s4d"}, {"card": "5c"}, {"card": "6h"}],
    }
    gemini = {
        "hero_hand": "AcKd",
        "streets": [{"board": "2h3s4d"}, {"card": "5c"}, {"card": "6h"}],
    }
    assert GeminiSessionManager._cards_disagreement(ocr, gemini) is None


def test_disagreement_flags_hero_diff():
    from gemini_session import GeminiSessionManager
    ocr = {"hero_hand": "AcKd", "streets": []}
    gemini = {"hero_hand": "AcKh", "streets": []}
    d = GeminiSessionManager._cards_disagreement(ocr, gemini)
    assert d == {"hero": {"ocr": "AcKd", "gemini": "AcKh"}}


def test_disagreement_flags_street_diff():
    from gemini_session import GeminiSessionManager
    ocr = {"hero_hand": "AcKd", "streets": [{"board": "2h3s4d"}, {"card": "5c"}]}
    gemini = {"hero_hand": "AcKd", "streets": [{"board": "2h3s4d"}, {"card": "5h"}]}
    d = GeminiSessionManager._cards_disagreement(ocr, gemini)
    assert d == {"streets": {"1": {"ocr": "5c", "gemini": "5h"}}}


def test_disagreement_flags_street_count_mismatch():
    """One side has more streets (turn+river vs just turn) — difference
    should be captured, not silently tolerated."""
    from gemini_session import GeminiSessionManager
    ocr = {"hero_hand": "AcKd", "streets": [{"board": "2h3s4d"}]}
    gemini = {"hero_hand": "AcKd", "streets": [{"board": "2h3s4d"}, {"card": "5h"}]}
    d = GeminiSessionManager._cards_disagreement(ocr, gemini)
    assert d == {"streets": {"1": {"ocr": "", "gemini": "5h"}}}


def test_disagreement_handles_empty_hands():
    from gemini_session import GeminiSessionManager
    assert GeminiSessionManager._cards_disagreement({}, {}) is None


# ── audit_labels flatten helpers ──

def test_board_strings_flattens_flop_turn_river():
    from ocr.classifier.audit_labels import _board_strings
    hand = {"streets": [
        {"board": "2h3s4d"},
        {"card": "5c"},
        {"card": "6h"},
    ]}
    assert _board_strings(hand) == ["2h", "3s", "4d", "5c", "6h"]


def test_board_strings_handles_missing_streets():
    from ocr.classifier.audit_labels import _board_strings
    assert _board_strings({}) == []
    assert _board_strings({"streets": []}) == []
    assert _board_strings({"streets": [{"actions": []}]}) == []


def test_board_strings_supports_partial_boards():
    from ocr.classifier.audit_labels import _board_strings
    # Folded preflop hand — no streets with cards
    hand_folded = {"streets": [{"actions": []}]}
    assert _board_strings(hand_folded) == []
    # Flop-only
    hand_flop = {"streets": [{"board": "2h3s4d"}]}
    assert _board_strings(hand_flop) == ["2h", "3s", "4d"]
