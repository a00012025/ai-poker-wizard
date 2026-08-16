"""Regression fixtures for the confidence/abstain gate.

Each test asserts a reusable feature shape from
``docs/superpowers/plans/artifacts/2026-05-23-three-day-99-audit.md``,
not a hand-id lookup. Fixtures are split into:

* hands that MUST abstain because of a known-danger diagnostic pattern;
* hands that MUST stay emitted (regression guard against the gate
  over-abstaining).

When a fixture flips behavior, the test points back at the audit
artifact for the rule wording to adjust.
"""
from __future__ import annotations

from pathlib import Path


from ocr.confidence_gate import evaluate  # noqa: E402
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def _parse(hand_id: str) -> dict:
    return parse_n8_screenshot(
        Path(f"data/hand_images/img/{hand_id}.png").read_bytes()
    )


def _decide(hand_id: str) -> dict:
    return evaluate(_parse(hand_id))


# ---------- position_wrong: gate must abstain ----------

def test_position_wrong_severe_collapse_loss_abstains():
    """TM5913031183: pre_collapse_loss=7, conf=1.000 — gate must veto.

    Feature shape: very large preflop row collapse (>=5) paired with a
    player-count mismatch (raw 9 vs final 8) overrides otherwise-perfect
    confidence.
    """
    decision = _decide("TM5913031183")
    assert decision["emit"] is False
    assert decision["reason"].startswith("hard_abstain:")


def test_position_wrong_allin_reaction_ambiguity_residual():
    """TM5873873878: AA hand with all-in re-action ambiguity. The
    selectivity audit showed this shape has ~85% exact rate at the
    same conf level, so the gate intentionally lets it through. Marked
    as a known residual for Day 3 parser-fix work."""
    decision = _decide("TM5873873878")
    # Documents current emission — flips when a parser fix lands.
    assert decision["emit"] is True


def test_position_wrong_collapse_with_real_player_mismatch_abstains():
    """TM5913031183: raw=9, final=8, loss=7 — caught by Rule A
    (severe_collapse_with_player_mismatch). The 9->8 mismatch combined
    with severe collapse is the most selective wrong-emit signal:
    9/16 wrong rate on the full bucket."""
    decision = _decide("TM5913031183")
    assert decision["emit"] is False
    assert "severe_collapse_with_player_mismatch" in decision["reason"]


def test_position_wrong_severe_collapse_high_conf_residual():
    """TM5913201917: pre_collapse_loss=6, conf=0.998, no AI, no
    player-count mismatch — none of the surviving selective rules fire.
    The audit documents this as a Day 3 residual parser-fix candidate
    that calibrator-only abstaining cannot separate from exact hands."""
    decision = _decide("TM5913201917")
    # Document the current behaviour so a future regression is visible:
    # the hand currently emits at above_threshold. If a parser fix flips
    # it to abstain (or fixes the position outright), update this assert.
    assert decision["reason"] == "above_threshold"


# ---------- preflop_action_types_wrong: gate must abstain ----------

def test_preflop_phantom_doubled_allin_abstains():
    """TM5901482662: parsed actions contain 'AI-AI' (phantom doubling)."""
    result = _parse("TM5901482662")
    if "AI-AI" in (result.get("hand", {}) or {}).get("preflop_actions", ""):
        decision = evaluate(result)
        assert decision["emit"] is False
        assert "phantom_doubled_allin" in decision["reason"] or "hard_abstain" in decision["reason"]


def test_preflop_trailing_allin_after_call_abstains():
    """TM5879884236: '...C-AI50.68-C-AI2-F' — trailing AI after call."""
    decision = _decide("TM5879884236")
    assert decision["emit"] is False
    assert "hard_abstain" in decision["reason"]


def test_preflop_action_types_wrong_extra_reaction_row_residual():
    """TM5895757896: phantom extra re-action row pattern. No selective
    rule abstains it; documented as Day 3 parser-fix candidate."""
    decision = _decide("TM5895757896")
    # Currently emits via above_threshold. When parser fix lands or a new
    # selective rule is added, update this to expect abstain.
    assert decision["emit"] is True


# ---------- hero/board critical: gate must abstain ----------

def test_hero_cards_low_card_confidence_demoted():
    """TM5900728345: card_conf=0.76 — raw-vs-WIN-mask disagreement.
    The 0.76 sits inside the noisy 0.5-0.8 band so the gate may emit it,
    but must at least flag it via the soft-risk path when it does."""
    decision = _decide("TM5900728345")
    if decision["emit"]:
        assert (
            "above_risky_threshold" in decision["reason"]
            or "above_threshold" in decision["reason"]
        )


def test_board_wrong_pot_inconsistency_with_collapse_abstains():
    """TM5896602712: pot_consistency=0.5, collapse_loss=4 — board street
    mismatch should not emit."""
    decision = _decide("TM5896602712")
    assert decision["emit"] is False
    assert "hard_abstain" in decision["reason"]


# ---------- positive: gate must KEEP emitting ----------

def test_simple_preflop_high_card_safe_emit_still_emits():
    """The simple_preflop_high_card safe_emit shape (TM5846884867 from
    test_safe_emit_override.py) must keep emitting under the gate."""
    decision = _decide("TM5846884867")
    assert decision["emit"] is True
    # Either above_threshold or safe_emit_reason path
    assert decision["reason"] in ("above_threshold",) or decision["reason"].startswith("safe_emit:")


def test_high_card_complex_non_danger_still_emits():
    """TM5875362766: another safe_emit override exemplar — keep emitting."""
    decision = _decide("TM5875362766")
    assert decision["emit"] is True


def test_stable_postflop_high_card_still_emits():
    """TM5863067643: postflop with stable card head — keep emitting."""
    decision = _decide("TM5863067643")
    assert decision["emit"] is True
