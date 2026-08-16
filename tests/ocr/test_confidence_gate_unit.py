"""Pure-function unit tests for the confidence_gate rules.

These tests do not invoke the OCR pipeline — they feed synthetic
``parser_output`` dicts to ``evaluate`` so every rule is exercised in
isolation and remains exercised even when an OCR fixture moves.
"""
from __future__ import annotations



from ocr.confidence_gate import evaluate  # noqa: E402


def _output(
    *,
    confidence: float = 0.95,
    parts: dict | None = None,
    diag: dict | None = None,
    actions: str = "F-R2-F-F-F-F-F-F",
    safe_emit: str | None = None,
) -> dict:
    return {
        "hand": {"preflop_actions": actions},
        "confidence": confidence,
        "confidence_parts": parts
        or {
            "pot_consistency": 1.0,
            "player_tracking": 1.0,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        "diagnostics": diag
        or {
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 8,
            "players_at_table_raw": 8,
            "players_at_table_final": 8,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.0,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        "safe_emit_reason": safe_emit,
    }


# ---------- Hard rule activation ----------

def test_rule_severe_collapse_with_player_mismatch_abstains_when_no_button():
    # Hard rule A: severe collapse + player-count mismatch + no button
    out = _output(diag={
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 13,  # loss=5
        "players_at_table_raw": 9,
        "players_at_table_final": 8,
        "estimate_used_reaction_signal": False,
        "dealer_button_conf": 0.0,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    })
    d = evaluate(out)
    assert d["emit"] is False
    assert "severe_collapse_with_player_mismatch" in d["reason"]


def test_rule_severe_collapse_bypassed_when_button_strong():
    out = _output(diag={
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 13,  # loss=5
        "players_at_table_raw": 9,
        "players_at_table_final": 8,
        "estimate_used_reaction_signal": False,
        "dealer_button_conf": 0.99,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    })
    d = evaluate(out)
    assert d["emit"] is True


def test_soft_risk_weak_tracking_with_allin_moderate_collapse_emits_by_default():
    # Soft-risk path is disabled by default (risky_emit_threshold=0.0).
    # Calibrator analysis showed the available features cannot reach
    # 99%@70% even with logistic regression, so the soft-risk gate is
    # opt-in.
    out = _output(
        confidence=0.90,
        parts={
            "pot_consistency": 1.0,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        diag={
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 12,  # loss=4
            "players_at_table_raw": 8,
            "players_at_table_final": 8,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.0,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        actions="F-F-F-F-AI11.58-F",
    )
    d = evaluate(out)
    assert d["emit"] is True


def test_soft_risk_explicit_threshold_abstains_when_opted_in():
    # When the caller opts in via risky_emit_threshold=0.95, the soft-
    # risk shape demotes the 0.90-conf hand.
    out = _output(
        confidence=0.90,
        parts={
            "pot_consistency": 1.0,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        diag={
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 12,  # loss=4
            "players_at_table_raw": 8,
            "players_at_table_final": 8,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.0,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        actions="F-F-F-F-AI11.58-F",
    )
    d = evaluate(out, risky_emit_threshold=0.95)
    assert d["emit"] is False
    assert "risky_below_threshold" in d["reason"]


def test_soft_risk_skipped_when_safe_emit_set():
    # Safe-emit reason indicates the parser already trusted the shape,
    # so the soft-risk gate is bypassed.
    out = _output(
        confidence=0.85,
        parts={
            "pot_consistency": 1.0,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        diag={
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 12,  # loss=4
            "players_at_table_raw": 8,
            "players_at_table_final": 8,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.0,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        actions="F-F-F-F-AI11.58-F",
        safe_emit="high_card_complex_non_danger",
    )
    d = evaluate(out)
    assert d["emit"] is True
    assert d["reason"].startswith("safe_emit:")


def test_soft_risk_skipped_for_high_collapse_no_allin():
    # Selectivity audit: weak-tracking-no-AI shape has ~94% exact rate,
    # so the gate intentionally lets it emit.
    out = _output(
        confidence=0.90,
        parts={
            "pot_consistency": 0.5,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        diag={
            "preflop_entries_count": 5,
            "preflop_entries_pre_collapse_count": 12,  # loss=7
            "players_at_table_raw": 5,
            "players_at_table_final": 5,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.0,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        actions="F-F-F-F-F-F",  # no AI
    )
    d = evaluate(out)
    assert d["emit"] is True
    assert d["reason"] == "above_threshold"


def test_rule6_very_low_card_confidence_abstains_regardless_of_button():
    # Cutoff is 0.5 — the 0.5-0.8 band turned out to be ~50% exact on
    # the full bucket, so we only hard-abstain below 0.5.
    out = _output(
        parts={
            "pot_consistency": 1.0,
            "player_tracking": 1.0,
            "ocr_confidence": 1.0,
            "card_confidence": 0.45,
        },
        diag={
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 8,
            "players_at_table_raw": 8,
            "players_at_table_final": 8,
            "estimate_used_reaction_signal": False,
            "dealer_button_conf": 0.99,  # even with strong button
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
    )
    d = evaluate(out)
    assert d["emit"] is False
    assert "low_card_conf" in d["reason"]


def test_rule7_doubled_allin_tokens_abstain():
    out = _output(actions="R2-F-F-C-AI-AI-F")
    d = evaluate(out)
    assert d["emit"] is False
    assert "phantom_doubled_allin" in d["reason"]


def test_rule8_sizeless_allin_mid_sequence_no_longer_abstains():
    # Rule was removed: selectivity audit showed 4 exact vs 2 wrong, so
    # the rule cost more coverage than it bought precision.
    out = _output(actions="F-F-F-C-AI-F-AI50.5-F")
    d = evaluate(out)
    assert d["emit"] is True


def test_rule9_trailing_allin_after_call_abstains():
    out = _output(actions="R2-F-F-F-F-F-C-AI50.7-C-AI2-F")
    d = evaluate(out)
    assert d["emit"] is False
    assert "trailing_allin_after_call" in d["reason"]


def test_rule11_pot_inconsistent_postflop_collapse_abstains():
    out = _output(parts={
        "pot_consistency": 0.5,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 1.0,
    }, diag={
        "preflop_entries_count": 5,
        "preflop_entries_pre_collapse_count": 9,  # loss=4
        "players_at_table_raw": 5,
        "players_at_table_final": 5,
        "estimate_used_reaction_signal": False,
        "dealer_button_conf": 0.0,
        "street_entries_count": {"flop": 2, "turn": 0, "river": 0},
    })
    d = evaluate(out)
    assert d["emit"] is False
    assert "pot_inconsistent_postflop_collapse" in d["reason"]


def test_rule11_does_not_fire_for_preflop_only_safe_emit_shape():
    # TM5875362766 profile: pot=0.5 loss=11 but no postflop entries — must
    # keep emitting via safe_emit / threshold path.
    out = _output(
        confidence=0.80,
        parts={
            "pot_consistency": 0.5,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 1.0,
        },
        diag={
            "preflop_entries_count": 8,
            "preflop_entries_pre_collapse_count": 19,  # loss=11
            "players_at_table_raw": 7,
            "players_at_table_final": 7,
            "estimate_used_reaction_signal": True,
            "dealer_button_conf": 0.99,  # strong button bypasses Rules 1-5,10
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        },
        actions="F-AI14.21-C-AI18.86-F-F-F-C",
        safe_emit="high_card_complex_non_danger",
    )
    d = evaluate(out)
    assert d["emit"] is True


# ---------- Emit paths ----------

def test_above_threshold_emits():
    out = _output(confidence=0.95)
    d = evaluate(out)
    assert d["emit"] is True
    assert d["reason"] == "above_threshold"


def test_below_threshold_with_safe_emit_emits():
    out = _output(confidence=0.80, safe_emit="simple_preflop_high_card")
    d = evaluate(out)
    assert d["emit"] is True
    assert d["reason"].startswith("safe_emit:")


def test_below_threshold_no_safe_emit_abstains():
    out = _output(confidence=0.80, safe_emit=None)
    d = evaluate(out)
    assert d["emit"] is False
    assert d["reason"] == "below_threshold"


def test_disable_hard_rules_replicates_legacy_behaviour():
    # A hand that would normally abstain via Rule 1 should emit when hard
    # rules are off.
    out = _output(confidence=0.95, diag={
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 13,  # loss=5
        "players_at_table_raw": 8,
        "players_at_table_final": 8,
        "estimate_used_reaction_signal": False,
        "dealer_button_conf": 0.0,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    })
    d = evaluate(out, enable_hard_rules=False)
    assert d["emit"] is True
