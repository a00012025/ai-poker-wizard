"""Phase 11.B.4 — v2 feature vector exposes the post-Phase-10 signals
the v1 27-feature vector lacks: ensemble usage/strength, per-card top-2
margins, rank source flags, raw-vs-masked suit swaps, and the
demote-to-Gemini structural-collapse flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.confidence_gate import _calibrator_features_v2, _V2_FEATURE_NAMES


def _stub_parser_output(
    *,
    ensemble_used=False,
    ensemble_votes=None,
    rank_source_0="classifier",
    rank_source_1="classifier",
    raw_suit_0="d", masked_suit_0="d",
    rank_top2_0=None, suit_top2_0=None,
    rank_top2_1=None, suit_top2_1=None,
    pre_collapse=None, post_collapse=None,
) -> dict:
    rank_top2_0 = rank_top2_0 or [("5", 0.90), ("3", 0.05)]
    suit_top2_0 = suit_top2_0 or [("d", 0.95), ("h", 0.03)]
    rank_top2_1 = rank_top2_1 or [("6", 0.85), ("8", 0.10)]
    suit_top2_1 = suit_top2_1 or [("d", 0.92), ("c", 0.05)]
    diag = {
        "ensemble_used": ensemble_used,
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 8,
        "players_at_table_raw": 8,
        "players_at_table_final": 8,
        "dealer_button_conf": 0.9,
        "estimate_used_reaction_signal": False,
        "street_entries_count": post_collapse or {"flop": 3, "turn": 2},
        "street_entries_pre_collapse_count": pre_collapse or {"flop": 3, "turn": 2},
    }
    details = [
        {
            "rank": rank_top2_0[0][0], "rank_conf": rank_top2_0[0][1],
            "suit": suit_top2_0[0][0], "suit_conf": suit_top2_0[0][1],
            "rank_top2": rank_top2_0, "suit_top2": suit_top2_0,
            "rank_source": rank_source_0,
            "raw_suit": raw_suit_0, "raw_suit_conf": 0.7,
            "masked_suit": masked_suit_0, "masked_suit_conf": suit_top2_0[0][1],
            "conf": min(rank_top2_0[0][1], suit_top2_0[0][1]),
            "ensemble_votes": ensemble_votes or [],
            "ensemble_conf": 0.9 if ensemble_used else 0.0,
        },
        {
            "rank": rank_top2_1[0][0], "rank_conf": rank_top2_1[0][1],
            "suit": suit_top2_1[0][0], "suit_conf": suit_top2_1[0][1],
            "rank_top2": rank_top2_1, "suit_top2": suit_top2_1,
            "rank_source": rank_source_1,
            "raw_suit": "d", "raw_suit_conf": 0.7,
            "masked_suit": "d", "masked_suit_conf": suit_top2_1[0][1],
            "conf": min(rank_top2_1[0][1], suit_top2_1[0][1]),
            "ensemble_votes": [],
            "ensemble_conf": 0.0,
        },
    ]
    return {
        "confidence": 0.9,
        "confidence_parts": {
            "pot_consistency": 1.0, "player_tracking": 1.0,
            "ocr_confidence": 0.9, "card_confidence": 0.85,
        },
        "hand": {"preflop_actions": "F-F-F-R2-F-F-C"},
        "diagnostics": diag,
        "hero_card_details": details,
        "safe_emit_reason": "",
    }


def test_v2_feature_count_matches_schema():
    feats = _calibrator_features_v2(_stub_parser_output())
    assert len(feats) == len(_V2_FEATURE_NAMES) == 40


def test_ensemble_used_flag_propagates():
    feats = _calibrator_features_v2(_stub_parser_output(ensemble_used=True))
    idx = _V2_FEATURE_NAMES.index("ensemble_used")
    assert feats[idx] == 1.0


def test_top2_margin_is_diff_between_top_and_runner_up():
    out = _stub_parser_output(
        rank_top2_0=[("5", 0.80), ("3", 0.15)],
    )
    feats = _calibrator_features_v2(out)
    idx = _V2_FEATURE_NAMES.index("hero0_rank_top2_margin")
    assert abs(feats[idx] - (0.80 - 0.15)) < 1e-9


def test_rank_source_corner_flag():
    out = _stub_parser_output(rank_source_0="corner_ocr")
    feats = _calibrator_features_v2(out)
    idx = _V2_FEATURE_NAMES.index("hero0_rank_source_is_corner")
    assert feats[idx] == 1.0


def test_raw_vs_masked_suit_swap_flag():
    out = _stub_parser_output(raw_suit_0="h", masked_suit_0="d")
    feats = _calibrator_features_v2(out)
    idx = _V2_FEATURE_NAMES.index("hero_raw_vs_masked_suit_swapped")
    assert feats[idx] == 1.0


def test_demote_to_gemini_fires_on_large_postflop_collapse():
    out = _stub_parser_output(
        pre_collapse={"flop": 3, "turn": 2, "river": 10},
        post_collapse={"flop": 3, "turn": 2, "river": 3},  # loss = 7
    )
    feats = _calibrator_features_v2(out)
    idx = _V2_FEATURE_NAMES.index("demote_to_gemini_fired")
    assert feats[idx] == 1.0


def test_demote_not_fired_when_loss_within_threshold():
    out = _stub_parser_output(
        pre_collapse={"flop": 3, "turn": 5, "river": 4},
        post_collapse={"flop": 3, "turn": 2, "river": 3},  # max loss = 3
    )
    feats = _calibrator_features_v2(out)
    idx = _V2_FEATURE_NAMES.index("demote_to_gemini_fired")
    assert feats[idx] == 0.0


def test_v1_features_preserved_in_v2_prefix():
    """First 27 features in v2 must be byte-identical to v1."""
    from ocr.confidence_gate import _calibrator_features
    out = _stub_parser_output()
    v1 = _calibrator_features(out)
    v2 = _calibrator_features_v2(out)
    assert v1 == v2[: len(v1)]
