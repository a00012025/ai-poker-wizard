"""Phase 11.D-a — v3 feature vector adds board-pipeline and
position-pipeline signals on top of v2. v3 is a strict superset: the
first 40 entries are byte-identical to v2; the 10 new entries target the
board_wrong (46%) and position_wrong (39%) emit errors the v2 schema was
blind to.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.confidence_gate import (
    _calibrator_features_v2,
    _calibrator_features_v3,
    _V2_FEATURE_NAMES,
    _V3_FEATURE_NAMES,
)


def _stub(
    *,
    board_details=None,
    street_entries=None,
    hero_position_source="preflop_index_order",
    hero_seat_index=0,
    players_final=6,
    hero_blind_consistent=True,
) -> dict:
    diag = {
        "ensemble_used": False,
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 6,
        "players_at_table_raw": players_final,
        "players_at_table_final": players_final,
        "dealer_button_conf": 0.9,
        "estimate_used_reaction_signal": False,
        "street_entries_count": street_entries if street_entries is not None
        else {"flop": 3, "turn": 2, "river": 1},
        "street_entries_pre_collapse_count": {"flop": 3, "turn": 2, "river": 1},
        "hero_position_source": hero_position_source,
        "hero_seat_index": hero_seat_index,
        "hero_blind_detected": 0.0,
        "hero_blind_consistent": hero_blind_consistent,
    }
    hero_details = [
        {"rank": "A", "rank_conf": 0.95, "suit": "s", "suit_conf": 0.9,
         "rank_top2": [("A", 0.95), ("K", 0.03)],
         "suit_top2": [("s", 0.9), ("c", 0.05)],
         "rank_source": "classifier", "raw_suit": "s", "raw_suit_conf": 0.9,
         "masked_suit": "s", "masked_suit_conf": 0.9, "conf": 0.9},
        {"rank": "K", "rank_conf": 0.88, "suit": "h", "suit_conf": 0.86,
         "rank_top2": [("K", 0.88), ("Q", 0.08)],
         "suit_top2": [("h", 0.86), ("d", 0.10)],
         "rank_source": "classifier", "raw_suit": "h", "raw_suit_conf": 0.86,
         "masked_suit": "h", "masked_suit_conf": 0.86, "conf": 0.86},
    ]
    return {
        "confidence": 0.9,
        "confidence_parts": {
            "pot_consistency": 1.0, "player_tracking": 1.0,
            "ocr_confidence": 0.9, "card_confidence": 0.85,
        },
        "hand": {"preflop_actions": "F-F-R2-F-F-C"},
        "diagnostics": diag,
        "hero_card_details": hero_details,
        "board_card_details": board_details if board_details is not None else [
            {"rank": "K", "rank_conf": 0.99, "suit": "s", "suit_conf": 0.98,
             "rank_top2": [("K", 0.99), ("Q", 0.005)],
             "suit_top2": [("s", 0.98), ("c", 0.01)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.98},
            {"rank": "9", "rank_conf": 0.97, "suit": "d", "suit_conf": 0.95,
             "rank_top2": [("9", 0.97), ("8", 0.02)],
             "suit_top2": [("d", 0.95), ("h", 0.03)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.95},
            {"rank": "3", "rank_conf": 0.96, "suit": "d", "suit_conf": 0.94,
             "rank_top2": [("3", 0.96), ("2", 0.02)],
             "suit_top2": [("d", 0.94), ("h", 0.03)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.94},
        ],
        "safe_emit_reason": "",
    }


def test_v3_feature_count_matches_schema():
    feats = _calibrator_features_v3(_stub())
    assert len(feats) == len(_V3_FEATURE_NAMES) == 50


def test_v3_is_strict_superset_of_v2():
    """First 40 features in v3 must be byte-identical to v2."""
    out = _stub()
    v2 = _calibrator_features_v2(out)
    v3 = _calibrator_features_v3(out)
    assert v2 == v3[: len(v2)]


def test_board_min_conf_is_min_over_board_cards():
    out = _stub(board_details=[
        {"rank": "K", "rank_conf": 0.99, "suit": "s", "suit_conf": 0.60,
         "rank_top2": [("K", 0.99), ("Q", 0.005)],
         "suit_top2": [("s", 0.60), ("c", 0.35)],
         "rank_source": "classifier", "corner_disagree": False, "conf": 0.60},
    ])
    feats = _calibrator_features_v3(out)
    idx = _V3_FEATURE_NAMES.index("board_min_conf")
    assert abs(feats[idx] - 0.60) < 1e-9


def test_board_corner_disagree_counts_overrides():
    out = _stub(board_details=[
        {"rank": "K", "rank_conf": 0.92, "suit": "s", "suit_conf": 0.9,
         "rank_top2": [("K", 0.92), ("Q", 0.05)],
         "suit_top2": [("s", 0.9), ("c", 0.05)],
         "rank_source": "corner_ocr", "corner_disagree": True, "conf": 0.9},
        {"rank": "9", "rank_conf": 0.97, "suit": "d", "suit_conf": 0.95,
         "rank_top2": [("9", 0.97), ("8", 0.02)],
         "suit_top2": [("d", 0.95), ("h", 0.03)],
         "rank_source": "classifier", "corner_disagree": False, "conf": 0.95},
    ])
    feats = _calibrator_features_v3(out)
    idx = _V3_FEATURE_NAMES.index("board_corner_disagree_count")
    assert feats[idx] == 1.0


def test_board_count_vs_street_mismatch_when_river_seen_but_board_short():
    # River entries present (expect >=5 board cards) but only 3 detected.
    out = _stub(
        street_entries={"flop": 3, "turn": 2, "river": 1},
        board_details=[
            {"rank": "K", "rank_conf": 0.99, "suit": "s", "suit_conf": 0.98,
             "rank_top2": [("K", 0.99)], "suit_top2": [("s", 0.98)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.98},
            {"rank": "9", "rank_conf": 0.97, "suit": "d", "suit_conf": 0.95,
             "rank_top2": [("9", 0.97)], "suit_top2": [("d", 0.95)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.95},
            {"rank": "3", "rank_conf": 0.96, "suit": "d", "suit_conf": 0.94,
             "rank_top2": [("3", 0.96)], "suit_top2": [("d", 0.94)],
             "rank_source": "classifier", "corner_disagree": False, "conf": 0.94},
        ],
    )
    feats = _calibrator_features_v3(out)
    idx = _V3_FEATURE_NAMES.index("board_count_vs_street_mismatch")
    assert feats[idx] == 1.0


def test_board_count_matches_when_no_street_no_board():
    # Preflop hand: no streets, no board → no mismatch, neutral board feats.
    out = _stub(street_entries={}, board_details=[])
    feats = _calibrator_features_v3(out)
    mism = _V3_FEATURE_NAMES.index("board_count_vs_street_mismatch")
    mc = _V3_FEATURE_NAMES.index("board_min_conf")
    assert feats[mism] == 0.0
    assert feats[mc] == 1.0  # no board → neutral (non-suspicious)


def test_position_source_one_hot():
    for src, name in [
        ("preflop_index_order", "pos_src_preflop_index"),
        ("hero_fold_recovery", "pos_src_fold_recovery"),
        ("blind_column", "pos_src_blind_column"),
    ]:
        feats = _calibrator_features_v3(_stub(hero_position_source=src))
        idx = _V3_FEATURE_NAMES.index(name)
        assert feats[idx] == 1.0, f"{src} should set {name}"
        # The other two must be 0.
        for other in ("pos_src_preflop_index", "pos_src_fold_recovery",
                      "pos_src_blind_column"):
            if other != name:
                assert feats[_V3_FEATURE_NAMES.index(other)] == 0.0


def test_hero_seat_y_norm():
    out = _stub(hero_seat_index=3, players_final=7)
    feats = _calibrator_features_v3(out)
    idx = _V3_FEATURE_NAMES.index("hero_seat_y_norm")
    assert abs(feats[idx] - (3 / 6)) < 1e-9


def test_hero_blind_consistent_flag():
    consistent = _calibrator_features_v3(_stub(hero_blind_consistent=True))
    inconsistent = _calibrator_features_v3(_stub(hero_blind_consistent=False))
    idx = _V3_FEATURE_NAMES.index("hero_blind_consistent")
    assert consistent[idx] == 1.0
    assert inconsistent[idx] == 0.0


def test_v3_robust_to_v2_era_records_missing_new_fields():
    """A v2-era parser_output (no board_card_details, no position diag)
    must still yield a full 50-dim vector with neutral defaults."""
    out = _stub()
    del out["board_card_details"]
    for k in ("hero_position_source", "hero_seat_index",
              "hero_blind_detected", "hero_blind_consistent"):
        out["diagnostics"].pop(k, None)
    feats = _calibrator_features_v3(out)
    assert len(feats) == 50
    # No board → neutral board_min_conf, no position source → all-zero one-hot.
    assert feats[_V3_FEATURE_NAMES.index("board_min_conf")] == 1.0
    assert feats[_V3_FEATURE_NAMES.index("pos_src_preflop_index")] == 0.0
