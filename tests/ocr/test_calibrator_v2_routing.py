"""Phase 11.C.3 — CalibratorScorer auto-detects v2 artifacts.

When data/calibrator/rf_model_v2.joblib exists, the scorer loads the
three v2 base models, computes v2 features, and averages predict_proba.
OCR_CALIBRATOR_VERSION=v1 forces fallback to the legacy single-model
path for rollback.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.confidence_gate import CalibratorScorer


def _stub_parser_output() -> dict:
    return {
        "confidence": 0.9,
        "confidence_parts": {
            "pot_consistency": 1.0, "player_tracking": 1.0,
            "ocr_confidence": 0.9, "card_confidence": 0.85,
        },
        "hand": {"preflop_actions": "F-F-R2-F-F-C"},
        "diagnostics": {
            "ensemble_used": False,
            "preflop_entries_count": 6,
            "preflop_entries_pre_collapse_count": 6,
            "players_at_table_raw": 6,
            "players_at_table_final": 6,
            "dealer_button_conf": 0.9,
            "estimate_used_reaction_signal": False,
            "street_entries_count": {},
            "street_entries_pre_collapse_count": {},
        },
        "hero_card_details": [
            {"rank": "A", "rank_conf": 0.95, "suit": "s", "suit_conf": 0.9,
             "rank_top2": [("A", 0.95), ("K", 0.03)],
             "suit_top2": [("s", 0.9), ("c", 0.05)],
             "rank_source": "classifier", "conf": 0.9},
            {"rank": "K", "rank_conf": 0.88, "suit": "h", "suit_conf": 0.86,
             "rank_top2": [("K", 0.88), ("Q", 0.08)],
             "suit_top2": [("h", 0.86), ("d", 0.10)],
             "rank_source": "classifier", "conf": 0.86},
        ],
        "safe_emit_reason": "",
    }


def _train_synth_v2(tmp_path: Path) -> Path:
    """Train tiny v2 models on synthetic data so we can exercise loading."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 40))
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)
    out = tmp_path / "calibrator"
    out.mkdir()
    feat_names = [f"f{i}" for i in range(40)]
    joblib.dump({"model": rf, "feature_names": feat_names},
                out / "rf_model_v2.joblib")
    joblib.dump({"model": gb, "feature_names": feat_names},
                out / "gb_model_v2.joblib")
    joblib.dump({"model": lr, "scaler": scaler, "feature_names": feat_names},
                out / "lr_model_v2.joblib")
    (out / "oof_v2.json").write_text("{}")
    return out


def test_v2_auto_detect_when_artifacts_present(tmp_path, monkeypatch):
    cal_dir = _train_synth_v2(tmp_path)
    monkeypatch.delenv("OCR_CALIBRATOR_VERSION", raising=False)
    scorer = CalibratorScorer(calibrator_dir=cal_dir)
    score = scorer.score(_stub_parser_output(), hand_id="H_TEST")
    assert score is not None
    assert 0.0 <= score <= 1.0
    assert scorer.version == "v2"


def test_v1_forced_via_env(tmp_path, monkeypatch):
    cal_dir = _train_synth_v2(tmp_path)
    monkeypatch.setenv("OCR_CALIBRATOR_VERSION", "v1")
    scorer = CalibratorScorer(calibrator_dir=cal_dir)
    # No v1 artifacts in this tmp dir → score returns None (caller falls back).
    score = scorer.score(_stub_parser_output(), hand_id="H_TEST")
    assert score is None
    assert scorer.version == "v1"


def _train_synth_v3(tmp_path: Path, *, with_isotonic: bool) -> Path:
    """Train tiny v3 models (50-dim) + optional isotonic so we can exercise
    the v3 loading + calibration path."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 50))
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)
    out = tmp_path / "calibrator"
    out.mkdir()
    feat_names = [f"f{i}" for i in range(50)]
    joblib.dump({"model": rf, "feature_names": feat_names},
                out / "rf_model_v3.joblib")
    joblib.dump({"model": gb, "feature_names": feat_names},
                out / "gb_model_v3.joblib")
    joblib.dump({"model": lr, "scaler": scaler, "feature_names": feat_names},
                out / "lr_model_v3.joblib")
    (out / "oof_v3.json").write_text("{}")
    if with_isotonic:
        avg = (rf.predict_proba(X)[:, 1] + gb.predict_proba(X)[:, 1]
               + lr.predict_proba(scaler.transform(X))[:, 1]) / 3.0
        iso = IsotonicRegression(out_of_bounds="clip").fit(avg, y)
        joblib.dump({"model": iso, "feature_key": "v3_features"},
                    out / "isotonic_v3.joblib")
    return out


def test_v3_auto_detected_and_scores_in_range(tmp_path, monkeypatch):
    cal_dir = _train_synth_v3(tmp_path, with_isotonic=True)
    monkeypatch.delenv("OCR_CALIBRATOR_VERSION", raising=False)
    scorer = CalibratorScorer(calibrator_dir=cal_dir)
    assert scorer.version == "v3"
    score = scorer.score(_stub_parser_output(), hand_id="H_TEST")
    assert score is not None
    assert 0.0 <= score <= 1.0
    # Isotonic was present → it must have been loaded.
    assert scorer._iso is not None


def test_v3_preferred_over_v2_when_both_present(tmp_path, monkeypatch):
    cal_dir = _train_synth_v3(tmp_path, with_isotonic=False)
    # Drop a v2 model alongside; v3 must still win the auto-detect.
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 40)); y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    joblib.dump({"model": rf, "feature_names": [f"f{i}" for i in range(40)]},
                cal_dir / "rf_model_v2.joblib")
    monkeypatch.delenv("OCR_CALIBRATOR_VERSION", raising=False)
    scorer = CalibratorScorer(calibrator_dir=cal_dir)
    assert scorer.version == "v3"
