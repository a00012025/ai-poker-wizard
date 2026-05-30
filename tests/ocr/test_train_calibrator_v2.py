"""Phase 11.C.1 — train_calibrator_v2.py reads all_records.jsonl files,
stacks RF + GB + LR base models on the v2 feature vector with 5-fold CV,
and saves the three .joblib artifacts plus an out-of-fold predictions
file the joint-evaluation step can consume.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _write_records(
    path: Path, n: int, *, seed: int, width: int = 40,
    feature_key: str = "v2_features",
) -> None:
    import random
    rng = random.Random(seed)
    with path.open("w") as fh:
        for i in range(n):
            label = 1 if rng.random() < 0.7 else 0
            feats = [rng.random() for _ in range(width)]
            # Bias one feature with the label so the model has signal.
            feats[0] = (0.8 + rng.random() * 0.2) if label else (rng.random() * 0.4)
            rec = {
                "hand_id": f"H{seed}_{i:04d}",
                feature_key: feats,
                "fields": {"hand_exact": bool(label)},
            }
            fh.write(json.dumps(rec) + "\n")


def test_train_calibrator_v2_smoke(tmp_path):
    train_path = tmp_path / "train_records.jsonl"
    val_path = tmp_path / "val_records.jsonl"
    out_dir = tmp_path / "calibrator"
    _write_records(train_path, 200, seed=0)
    _write_records(val_path, 60, seed=1)

    features_path = Path("data/calibrator/v2_features.txt")
    if not features_path.exists():
        pytest.skip("v2_features.txt missing")

    subprocess.run(
        [
            "python",
            "-m",
            "scripts.ocr.classifier.train_calibrator_v2",
            "--train", str(train_path),
            "--val", str(val_path),
            "--features", str(features_path),
            "--out-dir", str(out_dir),
        ],
        check=True,
    )

    for fname in ("rf_model_v2.joblib", "gb_model_v2.joblib",
                  "lr_model_v2.joblib", "oof_v2.json"):
        assert (out_dir / fname).exists(), f"{fname} missing"

    oof = json.loads((out_dir / "oof_v2.json").read_text())
    # OOF covers the joint train+val set (260 records).
    assert len(oof) == 260
    for hid, p in list(oof.items())[:5]:
        assert 0.0 <= p <= 1.0, f"{hid}={p}"


def test_train_v3_with_isotonic(tmp_path):
    """v3 path: reads v3_features, writes _v3 artifacts, and an isotonic
    calibrator file when --isotonic is passed."""
    train_path = tmp_path / "train_records.jsonl"
    val_path = tmp_path / "val_records.jsonl"
    out_dir = tmp_path / "calibrator"
    _write_records(train_path, 200, seed=0, width=50, feature_key="v3_features")
    _write_records(val_path, 60, seed=1, width=50, feature_key="v3_features")

    features_path = Path("data/calibrator/v3_features.txt")
    if not features_path.exists():
        pytest.skip("v3_features.txt missing")

    subprocess.run(
        [
            "python", "-m", "scripts.ocr.classifier.train_calibrator_v2",
            "--train", str(train_path),
            "--val", str(val_path),
            "--features", str(features_path),
            "--feature-key", "v3_features",
            "--out-suffix", "v3",
            "--isotonic",
            "--out-dir", str(out_dir),
        ],
        check=True,
    )

    for fname in ("rf_model_v3.joblib", "gb_model_v3.joblib",
                  "lr_model_v3.joblib", "oof_v3.json", "isotonic_v3.joblib"):
        assert (out_dir / fname).exists(), f"{fname} missing"

    oof = json.loads((out_dir / "oof_v3.json").read_text())
    assert len(oof) == 260
    for hid, p in list(oof.items())[:5]:
        assert 0.0 <= p <= 1.0, f"{hid}={p}"
