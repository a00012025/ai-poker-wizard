"""ocr_precision dumps per-hand diagnostics into diffs and a summary."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_ocr_precision_writes_diagnostics(tmp_path):
    out = tmp_path / "out"
    subprocess.run(
        [
            "python",
            "scripts/ocr_precision.py",
            "--limit",
            "5",
            "--workers",
            "1",
            "--out",
            str(out),
        ],
        check=True,
    )

    diffs_path = out / "diffs.jsonl"
    summary_path = out / "summary.json"
    diag_summary_path = out / "diagnostics_summary.json"
    calib_path = out / "calibration_summary.json"

    assert diffs_path.exists(), "diffs.jsonl missing"
    assert summary_path.exists(), "summary.json missing"
    assert diag_summary_path.exists(), "diagnostics_summary.json missing"
    assert calib_path.exists(), "calibration_summary.json missing"

    found_diag = False
    for line in diffs_path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("skipped") or rec.get("error"):
            continue
        assert "diagnostics" in rec, f"missing diagnostics in record: {rec.get('hand_id')}"
        found_diag = True
    assert found_diag

    diag_summary = json.loads(diag_summary_path.read_text())
    for key in (
        "dealer_button_detection_rate",
        "estimate_reaction_signal_rate",
        "pre_collapse_loss_histogram",
        "safe_emit_override_reasons",
    ):
        assert key in diag_summary

    summary = json.loads(summary_path.read_text())
    assert "safe_emit_overrides" in summary
    assert "safe_emit_override_exact" in summary
    assert "safe_emit_override_wrong" in summary
    assert "safe_emit_override_reasons" in summary

    calib = json.loads(calib_path.read_text())
    assert "ece_10bin" in calib
    assert "precision_coverage_curve" in calib


def test_ocr_precision_production_bucket(tmp_path):
    """--bucket production_test walks data/cards_v2/production_v1/ via the
    production split file and writes a separate summary."""
    import pytest

    split_path = Path("data/splits/production_v1.json")
    if not split_path.exists():
        pytest.skip("production_v1 split not seeded yet")
    out = tmp_path / "out"
    res = subprocess.run(
        [
            "python",
            "scripts/ocr_precision.py",
            "--split",
            str(split_path),
            "--bucket",
            "production_test",
            "--limit",
            "3",
            "--workers",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert (out / "summary.json").exists()
