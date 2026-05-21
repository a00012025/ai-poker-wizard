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

    assert diffs_path.exists(), "diffs.jsonl missing"
    assert summary_path.exists(), "summary.json missing"
    assert diag_summary_path.exists(), "diagnostics_summary.json missing"

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
    ):
        assert key in diag_summary
