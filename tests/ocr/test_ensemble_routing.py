"""When raw hero card_conf is < OCR_ENSEMBLE_FLOOR, the parser invokes
the ensemble and surfaces ensemble_used=True in diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.n8_parser import parse_n8_screenshot


def test_h3433_triggers_ensemble():
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    res = parse_n8_screenshot(img.read_bytes())
    diag = res.get("diagnostics") or {}
    assert diag.get("ensemble_used") is True, (
        f"H3433-class image must trigger the ensemble path; "
        f"diagnostics={diag}"
    )
