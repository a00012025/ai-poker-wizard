"""Phase 11.B.2 — ensemble routing fires when a hero card's raw conf
falls below OCR_ENSEMBLE_FLOOR (default 0.50), and the ensemble_used
flag bubbles all the way up to parse_n8_screenshot()['diagnostics'].

H3433 is the canonical case: the raw classifier reads 5d at conf ~0.16
because the WIN sticker partially occludes the rank glyph, so the
ensemble path must trigger and the diagnostic flag must show True.
"""
from __future__ import annotations

from pathlib import Path

import pytest


from ocr.n8_parser import parse_n8_screenshot


def test_h3433_triggers_ensemble_and_bubbles_diagnostic():
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    result = parse_n8_screenshot(img.read_bytes())
    diagnostics = result.get("diagnostics") or {}
    assert "ensemble_used" in diagnostics, \
        "diagnostics must expose ensemble_used flag"
    assert diagnostics["ensemble_used"] is True, (
        f"H3433 hero crop is low-conf (raw ~0.16) so ensemble should fire; "
        f"got {diagnostics['ensemble_used']!r}"
    )


def test_clean_hand_does_not_force_ensemble():
    """A clean PokerCraft hand with high-conf hero cards should NOT
    route through the ensemble path (ensemble_used = False)."""
    img_dir = Path("data/hand_images/img")
    candidates = list(img_dir.glob("TM*.png"))[:5]
    if not candidates:
        pytest.skip("PokerCraft corpus not present")

    any_clean_seen = False
    for img in candidates:
        result = parse_n8_screenshot(img.read_bytes())
        diagnostics = result.get("diagnostics") or {}
        if diagnostics.get("ensemble_used") is False:
            any_clean_seen = True
            break
    assert any_clean_seen, (
        "At least one of the first 5 PokerCraft images should have "
        "ensemble_used = False (clean hero crops, no fallback needed)"
    )
