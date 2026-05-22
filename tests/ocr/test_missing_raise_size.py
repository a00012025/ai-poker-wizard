"""Missing raise sizes lower confidence without discarding action structure."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_missing_raise_size_returns_low_confidence_hand_for_fallback_context():
    # The SB raise text is OCR'd without a numeric size.  This should not make
    # the whole deterministic parse disappear: hand_exact only needs the action
    # type, and production confidence gates can still fall back for solver-safe
    # sizing.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5863569047.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_hand"] == "KsKc"
    assert "R2" in result["hand"]["preflop_actions"]
    assert result["confidence_parts"]["ocr_confidence"] == 0.0
