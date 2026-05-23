"""Guarded low-confidence safe-emission overrides for OCR coverage lift."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402
import ocr_precision  # noqa: E402


def _parse(hand_id: str) -> dict:
    return parse_n8_screenshot(
        Path(f"data/hand_images/img/{hand_id}.png").read_bytes()
    )


def test_safe_emit_reasons_cover_three_reusable_predicates():
    cases = {
        "TM5846884867": "simple_preflop_high_card",
        "TM5875362766": "high_card_complex_non_danger",
        "TM5863067643": "stable_postflop_high_card",
    }

    for hand_id, reason in cases.items():
        result = _parse(hand_id)
        assert result["confidence"] < 0.88
        assert result.get("safe_emit_reason") == reason
        assert result["diagnostics"].get("safe_emit_reason") == reason


def test_known_danger_abstains_keep_no_safe_emit_reason():
    # These are high-card but structurally dangerous abstains: fragile all-in
    # sizing, reaction/table-size mismatch, or postflop re-action ambiguity.
    for hand_id in (
        "TM5846884903",
        "TM5867249527",
        "TM5913202014",
        "TM5962883091",
    ):
        result = _parse(hand_id)
        assert result["confidence"] < 0.88
        assert result.get("safe_emit_reason") is None
        assert result["diagnostics"].get("safe_emit_reason") is None


def test_precision_harness_emits_safe_overrides_but_abstains_danger_cases():
    ocr_precision._init_worker(
        "data/pokercraft_corpus/ground_truth/ground_truth.jsonl",
        emit_threshold=0.88,
    )

    positive = ocr_precision._run_one("data/hand_images/img/TM5846884867.png")
    assert positive.get("safe_emit_overridden") is True
    assert positive.get("parsed_none") is None
    assert positive["fields"]["hand_exact"] is True

    negative = ocr_precision._run_one("data/hand_images/img/TM5846884903.png")
    assert negative.get("safe_emit_reason") is None
    assert negative.get("abstained_confidence") is True
    assert negative.get("parsed_none") is True
