"""Phase 11.A.2 — backfill production_v1 corpus from snapshots where a
fresh Gemini reparse independently agrees with the stored OCR parse.

Two-signal agreement on (hero_hand, board, hero_position) is treated as
credible ground truth without manual verification. Records accepted via
this path are tagged ``source="gemini_reparse_agreement"`` in gt.jsonl.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .harvest_production import harvest_snapshot


def _board_from_streets(streets: Any) -> str | None:
    if not isinstance(streets, list) or not streets:
        return None
    first = streets[0]
    if not isinstance(first, dict):
        return None
    board = first.get("board")
    if isinstance(board, str) and board:
        return board
    return None


def _key_fields(parsed: dict | None) -> tuple[str | None, str | None, str | None]:
    if not isinstance(parsed, dict):
        return (None, None, None)
    return (
        parsed.get("hero_hand"),
        _board_from_streets(parsed.get("streets")),
        parsed.get("hero_position"),
    )


def is_agreement(ocr_parsed: dict | None, gemini_parsed: dict | None) -> bool:
    """True iff both parses are non-empty and agree on all three fields."""
    if not ocr_parsed or not gemini_parsed:
        return False
    ocr_key = _key_fields(ocr_parsed)
    gemini_key = _key_fields(gemini_parsed)
    # All three slots must be non-None and identical.
    if any(v is None for v in ocr_key) or any(v is None for v in gemini_key):
        return False
    return ocr_key == gemini_key


def backfill_one(
    *,
    hand_id: str,
    image_bytes: bytes,
    ocr_parsed: dict,
    gemini_parsed: dict | None,
    out_dir: Path,
) -> dict | None:
    """Accept the record into the corpus if Gemini reparse agrees.

    Returns the gt.jsonl entry (with ``source="gemini_reparse_agreement"``)
    on accept, ``None`` on reject. Side effect on accept: harvests labeled
    hero crops into ``out_dir`` using OCR's parse as ground truth.
    """
    if not is_agreement(ocr_parsed, gemini_parsed):
        return None
    n = harvest_snapshot(
        hand_id=hand_id,
        image_bytes=image_bytes,
        expected=ocr_parsed,
        out_dir=Path(out_dir),
    )
    if n < 2:
        return None
    return {
        "hand_id": hand_id,
        "ground_truth": ocr_parsed,
        "source": "gemini_reparse_agreement",
    }
