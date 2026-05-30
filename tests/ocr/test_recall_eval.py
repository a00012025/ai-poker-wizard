"""Recall eval (Gemini fallback for parse_none) — pure-logic coverage.

The network call is integration-only; here we lock the two pieces that can
silently regress: which records the fallback fires on, and the production
emit-guard that rejects an incomplete Gemini hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr_recall_eval import select_parse_none, gemini_parse_image


def _write_records(tmp_path, records) -> str:
    p = tmp_path / "all_records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return str(p)


def test_select_parse_none_excludes_emitted_and_abstain(tmp_path):
    recs = [
        {"hand_id": "A", "parsed_none": True},                          # true parse_none
        {"hand_id": "B", "parsed_none": True, "abstained_confidence": True},  # gated
        {"hand_id": "C", "fields": {"hand_exact": True}},               # emitted
    ]
    out = select_parse_none(_write_records(tmp_path, recs))
    assert [r["hand_id"] for r in out] == ["A"]


def test_select_parse_none_include_abstain(tmp_path):
    recs = [
        {"hand_id": "A", "parsed_none": True},
        {"hand_id": "B", "parsed_none": True, "abstained_confidence": True},
    ]
    out = select_parse_none(_write_records(tmp_path, recs), include_abstain=True)
    assert {r["hand_id"] for r in out} == {"A", "B"}


class _Resp:
    def __init__(self, text):
        self.text = text


class _Client:
    def __init__(self, text):
        self._text = text
        self.models = self

    def generate_content(self, *a, **k):
        return _Resp(self._text)


def test_gemini_parse_rejects_incomplete_hand():
    # Missing hero_hand → production emit-guard rejects (returns None).
    text = '```json\n{"hand": {"hero_position": "BTN", "preflop_actions": "F-C"}}\n```'
    assert gemini_parse_image(b"x", client=_Client(text)) is None


def test_gemini_parse_accepts_complete_hand():
    text = ('```json\n{"hand": {"hero_position": "BTN", '
            '"preflop_actions": "F-F-R2-C", "hero_hand": "AsKc", '
            '"players_at_table": 6}}\n```')
    hand = gemini_parse_image(b"x", client=_Client(text))
    assert hand is not None and hand["hero_hand"] == "AsKc"


def test_gemini_parse_returns_none_on_garbage():
    assert gemini_parse_image(b"x", client=_Client("no json")) is None
