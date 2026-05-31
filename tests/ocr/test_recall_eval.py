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

from ocr_recall_eval import (
    select_parse_none,
    gemini_parse_image,
    gemini_hero_hand_only,
    ocr_hand_from_record,
)


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


def test_select_parse_none_only_abstain(tmp_path):
    recs = [
        {"hand_id": "A", "parsed_none": True},
        {"hand_id": "B", "parsed_none": True, "abstained_confidence": True},
        {"hand_id": "C", "abstained_confidence": True, "parsed": {"hero_hand": "AsKs"}},
    ]
    out = select_parse_none(_write_records(tmp_path, recs), only_abstain=True)
    assert [r["hand_id"] for r in out] == ["B", "C"]


def test_ocr_hand_from_record_restores_parsed_streets():
    rec = {
        "parsed": {
            "hero_hand": "AsKs",
            "hero_position": "BB",
            "preflop_actions": "F-R2-C",
        },
        "parsed_streets": [["Ah", "Kd", "2c"], ["3s"], ["9h"]],
    }
    hand = ocr_hand_from_record(rec)
    assert hand["streets"] == [
        {"board": "AhKd2c", "actions": []},
        {"card": "3s", "actions": []},
        {"card": "9h", "actions": []},
    ]


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


def test_gemini_hero_hand_only_accepts_tight_json(monkeypatch):
    monkeypatch.setattr(
        "gemini_session.GeminiSessionManager._hero_cards_image_for_micro_read",
        staticmethod(lambda b, fallback_mime_type="image/png": (b, fallback_mime_type)),
    )
    text = '```json\n{"hero_hand": "Th2s"}\n```'
    got = gemini_hero_hand_only(
        b"x",
        ocr_hand={"hero_position": "SB", "players_at_table": 6},
        client=_Client(text),
    )
    assert got == "Th2s"


def test_gemini_hero_hand_only_rejects_non_card(monkeypatch):
    monkeypatch.setattr(
        "gemini_session.GeminiSessionManager._hero_cards_image_for_micro_read",
        staticmethod(lambda b, fallback_mime_type="image/png": (b, fallback_mime_type)),
    )
    assert gemini_hero_hand_only(b"x", client=_Client('{"hero_hand": "ZZZZ"}')) is None
