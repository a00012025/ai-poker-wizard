"""Diagnostics payload exposes per-hand parser internals."""
from __future__ import annotations

from pathlib import Path

from ocr.n8_parser import parse_n8_screenshot


SAMPLE = Path("data/hand_images/img/TM5846885824.png")


def test_diagnostics_payload_shape():
    result = parse_n8_screenshot(SAMPLE.read_bytes())
    diag = result.get("diagnostics")
    assert isinstance(diag, dict), "diagnostics block missing"

    for key in (
        "players_at_table_raw",
        "players_at_table_final",
        "estimate_used_reaction_signal",
        "dealer_button_seat",
        "dealer_button_conf",
        "preflop_entries_count",
        "preflop_entries_pre_collapse_count",
        "street_entries_count",
    ):
        assert key in diag, f"missing diagnostics key: {key}"

    assert isinstance(diag["street_entries_count"], dict)
    assert diag["players_at_table_final"] is None or 2 <= diag["players_at_table_final"] <= 9
