#!/usr/bin/env python3
"""Capture frozen GTOW Analyze fixtures for ledger regression tests."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gtow_analyze_api import iter_all_hands, hand_detail

FIX = Path(__file__).resolve().parent / "fixtures" / "gtow"
TARGETS = {
    "eef0b07b-23b6-4fe0-bcc6-41d83629583c": ["2026-05-30T00:00:00.000Z", "2026-05-31T00:00:00.000Z"],
    "bed8860a-442b-4478-a9b4-8acfd52b6143": ["2026-03-01T00:00:00.000Z", "2026-03-02T00:00:00.000Z"],
}

rows = {}
for hid, (a, b) in TARGETS.items():
    for row in iter_all_hands(a, b):
        if row["hand_id"] == hid:
            rows[hid] = row
            break
    assert hid in rows, f"list row not found for {hid}"
    det = hand_detail(hid)
    (FIX / f"detail_{hid[:8]}.json").write_text(json.dumps(det, indent=1))
    print(hid[:8], "detail game_points:",
          len(det["game_analysis"]["game_points"]))

(FIX / "list_rows.json").write_text(json.dumps(rows, indent=1))
print("fixtures written to", FIX)
