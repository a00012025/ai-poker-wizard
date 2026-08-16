#!/usr/bin/env python3
"""Acceptance gate for zero-loss list-only decision reconstruction.

Selects archived solved zero-loss hands that already have full-detail ledger
rows, then proves every runtime-accepted reconstruction matches the canonical
detail taxonomy tuple. Hands rejected by the runtime adapter are reported as
fallbacks and do not weaken the 100% fidelity gate.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from ledger_distill import (  # noqa: E402
    ListOnlyReconstructionError,
    _norm_code,
    distill_hand,
    distill_hand_from_list,
)
from spot_taxonomy import walk_spots  # noqa: E402

RAW = ROOT / "data" / "gtow_raw"


def load_list_index() -> dict[str, dict]:
    rows = {}
    for path in glob.glob(str(RAW / "list" / "*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                row = json.loads(line)
                rows[row["hand_id"]] = row
    return rows


def structural_tuple(decision: dict) -> tuple:
    return (
        decision["street"], decision["decision_idx"], decision["taken_code"],
        decision["correctness"], decision["spot_leaf"],
        decision["spot_category"], decision["facing"], decision["pot_type"],
        decision["position"],
    )


def detail_structure(list_row: dict, detail: dict) -> list[tuple]:
    _hand, decisions = distill_hand(list_row, detail)
    spots = list(walk_spots(list_row, detail))
    by_node = {(s["street"], s["decision_idx"]): s for s in spots}
    rows = []
    for decision in decisions:
        key = (decision["street"], decision["decision_idx"])
        spot = by_node[key]
        pot_type = spot.get("pot_type")
        if decision["street"] == "preflop":
            # The historical detail distiller can label a shove-open as
            # "unopened" because its generic pot helper ignores the AI token.
            # The official action-line category is authoritative here.
            pot_type = {
                "RFI": "unopened", "vsOpen": "SRP", "vsRaiseCall": "SRP",
                "vsSqueeze": "squeezed", "vs3bet": "3bet",
                "vsCold3bet": "3bet", "vs4bet": "4bet", "vsCold4bet": "4bet",
                "discarded": "limp",
            }.get(spot.get("category"), decision["pot_type"])
        rows.append(structural_tuple({
            **decision,
            "taken_code": _norm_code(decision["taken_code"]),
            "spot_leaf": spot["leaf"], "spot_category": spot["category"],
            "facing": spot.get("facing") or "unopened",
            "pot_type": pot_type,
        }))
    return rows


async def candidate_ids() -> list[str]:
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        rows = await conn.fetch("""
            SELECT h.gtow_hand_id
            FROM ledger_hands h
            WHERE h.source='online' AND h.detail_fetched
              AND h.solution_status='OK' AND h.total_ev_loss_bb=0
              AND EXISTS (
                SELECT 1 FROM ledger_decisions d WHERE d.gtow_hand_id=h.gtow_hand_id
              )
            ORDER BY h.gtow_hand_id
        """)
        return [r["gtow_hand_id"] for r in rows]
    finally:
        await conn.close()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    if args.sample < 200:
        ap.error("--sample must be at least 200")

    list_rows = load_list_index()
    ids = await candidate_ids()
    accepted = fallback = missing = 0
    divergences = []
    for hand_id in ids:
        list_row = list_rows.get(hand_id)
        if list_row is None:
            missing += 1
            continue
        detail_path = RAW / "detail" / list_row["played_at"][:7] / f"{hand_id}.json.gz"
        if not detail_path.exists():
            missing += 1
            continue
        try:
            _hand, reconstructed = distill_hand_from_list(list_row)
        except ListOnlyReconstructionError:
            fallback += 1
            continue
        with gzip.open(detail_path, "rt") as fh:
            detail = json.load(fh)
        actual = [structural_tuple(row) for row in reconstructed]
        expected = detail_structure(list_row, detail)
        if actual != expected:
            divergences.append((hand_id, expected, actual))
        accepted += 1
        if accepted >= args.sample:
            break

    exact_match = f"{accepted - len(divergences)}/{accepted}" if accepted else "0/0"
    print(f"RECONSTRUCT accepted={accepted} fallback={fallback} missing={missing} "
          f"divergent={len(divergences)} exact_match={exact_match}")
    for hand_id, expected, actual in divergences[:10]:
        print(f"DIVERGENCE {hand_id}\n  expected={expected}\n  actual={actual}")
    if accepted < args.sample:
        print(f"FAIL: needed {args.sample} runtime-accepted hands", file=sys.stderr)
        return 2
    return 0 if not divergences else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
