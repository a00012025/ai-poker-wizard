#!/usr/bin/env python3
"""Backfill action-line spot columns on ledger_decisions from the archived raw.

Pure re-distill of the taxonomy (no API): reads data/gtow_raw, runs
spot_taxonomy.walk_spots, and UPDATEs each ledger_decisions row matched by
(gtow_hand_id, street, decision_idx).

Default = INCREMENTAL/UPGRADE: hands whose decisions lack the current leaf,
parent-family or decision-depth contract (the daily/deploy job's case).
`--full` re-reads the whole archive after taxonomy semantics evolve. Raw stays
untouched either way.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
from ledger_distill import distill_hand
from spot_taxonomy import walk_spots

RAW = ROOT / "data" / "gtow_raw"

INCREMENTAL_MISSING_SQL = """
SELECT DISTINCT gtow_hand_id FROM ledger_decisions
WHERE source='online' AND (
  spot_leaf IS NULL OR spot_parent IS NULL OR played_depth_bb IS NULL
)
"""

READINESS_GAP_SQL = """
SELECT count(*) FROM ledger_decisions
WHERE source='online' AND NOT excluded AND NOT discarded
  AND (spot_leaf IS NULL OR spot_parent IS NULL OR played_depth_bb IS NULL)
"""

UPDATE_SQL = """
UPDATE ledger_decisions SET
  spot_category=$4, spot_leaf=$5, spot_parent=$6, spot_keys=$7,
  hero_cat=$8, villain_cat=$9, ip_oop=$10, flop_seq=$11, turn_seq=$12,
  eff_stack=$13, depth_band=$14, played_depth_bb=$15, solver_depth_bb=$16,
  board_suit=$17, board_conn=$18, board_paired=$19,
  discarded=$20, limp_origin=$21, confidence=$22, approx_flags=$23
WHERE gtow_hand_id=$1 AND street=$2 AND decision_idx=$3
"""


def _row(s: dict, decision: dict):
    t = s.get("tags", {})
    return (
        s["gtow_hand_id"], s["street"], s["decision_idx"],
        s["category"], s["leaf"], s.get("parent"), json.dumps(s["keys"]),
        s.get("hero_cat"), s.get("villain_cat"), s.get("ip_oop"),
        s.get("flop_seq"), s.get("turn_seq"),
        t.get("eff_stack"), t.get("depth_band"),
        decision.get("played_depth_bb"), decision.get("solver_depth_bb"),
        t.get("board_suit"), t.get("board_conn"), t.get("board_paired"),
        bool(s.get("discarded")), bool(s.get("limp_origin")),
        decision.get("confidence", 0.0), json.dumps(decision.get("approx_flags") or []),
    )


def load_list_index() -> dict:
    idx = {}
    for f in glob.glob(str(RAW / "list" / "*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                idx[r["hand_id"]] = r
    return idx


def select_files(files: list[str], target_ids: set[str] | None) -> list[str]:
    """target_ids=None → all files (--full, taxonomy evolution re-distill);
    otherwise only the archive files for hands still lacking spot columns —
    the daily job must not re-read the ENTIRE archive (O(history), grows
    forever) just to classify yesterday's hands."""
    if target_ids is None:
        return files
    return [f for f in files if Path(f).stem.replace(".json", "") in target_ids]


async def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="re-distill ALL archived hands (after taxonomy changes); "
                         "default = hands missing leaf or current depth/family fields")
    args = ap.parse_args()

    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    target_ids: set[str] | None = None
    if not args.full:
        target_ids = {r["gtow_hand_id"] for r in await conn.fetch(INCREMENTAL_MISSING_SQL)}
        if not target_ids:
            print("SPOTS backfill: nothing to do (ledger taxonomy/depth fields current)")
            await conn.close()
            return 0
    list_idx = load_list_index()
    files = select_files(glob.glob(str(RAW / "detail" / "*" / "*.json.gz")), target_ids)
    mode = "full" if args.full else f"incremental({len(target_ids)} hands)"
    print(f"list rows={len(list_idx)} detail files={len(files)} mode={mode}", flush=True)
    batch, n_spots, n_hands, n_missing = [], 0, 0, 0
    try:
        for i, f in enumerate(files):
            hid = Path(f).stem.replace(".json", "")
            lr = list_idx.get(hid)
            if lr is None:
                n_missing += 1
                continue
            with gzip.open(f, "rt") as fh:
                det = json.load(fh)
            n_hands += 1
            _hand, decisions = distill_hand(lr, det)
            by_node = {(d["street"], d["decision_idx"]): d for d in decisions}
            for s in walk_spots(lr, det):
                decision = by_node.get((s["street"], s["decision_idx"]))
                if decision is None:
                    continue
                batch.append(_row(s, decision))
                n_spots += 1
            if len(batch) >= 2000:
                await conn.executemany(UPDATE_SQL, batch); batch = []
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(files)} files, {n_spots} spots", flush=True)
        if batch:
            await conn.executemany(UPDATE_SQL, batch)
        classified = await conn.fetchval("SELECT count(*) FROM ledger_decisions WHERE spot_leaf IS NOT NULL")
        readiness_gap = await conn.fetchval(READINESS_GAP_SQL)
        print(f"SPOTS backfilled: spots={n_spots} hands={n_hands} missing_list={n_missing} "
              f"rows_with_spot_leaf={classified} readiness_gap={readiness_gap}")
        if readiness_gap:
            print("SPOTS backfill incomplete: honest online rows still miss current family/depth fields",
                  file=sys.stderr)
            return 2
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
