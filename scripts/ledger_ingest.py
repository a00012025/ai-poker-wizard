#!/usr/bin/env python3
"""Ingest GTOW Analyze hands into the ledger. Idempotent, resumable, loud.

Modes:
  --backfill --since 2026-03-01   full list sweep + full detail sweep
  --incremental                   re-sweep trailing 30 days (late-upload safe)
  --verify                        API total vs DB count since 3/1; exit 2 on mismatch
  --limit N                       cap detail fetches this run (dev/smoke)
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))

import gtow_analyze_api as gapi
from ledger_distill import distill_hand

RAW = ROOT / "data" / "gtow_raw"
EPOCH_SINCE = "2026-02-28T16:00:00.000Z"     # 2026-03-01 Taipei
HAND_COLS = [
    "gtow_hand_id", "played_at", "tournament_id", "tournament_name",
    "tournament_buyin", "file_name", "site", "position", "hero_hand",
    "boards", "pot_type", "total_players", "preflop_depth_bb",
    "total_ev_loss_bb", "total_ev_loss_pct_pot", "avg_gto_score",
    "winloss_bb", "hand_correctness", "solution_status",
]
DEC_COLS = [
    "gtow_hand_id", "street", "decision_idx", "source", "grader", "family",
    "texture", "gtow_texture", "depth_band", "position", "pot_type", "facing",
    "taken_code", "best_code", "correctness", "ev_loss_bb", "ev_loss_pct_pot",
    "taken_freq", "freq_diff", "gto_score", "hand_eq", "pot_bb", "gametype",
    "confidence", "approx_flags", "excluded", "played_at",
]


def raw_paths(hand_id: str, played_at: str):
    ym = played_at[:7]
    return (RAW / "list" / f"{ym}.jsonl.gz",
            RAW / "detail" / ym / f"{hand_id}.json.gz")


def _ts(v):  # ISO str -> aware datetime for asyncpg
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else v


def _hand_vals(h: dict) -> list:
    vals = [h.get(c) for c in HAND_COLS]
    vals[1] = _ts(vals[1])
    return vals


def _hand_upsert_sql() -> str:
    cols = ", ".join(HAND_COLS)
    ph = ", ".join(f"${i+1}" for i in range(len(HAND_COLS)))
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in HAND_COLS if c != "gtow_hand_id")
    return (f"INSERT INTO ledger_hands ({cols}) VALUES ({ph}) "
            f"ON CONFLICT (gtow_hand_id) DO UPDATE SET {upd}")


async def upsert_hand(conn, h: dict):
    await conn.execute(_hand_upsert_sql(), *_hand_vals(h))


async def upsert_hands_batch(conn, rows: list[dict]):
    if rows:
        await conn.executemany(_hand_upsert_sql(), [_hand_vals(h) for h in rows])


async def upsert_decisions(conn, decs: list[dict]):
    for d in decs:
        vals = [d.get(c) for c in DEC_COLS]
        vals[DEC_COLS.index("played_at")] = _ts(d["played_at"])
        vals[DEC_COLS.index("approx_flags")] = json.dumps(d["approx_flags"])
        cols = ", ".join(DEC_COLS)
        ph = ", ".join(f"${i+1}" for i in range(len(DEC_COLS)))
        upd = ", ".join(f"{c}=EXCLUDED.{c}"
                        for c in DEC_COLS if c not in ("gtow_hand_id", "street", "decision_idx"))
        await conn.execute(
            f"INSERT INTO ledger_decisions ({cols}) VALUES ({ph}) "
            f"ON CONFLICT (gtow_hand_id, street, decision_idx) DO UPDATE SET {upd}", *vals)


_LIST_BATCH = 500


async def sweep_list(conn, since_iso: str) -> tuple[int, int]:
    # Preload known ids once (one query) instead of a SELECT per hand — the
    # 30-day incremental re-sweep is mostly already-known hands.
    known_ids = {r["gtow_hand_id"]
                 for r in await conn.fetch("SELECT gtow_hand_id FROM ledger_hands")}
    new = known = 0
    batch: list[dict] = []
    empty_detail = {"game_analysis": {"game_points": []}}
    for row in gapi.iter_all_hands(since_iso):
        if row["hand_id"] in known_ids:
            known += 1
            continue
        known_ids.add(row["hand_id"])
        lp, _ = raw_paths(row["hand_id"], row["played_at"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(lp, "at") as f:
            f.write(json.dumps(row) + "\n")
        hand_row, _ = distill_hand(row, empty_detail)
        batch.append(hand_row)
        new += 1
        if len(batch) >= _LIST_BATCH:
            await upsert_hands_batch(conn, batch); batch = []
            print(f"  list sweep: {new} new...", flush=True)
    await upsert_hands_batch(conn, batch)
    return new, known


async def sweep_detail(conn, limit: int | None) -> tuple[int, int]:
    rows = await conn.fetch(
        "SELECT gtow_hand_id, played_at FROM ledger_hands "
        "WHERE NOT detail_fetched ORDER BY played_at")
    fetched = ndec = 0
    for r in rows:
        if limit and fetched >= limit:
            break
        hid = r["gtow_hand_id"]
        played = r["played_at"].isoformat()
        _, dp = raw_paths(hid, played)
        dp.parent.mkdir(parents=True, exist_ok=True)
        det = gapi.hand_detail(hid)
        with gzip.open(dp, "wt") as f:
            json.dump(det, f)
        lp, _ = raw_paths(hid, played)
        list_row = _find_list_row(lp, hid)
        hand_row, decs = distill_hand(list_row, det)
        async with conn.transaction():
            await upsert_hand(conn, hand_row)
            await upsert_decisions(conn, decs)
            await conn.execute(
                "UPDATE ledger_hands SET detail_fetched=true, raw_path=$2 "
                "WHERE gtow_hand_id=$1", hid, str(dp.relative_to(ROOT)))
        fetched += 1
        ndec += len(decs)
        if fetched % 100 == 0:
            print(f"  detail sweep: {fetched}/{len(rows)}", flush=True)
    return fetched, ndec


def _find_list_row(list_path: Path, hand_id: str) -> dict:
    with gzip.open(list_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["hand_id"] == hand_id:
                return row
    raise RuntimeError(f"list row for {hand_id} not in {list_path}")


async def verify(conn) -> int:
    api_total = gapi.list_hands(EPOCH_SINCE, limit=1)["total"]
    db_total = await conn.fetchval("SELECT count(*) FROM ledger_hands")
    if api_total == db_total:
        print(f"VERIFY OK api={api_total} db={db_total}")
        return 0
    print(f"VERIFY MISMATCH api={api_total} db={db_total}")
    return 2


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import os
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        if a.verify:
            return await verify(conn)
        if a.incremental:
            since_dt = datetime.now(timezone.utc) - timedelta(days=30)
            since = since_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            since = f"{a.since}T00:00:00.000Z" if "T" not in a.since else a.since
        n_new, n_known = await sweep_list(conn, since)
        n_det, n_dec = await sweep_detail(conn, a.limit)
        print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} skipped={n_known}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
