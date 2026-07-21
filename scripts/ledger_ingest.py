#!/usr/bin/env python3
"""Ingest GTOW Analyze hands into the ledger. Idempotent, resumable, loud.

Modes:
  --backfill --since 2026-03-01   full list sweep + full detail sweep
  --incremental                   re-sweep trailing 30 days (late-upload safe)
  --verify                        API total vs DB count since 3/1; exit 2 on mismatch
  --backfill-skipped              fetch detail for list-only zero-loss hands
  --limit N                       cap detail fetches this run (dev/smoke)
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))

import gtow_analyze_api as gapi
from ledger_distill import (
    ListOnlyReconstructionError,
    distill_hand,
    distill_hand_from_list,
    distill_hand_row,
    should_skip_zeroloss_detail,
)

RAW = ROOT / "data" / "gtow_raw"
EPOCH_SINCE = "2026-02-28T16:00:00.000Z"     # 2026-03-01 Taipei
HAND_COLS = [
    "gtow_hand_id", "played_at", "tournament_id", "tournament_name",
    "tournament_buyin", "file_name", "site", "position", "hero_hand",
    "boards", "pot_type", "total_players", "preflop_depth_bb",
    "total_ev_loss_bb", "total_ev_loss_pct_pot", "avg_gto_score",
    "winloss_bb", "hand_correctness", "solution_status", "detail_status",
]
DEC_COLS = [
    "gtow_hand_id", "street", "decision_idx", "source", "grader",
    "gtow_texture", "depth_band", "position", "pot_type", "facing",
    "played_depth_bb", "solver_depth_bb",
    "taken_code", "best_code", "correctness", "ev_loss_bb", "ev_loss_pct_pot",
    "taken_freq", "freq_diff", "gto_score", "hand_eq", "pot_bb", "gametype",
    "confidence", "approx_flags", "excluded", "played_at",
    "spot_category", "spot_leaf", "spot_parent", "spot_keys",
    "hero_cat", "villain_cat", "ip_oop", "flop_seq", "turn_seq",
    "eff_stack", "board_suit", "board_conn", "board_paired",
    "discarded", "limp_origin",
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
    vals[HAND_COLS.index("detail_status")] = h.get("detail_status") or "pending"
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
        vals[DEC_COLS.index("spot_keys")] = (
            json.dumps(d["spot_keys"]) if d.get("spot_keys") is not None else None)
        vals[DEC_COLS.index("discarded")] = bool(d.get("discarded", False))
        vals[DEC_COLS.index("limp_origin")] = bool(d.get("limp_origin", False))
        cols = ", ".join(DEC_COLS)
        ph = ", ".join(f"${i+1}" for i in range(len(DEC_COLS)))
        upd = ", ".join(f"{c}=EXCLUDED.{c}"
                        for c in DEC_COLS if c not in ("gtow_hand_id", "street", "decision_idx"))
        await conn.execute(
            f"INSERT INTO ledger_decisions ({cols}) VALUES ({ph}) "
            f"ON CONFLICT (gtow_hand_id, street, decision_idx) DO UPDATE SET {upd}", *vals)


_LIST_BATCH = 500

# Detail-fetch concurrency. The rate-limit probe (2026-07-21) found GTOW never
# 429s but soft-throttles via latency above ~10 req/s; detail latency (~0.8s)
# dominates, so concurrency (not a lower interval) is the lever. Defaults picked
# at the C=8 knee (~7.5 req/s, ~6x the old serial ~1.3/s) with server headroom
# for the live bot sharing the token. Env-tunable up to the ~C=12 ceiling.
_DETAIL_CONCURRENCY = max(1, int(os.getenv("GTOW_DETAIL_CONCURRENCY", "8")))
_DETAIL_MIN_INTERVAL = max(0.0, float(os.getenv("GTOW_DETAIL_MIN_INTERVAL", "0.08")))
_DETAIL_BATCH = 200          # fetch+write in batches to bound memory


async def _fetch_details_concurrent(hids: list[str], on_progress=None,
                                    _done_base: int = 0, _total: int | None = None
                                    ) -> dict:
    """Fetch hand details for `hids` concurrently under a semaphore + a shared
    min-interval pacer (threads run the blocking client via asyncio.to_thread).
    Returns {hid: detail_or_None}. Backoff/soft-status handling stays in the
    client; None means skip-and-retry-later (upload not ready / forbidden)."""
    sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)
    pace_lock = asyncio.Lock()
    next_slot = [time.monotonic()]
    results: dict = {}
    done = [0]
    total = _total if _total is not None else len(hids)
    if hids:
        # Warm the token + client-id cache once, serially, so the first
        # concurrent batch doesn't race N threads into simultaneous mints /
        # first-write of .gtow_client_id. (sweep_list usually warms it first;
        # this makes a detail-only run safe too.)
        await asyncio.to_thread(gapi._headers)

    async def _one(hid: str):
        async with sem:
            async with pace_lock:            # space out issue times a touch
                now = time.monotonic()
                wait = max(0.0, next_slot[0] - now)
                next_slot[0] = max(now, next_slot[0]) + _DETAIL_MIN_INTERVAL
            if wait:
                await asyncio.sleep(wait)
            results[hid] = await asyncio.to_thread(gapi.hand_detail, hid,
                                                   throttle=False)
        done[0] += 1
        if on_progress and (_done_base + done[0]) % 20 == 0:
            on_progress(_done_base + done[0], total)

    await asyncio.gather(*(_one(h) for h in hids))
    return results


async def sweep_list(conn, since_iso: str) -> tuple[int, int]:
    # Preload known ids once (one query) instead of a SELECT per hand — the
    # 30-day incremental re-sweep is mostly already-known hands.
    known_ids = {r["gtow_hand_id"]
                 for r in await conn.fetch("SELECT gtow_hand_id FROM ledger_hands")}
    new = known = 0
    batch: list[dict] = []
    for row in gapi.iter_all_hands(since_iso):
        if row["hand_id"] in known_ids:
            known += 1
            continue
        known_ids.add(row["hand_id"])
        lp, _ = raw_paths(row["hand_id"], row["played_at"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(lp, "at") as f:
            f.write(json.dumps(row) + "\n")
        hand_row = distill_hand_row(row)
        hand_row["detail_status"] = "pending"
        batch.append(hand_row)
        new += 1
        if len(batch) >= _LIST_BATCH:
            await upsert_hands_batch(conn, batch); batch = []
            print(f"  list sweep: {new} new...", flush=True)
    await upsert_hands_batch(conn, batch)
    return new, known


async def sweep_detail(conn, limit: int | None, *, backfill_skipped: bool = False
                       ) -> tuple[int, int, int, int]:
    status = "skipped_zeroloss" if backfill_skipped else "pending"
    rows = await conn.fetch(
        "SELECT gtow_hand_id, played_at FROM ledger_hands "
        "WHERE detail_status=$1 AND source='online' ORDER BY played_at", status)
    fetched = ndec = skipped_nodata = skipped_zeroloss = reconstruct_fallback = 0
    # Pass 1 (serial, fast): handle list-only zero-loss hands inline and collect
    # the ones that actually need a network detail fetch. `limit` caps fetch
    # attempts this run (dev/smoke); list-only hands never count against it.
    needs_detail: list[tuple[str, Path, dict]] = []
    for r in rows:
        if limit and len(needs_detail) >= limit:
            break
        hid = r["gtow_hand_id"]
        played = r["played_at"].isoformat()
        _, dp = raw_paths(hid, played)
        dp.parent.mkdir(parents=True, exist_ok=True)
        lp, _ = raw_paths(hid, played)
        list_row = _find_list_row(lp, hid)
        if not backfill_skipped and should_skip_zeroloss_detail(list_row):
            try:
                _hand_row, decs = distill_hand_from_list(list_row)
            except ListOnlyReconstructionError as exc:
                reconstruct_fallback += 1
                print(f"  list-only fallback {hid}: {exc}", flush=True)
            else:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM ledger_decisions WHERE gtow_hand_id=$1", hid)
                    await upsert_decisions(conn, decs)
                    await conn.execute(
                        "UPDATE ledger_hands SET detail_fetched=false, "
                        "detail_status='skipped_zeroloss' WHERE gtow_hand_id=$1", hid)
                skipped_zeroloss += 1
                ndec += len(decs)
                if skipped_zeroloss % 500 == 0:
                    print(f"  list-only sweep: {skipped_zeroloss}/{len(rows)}", flush=True)
                continue
        needs_detail.append((hid, dp, list_row))

    # Pass 2 (concurrent fetch) + pass 3 (serial DB write), batched to bound
    # memory. The "detail sweep: done/total" print format is unchanged so the
    # live progress bar's parser keeps working; total is the fetch-needing count.
    total_detail = len(needs_detail)

    def _emit(done, tot):
        print(f"  detail sweep: {done}/{tot}", flush=True)

    for start in range(0, total_detail, _DETAIL_BATCH):
        chunk = needs_detail[start:start + _DETAIL_BATCH]
        dets = await _fetch_details_concurrent(
            [hid for hid, _, _ in chunk], on_progress=_emit,
            _done_base=start, _total=total_detail)
        for hid, dp, list_row in chunk:
            det = dets.get(hid)
            if det is None:
                # no retrievable analysis yet (upload still processing /
                # forbidden / no solution) — leave detail_fetched=false so a
                # later run retries.
                skipped_nodata += 1
                continue
            with gzip.open(dp, "wt") as f:
                json.dump(det, f)
            hand_row, decs = distill_hand(list_row, det)
            async with conn.transaction():
                await upsert_hand(conn, hand_row)
                await conn.execute(
                    "DELETE FROM ledger_decisions WHERE gtow_hand_id=$1", hid)
                await upsert_decisions(conn, decs)
                await conn.execute(
                    "UPDATE ledger_hands SET detail_fetched=true, detail_status='fetched', raw_path=$2 "
                    "WHERE gtow_hand_id=$1", hid, str(dp.relative_to(ROOT)))
            fetched += 1
            ndec += len(decs)
    if total_detail:
        print(f"  detail sweep: {total_detail}/{total_detail}", flush=True)
    if skipped_nodata:
        print(f"  detail sweep: skipped {skipped_nodata} hands with no retrievable "
              f"analysis yet (will retry next run)", flush=True)
    return fetched, ndec, skipped_zeroloss, reconstruct_fallback


def _find_list_row(list_path: Path, hand_id: str) -> dict:
    with gzip.open(list_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["hand_id"] == hand_id:
                return row
    raise RuntimeError(f"list row for {hand_id} not in {list_path}")


async def verify(conn) -> int:
    api_total = gapi.list_hands(EPOCH_SINCE, limit=1)["total"]
    db_total = await conn.fetchval(
        "SELECT count(*) FROM ledger_hands WHERE source='online'")
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
    ap.add_argument("--backfill-skipped", action="store_true")
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import os
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        if a.verify:
            return await verify(conn)
        if a.backfill_skipped:
            n_new = n_known = 0
            n_det, n_dec, n_zero, n_fallback = await sweep_detail(
                conn, a.limit, backfill_skipped=True)
            print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} known={n_known} "
                  f"skipped_zeroloss={n_zero} reconstruct_fallback={n_fallback}")
            return 0
        if a.incremental:
            since_dt = datetime.now(timezone.utc) - timedelta(days=30)
            since = since_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            since = f"{a.since}T00:00:00.000Z" if "T" not in a.since else a.since
        n_new, n_known = await sweep_list(conn, since)
        n_det, n_dec, n_zero, n_fallback = await sweep_detail(conn, a.limit)
        print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} known={n_known} "
              f"skipped_zeroloss={n_zero} reconstruct_fallback={n_fallback}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
