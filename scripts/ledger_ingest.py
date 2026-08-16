#!/usr/bin/env python3
"""Ingest GTOW Analyze hands into the ledger. Idempotent, resumable, loud.

Modes:
  --backfill --since 2026-03-01   full list sweep + full detail sweep
  --incremental                   re-sweep latest ingested hand minus overlap
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
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import gtow_analyze_api as gapi
from ledger_distill import (
    ListOnlyReconstructionError,
    distill_hand,
    distill_hand_from_list,
    distill_hand_row,
    should_skip_zeroloss_detail,
)

RAW = ROOT / "data" / "gtow_raw"
GTOW_PLAYED_AT_TZ = ZoneInfo(os.getenv("GTOW_PLAYED_AT_TZ", "Asia/Taipei"))
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


def _gtow_archive_month(played_at) -> str:
    """Month key for GTOW raw archives.

    GTOW Analyze's `played_at` currently arrives as a GTOW/site wall-clock
    timestamp suffixed with `Z` (for example `2026-07-22T19:35:19Z` during a
    19:35 Taipei session).  Raw list/detail archives are keyed by that wall
    clock month.  Once stored in Postgres we normalize to real UTC, so datetime
    inputs must be converted back to the GTOW wall-clock timezone before
    choosing the archive month.
    """
    if isinstance(played_at, str):
        return played_at[:7]
    if isinstance(played_at, datetime):
        dt = played_at if played_at.tzinfo else played_at.replace(tzinfo=timezone.utc)
        return dt.astimezone(GTOW_PLAYED_AT_TZ).strftime("%Y-%m")
    raise TypeError(f"unsupported played_at for raw archive path: {played_at!r}")


def raw_paths(hand_id: str, played_at):
    ym = _gtow_archive_month(played_at)
    return (RAW / "list" / f"{ym}.jsonl.gz",
            RAW / "detail" / ym / f"{hand_id}.json.gz")


def _ts(v):  # GTOW Analyze played_at -> real UTC datetime for asyncpg
    if not isinstance(v, str):
        return v
    if v.endswith("Z"):
        # Despite the suffix, GTOW Analyze returns the HH's displayed/site
        # wall-clock time.  Treat it as GTOW_PLAYED_AT_TZ local time and store
        # a real UTC instant so DB day/week windows are honest.
        local = datetime.fromisoformat(v[:-1]).replace(tzinfo=GTOW_PLAYED_AT_TZ)
        return local.astimezone(timezone.utc)
    return datetime.fromisoformat(v)


def _gtow_wall_clock_iso(dt: datetime) -> str:
    """Format a datetime for GTOW Analyze filters.

    GTOW's list API returns and filters on site wall-clock strings with a `Z`
    suffix, so convert stored UTC back to GTOW_PLAYED_AT_TZ before formatting.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(GTOW_PLAYED_AT_TZ)
    return local.strftime("%Y-%m-%dT%H:%M:%S.000Z")


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


def _decision_upsert_sql() -> str:
    cols = ", ".join(DEC_COLS)
    ph = ", ".join(f"${i+1}" for i in range(len(DEC_COLS)))
    upd = ", ".join(f"{c}=EXCLUDED.{c}"
                    for c in DEC_COLS if c not in ("gtow_hand_id", "street", "decision_idx"))
    return (
        f"INSERT INTO ledger_decisions ({cols}) VALUES ({ph}) "
        f"ON CONFLICT (gtow_hand_id, street, decision_idx) DO UPDATE SET {upd}"
    )


async def upsert_hand(conn, h: dict):
    await conn.execute(_hand_upsert_sql(), *_hand_vals(h))


async def upsert_hands_batch(conn, rows: list[dict]):
    if rows:
        await conn.executemany(_hand_upsert_sql(), [_hand_vals(h) for h in rows])


def _decision_vals(d: dict) -> list:
    vals = [d.get(c) for c in DEC_COLS]
    vals[DEC_COLS.index("played_at")] = _ts(d["played_at"])
    vals[DEC_COLS.index("approx_flags")] = json.dumps(d["approx_flags"])
    vals[DEC_COLS.index("spot_keys")] = (
        json.dumps(d["spot_keys"]) if d.get("spot_keys") is not None else None)
    vals[DEC_COLS.index("discarded")] = bool(d.get("discarded", False))
    vals[DEC_COLS.index("limp_origin")] = bool(d.get("limp_origin", False))
    return vals


async def upsert_decisions(conn, decs: list[dict]):
    if decs:
        await conn.executemany(_decision_upsert_sql(), [_decision_vals(d) for d in decs])


_LIST_BATCH = 500
# GTOW rejects limits above 500. Use the maximum to minimize serial list calls.
_LIST_PAGE_SIZE = max(1, min(500, int(os.getenv("GTOW_LIST_PAGE_SIZE", "500"))))

# Detail-fetch concurrency. The rate-limit probe (2026-07-21) found GTOW never
# 429s but soft-throttles via latency above ~10 req/s; detail latency (~0.8s)
# dominates, so concurrency (not a lower interval) is the lever. Defaults picked
# at the C=8 knee (~7.5 req/s, ~6x the old serial ~1.3/s) with server headroom
# for the live bot sharing the token. Env-tunable up to the ~C=12 ceiling.
_DETAIL_CONCURRENCY = max(1, int(os.getenv("GTOW_DETAIL_CONCURRENCY", "8")))
_DETAIL_MIN_INTERVAL = max(0.0, float(os.getenv("GTOW_DETAIL_MIN_INTERVAL", "0.08")))
_DETAIL_BATCH = 200          # fetch+write in batches to bound memory
_DETAIL_PROGRESS_EVERY = max(1, int(os.getenv("GTOW_DETAIL_PROGRESS_EVERY", "10")))
_WRITE_PROGRESS_EVERY = max(1, int(os.getenv("GTOW_WRITE_PROGRESS_EVERY", "10")))
_LIST_ONLY_PROGRESS_EVERY = max(1, int(os.getenv("GTOW_LIST_ONLY_PROGRESS_EVERY", "100")))
_DETAIL_PREP_PROGRESS_EVERY = max(1, int(os.getenv("GTOW_DETAIL_PREP_PROGRESS_EVERY", "100")))
_LIST_ONLY_BATCH = max(1, int(os.getenv("GTOW_LIST_ONLY_BATCH", "200")))
_INCREMENTAL_OVERLAP_HOURS = max(0.0, float(os.getenv("GTOW_INCREMENTAL_OVERLAP_HOURS", "12")))
_INVALID_ACTIONS = object()


def _perf(event: str, **fields) -> None:
    """Emit machine-readable ingest timing for Docker logs.

    `src.ingest_runner` forwards these lines to the application logger without
    surfacing them in the Telegram progress message.
    """
    parts = [f"[ledger-perf] {event}"]
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


async def _fetch_details_concurrent(hids: list[str], on_progress=None,
                                    _done_base: int = 0, _total: int | None = None
                                    ) -> dict:
    """Fetch hand details for `hids` concurrently under a semaphore + a shared
    min-interval pacer (threads run the blocking client via asyncio.to_thread).
    Returns {hid: detail_or_None_or_invalid_sentinel}. Backoff/soft-status
    handling stays in the client; None means skip-and-retry-later (upload not
    ready / forbidden), while invalid actions are a permanent per-hand skip."""
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
            try:
                results[hid] = await asyncio.to_thread(
                    gapi.hand_detail, hid, throttle=False)
            except gapi.InvalidHandActionsError:
                results[hid] = _INVALID_ACTIONS
        done[0] += 1
        if on_progress and (_done_base + done[0]) % _DETAIL_PROGRESS_EVERY == 0:
            on_progress(_done_base + done[0], total)

    await asyncio.gather(*(_one(h) for h in hids))
    return results


async def sweep_list(conn, since_iso: str) -> tuple[int, int]:
    # Preload known ids once (one query) instead of a SELECT per hand — the
    # 30-day incremental re-sweep is mostly already-known hands.
    started = time.monotonic()
    known_started = time.monotonic()
    known_ids = {r["gtow_hand_id"]
                 for r in await conn.fetch("SELECT gtow_hand_id FROM ledger_hands")}
    known_query_s = time.monotonic() - known_started
    new = known = scanned = total = 0
    batch: list[dict] = []
    pages = 0
    api_s = archive_write_s = db_write_s = 0.0
    archived_ids_by_path: dict[Path, set[str]] = {}
    offset = 0
    while True:
        api_started = time.monotonic()
        page = gapi.list_hands(since_iso, offset=offset, limit=_LIST_PAGE_SIZE)
        api_s += time.monotonic() - api_started
        pages += 1
        items = page.get("items", [])
        total = int(page.get("total", 0) or 0)
        if not items:
            break
        archive_rows_by_path: dict[Path, list[dict]] = defaultdict(list)
        for row in items:
            scanned += 1
            hid = row["hand_id"]
            lp, _ = raw_paths(hid, row["played_at"])
            archived_ids = archived_ids_by_path.get(lp)
            if archived_ids is None:
                archived_ids = set()
                if lp.exists():
                    with gzip.open(lp, "rt") as f:
                        for line in f:
                            archived = json.loads(line)
                            if archived.get("hand_id"):
                                archived_ids.add(archived["hand_id"])
                archived_ids_by_path[lp] = archived_ids
            if hid not in archived_ids:
                archived_ids.add(hid)
                archive_rows_by_path[lp].append(row)

            if hid in known_ids:
                known += 1
                continue
            known_ids.add(hid)
            hand_row = distill_hand_row(row)
            hand_row["detail_status"] = "pending"
            batch.append(hand_row)
            new += 1
            if len(batch) >= _LIST_BATCH:
                db_started = time.monotonic()
                await upsert_hands_batch(conn, batch)
                db_write_s += time.monotonic() - db_started
                batch = []
                print(f"  list write: {new} new...", flush=True)
        archive_started = time.monotonic()
        for lp, archive_rows in archive_rows_by_path.items():
            lp.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(lp, "at") as f:
                for row in archive_rows:
                    f.write(json.dumps(row) + "\n")
        archive_write_s += time.monotonic() - archive_started
        offset += len(items)
        print(f"  list scan: {scanned}/{total} ({new} new)", flush=True)
        if offset >= total:
            break
    db_started = time.monotonic()
    await upsert_hands_batch(conn, batch)
    db_write_s += time.monotonic() - db_started
    _perf("list", since=since_iso, elapsed_s=time.monotonic() - started,
          known_query_s=known_query_s, api_s=api_s,
          archive_write_s=archive_write_s, db_write_s=db_write_s,
          pages=pages, scanned=scanned, new=new, known=known)
    return new, known


async def incremental_since(conn) -> str:
    """Return the GTOW list lower bound for incremental ingest.

    The first implementation always re-swept trailing 30 days to catch late HH
    uploads, which is safe but scans thousands of already-known hands. Use the
    latest ingested online hand as a watermark with a generous overlap; verify
    still escalates to full backfill if this misses anything.
    """
    max_played = await conn.fetchval(
        "SELECT max(played_at) FROM ledger_hands WHERE source='online'")
    if max_played is None:
        since_dt = datetime.now(timezone.utc) - timedelta(days=30)
    else:
        since_dt = max_played - timedelta(hours=_INCREMENTAL_OVERLAP_HOURS)
    since = _gtow_wall_clock_iso(since_dt)
    _perf("incremental_since", max_played=max_played.isoformat() if max_played else None,
          overlap_hours=_INCREMENTAL_OVERLAP_HOURS, since=since)
    return since


async def _repair_missing_list_rows(rows) -> int:
    """Recover raw list rows for pending DB hands after archive loss.

    The list endpoint supports a narrow played-at range, so repair each missing
    hand around its exact wall-clock timestamp instead of re-scanning all
    history. This is primarily for a fresh/replaced host whose Supabase ledger
    survived but whose local bind-mounted raw archive did not.
    """
    repaired = 0
    for idx, r in enumerate(rows, start=1):
        played_at = r["played_at"]
        since = _gtow_wall_clock_iso(played_at - timedelta(seconds=1))
        until = _gtow_wall_clock_iso(played_at + timedelta(seconds=1))
        offset = 0
        row = None
        while True:
            page = await asyncio.to_thread(
                gapi.list_hands, since, until, offset, _LIST_PAGE_SIZE)
            items = page.get("items", [])
            row = next((item for item in items
                        if item.get("hand_id") == r["gtow_hand_id"]), None)
            if row is not None:
                break
            offset += len(items)
            if not items or offset >= int(page.get("total", 0) or 0):
                break
        if row is None:
            continue
        lp, _ = raw_paths(row["hand_id"], row["played_at"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(lp, "at") as f:
            f.write(json.dumps(row) + "\n")
        repaired += 1
        if idx % 10 == 0:
            print(f"  archive repair sweep: {idx}/{len(rows)} ({repaired} repaired)",
                  flush=True)
    if rows:
        print(f"  archive repair sweep: {len(rows)}/{len(rows)} ({repaired} repaired)",
              flush=True)
    return repaired


async def sweep_detail(conn, limit: int | None, *, backfill_skipped: bool = False
                       ) -> tuple[int, int, int, int, int]:
    started = time.monotonic()
    status = "skipped_zeroloss" if backfill_skipped else "pending"
    pending_started = time.monotonic()
    rows = await conn.fetch(
        "SELECT gtow_hand_id, played_at FROM ledger_hands "
        "WHERE detail_status=$1 AND source='online' ORDER BY played_at", status)
    pending_query_s = time.monotonic() - pending_started
    load_started = time.monotonic()
    list_rows = _load_list_rows(rows, allow_missing=True)
    missing_archive_rows = [r for r in rows if r["gtow_hand_id"] not in list_rows]
    repaired_archive_rows = 0
    if missing_archive_rows:
        print(f"  detail prep: repairing {len(missing_archive_rows)} pending hands "
              "missing raw list archive", flush=True)
        repaired_archive_rows = await _repair_missing_list_rows(missing_archive_rows)
        list_rows = _load_list_rows(rows, allow_missing=True)
    missing_archive_hids = [r["gtow_hand_id"] for r in rows
                            if r["gtow_hand_id"] not in list_rows]
    if missing_archive_hids:
        print(
            "  detail prep: skipped "
            f"{len(missing_archive_hids)} pending hands missing raw list archive "
            "after targeted repair (left pending)",
            flush=True,
        )
        rows = [r for r in rows if r["gtow_hand_id"] in list_rows]
    load_list_rows_s = time.monotonic() - load_started
    fetched = ndec = skipped_nodata = skipped_zeroloss = skipped_invalid = 0
    reconstruct_fallback = 0
    # Pass 1 (serial, fast): handle list-only zero-loss hands inline and collect
    # the ones that actually need a network detail fetch. `limit` caps fetch
    # attempts this run (dev/smoke); list-only hands never count against it.
    needs_detail: list[tuple[str, Path, dict]] = []
    zero_hids: list[str] = []
    zero_decs: list[dict] = []
    prep_started = time.monotonic()
    zero_write_s = fetch_s = detail_write_s = 0.0

    async def flush_zero_batch():
        nonlocal zero_write_s
        if not zero_hids:
            return
        flush_started = time.monotonic()
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM ledger_decisions WHERE gtow_hand_id = ANY($1::text[])",
                zero_hids)
            await upsert_decisions(conn, zero_decs)
            await conn.execute(
                "UPDATE ledger_hands SET detail_fetched=false, "
                "detail_status='skipped_zeroloss' "
                "WHERE gtow_hand_id = ANY($1::text[])",
                zero_hids)
        zero_write_s += time.monotonic() - flush_started
        zero_hids.clear()
        zero_decs.clear()

    for idx, r in enumerate(rows, start=1):
        if limit and len(needs_detail) >= limit:
            break
        hid = r["gtow_hand_id"]
        played = r["played_at"]
        _, dp = raw_paths(hid, played)
        dp.parent.mkdir(parents=True, exist_ok=True)
        list_row = list_rows[hid]
        if idx % _DETAIL_PREP_PROGRESS_EVERY == 0:
            print(f"  detail prep: {idx}/{len(rows)}", flush=True)
        if not backfill_skipped and should_skip_zeroloss_detail(list_row):
            try:
                _hand_row, decs = distill_hand_from_list(list_row)
            except ListOnlyReconstructionError as exc:
                reconstruct_fallback += 1
                print(f"  list-only fallback {hid}: {exc}", flush=True)
            else:
                zero_hids.append(hid)
                zero_decs.extend(decs)
                skipped_zeroloss += 1
                ndec += len(decs)
                if len(zero_hids) >= _LIST_ONLY_BATCH:
                    await flush_zero_batch()
                if idx % _LIST_ONLY_PROGRESS_EVERY == 0:
                    print(f"  list-only sweep: {idx}/{len(rows)} "
                          f"({skipped_zeroloss} zero-loss)", flush=True)
                continue
        needs_detail.append((hid, dp, list_row))
        if idx % _LIST_ONLY_PROGRESS_EVERY == 0:
            print(f"  list-only sweep: {idx}/{len(rows)} "
                  f"({skipped_zeroloss} zero-loss)", flush=True)
    await flush_zero_batch()
    if rows:
        print(f"  list-only sweep: {min(len(rows), idx if 'idx' in locals() else 0)}/{len(rows)} "
              f"({skipped_zeroloss} zero-loss)", flush=True)
    if rows:
        print(f"  detail prep: {min(len(rows), idx if 'idx' in locals() else 0)}/{len(rows)}", flush=True)
    prep_s = time.monotonic() - prep_started

    # Pass 2 (concurrent fetch) + pass 3 (serial DB write), batched to bound
    # memory. The "detail sweep: done/total" print format is unchanged so the
    # live progress bar's parser keeps working; total is the fetch-needing count.
    total_detail = len(needs_detail)

    def _emit(done, tot):
        print(f"  detail sweep: {done}/{tot}", flush=True)

    for start in range(0, total_detail, _DETAIL_BATCH):
        chunk = needs_detail[start:start + _DETAIL_BATCH]
        fetch_started = time.monotonic()
        dets = await _fetch_details_concurrent(
            [hid for hid, _, _ in chunk], on_progress=_emit,
            _done_base=start, _total=total_detail)
        fetch_s += time.monotonic() - fetch_started
        written_in_chunk = 0
        hand_rows: list[dict] = []
        hand_updates: list[tuple[str, str]] = []
        all_decs: list[dict] = []
        fetched_hids: list[str] = []
        invalid_hids: list[str] = []
        write_started = time.monotonic()
        for hid, dp, list_row in chunk:
            det = dets.get(hid)
            if det is _INVALID_ACTIONS:
                invalid_hids.append(hid)
                skipped_invalid += 1
                continue
            if det is None:
                # no retrievable analysis yet (upload still processing /
                # forbidden / no solution) — leave detail_fetched=false so a
                # later run retries.
                skipped_nodata += 1
                continue
            with gzip.open(dp, "wt") as f:
                json.dump(det, f)
            hand_row, decs = distill_hand(list_row, det)
            hand_rows.append(hand_row)
            fetched_hids.append(hid)
            all_decs.extend(decs)
            hand_updates.append((hid, str(dp.relative_to(ROOT))))
            fetched += 1
            ndec += len(decs)
            written_in_chunk += 1
            if written_in_chunk % _WRITE_PROGRESS_EVERY == 0:
                print(f"  detail write: {start + written_in_chunk}/{total_detail}", flush=True)
        if fetched_hids or invalid_hids:
            async with conn.transaction():
                if fetched_hids:
                    await upsert_hands_batch(conn, hand_rows)
                    await conn.execute(
                        "DELETE FROM ledger_decisions WHERE gtow_hand_id = ANY($1::text[])",
                        fetched_hids)
                    await upsert_decisions(conn, all_decs)
                    await conn.executemany(
                        "UPDATE ledger_hands SET detail_fetched=true, "
                        "detail_status='fetched', raw_path=$2 WHERE gtow_hand_id=$1",
                        hand_updates)
                if invalid_hids:
                    await conn.execute(
                        "DELETE FROM ledger_decisions WHERE gtow_hand_id = ANY($1::text[])",
                        invalid_hids)
                    await conn.execute(
                        "UPDATE ledger_hands SET detail_fetched=false, "
                        "detail_status='skipped_invalid_actions', raw_path=NULL "
                        "WHERE gtow_hand_id = ANY($1::text[])",
                        invalid_hids)
        detail_write_s += time.monotonic() - write_started
    if total_detail:
        print(f"  detail sweep: {total_detail}/{total_detail}", flush=True)
    if skipped_nodata:
        print(f"  detail sweep: skipped {skipped_nodata} hands with no retrievable "
              f"analysis yet (will retry next run)", flush=True)
    if skipped_invalid:
        print(f"  detail sweep: skipped {skipped_invalid} hands permanently rejected "
              f"by GTOW as incorrect actions", flush=True)
    _perf("detail", elapsed_s=time.monotonic() - started,
          pending_query_s=pending_query_s, load_list_rows_s=load_list_rows_s,
          prep_s=prep_s, zero_write_s=zero_write_s, fetch_s=fetch_s,
          detail_write_s=detail_write_s, pending=len(rows),
          needs_detail=total_detail, fetched=fetched,
          skipped_nodata=skipped_nodata, skipped_zeroloss=skipped_zeroloss,
          skipped_invalid=skipped_invalid,
          repaired_archive_rows=repaired_archive_rows,
          skipped_missing_archive=len(missing_archive_hids),
          reconstruct_fallback=reconstruct_fallback, decisions=ndec)
    return fetched, ndec, skipped_zeroloss, reconstruct_fallback, skipped_invalid


def _find_list_row(list_path: Path, hand_id: str) -> dict:
    with gzip.open(list_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["hand_id"] == hand_id:
                return row
    raise RuntimeError(f"list row for {hand_id} not in {list_path}")


def _load_list_rows(rows, *, allow_missing: bool = False) -> dict[str, dict]:
    """Load all list archive rows needed by a detail pass.

    The old path called _find_list_row() per pending hand, reopening and
    scanning the same monthly gzip thousands of times before the first detail
    progress line. Load each touched month once instead.
    """
    wanted_by_path: dict[Path, set[str]] = defaultdict(set)
    for r in rows:
        hid = r["gtow_hand_id"]
        played = r["played_at"]
        lp, _ = raw_paths(hid, played)
        wanted_by_path[lp].add(hid)

    found: dict[str, dict] = {}
    for list_path, wanted in wanted_by_path.items():
        if allow_missing and not list_path.exists():
            continue
        with gzip.open(list_path, "rt") as f:
            for line in f:
                row = json.loads(line)
                hid = row.get("hand_id")
                if hid in wanted:
                    found[hid] = row
                    if len(found.keys() & wanted) == len(wanted):
                        break

    missing = [r["gtow_hand_id"] for r in rows if r["gtow_hand_id"] not in found]
    if missing and not allow_missing:
        raise RuntimeError(
            f"list rows missing from raw archive: {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )
    return found


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
            n_det, n_dec, n_zero, n_fallback, n_invalid = await sweep_detail(
                conn, a.limit, backfill_skipped=True)
            print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} known={n_known} "
                  f"skipped_zeroloss={n_zero} skipped_invalid={n_invalid} "
                  f"reconstruct_fallback={n_fallback}")
            return 0
        if a.incremental:
            since = await incremental_since(conn)
        else:
            since = f"{a.since}T00:00:00.000Z" if "T" not in a.since else a.since
        n_new, n_known = await sweep_list(conn, since)
        n_det, n_dec, n_zero, n_fallback, n_invalid = await sweep_detail(conn, a.limit)
        print(f"INGEST list={n_new} detail={n_det} decisions={n_dec} known={n_known} "
              f"skipped_zeroloss={n_zero} skipped_invalid={n_invalid} "
              f"reconstruct_fallback={n_fallback}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
