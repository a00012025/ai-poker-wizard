#!/usr/bin/env python3
"""Rebuild ledger_sessions from ledger_hands timestamps (gap>60min clustering)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

GAP = timedelta(minutes=60)
WINDOW = timedelta(minutes=10)


def cluster_sessions(hands: list[dict]) -> list[dict]:
    sessions, cur = [], []
    for h in hands:
        if cur and h["played_at"] - cur[-1]["played_at"] > GAP:
            sessions.append(_finish(cur)); cur = []
        cur.append(h)
    if cur:
        sessions.append(_finish(cur))
    return sessions


def _max_concurrent_tables(hands: list[dict]) -> int:
    """Max distinct tournaments within a ±WINDOW neighborhood.

    The old implementation rebuilt a set by scanning every hand for every
    hand in a session (O(n²)); a 36k-hand rebuild spent most of its time here.
    Hands are already sorted by played_at, so a sliding window preserves the
    same semantics in O(n).
    """
    counts: dict[str, int] = {}
    left = right = 0
    max_cc = 1
    for h in hands:
        lo = h["played_at"] - WINDOW
        hi = h["played_at"] + WINDOW
        while right < len(hands) and hands[right]["played_at"] <= hi:
            tid = hands[right].get("tournament_id")
            if tid:
                counts[tid] = counts.get(tid, 0) + 1
            right += 1
        while left < len(hands) and hands[left]["played_at"] < lo:
            tid = hands[left].get("tournament_id")
            if tid:
                nxt = counts.get(tid, 0) - 1
                if nxt > 0:
                    counts[tid] = nxt
                else:
                    counts.pop(tid, None)
            left += 1
        max_cc = max(max_cc, len(counts) or 1)
    return max_cc


def _finish(hands: list[dict]) -> dict:
    tourneys = sorted({h["tournament_id"] for h in hands if h["tournament_id"]})
    max_cc = _max_concurrent_tables(hands)
    start, end = hands[0]["played_at"], hands[-1]["played_at"]
    return {"started_at": start, "ended_at": end,
            "duration_min": (end - start).total_seconds() / 60,
            "tournaments": tourneys, "max_concurrent_tables": max_cc,
            "hands_count": len(hands),
            "hand_ids": [h["gtow_hand_id"] for h in hands]}


async def rebuild():
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        started = time.monotonic()
        # online-only: live hands carry a synthetic capture timestamp, not a
        # real play time — clustering them would fabricate sessions.
        fetch_started = time.monotonic()
        rows = await conn.fetch(
            "SELECT gtow_hand_id, played_at, tournament_id FROM ledger_hands "
            "WHERE source='online' ORDER BY played_at")
        fetch_s = time.monotonic() - fetch_started
        cluster_started = time.monotonic()
        sessions = cluster_sessions([dict(r) for r in rows])
        cluster_s = time.monotonic() - cluster_started
        write_started = time.monotonic()
        async with conn.transaction():
            await conn.execute("UPDATE ledger_hands SET session_id=NULL")
            await conn.execute("DELETE FROM ledger_sessions")
            assignments: list[tuple[str, int]] = []
            for s in sessions:
                sid = await conn.fetchval(
                    "INSERT INTO ledger_sessions (started_at, ended_at, duration_min, "
                    "tournaments, max_concurrent_tables, hands_count) "
                    "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    s["started_at"], s["ended_at"], s["duration_min"],
                    json.dumps(s["tournaments"]), s["max_concurrent_tables"], s["hands_count"])
                assignments.extend((hid, sid) for hid in s["hand_ids"])
            await conn.execute(
                "CREATE TEMP TABLE tmp_ledger_session_assign "
                "(gtow_hand_id text PRIMARY KEY, session_id bigint) ON COMMIT DROP")
            await conn.copy_records_to_table(
                "tmp_ledger_session_assign",
                records=assignments,
                columns=("gtow_hand_id", "session_id"),
            )
            await conn.execute(
                "UPDATE ledger_hands h SET session_id=a.session_id "
                "FROM tmp_ledger_session_assign a "
                "WHERE h.gtow_hand_id=a.gtow_hand_id AND h.source='online'")
        write_s = time.monotonic() - write_started
        elapsed_s = time.monotonic() - started
        print(
            "[session-perf] rebuild "
            f"elapsed_s={elapsed_s:.3f} fetch_s={fetch_s:.3f} "
            f"cluster_s={cluster_s:.3f} write_s={write_s:.3f} "
            f"hands={len(rows)} sessions={len(sessions)}",
            flush=True,
        )
        print(f"SESSIONS rebuilt: {len(sessions)}")
    finally:
        await conn.close()


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        asyncio.run(rebuild())
