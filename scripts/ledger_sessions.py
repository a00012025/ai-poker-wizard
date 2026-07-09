#!/usr/bin/env python3
"""Rebuild ledger_sessions from ledger_hands timestamps (gap>60min clustering)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
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


def _finish(hands: list[dict]) -> dict:
    tourneys = sorted({h["tournament_id"] for h in hands if h["tournament_id"]})
    max_cc = 1
    for h in hands:
        cc = {g["tournament_id"] for g in hands
              if g["tournament_id"] and abs((g["played_at"] - h["played_at"]).total_seconds())
              <= WINDOW.total_seconds()}
        max_cc = max(max_cc, len(cc) or 1)
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
        rows = await conn.fetch(
            "SELECT gtow_hand_id, played_at, tournament_id FROM ledger_hands ORDER BY played_at")
        sessions = cluster_sessions([dict(r) for r in rows])
        async with conn.transaction():
            await conn.execute("UPDATE ledger_hands SET session_id=NULL")
            await conn.execute("DELETE FROM ledger_sessions")
            for s in sessions:
                sid = await conn.fetchval(
                    "INSERT INTO ledger_sessions (started_at, ended_at, duration_min, "
                    "tournaments, max_concurrent_tables, hands_count) "
                    "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    s["started_at"], s["ended_at"], s["duration_min"],
                    json.dumps(s["tournaments"]), s["max_concurrent_tables"], s["hands_count"])
                await conn.execute(
                    "UPDATE ledger_hands SET session_id=$1 WHERE gtow_hand_id = ANY($2)",
                    sid, s["hand_ids"])
        print(f"SESSIONS rebuilt: {len(sessions)}")
    finally:
        await conn.close()


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        asyncio.run(rebuild())
