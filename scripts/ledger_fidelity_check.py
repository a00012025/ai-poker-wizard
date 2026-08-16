#!/usr/bin/env python3
"""Fidelity check: 20 random lossy hands — ledger vs re-fetched API detail."""
import asyncio
import os
import random
from pathlib import Path
from zoneinfo import ZoneInfo

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
from gtow_analyze_api import hand_detail

TPE = ZoneInfo("Asia/Taipei")


async def main():
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    hands = await conn.fetch(
        "SELECT gtow_hand_id, played_at, total_ev_loss_bb FROM ledger_hands "
        "WHERE total_ev_loss_bb > 0.1 AND detail_fetched AND source='online' "
        "ORDER BY random() LIMIT 20")
    lines, mismatches = [], 0
    for h in hands:
        det = hand_detail(h["gtow_hand_id"])
        if det is None:
            continue
        api_loss = sum(
            float(a["ev_loss"]) for gp in det["game_analysis"]["game_points"]
            for a in (gp.get("analysis_solved") or {}).get("available_actions", [])
            if a.get("selected") and a.get("ev_loss") is not None)
        db_loss = await conn.fetchval(
            "SELECT COALESCE(sum(ev_loss_bb),0) FROM ledger_decisions WHERE gtow_hand_id=$1",
            h["gtow_hand_id"])
        ok = abs(api_loss - db_loss) < 1e-4
        mismatches += (not ok)
        lines.append(f"| {h['gtow_hand_id'][:8]} | {h['played_at'].astimezone(TPE):%m-%d} "
                     f"| {db_loss:.3f} | {api_loss:.3f} | {'✅' if ok else '❌'} |")
    report = ("# Fidelity check (20 random lossy hands)\n\n"
              "| hand | date | ledger bb | api bb | match |\n|---|---|---|---|---|\n"
              + "\n".join(lines) + f"\n\nmismatches: {mismatches}/{len(lines)}\n")
    (ROOT / "data/scorecards/fidelity_report.md").write_text(report)
    print(report)
    await conn.close()
    return 1 if mismatches else 0


raise SystemExit(asyncio.run(main()))
