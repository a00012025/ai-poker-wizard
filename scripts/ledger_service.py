#!/usr/bin/env python3
"""Ledger query service for the Telegram bot + LLM follow-up tools.

- resolve_owner_chat_id: N=1 system owner (OWNER_CHAT_ID env or sole active user).
- query_ledger_summary / query_ledger_hands: grounded answers over the action-line
  ledger, always with n. All stats exclude excluded + discarded decisions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_leaderboard import analyze_table_url

TPE = ZoneInfo("Asia/Taipei")


async def resolve_owner_chat_id(pool) -> int | None:
    env = os.getenv("OWNER_CHAT_ID")
    if env:
        return int(env)
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT user_id FROM users WHERE is_active")
    return rows[0]["user_id"] if len(rows) == 1 else None


def _summary_sql(category: str | None, hero_cat: str | None, days: int | None):
    """Pure WHERE-builder for the summary aggregate. Returns (sql, args).

    source='online' only (§5.2): live hands are selectively recorded, so their
    averages are biased — they must never blend into the summary stats."""
    where = ["NOT excluded", "NOT discarded", "spot_leaf IS NOT NULL",
             "source='online'", "confidence >= 0.8"]
    args: list = []
    if category:
        args.append(category); where.append(f"spot_category = ${len(args)}")
    if hero_cat:
        args.append(hero_cat); where.append(f"hero_cat = ${len(args)}")
    if days:
        args.append(days); where.append(f"played_at >= now() - make_interval(days => ${len(args)})")
    sql = (f"SELECT count(*) n, sum(ev_loss_bb) total_bb, avg(ev_loss_bb)*100 per100 "
           f"FROM ledger_decisions WHERE {' AND '.join(where)}")
    return sql, args


def _top_spots_sql(category: str | None, hero_cat: str | None, days: int | None, limit: int):
    where = ["NOT excluded", "NOT discarded", "spot_parent IS NOT NULL",
             "source='online'", "confidence >= 0.8"]
    args: list = []
    if category:
        args.append(category); where.append(f"spot_category = ${len(args)}")
    if hero_cat:
        args.append(hero_cat); where.append(f"hero_cat = ${len(args)}")
    if days:
        args.append(days); where.append(f"played_at >= now() - make_interval(days => ${len(args)})")
    args.append(limit)
    sql = (f"SELECT spot_parent spot_leaf, count(*) n, sum(ev_loss_bb) total_bb, avg(ev_loss_bb)*100 per100 "
           f"FROM ledger_decisions WHERE {' AND '.join(where)} "
           f"GROUP BY spot_parent HAVING count(*) >= 25 ORDER BY sum(ev_loss_bb) DESC LIMIT ${len(args)}")
    return sql, args


def _excluded_count_sql(category: str | None, days: int | None):
    """Excluded/discarded caveat count with the SAME scope (source/category/
    window) as the summary stats it is shown beside — a wider-scope count would
    misstate the honesty caveat (§5.2)."""
    where = ["(excluded OR discarded)", "source='online'"]
    args: list = []
    if category:
        args.append(category); where.append(f"spot_category = ${len(args)}")
    if days:
        args.append(days); where.append(f"played_at >= now() - make_interval(days => ${len(args)})")
    return f"SELECT count(*) FROM ledger_decisions WHERE {' AND '.join(where)}", args


async def query_ledger_summary(pool, category=None, hero_cat=None, days=None) -> dict:
    sql, args = _summary_sql(category, hero_cat, days)
    tsql, targs = _top_spots_sql(category, hero_cat, days, 10)
    esql, eargs = _excluded_count_sql(category, days)
    async with pool.acquire() as conn:
        agg = await conn.fetchrow(sql, *args)
        tops = await conn.fetch(tsql, *targs)
        exc = await conn.fetchval(esql, *eargs)
    return {
        "n": agg["n"] or 0, "total_bb": float(agg["total_bb"] or 0),
        "per100": float(agg["per100"] or 0), "excluded_n": exc or 0,
        "window_days": days,
        "top_spots": [{"spot": r["spot_leaf"], "n": r["n"],
                       "total_bb": float(r["total_bb"] or 0),
                       "per100": float(r["per100"] or 0)} for r in tops],
    }


def progress_sql(category: str | None, spot_leaf: str | None):
    """Weekly EV-loss series builder (bb/100 with n), newest weeks first.

    EV-weighted only — the frequency-era deviation-rate trend is retired
    (§7.3). Returns (sql, args) with the LIMIT (weeks) as the LAST positional
    parameter, to be appended by the caller."""
    where = ["NOT excluded", "NOT discarded", "spot_leaf IS NOT NULL",
             "source='online'", "confidence >= 0.8"]
    args: list = []
    if category:
        args.append(category); where.append(f"spot_category = ${len(args)}")
    if spot_leaf:
        args.append(spot_leaf); where.append(f"spot_leaf = ${len(args)}")
    sql = (f"SELECT to_char((played_at AT TIME ZONE 'Asia/Taipei'), 'IYYY-\"W\"IW') week, "
           f"count(*) n, avg(ev_loss_bb)*100 per100 "
           f"FROM ledger_decisions WHERE {' AND '.join(where)} "
           f"GROUP BY 1 ORDER BY 1 DESC LIMIT ${len(args) + 1}")
    return sql, args


async def query_progress_series(pool, category=None, spot_leaf=None, weeks=8) -> list[dict]:
    """Last `weeks` ISO weeks of EV loss/100 (ascending for display)."""
    sql, args = progress_sql(category, spot_leaf)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args, weeks)
    return [{"week": r["week"], "n": r["n"], "per100": float(r["per100"] or 0)}
            for r in reversed(rows)]


async def fetch_latest_scorecard(pool) -> dict | None:
    """Latest weekly scorecard row as {week, data} (data_json normalized)."""
    import json
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT week, data_json FROM scorecards ORDER BY created_at DESC LIMIT 1")
    if not row:
        return None
    data = row["data_json"]
    if isinstance(data, str):
        data = json.loads(data)
    return {"week": row["week"], "data": data or {}}


def _hands_sql(category: str | None, spot: str | None, min_ev_loss: float,
               days: int | None, limit: int):
    """Pure WHERE-builder for the worst-hands listing. source='online' only
    (§5.2): live hands are selectively recorded — they surface via the drill
    queue / 線下 sections, never blended into this list (their Analyze review
    links would be meaningless anyway)."""
    where = ["NOT d.excluded", "NOT d.discarded", "d.source='online'",
             "d.confidence >= 0.8", "d.ev_loss_bb >= $1"]
    args: list = [float(min_ev_loss)]
    if category:
        args.append(category); where.append(f"d.spot_category = ${len(args)}")
    if spot:
        args.append(spot); where.append(f"d.spot_leaf = ${len(args)}")
    if days:
        args.append(days); where.append(f"d.played_at >= now() - make_interval(days => ${len(args)})")
    args.append(min(limit, 10))
    sql = (f"SELECT d.gtow_hand_id, h.played_at, h.hero_hand, h.position, h.boards, "
           f"d.spot_leaf, d.ev_loss_bb, d.correctness "
           f"FROM ledger_decisions d JOIN ledger_hands h ON h.gtow_hand_id=d.gtow_hand_id "
           f"WHERE {' AND '.join(where)} ORDER BY d.ev_loss_bb DESC LIMIT ${len(args)}")
    return sql, args


async def query_ledger_hands(pool, category=None, spot=None, min_ev_loss=0.5,
                             days=90, limit=5) -> list[dict]:
    sql, args = _hands_sql(category, spot, min_ev_loss, days, limit)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    out = []
    for r in rows:
        day = r["played_at"].astimezone(TPE).strftime("%Y-%m-%d")
        out.append({"hand_id": r["gtow_hand_id"][:8], "played_at": day,
                    "hero_hand": r["hero_hand"], "position": r["position"],
                    "boards": r["boards"], "spot": r["spot_leaf"],
                    "ev_loss_bb": float(r["ev_loss_bb"] or 0), "correctness": r["correctness"],
                    "review_url": analyze_table_url(day, day)})
    return out
