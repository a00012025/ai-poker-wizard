#!/usr/bin/env python3
"""Leak service — all DB query methods for leak detection and coaching memory.

Shared by LLM tools (real-time queries) and weekly report job (aggregation).
All methods accept an asyncpg Pool and use parameterized queries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import asyncpg

logger = logging.getLogger("poker_bot")


# ── Deviation Insertion ──

async def insert_deviation(
    pool: asyncpg.Pool,
    chat_id: int,
    hand_history_id: int | None,
    street: str,
    action_index: int,
    spot_category: str,
    position: str,
    hero_action: str,
    gto_action: str,
    hero_freq: float | None,
    gto_freq: float | None,
    ev_loss_estimate: float | None,
    board_texture: str | None,
    effective_bb: float | None,
    is_deviation: bool,
    meta: dict | None = None,
    played_at: datetime | None = None,
) -> None:
    """Insert a single deviation row. ON CONFLICT DO NOTHING (idempotent)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO deviations (
                    chat_id, hand_history_id, street, action_index,
                    spot_category, position, hero_action, gto_action,
                    hero_freq, gto_freq, ev_loss_estimate,
                    board_texture, effective_bb, is_deviation, meta, played_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (hand_history_id, street, action_index) DO NOTHING
                """,
                chat_id, hand_history_id, street, action_index,
                spot_category, position, hero_action, gto_action,
                hero_freq, gto_freq, ev_loss_estimate,
                board_texture, effective_bb, is_deviation,
                json.dumps(meta) if meta else None, played_at,
            )
    except Exception as e:
        logger.warning(f"Failed to insert deviation for chat_id={chat_id}: {e}")


async def insert_deviations_batch(
    pool: asyncpg.Pool,
    rows: list[dict],
) -> int:
    """Insert multiple deviation rows. Returns count of inserted rows."""
    if not rows:
        return 0
    inserted = 0
    async with pool.acquire() as conn:
        for row in rows:
            try:
                result = await conn.execute(
                    """
                    INSERT INTO deviations (
                        chat_id, hand_history_id, street, action_index,
                        spot_category, position, hero_action, gto_action,
                        hero_freq, gto_freq, ev_loss_estimate,
                        board_texture, effective_bb, is_deviation, meta, played_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (hand_history_id, street, action_index) DO NOTHING
                    """,
                    row["chat_id"], row.get("hand_history_id"),
                    row["street"], row.get("action_index", 0),
                    row["spot_category"], row["position"],
                    row["hero_action"], row["gto_action"],
                    row.get("hero_freq"), row.get("gto_freq"),
                    row.get("ev_loss_estimate"), row.get("board_texture"),
                    row.get("effective_bb"), row["is_deviation"],
                    json.dumps(row["meta"]) if row.get("meta") else None,
                    row.get("played_at"),
                )
                if result and "INSERT" in result:
                    inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert deviation: {e}")
    return inserted


# ── Leak Queries (used by LLM tools) ──

async def query_leaks(
    pool: asyncpg.Pool,
    chat_id: int,
    spot_category: str | None = None,
    street: str | None = None,
    position: str | None = None,
    min_samples: int = 5,
    limit: int = 10,
) -> list[dict]:
    """Query aggregated leak data for a user.

    Returns top leaks ranked by deviation_rate * sample_count.
    Each row: {spot_category, sample_count, deviation_count, deviation_rate,
               avg_hero_freq, avg_gto_freq, top_gto_action}
    """
    conditions = ["chat_id = $1"]
    params: list[Any] = [chat_id]
    idx = 2

    if spot_category:
        conditions.append(f"spot_category = ${idx}")
        params.append(spot_category)
        idx += 1
    if street:
        conditions.append(f"street = ${idx}")
        params.append(street)
        idx += 1
    if position:
        conditions.append(f"position = ${idx}")
        params.append(position)
        idx += 1

    where = " AND ".join(conditions)

    query = f"""
        SELECT
            spot_category,
            COUNT(*) AS sample_count,
            SUM(CASE WHEN is_deviation THEN 1 ELSE 0 END) AS deviation_count,
            AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) AS deviation_rate,
            AVG(hero_freq) AS avg_hero_freq,
            AVG(gto_freq) AS avg_gto_freq,
            MODE() WITHIN GROUP (ORDER BY gto_action) AS top_gto_action
        FROM deviations
        WHERE {where}
        GROUP BY spot_category
        HAVING COUNT(*) >= ${idx}
        ORDER BY AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) * COUNT(*) DESC
        LIMIT ${idx + 1}
    """
    params.extend([min_samples, limit])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [dict(r) for r in rows]


async def query_stats(
    pool: asyncpg.Pool,
    chat_id: int,
    days: int | None = None,
) -> dict:
    """Get overall stats for a user.

    Returns: {total_hands, total_deviations, deviation_rate,
              by_street: {preflop: {count, deviation_rate}, ...},
              worst_spots: [...], most_improved: [...]}
    """
    time_filter = ""
    params: list[Any] = [chat_id]
    if days:
        time_filter = " AND created_at >= $2"
        params.append(datetime.utcnow() - timedelta(days=days))

    async with pool.acquire() as conn:
        # Overall counts
        row = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_deviation THEN 1 ELSE 0 END) AS deviations,
                COUNT(DISTINCT hand_history_id) AS hands
            FROM deviations
            WHERE chat_id = $1{time_filter}
            """,
            *params,
        )
        total = row["total"] or 0
        deviations = row["deviations"] or 0
        hands = row["hands"] or 0

        # By street
        street_rows = await conn.fetch(
            f"""
            SELECT
                street,
                COUNT(*) AS count,
                AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) AS deviation_rate
            FROM deviations
            WHERE chat_id = $1{time_filter}
            GROUP BY street
            ORDER BY street
            """,
            *params,
        )

        # Worst spots (top 5 by deviation_rate * count)
        worst_rows = await conn.fetch(
            f"""
            SELECT
                spot_category,
                COUNT(*) AS sample_count,
                AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) AS deviation_rate
            FROM deviations
            WHERE chat_id = $1{time_filter}
            GROUP BY spot_category
            HAVING COUNT(*) >= 5
            ORDER BY AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) * COUNT(*) DESC
            LIMIT 5
            """,
            *params,
        )

    return {
        "total_decisions": total,
        "total_deviations": deviations,
        "total_hands": hands,
        "deviation_rate": deviations / total if total > 0 else 0,
        "by_street": {
            r["street"]: {
                "count": r["count"],
                "deviation_rate": float(r["deviation_rate"]),
            }
            for r in street_rows
        },
        "worst_spots": [
            {
                "spot_category": r["spot_category"],
                "sample_count": r["sample_count"],
                "deviation_rate": float(r["deviation_rate"]),
            }
            for r in worst_rows
        ],
    }


async def query_progress(
    pool: asyncpg.Pool,
    chat_id: int,
    spot_category: str,
    weeks: int = 4,
) -> list[dict]:
    """Get week-over-week trend for a specific spot category.

    Returns list of {week, sample_count, deviation_rate} for the last N weeks.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                DATE_TRUNC('week', created_at)::date AS week_start,
                COUNT(*) AS sample_count,
                AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) AS deviation_rate
            FROM deviations
            WHERE chat_id = $1
              AND spot_category = $2
              AND created_at >= NOW() - ($3 || ' weeks')::interval
            GROUP BY DATE_TRUNC('week', created_at)
            ORDER BY week_start
            """,
            chat_id, spot_category, str(weeks),
        )

    return [
        {
            "week": r["week_start"].isoformat(),
            "sample_count": r["sample_count"],
            "deviation_rate": float(r["deviation_rate"]),
        }
        for r in rows
    ]


# ── Leak Report Generation (used by weekly job) ──

async def generate_leak_report(
    pool: asyncpg.Pool,
    chat_id: int,
    period: str,
    min_samples: int = 5,
) -> list[dict]:
    """Generate leak report for a period and store in leak_reports table.

    period: e.g. "2026-W13"
    Returns list of leak report rows.
    """
    # Get deviations for this period
    # Parse week period: "2026-W13" → date range
    year, week_str = period.split("-W")
    week_num = int(week_str)
    # Monday of that week
    week_start = datetime.fromisocalendar(int(year), week_num, 1)
    week_end = week_start + timedelta(days=7)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                spot_category,
                COUNT(*) AS sample_count,
                AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) AS deviation_rate,
                AVG(CASE WHEN is_deviation THEN ev_loss_estimate ELSE NULL END) AS avg_ev_loss,
                SUM(CASE WHEN is_deviation THEN COALESCE(ev_loss_estimate, 0) ELSE 0 END) AS total_ev_loss
            FROM deviations
            WHERE chat_id = $1
              AND created_at >= $2
              AND created_at < $3
            GROUP BY spot_category
            HAVING COUNT(*) >= $4
            ORDER BY AVG(CASE WHEN is_deviation THEN 1.0 ELSE 0.0 END) * COUNT(*) DESC
            """,
            chat_id, week_start, week_end, min_samples,
        )

        if not rows:
            return []

        # Get previous period for trend
        prev_period = f"{year}-W{week_num - 1:02d}" if week_num > 1 else f"{int(year) - 1}-W52"

        reports = []
        for r in rows:
            spot = r["spot_category"]

            # Get trend from previous leak_report
            prev_report = await conn.fetchrow(
                """
                SELECT deviation_rate FROM leak_reports
                WHERE chat_id = $1 AND report_period = $2 AND spot_category = $3
                """,
                chat_id, prev_period, spot,
            )

            trend = "stable"
            trend_delta = 0.0
            if prev_report:
                prev_rate = prev_report["deviation_rate"]
                current_rate = float(r["deviation_rate"])
                trend_delta = current_rate - prev_rate
                if trend_delta < -0.05:
                    trend = "improving"
                elif trend_delta > 0.05:
                    trend = "worsening"

            report = {
                "chat_id": chat_id,
                "report_period": period,
                "spot_category": spot,
                "sample_count": r["sample_count"],
                "deviation_rate": float(r["deviation_rate"]),
                "avg_ev_loss": float(r["avg_ev_loss"]) if r["avg_ev_loss"] else None,
                "total_ev_loss": float(r["total_ev_loss"]) if r["total_ev_loss"] else None,
                "trend": trend,
                "trend_delta": trend_delta,
            }
            reports.append(report)

    # Store reports
    async with pool.acquire() as conn:
        for report in reports:
            await conn.execute(
                """
                INSERT INTO leak_reports (
                    chat_id, report_period, spot_category, sample_count,
                    deviation_rate, avg_ev_loss, total_ev_loss, trend, trend_delta
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (chat_id, report_period, spot_category)
                DO UPDATE SET
                    sample_count = $4, deviation_rate = $5, avg_ev_loss = $6,
                    total_ev_loss = $7, trend = $8, trend_delta = $9
                """,
                report["chat_id"], report["report_period"],
                report["spot_category"], report["sample_count"],
                report["deviation_rate"], report["avg_ev_loss"],
                report["total_ev_loss"], report["trend"], report["trend_delta"],
            )

    return reports


# ── Tilt Detection ──

async def detect_tilt(
    pool: asyncpg.Pool,
    chat_id: int,
    window_size: int = 10,
    min_hands: int = 5,
    threshold: float = 0.5,
) -> dict | None:
    """Detect tilt by checking overall deviation rate in a moving window.

    Returns {deviation_rate, window_size, is_tilting} or None if insufficient data.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT is_deviation
            FROM deviations
            WHERE chat_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            chat_id, window_size,
        )

    if len(rows) < min_hands:
        return None

    deviations = sum(1 for r in rows if r["is_deviation"])
    rate = deviations / len(rows)

    return {
        "deviation_rate": rate,
        "window_size": len(rows),
        "deviations_in_window": deviations,
        "is_tilting": rate >= threshold,
    }


# ── Hand of the Week ──

async def get_hand_of_the_week(
    pool: asyncpg.Pool,
    chat_id: int,
    period_start: datetime,
    period_end: datetime,
) -> dict | None:
    """Find the most instructive hand from the period.

    Picks the hand with the highest EV loss (or most deviations if no EV data).
    Returns {hand_history_id, street, hero_action, gto_action, ev_loss_estimate,
             spot_category} or None.
    """
    async with pool.acquire() as conn:
        # Try highest EV loss first
        row = await conn.fetchrow(
            """
            SELECT hand_history_id, street, hero_action, gto_action,
                   ev_loss_estimate, spot_category, hero_freq, gto_freq
            FROM deviations
            WHERE chat_id = $1
              AND created_at >= $2
              AND created_at < $3
              AND is_deviation = TRUE
            ORDER BY COALESCE(ev_loss_estimate, 0) DESC, hero_freq ASC
            LIMIT 1
            """,
            chat_id, period_start, period_end,
        )

    if row:
        return dict(row)
    return None


# ── Session Narrative ──

async def get_session_hands(
    pool: asyncpg.Pool,
    chat_id: int,
    hand_history_ids: list[int],
) -> list[dict]:
    """Get deviation data for a list of hand history IDs (for session narrative).

    Returns deviations sorted by created_at (chronological).
    """
    if not hand_history_ids:
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT hand_history_id, street, spot_category, hero_action,
                   gto_action, hero_freq, gto_freq, is_deviation,
                   ev_loss_estimate, board_texture, created_at
            FROM deviations
            WHERE chat_id = $1
              AND hand_history_id = ANY($2)
            ORDER BY created_at
            """,
            chat_id, hand_history_ids,
        )

    return [dict(r) for r in rows]


def split_sessions(
    hands: list[dict],
    gap_minutes: int = 30,
) -> list[list[dict]]:
    """Split hands into sessions based on time gaps.

    hands: list of dicts with 'played_at' or 'created_at' datetime field.
    Returns list of sessions, each a list of hands.
    """
    if not hands:
        return []

    # Sort by timestamp
    def _get_time(h: dict) -> datetime:
        t = h.get("played_at") or h.get("created_at")
        if isinstance(t, str):
            return datetime.fromisoformat(t)
        return t

    sorted_hands = sorted(hands, key=_get_time)
    gap = timedelta(minutes=gap_minutes)

    sessions: list[list[dict]] = [[sorted_hands[0]]]
    for h in sorted_hands[1:]:
        if _get_time(h) - _get_time(sessions[-1][-1]) > gap:
            sessions.append([h])
        else:
            sessions[-1].append(h)

    return sessions
