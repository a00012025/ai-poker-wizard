#!/usr/bin/env python3
"""Weekly leak report generation + session narrative + tilt detection + hand-of-the-week.

Called by PTB JobQueue every Sunday. Generates a report for each active user
and sends it via Telegram.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import asyncpg

logger = logging.getLogger("poker_bot")

# Spot category display names (Traditional Chinese)
SPOT_NAMES = {
    "open_raise": "開局加注",
    "facing_open": "面對加注",
    "facing_3bet": "面對 3-bet",
    "squeeze": "擠壓加注",
    "facing_4bet": "面對 4-bet",
    "limp_pot": "跛入底池",
    "cbet_ip": "IP C-bet",
    "cbet_oop": "OOP C-bet",
    "facing_cbet_ip": "IP 面對 C-bet",
    "facing_cbet_oop": "OOP 面對 C-bet",
    "probe": "探測下注",
    "facing_probe": "面對探測下注",
    "donk": "Donk bet",
    "check_raise": "Check-raise",
}


async def generate_weekly_report(
    pool: asyncpg.Pool,
    chat_id: int,
    period_end: datetime | None = None,
) -> str | None:
    """Generate weekly report text for a user. Returns formatted message or None if no data."""
    from leak_service import query_leaks, query_stats, detect_tilt, get_hand_of_the_week

    if period_end is None:
        period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=7)

    # Get this week's stats
    stats = await query_stats(pool, chat_id, days=7)
    if stats["total_decisions"] == 0:
        return None  # No data this week

    # Get leaks
    leaks = await query_leaks(pool, chat_id, min_samples=3, limit=5)

    # Get tilt data
    tilt_data = await detect_tilt(pool, chat_id)

    # Get hand of the week
    hotw = await get_hand_of_the_week(pool, chat_id, period_start, period_end)

    # Get previous week stats for comparison
    prev_stats = await _get_prev_week_stats(pool, chat_id, period_start)

    # Format dates
    start_str = period_start.strftime("%m/%d")
    end_str = period_end.strftime("%m/%d")

    lines = [f"📊 每週偏離報告（{start_str}-{end_str}）\n"]
    lines.append(f"分析手牌數: {stats['total_hands']}")
    lines.append(f"決策點: {stats['total_decisions']}")
    overall_rate = stats["deviation_rate"] * 100
    lines.append(f"整體偏離率: {overall_rate:.0f}%")

    if prev_stats and prev_stats["total_decisions"] > 0:
        prev_rate = prev_stats["deviation_rate"] * 100
        delta = overall_rate - prev_rate
        if abs(delta) > 2:
            arrow = "📈" if delta > 0 else "📉"
            lines.append(f"{arrow} 較上週{'上升' if delta > 0 else '下降'} {abs(delta):.0f}%")

    # Top leaks
    top_leaks = [l for l in leaks if l["deviation_rate"] > 0.15]  # Only show significant leaks
    if top_leaks:
        lines.append("\n🔴 主要弱點:")
        for i, leak in enumerate(top_leaks[:3], 1):
            name = SPOT_NAMES.get(leak["spot_category"], leak["spot_category"])
            rate = leak["deviation_rate"] * 100
            lines.append(f"{i}. {name} (n={leak['sample_count']}) — 偏離率 {rate:.0f}%")

    # Improving spots (from leak_reports if available)
    improving = await _get_improving_spots(pool, chat_id)
    if improving:
        lines.append("\n📈 進步中:")
        for spot in improving[:2]:
            name = SPOT_NAMES.get(spot["spot_category"], spot["spot_category"])
            lines.append(f"✅ {name} — 偏離率下降 {abs(spot['trend_delta'])*100:.0f}%")

    # Tracking (low sample spots)
    tracking = [l for l in leaks if l["sample_count"] < 10 and l["deviation_rate"] > 0.2]
    if tracking:
        lines.append("\n👀 追蹤中（樣本不足）:")
        for t in tracking[:3]:
            name = SPOT_NAMES.get(t["spot_category"], t["spot_category"])
            needed = 10 - t["sample_count"]
            lines.append(f"- {name} (n={t['sample_count']}) — 還需 {needed} 手")

    # Tilt detection
    if tilt_data and tilt_data["is_tilting"]:
        tilt_rate = tilt_data["deviation_rate"] * 100
        lines.append(
            f"\n⚠️ 上頭偵測: 最近 {tilt_data['window_size']} 個決策中 "
            f"{tilt_data['deviations_in_window']} 個偏離 ({tilt_rate:.0f}%)"
        )

    # Hand of the week
    if hotw:
        lines.append(f"\n🃏 本週最值得學習的手牌:")
        name = SPOT_NAMES.get(hotw["spot_category"], hotw["spot_category"])
        lines.append(
            f"  {name} — Hero {hotw['hero_action']} vs GTO {hotw['gto_action']}"
        )
        if hotw.get("hero_freq") is not None:
            lines.append(f"  Hero 行動頻率: {hotw['hero_freq']:.0f}%")
        if hotw.get("hand_history_id"):
            lines.append(f"  Hand ID: H{hotw['hand_history_id']}")

    # Training suggestion
    if top_leaks:
        name = SPOT_NAMES.get(top_leaks[0]["spot_category"], top_leaks[0]["spot_category"])
        lines.append(f"\n🎯 本週練習重點:")
        lines.append(f"在 GTO Wizard 練習 {name} 場景")

    return "\n".join(lines)


async def _get_prev_week_stats(
    pool: asyncpg.Pool,
    chat_id: int,
    current_week_start: datetime,
) -> dict | None:
    """Get stats from the previous week for comparison."""
    prev_end = current_week_start
    prev_start = prev_end - timedelta(days=7)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_deviation THEN 1 ELSE 0 END) AS deviations,
                COUNT(DISTINCT hand_history_id) AS hands
            FROM deviations
            WHERE chat_id = $1
              AND created_at >= $2
              AND created_at < $3
            """,
            chat_id, prev_start, prev_end,
        )

    total = row["total"] or 0
    if total == 0:
        return None

    return {
        "total_decisions": total,
        "total_deviations": row["deviations"] or 0,
        "total_hands": row["hands"] or 0,
        "deviation_rate": (row["deviations"] or 0) / total,
    }


async def _get_improving_spots(
    pool: asyncpg.Pool,
    chat_id: int,
) -> list[dict]:
    """Find spots that are improving (from leak_reports)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT spot_category, trend, trend_delta
            FROM leak_reports
            WHERE chat_id = $1
              AND trend = 'improving'
            ORDER BY created_at DESC
            LIMIT 5
            """,
            chat_id,
        )
    return [dict(r) for r in rows]


async def send_weekly_reports(
    pool: asyncpg.Pool,
    bot: Any,  # telegram.Bot instance
) -> int:
    """Generate and send weekly reports to all active users. Returns count of reports sent."""
    # Get all active users with deviations data
    async with pool.acquire() as conn:
        users = await conn.fetch(
            """
            SELECT DISTINCT d.chat_id
            FROM deviations d
            JOIN users u ON u.user_id = d.chat_id
            WHERE u.is_active = TRUE
              AND d.created_at >= NOW() - INTERVAL '7 days'
            """
        )

    sent = 0
    for user_row in users:
        chat_id = user_row["chat_id"]
        try:
            report = await generate_weekly_report(pool, chat_id)
            if report:
                await bot.send_message(chat_id=chat_id, text=report)
                sent += 1
                logger.info(f"Sent weekly report to chat_id={chat_id}")
        except Exception as e:
            logger.warning(f"Failed to send weekly report to chat_id={chat_id}: {e}")
            continue

    logger.info(f"Weekly report: sent {sent}/{len(users)} reports")
    return sent
