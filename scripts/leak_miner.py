"""Leak miner — cluster deviations by rich group key, rank by SUM(ev_loss).

Feeds the weekly report generator and the query_my_leaks LLM tool.
Single source of truth: both push (weekly_report.py) and pull
(query_my_leaks tool) read through this module.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import logging

logger = logging.getLogger("poker_bot")


@dataclass
class ClusterKey:
    """Composite grouping key. All fields carry clustering identity."""
    pot_type:       str | None
    street:         str
    gtow_hero_role: str | None
    villain_pos:    str | None
    hero_pos:       str
    spot_category:  str
    board_texture:  str | None  # None for preflop

    def to_dict(self) -> dict[str, Any]:
        return {
            "pot_type":       self.pot_type,
            "street":         self.street,
            "gtow_hero_role": self.gtow_hero_role,
            "villain_pos":    self.villain_pos,
            "hero_pos":       self.hero_pos,
            "spot_category":  self.spot_category,
            "board_texture":  self.board_texture,
        }


@dataclass
class Cluster:
    key:                 ClusterKey
    sample_count:        int
    total_ev_loss_bb:    float
    avg_ev_loss_bb:      float
    aggression_label:    str   # "too_passive" | "too_aggressive" | "mixed" | "aligned"
    passive_ratio:       float  # 0..1
    aggressive_ratio:    float  # 0..1
    top_hand_ids:        list[int]   # top 3 by individual ev_loss DESC
    top_deviation_ids:   list[int]   # parallel to top_hand_ids (for custom-spot URL builder)
    effective_bb_median: float
    gtow_type:           str | None  # for URL builder (from the cluster's dominant row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key":                 self.key.to_dict(),
            "sample_count":        self.sample_count,
            "total_ev_loss_bb":    round(self.total_ev_loss_bb, 2),
            "avg_ev_loss_bb":      round(self.avg_ev_loss_bb, 3),
            "aggression_label":    self.aggression_label,
            "passive_ratio":       round(self.passive_ratio, 2),
            "aggressive_ratio":    round(self.aggressive_ratio, 2),
            "top_hand_ids":        list(self.top_hand_ids),
            "top_deviation_ids":   list(self.top_deviation_ids),
            "effective_bb_median": round(self.effective_bb_median, 1),
            "gtow_type":           self.gtow_type,
        }


async def mine_clusters(
    pool: asyncpg.Pool,
    chat_id: int,
    start: datetime,
    end: datetime,
    min_sample: int = 5,
    top_k: int = 5,
) -> list[Cluster]:
    """Group deviations into clusters and return the top-k by total EV loss.

    Grouping key: (pot_type, street, gtow_hero_role, villain_pos, hero_pos,
                   spot_category, board_texture)
    Filter:       COUNT(*) >= min_sample, SUM(ev_loss_estimate) > 0
    Rank:         SUM(ev_loss_estimate) DESC

    Excludes rows where ev_loss_estimate IS NULL (pre-backfill data).
    """
    sql = """
    WITH cluster_rows AS (
      SELECT
        id                         AS deviation_id,
        meta->>'pot_type'          AS pot_type,
        street,
        meta->>'gtow_hero_role'    AS hero_role,
        meta->>'villain_pos'       AS villain_pos,
        position                   AS hero_pos,
        spot_category,
        board_texture,
        ev_loss_estimate,
        effective_bb,
        hand_history_id,
        meta->>'aggression_direction' AS agg_dir,
        meta->>'gtow_type'         AS gtow_type
      FROM deviations
      WHERE chat_id = $1
        AND created_at >= $2
        AND created_at <  $3
        AND ev_loss_estimate IS NOT NULL
        AND ev_loss_estimate > 0
    ),
    grouped AS (
      SELECT
        pot_type, street, hero_role, villain_pos, hero_pos, spot_category,
        board_texture,
        COUNT(*)                                                 AS n,
        SUM(ev_loss_estimate)                                    AS total_loss,
        AVG(ev_loss_estimate)                                    AS avg_loss,
        COUNT(*) FILTER (WHERE agg_dir = 'too_passive')          AS n_passive,
        COUNT(*) FILTER (WHERE agg_dir = 'too_aggressive')       AS n_aggressive,
        COUNT(*) FILTER (WHERE agg_dir = 'aligned')              AS n_aligned,
        COUNT(*) FILTER (WHERE agg_dir = 'mixed')                AS n_mixed,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY effective_bb) AS eff_bb_median,
        (array_agg(hand_history_id ORDER BY ev_loss_estimate DESC))[1:3]
                                                                 AS top_hands,
        (array_agg(deviation_id    ORDER BY ev_loss_estimate DESC))[1:3]
                                                                 AS top_dev_ids,
        (array_agg(gtow_type ORDER BY ev_loss_estimate DESC))[1] AS dom_gtow_type
      FROM cluster_rows
      GROUP BY 1,2,3,4,5,6,7
      HAVING COUNT(*) >= $4
    )
    SELECT *
    FROM grouped
    ORDER BY total_loss DESC
    LIMIT $5
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, chat_id, start, end, min_sample, top_k)
    except Exception as e:
        logger.warning(f"[chat={chat_id}] mine_clusters query failed: {e}")
        return []

    out: list[Cluster] = []
    for r in rows:
        label = _label_aggression(
            n_passive=r["n_passive"],
            n_aggressive=r["n_aggressive"],
            n_aligned=r["n_aligned"],
            n_mixed=r["n_mixed"],
        )
        n = r["n"]
        passive_ratio    = (r["n_passive"]    / n) if n else 0.0
        aggressive_ratio = (r["n_aggressive"] / n) if n else 0.0
        key = ClusterKey(
            pot_type=r["pot_type"],
            street=r["street"],
            gtow_hero_role=r["hero_role"],
            villain_pos=r["villain_pos"],
            hero_pos=r["hero_pos"],
            spot_category=r["spot_category"],
            board_texture=r["board_texture"],
        )
        top_hands = [int(h) for h in (r["top_hands"] or []) if h is not None]
        top_dev_ids = [int(d) for d in (r["top_dev_ids"] or []) if d is not None]
        out.append(Cluster(
            key=key,
            sample_count=int(n),
            total_ev_loss_bb=float(r["total_loss"] or 0.0),
            avg_ev_loss_bb=float(r["avg_loss"] or 0.0),
            aggression_label=label,
            passive_ratio=passive_ratio,
            aggressive_ratio=aggressive_ratio,
            top_hand_ids=top_hands,
            top_deviation_ids=top_dev_ids,
            effective_bb_median=float(r["eff_bb_median"] or 0.0),
            gtow_type=r["dom_gtow_type"],
        ))
    return out


def _label_aggression(
    n_passive: int,
    n_aggressive: int,
    n_aligned: int,
    n_mixed: int,
    threshold: float = 0.7,
) -> str:
    """Label a cluster's direction only if one side is >= threshold of non-aligned rows.

    Otherwise 'mixed'. If all rows are aligned, returns 'aligned'.
    """
    total = n_passive + n_aggressive + n_aligned + n_mixed
    if total == 0:
        return "mixed"
    non_aligned = n_passive + n_aggressive + n_mixed
    if non_aligned == 0:
        return "aligned"
    if n_passive / non_aligned >= threshold:
        return "too_passive"
    if n_aggressive / non_aligned >= threshold:
        return "too_aggressive"
    return "mixed"
