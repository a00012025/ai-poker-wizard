#!/usr/bin/env python3
"""Deviation capture for the live coaching flow (screenshot/text analyses).

Writes per-decision deviations extracted after each analysis
(gemini_session._extract_deviations → insert_deviation). The frequency-era
query/report layer that used to live here (query_leaks / query_stats /
query_progress / weekly report helpers) was retired per North Star §7.3/§12 —
all stats surfaces now read the EV-weighted ledger (ledger_service /
scorecard). The `deviations` table remains as the live-analysis capture
snapshot only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone

import asyncpg

logger = logging.getLogger("poker_bot")


# ── Shared zh-TW labels (single source of truth for LLM tools + reports) ──

SPOT_DESCRIPTIONS_ZH: dict[str, str] = {
    # Preflop
    "open_raise":       "開局加注範圍",
    "facing_open":      "面對加注時的應對",
    "hero_3bet":        "主動 3-bet 時機",
    "facing_3bet":      "面對 3-bet 的防禦",
    "facing_4bet":      "面對 4-bet 的應對",
    "squeeze":          "擠壓加注時機",
    "vs_squeeze":       "被 squeeze 後的應對",
    "possible_squeeze": "錯過 squeeze 機會",
    "limp_pot":         "跛入底池策略",
    # Postflop
    "cbet_ip":          "位置內 C-bet",
    "cbet_oop":         "位置外 C-bet",
    "facing_cbet_ip":   "位置內面對 C-bet",
    "facing_cbet_oop":  "位置外面對 C-bet",
    "probe":            "探測性下注",
    "facing_probe":     "面對探測性下注",
    "donk":             "Donk bet",
    "check_raise":      "Check-raise",
}

AGGRESSION_DIRECTION_ZH: dict[str, str] = {
    "too_passive":    "太 passive（應更主動）",
    "too_aggressive": "太 aggressive（應更收斂）",
    "mixed":          "混合方向",
    "aligned":        "頻率大致正確但 EV 有落差",
}



# ── DeviationMeta (typed JSONB access) ──

@dataclass
class DeviationMeta:
    """Typed view of the `deviations.meta` JSONB column.

    All fields are optional; `to_jsonb()` drops None entries so we only
    store what we actually know. `from_jsonb()` tolerates extra/unknown
    keys for forward-compat.
    """
    villain_pos: str | None = None
    preflop_line_key: str | None = None
    pot_type: str | None = None
    # "too_passive" | "too_aggressive" | "aligned" | "mixed" | None
    aggression_direction: str | None = None
    gtow_type: str | None = None
    # "aggressor" | "caller" | "squeezer" | "3bettor" | ...
    gtow_hero_role: str | None = None
    gto_dominant_action: str | None = None  # highest frequency
    gto_best_ev_action: str | None = None   # highest EV

    def to_jsonb(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_jsonb(cls, d: dict | None) -> "DeviationMeta":
        if not d:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ── EV loss + aggression helpers ──

_PASSIVE_CODES = {"F", "X", "C"}
_AGGRESSIVE_PREFIXES = ("R", "AI", "B")  # B reserved; not currently emitted.


def _is_passive(code: str) -> bool:
    return code in _PASSIVE_CODES


def _is_aggressive(code: str) -> bool:
    if not code:
        return False
    return any(code.startswith(p) for p in _AGGRESSIVE_PREFIXES)


def compute_ev_loss(
    action_evs: dict[str, float] | None,
    hero_code: str | None,
) -> float | None:
    """Return max(0, best_ev - hero_ev) in bb, or None if data is missing.

    Floating-point safe: if hero_ev barely exceeds max_ev due to rounding,
    clamps to 0 rather than returning a negative loss.
    """
    if not action_evs or hero_code is None:
        return None
    if hero_code not in action_evs:
        return None
    hero_ev = action_evs[hero_code]
    if hero_ev is None:
        return None
    try:
        max_ev = max(action_evs.values())
    except ValueError:
        return None
    loss = max_ev - hero_ev
    return loss if loss > 0.0 else 0.0


def pick_best_ev_action(action_evs: dict[str, float] | None) -> str | None:
    """Return the action code with the highest EV, or None."""
    if not action_evs:
        return None
    return max(action_evs, key=lambda k: action_evs[k])


def classify_aggression_direction(
    hero_code: str | None,
    gto_best_code: str | None,
) -> str | None:
    """Is hero playing more passively or more aggressively than GTO wants?

    Returns one of: "aligned", "too_passive", "too_aggressive", "mixed".
    Returns None if either code is missing.
    """
    if not hero_code or not gto_best_code:
        return None
    if hero_code == gto_best_code:
        return "aligned"
    hp, ha = _is_passive(hero_code), _is_aggressive(hero_code)
    gp, ga = _is_passive(gto_best_code), _is_aggressive(gto_best_code)
    if hp and ga:
        return "too_passive"
    if ha and gp:
        return "too_aggressive"
    return "mixed"


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


