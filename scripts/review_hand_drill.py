"""Select one reviewed hero decision for a precise GTOW Drill."""

from __future__ import annotations

import re
import uuid
from typing import Callable
from urllib.parse import urlsplit


_STREET_LABELS = {
    "preflop": "翻前", "flop": "翻牌", "turn": "轉牌", "river": "河牌",
}


def parse_shared_hand_id(value: str) -> str | None:
    """Accept one GTOW shared-hand URL or its canonical UUID."""
    value = (value or "").strip()
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        if (parts.scheme != "https" or parts.netloc != "app.gtowizard.com"
                or parts.fragment):
            return None
        match = re.fullmatch(r"/shared-hand/([0-9a-fA-F-]{36})/?", parts.path)
        if not match:
            return None
        value = match.group(1)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    return str(parsed) if str(parsed) == value.lower() else None


def hand_row_from_analyze_detail(hand_id: str, detail: dict) -> dict | None:
    """Recover the hand metadata needed to reconstruct an uningested hand."""
    if not isinstance(detail, dict) or str(detail.get("id")) != hand_id:
        return None
    players = detail.get("players") or []
    heroes = [player for player in players if player.get("is_hero") is True]
    if len(heroes) != 1:
        return None
    hero = heroes[0]
    try:
        stack = float(hero.get("stack"))
        players_dealt = int(detail.get("players_dealt") or len(players))
    except (TypeError, ValueError):
        return None
    if not (hero.get("position") and hero.get("hand") and stack > 0
            and players_dealt >= 2):
        return None
    return {
        "gtow_hand_id": hand_id,
        "position": hero["position"],
        "hero_hand": hero["hand"],
        "total_players": players_dealt,
        "preflop_depth_bb": stack,
    }


def review_drill_spots(hand: dict, hand_id: str) -> list[dict]:
    """One stable callback choice per hero decision in a saved hand."""
    if not (re.fullmatch(r"[A-Za-z0-9_]+", hand_id or "")
            or parse_shared_hand_id(hand_id) == hand_id):
        return []

    from scorecard import spot_desc_zh
    from spot_naming import compact_spot_name
    from spot_taxonomy import walk_spots_from_parsed

    options = []
    for spot in walk_spots_from_parsed(hand):
        if spot.get("discarded"):
            continue
        street = spot["street"]
        index = int(spot["decision_idx"])
        callback_data = f"hds:{hand_id}:{street}:{index}"
        if len(callback_data.encode("utf-8")) > 64:
            continue
        decision = {
            "street": street,
            "decision_idx": index,
            "spot_category": spot["category"],
            "spot_leaf": spot["leaf"],
            "position": spot["hero_pos"],
            "hero_cat": spot["hero_cat"],
            "villain_cat": spot["villain_cat"],
            "ip_oop": spot["ip_oop"],
            "pot_type": spot["pot_type"],
            "eff_stack": spot["tags"]["eff_stack"],
            "gtow_hand_id": hand_id,
        }
        desc = spot_desc_zh({**decision, "hero_pos": spot["hero_pos"]})
        label = f"{_STREET_LABELS.get(street, street)} {index + 1} · {desc}"
        options.append({
            **decision,
            "label": label[:60],
            "drill_name": compact_spot_name(decision),
            "callback_data": callback_data,
        })
    return options


def build_review_drill_url(
    hand: dict,
    option: dict,
    *,
    url_builder: Callable[[dict], str | None] | None = None,
) -> str | None:
    """Use the shared exact/standard Trainer URL policy for this decision."""
    if url_builder is None:
        from live_flow import drill_url_for

        url_builder = drill_url_for
    return url_builder({**option, "_hand": hand})
