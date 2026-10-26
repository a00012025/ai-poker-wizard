"""Select one reviewed hero decision for a precise GTOW Drill."""

from __future__ import annotations

import re
from typing import Callable


_STREET_LABELS = {
    "preflop": "翻前", "flop": "翻牌", "turn": "轉牌", "river": "河牌",
}


def review_drill_spots(hand: dict, hand_id: str) -> list[dict]:
    """One stable callback choice per hero decision in a saved hand."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", hand_id or ""):
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
