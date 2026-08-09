#!/usr/bin/env python3
"""Deterministic teaching skeletons for the initial coaching reply.

The solver decides *what is true*.  This module compresses those facts into a
small teaching card: the relevant range structure, the exact combo's role, and
whether blockers or size construction are actually supported by the node.  The
LLM may explain this card, but it must not invent a different range story.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from coach_causal_rules import select_causal_mechanisms
import gto_formatter as gf


_MADE_ZH = {
    "no_made_hand": "未成牌",
    "king_high": "K 高",
    "ace_high": "A 高",
    "low_pair": "小對",
    "third_pair": "第三對",
    "second_pair": "第二對",
    "underpair": "低口袋對",
    "top_pair": "頂對",
    "overpair": "超對",
    "over_pair": "超對",
    "two_pair": "兩對",
    "set": "set",
    "trips": "三條",
    "straight": "順子",
    "flush": "同花",
    "full_house": "葫蘆",
    "quads": "四條",
    "straight_flush": "同花順",
}

_CATEGORY_ALIASES = {
    "fullhouse": "full_house",
    "over_pair": "overpair",
}

_DRAW_ZH = {
    "no_draw": "沒有聽牌",
    "onecard_bdfd": "單張後門同花潛力",
    "twocards_bdfd": "雙張後門同花潛力",
    "gutshot": "卡順聽牌",
    "oesd": "兩頭順子聽牌",
    "flush_draw": "同花聽牌",
    "nut_flush_draw": "堅果同花聽牌",
    "combo_draw": "複合聽牌",
}

_STRONG_CATEGORIES = (
    "straight_flush", "quads", "full_house", "flush", "straight",
    "set", "trips", "two_pair", "overpair", "top_pair",
)
_POLAR_VALUE = {
    "straight_flush", "quads", "full_house", "flush", "straight", "set", "trips",
}
_POLAR_AIR = {"no_made_hand", "king_high", "ace_high"}

_CATEGORY_TERMS = {
    "flush": ("同花", "flush"),
    "straight": ("順子", "straight"),
    "set": ("set", "暗三條"),
    "trips": ("trips", "明三條", "三條"),
    "two_pair": ("兩對", "two pair"),
    "top_pair": ("頂對", "top pair"),
    "overpair": ("超對", "overpair"),
    "full_house": ("葫蘆", "full house"),
    "quads": ("四條", "quads"),
    "flush_draw": ("同花聽牌", "梅花聽牌", "花聽", "flush draw"),
    "backdoor_flush": ("後門同花", "後門花", "backdoor flush"),
    "straight_draw": ("卡順", "兩頭順", "順子聽牌", "gutshot", "oesd"),
}

_VULNERABLE_MADE_HANDS = {
    "low_pair", "third_pair", "second_pair", "underpair", "top_pair", "overpair",
}
_VALUE_MADE_HANDS = {
    "top_pair", "overpair", "two_pair", "set", "trips", "straight", "flush",
    "full_house", "quads", "straight_flush",
}
_UNPAIRED_CATEGORIES = {"no_made_hand", "king_high", "ace_high"}
_EXACT_MADE_TERMS = {
    "low_pair": ("小對", "小對子", "low pair"),
    "third_pair": ("第三對", "third pair"),
    "second_pair": ("第二對", "second pair"),
    "underpair": ("低口袋對", "underpair"),
    "top_pair": ("頂對", "top pair"),
    "overpair": ("超對", "overpair"),
    "two_pair": ("兩對", "two pair"),
    "set": ("set", "暗三條"),
    "trips": ("trips", "明三條"),
    "straight": ("順子", "straight"),
    "flush": ("同花", "flush"),
    "full_house": ("葫蘆", "full house"),
    "quads": ("四條", "quads"),
    "straight_flush": ("同花順", "straight flush"),
}
_PREFLOP_ROLE_LABELS = {
    "opener": "opener",
    "3bettor": "3-bettor",
    "4bettor": "4-bettor",
    "5bettor": "5-bettor",
    "caller": "caller",
    "limper": "limper",
    "checker": "checker",
    "unknown": "角色不明",
}


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: float) -> int:
    return int(round(100 * value))


def _normalize_category(name: str | None) -> str | None:
    if not name:
        return name
    return _CATEGORY_ALIASES.get(name, name)


def _raw_hero_hand(context: dict) -> str | None:
    raw = (context.get("hand") or {}).get("hero_hand")
    if raw and gf.combo_index_for_hand(raw) is not None:
        return raw
    hand = context.get("hero_hand")
    if hand and gf.combo_index_for_hand(hand) is not None:
        return hand
    return None


def _players(solution: dict) -> dict[str, dict]:
    return {
        (row.get("player") or {}).get("position"): row
        for row in (solution.get("players_info") or [])
        if (row.get("player") or {}).get("position")
    }


def _category_shares(player_info: dict) -> dict[str, float]:
    shares: dict[str, float] = {}
    for row in player_info.get("hand_categories") or []:
        name = _normalize_category(row.get("name"))
        if name:
            shares[name] = shares.get(name, 0.0) + _float(row.get("total_frequency"))
    return shares


def _advanced_equity_profile(container: dict) -> dict | None:
    """Collapse GTOW's advanced buckets into stable teaching regions.

    The 90–100 bucket is a *top-equity region*, not proof of literal nuts.  The
    broader strong/middle/weak split is used only to compare two ranges or two
    action buckets at the same solved node.
    """
    rows = container.get("equity_buckets_advanced") or []
    shares = {
        row.get("name"): _float(row.get("total_frequency"), -1.0)
        for row in rows
        if row.get("name") and _float(row.get("total_frequency"), -1.0) >= 0
    }
    if not shares:
        return None
    total = sum(shares.values())
    if total <= 0:
        return None
    normalized = {name: value / total for name, value in shares.items()}
    top = normalized.get("hands_90_100", 0.0)
    strong = sum(
        value for name, value in normalized.items()
        if name in {"hands_70_80", "hands_80_90", "hands_90_100"}
    )
    middle = sum(
        value for name, value in normalized.items()
        if name in {"hands_50_60", "hands_60_70"}
    )
    weak = max(0.0, 1.0 - strong - middle)
    return {
        "top_90_100": top,
        "strong": strong,
        "middle": middle,
        "weak": weak,
        "extremes": strong + weak,
        "buckets": normalized,
    }


def _range_structure(hero: str, villain: str, hero_pi: dict,
                     villain_pi: dict) -> dict | None:
    """Quantify average/top-end range structure without claiming causality."""
    hero_profile = _advanced_equity_profile(hero_pi)
    villain_profile = _advanced_equity_profile(villain_pi)
    if not hero_profile or not villain_profile:
        return None
    top_gap = hero_profile["top_90_100"] - villain_profile["top_90_100"]
    strong_gap = hero_profile["strong"] - villain_profile["strong"]
    top_owner = hero if top_gap >= 0.02 else (villain if top_gap <= -0.02 else None)
    strong_owner = hero if strong_gap >= 0.05 else (villain if strong_gap <= -0.05 else None)
    return {
        "hero": hero_profile,
        "villain": villain_profile,
        "nut_region": {
            "label": "90–100% equity 頂端區域",
            "hero_share": hero_profile["top_90_100"],
            "villain_share": villain_profile["top_90_100"],
            "gap": top_gap,
            "owner": top_owner,
            "scope": "頂端 equity proxy，不等於 literal nuts，也不能單獨決定 size",
        },
        "strong_region": {
            "label": "70–100% equity 強端區域",
            "hero_share": hero_profile["strong"],
            "villain_share": villain_profile["strong"],
            "gap": strong_gap,
            "owner": strong_owner,
        },
    }


def _category_name(solution: dict, player_info: dict, combo_idx: int) -> str | None:
    categories = solution.get("hand_categories_range") or []
    if combo_idx >= len(categories):
        return None
    names = {
        row.get("index"): _normalize_category(row.get("name"))
        for row in (player_info.get("hand_categories") or [])
    }
    return names.get(categories[combo_idx])


def _draw_name(solution: dict, player_info: dict, combo_idx: int) -> str | None:
    categories = solution.get("draw_categories_range") or []
    if combo_idx >= len(categories):
        return None
    names = {
        row.get("index"): row.get("name")
        for row in (player_info.get("draw_categories") or [])
    }
    return names.get(categories[combo_idx])


def _deterministic_hand_truth(hero_hand: str, board: str) -> dict | None:
    """Cross-check exact combo categories with the repo's poker-rules evaluator."""
    try:
        from hand_eval import evaluate

        truth = evaluate(hero_hand, board)
    except (ImportError, TypeError, ValueError, IndexError):
        return None
    return truth if truth.get("made_hand") else None


def _range_band(percentile: float | None) -> str:
    if percentile is None or percentile < 0:
        return "位置不明"
    if percentile < 0.20:
        return "range 底端"
    if percentile < 0.40:
        return "range 偏下段"
    if percentile < 0.65:
        return "range 中段"
    if percentile < 0.85:
        return "range 偏上段"
    return "range 頂端"


def _action_label(action_solution: dict, facing_bet: bool) -> str:
    action = action_solution.get("action") or {}
    code = action.get("code") or ""
    if code == "X":
        return "check"
    if code == "C":
        return "call"
    if code == "F":
        return "fold"
    if code == "RAI" or action.get("allin"):
        return "all-in"
    ratio = _float(action.get("betsize_by_pot"), -1.0)
    verb = "raise" if facing_bet else "bet"
    if ratio >= 0:
        return f"{verb} {_pct(ratio)}% pot"
    return verb


def _exact_actions(solution: dict, combo_idx: int) -> list[dict]:
    facing_bet = any(
        (row.get("action") or {}).get("code") in {"F", "C"}
        for row in (solution.get("action_solutions") or [])
    )
    rows = []
    for action_solution in (solution.get("action_solutions") or []):
        strategies = action_solution.get("strategy") or []
        evs = action_solution.get("evs") or []
        if combo_idx >= len(strategies) or combo_idx >= len(evs):
            continue
        action = action_solution.get("action") or {}
        rows.append({
            "code": action.get("code"),
            "label": _action_label(action_solution, facing_bet),
            "frequency": _float(strategies[combo_idx]),
            "ev_bb": _float(evs[combo_idx]),
            "pot_ratio": _float(action.get("betsize_by_pot"), -1.0),
        })
    return rows


def _preflop_decision(context: dict, spot: dict, solution: dict,
                      hero_hand: str) -> dict | None:
    """Build the exact-combo verdict needed to cover one preflop decision.

    Preflop uses the 169 hand-class arrays, so it cannot share the richer
    postflop causal card.  It still carries the same action/mix/EV contract,
    which is enough for a concise, deterministic review line.
    """
    if spot.get("street") != "preflop":
        return None
    hero = spot.get("solver_hero_pos") or context.get("hero_position")
    players = _players(solution)
    hero_pi = players.get(hero)
    if not hero_pi or (solution.get("game") or {}).get("active_position") != hero:
        return None
    player_range = hero_pi.get("range") or []
    if len(player_range) != 169:
        return None
    from hh_deviation_check import HAND_TO_169

    combo_idx = HAND_TO_169.get(gf.normalize_hand_name(hero_hand))
    if combo_idx is None or _float(player_range[combo_idx]) <= 0:
        return None
    actions = _exact_actions(solution, combo_idx)
    if not actions:
        return None
    actual_code = spot.get("taken_code")
    if not actual_code:
        full_line = (
            context.get("preflop_actions")
            or (context.get("hand") or {}).get("preflop_actions")
            or ""
        )
        prefix = (spot.get("params") or {}).get("preflop_actions") or ""
        full_tokens = [token for token in str(full_line).split("-") if token]
        prefix_tokens = [token for token in str(prefix).split("-") if token]
        if len(prefix_tokens) < len(full_tokens):
            actual_code = full_tokens[len(prefix_tokens)]
    actual = next((row for row in actions if row["code"] == actual_code), None)
    best = (
        actual
        if actual and actual.get("frequency", 0.0) >= 0.01
        else _best_in_mix_action(actions)
    )
    preferred = max(actions, key=lambda row: row["frequency"])
    ev_loss = max(0.0, best["ev_bb"] - actual["ev_bb"]) if actual else 0.0
    pot = _float((solution.get("game") or {}).get("pot"))
    range_plan = _range_plan(solution)
    return {
        "street": "preflop",
        "board": "",
        "hero": hero,
        "villain": None,
        "hero_hand": hero_hand,
        "actual_action": actual,
        "preferred_action": preferred,
        "best_action_by_ev": best,
        "available_actions": actions,
        "ev_loss_bb": ev_loss,
        "ev_loss_pot": ev_loss / pot if pot > 0 else None,
        "range_plan": range_plan,
        "action_contract": _action_contract(actions, range_plan),
        "confidence": "high",
        "scope": "只適用於這個深度與 preflop action line",
    }


def _decision_verdict(decision: dict) -> str:
    """Render one compact, deterministic verdict for the narrator contract."""
    actual = decision.get("actual_action")
    preferred = decision["preferred_action"]
    loss = decision.get("ev_loss_pot") or 0.0
    if not actual:
        return f"solver 最偏好 {preferred['label']}"
    if loss >= 0.003:
        return f"{actual['label']} 有實質 EV 損失，應偏向 {preferred['label']}"
    if actual.get("code") == preferred.get("code"):
        purity = "幾乎純用" if preferred.get("frequency", 0.0) >= 0.97 else "最常用"
        return f"{actual['label']} 正確，solver {purity}這個動作"
    if actual.get("frequency", 0.0) >= 0.01:
        return (
            f"{actual['label']} 是 solver 保留的 mix，"
            f"但主要動作是 {preferred['label']}"
        )
    return (
        f"{actual['label']} 的 EV 影響很小，但不在可採信的 solver mix，"
        f"應偏向 {preferred['label']}"
    )


def _label_all_decisions(decisions: list[dict]) -> None:
    """Assign stable labels that distinguish multiple decisions on one street."""
    street_names = {
        "preflop": "Preflop", "flop": "Flop", "turn": "Turn", "river": "River",
    }
    totals: dict[str, int] = {}
    seen: dict[str, int] = {}
    for decision in decisions:
        street = decision["street"]
        totals[street] = totals.get(street, 0) + 1
    circled = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}
    for index, decision in enumerate(decisions, 1):
        street = decision["street"]
        seen[street] = seen.get(street, 0) + 1
        suffix = ""
        if totals[street] > 1:
            suffix = " " + circled.get(seen[street], str(seen[street]))
        decision["decision_id"] = f"D{index}"
        decision["coverage_label"] = street_names[street] + suffix
        decision["coverage_verdict"] = _decision_verdict(decision)


def _best_in_mix_action(actions: list[dict]) -> dict:
    """Use the same 1% in-mix EV basis as the canonical deviation grader."""
    from hh_deviation_check import _best_in_mix

    frequencies = {row["code"]: row["frequency"] for row in actions if row.get("code")}
    evs = {row["code"]: row["ev_bb"] for row in actions if row.get("code")}
    code, _ = _best_in_mix(frequencies, evs)
    return next(row for row in actions if row.get("code") == code)


def _range_plan(solution: dict) -> dict:
    action_solutions = solution.get("action_solutions") or []
    frequencies = {
        (row.get("action") or {}).get("code"): _float(row.get("total_frequency"))
        for row in action_solutions
        if (row.get("action") or {}).get("code")
    }
    facing_bet = "F" in frequencies or "C" in frequencies
    check = frequencies.get("X", 0.0)
    fold = frequencies.get("F", 0.0)
    aggression = sum(
        freq for code, freq in frequencies.items()
        if code and code.startswith("R")
    )
    if check >= 0.97:
        text = "solver 讓整個 range 幾乎全部 check"
        strength = "very_strong"
    elif facing_bet and fold <= 0.12:
        text = f"solver 讓整個 range 繼續約 {_pct(1 - fold)}%，防守非常寬"
        strength = "strong"
    elif aggression >= 0.75:
        text = f"solver 讓整個 range 約 {_pct(aggression)}% 採取進攻動作"
        strength = "strong"
    elif check >= 0.70:
        text = f"solver 的整體 range 以 check 為主（約 {_pct(check)}%）"
        strength = "medium"
    else:
        text = "solver 的整體 range 採混合策略"
        strength = "weak"
    return {
        "text": text,
        "strength": strength,
        "frequencies": frequencies,
        "facing_bet": facing_bet,
    }


def _action_contract(actions: list[dict], range_plan: dict) -> dict:
    """Keep continue frequency distinct from its call/raise components."""
    frequencies = {row["code"]: row["frequency"] for row in actions if row.get("code")}
    facing_bet = range_plan["facing_bet"]
    continue_frequency = (
        sum(freq for code, freq in frequencies.items() if code != "F")
        if facing_bet else None
    )
    preferred = max(actions, key=lambda row: row["frequency"])
    mixed = [row for row in actions if row["frequency"] >= 0.01]
    if facing_bet and continue_frequency is not None and continue_frequency >= 0.995:
        if preferred["frequency"] >= 0.97:
            summary = f"這個 combo 不棄牌，幾乎純 {preferred['label']}"
        else:
            labels = "、".join(row["label"] for row in mixed if row["code"] != "F")
            summary = (
                f"這個 combo 100% 繼續，在 {labels} 之間分配；"
                "不可把『繼續』改寫成單一 action"
            )
    elif facing_bet:
        summary = f"這個 combo 主要採取 {preferred['label']}，其餘 action 只是混合"
    elif preferred["frequency"] >= 0.97:
        summary = f"這個 combo 幾乎純 {preferred['label']}"
    else:
        labels = "、".join(row["label"] for row in mixed)
        summary = f"這個 combo 在 {labels} 之間混合，以 {preferred['label']} 為主"
    return {
        "mode": "facing_bet" if facing_bet else "unopened_action",
        "frequencies": frequencies,
        "continue_frequency": continue_frequency,
        "preferred_code": preferred["code"],
        "preferred_label": preferred["label"],
        "summary": summary,
    }


def _preflop_roles(solution: dict, spot: dict) -> dict[str, str]:
    """Replay the solver line and assign durable preflop roles to its actors."""
    tokens = [
        token for token in str((spot.get("params") or {}).get("preflop_actions") or "").split("-")
        if token
    ]
    game_players = (solution.get("game") or {}).get("players") or []
    try:
        from gtow_action_resolver import POSITION_ORDERS, _replay_preflop_actors

        positions = POSITION_ORDERS.get(len(game_players))
        if not positions:
            return {}
        actors = _replay_preflop_actors(tokens, positions)
    except (ImportError, ValueError, IndexError):
        return {}

    roles: dict[str, str] = {}
    raise_count = 0
    for actor, token in zip(actors, tokens):
        if token.startswith("R") or token.startswith("AI"):
            raise_count += 1
            role = {
                1: "opener", 2: "3bettor", 3: "4bettor", 4: "5bettor",
            }.get(raise_count, "5bettor")
            roles[actor] = role
        elif token == "C" and actor not in roles:
            roles[actor] = "caller" if raise_count else "limper"
        elif token == "X" and actor not in roles:
            roles[actor] = "checker"
    return roles


def _node_context(solution: dict, spot: dict, hero: str, villain: str) -> dict:
    game = solution.get("game") or {}
    players = {
        row.get("position"): row
        for row in game.get("players") or [] if row.get("position")
    }
    roles = _preflop_roles(solution, spot)
    hero_player = players.get(hero) or {}
    villain_player = players.get(villain) or {}
    pot = _float(game.get("pot"))
    current_stacks = [
        _float(player.get("current_stack"), -1.0)
        for player in (hero_player, villain_player)
    ]
    effective_remaining = (
        min(current_stacks) if pot > 0 and all(stack >= 0 for stack in current_stacks)
        else None
    )
    effective_spr = effective_remaining / pot if effective_remaining is not None else None
    hero_role = roles.get(hero, "unknown")
    villain_role = roles.get(villain, "unknown")
    hero_relative = hero_player.get("relative_postflop_position") or "unknown"
    villain_relative = villain_player.get("relative_postflop_position") or "unknown"
    return {
        "hero_preflop_role": hero_role,
        "villain_preflop_role": villain_role,
        "hero_relative_position": hero_relative,
        "villain_relative_position": villain_relative,
        "pot_odds": _float(game.get("pot_odds"), -1.0),
        "effective_spr": effective_spr,
        "actor_lock": (
            f"Hero={hero}（{_PREFLOP_ROLE_LABELS[hero_role]}，{hero_relative}）；"
            f"Villain={villain}（{_PREFLOP_ROLE_LABELS[villain_role]}，{villain_relative}）"
        ),
    }


def _combo_equity(player_info: dict, combo_idx: int) -> float | None:
    equities = player_info.get("hand_eqs") or []
    if combo_idx >= len(equities):
        return None
    value = _float(equities[combo_idx], -1.0)
    return value if value >= 0 else None


def _range_equity_story(hero: str, villain: str, hero_pi: dict, villain_pi: dict,
                        range_plan: dict, action_contract: dict,
                        actual: dict | None, preferred: dict) -> dict:
    hero_eq = _float(hero_pi.get("total_eq"), -1.0)
    villain_eq = _float(villain_pi.get("total_eq"), -1.0)
    gap = hero_eq - villain_eq if hero_eq >= 0 and villain_eq >= 0 else None
    check = range_plan["frequencies"].get("X", 0.0)
    aggression = sum(
        frequency for code, frequency in range_plan["frequencies"].items()
        if (code or "").startswith("R")
    )
    actual_code = (actual or {}).get("code") or ""
    preferred_code = preferred.get("code") or ""
    is_sizing_choice = (
        actual_code.startswith("R") and preferred_code.startswith("R")
    )
    use = "omit"
    interpretation = "平均 range equity 不改變主要因果解釋，教練回覆應省略"
    if gap is None:
        interpretation = "range equity 資料不完整，教練回覆應省略"
    elif is_sizing_choice:
        use = "prevents_bad_inference"
        interpretation = "平均 range equity 無法選擇下注 size，應改看各 size 的 range construction"
    elif check >= 0.95 and gap <= -0.04:
        use = "supports_plan"
        interpretation = (
            f"{hero} 的 range equity 劣勢與近乎全 range check 同方向；"
            "只需用來支持整體計畫，不必列原始數字"
        )
    elif aggression >= 0.75 and gap >= 0.04:
        use = "supports_plan"
        interpretation = (
            f"{hero} 的 range equity 優勢支持高頻進攻容量；"
            "單一 combo 是否入選仍由牌力角色與 blocker 決定"
        )
    elif (
        gap <= -0.04
        and actual_code == "F"
        and (action_contract.get("continue_frequency") or 0.0) >= 0.80
    ):
        use = "prevents_bad_inference"
        interpretation = (
            f"{hero} 的平均 range equity 雖落後，但這個 combo 仍應"
            f" {preferred['label']}；range 劣勢不能直接翻譯成 fold"
        )
    return {
        "use": use,
        "hero": hero,
        "villain": villain,
        "hero_equity": hero_eq if hero_eq >= 0 else None,
        "villain_equity": villain_eq if villain_eq >= 0 else None,
        "gap": gap,
        "interpretation": interpretation,
    }


def _defense_price_story(node: dict, combo_equity: float | None,
                         actual: dict | None, preferred: dict,
                         action_contract: dict) -> dict | None:
    pot_odds = node.get("pot_odds")
    if (
        pot_odds is None or pot_odds < 0 or combo_equity is None
        or (actual or {}).get("code") != "F"
        or preferred.get("code") == "F"
        or (action_contract.get("continue_frequency") or 0.0) < 0.80
        or combo_equity < pot_odds + 0.05
    ):
        return None
    return {
        "pot_odds": pot_odds,
        "combo_equity": combo_equity,
        "interpretation": (
            "這個 combo 的 raw equity 明顯高於面對此 size 的直接 pot odds，"
            "而且 solver 讓它高頻繼續；下注價格與防守門檻比平均 range equity 更直接"
        ),
    }


def _equity_denial_story(solution: dict, villain_pi: dict, node: dict,
                         made_category: str | None, preferred: dict) -> dict | None:
    spr = node.get("effective_spr")
    if (
        made_category not in _VULNERABLE_MADE_HANDS
        or preferred.get("code") != "RAI"
        or preferred.get("frequency", 0.0) < 0.70
        or spr is None or spr > 0.80
    ):
        return None
    made_shares = _category_shares(villain_pi)
    unpaired_share = sum(made_shares.get(name, 0.0) for name in _UNPAIRED_CATEGORIES)
    draw_share = sum(
        _float(row.get("total_frequency"))
        for row in villain_pi.get("draw_categories") or []
        if row.get("name") != "no_draw"
    )
    if max(unpaired_share, draw_share) < 0.15:
        return None
    return {
        "effective_spr": spr,
        "unpaired_share": unpaired_share,
        "draw_share": draw_share,
        "interpretation": (
            f"低 SPR 下，Hero 的{_MADE_ZH.get(made_category, made_category)}是脆弱成牌；"
            "all-in 向對手範圍中的未成牌與聽牌收取完整 realization 代價，"
            "避免讓它們便宜看到後續牌。這不是 range equity 優勢的推論"
        ),
    }


def _decision_type(preferred: dict, made_category: str | None,
                   range_plan: dict, equity_denial: dict | None) -> str:
    code = preferred.get("code") or ""
    if range_plan["facing_bet"] and code in {"F", "C"}:
        return "defense"
    if code.startswith("R"):
        if equity_denial:
            return "protection"
        if made_category in _VALUE_MADE_HANDS:
            return "value"
        if made_category in _UNPAIRED_CATEGORIES:
            return "bluff"
    return "strategy"


def _mix_strategy_story(actions: list[dict], actual: dict | None,
                        preferred: dict, ev_loss: float) -> dict | None:
    """Explain an equilibrium mix without turning frequency into correctness."""
    if (
        not actual
        or actual.get("frequency", 0.0) < 0.01
        or actual.get("code") == preferred.get("code")
        or ev_loss > 1e-9
    ):
        return None
    mixed = [
        row for row in actions
        if row.get("frequency", 0.0) >= 0.05
        or row.get("code") in {actual.get("code"), preferred.get("code")}
    ]
    if len(mixed) < 2:
        return None
    labels = [row["label"] for row in sorted(mixed, key=lambda row: -row["frequency"])]
    return {
        "actual_label": actual["label"],
        "actual_frequency": actual["frequency"],
        "preferred_label": preferred["label"],
        "preferred_frequency": preferred["frequency"],
        "mixed_labels": labels,
        "interpretation": (
            f"這個 combo 明確在 {'、'.join(labels)}之間混合；"
            "實戰動作是 solver 保留的分支，"
            "最高頻動作只是較常用，不是唯一正解"
        ),
    }


def _range_evidence(hero: str, villain: str, hero_pi: dict, villain_pi: dict) -> list[dict]:
    hero_cats = _category_shares(hero_pi)
    villain_cats = _category_shares(villain_pi)
    candidates = []
    for category in _STRONG_CATEGORIES:
        hero_share = hero_cats.get(category, 0.0)
        villain_share = villain_cats.get(category, 0.0)
        gap = hero_share - villain_share
        if max(hero_share, villain_share) < 0.025 or abs(gap) < 0.025:
            continue
        owner = hero if gap > 0 else villain
        candidates.append({
            "category": category,
            "label": _MADE_ZH.get(category, category),
            "owner": owner,
            "hero_share": hero_share,
            "villain_share": villain_share,
            "gap": gap,
        })
    # Keep hand-strength order instead of sorting only by percentage gap.  The
    # coaching question is "who owns the top of the range?", so a meaningful
    # flush/set gap is more instructive than a slightly larger two-pair gap.
    return candidates[:2]


def _action_composition(player_info: dict, code: str) -> dict[str, float]:
    masses = {}
    for category in (player_info.get("hand_categories") or []):
        mass = _float((category.get("actions_total_combos") or {}).get(code))
        if mass > 0:
            name = _normalize_category(category.get("name"))
            if name:
                masses[name] = masses.get(name, 0.0) + mass
    total = sum(masses.values())
    if total <= 0:
        return {}
    return {name: mass / total for name, mass in masses.items()}


def _size_structure(solution: dict, player_info: dict) -> dict | None:
    bets = []
    for row in (solution.get("action_solutions") or []):
        action = row.get("action") or {}
        code = action.get("code") or ""
        if not code.startswith("R") or _float(row.get("total_frequency")) < 0.005:
            continue
        ratio = _float(action.get("betsize_by_pot"), -1.0)
        if ratio < 0:
            continue
        composition = _action_composition(player_info, code)
        equity_profile = _advanced_equity_profile(row)
        if not composition and not equity_profile:
            continue
        polar_share = sum(composition.get(cat, 0.0) for cat in _POLAR_VALUE | _POLAR_AIR)
        bets.append({
            "code": code,
            "ratio": ratio,
            "frequency": _float(row.get("total_frequency")),
            "composition": composition,
            "polar_share": polar_share,
            "equity_profile": equity_profile,
        })
    if len(bets) < 2:
        return None
    largest = max(bets, key=lambda row: row["ratio"])
    common_smaller = max(
        (row for row in bets if row is not largest),
        key=lambda row: row["frequency"],
        default=None,
    )
    if not common_smaller:
        return None
    category_support = bool(
        largest["composition"] and common_smaller["composition"]
        and largest["polar_share"] - common_smaller["polar_share"] >= 0.10
    )
    large_profile = largest["equity_profile"]
    small_profile = common_smaller["equity_profile"]
    equity_support = bool(
        large_profile and small_profile
        and large_profile["strong"] >= 0.10
        and large_profile["weak"] >= 0.10
        and large_profile["strong"] - small_profile["strong"] >= 0.08
        and large_profile["weak"] - small_profile["weak"] >= 0.08
        and small_profile["middle"] - large_profile["middle"] >= 0.08
    )
    if not category_support and not equity_support:
        return None
    large_top = sorted(largest["composition"].items(), key=lambda item: -item[1])[:3]
    return {
        "larger_size": f"{_pct(largest['ratio'])}% pot",
        "smaller_size": f"{_pct(common_smaller['ratio'])}% pot",
        "evidence_source": (
            "categories_and_advanced_equity_buckets"
            if category_support and equity_support
            else ("advanced_equity_buckets" if equity_support else "hand_categories")
        ),
        "larger_profile": large_profile,
        "smaller_profile": small_profile,
        "large_size_main_categories": [
            {"category": name, "label": _MADE_ZH.get(name, name), "share": share}
            for name, share in large_top
        ],
        "interpretation": (
            "較大 size 的實際 action range 同時有更多強端與弱端、較少中段；"
            "這支持它比常用小 size 更 polar，但不證明平均 range equity 決定 size"
            if equity_support else
            "較大 size 的已驗證牌型組成比常用小 size 更偏向強牌與空氣"
        ),
    }


def _same_class_sensitivity(solution: dict, player_info: dict, hero_hand: str) -> dict:
    hero_class = gf.normalize_hand_name(hero_hand)
    board = {
        (solution.get("game") or {}).get("board", "")[i:i + 2]
        for i in range(0, len((solution.get("game") or {}).get("board", "")), 2)
    }
    rows = []
    player_range = player_info.get("range") or []
    blocker_rate = solution.get("blocker_rate") or []
    trash_rate = solution.get("unblocker_rate") or []
    for idx, combo in enumerate(gf._COMBO_INDEX):
        if gf._combo_to_hand_name(*combo) != hero_class or board.intersection(combo):
            continue
        weight = _float(player_range[idx]) if idx < len(player_range) else 0.0
        if weight <= 1e-6:
            continue
        actions = _exact_actions(solution, idx)
        rows.append({
            "actions": {row["code"]: row["frequency"] for row in actions},
            "value": _float(blocker_rate[idx], -1.0) if idx < len(blocker_rate) else -1.0,
            "trash": _float(trash_rate[idx], -1.0) if idx < len(trash_rate) else -1.0,
        })
    if len(rows) < 2:
        return {"level": "unknown", "max_action_spread": 0.0}
    action_codes = {code for row in rows for code in row["actions"]}
    max_spread = max(
        (
            max(row["actions"].get(code, 0.0) for row in rows)
            - min(row["actions"].get(code, 0.0) for row in rows)
            for code in action_codes
        ),
        default=0.0,
    )
    valid_values = [row["value"] for row in rows if row["value"] >= 0]
    valid_trash = [row["trash"] for row in rows if row["trash"] >= 0]
    removal_spread = max(
        (max(values) - min(values) for values in (valid_values, valid_trash) if values),
        default=0.0,
    )
    level = "high" if max_spread >= 0.35 or removal_spread >= 3 else "low"
    return {
        "level": level,
        "max_action_spread": max_spread,
        "removal_spread": removal_spread,
        "combo_count": len(rows),
    }


def _same_class_action_plan(solution: dict, player_info: dict, hero_hand: str,
                            preferred: dict) -> dict | None:
    """Confirm that reachable suit combos share the same preferred action.

    This is narrower than a polarization claim.  It supports wording such as
    "the solver assigns this value hand class to the all-in bucket" without
    pretending that average range equity explains why that size exists.
    """
    preferred_code = preferred.get("code") or ""
    if not preferred_code:
        return None
    hero_class = gf.normalize_hand_name(hero_hand)
    board = {
        (solution.get("game") or {}).get("board", "")[i:i + 2]
        for i in range(0, len((solution.get("game") or {}).get("board", "")), 2)
    }
    player_range = player_info.get("range") or []
    frequencies = []
    weights = []
    for idx, combo in enumerate(gf._COMBO_INDEX):
        if gf._combo_to_hand_name(*combo) != hero_class or board.intersection(combo):
            continue
        weight = _float(player_range[idx]) if idx < len(player_range) else 0.0
        if weight <= 1e-6:
            continue
        actions = _exact_actions(solution, idx)
        frequency = next(
            (row["frequency"] for row in actions if row.get("code") == preferred_code),
            0.0,
        )
        frequencies.append(frequency)
        weights.append(weight)
    if len(frequencies) < 2 or min(frequencies) < 0.85:
        return None
    weighted_frequency = sum(
        frequency * weight for frequency, weight in zip(frequencies, weights)
    ) / sum(weights)
    return {
        "hand_class": hero_class,
        "action_code": preferred_code,
        "action_label": preferred["label"],
        "combo_count": len(frequencies),
        "minimum_frequency": min(frequencies),
        "weighted_frequency": weighted_frequency,
        "interpretation": (
            f"同一 {hero_class} hand class 的所有可達花色組合都一致偏好"
            f" {preferred['label']}；這證明的是 class-level size allocation，"
            "平均 range equity 不能解釋這個 size，也沒有足夠資料推廣到整體 range construction"
        ),
    }


def _size_choice_story(solution: dict, player_info: dict, hero_hand: str,
                       made_category: str | None, actual: dict | None,
                       preferred: dict) -> dict | None:
    """Select a conservative, exact-class explanation for a value size error."""
    actual_code = (actual or {}).get("code") or ""
    preferred_code = preferred.get("code") or ""
    if (
        made_category not in _VALUE_MADE_HANDS
        or not actual_code.startswith("R")
        or not preferred_code.startswith("R")
        or actual_code == preferred_code
        or preferred.get("frequency", 0.0) < 0.70
    ):
        return None
    plan = _same_class_action_plan(solution, player_info, hero_hand, preferred)
    if not plan:
        return None
    return {
        **plan,
        "actual_label": actual["label"],
        "preferred_label": preferred["label"],
    }


def _blocker_story(solution: dict, player_info: dict, combo_idx: int,
                   hero_hand: str, made_category: str | None,
                   aggressive_code: str | None, street: str) -> dict | None:
    if street not in {"turn", "river"} or made_category not in {
        "no_made_hand", "king_high", "ace_high"
    } or not (aggressive_code or "").startswith("R"):
        return None
    values = solution.get("blocker_rate") or []
    trash = solution.get("unblocker_rate") or []
    if combo_idx >= len(values) or combo_idx >= len(trash):
        return None
    value_score = _float(values[combo_idx], -1.0)
    trash_score = _float(trash[combo_idx], -1.0)
    if value_score < 0 or trash_score < 0:
        return None
    score_gap = value_score - trash_score
    if score_gap >= 3:
        direction = "favorable"
        interpretation = "blocker 對 bluff 選牌有利，但不是下注成立的唯一原因"
    elif score_gap <= -3:
        direction = "unfavorable"
        interpretation = "blocker 對這個 bluff 不利；它不是支持下注的理由"
    else:
        direction = "neutral"
        interpretation = "blocker 沒有明確提供優勢，不應把它寫成主要原因"
    sensitivity = _same_class_sensitivity(solution, player_info, hero_hand)
    return {
        "direction": direction,
        "interpretation": interpretation,
        "value_removal": value_score,
        "trash_removal": trash_score,
        "same_class_suit_sensitivity": sensitivity["level"],
    }


def _opponent_card_action_effects(solution: dict, action_code: str | None) -> dict | None:
    """Extract GTOW's blocker-tab delta with its original direction intact.

    At a Hero action node this field answers: "if Villain holds card X, how
    does Hero's action frequency change?"  It is intentionally kept separate
    from Hero's value/trash removal scores.  Using Hero's own cards here would
    reverse the condition and create a convincing but false blocker story.
    """
    if not action_code:
        return None
    effects = []
    for row in solution.get("blockers_frequencies") or []:
        card = row.get("card")
        if not card:
            continue
        delta = next(
            (
                _float(action.get("frequency"))
                for action in row.get("actions") or []
                if action.get("action") == action_code
            ),
            None,
        )
        if delta is None or abs(delta) < 0.0005:
            continue
        effects.append({
            "card": card,
            "delta": delta,
            "delta_pp": 100 * delta,
            "direction": "increase" if delta > 0 else "decrease",
        })
    if not effects:
        return None
    effects.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return {
        "semantics": "conditional_on_villain_card",
        "action_code": action_code,
        "largest_effects": effects[:3],
        "scope": (
            "只表示 Villain 持有該單卡時 Hero action frequency 的條件差；"
            "不可解讀成 Hero 手牌 blocker，亦不可命名被移除的具體 combo"
        ),
    }


def _decision(context: dict, spot: dict, solution: dict, hero_hand: str) -> dict | None:
    street = spot.get("street") or ""
    if street not in {"flop", "turn", "river"}:
        return None
    combo_idx = gf.combo_index_for_hand(hero_hand)
    if combo_idx is None:
        return None
    hero = spot.get("solver_hero_pos") or context.get("hero_position")
    players = _players(solution)
    hero_pi = players.get(hero)
    if not hero_pi or (solution.get("game") or {}).get("active_position") != hero:
        return None
    villain_rows = [(pos, pi) for pos, pi in players.items() if pos != hero]
    if not villain_rows:
        return None
    villain, villain_pi = villain_rows[0]
    player_range = hero_pi.get("range") or []
    reach_weight = _float(player_range[combo_idx]) if combo_idx < len(player_range) else 0.0
    if reach_weight <= 0:
        return None

    actions = _exact_actions(solution, combo_idx)
    if not actions:
        return None
    actual_code = spot.get("taken_code")
    actual = next((row for row in actions if row["code"] == actual_code), None)
    best = (
        actual
        if actual and actual.get("frequency", 0.0) >= 0.01
        else _best_in_mix_action(actions)
    )
    preferred = max(actions, key=lambda row: row["frequency"])
    ev_loss = max(0.0, best["ev_bb"] - actual["ev_bb"]) if actual else 0.0
    pot = _float((solution.get("game") or {}).get("pot"))
    ev_loss_pot = ev_loss / pot if pot > 0 else None
    board = (solution.get("game") or {}).get("board") or ""
    made_category = _category_name(solution, hero_pi, combo_idx)
    draw_category = _draw_name(solution, hero_pi, combo_idx)
    hand_truth = _deterministic_hand_truth(hero_hand, board)
    if hand_truth:
        truth_made = hand_truth.get("made_hand")
        if truth_made == "high_card":
            truth_made = "no_made_hand"
        # GTOW's range-relative pair buckets are more precise than the generic
        # evaluator (for example 88 on J93 is third_pair, not merely low_pair).
        # Use poker-rules made-hand truth only when the solved node lacks one.
        if not made_category and truth_made in _MADE_ZH:
            made_category = truth_made
        truth_draws = hand_truth.get("draws") or []
        for candidate in ("nut_flush_draw", "flush_draw", "combo_draw", "oesd", "gutshot"):
            if candidate in truth_draws:
                draw_category = candidate
                break
    percentiles = hero_pi.get("eq_percentile") or []
    percentile = _float(percentiles[combo_idx], -1.0) if combo_idx < len(percentiles) else -1.0
    band = _range_band(percentile if percentile >= 0 else None)

    range_plan = _range_plan(solution)
    action_contract = _action_contract(actions, range_plan)
    node_context = _node_context(solution, spot, hero, villain)
    combo_equity = _combo_equity(hero_pi, combo_idx)
    evidence = _range_evidence(hero, villain, hero_pi, villain_pi)
    range_structure = _range_structure(hero, villain, hero_pi, villain_pi)
    size_structure = _size_structure(solution, hero_pi)
    aggressive_code = next(
        (
            action.get("code") for action in (actual, preferred)
            if action and (action.get("code") or "").startswith("R")
        ),
        None,
    )
    blocker = _blocker_story(
        solution, hero_pi, combo_idx, hero_hand, made_category, aggressive_code, street,
    )
    opponent_card_effects = _opponent_card_action_effects(
        solution, preferred.get("code"),
    )
    range_equity = _range_equity_story(
        hero, villain, hero_pi, villain_pi, range_plan, action_contract, actual, preferred,
    )
    defense_price = _defense_price_story(
        node_context, combo_equity, actual, preferred, action_contract,
    )
    equity_denial = _equity_denial_story(
        solution, villain_pi, node_context, made_category, preferred,
    )
    size_choice = _size_choice_story(
        solution, hero_pi, hero_hand, made_category, actual, preferred,
    )
    mix_strategy = _mix_strategy_story(actions, actual, preferred, ev_loss)
    decision = {
        "street": street,
        "board": board,
        "hero": hero,
        "villain": villain,
        "hero_hand": hero_hand,
        "hero_role": {
            "range_band": band,
            "made_hand": made_category,
            "made_hand_label": _MADE_ZH.get(made_category, made_category or "未知牌型"),
            "draw": draw_category,
            "draw_label": _DRAW_ZH.get(draw_category, draw_category or "聽牌狀態未知"),
            "combo_equity": combo_equity,
            "deterministic_hand": hand_truth,
        },
        "actual_action": actual,
        "preferred_action": preferred,
        "best_action_by_ev": best,
        "available_actions": actions,
        "ev_loss_bb": ev_loss,
        "ev_loss_pot": ev_loss_pot,
        "range_plan": range_plan,
        "action_contract": action_contract,
        "node_context": node_context,
        "range_equity": range_equity,
        "range_structure": range_structure,
        "defense_price": defense_price,
        "equity_denial": equity_denial,
        "range_evidence": evidence,
        "size_structure": size_structure,
        "size_choice": size_choice,
        "mix_strategy": mix_strategy,
        "blocker": blocker,
        "opponent_card_effects": opponent_card_effects,
        "confidence": "medium" if reach_weight < 0.005 else "high",
        "scope": (
            "這個 combo 因前街低頻線只少量到達此節點；結論只適用於目前深度、牌面與 action line"
            if reach_weight < 0.005
            else "只適用於這個深度、牌面與 action line；range 事實來自同一個已評分 solver node"
        ),
    }
    decision["decision_type"] = _decision_type(
        preferred, made_category, range_plan, equity_denial,
    )
    mechanisms = select_causal_mechanisms(decision)
    decision["causal_mechanisms"] = mechanisms
    decision["drivers"] = {
        "primary": mechanisms[0]["title"],
        "secondary": mechanisms[1]["title"] if len(mechanisms) > 1 else None,
    }
    return decision


def _teaching_score(decision: dict) -> float:
    score = min(5.0, (decision.get("ev_loss_pot") or 0.0) * 20)
    if decision["range_plan"]["strength"] in {"very_strong", "strong"}:
        score += 1.5
    score += min(2.0, sum(abs(row["gap"]) for row in decision["range_evidence"]) * 8)
    if decision.get("size_structure"):
        score += 1.0
    if decision.get("size_choice"):
        score += 1.5
    if decision.get("blocker") and decision["blocker"]["direction"] != "neutral":
        score += 1.0
    if decision.get("equity_denial") or decision.get("defense_price"):
        score += 1.5
    if (decision.get("range_equity") or {}).get("use") != "omit":
        score += 0.5
    return score


def build_teaching_digest(context: dict) -> dict | None:
    """Cover every solved Hero decision and select at most two to teach deeply."""
    if context.get("no_hero_hand"):
        return None
    hero_hand = _raw_hero_hand(context)
    if not hero_hand:
        return None
    all_decisions = []
    postflop_decisions = []
    for spot, solution in zip(context.get("hero_spots") or [], context.get("solutions") or []):
        if not solution:
            continue
        item = (
            _preflop_decision(context, spot, solution, hero_hand)
            if spot.get("street") == "preflop"
            else _decision(context, spot, solution, hero_hand)
        )
        if item:
            all_decisions.append(item)
            if item["street"] != "preflop":
                postflop_decisions.append(item)
    if not all_decisions:
        return None

    _label_all_decisions(all_decisions)

    deviations = [
        row for row in postflop_decisions
        if (row.get("ev_loss_pot") or 0.0) >= 0.003
    ]
    selected = []
    if deviations:
        selected.append(max(deviations, key=lambda row: row.get("ev_loss_pot") or 0.0))
    remaining = [row for row in postflop_decisions if row not in selected]
    if remaining:
        best_teaching = max(remaining, key=_teaching_score)
        if not selected or _teaching_score(best_teaching) >= 2.0:
            selected.append(best_teaching)
    if not selected and postflop_decisions:
        selected = [max(postflop_decisions, key=_teaching_score)]
    selected.sort(key=lambda row: ("flop", "turn", "river").index(row["street"]))

    allowed_categories = set()
    allowed_percentages = set()
    allowed_bb = set()
    for row in all_decisions:
        for action_key in ("actual_action", "preferred_action", "best_action_by_ev"):
            action = row.get(action_key)
            if action:
                allowed_percentages.add(round(100 * action["frequency"], 1))
                if action["pot_ratio"] >= 0:
                    allowed_percentages.add(round(100 * action["pot_ratio"], 1))
        if row.get("ev_loss_bb") is not None:
            allowed_bb.add(round(row["ev_loss_bb"], 3))
        if row.get("ev_loss_pot") is not None:
            allowed_percentages.add(round(100 * row["ev_loss_pot"], 1))
    for row in selected:
        role = row["hero_role"]
        if role.get("made_hand"):
            allowed_categories.add(role["made_hand"])
        if role.get("draw") in {"flush_draw", "nut_flush_draw"}:
            allowed_categories.add("flush_draw")
        if role.get("draw") in {"onecard_bdfd", "twocards_bdfd"}:
            allowed_categories.add("backdoor_flush")
        if role.get("draw") in {"gutshot", "oesd"}:
            allowed_categories.add("straight_draw")
        for evidence in row["range_evidence"]:
            allowed_categories.add(evidence["category"])
            allowed_percentages |= {
                round(100 * evidence["hero_share"], 1),
                round(100 * evidence["villain_share"], 1),
            }
        for action_key in ("actual_action", "preferred_action", "best_action_by_ev"):
            action = row.get(action_key)
            if action:
                allowed_percentages.add(round(100 * action["frequency"], 1))
                if action["pot_ratio"] >= 0:
                    allowed_percentages.add(round(100 * action["pot_ratio"], 1))
        for action in row.get("available_actions") or []:
            if action.get("frequency", 0.0) >= 0.005 and action.get("pot_ratio", -1.0) >= 0:
                allowed_percentages.add(round(100 * action["pot_ratio"], 1))
        allowed_percentages |= {
            round(100 * frequency, 1)
            for frequency in row["range_plan"]["frequencies"].values()
        }
        allowed_percentages |= {
            round(100 * frequency, 1)
            for frequency in row["action_contract"]["frequencies"].values()
        }
        aggression = sum(
            frequency
            for code, frequency in row["range_plan"]["frequencies"].items()
            if (code or "").startswith("R")
        )
        allowed_percentages.add(round(100 * aggression, 1))
        if row["range_plan"]["facing_bet"]:
            allowed_percentages.add(round(
                100 * (1 - row["range_plan"]["frequencies"].get("F", 0.0)), 1,
            ))
        continue_frequency = row["action_contract"].get("continue_frequency")
        if continue_frequency is not None:
            allowed_percentages.add(round(100 * continue_frequency, 1))
        if row.get("defense_price"):
            allowed_percentages.add(round(100 * row["defense_price"]["pot_odds"], 1))
        if row.get("ev_loss_bb") is not None:
            allowed_bb.add(round(row["ev_loss_bb"], 3))
        if row.get("ev_loss_pot") is not None:
            allowed_percentages.add(round(100 * row["ev_loss_pot"], 1))
        if row.get("size_structure"):
            allowed_categories |= {
                item["category"] for item in row["size_structure"]["large_size_main_categories"]
            }
            for key in ("larger_size", "smaller_size"):
                match = re.search(r"\d+(?:\.\d+)?", row["size_structure"][key])
                if match:
                    allowed_percentages.add(float(match.group()))

    caveats = []
    validation_warning = (context.get("validation") or {}).get("user_warning")
    if validation_warning:
        caveats.append("手牌解析有 validation warning，教學結論需降級看待")
    if any((spot.get("depth_caveat") for spot in (context.get("hero_spots") or []))):
        caveats.append("不同街使用的 solver depth bucket 不完全一致")
    digest_confidence = (
        "medium"
        if caveats or any(row.get("confidence") == "medium" for row in all_decisions)
        else "high"
    )
    return {
        "confidence": digest_confidence,
        "all_decisions": all_decisions,
        "decisions": selected[:2],
        "allowed_categories": sorted(allowed_categories),
        "allowed_percentages": sorted(allowed_percentages),
        "allowed_bb": sorted(allowed_bb),
        "caveats": caveats,
    }


def _category_sentence(decision: dict) -> str | None:
    evidence = decision.get("range_evidence") or []
    if not evidence:
        return None
    grouped: dict[str, list[str]] = {}
    for row in evidence:
        grouped.setdefault(row["owner"], []).append(row["label"])
    parts = []
    for owner, labels in grouped.items():
        parts.append(f"{owner} 的{'、'.join(labels)}較多")
    return "；".join(parts)


def build_decision_evidence(context: dict, spot: dict,
                            solution: dict) -> dict | None:
    """Build the full causal card for one exact cached Hero decision.

    Unlike ``build_teaching_digest``, this does not rank or discard decisions.
    Follow-up tools already resolved a precise node and need that node's range
    structure, removal metrics, and causal gates.
    """
    hero_hand = _raw_hero_hand(context)
    if not hero_hand or not solution:
        return None
    return _decision(context, spot, solution, hero_hand)


def render_decision_evidence(decision: dict | None) -> list[str]:
    """Render compact machine-grounded causal facts for a follow-up LLM."""
    if not decision:
        return []
    lines = []
    role = decision.get("hero_role") or {}
    lines.append(
        "  Hero range 角色："
        f"{role.get('range_band', '未知區段')}的"
        f"{role.get('made_hand_label', '未知牌型')}，"
        f"{role.get('draw_label', '聽牌狀態未知')}"
    )
    drivers = decision.get("drivers") or {}
    if drivers.get("primary"):
        suffix = f"；次要機制：{drivers['secondary']}" if drivers.get("secondary") else ""
        lines.append(f"  因果優先序：{drivers['primary']}{suffix}")

    range_equity = decision.get("range_equity") or {}
    if range_equity.get("use") != "omit":
        hero_eq = range_equity.get("hero_equity")
        villain_eq = range_equity.get("villain_equity")
        numbers = ""
        if hero_eq is not None and villain_eq is not None:
            numbers = (
                f"（{range_equity.get('hero')} {_pct(hero_eq)}% vs "
                f"{range_equity.get('villain')} {_pct(villain_eq)}%）"
            )
        lines.append(
            f"  Range equity gate={range_equity.get('use')}{numbers}："
            f"{range_equity.get('interpretation')}"
        )

    structure = decision.get("range_structure") or {}
    ownership = []
    for key in ("nut_region", "strong_region"):
        row = structure.get(key) or {}
        if row.get("owner"):
            ownership.append(
                f"{row['owner']} 的{row.get('label')}較多"
                f"（Hero {_pct(row.get('hero_share', 0.0))}% / "
                f"Villain {_pct(row.get('villain_share', 0.0))}%）"
            )
    if ownership:
        lines.append(
            "  Range 強端 proxy：" + "；".join(ownership)
            + "；90–100% 區域不是 literal nuts"
        )

    category_story = _category_sentence(decision)
    if category_story:
        lines.append(f"  可辨認強牌類別：{category_story}")
    if decision.get("defense_price"):
        lines.append(f"  Pot-odds gate：{decision['defense_price']['interpretation']}")
    if decision.get("equity_denial"):
        lines.append(f"  Equity denial：{decision['equity_denial']['interpretation']}")
    if decision.get("size_choice"):
        lines.append(f"  Exact-class sizing：{decision['size_choice']['interpretation']}")
    if decision.get("size_structure"):
        size = decision["size_structure"]
        labels = "、".join(
            row["label"] for row in size.get("large_size_main_categories", [])[:2]
        )
        category_note = f"；較大 size 主要是 {labels}" if labels else ""
        lines.append(
            f"  Size construction：{size['larger_size']} 比 {size['smaller_size']} 更 polar；"
            f"{size['interpretation']}{category_note}"
        )

    blocker = decision.get("blocker")
    if blocker:
        lines.append(
            "  GTOW removal metrics："
            f"value removal {blocker['value_removal']:.2f}；"
            f"trash removal {blocker['trash_removal']:.2f}；"
            f"方向 {blocker['direction']}；{blocker['interpretation']}；"
            f"同 hand class 花色敏感度 {blocker['same_class_suit_sensitivity']}"
        )
    card_effects = decision.get("opponent_card_effects")
    if card_effects:
        rendered = "、".join(
            f"Villain 持 {row['card']} 時 Hero 該 action "
            f"{'增加' if row['direction'] == 'increase' else '減少'} "
            f"{abs(row['delta_pp']):.1f}pp"
            for row in card_effects.get("largest_effects", [])[:2]
        )
        if rendered:
            lines.append(
                f"  Opponent-card conditional delta：{rendered}；"
                "這不是 Hero 手牌 blocker"
            )
    lines.append(f"  適用邊界：{decision.get('scope')}")
    return lines


def render_prompt_block(digest: dict | None) -> str:
    """Render the digest as a compact contract for the coaching LLM."""
    if not digest:
        return ""
    lines = [
        "【Deterministic 教學骨架｜唯一可用的因果材料】",
        f"資料信心：{digest['confidence']}；保真度：同一個已評分 solver node。",
        "",
        "【逐點判定｜核心判斷必須依序涵蓋每一點】",
    ]
    for decision in digest.get("all_decisions") or digest["decisions"]:
        lines.append(
            f"• {decision['decision_id']}｜{decision['coverage_label']}："
            f"{decision['coverage_verdict']}。"
        )
    lines.extend([
        "• 上述每點只需一句；非焦點不得自行補因果，exact combo mix 本身就是足夠理由。",
        "",
        "【深講焦點｜只在 *為什麼* 展開】",
    ])
    for index, decision in enumerate(digest["decisions"], 1):
        role = decision["hero_role"]
        actual = decision.get("actual_action")
        preferred = decision["preferred_action"]
        loss = decision.get("ev_loss_pot")
        if actual and loss is not None and loss >= 0.003:
            verdict = (
                f"Hero 的 {actual['label']} 比最佳 EV 動作少 {decision['ev_loss_bb']:.2f}bb"
                f"（約 {100 * loss:.1f}% pot）"
            )
        elif actual:
            if actual.get("code") == preferred.get("code"):
                purity = (
                    "幾乎純用此動作"
                    if preferred.get("frequency", 0.0) >= 0.97
                    else "是最高頻動作"
                )
                verdict = f"Hero 的 {actual['label']} 沒有實質 EV 損失，而且{purity}"
            elif actual.get("frequency", 0.0) >= 0.01:
                verdict = (
                    f"Hero 的 {actual['label']} 沒有實質 EV 損失，"
                    "而且是 solver 保留的 mix 分支"
                )
            else:
                verdict = (
                    f"Hero 的 {actual['label']} EV 代價低於實質門檻，"
                    "但不在可採信的 solver mix；不要稱為低頻 mix 分支"
                )
        else:
            verdict = f"solver 最偏好 {preferred['label']}"
        lines.extend([
            f"焦點 {index}｜{decision['street'].capitalize()} {decision['board']}",
            f"• 核心判定：{verdict}；solver 最常用 {preferred['label']}（約 {_pct(preferred['frequency'])}%）。",
            f"• Actor lock：{decision['node_context']['actor_lock']}；不可交換位置、preflop role 或 IP/OOP。",
            f"• Hero 角色：{role['range_band']}的{role['made_hand_label']}，{role['draw_label']}。",
            f"• Exact combo action：{decision['action_contract']['summary']}。",
            f"• 主要機制：{decision['drivers']['primary']}。",
            f"• 已觀測 range plan：{decision['range_plan']['text']}。",
        ])
        for mechanism in decision.get("causal_mechanisms") or []:
            forbidden = "、".join(mechanism["forbidden_inferences"])
            lines.append(
                f"• Rule contract [{mechanism['id']}｜{mechanism['evidence_tier']}｜"
                f"{mechanism['lane']}]：可說「{mechanism['claim_scope']}」；"
                f"不可外推「{forbidden}」。"
            )
        range_equity = decision.get("range_equity") or {}
        lines.append(
            f"• Range-equity gate（{range_equity.get('use', 'omit')}）："
            f"{range_equity.get('interpretation', '省略')}。"
        )
        range_structure = decision.get("range_structure") or {}
        nut_region = range_structure.get("nut_region") or {}
        strong_region = range_structure.get("strong_region") or {}
        if nut_region.get("owner") or strong_region.get("owner"):
            ownership = []
            if nut_region.get("owner"):
                ownership.append(
                    f"{nut_region['owner']} 的 90–100% equity 頂端區域更多"
                )
            if strong_region.get("owner"):
                ownership.append(
                    f"{strong_region['owner']} 的 70–100% equity 強端更多"
                )
            lines.append(
                "• Advanced equity structure：" + "；".join(ownership)
                + "；這是 range 結構 proxy，不等於 literal nuts，也不能單獨決定 size。"
            )
        if decision.get("defense_price"):
            lines.append(f"• Price/defense：{decision['defense_price']['interpretation']}。")
        if decision.get("equity_denial"):
            lines.append(f"• Equity denial：{decision['equity_denial']['interpretation']}。")
        if decision.get("mix_strategy"):
            lines.append(f"• Mix contract：{decision['mix_strategy']['interpretation']}。")
        if decision.get("size_choice"):
            lines.append(f"• Exact-class sizing：{decision['size_choice']['interpretation']}。")
        category_story = _category_sentence(decision)
        if category_story:
            lines.append(f"• 已驗證強牌結構：{category_story}。")
        if (
            decision.get("size_structure")
            and decision["drivers"].get("secondary") == "不同 bet size 的 range construction"
        ):
            size = decision["size_structure"]
            labels = "、".join(row["label"] for row in size["large_size_main_categories"][:2])
            category_note = f"；較大 size 主要由 {labels} 構成" if labels else ""
            lines.append(
                f"• Size construction：{size['larger_size']} 比 {size['smaller_size']} 更極化；"
                f"{size['interpretation']}{category_note}。"
            )
        blocker = decision.get("blocker")
        if blocker:
            lines.append(
                f"• Blocker：{blocker['interpretation']}；同 hand class 花色敏感度 "
                f"{blocker['same_class_suit_sensitivity']}。"
            )
        card_effects = decision.get("opponent_card_effects")
        if card_effects:
            effects = "、".join(
                f"Villain 持 {row['card']} 時該 action "
                f"{'上升' if row['direction'] == 'increase' else '下降'}"
                for row in card_effects["largest_effects"][:2]
            )
            lines.append(
                f"• Opponent-card conditional delta：{effects}；{card_effects['scope']}。"
            )
        if decision["drivers"].get("secondary"):
            lines.append(f"• 次要機制：{decision['drivers']['secondary']}。")
        lines.append(f"• 適用邊界：{decision['scope']}。")
    if digest.get("caveats"):
        lines.append("• 降級註記：" + "；".join(digest["caveats"]) + "。")
    lines.extend([
        "",
        "【輸出契約】",
        "只輸出三段：*核心判斷*、*為什麼*、*你要記得*。",
        "第一個字必須是 *核心判斷* 的星號；不要輸出 solver 校正、近似解、寒暄或其他前言。",
        "*核心判斷* 必須按 D1、D2…順序，每個決策恰好一個 bullet；每行用 `•` 開頭（不用 `-`），並使用骨架提供的 Preflop／Flop ①／Turn 等原樣標籤。",
        "每個 bullet 只寫『Hero 動作如何＋solver 偏好』，通常一句；不得省略打對的決策，也不要抄完整 action table。",
        "逐點判定可評價所有 D# 的正誤或 mix；只有深講焦點可以在 *為什麼* 展開 range、牌型、blocker 或因果機制。",
        "若沒有深講焦點，*為什麼* 只說 exact combo strategy 已足以判定，不補一般牌理。",
        "核心判定是硬契約：『沒有實質 EV 損失』的 solver mix 分支不得稱為錯誤；頻率較低只代表較少採用。",
        "若核心判定寫『EV 代價低於實質門檻，但不在可採信的 solver mix』，只能說 EV 影響很小；不可稱為 solver 保留、可用或低頻 mix。",
        "不要逐項重述骨架，不要展示 percentile、removal score 或完整頻率表；全文最多引用 3 個真正有教學價值的數字。",
        "數字配額：每個焦點最多 1 個，優先保留 EV loss 或 preferred frequency；不要同時寫 bb 與 % pot，也不要自行估算 SPR。",
        "*為什麼*只選主要機制與最多一個次要機制，使用『同花／set／順子／兩對／頂對／未成牌』等人能理解的 range 詞彙。",
        "Actor lock 是硬契約：不得把 Hero/Villain、opener/caller/3-bettor 或 IP/OOP 對調。",
        "『100% 繼續』不等於『100% call』；只能沿用 Exact combo action 的 action bucket。",
        "Range equity 只有 gate 為 supports_plan 或 prevents_bad_inference 時才能提；gate=omit 時完全省略。",
        "只有骨架明示 Equity denial 時，才能談低 SPR、脆弱成牌與拒絕便宜 realization；不得虛構對手精確 fold equity。",
        "Pot odds／raw equity 只能支持『繼續而非 fold』，不能單獨用來選 call、raise 或 all-in。",
        "只有骨架明示 Exact-class sizing 時，才能說同 hand class 被分配到某個 size；這不等於整體 range 更極化。",
        "Blocker 只能沿用骨架給的方向；不可自行聲稱 Hero 阻擋某個具體 combo、順子、同花或 nuts。",
        "Opponent-card conditional delta 是『Villain 持該牌時 Hero 策略如何變』；不得倒轉成 Hero 持牌造成的 blocker 故事。",
        "不得把脆弱成牌稱為半詐唬，也不要自行加入乾濕、連接性、驚悚牌等未驗證的 board texture。",
        "強牌類別差距只能說『誰的同花／set／兩對等更多』；不可擴寫成整體 range 或牌面必然有利／不利。",
        "不可列舉骨架沒有提供的具體手牌；不可把相關性寫成唯一因果。",
        "沒有 opponent response facts；禁止聲稱對手一定會 call/fold、為了誘導詐唬、或為了保護 check range。",
        "骨架只證明目前這一個 action；不得把 check 擴寫成 check-fold、check-call 或 check-raise。",
        "*為什麼*最多三句，只解釋深講焦點；*你要記得*只給一條可帶走的 heuristic，並保留 exact-node 邊界。",
        "若上層要求 FOLLOWUP，每題必須另起一行並以 `FOLLOWUP:` 開頭；不得用普通 bullet 或編號代替。",
    ])
    return "\n".join(lines)


def render_fallback(digest: dict) -> str:
    """Deterministic three-section answer used when the prose audit fails."""
    coverage = digest.get("all_decisions") or digest["decisions"]
    focus = digest["decisions"]
    primary = focus[0] if focus else coverage[0]

    def _core(decision: dict) -> str:
        actual = decision.get("actual_action")
        preferred = decision["preferred_action"]
        loss = decision.get("ev_loss_pot") or 0.0
        if actual and loss >= 0.003:
            severity = "小漏洞" if loss < 0.03 else ("明顯失誤" if loss < 0.10 else "嚴重錯誤")
            return f"{actual['label']} 是{severity}，應偏向 {preferred['label']}"
        if actual:
            if actual.get("code") == preferred.get("code"):
                return f"{actual['label']} 是正確選擇"
            if actual.get("frequency", 0.0) < 0.01:
                return (
                    f"{actual['label']} EV 影響很小，"
                    f"但不在可採信的 solver mix，應偏向 {preferred['label']}"
                )
            return (
                f"{actual['label']} 沒有實質 EV 損失，是 solver 保留的 mix，"
                f"主要仍是 {preferred['label']}"
            )
        return f"應偏向 {preferred['label']}"

    core_parts = [
        f"• {decision['coverage_label']}：{_core(decision)}"
        for decision in coverage
    ]
    reasons = []
    for decision in focus:
        role = decision["hero_role"]
        pieces = [
            f"{decision['street'].capitalize()} 時它是 {role['range_band']}的"
            f"{role['made_hand_label']}，{role['draw_label']}"
        ]
        if decision.get("equity_denial"):
            pieces.append(decision["equity_denial"]["interpretation"])
        elif decision.get("mix_strategy"):
            pieces.append(decision["mix_strategy"]["interpretation"])
        elif decision.get("defense_price"):
            pieces.append(decision["defense_price"]["interpretation"])
        elif decision.get("size_choice"):
            size_choice = decision["size_choice"]
            pieces.append(
                f"同一 {size_choice['hand_class']} 手牌類別的可達花色組合都偏好"
                f" {size_choice['preferred_label']}；這是同類手牌的尺寸分配，"
                "不是由平均 range equity 推出"
            )
        elif (decision.get("range_equity") or {}).get("use") != "omit":
            pieces.append(decision["range_equity"]["interpretation"])
        category_story = _category_sentence(decision)
        if category_story:
            pieces.append(category_story)
        size = decision.get("size_structure")
        if size and decision.get("decision_type") == "value":
            labels = "、".join(
                item["label"] for item in size["large_size_main_categories"][:2]
            )
            if labels:
                pieces.append(
                    f"較大的 {size['larger_size']} range 更極化，主要包含 {labels}"
                )
            else:
                pieces.append(size["interpretation"])
        blocker = decision.get("blocker")
        if blocker and blocker["direction"] != "neutral":
            pieces.append(blocker["interpretation"])
        elif not category_story:
            pieces.append(decision["range_plan"]["text"])
        reasons.append("；".join(pieces) + "。")

    # The common two-street teaching shape (bad bluff candidate on one street,
    # range-created bluff capacity on the next) reads better as one causal
    # contrast than as two repeated blocker sentences.
    if len(focus) == 2:
        first, second = focus
        second_categories = _category_sentence(second)
        first_blocker = first.get("blocker")
        second_blocker = second.get("blocker")
        if first_blocker and second_categories and second.get("decision_type") == "bluff":
            first_role = first["hero_role"]
            contrast = (
                f"{first['street'].capitalize()} 時它是 {first_role['range_band']}的"
                f"{first_role['made_hand_label']}，{first_role['draw_label']}，而且 blocker 不利，"
                f"不是好的 bluff 候選。{second['street'].capitalize()} 時 {second_categories}，"
                "強牌結構讓 range 能容納 bluff；這手牌的 blocker 仍不利，所以正確下注並不是靠 blocker。"
            )
            if not second_blocker or second_blocker.get("direction") == "neutral":
                contrast = contrast.replace(
                    "；這手牌的 blocker 仍不利，所以正確下注並不是靠 blocker。", "。"
                )
            reasons = [contrast]
    if not reasons:
        reasons = [
            "這手只有 preflop 決策；exact hand class 的 solver action 分配"
            "已足以判定，不需要另外補一般牌理。"
        ]

    lesson = {
        "低 SPR 下的 equity denial 與脆弱成牌保護": "低 SPR 面對大注時，脆弱成牌可能用 jam 向未成牌與聽牌收取 realization 代價；不要只看平均 range equity。",
        "exact combo 的 mixed strategy 分配": "solver 明確保留的 mix 分支都不是錯誤；把頻率偏好和 EV 損失分開學，並且只在同一個 node 套用。",
        "下注價格、整體防守門檻與 Hero 在自身 range 的位置": "range 劣勢不等於 fold；先比較下注價格與防守門檻，再看 exact combo 在自身 range 的位置。",
        "價值 hand class 的 size allocation": "價值牌的 size 要看 solver 把這個 hand class 分配到哪個下注 bucket；平均 range equity 不能替你選 size。",
        "整體 range strategy": "當 solver 在同一節點採近乎全 range 的計畫時，先服從 range-level strategy，再談單一 combo 的微調。",
        "整體防守結構與 Hero 在自身 range 的位置": "range 劣勢不代表每個邊緣 combo 都要 fold；先看整體防守寬度，再看這手牌在自身 range 的位置。",
        "雙方強牌結構": "先找雙方誰擁有更多可辨認的強牌類別，再結合 exact combo 的角色與 EV 判斷行動。",
    }.get(
        (primary.get("drivers") or {}).get("primary"),
        "先判斷這手牌在自身 range 的角色與 EV，再服從這個 node 的 solver action。",
    )
    if (
        (primary.get("drivers") or {}).get("primary") == "雙方強牌結構"
        and any(
            (row.get("blocker") or {}).get("direction") not in {None, "neutral"}
            for row in digest["decisions"]
        )
    ):
        lesson = "先找雙方誰擁有更多可辨認的強牌類別，再用 blocker 排序 value 或 bluff 候選，而不是反過來編理由。"
    if (
        len(focus) > 1
        and any(row.get("decision_type") == "bluff" for row in focus)
        and _category_sentence(focus[1])
    ):
        lesson = "先看雙方強牌結構決定 range 能否容納 bluff，再用 combo 角色、EV 與 blocker 排序候選牌。"
    if any(
        row.get("decision_type") == "value"
        and (row.get("size_choice") or row.get("size_structure"))
        for row in focus
    ):
        lesson = (
            "價值牌先看同類手牌被分到哪個 size；只有比較過各 size 的實際 "
            "range construction，才能判斷較大 size 是否同時含更多強牌與 bluff。"
        )
    reach_note = ""
    if any(decision.get("confidence") == "medium" for decision in coverage):
        reach_note = "River 是前街低頻線後的條件式結論。"
    core_text = "\n".join(core_parts)
    return (
        f"*核心判斷*\n{core_text}\n\n"
        f"*為什麼*\n{' '.join(reasons)}\n\n"
        f"*你要記得*\n{lesson}{reach_note}這條結論只適用目前的深度、牌面與 action line。"
    )


@dataclass
class AuditResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _covered_decisions(digest: dict) -> list[dict]:
    """Every decision that must receive a verdict, including preflop."""
    return digest.get("all_decisions") or digest.get("decisions") or []


def _body_without_followups(text: str) -> str:
    kept = []
    seen_takeaway = False
    followup_block = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if "你要記得" in stripped:
            seen_takeaway = True
        if re.match(
            r"^(?:[*#_\s-]*follow[ -]?up|[*#_\s-]*你可以問|[*#_\s-]*提問建議)",
            stripped,
            re.I,
        ):
            followup_block = True
            continue
        if followup_block or re.match(r"^FOLLOWUP\s*[:：]", stripped, re.I):
            continue
        if (
            seen_takeaway
            and re.match(r"^(?:[•*+-]|\d+[.)、])\s*", stripped)
            and ("？" in stripped or "?" in stripped)
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


_ASCII_CARD_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[2-9TJQKA][cdhs])"
    r"(?:[\s,/-]*(?:[2-9TJQKA][cdhs]))*(?![A-Za-z0-9])",
    re.I,
)
_ASCII_CARD_RE = re.compile(r"([2-9TJQKA])([cdhs])", re.I)
_EMOJI_CARD_RE = re.compile(
    r"([2-9TJQKA])\s*(♣|♧|☘(?:️)?|♦|♢|🔷|♥(?:️)?|♡|♠(?:️)?|♤)",
    re.I,
)
_SUIT_NORMALIZATION = {
    "c": "c", "♣": "c", "♧": "c", "☘": "c", "☘️": "c",
    "d": "d", "♦": "d", "♢": "d", "🔷": "d",
    "h": "h", "♥": "h", "♥️": "h", "♡": "h",
    "s": "s", "♠": "s", "♠️": "s", "♤": "s",
}


def _normalized_cards(text: str) -> list[tuple[int, int, str]]:
    cards = []
    for run in _ASCII_CARD_RUN_RE.finditer(text or ""):
        for match in _ASCII_CARD_RE.finditer(run.group(0)):
            start = run.start() + match.start()
            end = run.start() + match.end()
            cards.append((start, end, match.group(1).upper() + match.group(2).lower()))
    for match in _EMOJI_CARD_RE.finditer(text or ""):
        suit = _SUIT_NORMALIZATION.get(match.group(2).lower())
        if suit:
            cards.append((match.start(), match.end(), match.group(1).upper() + suit))
    return sorted(set(cards))


def _audit_exact_combos(body: str, digest: dict) -> list[str]:
    allowed = {
        frozenset((row["hero_hand"][:2], row["hero_hand"][2:4]))
        for row in _covered_decisions(digest)
        if len(row.get("hero_hand") or "") == 4
    }
    board_cards = {
        card
        for row in _covered_decisions(digest)
        for _, _, card in _normalized_cards(row.get("board") or "")
    }
    violations = []
    cards = _normalized_cards(body)
    for left, right in zip(cards, cards[1:]):
        gap = body[left[1]:right[0]]
        if len(gap) > 3 or gap.strip(" -/,，"):
            continue
        pair = frozenset((left[2], right[2]))
        if left[2] in board_cards and right[2] in board_cards:
            continue
        if pair not in allowed:
            violations.append(f"unsupported exact combo {left[2]}{right[2]}")
    return violations


_ROLE_PATTERNS = {
    "opener": r"(?:opener|PFR|率先加注者|開牌者|開池者|開牌範圍|open(?:ed)?)",
    "caller": r"(?:caller|跟注者|平跟者)",
    "3bettor": r"(?:3[- ]?bettor|三下注者)",
    "4bettor": r"(?:4[- ]?bettor|四下注者)",
    "aggressor": r"(?:preflop\s*加注者|翻牌前加注者)",
}


def _claimed_role(fragment: str) -> str | None:
    for role, pattern in _ROLE_PATTERNS.items():
        if re.search(pattern, fragment, re.I):
            return role
    return None


def _role_matches(claimed: str, expected: str | None) -> bool:
    if claimed == "aggressor":
        return expected in {"opener", "3bettor", "4bettor", "5bettor"}
    return claimed == expected


def _audit_actor_contract(body: str, digest: dict) -> list[str]:
    violations = []
    for decision in digest["decisions"]:
        hero, villain = decision["hero"], decision["villain"]
        node = decision["node_context"]
        # Explicit aliases such as "CO（我們）" are high-signal and easy to lock.
        for match in re.finditer(
            r"(?<![A-Za-z0-9+])(UTG\+?[12]?|LJ|HJ|CO|BTN|SB|BB)\s*[（(]\s*"
            r"(我們|Hero|你|對手|Villain)",
            body,
            re.I,
        ):
            position = match.group(1).upper().replace("UTG1", "UTG+1").replace("UTG2", "UTG+2")
            alias = match.group(2).lower()
            expected = villain if alias in {"對手", "villain"} else hero
            if position != expected:
                violations.append(f"actor inversion {position} is not {match.group(2)}")

        role_by_actor = {
            hero: node.get("hero_preflop_role"),
            villain: node.get("villain_preflop_role"),
        }
        for actor, expected_role in role_by_actor.items():
            for match in re.finditer(
                rf"(?<![A-Za-z0-9+]){re.escape(actor)}(?![A-Za-z0-9+])[^。；\n]{{0,24}}"
                rf"({'|'.join(_ROLE_PATTERNS.values())})",
                body,
                re.I,
            ):
                claimed = _claimed_role(match.group(0))
                if claimed and not _role_matches(claimed, expected_role):
                    violations.append(
                        f"actor role mismatch {actor}:{claimed}!={expected_role}"
                    )

        for alias_pattern, expected_role, label in (
            (r"(?:Hero|你|我們)", node.get("hero_preflop_role"), "Hero"),
            (r"(?:Villain|對手)", node.get("villain_preflop_role"), "Villain"),
        ):
            for match in re.finditer(
                rf"{alias_pattern}[^。；，,\n但而]{{0,24}}"
                rf"({'|'.join(_ROLE_PATTERNS.values())})",
                body,
                re.I,
            ):
                claimed = _claimed_role(match.group(0))
                if claimed and not _role_matches(claimed, expected_role):
                    violations.append(
                        f"actor role mismatch {label}:{claimed}!={expected_role}"
                    )

        expected_relative = node.get("hero_relative_position")
        for match in re.finditer(
            r"(?:Hero|你|我們)[^。；\n]{0,20}(IP|OOP|有利位置|不利位置)",
            body,
            re.I,
        ):
            token = match.group(1).upper()
            claimed = "IP" if token in {"IP", "有利位置"} else "OOP"
            if expected_relative in {"IP", "OOP"} and claimed != expected_relative:
                violations.append(
                    f"relative-position mismatch Hero:{claimed}!={expected_relative}"
                )
    return violations


_ACTION_TEXT_PATTERNS = {
    "allin": r"(?:all[- ]?in|全下|打光)",
    "call": r"(?:call|跟注)",
    "fold": r"(?:fold|棄牌)",
    "check": r"(?:check|過牌)",
    "raise": r"(?:raise|加注)",
    "bet": r"(?:bet|下注)",
    "continue": r"(?:continue|繼續|防守)",
}


def _action_family(code: str, facing_bet: bool) -> str:
    if code == "F":
        return "fold"
    if code == "C":
        return "call"
    if code == "X":
        return "check"
    if code == "RAI":
        return "allin"
    if code.startswith("R"):
        return "raise" if facing_bet else "bet"
    return "other"


def _family_frequency(decision: dict, family: str, range_level: bool) -> float | None:
    if family == "continue":
        if range_level:
            return 1 - decision["range_plan"]["frequencies"].get("F", 0.0)
        return decision["action_contract"].get("continue_frequency")
    frequencies = (
        decision["range_plan"]["frequencies"]
        if range_level else decision["action_contract"]["frequencies"]
    )
    return sum(
        frequency for code, frequency in frequencies.items()
        if _action_family(code, decision["range_plan"]["facing_bet"]) == family
    )


def _sized_family_frequency(decision: dict, family: str, range_level: bool,
                            pot_ratio: float) -> float | None:
    """Frequency for one concrete bet/raise size rather than all aggression."""
    frequencies = (
        decision["range_plan"]["frequencies"]
        if range_level else decision["action_contract"]["frequencies"]
    )
    matches = [
        row for row in decision.get("available_actions") or []
        if abs(row.get("pot_ratio", -1.0) - pot_ratio) <= 0.015
        and _action_family(
            row.get("code") or "", decision["range_plan"]["facing_bet"],
        ) == family
    ]
    if not matches:
        return None
    return sum(frequencies.get(row["code"], 0.0) for row in matches)


def _nearest_action_family(sentence: str, start: int, end: int) -> str | None:
    """Bind a percentage to its closest action noun, not any noun nearby."""
    candidates = []
    for family, pattern in _ACTION_TEXT_PATTERNS.items():
        for match in re.finditer(pattern, sentence, re.I):
            if match.end() <= start:
                distance = start - match.end()
                bridge = sentence[match.end():start]
            elif match.start() >= end:
                distance = match.start() - end
                bridge = sentence[end:match.start()]
            else:
                distance = 0
                bridge = ""
            if re.search(r"[，,；;]", bridge):
                continue
            if distance <= 28:
                candidates.append((distance, match.start(), family))
    return min(candidates)[2] if candidates else None


def _audit_action_frequency_claims(body: str, digest: dict) -> list[str]:
    violations = []
    for sentence in re.split(r"[。！？\n]", body):
        for number in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|％)", sentence):
            tail = sentence[number.end():number.end() + 8]
            if re.search(r"(?:pot|底池)", tail, re.I):
                continue
            window = sentence[max(0, number.start() - 28):number.end() + 28]
            family = _nearest_action_family(sentence, number.start(), number.end())
            if not family:
                continue
            level_prefix = re.split(
                r"[，,；;]", sentence[:number.start()],
            )[-1]
            level_suffix = re.split(
                r"[，,；;]", sentence[number.end():],
            )[0][:18]
            range_level = bool(re.search(
                r"(?:整體|全\s*range|range)",
                level_prefix,
                re.I,
            )) or bool(re.search(r"(?:範圍|range)", level_suffix, re.I))
            candidates = _covered_decisions(digest)
            street_matches = [
                match for match in re.finditer(
                    r"(preflop|翻牌前|flop|turn|river|翻牌|轉牌|河牌)", sentence, re.I,
                )
                if match.start() <= number.start()
            ]
            street_match = street_matches[-1] if street_matches else None
            if street_match:
                street = {
                    "翻牌前": "preflop", "翻牌": "flop",
                    "轉牌": "turn", "河牌": "river",
                }.get(street_match.group(1).lower(), street_match.group(1).lower())
                candidates = [row for row in candidates if row["street"] == street]
            size_matches = [
                match for match in re.finditer(
                    r"(\d+(?:\.\d+)?)\s*(?:%|％)\s*(?:pot|底池)",
                    sentence[:number.start()],
                    re.I,
                )
                if number.start() - match.end() <= 40
                and _nearest_action_family(
                    sentence, match.start(), match.end(),
                ) == family
            ]
            size_ratio = float(size_matches[-1].group(1)) / 100 if size_matches else None
            claimed = float(number.group(1)) / 100
            if not any(
                (
                    frequency := (
                        _sized_family_frequency(row, family, range_level, size_ratio)
                        if size_ratio is not None else
                        _family_frequency(row, family, range_level)
                    )
                ) is not None
                and abs(claimed - frequency) <= 0.011
                for row in candidates
            ):
                violations.append(
                    f"action-frequency mismatch {family} {number.group(1)}%"
                )
    return violations


def _audit_category_ownership(body: str, digest: dict) -> list[str]:
    violations = []
    for decision in digest["decisions"]:
        expected = {
            row["category"]: row["owner"] for row in decision.get("range_evidence") or []
        }
        for category, owner in expected.items():
            terms = _CATEGORY_TERMS.get(category, (_MADE_ZH.get(category, category),))
            for actor in (decision["hero"], decision["villain"]):
                pattern = (
                    rf"(?<![A-Za-z0-9+]){re.escape(actor)}(?![A-Za-z0-9+])"
                    rf"[^。；，,、\n而]{{0,30}}(?:{'|'.join(re.escape(term) for term in terms)})"
                    rf"[^。；，,、\n而]{{0,12}}(?:更多|較多|more)"
                )
                if actor != owner and re.search(pattern, body, re.I):
                    violations.append(f"category owner mismatch {category}:{actor}!={owner}")
    return violations


def _audit_exact_hand_categories(body: str, digest: dict) -> list[str]:
    """Keep range-level category facts from becoming Hero's exact hand type."""
    violations = []
    term_rows = sorted(
        (
            (term, category)
            for category, terms in _EXACT_MADE_TERMS.items()
            for term in terms
        ),
        key=lambda row: len(row[0]),
        reverse=True,
    )
    category_pattern = "|".join(re.escape(term) for term, _ in term_rows)
    term_to_category = {term.lower(): category for term, category in term_rows}
    street_aliases = {
        "flop": "flop", "翻牌": "flop",
        "turn": "turn", "轉牌": "turn",
        "river": "river", "河牌": "river",
    }
    for sentence in re.split(r"[。！？\n]", body):
        if not sentence.strip():
            continue
        street_match = re.search(
            r"(?:flop|turn|river|翻牌|轉牌|河牌)", sentence, re.I,
        )
        street = street_aliases.get(street_match.group(0).lower()) if street_match else None
        candidates = [
            row for row in digest.get("decisions") or []
            if street is None or row.get("street") == street
        ]
        if not candidates:
            continue
        hero_positions = sorted({row.get("hero") for row in candidates if row.get("hero")})
        actor_pattern = "|".join(
            [r"你(?:的|這手)?", r"Hero(?:的)?", r"我們(?:的)?"]
            + [rf"{re.escape(position)}(?:的)?" for position in hero_positions]
        )
        exact_pattern = re.compile(
            rf"(?:{actor_pattern})[^。；，,\n]{{0,20}}?(?P<category>{category_pattern}|三條)",
            re.I,
        )
        for match in exact_pattern.finditer(sentence):
            fragment = match.group(0)
            suffix = sentence[match.end():match.end() + 12]
            if re.search(
                r"(?:range|範圍|牌組|組合|比例|更多|較多|more)",
                fragment,
                re.I,
            ):
                continue
            if re.search(r"(?:對手|Villain)[^。；，,\n]*$", fragment, re.I):
                continue
            if re.search(r"(?:更多|較多|比例)", suffix, re.I):
                continue
            token = match.group("category")
            claimed = "trips" if token == "三條" else term_to_category[token.lower()]
            expected = {
                (row.get("hero_role") or {}).get("made_hand") for row in candidates
            }
            compatible = (
                bool(expected.intersection({"set", "trips"}))
                if token == "三條" else claimed in expected
            )
            if not compatible:
                expected_label = "/".join(sorted(item for item in expected if item)) or "unknown"
                violations.append(
                    f"exact-combo category mismatch {street or 'unbound'}:"
                    f"{claimed}!={expected_label}"
                )
    return violations


_STREET_ALIASES = {
    "preflop": "preflop", "翻牌前": "preflop",
    "flop": "flop", "翻牌": "flop",
    "turn": "turn", "轉牌": "turn",
    "river": "river", "河牌": "river",
}


def _mentioned_streets(body: str) -> set[str]:
    """Return explicit streets without reading 翻牌 inside 翻牌前."""
    mentions = set()
    for match in re.finditer(
        r"preflop|翻牌前|flop|翻牌|turn|轉牌|river|河牌",
        body or "",
        re.I,
    ):
        mentions.add(_STREET_ALIASES[match.group(0).lower()])
    return mentions


def _audit_selected_streets(body: str, digest: dict) -> list[str]:
    """The narrator may issue verdicts only for solved, covered streets."""
    selected = {row.get("street") for row in _covered_decisions(digest)}
    return [
        f"unsupported street {street}"
        for street in sorted(_mentioned_streets(body) - selected)
    ]


def _audit_decision_coverage(body: str, digest: dict) -> list[str]:
    """Every solved Hero decision must appear in the core review exactly once."""
    core = re.split(r"\*為什麼\*", body or "", maxsplit=1)[0]
    violations = []
    for decision in _covered_decisions(digest):
        label = decision.get("coverage_label")
        if label and len(re.findall(re.escape(label), core, re.I)) != 1:
            violations.append(f"missing decision coverage {decision.get('decision_id', label)}")
    return violations


def _audit_action_verdicts(body: str, digest: dict) -> list[str]:
    """Prevent frequency preferences from being rewritten as EV mistakes."""
    violations = []
    street_terms = {
        "preflop": r"(?:preflop|翻牌前)",
        "flop": r"(?:flop|翻牌)",
        "turn": r"(?:turn|轉牌)",
        "river": r"(?:river|河牌)",
    }
    negative = re.compile(
        r"(?:錯誤|失誤|漏洞|打錯|不正確|不應該|不該|偏離|"
        r"(?:有|造成|產生|損失)[^。；\n]{0,8}EV\s*(?:loss|損失))",
        re.I,
    )
    negation = re.compile(
        r"(?:不是|並非|不算|不等於|沒有|無)[^。；\n]{0,10}"
        r"(?:錯誤|失誤|漏洞|EV\s*(?:loss|損失))",
        re.I,
    )
    positive = re.compile(
        r"(?:正確|沒(?:有)?實質\s*EV\s*損失|沒有\s*EV\s*損失|"
        r"不是[^。；\n]{0,6}(?:錯誤|失誤)|solver\s*(?:保留|支持)|mix\s*分支)",
        re.I,
    )

    def names_actual(sentence: str, decision: dict) -> bool:
        actual = decision.get("actual_action") or {}
        family = _action_family(
            actual.get("code") or "", decision["range_plan"]["facing_bet"],
        )
        pattern = _ACTION_TEXT_PATTERNS.get(family)
        return bool(
            (pattern and re.search(pattern, sentence, re.I))
            or re.search(r"(?:這個|這次|實戰)(?:動作|選擇|打法)|Hero 的行動", sentence, re.I)
        )

    core = re.split(r"\*為什麼\*", body or "", maxsplit=1)[0]
    core_sentences = [row for row in re.split(r"[。！？\n]", core) if row.strip()]
    for sentence in re.split(r"[。！？\n]", body or ""):
        if not sentence.strip():
            continue
        for decision in _covered_decisions(digest):
            street = decision.get("street")
            if not street or not re.search(street_terms[street], sentence, re.I):
                continue
            if not names_actual(sentence, decision):
                continue
            has_material_loss = (decision.get("ev_loss_pot") or 0.0) >= 0.003
            if not has_material_loss and negative.search(sentence) and not negation.search(sentence):
                violations.append(f"verdict mismatch {street}:in-mix-called-error")
            if has_material_loss and positive.search(sentence):
                violations.append(f"verdict mismatch {street}:loss-called-correct")
    if len(_covered_decisions(digest)) == 1:
        decision = _covered_decisions(digest)[0]
        street = decision["street"]
        has_material_loss = (decision.get("ev_loss_pot") or 0.0) >= 0.003
        for sentence in core_sentences:
            if not names_actual(sentence, decision):
                continue
            if not has_material_loss and negative.search(sentence) and not negation.search(sentence):
                violations.append(f"verdict mismatch {street}:in-mix-called-error")
            if has_material_loss and positive.search(sentence):
                violations.append(f"verdict mismatch {street}:loss-called-correct")
    return violations


def _audit_actual_mix_status(body: str, digest: dict) -> list[str]:
    """An off-mix action may be cheap in EV without becoming a solver branch."""
    violations = []
    for decision in _covered_decisions(digest):
        actual = decision.get("actual_action") or {}
        if actual.get("frequency", 0.0) >= 0.01:
            continue
        family = _action_family(
            actual.get("code") or "", decision["range_plan"]["facing_bet"],
        )
        action_pattern = _ACTION_TEXT_PATTERNS.get(family)
        if not action_pattern:
            continue
        for sentence in re.split(r"[。！？；\n]", body or ""):
            if not re.search(action_pattern, sentence, re.I):
                continue
            claims_supported = (
                re.search(r"(?:solver[^。；\n]{0,12}(?:保留|支持)|mix\s*分支)", sentence, re.I)
                or re.search(rf"(?:低頻|少量)[^。；\n]{{0,12}}{action_pattern}", sentence, re.I)
                or re.search(rf"{action_pattern}[^。；\n]{{0,10}}(?:可用|可採用)", sentence, re.I)
            )
            explicitly_off_mix = re.search(
                r"(?:不在|未進入|不是)[^。；\n]{0,12}(?:solver\s*)?mix|"
                r"(?:solver[^。；\n]{0,12}(?:不採用|未保留))",
                sentence,
                re.I,
            )
            if claims_supported and not explicitly_off_mix:
                violations.append(f"off-mix action called supported {decision['street']}")
                break
    return violations


def _audit_range_structure_causality(body: str, digest: dict) -> list[str]:
    """Secondary category ownership is context, not a derived action rule."""
    violations = []
    action_terms = (
        r"(?:策略|混合|bet|check|call|fold|raise|all[- ]?in|"
        r"下注|過牌|跟注|棄牌|加注|全下)"
    )
    causal_terms = r"(?:因此|所以|導致|決定|使得|讓|支撐|支持)"
    for decision in digest.get("decisions") or []:
        primary = (decision.get("drivers") or {}).get("primary")
        if primary == "雙方強牌結構":
            continue
        labels = [row.get("label") for row in decision.get("range_evidence") or []]
        labels = [label for label in labels if label]
        if not labels:
            continue
        category_terms = "|".join(re.escape(label) for label in labels)
        for clause in re.split(r"[。！？；，,\n]", body or ""):
            if not re.search(category_terms, clause, re.I):
                continue
            if (
                re.search(causal_terms, clause, re.I)
                and re.search(action_terms, clause, re.I)
            ):
                violations.append(
                    f"unsupported category-to-strategy causality {decision['street']}"
                )
                break
    return violations


def _audit_unsupported_categories(body: str, allowed: set[str]) -> list[str]:
    """Match human category terms without double-counting draws as made hands."""
    violations = []
    for category, terms in _CATEGORY_TERMS.items():
        if category in allowed:
            continue
        searchable = body
        if category == "flush":
            searchable = re.sub(
                r"(?:後門)?同花(?:聽牌|潛力)|flush\s+draw|backdoor\s+flush",
                "",
                searchable,
                flags=re.I,
            )
        elif category == "straight":
            searchable = re.sub(
                r"(?:順子聽牌|卡順|兩頭順)|straight\s+draw|gutshot|oesd",
                "",
                searchable,
                flags=re.I,
            )
        mentioned = False
        for term in terms:
            if term == "三條" and allowed.intersection({"set", "trips"}):
                continue
            if re.search(re.escape(term), searchable, re.I):
                mentioned = True
                break
        if mentioned:
            violations.append(f"unsupported category {category}")
    return violations


def audit_draft(text: str, digest: dict, source_texts: list[str] | None = None) -> AuditResult:
    """Reject unsupported combo/category/blocker claims in an initial reply.

    This deliberately audits factual nouns rather than prose style.  The LLM
    still has room to explain the verified mechanisms in its own words.
    """
    body = _body_without_followups(text)
    lowered = body.lower()
    violations = []

    # Keep the established combo whitelist, now fed with the teaching card too.
    try:
        import coach_facts

        covered = _covered_decisions(digest)
        anchor = (digest.get("decisions") or covered)[0]
        facts = coach_facts.initial_verdict_facts(
            list(source_texts or []) + [render_prompt_block(digest)],
            {"hand": {"hero_hand": anchor["hero_hand"]},
             "hero_hand": anchor["hero_hand"]},
        )
        board = anchor.get("board") or ""
        verdict = coach_facts.verify_claims(body, facts, board)
        violations.extend(f"unsupported combo {token}" for token in verdict.violations)
    except Exception:
        # The category/blocker gates below remain active even if the legacy
        # whitelist helper is unavailable during a partial deployment.
        pass
    violations.extend(_audit_exact_combos(body, digest))
    violations.extend(_audit_actor_contract(body, digest))
    violations.extend(_audit_action_frequency_claims(body, digest))
    violations.extend(_audit_category_ownership(body, digest))
    violations.extend(_audit_exact_hand_categories(body, digest))
    violations.extend(_audit_selected_streets(body, digest))
    violations.extend(_audit_decision_coverage(body, digest))
    violations.extend(_audit_action_verdicts(body, digest))
    violations.extend(_audit_actual_mix_status(body, digest))
    violations.extend(_audit_range_structure_causality(body, digest))

    allowed = set(digest.get("allowed_categories") or [])
    violations.extend(_audit_unsupported_categories(body, allowed))

    supports_blocker = any(row.get("blocker") for row in digest["decisions"])
    if not supports_blocker and ("blocker" in lowered or "阻斷" in body):
        violations.append("unsupported blocker explanation")
    if re.search(
        r"(?:阻擋|\bblocks?\b|\bblocking\b|\bblocked\b)[^。；\n]{0,32}"
        r"(?:同花|順子|flush|straight|nuts|set|兩對|關鍵牌|強牌|跟注範圍|call range)",
        lowered,
        re.I,
    ):
        violations.append("unsupported blocker target")
    nut_claim_body = body
    if any(
        (row.get("hero_role") or {}).get("draw") == "nut_flush_draw"
        for row in digest["decisions"]
    ):
        nut_claim_body = re.sub(
            r"(?:\bnuts?\b|堅果)\s*(?:同花聽牌|花聽|flush\s+draw)",
            "",
            nut_claim_body,
            flags=re.I,
        )
    if re.search(r"\b(?:nuts?|nut advantage)\b|堅果", nut_claim_body, re.I):
        violations.append("unsupported nuts claim")

    supports_range_equity = any(
        (row.get("range_equity") or {}).get("use") != "omit"
        for row in digest["decisions"]
    )
    range_equity_claim_body = re.sub(
        r"(?:不是|並非|不靠|不能(?:只)?靠|不要(?:只)?看)[^。；\n]{0,20}"
        r"(?:range\s+equity|範圍勝率)[^。；\n]{0,16}",
        "",
        body,
        flags=re.I,
    )
    if not supports_range_equity and re.search(
        r"(?:range|範圍|整體)[^。；\n]{0,20}(?:equity|勝率)[^。；\n]{0,16}(?:優勢|劣勢|領先|落後)"
        r"|(?:range equity|範圍勝率)",
        range_equity_claim_body,
        re.I,
    ):
        violations.append("unsupported range-equity explanation")

    supports_equity_denial = any(row.get("equity_denial") for row in digest["decisions"])
    if not supports_equity_denial and re.search(
        r"(?:equity denial|deny\s+equity|拒絕[^。；\n]{0,20}(?:equity|勝率|實現)|"
        r"(?:未成牌|聽牌)[^。；\n]{0,24}(?:免費|便宜)[^。；\n]{0,12}(?:realization|看到|實現))",
        lowered,
        re.I,
    ):
        violations.append("unsupported equity-denial explanation")
    if re.search(r"\bSPR\b|籌碼底池比", body, re.I) and not supports_equity_denial:
        violations.append("unsupported SPR explanation")
    if re.search(
        r"(?:raw\s+equity|原始勝率)[^。；\n]{0,40}"
        r"(?:all[- ]?in|全下|最激進|激進)|"
        r"(?:all[- ]?in|全下|最激進|激進)[^。；\n]{0,40}"
        r"(?:raw\s+equity|原始勝率)",
        body,
        re.I,
    ):
        violations.append("raw equity used to choose aggressive action")

    supports_size_choice = any(row.get("size_choice") for row in digest["decisions"])
    if not supports_size_choice and re.search(
        r"(?:hand class|同類手牌|同一[^。；\n]{0,16}(?:牌型|手牌)|"
        r"所有可達[^。；\n]{0,16}(?:combo|組合))[^。；\n]{0,36}"
        r"(?:size|bucket|下注尺度|分配到|all[- ]?in|全下)",
        body,
        re.I,
    ):
        violations.append("unsupported exact-class sizing claim")

    if re.search(
        r"(?:誘導|引誘)[^。；\n]{0,20}(?:詐唬|bluff)|"
        r"(?:保護|平衡)[^。；\n]{0,20}(?:check|過牌)[^。；\n]{0,12}(?:range|範圍)|"
        r"(?:用來|為了|透過)[^。；\n]{0,16}(?:平衡|保護)[^。；\n]{0,16}(?:整體)?(?:策略|range|範圍)|"
        r"(?:控制底池|pot control)",
        body,
        re.I,
    ):
        violations.append("unsupported induced-action explanation")
    if re.search(
        r"(?:check|過牌)\s*[-/]?\s*(?:fold|call|raise|棄牌|跟注|加注)",
        body,
        re.I,
    ):
        violations.append("unsupported future action plan")
    if re.search(
        r"(?:對手|Villain|UTG\+?[12]?|LJ|HJ|CO|BTN|SB|BB)[^。；\n]{0,36}"
        r"(?:一定|會|不會|很難|容易|可以|足以|無法|更有可能|較可能)[^。；\n]{0,16}"
        r"(?:跟注|棄牌|call|fold)|"
        r"(?:一定|會|不會|很難|容易|可以|足以|無法|更有可能|較可能)"
        r"[^。；\n]{0,20}(?:對手|Villain)[^。；\n]{0,20}"
        r"(?:跟注|棄牌|call|fold)|"
        r"(?:跟注|棄牌|call|fold)[^。；\n]{0,12}(?:的)?(?:對手|Villain)|"
        r"(?:詐唬|bluff)[^。；\n]{0,12}(?:成功率|成功機會)[^。；\n]{0,8}(?:高|低)",
        body,
        re.I,
    ):
        violations.append("unsupported opponent-response claim")

    supported_sprs = [
        row["equity_denial"]["effective_spr"]
        for row in digest["decisions"] if row.get("equity_denial")
    ]
    for match in re.finditer(
        r"\bSPR\b\s*(?:極低|很低|低)?\s*[（(]?\s*"
        r"(?:約(?:為)?|大約|為|=)?\s*(\d+(?:\.\d+)?)(?!\s*[- ]?bet\b|[A-Za-z])",
        body,
        re.I,
    ):
        prefix = body[max(0, match.start() - 8):match.start()]
        if re.search(r"(?:<|>|低於|高於|小於|大於)$", prefix):
            continue
        claimed = float(match.group(1))
        if supported_sprs and not any(abs(claimed - value) <= 0.10 for value in supported_sprs):
            violations.append(f"SPR mismatch {match.group(1)}")

    if re.search(r"(?:半詐唬|semi[- ]?bluff)", lowered, re.I):
        violations.append("unsupported semi-bluff label")

    supports_range_direction = any(
        (row.get("range_equity") or {}).get("use") == "supports_plan"
        or "雖落後" in (row.get("range_equity") or {}).get("interpretation", "")
        for row in digest["decisions"]
    )
    if not supports_range_direction and re.search(
        r"(?:牌面|flop|turn|river|翻牌|轉牌|河牌)[^。；\n]{0,24}"
        r"(?:對)?(?:你|Hero|我們|自己)?(?:的)?(?:\s*range|範圍)"
        r"[^。；\n]{0,8}(?:有利|不利)|"
        r"(?:雙方|Hero|Villain|你|我們|對手)[^。；\n]{0,20}"
        r"(?:range|範圍)[^。；\n]{0,12}(?:偏弱|偏強|更弱|更強)|"
        r"(?:牌面|board)(?:結構)?[^。；\n]{0,24}(?:對|更)"
        r"[^。；\n]{0,24}(?:有利|不利)",
        body,
        re.I,
    ):
        violations.append("unsupported broad range-advantage claim")

    supports_polar = any(
        row.get("size_structure")
        and row.get("drivers", {}).get("secondary") == "不同 bet size 的 range construction"
        for row in digest["decisions"]
    )
    if not supports_polar and ("極化" in body or "polarized" in lowered or "polarization" in lowered):
        violations.append("unsupported polarization claim")
    if re.search(
        r"價值(?:下注)?式?詐唬|\bvalue[- ]?bet(?:ting)?[- ]?bluff\b",
        lowered,
        re.I,
    ):
        violations.append("contradictory value-bluff label")
    for sentence in re.split(r"[。！？\n]", body):
        sentence_lower = sentence.lower()
        names_a_street = re.search(
            r"(?:flop|turn|river|翻牌|轉牌|河牌)", sentence_lower,
        )
        claims_transition = re.search(
            r"(?:幫助|增強|提升|改善|削弱|改變|help|improv|strengthen|weaken|shift)",
            sentence_lower,
        )
        names_range = "range" in sentence_lower or "範圍" in sentence
        if names_a_street and claims_transition and names_range:
            violations.append("unsupported range-transition claim")
            break
    if re.search(
        r"乾燥|濕潤|動態牌面|靜態牌面|連接性(?:強|高)|協調性(?:強|高)|驚悚牌|"
        r"牌面成對|翻牌成對|轉牌成對|河牌成對|低張連接|同花可能|"
        r"\bdry\b|\bwet\b|dynamic board|static board|connected board|scare card",
        lowered,
        re.I,
    ):
        violations.append("unsupported board-texture claim")

    allowed_positions = {
        position
        for row in _covered_decisions(digest)
        for position in (row.get("hero"), row.get("villain"))
        if position
    }
    mentioned_positions = set(re.findall(
        r"(?<![A-Za-z0-9+])(?:UTG\+?[12]?|LJ|HJ|CO|BTN|SB|BB)(?![A-Za-z0-9+])",
        body,
        re.I,
    ))
    normalized_positions = {
        position.upper().replace("UTG1", "UTG+1").replace("UTG2", "UTG+2")
        for position in mentioned_positions
    }
    for position in sorted(normalized_positions - allowed_positions):
        violations.append(f"unsupported position {position}")

    required_headers = ("*核心判斷*", "*為什麼*", "*你要記得*")
    if any(header not in body for header in required_headers):
        violations.append("missing teaching structure")
    response_limit = min(720, 360 + max(0, len(_covered_decisions(digest)) - 2) * 80)
    if len(re.sub(r"\s+", "", body)) > response_limit:
        violations.append("response too long")
    if any(row.get("confidence") == "medium" for row in _covered_decisions(digest)):
        if not re.search(r"低頻|少量到達|條件式|rare|low[- ]frequency", lowered, re.I):
            violations.append("missing low-reach caveat")

    numeric_claims = re.findall(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(%|％|bb)", body, re.I,
    )
    frequency_claims = re.findall(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:%|％)(?!\s*(?:pot|底池))",
        body,
        re.I,
    )
    bb_claims = re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*bb", body, re.I)
    if len(frequency_claims) + len(bb_claims) > 3:
        violations.append("too many numeric claims")
    allowed_percentages = list(digest.get("allowed_percentages") or [])
    allowed_bb = list(digest.get("allowed_bb") or [])
    for source in source_texts or []:
        for value, unit in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(%|％|bb)", source, re.I,
        ):
            target = allowed_bb if unit.lower() == "bb" else allowed_percentages
            target.append(float(value))
    for raw_value, unit in numeric_claims:
        value = float(raw_value)
        candidates = allowed_bb if unit.lower() == "bb" else allowed_percentages
        tolerance = 0.02 if unit.lower() == "bb" else 1.0
        if not any(abs(value - allowed) <= tolerance for allowed in candidates):
            violations.append(f"unsupported numeric claim {raw_value}{unit}")
    return AuditResult(ok=not violations, violations=sorted(set(violations)))
