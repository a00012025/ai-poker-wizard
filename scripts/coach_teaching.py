#!/usr/bin/env python3
"""Deterministic teaching skeletons for the initial coaching reply.

The solver decides *what is true*.  This module compresses those facts into a
small teaching card: the relevant range structure, the exact combo's role, and
whether blockers or size construction are actually supported by the node.  The
LLM may explain this card, but it must not invent a different range story.
"""
from __future__ import annotations

import logging
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
    "backdoor_flush_draw": "後門同花潛力",
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

# Action-range morphology is intentionally based on the solved action bucket,
# not a fixed slogan such as "big bet = nuts or air".  Two pair+ forms the
# durable value core; one-pair hands sit in the merge/bridge region; weak pairs
# and unpaired hands may supply the low-equity end of a polar construction.
_ACTION_VALUE_CORE = {
    "straight_flush", "quads", "full_house", "flush", "straight",
    "set", "trips", "two_pair",
}
_ACTION_MIDDLE = {
    "overpair", "top_pair", "second_pair", "third_pair",
}
_ACTION_WEAK = {
    "low_pair", "underpair", "no_made_hand", "king_high", "ace_high",
}

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
_LIVE_DRAW_CATEGORIES = {
    "gutshot", "oesd", "flush_draw", "nut_flush_draw", "combo_draw",
}
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
        "scope": "同一深度與 preflop node 的已驗證事實",
    }


def _off_tree_decision(context: dict, spot: dict, solution: dict,
                       hero_hand: str) -> dict | None:
    """Keep a real Hero action visible when the exact combo has 0% reach.

    There is no counterfactual EV comparison at this node, so this record is
    deliberately coverage-only: it can say why the action is ungraded, but it
    can never become a deep causal focus or a correct/incorrect verdict.
    """
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
    player_range = hero_pi.get("range") or []
    if combo_idx < len(player_range) and _float(player_range[combo_idx]) > 0:
        return None

    action_rows = solution.get("action_solutions") or []
    facing_bet = any(
        (row.get("action") or {}).get("code") in {"F", "C"}
        for row in action_rows
    )
    actual_code = spot.get("taken_code") or ""
    actual_solution = next(
        (
            row for row in action_rows
            if (row.get("action") or {}).get("code") == actual_code
        ),
        None,
    )
    if actual_solution:
        action = actual_solution.get("action") or {}
        actual_label = _action_label(actual_solution, facing_bet)
        pot_ratio = _float(action.get("betsize_by_pot"), -1.0)
    else:
        actual_label = {
            "X": "check", "C": "call", "F": "fold", "RAI": "all-in",
        }.get(actual_code, "實戰動作")
        pot_ratio = -1.0
    actual = {
        "code": actual_code,
        "label": actual_label,
        "frequency": 0.0,
        "ev_bb": 0.0,
        "pot_ratio": pot_ratio,
    }
    range_plan = _range_plan(solution)
    villain = next((position for position in players if position != hero), None)
    return {
        "street": street,
        "board": (solution.get("game") or {}).get("board") or "",
        "hero": hero,
        "villain": villain,
        "hero_hand": hero_hand,
        "actual_action": actual,
        "preferred_action": None,
        "best_action_by_ev": None,
        "available_actions": [],
        "ev_loss_bb": None,
        "ev_loss_pot": None,
        "range_plan": range_plan,
        "action_contract": {
            "mode": "off_tree",
            "frequencies": {},
            "continue_frequency": None,
            "summary": "exact combo 0% 到達，沒有 action EV 對照",
        },
        "off_tree": True,
        "confidence": "off_tree",
        "scope": "這個 exact combo 0% 到達此節點，不能判定實戰動作對錯",
    }


def _decision_verdict(decision: dict) -> str:
    """Render one compact, deterministic verdict for the narrator contract."""
    def scoped(verdict: str) -> str:
        if decision.get("confidence") == "medium":
            return verdict + "（這個 combo 只少量到達此節點）"
        return verdict

    actual = decision.get("actual_action")
    if decision.get("off_tree"):
        label = (actual or {}).get("label") or "實戰動作"
        return (
            f"{label} 屬 off-tree：這個 combo 0% 到達此節點，"
            "沒有 solver 對照，無法判定對錯"
        )
    preferred = decision["preferred_action"]
    loss = decision.get("ev_loss_pot") or 0.0
    if not actual:
        return scoped(f"solver 最偏好 {preferred['label']}")
    if loss >= 0.003:
        return scoped(f"{actual['label']} 有實質 EV 損失，應偏向 {preferred['label']}")
    if actual.get("code") == preferred.get("code"):
        purity = "幾乎純用" if preferred.get("frequency", 0.0) >= 0.97 else "最常用"
        return scoped(f"{actual['label']} 正確，solver {purity}這個動作")
    if actual.get("frequency", 0.0) >= 0.01:
        return scoped(
            f"{actual['label']} 是 solver 保留的 mix，"
            f"但主要動作是 {preferred['label']}"
        )
    return scoped(
        f"{actual['label']} 的 EV 影響很小，但不在可採信的 solver mix，"
        f"應偏向 {preferred['label']}"
    )


def _decision_brief_reason(decision: dict) -> str:
    """One grounded reason for the sequential street-by-street review."""
    if decision.get("off_tree"):
        return "這個 exact combo 在 solver 中 0% 到達，缺少可比較的 action EV"
    if decision.get("street") == "preflop":
        preferred = decision.get("preferred_action") or {}
        purity = "幾乎純用" if preferred.get("frequency", 0.0) >= 0.97 else "以此動作為主"
        return f"這個 hand class 在目前 preflop node {purity} {preferred.get('label', '該動作')}"
    if decision.get("draw_aggression"):
        story = decision["draw_aggression"]
        return (
            f"目前仍是未成牌，equity 來自{'與'.join(story['draw_labels'])}，"
            "solver 把它分配到 raise/all-in 而非被動 call"
        )
    if decision.get("showdown_value"):
        story = decision["showdown_value"]
        return (
            f"{story['made_hand_label']}仍領先對手下注 range 中的未成牌聽牌，"
            "配合價格不應 fold"
        )
    if decision.get("equity_denial"):
        return decision["equity_denial"]["interpretation"].split("。", 1)[0]
    if decision.get("check_story"):
        return decision["check_story"]["interpretation"].split("；", 1)[0]
    if decision.get("aggression_job"):
        return decision["aggression_job"]["interpretation"].split("；", 1)[0]
    if decision.get("mix_strategy"):
        return decision["mix_strategy"]["interpretation"].split("；", 1)[0]
    role = decision.get("hero_role") or {}
    plan = decision.get("range_plan") or {}
    if plan.get("strength") in {"very_strong", "strong"}:
        return (
            f"{plan.get('text')}，而這手是{role.get('range_band')}的"
            f"{role.get('made_hand_label')}、{role.get('draw_summary', role.get('draw_label'))}"
        )
    contract = decision.get("action_contract") or {}
    if contract.get("summary"):
        return contract["summary"]
    return "exact combo 的 solver action 與目前牌力角色一致"


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
        decision["coverage_reason"] = _decision_brief_reason(decision)


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
            f"{hero} 的 range equity 落後，而 solver 同時採近乎全 range check；"
            "這只描述整體策略方向，不決定單一 combo 的動作"
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


def _verified_live_draws(hand_truth: dict | None) -> list[str]:
    """Return every poker-rules draw that can improve on the next card."""
    return [
        draw for draw in (hand_truth or {}).get("draws") or []
        if draw in _LIVE_DRAW_CATEGORIES
    ]


def _draw_aggression_story(made_category: str | None,
                           hand_truth: dict | None,
                           combo_equity: float | None,
                           preferred: dict,
                           action_contract: dict) -> dict | None:
    """Explain where an unmade aggressive draw gets its equity.

    The current node proves the exact combo's raw equity and action
    allocation.  It does not expose Villain's response to the raise, so fold
    pressure may be described qualitatively but never assigned a percentage.
    """
    live_draws = _verified_live_draws(hand_truth)
    has_multiple_sources = (
        len(live_draws) >= 2 or "combo_draw" in live_draws
    )
    frequencies = action_contract.get("frequencies") or {}
    aggressive_frequency = sum(
        frequency for code, frequency in frequencies.items()
        if (code or "").startswith("R")
    )
    call_frequency = frequencies.get("C", 0.0)
    if (
        made_category not in _UNPAIRED_CATEGORIES
        or not has_multiple_sources
        or combo_equity is None
        or not (preferred.get("code") or "").startswith("R")
        or aggressive_frequency < 0.90
        or call_frequency > 0.05
    ):
        return None
    draw_labels = [_DRAW_ZH.get(draw, draw) for draw in live_draws]
    return {
        "live_draws": live_draws,
        "draw_labels": draw_labels,
        "draw_summary": "＋".join(draw_labels),
        "combo_equity": combo_equity,
        "aggressive_frequency": aggressive_frequency,
        "call_frequency": call_frequency,
        "preferred_label": preferred["label"],
        "interpretation": (
            f"Hero 仍是未成牌，改善 equity 來自{'與'.join(draw_labels)}。"
            f"Solver 幾乎不 call，而把它分到 raise/all-in、以 {preferred['label']} 為主；"
            "這是有改善 equity 的半詐唬。聽牌讓被跟注後仍能改善，加注則施加"
            "棄牌壓力，但目前資料不能量化 fold equity"
        ),
    }


def _villain_unpaired_story(solution: dict, villain_pi: dict) -> dict | None:
    """Join Villain's made/draw category arrays at the current solved node."""
    made_indices = solution.get("hand_categories_range") or []
    draw_indices = solution.get("draw_categories_range") or []
    villain_range = villain_pi.get("range") or []
    made_names = {
        row.get("index"): _normalize_category(row.get("name"))
        for row in villain_pi.get("hand_categories") or []
    }
    draw_names = {
        row.get("index"): row.get("name")
        for row in villain_pi.get("draw_categories") or []
    }
    total = sum(_float(weight) for weight in villain_range)
    if total <= 0:
        return None
    draw_mass = 0.0
    air_mass = 0.0
    live_draws: set[str] = set()
    for index, raw_weight in enumerate(villain_range):
        weight = _float(raw_weight)
        if (
            weight <= 0 or index >= len(made_indices)
            or index >= len(draw_indices)
        ):
            continue
        made = made_names.get(made_indices[index])
        draw = draw_names.get(draw_indices[index])
        if made not in _UNPAIRED_CATEGORIES:
            continue
        if draw in _LIVE_DRAW_CATEGORIES:
            draw_mass += weight
            live_draws.add(draw)
        elif draw == "no_draw":
            air_mass += weight
    return {
        "unpaired_draw_share": draw_mass / total,
        "unpaired_air_share": air_mass / total,
        "live_draws": sorted(live_draws),
    }


def _showdown_value_story(solution: dict, villain: str, villain_pi: dict,
                           made_category: str | None,
                           defense_price: dict | None) -> dict | None:
    """Show when a made hand still beats a meaningful unpaired draw region."""
    if made_category not in _VULNERABLE_MADE_HANDS or not defense_price:
        return None
    unpaired = _villain_unpaired_story(solution, villain_pi)
    if not unpaired or unpaired["unpaired_draw_share"] < 0.10:
        return None
    draw_labels = [
        _DRAW_ZH.get(draw, draw) for draw in unpaired["live_draws"]
    ]
    made_label = _MADE_ZH.get(made_category, made_category)
    draw_text = "、".join(draw_labels)
    return {
        **unpaired,
        "villain": villain,
        "made_hand_label": made_label,
        "draw_labels": draw_labels,
        "interpretation": (
            f"Hero 的{made_label}目前仍領先 {villain} 下注 range 中的一批未成牌，"
            f"包括{draw_text}，所以仍有攤牌價值。配合下注價格，Hero 不需要領先"
            "整個下注 range 也能繼續；這只支持不 fold，不能單靠它區分 call 與 raise"
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


def _representative_classes_for_action(
    player_info: dict, code: str, *, limit: int = 8,
) -> list[dict]:
    rows = []
    for hand_class, counter in (player_info.get("simple_hand_counters") or {}).items():
        mass = _float((counter.get("actions_total_combos") or {}).get(code))
        if mass <= 0:
            continue
        rows.append({"hand_class": hand_class, "mass": mass})
    rows.sort(key=lambda row: -row["mass"])
    total = sum(row["mass"] for row in rows)
    for row in rows:
        row["share"] = row["mass"] / total if total else 0.0
    return rows[:limit]


def _action_range_profile(
    solution: dict, player_info: dict, code: str | None,
) -> dict | None:
    """Describe the solved range that takes one aggressive action.

    The shape is a statement about the action bucket at this exact node.  It
    never assumes that a size is polar merely because it is large.
    """
    if not code or not code.startswith("R"):
        return None
    action_solution = next(
        (
            row for row in (solution.get("action_solutions") or [])
            if (row.get("action") or {}).get("code") == code
        ),
        None,
    )
    if not action_solution:
        return None
    composition = _action_composition(player_info, code)
    if not composition:
        return None
    value_share = sum(composition.get(name, 0.0) for name in _ACTION_VALUE_CORE)
    middle_share = sum(composition.get(name, 0.0) for name in _ACTION_MIDDLE)
    weak_share = sum(composition.get(name, 0.0) for name in _ACTION_WEAK)
    accounted = value_share + middle_share + weak_share
    middle_share += max(0.0, 1.0 - accounted)
    ordered = sorted(composition.items(), key=lambda item: -item[1])
    value_categories = [
        _MADE_ZH.get(name, name)
        for name, share in ordered if name in _ACTION_VALUE_CORE and share >= 0.01
    ]
    middle_categories = [
        _MADE_ZH.get(name, name)
        for name, share in ordered if name in _ACTION_MIDDLE and share >= 0.01
    ]
    weak_categories = [
        _MADE_ZH.get(name, name)
        for name, share in ordered if name in _ACTION_WEAK and share >= 0.01
    ]
    if value_share >= 0.35 and weak_share >= 0.12 and middle_share <= 0.30:
        shape = "polar"
        shape_label = "明顯偏極化"
        value_threshold = (
            "、".join(value_categories[:4]) + " 為主要價值端"
            if value_categories else "兩對以上為主"
        )
        value_text = "、".join(value_categories[:4]) or "已驗證的強牌"
        weak_text = "、".join(weak_categories[:4]) or "已驗證的弱端"
        interpretation = (
            f"此 action range 的價值端主要由 {value_text} 構成，"
            f"弱端主要由 {weak_text} 構成，中段成牌較少；"
            "可描述成已驗證價值端加弱端詐唬候選，"
            "但不能簡化成 literal nuts-or-air"
        )
    elif middle_share >= 0.35:
        shape = "merged"
        shape_label = "merged／線性"
        if composition.get("top_pair", 0.0) >= 0.10:
            value_threshold = "頂對以上為主要價值端"
        elif composition.get("second_pair", 0.0) >= 0.10:
            value_threshold = "第二對以上也進入薄價值端"
        else:
            value_threshold = "價值端與中段牌力混合"
        interpretation = (
            "此 action range 含有大量一對類中段牌力，不是只由強牌與空氣組成"
        )
    else:
        shape = "mixed"
        shape_label = "混合結構"
        value_threshold = "沒有單一清楚門檻"
        interpretation = "此 action range 同時含強端、中段與弱端，沒有足夠證據貼純極化標籤"
    return {
        "action_code": code,
        "action_label": _action_label(
            action_solution,
            any(
                (row.get("action") or {}).get("code") in {"F", "C"}
                for row in (solution.get("action_solutions") or [])
            ),
        ),
        "shape": shape,
        "shape_label": shape_label,
        "value_threshold": value_threshold,
        "value_core_share": value_share,
        "middle_share": middle_share,
        "weak_share": weak_share,
        "main_categories": [
            {
                "category": name,
                "label": _MADE_ZH.get(name, name),
                "share": share,
            }
            for name, share in ordered[:6]
        ],
        "value_categories": value_categories,
        "middle_categories": middle_categories,
        "weak_categories": weak_categories,
        "representative_classes": _representative_classes_for_action(
            player_info, code,
        ),
        "interpretation": interpretation,
    }


def _range_strength_targets(
    solution: dict, player_info: dict, hero_hand: str,
) -> dict | None:
    """Locate Hero's current made hand against Villain's reachable range."""
    try:
        from hand_eval import showdown_rank_key

        board = (solution.get("game") or {}).get("board") or ""
        hero_key = showdown_rank_key(hero_hand, board)
    except (ImportError, TypeError, ValueError, IndexError):
        return None
    range_arr = player_info.get("range") or []
    if len(range_arr) != len(gf._COMBO_INDEX):
        return None
    board_cards = gf._get_board_cards(board)
    hero_cards = {hero_hand[:2], hero_hand[2:]}
    category_indices = solution.get("hand_categories_range") or []
    category_names = {
        row.get("index"): _normalize_category(row.get("name"))
        for row in (player_info.get("hand_categories") or [])
    }
    buckets = {"ahead_of": {}, "behind": {}, "ties": {}}
    classes = {"ahead_of": {}, "behind": {}, "ties": {}}
    totals = {key: 0.0 for key in buckets}
    for idx, (c1, c2) in enumerate(gf._COMBO_INDEX):
        weight = _float(range_arr[idx])
        if (
            weight <= 1e-9 or c1 in board_cards or c2 in board_cards
            or c1 in hero_cards or c2 in hero_cards
        ):
            continue
        try:
            villain_key = showdown_rank_key(c1 + c2, board)
        except ValueError:
            continue
        target = "ahead_of" if hero_key > villain_key else ("behind" if hero_key < villain_key else "ties")
        category = (
            category_names.get(category_indices[idx])
            if idx < len(category_indices) else None
        ) or "unknown"
        hand_class = gf._combo_to_hand_name(c1, c2)
        buckets[target][category] = buckets[target].get(category, 0.0) + weight
        classes[target][hand_class] = classes[target].get(hand_class, 0.0) + weight
        totals[target] += weight
    total = sum(totals.values())
    if total <= 0:
        return None

    def summarize(target: str) -> dict:
        return {
            "share": totals[target] / total,
            "categories": [
                {"category": name, "label": _MADE_ZH.get(name, name), "mass": mass}
                for name, mass in sorted(
                    buckets[target].items(), key=lambda item: -item[1],
                )[:4]
            ],
            "classes": [
                {"hand_class": name, "mass": mass}
                for name, mass in sorted(
                    classes[target].items(), key=lambda item: -item[1],
                )[:5]
            ],
        }

    return {target: summarize(target) for target in buckets}


def _response_action_family(code: str) -> str:
    if code == "F":
        return "fold"
    if code in {"C", "X"}:
        return "continue"
    if code.startswith("R"):
        return "raise"
    return "other"


def _opponent_response_profile(
    response_solution: dict, hero_hand: str,
) -> dict | None:
    """Classify the opponent's solved response into value/bluff/protection targets."""
    actor = (response_solution.get("game") or {}).get("active_position")
    actor_pi = _players(response_solution).get(actor)
    board = (response_solution.get("game") or {}).get("board") or ""
    if not actor_pi:
        return None
    try:
        from hand_eval import showdown_rank_key

        hero_key = showdown_rank_key(hero_hand, board)
    except (ImportError, TypeError, ValueError, IndexError):
        return None
    range_arr = actor_pi.get("range") or []
    if len(range_arr) != len(gf._COMBO_INDEX):
        return None
    category_indices = response_solution.get("hand_categories_range") or []
    category_names = {
        row.get("index"): _normalize_category(row.get("name"))
        for row in (actor_pi.get("hand_categories") or [])
    }
    draw_indices = response_solution.get("draw_categories_range") or []
    draw_names = {
        row.get("index"): row.get("name")
        for row in (actor_pi.get("draw_categories") or [])
    }
    action_rows = [
        row for row in (response_solution.get("action_solutions") or [])
        if (row.get("action") or {}).get("code")
    ]
    if not action_rows:
        return None
    overall = {"fold": 0.0, "continue": 0.0, "raise": 0.0}
    for row in action_rows:
        family = _response_action_family((row.get("action") or {}).get("code") or "")
        if family in overall:
            overall[family] += _float(row.get("total_frequency"))

    target_names = (
        "continues_worse", "folds_better", "continues_better",
        "folds_worse_with_equity", "folds_worse", "ties",
    )
    targets = {
        name: {"mass": 0.0, "categories": {}, "classes": {}, "draws": {}}
        for name in target_names
    }
    indifferent = {
        "mass": 0.0, "categories": {}, "classes": {}, "action_families": {},
    }
    board_cards = gf._get_board_cards(board)
    hero_cards = {hero_hand[:2], hero_hand[2:]}
    for idx, (c1, c2) in enumerate(gf._COMBO_INDEX):
        weight = _float(range_arr[idx])
        if (
            weight <= 1e-9 or c1 in board_cards or c2 in board_cards
            or c1 in hero_cards or c2 in hero_cards
        ):
            continue
        try:
            villain_key = showdown_rank_key(c1 + c2, board)
        except ValueError:
            continue
        comparison = "worse" if villain_key < hero_key else ("better" if villain_key > hero_key else "tie")
        category = (
            category_names.get(category_indices[idx])
            if idx < len(category_indices) else None
        ) or "unknown"
        draw = draw_names.get(draw_indices[idx]) if idx < len(draw_indices) else None
        hand_class = gf._combo_to_hand_name(c1, c2)
        family_frequencies: dict[str, float] = {}
        for action_row in action_rows:
            code = (action_row.get("action") or {}).get("code") or ""
            strategies = action_row.get("strategy") or []
            frequency = _float(strategies[idx]) if idx < len(strategies) else 0.0
            family = _response_action_family(code)
            if family in {"fold", "continue", "raise"}:
                family_frequencies[family] = family_frequencies.get(family, 0.0) + frequency
        meaningful_families = [
            frequency for frequency in family_frequencies.values()
            if frequency >= 0.10
        ]
        if len(meaningful_families) >= 2 and max(meaningful_families) <= 0.90:
            indifferent["mass"] += weight
            indifferent["categories"][category] = (
                indifferent["categories"].get(category, 0.0) + weight
            )
            indifferent["classes"][hand_class] = (
                indifferent["classes"].get(hand_class, 0.0) + weight
            )
            for family, frequency in family_frequencies.items():
                indifferent["action_families"][family] = (
                    indifferent["action_families"].get(family, 0.0)
                    + weight * frequency
                )
        for action_row in action_rows:
            code = (action_row.get("action") or {}).get("code") or ""
            strategies = action_row.get("strategy") or []
            frequency = _float(strategies[idx]) if idx < len(strategies) else 0.0
            mass = weight * frequency
            if mass <= 1e-9:
                continue
            family = _response_action_family(code)
            if comparison == "tie":
                target = "ties"
            elif family == "fold" and comparison == "better":
                target = "folds_better"
            elif (
                family == "fold" and comparison == "worse" and len(board) < 10
                and (
                    (draw and draw != "no_draw")
                    or category in _UNPAIRED_CATEGORIES
                )
            ):
                target = "folds_worse_with_equity"
            elif family == "fold":
                target = "folds_worse"
            elif comparison == "worse":
                target = "continues_worse"
            else:
                target = "continues_better"
            bucket = targets[target]
            bucket["mass"] += mass
            bucket["categories"][category] = bucket["categories"].get(category, 0.0) + mass
            bucket["classes"][hand_class] = bucket["classes"].get(hand_class, 0.0) + mass
            if draw and draw != "no_draw":
                bucket["draws"][draw] = bucket["draws"].get(draw, 0.0) + mass

    total_mass = sum(bucket["mass"] for bucket in targets.values())
    if total_mass <= 0:
        return None
    for bucket in targets.values():
        bucket["share"] = bucket["mass"] / total_mass
        bucket["categories"] = [
            {"category": name, "label": _MADE_ZH.get(name, name), "mass": mass}
            for name, mass in sorted(
                bucket["categories"].items(), key=lambda item: -item[1],
            )[:4]
        ]
        bucket["classes"] = [
            {"hand_class": name, "mass": mass}
            for name, mass in sorted(
                bucket["classes"].items(), key=lambda item: -item[1],
            )[:5]
        ]
        bucket["draws"] = [
            {"draw": name, "label": _DRAW_ZH.get(name, name), "mass": mass}
            for name, mass in sorted(
                bucket["draws"].items(), key=lambda item: -item[1],
            )[:3]
        ]
    indifferent["share"] = indifferent["mass"] / total_mass
    indifferent["categories"] = [
        {"category": name, "label": _MADE_ZH.get(name, name), "mass": mass}
        for name, mass in sorted(
            indifferent["categories"].items(), key=lambda item: -item[1],
        )[:4]
    ]
    indifferent["classes"] = [
        {"hand_class": name, "mass": mass}
        for name, mass in sorted(
            indifferent["classes"].items(), key=lambda item: -item[1],
        )[:5]
    ]
    if indifferent["mass"] > 0:
        indifferent["action_families"] = {
            family: mass / indifferent["mass"]
            for family, mass in indifferent["action_families"].items()
        }
    return {
        "actor": actor,
        "board": board,
        "overall": overall,
        "targets": targets,
        "indifferent": indifferent,
        "scope": (
            "只描述對手在這個 exact action 後的 solved response；"
            "價值、詐唬與 protection 目標均由目前已成牌比較和實際 fold/continue bucket 推得"
        ),
    }


def _aggressive_branch_action(
    actions: list[dict], actual: dict | None, preferred: dict,
) -> dict | None:
    """Choose the action whose job should be explained.

    Explain the real bet/raise when Hero took one.  When Hero checked, explain
    the solver's main aggressive alternative so the check can be contrasted
    against an actual value/bluff/protection branch rather than generic prose.
    """
    if actual and (actual.get("code") or "").startswith("R"):
        return actual
    if (preferred.get("code") or "").startswith("R"):
        return preferred
    aggressive = [row for row in actions if (row.get("code") or "").startswith("R")]
    return max(aggressive, key=lambda row: row.get("frequency", 0.0), default=None)


def _target_names(bucket: dict | None, *, limit: int = 3) -> list[str]:
    if not bucket:
        return []
    floor = _float(bucket.get("mass")) * 0.05
    classes = [
        row["hand_class"] for row in bucket.get("classes") or []
        if _float(row.get("mass")) >= floor
    ]
    if classes:
        return classes[:limit]
    return [row["label"] for row in (bucket.get("categories") or [])[:limit]]


def _target_phrase(bucket: dict | None, *, limit: int = 4) -> str:
    if not bucket:
        return ""
    floor = _float(bucket.get("mass")) * 0.05
    categories = [
        row["label"] for row in bucket.get("categories") or []
        if _float(row.get("mass")) >= floor
    ][:3]
    classes = [
        row["hand_class"] for row in bucket.get("classes") or []
        if _float(row.get("mass")) >= floor
    ][:limit]
    if categories and classes:
        return f"{'、'.join(categories)}（如 {'、'.join(classes)}）"
    return "、".join(categories or classes)


def _aggression_job_story(decision: dict) -> dict | None:
    profile = decision.get("opponent_response_profile") or {}
    targets = profile.get("targets") or {}
    if not profile:
        return None
    value_targets = _target_names(targets.get("continues_worse"))
    bluff_targets = _target_names(targets.get("folds_better"))
    protection_targets = _target_names(targets.get("folds_worse_with_equity"))
    better_continues = _target_names(targets.get("continues_better"))
    indifferent = profile.get("indifferent") or {}
    indifferent_targets = (
        _target_names(indifferent, limit=4)
        if _float(indifferent.get("share")) >= 0.03 else []
    )
    role = decision.get("hero_role") or {}
    made = role.get("made_hand")
    live_draws = role.get("live_draws") or []
    is_unmade_draw = made in _UNPAIRED_CATEGORIES and bool(live_draws)
    if is_unmade_draw:
        # A draw occasionally being called by an even weaker draw is not the
        # useful meaning of "value bet".  Its main job is fold pressure plus
        # retained improvement equity when stronger hands continue.
        value_targets = []
    jobs = []
    if value_targets:
        jobs.append("value")
    if bluff_targets:
        jobs.append("bluff")
    if protection_targets:
        jobs.append("protection")
    if is_unmade_draw:
        combo_job = "semi_bluff"
    elif len(jobs) >= 2 or (value_targets and live_draws):
        combo_job = "hybrid"
    elif jobs:
        combo_job = jobs[0]
    else:
        combo_job = "range_construction"
    pieces = []
    if value_targets:
        pieces.append(
            f"較差的{_target_phrase(targets.get('continues_worse'))}仍會繼續，"
            "形成 value 來源"
        )
    if bluff_targets:
        pieces.append(
            f"較好的{_target_phrase(targets.get('folds_better'))}會棄牌，"
            "形成詐唬收益"
        )
    if protection_targets:
        pieces.append(
            f"目前較差但仍有改善 equity 的"
            f"{_target_phrase(targets.get('folds_worse_with_equity'))}會棄牌，"
            "形成 protection／equity denial"
        )
    if better_continues:
        pieces.append(
            f"更強的{_target_phrase(targets.get('continues_better'))}仍會繼續，"
            "不能把此動作當純 value"
        )
    if is_unmade_draw:
        pieces.append(
            f"被跟注時仍靠{role.get('draw_summary')}保留改善 equity，"
            "所以這是半詐唬而不是薄 value"
        )
    if indifferent_targets:
        families = [
            {"fold": "fold", "continue": "call/continue", "raise": "raise/all-in"}.get(
                family, family,
            )
            for family, frequency in (indifferent.get("action_families") or {}).items()
            if frequency >= 0.10
        ]
        pieces.append(
            f"同時把{_target_phrase(indifferent)}推進"
            f"{'／'.join(families)}混合的 indifferent 邊界，"
            "讓這部分 range 面臨最接近無差異的困難決策"
        )
    return {
        "combo_job": combo_job,
        "is_alternative": not (
            decision.get("aggressive_branch_is_actual")
            or decision.get("aggressive_branch_is_preferred")
        ),
        "value_targets": value_targets,
        "bluff_targets": bluff_targets,
        "protection_targets": protection_targets,
        "better_continues": better_continues,
        "indifferent_targets": indifferent_targets,
        "interpretation": "；".join(pieces),
        "scope": profile.get("scope"),
    }


def _check_story(decision: dict) -> dict | None:
    actual = decision.get("actual_action") or {}
    # Explain checks only when the exact combo is genuinely in solver's check
    # mix.  A zero-frequency check with a pure recommended bet is a mistake to
    # correct, not an action that deserves a fabricated pot-control/free-card
    # rationale.
    if actual.get("code") != "X" or _float(actual.get("frequency")) < 0.01:
        return None
    strength = decision.get("relative_strength") or {}
    ahead = _target_names(strength.get("ahead_of"))
    behind = _target_names(strength.get("behind"))
    role = decision.get("hero_role") or {}
    node = decision.get("node_context") or {}
    branch_job = decision.get("aggression_job") or {}
    reasons = []
    if ahead or behind:
        relation = []
        if ahead:
            relation.append(f"目前領先{'、'.join(ahead)}")
        if behind:
            relation.append(f"落後{'、'.join(behind)}")
        reasons.append(
            f"Hero 位於{role.get('range_band')}，" + "、但".join(relation)
        )
    if role.get("live_draws") and node.get("hero_relative_position") == "IP":
        reasons.append(
            f"IP 過牌可免費保留{role.get('draw_summary')}到下一張牌的 realization"
        )
    response = decision.get("opponent_response_profile") or {}
    raise_frequency = (response.get("overall") or {}).get("raise", 0.0)
    if raise_frequency >= 0.10:
        reasons.append("避免把中段牌力立即放進會遭遇大量 raise／all-in 的分支")
    if branch_job.get("interpretation"):
        reasons.append(
            f"若改走 {decision['action_range_profile']['action_label']}，其工作是："
            f"{branch_job['interpretation']}"
        )
    if not reasons:
        return None
    check_comp = decision.get("check_range_composition") or {}
    protected = sum(
        check_comp.get(name, 0.0)
        for name in _ACTION_VALUE_CORE | {"overpair", "top_pair"}
    )
    return {
        "ahead_of": ahead,
        "behind": behind,
        "free_card": bool(
            role.get("live_draws") and node.get("hero_relative_position") == "IP"
        ),
        "alternative_action": (decision.get("action_range_profile") or {}).get("action_label"),
        "alternative_job": branch_job,
        "check_strong_share": protected,
        "interpretation": "；".join(reasons),
    }


def _response_params(decision: dict) -> tuple[str, dict] | None:
    branch = decision.get("aggressive_branch") or {}
    code = branch.get("code") or ""
    street = decision.get("street")
    key = {
        "flop": "flop_actions", "turn": "turn_actions", "river": "river_actions",
    }.get(street)
    params = dict(decision.get("_spot_params") or {})
    if not code.startswith("R") or not key or not params:
        return None
    params[key] = "-".join(filter(None, (params.get(key, ""), code)))
    return f"{street}:{code}", params


def _enrich_action_job(
    decision: dict, context: dict, *, response_loader=None,
) -> None:
    response_request = _response_params(decision)
    if not response_request:
        return
    cache_key, params = response_request
    injected = context.get("_coach_response_solutions") or {}
    response_solution = injected.get(cache_key)
    if response_solution is None and response_loader:
        try:
            response_solution = response_loader(params)
        except Exception as exc:  # evidence enrichment must fail closed
            logging.getLogger(__name__).warning(
                "Coach response-node enrichment failed for %s: %s", cache_key, exc,
            )
    if not response_solution:
        return
    decision["opponent_response_profile"] = _opponent_response_profile(
        response_solution, decision["hero_hand"],
    )
    decision["aggression_job"] = _aggression_job_story(decision)
    decision["check_story"] = _check_story(decision)
    mechanisms = select_causal_mechanisms(decision)
    decision["causal_mechanisms"] = mechanisms
    decision["drivers"] = {
        "primary": mechanisms[0]["title"],
        "secondary": mechanisms[1]["title"] if len(mechanisms) > 1 else None,
    }


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
    live_draws = _verified_live_draws(hand_truth)
    draw_summary = "＋".join(
        _DRAW_ZH.get(draw, draw) for draw in live_draws
    ) or _DRAW_ZH.get(draw_category, draw_category or "聽牌狀態未知")
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
    showdown_value = _showdown_value_story(
        solution, villain, villain_pi, made_category, defense_price,
    )
    draw_aggression = _draw_aggression_story(
        made_category, hand_truth, combo_equity, preferred, action_contract,
    )
    equity_denial = _equity_denial_story(
        solution, villain_pi, node_context, made_category, preferred,
    )
    size_choice = _size_choice_story(
        solution, hero_pi, hero_hand, made_category, actual, preferred,
    )
    mix_strategy = _mix_strategy_story(actions, actual, preferred, ev_loss)
    aggressive_branch = _aggressive_branch_action(actions, actual, preferred)
    aggressive_branch_is_actual = bool(
        actual and aggressive_branch
        and actual.get("code") == aggressive_branch.get("code")
    )
    aggressive_branch_is_preferred = bool(
        aggressive_branch
        and preferred.get("code") == aggressive_branch.get("code")
    )
    action_range_profile = _action_range_profile(
        solution, hero_pi, (aggressive_branch or {}).get("code"),
    )
    relative_strength = _range_strength_targets(
        solution, villain_pi, hero_hand,
    )
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
            "live_draws": live_draws,
            "draw_summary": draw_summary,
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
        "showdown_value": showdown_value,
        "draw_aggression": draw_aggression,
        "equity_denial": equity_denial,
        "range_evidence": evidence,
        "size_structure": size_structure,
        "size_choice": size_choice,
        "mix_strategy": mix_strategy,
        "aggressive_branch": aggressive_branch,
        "aggressive_branch_is_actual": aggressive_branch_is_actual,
        "aggressive_branch_is_preferred": aggressive_branch_is_preferred,
        "action_range_profile": action_range_profile,
        "relative_strength": relative_strength,
        "check_range_composition": _action_composition(hero_pi, "X"),
        "opponent_response_profile": None,
        "aggression_job": None,
        "check_story": None,
        "blocker": blocker,
        "opponent_card_effects": opponent_card_effects,
        "_spot_params": dict(spot.get("params") or {}),
        "confidence": "medium" if reach_weight < 0.005 else "high",
        "scope": (
            "這個 combo 因前街低頻線只少量到達此節點"
            if reach_weight < 0.005
            else "range 事實來自同一個已評分 solver node"
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
    if (
        decision.get("equity_denial") or decision.get("defense_price")
        or decision.get("showdown_value") or decision.get("draw_aggression")
    ):
        score += 1.5
    if (decision.get("range_equity") or {}).get("use") != "omit":
        score += 0.5
    if decision.get("action_range_profile"):
        score += 1.0
    return score


def build_teaching_digest(context: dict, *, response_loader=None) -> dict | None:
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
        if item is None and spot.get("street") != "preflop":
            item = _off_tree_decision(context, spot, solution, hero_hand)
        if item:
            all_decisions.append(item)
            if item["street"] != "preflop" and not item.get("off_tree"):
                postflop_decisions.append(item)
    if not all_decisions:
        return None

    deviations = [
        row for row in postflop_decisions
        if (row.get("ev_loss_pot") or 0.0) >= 0.003
    ]
    selected = sorted(
        deviations,
        key=lambda row: row.get("ev_loss_pot") or 0.0,
        reverse=True,
    )[:2]
    if not selected and postflop_decisions:
        # With no EV mistake, give the narrator two evidence-rich candidates.
        # It may teach either one or connect them into a cross-street strategy
        # story; the first solver card already handles exhaustive coverage.
        selected = sorted(
            postflop_decisions,
            key=_teaching_score,
            reverse=True,
        )[:2]
    selected.sort(key=lambda row: ("flop", "turn", "river").index(row["street"]))

    # Successor nodes are fetched only for the 1–2 selected teaching focuses.
    # Existing unit callers remain pure unless they inject cached nodes or a
    # loader; production analysis supplies a per-user authenticated loader.
    for row in selected:
        _enrich_action_job(row, context, response_loader=response_loader)
    _label_all_decisions(all_decisions)

    allowed_categories = set()
    allowed_percentages = set()
    allowed_bb = set()
    for row in all_decisions:
        role = row.get("hero_role") or {}
        if role.get("made_hand"):
            allowed_categories.add(role["made_hand"])
        all_live_draws = set(role.get("live_draws") or [])
        if role.get("draw") in {"flush_draw", "nut_flush_draw"} or all_live_draws.intersection(
            {"flush_draw", "nut_flush_draw", "combo_draw"}
        ):
            allowed_categories.add("flush_draw")
        if role.get("draw") in {"onecard_bdfd", "twocards_bdfd", "backdoor_flush_draw"}:
            allowed_categories.add("backdoor_flush")
        if role.get("draw") in {"gutshot", "oesd"} or all_live_draws.intersection(
            {"gutshot", "oesd", "combo_draw"}
        ):
            allowed_categories.add("straight_draw")
        for action_key in ("actual_action", "preferred_action", "best_action_by_ev"):
            action = row.get(action_key)
            if action:
                allowed_percentages.add(round(100 * action["frequency"], 1))
                if action["pot_ratio"] >= 0:
                    allowed_percentages.add(round(100 * action["pot_ratio"], 1))
        for action in row.get("available_actions") or []:
            if action.get("frequency", 0.0) >= 0.005 and action.get("pot_ratio", -1.0) >= 0:
                allowed_percentages.add(round(100 * action["pot_ratio"], 1))
        if row.get("ev_loss_bb") is not None:
            allowed_bb.add(round(row["ev_loss_bb"], 3))
        if row.get("ev_loss_pot") is not None:
            allowed_percentages.add(round(100 * row["ev_loss_pot"], 1))
    for row in selected:
        role = row["hero_role"]
        if role.get("made_hand"):
            allowed_categories.add(role["made_hand"])
        live_draws = set(role.get("live_draws") or [])
        if role.get("draw") in {"flush_draw", "nut_flush_draw"} or live_draws.intersection(
            {"flush_draw", "nut_flush_draw", "combo_draw"}
        ):
            allowed_categories.add("flush_draw")
        if role.get("draw") in {"onecard_bdfd", "twocards_bdfd"}:
            allowed_categories.add("backdoor_flush")
        if role.get("draw") in {"gutshot", "oesd"} or live_draws.intersection(
            {"gutshot", "oesd", "combo_draw"}
        ):
            allowed_categories.add("straight_draw")
        showdown = row.get("showdown_value") or {}
        for draw in showdown.get("live_draws") or []:
            if draw in {"flush_draw", "nut_flush_draw", "combo_draw"}:
                allowed_categories.add("flush_draw")
            if draw in {"gutshot", "oesd", "combo_draw"}:
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
        action_profile = row.get("action_range_profile") or {}
        for item in action_profile.get("main_categories") or []:
            allowed_categories.add(item["category"])
        response = row.get("opponent_response_profile") or {}
        for bucket in (response.get("targets") or {}).values():
            for item in bucket.get("categories") or []:
                allowed_categories.add(item["category"])
        for frequency in (response.get("overall") or {}).values():
            allowed_percentages.add(round(100 * frequency, 1))

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


def _range_first_overview(decision: dict) -> dict | None:
    """Put whole-range construction before the exact combo's assignment."""
    profile = decision.get("action_range_profile") or {}
    if not profile:
        return None
    hero = decision.get("hero") or "Hero"
    villain = decision.get("villain") or "Villain"
    structure = decision.get("range_structure") or {}
    top = structure.get("nut_region") or {}
    strong = structure.get("strong_region") or {}
    structure_parts = []
    if top.get("owner") == hero and strong.get("owner") == hero:
        structure_parts.append(
            f"{hero} 在 equity 頂端與強端的份量都比 {villain} 厚"
        )
    else:
        if top.get("owner"):
            structure_parts.append(f"{top['owner']} 的 equity 頂端份量較厚")
        if strong.get("owner"):
            structure_parts.append(f"{strong['owner']} 的強端份量較厚")
    category_story = _category_sentence(decision)
    if category_story:
        structure_parts.append(category_story)

    frequencies = (decision.get("range_plan") or {}).get("frequencies") or {}
    check = _float(frequencies.get("X"))
    aggression = sum(
        _float(frequency)
        for code, frequency in frequencies.items()
        if (code or "").startswith("R")
    )
    if check >= 0.05 and aggression >= 0.05:
        plan_text = "整體策略仍在 check 與進攻之間混合，不是全 range 開火"
    elif aggression >= 0.97:
        plan_text = "整體 range 幾乎全部進攻"
    else:
        plan_text = (decision.get("range_plan") or {}).get("text")

    value_categories = profile.get("value_categories") or []
    weak_categories = profile.get("weak_categories") or []
    if profile.get("shape") == "polar":
        construction = (
            f"{profile['action_label']} 是{profile['shape_label']}："
            f"價值端主要由 {'、'.join(value_categories[:4]) or '已驗證強牌'} 構成，"
            f"弱端則從 {'、'.join(weak_categories[:4]) or '已驗證弱牌'} 挑選詐唬候選"
        )
    elif profile.get("shape") == "merged":
        construction = (
            f"{profile['action_label']} 是{profile['shape_label']}，"
            f"中段主要包含{'、'.join((profile.get('middle_categories') or [])[:4]) or '一對類牌力'}"
        )
    else:
        construction = f"{profile['action_label']} 採{profile['shape_label']}"
    parts = structure_parts + [part for part in (plan_text, construction) if part]
    return {
        "hero": hero,
        "villain": villain,
        "value_categories": value_categories,
        "weak_categories": weak_categories,
        "interpretation": "先看整體 range：" + "；".join(parts),
        "scope": (
            "先描述雙方 range 的強端厚度與此 action bucket 的價值／弱端組成，"
            "再描述 exact combo；不得把頂端 equity proxy 稱為 literal nut advantage"
        ),
    }


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
    if decision.get("action_range_profile"):
        profile = decision["action_range_profile"]
        lines.append(
            f"  Action range：{profile['action_label']} 是 {profile['shape_label']}；"
            f"{profile['interpretation']}"
        )
    if decision.get("aggression_job"):
        lines.append(
            f"  Action job：{decision['aggression_job']['interpretation']}"
        )
    if decision.get("check_story"):
        lines.append(f"  Check job：{decision['check_story']['interpretation']}")

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
        "【全手背景事實｜供選焦點，不要逐點重述】",
    ]
    for decision in digest.get("all_decisions") or digest["decisions"]:
        lines.append(
            f"• {decision['coverage_label']}："
            f"{decision['coverage_verdict']}；簡短理由：{decision['coverage_reason']}。"
        )
    lines.extend([
        "• 這些是第一則 solver 卡片已呈現的背景；第二則訊息不必逐一提到，也不要改寫成另一張 action table。",
        "• 可以用其中一點交代整手總評，再把篇幅留給下方 1–2 個教學焦點。",
        "• 若選到 off-tree 節點，只能說缺少 exact-combo solver 對照、無法判定對錯；不得自行補策略理由。",
        "",
        "【教練候選焦點｜從中挑最有價值的 1–2 個自然展開】",
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
            f"• 主要機制：{decision['drivers']['primary']}。",
            f"• 已觀測 range plan：{decision['range_plan']['text']}。",
        ])
        range_first = _range_first_overview(decision)
        if range_first:
            lines.append(
                f"• Range-first overview：{range_first['interpretation']}；"
                f"{range_first['scope']}。"
            )
        lines.extend([
            f"• Hero 角色：{role['range_band']}的{role['made_hand_label']}，{role.get('draw_summary', role['draw_label'])}。",
            f"• Exact combo action：{decision['action_contract']['summary']}。",
        ])
        action_profile = decision.get("action_range_profile") or {}
        if action_profile:
            if decision.get("aggressive_branch_is_actual"):
                branch_kind = "實戰進攻分支"
            elif decision.get("aggressive_branch_is_preferred"):
                branch_kind = "solver 建議進攻分支"
            else:
                branch_kind = "solver 的進攻替代分支"
            main = "、".join(
                f"{row['label']} {_pct(row['share'])}%"
                for row in action_profile.get("main_categories") or []
            )
            classes = "、".join(
                row["hand_class"]
                for row in (action_profile.get("representative_classes") or [])[:6]
            )
            lines.append(
                f"• Action range morphology（{branch_kind} {action_profile['action_label']}）："
                f"{action_profile['shape_label']}；{action_profile['interpretation']}；"
                f"價值門檻={action_profile['value_threshold']}；"
                f"主要牌型={main or '資料不足'}；代表 hand classes={classes or '資料不足'}。"
            )
        response = decision.get("opponent_response_profile") or {}
        if response:
            overall = response.get("overall") or {}
            lines.append(
                "• Opponent solved response："
                f"fold {_pct(overall.get('fold', 0.0))}% / "
                f"continue {_pct(overall.get('continue', 0.0))}% / "
                f"raise/all-in {_pct(overall.get('raise', 0.0))}%；"
                f"{response.get('scope')}。"
            )
        if decision.get("aggression_job"):
            job = decision["aggression_job"]
            lines.append(
                f"• Exact combo action job（"
                f"{'替代分支；' if job.get('is_alternative') else ''}{job['combo_job']}）："
                f"{job['interpretation']}。"
            )
        if decision.get("check_story"):
            lines.append(
                f"• Check job：{decision['check_story']['interpretation']}。"
            )
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
        if not range_first and (nut_region.get("owner") or strong_region.get("owner")):
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
        if decision.get("showdown_value"):
            lines.append(f"• Showdown value：{decision['showdown_value']['interpretation']}。")
        if decision.get("draw_aggression"):
            lines.append(
                f"• Equity 來源／進攻分配：{decision['draw_aggression']['interpretation']}。"
            )
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
        selected_rule_ids = {
            row.get("id") for row in decision.get("causal_mechanisms") or []
        }
        if card_effects and "blocker_candidate_ranking" in selected_rule_ids:
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
    if digest.get("caveats"):
        lines.append("• 降級註記：" + "；".join(digest["caveats"]) + "。")
    lines.extend([
        "",
        "【輸出契約】",
        "第二則訊息一定要有內容，即使全手打對也要提供一個具體、牌局相關的策略觀察。",
        "先用一句話總評整手，再挑 1–2 個最有教學價值的焦點；格式與段落由你決定，不必使用固定標題。",
        "不要逐點重述第一則 solver 卡片，不必逐一提到背景事實中的每個決策，也不要逐街稱讚。",
        "有實質 EV 錯誤時優先解釋最昂貴或最早的根本偏差；沒有錯誤時，解釋最有意思的 mix、牌力角色、尺寸或跨街策略節奏。",
        "只能從教練候選焦點提供的 range、牌型、blocker 與因果材料展開；背景事實只用來維持整手總評正確。",
        "off-tree 的 D# 只能說明沒有 exact-combo solver 對照、無法判定對錯；不得猜測該動作的 EV 或策略理由。",
        "若只有 preflop 或沒有可深講的 postflop 機制，仍要自然說明該 hand class 的策略定位；不要輸出系統如何判定正誤。",
        "核心判定是硬契約：『沒有實質 EV 損失』的 solver mix 分支不得稱為錯誤；頻率較低只代表較少採用。",
        "若你選擇討論『這個 combo 只少量到達此節點』的焦點，必須保留低到達率 caveat。",
        "低到達率 caveat 到『這個 combo 只少量到達此節點』為止；正文不得再加 generic node 邊界收尾。",
        "若核心判定寫『EV 代價低於實質門檻，但不在可採信的 solver mix』，只能說 EV 影響很小；不可稱為 solver 保留、可用或低頻 mix。",
        "不要逐項重述骨架，不要展示 percentile、removal score 或完整頻率表；全文最多引用 3 個真正有教學價值的數字。",
        "數字配額：每個焦點最多 1 個，優先保留 EV loss 或 preferred frequency；不要同時寫 bb 與 % pot，也不要自行估算 SPR。",
        "每個焦點只選主要機制與最多一個次要機制，使用『同花／set／順子／兩對／頂對／未成牌』等人能理解的 range 詞彙。",
        "每個 postflop 焦點在總評後都先講雙方整體 range：誰的 equity 頂端／強端較厚、整體 check／進攻計畫，以及所選 action bucket 的價值端與弱端組成；之後才講 exact combo 在其中負責什麼。",
        "若 exact combo 對某個 bet／raise 是近乎純進攻（至少 97%），而實戰 check 低於 1%，只能解釋 solver 建議的進攻及 check 犧牲的收益；不得替 0% 過牌補理由（包括 pot control、免費看牌或 equity realization）。",
        "每個進攻焦點先講 action range 是 merged、polar 或 mixed，再講 exact combo 在其中負責什麼；不可只說它是低頻／高頻 mix。",
        "下注或加注的理由必須依 Opponent solved response 分開：較差牌繼續=value、較好牌棄掉=bluff、目前較差但有改善 equity 的牌棄掉=protection／equity denial；同時命中多項時稱 hybrid。",
        "不能只交代 action range 形狀：Exact combo action job 中 value／bluff／protection 哪些欄位非空，就至少各點名一個骨架提供的主要 target；check 的替代進攻分支也遵守此規則。",
        "若 Exact combo action job 列出 indifferent 邊界，可說該 size 把哪些牌推入 fold／call／raise 的混合困難決策；沒有 combo-level 顯著混合時不得自行使用 indifferent。",
        "若 Action range morphology 寫明偏極化，必須沿用骨架列出的實際 value_categories 與 weak_categories 說明價值端／詐唬候選組成；不得固定套『兩對以上』，也不得縮寫成 literal nuts-or-air。",
        "只有骨架存在 Check job 時，過牌焦點才可交代 Hero 在自身 range 的位置、目前領先／落後的主要範圍，以及保留 equity realization 或避開反擊的作用；再用替代進攻分支說明沒有選 bet/raise 犧牲了什麼。",
        "Actor lock 是硬契約：不得把 Hero/Villain、opener/caller/3-bettor 或 IP/OOP 對調。",
        "『100% 繼續』不等於『100% call』；只能沿用 Exact combo action 的 action bucket。",
        "Range equity 只有 gate 為 supports_plan 或 prevents_bad_inference 時才能提；gate=omit 時完全省略。",
        "只有骨架明示 Equity denial 或 Exact combo action job 的 protection target 時，才能談拒絕對手 equity；低 SPR／脆弱成牌仍只限 Equity denial 欄位。",
        "Pot odds／raw equity 只能支持『繼續而非 fold』，不能單獨用來選 call、raise 或 all-in。",
        "只有骨架明示 Equity 來源／進攻分配，或 Exact combo action job=semi_bluff 時，才能把未成牌稱為半詐唬；必須同時說明已驗證聽牌是被跟注後的改善 equity 來源。",
        "只有骨架明示 Exact-class sizing 時，才能說同 hand class 被分配到某個 size；這不等於整體 range 更極化。",
        "Blocker 只能沿用骨架給的方向；不可自行聲稱 Hero 阻擋某個具體 combo、順子、同花或 nuts。",
        "Opponent-card conditional delta 是『Villain 持該牌時 Hero 策略如何變』；不得倒轉成 Hero 持牌造成的 blocker 故事。",
        "不得把沒有 live draw 的脆弱成牌稱為半詐唬；沒有 Equity 來源／進攻分配或 action job=semi_bluff 事實時也不得自行使用半詐唬。不要加入乾濕、連接性、驚悚牌等未驗證的 board texture。",
        "強牌類別差距只能說『誰的同花／set／兩對等更多』；不可擴寫成整體 range 或牌面必然有利／不利。",
        "不可列舉骨架沒有提供的具體手牌；不可把相關性寫成唯一因果。",
        "只有 Opponent solved response 提供的 hand classes／牌型與 action bucket 可以拿來說對手會 call/fold/raise；不得擴寫未列出的牌或誘導故事。",
        "骨架只證明目前這一個 action；不得把 check 擴寫成 check-fold、check-call 或 check-raise。",
        "正文以 2–5 個短段落為原則；不要硬塞通用 heuristic，也不要固定用 exact-node 邊界收尾。",
        "若上層要求 FOLLOWUP，每題必須另起一行並以 `FOLLOWUP:` 開頭；不得用普通 bullet 或編號代替。",
    ])
    return "\n".join(lines)


def render_fallback(digest: dict) -> str:
    """Selective natural-language safety net used when both LLM drafts fail."""
    coverage = digest.get("all_decisions") or digest["decisions"]
    focus = digest["decisions"]
    primary = focus[0] if focus else coverage[0]
    reasons = []
    for decision in focus:
        role = decision["hero_role"]
        pieces = []
        range_first = _range_first_overview(decision)
        if range_first:
            pieces.append(range_first["interpretation"])
        if decision.get("check_story"):
            check = decision["check_story"]
            relation = []
            if check.get("ahead_of"):
                relation.append("領先 " + "、".join(check["ahead_of"][:2]))
            if check.get("behind"):
                relation.append("落後 " + "、".join(check["behind"][:2]))
            check_reason = (
                f"{decision['street'].capitalize()} 過牌的理由是：{decision['hero_hand']} 位於"
                f" {role['range_band']}，{'、'.join(relation)}"
            )
            if check.get("free_card"):
                check_reason += (
                    f"；IP 可以免費保留{role.get('draw_summary')}的 realization"
                )
            pieces.append(check_reason)
            job = decision.get("aggression_job") or {}
            profile = decision.get("action_range_profile") or {}
            targets = []
            if job.get("value_targets"):
                targets.append("向 " + "、".join(job["value_targets"][:2]) + " 取 value")
            if job.get("bluff_targets"):
                targets.append("逼 " + "、".join(job["bluff_targets"][:2]) + " 棄牌")
            if job.get("protection_targets"):
                targets.append("拒絕 " + "、".join(job["protection_targets"][:2]) + " 的 equity")
            pieces.append(
                f"相對地，{profile.get('action_label')} 是{profile.get('shape_label')}，"
                + "、".join(targets)
            )
        elif decision.get("aggression_job"):
            profile = decision.get("action_range_profile") or {}
            job = decision["aggression_job"]
            targets = []
            if job.get("value_targets"):
                targets.append("向 " + "、".join(job["value_targets"][:2]) + " 取 value")
            if job.get("bluff_targets"):
                targets.append("逼 " + "、".join(job["bluff_targets"][:3]) + " 棄牌")
            if job.get("protection_targets"):
                targets.append("拒絕 " + "、".join(job["protection_targets"][:2]) + " 的 equity")
            job_label = {
                "value": "價值下注",
                "bluff": "詐唬",
                "protection": "保護下注",
                "semi_bluff": "半詐唬",
                "hybrid": "複合任務",
            }.get(job.get("combo_job"), "進攻候選")
            pieces.append(
                f"再看 exact combo：{decision['hero_hand']} 是 {role['range_band']}的"
                f"{role['made_hand_label']}，在 {profile.get('action_label', '這個進攻動作')}"
                f" 中負責{job_label}；" + "、".join(targets)
            )
            if job.get("combo_job") == "semi_bluff":
                pieces.append(
                    f"{decision['hero_hand']} 被跟注後仍靠{role.get('draw_summary')}改善，屬半詐唬"
                )
            if job.get("indifferent_targets"):
                pieces.append(
                    "同時把 " + "、".join(job["indifferent_targets"][:2])
                    + " 推入 indifferent 決策"
                )
        elif decision.get("draw_aggression"):
            pieces.append(decision["draw_aggression"]["interpretation"])
        elif decision.get("equity_denial"):
            pieces.append(decision["equity_denial"]["interpretation"])
        elif decision.get("mix_strategy"):
            pieces.append(decision["mix_strategy"]["interpretation"])
        elif decision.get("showdown_value"):
            pieces.append(decision["showdown_value"]["interpretation"])
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
        if not pieces:
            pieces.append(
                f"{decision['street'].capitalize()} 時它是 {role['range_band']}的"
                f"{role['made_hand_label']}，{role.get('draw_summary', role['draw_label'])}"
            )
        category_story = _category_sentence(decision)
        if category_story and not (
            decision.get("check_story") or decision.get("aggression_job")
        ):
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
        if (
            blocker and blocker["direction"] != "neutral"
            and not (decision.get("check_story") or decision.get("aggression_job"))
        ):
            pieces.append(blocker["interpretation"])
        elif (
            not category_story
            and not any(decision.get(key) for key in (
                "check_story", "aggression_job", "draw_aggression",
                "equity_denial", "mix_strategy",
                "showdown_value", "defense_price", "size_choice",
            ))
        ):
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
        if any(row.get("off_tree") for row in coverage):
            reasons = [
                "沒有可深講的已評分 postflop 節點；off-tree 只表示缺少 "
                "exact-combo solver 對照，不能補一般牌理。"
            ]
        else:
            reasons = [
                "這手只有 preflop 決策；exact hand class 的 solver action 分配"
                "已足以判定，不需要另外補一般牌理。"
            ]

    off_tree = [row for row in coverage if row.get("off_tree")]
    if primary.get("off_tree"):
        summary = (
            f"{primary['coverage_label']} 的實戰動作屬 off-tree，"
            "沒有 exact-combo solver 對照，無法判定對錯。"
        )
    elif len(coverage) == 1 and coverage[0]["street"] == "preflop":
        summary = (
            f"這手只有 preflop 決策：{primary['coverage_verdict']}；"
            f"{primary['coverage_reason']}。"
        )
    elif (
        (primary.get("ev_loss_pot") or 0.0) >= 0.003
        or (
            primary.get("actual_action")
            and primary["actual_action"].get("frequency", 0.0) < 0.01
        )
    ):
        summary = (
            f"整手最需要修正的是 {primary['coverage_label']}："
            f"{primary['coverage_verdict']}；{primary['coverage_reason']}。"
        )
    else:
        focus_verdicts = "；".join(
            f"{row['coverage_label']} {row['coverage_verdict']}"
            for row in focus
        )
        summary = f"整手沒有實質 EV 損失；{focus_verdicts}。"
        if off_tree:
            first = off_tree[0]
            summary += (
                f" {first['coverage_label']} 的實戰動作屬 off-tree，"
                "沒有 exact-combo solver 對照，無法判定對錯。"
            )
    return summary + "\n\n" + "\n\n".join(reasons)


@dataclass
class AuditResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _covered_decisions(digest: dict) -> list[dict]:
    """Every decision that must receive a verdict, including preflop."""
    return digest.get("all_decisions") or digest.get("decisions") or []


def _body_without_followups(text: str) -> str:
    kept = []
    followup_block = False
    for line in (text or "").splitlines():
        stripped = line.strip()
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
            re.match(r"^(?:[•*+-]|\d+[.)、])\s*", stripped)
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
            candidates = [
                row for row in _covered_decisions(digest)
                if not row.get("off_tree")
            ]
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


def _audit_off_tree_verdicts(body: str, digest: dict) -> list[str]:
    """An unreachable exact combo must stay ungraded in freeform prose."""
    violations = []
    street_terms = {
        "preflop": r"(?:preflop|翻牌前)",
        "flop": r"(?:flop|翻牌)",
        "turn": r"(?:turn|轉牌)",
        "river": r"(?:river|河牌)",
    }
    verdict_terms = re.compile(
        r"(?:正確|錯誤|失誤|漏洞|打對|打錯|可接受|合理|"
        r"EV\s*(?:loss|損失)|solver\s*(?:保留|支持)|mix\s*分支)",
        re.I,
    )
    ungraded_terms = re.compile(
        r"off[- ]?tree|0\s*%[^。；\n]{0,12}到達|"
        r"(?:沒有|無)[^。；\n]{0,12}solver[^。；\n]{0,12}對照|"
        r"無法[^。；\n]{0,12}判定|不能[^。；\n]{0,12}判定",
        re.I,
    )
    for decision in _covered_decisions(digest):
        if not decision.get("off_tree"):
            continue
        actual = decision.get("actual_action") or {}
        family = _action_family(
            actual.get("code") or "", decision["range_plan"]["facing_bet"],
        )
        action_pattern = _ACTION_TEXT_PATTERNS.get(family)
        same_street_count = sum(
            row.get("street") == decision.get("street")
            for row in _covered_decisions(digest)
        )
        for sentence in re.split(r"[。！？\n]", body or ""):
            if not re.search(street_terms[decision["street"]], sentence, re.I):
                continue
            if (
                same_street_count > 1
                and action_pattern
                and not re.search(action_pattern, sentence, re.I)
            ):
                continue
            if verdict_terms.search(sentence) and not ungraded_terms.search(sentence):
                violations.append(f"off-tree decision graded {decision.get('street')}")
                break
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
            if decision.get("off_tree"):
                continue
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
    if (
        len(_covered_decisions(digest)) == 1
        and not _covered_decisions(digest)[0].get("off_tree")
    ):
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
        if decision.get("off_tree"):
            continue
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
    # The deterministic card already covers every decision.  The narrator is a
    # selective teaching layer, so omission is intentional; facts it chooses to
    # mention are still audited below.
    violations.extend(_audit_off_tree_verdicts(body, digest))
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

    supports_equity_denial = any(
        row.get("equity_denial")
        or (row.get("aggression_job") or {}).get("protection_targets")
        for row in digest["decisions"]
    )
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

    induced_claim = re.search(
        r"(?:誘導|引誘)[^。；\n]{0,20}(?:詐唬|bluff)|"
        r"(?:保護|平衡)[^。；\n]{0,20}(?:check|過牌)[^。；\n]{0,12}(?:range|範圍)|"
        r"(?:用來|為了|透過)[^。；\n]{0,16}(?:平衡|保護)[^。；\n]{0,16}(?:整體)?(?:策略|range|範圍)",
        body,
        re.I,
    )
    supports_check_story = any(row.get("check_story") for row in digest["decisions"])
    pure_aggressive_corrections = [
        row for row in digest["decisions"]
        if (row.get("actual_action") or {}).get("code") == "X"
        and _float((row.get("actual_action") or {}).get("frequency")) < 0.01
        and ((row.get("preferred_action") or {}).get("code") or "").startswith("R")
        and _float((row.get("preferred_action") or {}).get("frequency")) >= 0.97
    ]
    if pure_aggressive_corrections and re.search(
        r"(?:check|過牌)[^。；\n]{0,28}"
        r"(?:理由|控制底池|pot control|免費|保留[^。；\n]{0,10}(?:equity|勝率)|"
        r"realization|避免[^。；\n]{0,10}(?:反擊|加注))",
        body,
        re.I,
    ):
        violations.append("unsupported check rationale for pure aggression")
    for decision in pure_aggressive_corrections:
        profile = decision.get("action_range_profile") or {}
        if profile.get("shape") != "polar":
            continue
        value_terms = profile.get("value_categories") or []
        weak_terms = profile.get("weak_categories") or []
        has_value_side = bool(
            re.search(r"價值(?:端|範圍|下注)", body)
            and any(term in body for term in value_terms)
        )
        has_bluff_side = bool(
            re.search(r"詐唬|bluff", body, re.I)
            and any(term in body for term in weak_terms)
        )
        if not (has_value_side and has_bluff_side):
            violations.append("missing range-first value/bluff construction")
    unsupported_pot_control = bool(
        re.search(r"(?:控制底池|pot control)", body, re.I)
        and not supports_check_story
    )
    if induced_claim or unsupported_pot_control:
        violations.append("unsupported induced-action explanation")
    supports_response = any(
        row.get("opponent_response_profile") for row in digest["decisions"]
    )
    if not supports_response and re.search(
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
    supports_indifferent = any(
        (row.get("aggression_job") or {}).get("indifferent_targets")
        for row in digest["decisions"]
    )
    if not supports_indifferent and re.search(
        r"indifferent|無差異|困難決策", body, re.I,
    ):
        violations.append("unsupported indifferent-response claim")

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

    supports_draw_aggression = any(
        row.get("draw_aggression")
        or (row.get("aggression_job") or {}).get("combo_job") == "semi_bluff"
        for row in digest["decisions"]
    )
    if (
        re.search(r"(?:半詐唬|semi[- ]?bluff)", lowered, re.I)
        and not supports_draw_aggression
    ):
        violations.append("unsupported semi-bluff label")
    requires_draw_source = any(
        row.get("draw_aggression") for row in digest["decisions"]
    )
    if requires_draw_source and not re.search(
        r"(?:equity|勝率)[^。；\n]{0,16}(?:來源|來自|改善)|"
        r"(?:改善牌|補成)[^。；\n]{0,16}(?:同花|順子)|"
        r"(?:同花聽牌|花聽)[^。；\n]{0,24}(?:卡順|順子聽牌)",
        body,
        re.I,
    ):
        violations.append("missing draw-equity source")

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

    polar_streets = {
        row.get("street")
        for row in digest["decisions"]
        if (
            row.get("size_structure")
            and row.get("drivers", {}).get("secondary") == "不同 bet size 的 range construction"
        ) or (row.get("action_range_profile") or {}).get("shape") == "polar"
    }
    for sentence in re.split(r"[。！？\n]", body):
        if not re.search(r"極化|polarized|polarization", sentence, re.I):
            continue
        named = {
            street for alias, street in _STREET_ALIASES.items()
            if re.search(re.escape(alias), sentence, re.I)
        }
        if not named:
            mentioned_anywhere = _mentioned_streets(body)
            if len(mentioned_anywhere) == 1:
                named = mentioned_anywhere
        if not polar_streets or (named and not named.intersection(polar_streets)):
            violations.append("unsupported polarization claim")
    if re.search(
        r"價值(?:下注)?式?詐唬|\bvalue[- ]?bet(?:ting)?[- ]?bluff\b",
        lowered,
        re.I,
    ):
        violations.append("contradictory value-bluff label")
    for sentence in re.split(r"[。！？\n]", body):
        sentence_lower = sentence.lower()
        transition_sentence = re.sub(
            r"(?:改善|補成)[^。；\n]{0,8}(?:equity|勝率|牌|來源)|"
            r"(?:equity|勝率)[^。；\n]{0,8}(?:改善|來源)",
            "",
            sentence_lower,
            flags=re.I,
        )
        names_a_street = re.search(
            r"(?:flop|turn|river|翻牌|轉牌|河牌)", transition_sentence,
        )
        claims_transition = re.search(
            r"(?:幫助|增強|提升|改善|削弱|改變|help|improv|strengthen|weaken|shift)",
            transition_sentence,
        )
        names_range = "range" in transition_sentence or "範圍" in transition_sentence
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

    compact_length = len(re.sub(r"\s+", "", body))
    if compact_length < 20:
        violations.append("coaching response too short")
    grounded_terms = re.compile(
        r"(?:"
        r"check|bet|call|fold|raise|all[- ]?in|"
        r"過牌|下注|跟注|棄牌|加注|全下|"
        r"同花|順子|兩對|頂對|超對|set|三條|葫蘆|四條|"
        r"未成牌|高牌|小對|中對|底對|聽牌|blocker|阻斷|"
        r"SPR|底池賠率|防守門檻|realization|equity denial|"
        r"range[^。；\n]{0,16}(?:頂端|底端|偏上|偏下|equity[^。；\n]{0,8}(?:優勢|劣勢|領先|落後))"
        r")",
        re.I,
    )
    if compact_length >= 20 and not grounded_terms.search(body):
        violations.append("missing grounded teaching content")
    response_limit = min(900, 520 + max(0, len(_covered_decisions(digest)) - 2) * 80)
    if compact_length > response_limit:
        violations.append("response too long")
    medium_decisions = [
        row for row in _covered_decisions(digest)
        if row.get("confidence") == "medium"
    ]
    mentioned_streets = _mentioned_streets(body)
    discusses_medium = (
        len(_covered_decisions(digest)) == 1 and bool(medium_decisions)
    ) or any(row.get("street") in mentioned_streets for row in medium_decisions)
    if discusses_medium:
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
