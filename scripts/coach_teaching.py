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
    "trips": "trips",
    "straight": "順子",
    "flush": "同花",
    "full_house": "葫蘆",
    "quads": "四條",
    "straight_flush": "同花順",
}

_DRAW_ZH = {
    "no_draw": "沒有聽牌",
    "onecard_bdfd": "單張後門同花潛力",
    "twocards_bdfd": "雙張後門同花潛力",
    "gutshot": "卡順聽牌",
    "oesd": "兩頭順子聽牌",
    "flush_draw": "同花聽牌",
    "nut_flush_draw": "nuts 同花聽牌",
    "combo_draw": "複合聽牌",
}

_STRONG_CATEGORIES = (
    "straight_flush", "quads", "full_house", "flush", "straight",
    "set", "trips", "two_pair", "overpair", "over_pair", "top_pair",
)
_POLAR_VALUE = {
    "straight_flush", "quads", "full_house", "flush", "straight", "set", "trips",
}
_POLAR_AIR = {"no_made_hand", "king_high", "ace_high"}

_CATEGORY_TERMS = {
    "flush": ("同花", "flush"),
    "straight": ("順子", "straight"),
    "set": ("set", "三條"),
    "trips": ("trips",),
    "two_pair": ("兩對", "two pair"),
    "top_pair": ("頂對", "top pair"),
    "overpair": ("超對", "overpair"),
    "full_house": ("葫蘆", "full house"),
    "quads": ("四條", "quads"),
    "flush_draw": ("同花聽牌", "梅花聽牌", "花聽", "flush draw", "後門同花"),
    "straight_draw": ("卡順", "兩頭順", "順子聽牌", "gutshot", "oesd"),
}


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: float) -> int:
    return int(round(100 * value))


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
    return {
        row.get("name"): _float(row.get("total_frequency"))
        for row in (player_info.get("hand_categories") or [])
        if row.get("name")
    }


def _category_name(solution: dict, player_info: dict, combo_idx: int) -> str | None:
    categories = solution.get("hand_categories_range") or []
    if combo_idx >= len(categories):
        return None
    names = {
        row.get("index"): row.get("name")
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
            masses[category.get("name")] = mass
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
        if not composition:
            continue
        polar_share = sum(composition.get(cat, 0.0) for cat in _POLAR_VALUE | _POLAR_AIR)
        bets.append({
            "code": code,
            "ratio": ratio,
            "frequency": _float(row.get("total_frequency")),
            "composition": composition,
            "polar_share": polar_share,
        })
    if len(bets) < 2:
        return None
    largest = max(bets, key=lambda row: row["ratio"])
    common_smaller = max(
        (row for row in bets if row is not largest),
        key=lambda row: row["frequency"],
        default=None,
    )
    if not common_smaller or largest["polar_share"] - common_smaller["polar_share"] < 0.10:
        return None
    large_top = sorted(largest["composition"].items(), key=lambda item: -item[1])[:3]
    return {
        "larger_size": f"{_pct(largest['ratio'])}% pot",
        "smaller_size": f"{_pct(common_smaller['ratio'])}% pot",
        "large_size_main_categories": [
            {"category": name, "label": _MADE_ZH.get(name, name), "share": share}
            for name, share in large_top
        ],
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


def _driver(decision: dict) -> tuple[str, str | None]:
    range_plan = decision["range_plan"]
    evidence = decision["range_evidence"]
    blocker = decision.get("blocker")
    size_structure = decision.get("size_structure")
    if range_plan["strength"] == "very_strong":
        primary = "整體 range strategy"
    elif range_plan["facing_bet"] and (1 - range_plan["frequencies"].get("F", 0.0)) >= 0.85:
        primary = "整體防守結構與 Hero 在自身 range 的位置"
    elif sum(abs(row["gap"]) for row in evidence) >= 0.10:
        primary = "雙方強牌結構"
    else:
        primary = "Hero 這個 combo 的 range 角色與 EV"
    if blocker and blocker["direction"] != "neutral":
        secondary = "blocker 只用來排序候選 combo"
    elif size_structure:
        secondary = "不同 bet size 的 range construction"
    elif evidence and primary != "雙方強牌結構":
        secondary = "雙方強牌結構"
    else:
        secondary = None
    return primary, secondary


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
    best = max(actions, key=lambda row: row["ev_bb"])
    actual_code = spot.get("taken_code")
    actual = next((row for row in actions if row["code"] == actual_code), None)
    preferred = max(actions, key=lambda row: row["frequency"])
    ev_loss = max(0.0, best["ev_bb"] - actual["ev_bb"]) if actual else 0.0
    pot = _float((solution.get("game") or {}).get("pot"))
    ev_loss_pot = ev_loss / pot if pot > 0 else None
    made_category = _category_name(solution, hero_pi, combo_idx)
    draw_category = _draw_name(solution, hero_pi, combo_idx)
    percentiles = hero_pi.get("eq_percentile") or []
    percentile = _float(percentiles[combo_idx], -1.0) if combo_idx < len(percentiles) else -1.0
    band = _range_band(percentile if percentile >= 0 else None)

    range_plan = _range_plan(solution)
    evidence = _range_evidence(hero, villain, hero_pi, villain_pi)
    size_structure = _size_structure(solution, hero_pi)
    aggressive_code = (
        actual.get("code") if actual and (actual.get("code") or "").startswith("R")
        else best.get("code")
    )
    blocker = _blocker_story(
        solution, hero_pi, combo_idx, hero_hand, made_category, aggressive_code, street,
    )
    decision = {
        "street": street,
        "board": (solution.get("game") or {}).get("board") or "",
        "hero": hero,
        "villain": villain,
        "hero_hand": hero_hand,
        "hero_role": {
            "range_band": band,
            "made_hand": made_category,
            "made_hand_label": _MADE_ZH.get(made_category, made_category or "未知牌型"),
            "draw": draw_category,
            "draw_label": _DRAW_ZH.get(draw_category, draw_category or "聽牌狀態未知"),
        },
        "actual_action": actual,
        "preferred_action": preferred,
        "best_action_by_ev": best,
        "ev_loss_bb": ev_loss,
        "ev_loss_pot": ev_loss_pot,
        "range_plan": range_plan,
        "range_evidence": evidence,
        "size_structure": size_structure,
        "blocker": blocker,
        "confidence": "medium" if reach_weight < 0.005 else "high",
        "scope": (
            "這個 combo 因前街低頻線只少量到達此節點；結論只適用於目前深度、牌面與 action line"
            if reach_weight < 0.005
            else "只適用於這個深度、牌面與 action line；range 事實來自同一個已評分 solver node"
        ),
    }
    primary, secondary = _driver(decision)
    decision["drivers"] = {"primary": primary, "secondary": secondary}
    return decision


def _teaching_score(decision: dict) -> float:
    score = min(5.0, (decision.get("ev_loss_pot") or 0.0) * 20)
    if decision["range_plan"]["strength"] in {"very_strong", "strong"}:
        score += 1.5
    score += min(2.0, sum(abs(row["gap"]) for row in decision["range_evidence"]) * 8)
    if decision.get("size_structure"):
        score += 1.0
    if decision.get("blocker") and decision["blocker"]["direction"] != "neutral":
        score += 1.0
    return score


def build_teaching_digest(context: dict) -> dict | None:
    """Build a small, solver-grounded digest for at most two postflop decisions."""
    if context.get("no_hero_hand"):
        return None
    hero_hand = _raw_hero_hand(context)
    if not hero_hand:
        return None
    decisions = []
    for spot, solution in zip(context.get("hero_spots") or [], context.get("solutions") or []):
        if not solution:
            continue
        item = _decision(context, spot, solution, hero_hand)
        if item:
            decisions.append(item)
    if not decisions:
        return None

    deviations = [row for row in decisions if (row.get("ev_loss_pot") or 0.0) >= 0.003]
    selected = []
    if deviations:
        selected.append(max(deviations, key=lambda row: row.get("ev_loss_pot") or 0.0))
    remaining = [row for row in decisions if row not in selected]
    if remaining:
        best_teaching = max(remaining, key=_teaching_score)
        if not selected or _teaching_score(best_teaching) >= 2.0:
            selected.append(best_teaching)
    if not selected:
        selected = [max(decisions, key=_teaching_score)]
    selected.sort(key=lambda row: ("flop", "turn", "river").index(row["street"]))

    allowed_categories = set()
    allowed_percentages = set()
    allowed_bb = set()
    for row in selected:
        role = row["hero_role"]
        if role.get("made_hand"):
            allowed_categories.add(role["made_hand"])
        if role.get("draw") in {"flush_draw", "nut_flush_draw"}:
            allowed_categories.add("flush_draw")
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
        allowed_percentages |= {
            round(100 * frequency, 1)
            for frequency in row["range_plan"]["frequencies"].values()
        }
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
        if caveats or any(row.get("confidence") == "medium" for row in selected)
        else "high"
    )
    return {
        "confidence": digest_confidence,
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
        parts.append(f"{owner} 的{'、'.join(labels)} 更多")
    return "；".join(parts)


def render_prompt_block(digest: dict | None) -> str:
    """Render the digest as a compact contract for the coaching LLM."""
    if not digest:
        return ""
    lines = [
        "【Deterministic 教學骨架｜唯一可用的因果材料】",
        f"資料信心：{digest['confidence']}；保真度：同一個已評分 solver node。",
    ]
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
            verdict = f"Hero 的 {actual['label']} 沒有實質 EV 損失"
        else:
            verdict = f"solver 最偏好 {preferred['label']}"
        lines.extend([
            f"焦點 {index}｜{decision['street'].capitalize()} {decision['board']}",
            f"• 核心判定：{verdict}；solver 最常用 {preferred['label']}（約 {_pct(preferred['frequency'])}%）。",
            f"• Hero 角色：{role['range_band']}的{role['made_hand_label']}，{role['draw_label']}。",
            f"• 主要機制：{decision['drivers']['primary']}。",
            f"• 已觀測 range plan：{decision['range_plan']['text']}。",
        ])
        category_story = _category_sentence(decision)
        if category_story:
            lines.append(f"• 已驗證強牌結構：{category_story}。")
        if (
            decision.get("size_structure")
            and decision["drivers"].get("secondary") == "不同 bet size 的 range construction"
        ):
            size = decision["size_structure"]
            labels = "、".join(row["label"] for row in size["large_size_main_categories"][:2])
            lines.append(
                f"• Size construction：{size['larger_size']} 比 {size['smaller_size']} 更極化；"
                f"較大 size 主要由 {labels} 構成。"
            )
        blocker = decision.get("blocker")
        if blocker:
            lines.append(
                f"• Blocker：{blocker['interpretation']}；同 hand class 花色敏感度 "
                f"{blocker['same_class_suit_sensitivity']}。"
            )
        if decision["drivers"].get("secondary"):
            lines.append(f"• 次要機制：{decision['drivers']['secondary']}。")
        lines.append(f"• 適用邊界：{decision['scope']}。")
    if digest.get("caveats"):
        lines.append("• 降級註記：" + "；".join(digest["caveats"]) + "。")
    lines.extend([
        "",
        "【輸出契約】",
        "只輸出三段：*核心判斷*、*為什麼*、*你要記得*；每段最多兩句，全文約 180–300 字。",
        "GTO street summary 會由系統另外顯示；禁止再加 Preflop／Flop／Turn／River 逐街複述。",
        "不要逐項重述骨架，不要展示 percentile、removal score 或完整頻率表；最多引用 3 個真正有教學價值的數字。",
        "*為什麼*只選主要機制與最多一個次要機制，使用『同花／set／順子／兩對／頂對／未成牌』等人能理解的 range 詞彙。",
        "Blocker 只能沿用骨架給的方向；不可自行聲稱 Hero 阻擋某個具體 combo、順子、同花或 nuts。",
        "不可列舉骨架沒有提供的具體手牌；不可把相關性寫成唯一因果。",
        "*你要記得*給一條可帶走的 heuristic，並保留 exact-node 邊界，避免泛化成所有相似牌面。",
    ])
    return "\n".join(lines)


def render_fallback(digest: dict) -> str:
    """Deterministic three-section answer used when the prose audit fails."""
    primary = digest["decisions"][0]

    def _core(decision: dict) -> str:
        actual = decision.get("actual_action")
        preferred = decision["preferred_action"]
        loss = decision.get("ev_loss_pot") or 0.0
        street = decision["street"].capitalize()
        if actual and loss >= 0.003:
            severity = "小漏洞" if loss < 0.03 else ("明顯失誤" if loss < 0.10 else "嚴重錯誤")
            return f"{street} 的 {actual['label']} 是{severity}，應偏向 {preferred['label']}"
        if actual:
            if actual.get("code") == preferred.get("code"):
                return f"{street} 的 {actual['label']} 是正確選擇"
            return f"{street} 的 {actual['label']} 沒有實質 EV 損失"
        return f"{street} 應偏向 {preferred['label']}"

    core_parts = [_core(decision) for decision in digest["decisions"]]
    reasons = []
    for decision in digest["decisions"]:
        role = decision["hero_role"]
        pieces = [
            f"{decision['street'].capitalize()} 時它是 {role['range_band']}的"
            f"{role['made_hand_label']}，{role['draw_label']}"
        ]
        category_story = _category_sentence(decision)
        if category_story:
            pieces.append(category_story)
        blocker = decision.get("blocker")
        if blocker and blocker["direction"] != "neutral":
            pieces.append(blocker["interpretation"])
        elif not category_story:
            pieces.append(decision["range_plan"]["text"])
        reasons.append("；".join(pieces) + "。")

    # The common two-street teaching shape (bad bluff candidate on one street,
    # range-created bluff capacity on the next) reads better as one causal
    # contrast than as two repeated blocker sentences.
    if len(digest["decisions"]) == 2:
        first, second = digest["decisions"]
        second_categories = _category_sentence(second)
        first_blocker = first.get("blocker")
        second_blocker = second.get("blocker")
        if first_blocker and second_categories:
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

    lesson = {
        "整體 range strategy": "當 solver 在同一節點採近乎全 range 的計畫時，先服從 range-level strategy，再談單一 combo 的微調。",
        "整體防守結構與 Hero 在自身 range 的位置": "range 劣勢不代表每個邊緣 combo 都要 fold；先看整體防守寬度，再看這手牌在自身 range 的位置。",
        "雙方強牌結構": "先找雙方誰擁有更多可辨認的強牌類別，再用 blocker 排序 value 或 bluff 候選，而不是反過來編理由。",
    }.get(
        primary["drivers"]["primary"],
        "先判斷這手牌在自身 range 的角色與 EV，再用 blocker 做次要排序。",
    )
    if len(digest["decisions"]) > 1 and _category_sentence(digest["decisions"][1]):
        lesson = "先看雙方強牌結構決定 range 能否容納 bluff，再用 combo 角色、EV 與 blocker 排序候選牌。"
    reach_note = ""
    if any(decision.get("confidence") == "medium" for decision in digest["decisions"]):
        reach_note = "River 是前街低頻線後的條件式結論。"
    return (
        f"*核心判斷*\n{'；'.join(core_parts)}。\n\n"
        f"*為什麼*\n{' '.join(reasons)}\n\n"
        f"*你要記得*\n{lesson}{reach_note}這條結論只適用目前的深度、牌面與 action line。"
    )


@dataclass
class AuditResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _body_without_followups(text: str) -> str:
    return "\n".join(
        line for line in (text or "").splitlines()
        if not line.strip().startswith("FOLLOWUP:")
    )


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

        facts = coach_facts.initial_verdict_facts(
            list(source_texts or []) + [render_prompt_block(digest)],
            {"hand": {"hero_hand": digest["decisions"][0]["hero_hand"]},
             "hero_hand": digest["decisions"][0]["hero_hand"]},
        )
        board = digest["decisions"][0].get("board") or ""
        verdict = coach_facts.verify_claims(body, facts, board)
        violations.extend(f"unsupported combo {token}" for token in verdict.violations)
    except Exception:
        # The category/blocker gates below remain active even if the legacy
        # whitelist helper is unavailable during a partial deployment.
        pass

    allowed = set(digest.get("allowed_categories") or [])
    for category, terms in _CATEGORY_TERMS.items():
        if category in allowed:
            continue
        if any(term.lower() in lowered for term in terms):
            violations.append(f"unsupported category {category}")

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
    if re.search(r"\b(?:nuts?|nut advantage)\b|堅果", lowered, re.I):
        violations.append("unsupported nuts claim")

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
    if re.search(r"乾燥|濕潤|動態牌面|靜態牌面|\bdry\b|\bwet\b|dynamic board|static board", lowered, re.I):
        violations.append("unsupported board-texture claim")

    allowed_positions = {
        position
        for row in digest["decisions"]
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
    if len(re.sub(r"\s+", "", body)) > 360:
        violations.append("response too long")
    if any(row.get("confidence") == "medium" for row in digest["decisions"]):
        if not re.search(r"低頻|少量到達|條件式|rare|low[- ]frequency", lowered, re.I):
            violations.append("missing low-reach caveat")

    numeric_claims = re.findall(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(%|％|bb)", body, re.I,
    )
    if len(numeric_claims) > 3:
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
