"""Replay a parsed hand through GTOW `next_actions` to resolve raw bb sizes
into GTOW raise codes (R2.1, R1.9, R5.2, ...), for custom-spot URL assembly.

Key decisions:
  - MTTGeneral preflop trees are 8-max. Hands with fewer players are padded
    with extra leading folds so hero's physical position maps onto the 8-max
    seat order. (5-max BTN → 8-max BTN; 6-max CO → 8-max CO.)
  - Cash games use the raw players_at_table preflop tree (no padding).
  - Each action is resolved independently via a next_actions call up to that
    decision point, snapping raw bb to the closest R* code (absolute distance
    match; same heuristic as find_closest_action).
  - action_index in the deviations table is HERO-SCOPED (counts hero's Nth
    decision on the street), matching scripts/backfill_ev_loss.py
    `_walk_to_decision`. The resolver converts it to raw stream index when
    truncating the street's actions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_api import (
    get_next_actions,
    nearest_depth,
    nearest_cash_depth,
    find_closest_action,
)

POSITION_ORDERS: dict[int, list[str]] = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}

MTT_TREE_SIZE = 8  # MTTGeneral preflop tree
STREET_ORDER = ("preflop", "flop", "turn", "river")


def _is_cash(gametype: str) -> bool:
    return (gametype or "").startswith("Cash")


def _pad_preflop_to_mtt_tree(
    preflop_actions_raw: str,
    players_at_table: int,
    hero_position: str,
) -> tuple[str, str, list[str]]:
    """Pad a shorter-table preflop line to the 8-max MTT tree.

    Returns (padded_action_string_with_original_codes, hero_position_8max,
             ordered_positions_list_8max).
    """
    if players_at_table >= MTT_TREE_SIZE:
        return preflop_actions_raw, hero_position, POSITION_ORDERS[players_at_table][:MTT_TREE_SIZE]

    pad = MTT_TREE_SIZE - players_at_table
    prefix = "F-" * pad
    padded = prefix + (preflop_actions_raw or "")
    padded = padded.rstrip("-")
    return padded, hero_position, POSITION_ORDERS[MTT_TREE_SIZE]


def _resolve_one_raise(
    gametype: str,
    depth: float,
    preflop_actions: str,
    board: str,
    flop_actions: str,
    turn_actions: str,
    river_actions: str,
    target_size: float,
) -> str:
    """Call next_actions at the current node and snap target_size to R* code."""
    resp = get_next_actions(
        gametype=gametype, depth=depth, stacks="",
        preflop_actions=preflop_actions, board=board,
        flop_actions=flop_actions, turn_actions=turn_actions,
        river_actions=river_actions,
    )
    available = resp.get("next_actions", {}).get("available_actions", []) or []
    code = find_closest_action(available, target_size)

    if target_size > 0 and code == "X":
        raise ValueError(
            f"no raise options at this node (target={target_size}) — off-tree"
        )
    return code


def _resolve_preflop_codes(
    gametype: str,
    depth: float,
    raw_preflop: str,
    hero_pad: int,  # unused but kept for call-site clarity
) -> str:
    """Walk preflop action-by-action, replacing each R-size with the GTOW code."""
    if not raw_preflop:
        return ""

    tokens = raw_preflop.split("-")
    out_tokens: list[str] = []
    prefix_history = ""

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("R"):
            try:
                target = float(tok[1:])
            except ValueError:
                out_tokens.append(tok)
                prefix_history = "-".join(out_tokens)
                continue
            code = _resolve_one_raise(
                gametype=gametype, depth=depth,
                preflop_actions=prefix_history,
                board="", flop_actions="", turn_actions="", river_actions="",
                target_size=target,
            )
            out_tokens.append(code)
        else:
            out_tokens.append(tok)
        prefix_history = "-".join(out_tokens)

    return "-".join(out_tokens)


def _hero_action_to_raw_index(
    raw_actions: list[dict],
    hero_pos: str,
    hero_action_index: int,
) -> int:
    """Convert a hero-scoped action_index (0 = hero's 1st decision on street)
    to a raw stream index (how many actions on the street precede it).

    action_index semantic matches backfill_ev_loss.py _walk_to_decision:
    counts occurrences of hero_pos in raw_actions.

    Returns the RAW index of hero's (hero_action_index)-th occurrence.
    If hero has fewer actions on the street than requested, returns len(raw_actions)
    (meaning: emit all street actions, no hero action to stop before).
    """
    hero_count = 0
    for i, act in enumerate(raw_actions):
        if act.get("position") == hero_pos:
            if hero_count == hero_action_index:
                return i
            hero_count += 1
    return len(raw_actions)


def _resolve_street_codes(
    gametype: str,
    depth: float,
    preflop_actions: str,
    board_so_far: str,
    prior_streets: dict[str, str],
    street_key: str,  # "flop" | "turn" | "river"
    raw_actions: list[dict],
    stop_after_n: int,
) -> str:
    """Resolve actions for one postflop street, emitting only actions[0:stop_after_n]."""
    out_tokens: list[str] = []
    for i, act in enumerate(raw_actions):
        if i >= stop_after_n:
            break
        action = act.get("action", "")
        if action.startswith("R"):
            target = float(act.get("size") or action[1:] or 0)
            code = _resolve_one_raise(
                gametype=gametype, depth=depth,
                preflop_actions=preflop_actions,
                board=board_so_far,
                flop_actions=prior_streets.get("flop", "") if street_key != "flop" else "-".join(out_tokens),
                turn_actions=prior_streets.get("turn", "") if street_key != "turn" else "-".join(out_tokens),
                river_actions=prior_streets.get("river", "") if street_key != "river" else "-".join(out_tokens),
                target_size=target,
            )
            out_tokens.append(code)
        else:
            out_tokens.append(action)
    return "-".join(out_tokens)


def _identify_villain(
    hand_data: dict,
    hero_pos_8max: str,
    preflop_codes: str,
    street: str,
) -> str | None:
    """Identify the HU postflop opponent.

    Walks preflop actions in 8-max position order, returns the LAST non-hero
    position with a non-fold action. For postflop, also sanity-checks that
    streets[] has <=2 distinct actors.
    """
    positions = POSITION_ORDERS[MTT_TREE_SIZE]
    tokens = preflop_codes.split("-") if preflop_codes else []
    last_villain: str | None = None
    for pos, tok in zip(positions, tokens):
        if pos == hero_pos_8max:
            continue
        if tok in ("F", ""):
            continue
        last_villain = pos

    if street == "preflop":
        return last_villain

    for s in hand_data.get("streets", []) or []:
        actors = {a.get("position") for a in (s.get("actions") or [])}
        actors.discard(None)
        if not actors:
            # Street recorded but not played (e.g. hand ended earlier) — skip.
            continue
        if len(actors) > 2 or (last_villain and last_villain not in actors and hero_pos_8max not in actors):
            return None
    return last_villain


def resolve_actions_for_deviation(
    hand_data: dict[str, Any],
    street: str,
    action_index: int,
) -> dict[str, Any]:
    """Replay a hand and return GTOW-formatted action fields for a deviation.

    Args:
        hand_data: parsed hand (shape: hand_histories.hand_data).
        street: "preflop" | "flop" | "turn" | "river" — hero's decision street.
        action_index: HERO-scoped — hero's Nth decision on that street (0-based).

    Returns a dict with keys:
        preflop_actions, flop_actions, turn_actions, river_actions,
        hero_pos, villain_pos, history_spot, depth, gametype

    Raises on malformed hand_data. Caller catches to fall back to bucket URL.
    """
    if street not in STREET_ORDER:
        raise ValueError(f"street must be one of {STREET_ORDER}, got {street!r}")

    gametype = hand_data.get("gametype") or "MTTGeneral"
    effective_bb = float(hand_data.get("effective_bb") or 30.0)
    hero_pos_raw = hand_data.get("hero_position") or ""
    players = int(hand_data.get("players_at_table") or 8)
    raw_preflop = hand_data.get("preflop_actions") or ""

    depth = nearest_cash_depth(effective_bb) if _is_cash(gametype) else nearest_depth(effective_bb)

    if _is_cash(gametype):
        padded_preflop = raw_preflop
        hero_pos_8 = hero_pos_raw
    else:
        padded_preflop, hero_pos_8, _ = _pad_preflop_to_mtt_tree(
            raw_preflop, players, hero_pos_raw,
        )

    if street == "preflop":
        tokens = padded_preflop.split("-") if padded_preflop else []
        positions = POSITION_ORDERS[MTT_TREE_SIZE] if not _is_cash(gametype) else POSITION_ORDERS[players]
        hero_slot = positions.index(hero_pos_8)
        truncated = "-".join(tokens[:hero_slot + action_index]) if action_index else "-".join(tokens[:hero_slot])
        preflop_codes = _resolve_preflop_codes(gametype, depth, truncated, 0)
        flop_codes = turn_codes = river_codes = ""
    else:
        preflop_codes = _resolve_preflop_codes(gametype, depth, padded_preflop, 0)
        flop_codes = turn_codes = river_codes = ""
        streets = hand_data.get("streets") or []
        board_parts: list[str] = []
        prior: dict[str, str] = {}
        target_idx = STREET_ORDER.index(street)
        for i, s in enumerate(streets):
            s_name = STREET_ORDER[i + 1]  # streets[0]=flop, [1]=turn, [2]=river
            if i == 0:
                board_parts.append(s.get("board") or "")
            else:
                board_parts.append(s.get("card") or "")
            board_now = "".join(board_parts)
            actions = s.get("actions") or []
            if i + 1 < target_idx:
                resolved = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=len(actions),
                )
                prior[s_name] = resolved
                if s_name == "flop":
                    flop_codes = resolved
                elif s_name == "turn":
                    turn_codes = resolved
                elif s_name == "river":
                    river_codes = resolved
            elif i + 1 == target_idx:
                # Convert hero-scoped action_index to raw-stream index
                raw_stop = _hero_action_to_raw_index(actions, hero_pos_8, action_index)
                resolved = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=raw_stop,
                )
                if s_name == "flop":
                    flop_codes = resolved
                elif s_name == "turn":
                    turn_codes = resolved
                elif s_name == "river":
                    river_codes = resolved
                break

    def _count(s: str) -> int:
        return len([t for t in s.split("-") if t]) if s else 0

    history_spot = _count(preflop_codes) + _count(flop_codes) + _count(turn_codes) + _count(river_codes)

    villain_pos = _identify_villain(hand_data, hero_pos_8, preflop_codes, street)

    return {
        "preflop_actions": preflop_codes,
        "flop_actions":    flop_codes,
        "turn_actions":    turn_codes,
        "river_actions":   river_codes,
        "hero_pos":        hero_pos_8,
        "villain_pos":     villain_pos,
        "history_spot":    history_spot,
        "depth":           depth,
        "gametype":        gametype,
    }
