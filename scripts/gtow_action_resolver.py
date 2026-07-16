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
    decision on the street). The resolver converts it to raw stream index when
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
    if players_at_table == 9:
        # MTTGeneral exposes an 8-max tree.  A 9-max hand maps safely only when
        # the extra earliest seat (physical UTG) folded: remove that fold and
        # shift UTG+1/UTG+2 onto solver UTG/UTG+1.  Never erase a voluntary UTG
        # action — that would change the spot rather than approximate seating.
        tokens = [t for t in (preflop_actions_raw or "").split("-") if t]
        if not tokens or tokens[0] != "F":
            raise ValueError("9-max MTT review cannot drop a non-folding UTG seat")
        hero_8max = {"UTG+1": "UTG", "UTG+2": "UTG+1"}.get(
            hero_position, hero_position)
        return "-".join(tokens[1:]), hero_8max, POSITION_ORDERS[MTT_TREE_SIZE]
    if players_at_table >= MTT_TREE_SIZE:
        return preflop_actions_raw, hero_position, POSITION_ORDERS[MTT_TREE_SIZE]

    pad = MTT_TREE_SIZE - players_at_table
    prefix = "F-" * pad
    padded = prefix + (preflop_actions_raw or "")
    padded = padded.rstrip("-")
    return padded, hero_position, POSITION_ORDERS[MTT_TREE_SIZE]


def _replay_preflop_actors(tokens: list[str], positions: list[str]) -> list[str]:
    """Assign each preflop token to the position that took it.

    Replays standard preflop betting: action rotates clockwise; a raise
    reopens the action for every still-live player except the raiser, so
    tokens beyond round 1 (responses to a 3-bet/4-bet) map back to the right
    seats. Returns a list parallel to ``tokens`` of actor position strings.
    """
    n = len(positions)
    folded = [False] * n
    pending = set(range(n))  # seats that still owe an action before betting closes
    actors: list[str] = []
    seat = 0
    for tok in tokens:
        steps = 0
        while (folded[seat] or seat not in pending) and steps <= n:
            seat = (seat + 1) % n
            steps += 1
        actors.append(positions[seat])
        pending.discard(seat)
        t = (tok or "").strip()
        if t in ("F", ""):
            folded[seat] = True
        elif t.startswith("R") or t == "AI":
            pending = {s for s in range(n) if not folded[s] and s != seat}
        # C / X close nothing
        seat = (seat + 1) % n
    return actors


def _collapse_coldcall_folders(
    preflop_actions: str,
    positions: list[str],
    hero_pos: str,
) -> str:
    """Collapse non-hero cold-call-then-fold players into a single fold.

    A multiway pot where an extra player flat-calls preflop and then folds
    before the flop (e.g. CO cold-calls hero's open, the SB 3-bets, CO folds)
    is off-tree for the heads-up GTOW solution node the bot analysed. Such a
    player reaches neither the flop nor changes the eventual HU pot, so we
    approximate their first action as a fold and drop their later fold — the
    line stays on-tree and lands on the same HU flop (H3480).

    Preserved as-is:
      - Raisers (openers, 3-bettors): their bet defines the node.
      - Players still live on the flop (last preflop action is not a fold).
      - Hero.
    """
    tokens = [t for t in (preflop_actions or "").split("-") if t]
    if not tokens:
        return preflop_actions
    actors = _replay_preflop_actors(tokens, positions)

    by_actor: dict[str, list[int]] = {}
    for i, pos in enumerate(actors):
        by_actor.setdefault(pos, []).append(i)

    drop: set[int] = set()
    fold_in_place: set[int] = set()
    for pos, idxs in by_actor.items():
        if pos == hero_pos:
            continue
        toks = [tokens[i] for i in idxs]
        has_raise = any(t.startswith("R") or t == "AI" for t in toks)
        has_call = "C" in toks
        ends_folded = toks[-1] == "F"
        if has_raise or not has_call or not ends_folded:
            continue
        # cold-call-then-fold: keep one fold at the first action, drop the rest
        fold_in_place.add(idxs[0])
        drop.update(idxs[1:])

    if not drop and not fold_in_place:
        return preflop_actions
    out = [
        "F" if i in fold_in_place else tok
        for i, tok in enumerate(tokens)
        if i not in drop
    ]
    return "-".join(out)


def _resolve_one_raise(
    gametype: str,
    depth: float,
    preflop_actions: str,
    board: str,
    flop_actions: str,
    turn_actions: str,
    river_actions: str,
    target_size: float,
    actual_pot: float = 0.0,
) -> str:
    """Call next_actions at the current node and snap target_size to R* code.

    When ``actual_pot`` is given (>0), the opening bet of a postflop street is
    snapped by *pot ratio* rather than absolute bb — matching the analysis
    pipeline's ``_find_action_by_pot_pct``. This matters in multiway pots where
    the deep-link's preflop line drops dead money (folded cold-callers), so the
    solver's modeled pot is smaller than the real pot and a bet that was e.g.
    ~1/3 of the real pot would otherwise snap to the 1/2-pot bucket (H3480).
    """
    resp = get_next_actions(
        gametype=gametype, depth=depth, stacks="",
        preflop_actions=preflop_actions, board=board,
        flop_actions=flop_actions, turn_actions=turn_actions,
        river_actions=river_actions,
    )
    available = resp.get("next_actions", {}).get("available_actions", []) or []
    # This only ever resolves a raise/bet token, so the answer must be a raise or
    # all-in (code "R*"/"RAI") — never Call/Check/Fold. The solver may offer a
    # single 3-bet size, so an undersized 3-bet (e.g. 5bb that sits between Call
    # at 2.1 and the solver's R8.2) would otherwise mis-snap to Call and put the
    # line off-tree (H3490). Restrict candidates to raises before matching.
    raises = [a for a in available
              if str(a.get("action", {}).get("code", "")).startswith("R")]
    if not raises:
        raise ValueError(
            f"no raise options at this node (target={target_size}) — off-tree"
        )
    if actual_pot > 0:
        # Pot-ratio snapping with shared all-in protection: _find_action_by_pot_pct
        # keeps a near-shove on the all-in node and otherwise drives the bucket by
        # pot ratio (the guard used to live here; now shared so the two pipelines
        # can't drift apart — H3480).
        from analyze_hand import _find_action_by_pot_pct
        code = _find_action_by_pot_pct(raises, target_size, actual_pot)
    else:
        code = find_closest_action(raises, target_size)
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

    action_index semantic: counts occurrences of hero_pos in raw_actions
    (hero-scoped, same convention as gemini_session._extract_deviations).

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
    actual_pot: float = 0.0,
) -> tuple[str, float]:
    """Resolve actions for one postflop street, emitting only actions[0:stop_after_n].

    Returns (action_string, updated_actual_pot). When ``actual_pot`` > 0 the
    street's OPENING bet is snapped by pot ratio (the meaningful signal in a
    multiway pot where dead money inflates the bet's apparent solver-pot
    fraction); later raises keep absolute snapping. The running pot is advanced
    through every emitted action so downstream streets see the right pot.
    """
    out_tokens: list[str] = []
    outstanding_bet = 0.0
    street_investments: dict[str, float] = {}
    for i, act in enumerate(raw_actions):
        if i >= stop_after_n:
            break
        action = act.get("action", "")
        pos = act.get("position")
        target = 0.0
        # Legacy live rows may persist a sized opening bet as generic ``B``
        # plus ``size``.  It is semantically the same wager as ``R{size}`` and
        # must be resolved through GTOW before URL validation; emitting raw B
        # makes GTOW discard the entire custom history.
        if action.startswith("R") or action == "B":
            target = float(act.get("size") or action[1:] or 0)
            code = _resolve_one_raise(
                gametype=gametype, depth=depth,
                preflop_actions=preflop_actions,
                board=board_so_far,
                flop_actions=prior_streets.get("flop", "") if street_key != "flop" else "-".join(out_tokens),
                turn_actions=prior_streets.get("turn", "") if street_key != "turn" else "-".join(out_tokens),
                river_actions=prior_streets.get("river", "") if street_key != "river" else "-".join(out_tokens),
                target_size=target,
                # Pot-ratio snap only the street's opening bet.
                actual_pot=actual_pot if outstanding_bet == 0 else 0.0,
            )
            out_tokens.append(code)
        else:
            out_tokens.append(action)

        # Advance the running pot (mirrors analyze_hand's postflop tracking).
        if actual_pot > 0:
            if action in ("X", "F", ""):
                pass
            elif action == "C":
                prev = street_investments.get(pos, 0.0)
                actual_pot += outstanding_bet - prev
                street_investments[pos] = outstanding_bet
            else:  # bet / raise / all-in
                prev = street_investments.get(pos, 0.0)
                actual_pot += target - prev
                street_investments[pos] = target
                outstanding_bet = target
    return "-".join(out_tokens), actual_pot


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
        actors = _replay_preflop_actors(tokens, positions)
        hero_actions = [i for i, actor in enumerate(actors) if actor == hero_pos_8]
        if action_index >= len(hero_actions):
            raise ValueError(
                f"hero preflop decision {action_index} not found for {hero_pos_8}")
        # Stop immediately before hero's Nth decision.  Hero's response to a
        # 3bet/4bet is not adjacent to their opening seat in the token stream;
        # locate it by replaying the betting order instead of hero_slot + N.
        truncated = "-".join(tokens[:hero_actions[action_index]])
        preflop_codes = _resolve_preflop_codes(gametype, depth, truncated, 0)
        flop_codes = turn_codes = river_codes = ""
    else:
        # Collapse extra cold-callers who folded before the flop into folds so
        # the preflop line stays on-tree for the heads-up postflop node the bot
        # analysed (H3480: CO cold-calls then folds to the SB 3-bet).
        collapse_positions = (
            POSITION_ORDERS[players] if _is_cash(gametype) else POSITION_ORDERS[MTT_TREE_SIZE]
        )
        hu_preflop = _collapse_coldcall_folders(padded_preflop, collapse_positions, hero_pos_8)
        preflop_codes = _resolve_preflop_codes(gametype, depth, hu_preflop, 0)
        flop_codes = turn_codes = river_codes = ""

        # Real pot at the flop, from the ORIGINAL (un-collapsed) preflop line so
        # dead money from folded cold-callers is counted. Postflop opening bets
        # are snapped against this pot by ratio (see _resolve_street_codes).
        from analyze_hand import _compute_preflop_pot
        ante = 0.0 if _is_cash(gametype) else 0.125
        pot_players = players if _is_cash(gametype) else MTT_TREE_SIZE
        actual_pot = _compute_preflop_pot(
            padded_preflop, effective_bb, num_players=pot_players, ante_per_player=ante,
        )

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
                resolved, actual_pot = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=len(actions), actual_pot=actual_pot,
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
                resolved, actual_pot = _resolve_street_codes(
                    gametype, depth, preflop_codes, board_now, prior, s_name,
                    actions, stop_after_n=raw_stop, actual_pot=actual_pot,
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
