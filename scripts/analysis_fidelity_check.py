#!/usr/bin/env python3
"""Compare GTOW Analyzer truth with the repository's ``analyze_hand_full``.

The online ledger is the sampling index; archived/live GTOW hand detail is the
truth source because the flattened ledger rows do not retain the full betting
stream.  Results are written one hand at a time to JSONL, so a long run can be
resumed safely after token/API/process interruption.

Examples:
    python scripts/analysis_fidelity_check.py --sample-size 30
    python scripts/analysis_fidelity_check.py --sample-size 400 --resume
    python scripts/analysis_fidelity_check.py --hand-id bee60039-cf87-4beb-8443-3b1d73b59a51
    python scripts/analysis_fidelity_check.py --sample-size 30 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_hand import POSITION_ORDERS, analyze_hand_full
from gto_formatter import (
    _combo_idx_in_player_range,
    _get_action_strategy_frequencies,
    combo_index_for_hand,
    normalize_hand_name,
)
from gtow_analyze_api import hand_detail
from gto_api import nearest_cash_depth, nearest_depth
from gtow_solution_url import _canonical_board_str, _parsed_hand_from_analyze
from hh_deviation_check import _get_action_evs_postflop, _get_action_evs_preflop


STREETS = ("preflop", "flop", "turn", "river")
STREET_FROM_GTOW = {
    "PREFLOP": "preflop", "FLOP": "flop", "TURN": "turn", "RIVER": "river",
}
RARE_STRATA = (
    "fivebet", "heads_up", "nine_max", "fourbet", "squeeze", "allin",
    "multi_decision", "sizing_snap", "depth_snap", "no_solution",
)
DEFAULT_OUTPUT = ROOT / "data" / "analysis_fidelity"


@dataclass(frozen=True)
class Thresholds:
    ev_bb: float = 0.05
    frequency: float = 0.05
    depth_bb: float = 0.01
    sizing_relative: float = 0.15


def normalize_code(code: str | None) -> str:
    """Normalize action aliases without hiding meaningful sizing differences."""
    code = (code or "").strip()
    if code == "RAI":
        return "AI"
    if code.startswith("B"):
        return "R" + code[1:] if len(code) > 1 else "R"
    return code


def _codes_compatible(a: str | None, b: str | None,
                      sizing_relative: float) -> bool:
    a, b = normalize_code(a), normalize_code(b)
    if a == b:
        return True
    if not (a.startswith("R") and b.startswith("R")):
        return False
    try:
        av, bv = float(a[1:]), float(b[1:])
    except ValueError:
        return False
    return abs(av - bv) / max(av, bv, 1e-9) <= sizing_relative


def _action_lines_compatible(a: str | None, b: str | None,
                             sizing_relative: float) -> bool:
    aa = [x for x in str(a or "").split("-") if x]
    bb = [x for x in str(b or "").split("-") if x]
    return len(aa) == len(bb) and all(
        _codes_compatible(x, y, sizing_relative) for x, y in zip(aa, bb)
    )


def _has_allin_numeric_drift(gtow: dict, own: dict) -> bool:
    """True when one tree stores a numeric raise where the other stores AI.

    This intentionally compares the archived solved sequence as stored. Blindly
    rewriting historical numeric codes to AI changed otherwise matching nodes;
    a remaining numeric/AI split is therefore treated as a versioned-tree
    semantic boundary, not parity evidence that local reconstruction can repair.
    """
    for field in ("preflop_actions", "flop_actions", "turn_actions", "river_actions"):
        ga = [normalize_code(x) for x in str(gtow.get(field) or "").split("-") if x]
        oa = [normalize_code(x) for x in str(own.get(field) or "").split("-") if x]
        if len(ga) != len(oa):
            continue
        for left, right in zip(ga, oa):
            if (left == "AI" and right.startswith("R")) or (
                    right == "AI" and left.startswith("R")):
                return True
    return False


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _joined(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(str(x) for x in value)
    return str(value or "")


def reconstruct_analyze_hand(hand_row: dict, detail: dict) -> dict:
    """Rebuild the exact parsed-hand shape from GTOW's real action stream."""
    hero_pos = hand_row.get("position") or hand_row.get("player_position") or ""
    depth = hand_row.get("preflop_depth_bb", hand_row.get("preflop_game_depth", 0))
    gametype = "MTTGeneral"
    gps = ((detail.get("game_analysis") or {}).get("game_points")) or []
    for gp in gps:
        if gp.get("gametype"):
            gametype = gp["gametype"]
            break
    hand = _parsed_hand_from_analyze(detail, hero_pos, float(depth or 0), gametype)
    players = int(
        hand_row.get("total_players")
        or detail.get("players_dealt")
        or hand.get("players_at_table")
        or 8
    )
    hand.update({
        "hand_id": hand_row.get("gtow_hand_id") or hand_row.get("hand_id"),
        "hero_hand": hand_row.get("hero_hand") or "",
        "num_players": players,
        "players_at_table": players,
        "game_format": "cash" if str(detail.get("format", "")).lower() == "cash" else "mtt",
    })
    # The Analyze list row commonly stores hero's physical stack, whereas each
    # graded game-point records the solver avatar depth actually used for the
    # decision. Feed analyze_hand the latter; keep the row value as audit
    # metadata rather than manufacturing downstream sizing/depth mismatches.
    for gp in gps:
        action = gp.get("real_game_action") or {}
        gp_depth = _float(gp.get("depth"))
        available = (gp.get("analysis_solved") or {}).get("available_actions") or []
        if (action.get("position") == hero_pos and gp_depth is not None
                and any(a.get("selected") for a in available)):
            hand["ledger_preflop_depth_bb"] = float(depth or 0)
            if hand["game_format"] == "mtt" and abs((gp_depth % 1) - 0.125) < 1e-6:
                hand["effective_bb"] = gp_depth - 0.125
            else:
                hand["effective_bb"] = gp_depth
            break
    _restore_analyze_allins(hand, detail)
    _attach_real_stacks_and_effective(hand, detail, hero_pos, players)
    _truncate_after_hero_fold(hand, detail, hero_pos)
    return hand


def _restore_analyze_allins(hand: dict, detail: dict) -> None:
    """Translate GTOW ``RAI`` into analyze_hand's explicit ``AI`` contract.

    ``_parsed_hand_from_analyze`` intentionally turns RAI into a numeric raise
    for the deep-link resolver. The analysis engine needs the opposite: an AI
    token/flag so effective-stack and terminal-action logic can recognize it.
    """
    gps = ((detail.get("game_analysis") or {}).get("game_points")) or []
    by_street: dict[str, list[dict]] = defaultdict(list)
    for gp in gps:
        rg = gp.get("real_game") or {}
        street = STREET_FROM_GTOW.get(
            str((rg.get("current_street") or {}).get("type", "")).upper())
        action = gp.get("real_game_action") or {}
        if street and action.get("code"):
            by_street[street].append(action)

    preflop = [p for p in str(hand.get("preflop_actions") or "").split("-") if p]
    for i, action in enumerate(by_street["preflop"]):
        if i < len(preflop) and action.get("code") == "RAI":
            preflop[i] = f"AI{float(action.get('betsize') or 0):g}"
    hand["preflop_actions"] = "-".join(preflop)

    for street_name, street in zip(("flop", "turn", "river"), hand.get("streets") or []):
        real_actions = by_street[street_name]
        for i, action in enumerate(street.get("actions") or []):
            if i < len(real_actions) and real_actions[i].get("code") == "RAI":
                action["action"] = "AI"
                action["allin"] = True
                action["size"] = float(real_actions[i].get("betsize") or action.get("size") or 0)


def _attach_real_stacks_and_effective(hand: dict, detail: dict,
                                      hero_pos: str, players_at_table: int) -> None:
    """Rebuild HH-parser-equivalent effective depth from GTOW real seats.

    ``ledger_hands.preflop_depth_bb`` is commonly hero's depth, not the stack
    that bound the played spot.  GTOW detail retains every physical starting
    stack, so postflop hands use the shortest player who actually reached the
    flop; preflop-only folds use players still live when hero acted.
    """
    gps = ((detail.get("game_analysis") or {}).get("game_points")) or []
    first_players = next(
        ((gp.get("real_game") or {}).get("players") for gp in gps
         if (gp.get("real_game") or {}).get("players")),
        [],
    )
    stacks = {
        p.get("position"): float(p.get("stack") or 0)
        for p in first_players if p.get("position") and _float(p.get("stack")) is not None
    }
    hero_stack = stacks.get(hero_pos)
    if not hero_stack:
        return
    first_real_game = next(
        ((gp.get("real_game") or {}) for gp in gps
         if (gp.get("real_game") or {}).get("players")),
        {},
    )
    pot_before = _float(first_real_game.get("pot"))
    posted = sum(
        float(p.get("chips_on_table") or 0)
        for p in first_real_game.get("players") or []
    )
    if pot_before is not None and players_at_table > 0:
        ante = (pot_before - posted) / players_at_table
        if 0 <= ante <= 0.5:
            hand["ante_per_player"] = round(ante, 6)
    order = POSITION_ORDERS.get(players_at_table)
    if order and all(pos in stacks for pos in order):
        hand["player_stacks"] = [stacks[pos] for pos in order]
    hand["hero_starting_stack"] = hero_stack

    folded: set[str] = set()
    live_at_hero: set[str] | None = None
    reached_flop = False
    hero_folded_preflop = False
    for gp in gps:
        rg = gp.get("real_game") or {}
        street = STREET_FROM_GTOW.get(
            str((rg.get("current_street") or {}).get("type", "")).upper())
        action = gp.get("real_game_action") or {}
        pos, code = action.get("position"), action.get("code")
        if street != "preflop":
            reached_flop = True
            break
        if pos == hero_pos and live_at_hero is None:
            live_at_hero = set(stacks) - folded - {hero_pos}
        if code == "F" and pos:
            folded.add(pos)
            if pos == hero_pos:
                hero_folded_preflop = True

    opponents = (
        (live_at_hero or (set(stacks) - folded - {hero_pos}))
        if hero_folded_preflop
        else (set(stacks) - folded - {hero_pos})
    )
    opponent_stacks = [stacks[p] for p in opponents if stacks.get(p, 0) > 0]
    # Exact HU has one unambiguous binding opponent. In multiway/all-in side-pot
    # hands the shortest stack is often NOT the decision being graded (e.g. a
    # 9bb caller plus a 50bb shover); retain hero depth and let analyze_hand's
    # per-node resolver select the relevant cover/jam depth.
    # This also covers a preflop all-in runout whose detail has no postflop
    # action game-points: final folds can still leave one exact opponent.
    if not hero_folded_preflop and len(opponent_stacks) == 1:
        hand["effective_bb"] = min(hero_stack, min(opponent_stacks))


def _truncate_after_hero_fold(hand: dict, detail: dict, hero_pos: str) -> None:
    """Stop replay once hero folds; later table action is not hero analysis input."""
    gps = ((detail.get("game_analysis") or {}).get("game_points")) or []
    fold_street = None
    preflop_before_fold: list[str] = []
    for gp in gps:
        real_game = gp.get("real_game") or {}
        street = STREET_FROM_GTOW.get(
            str((real_game.get("current_street") or {}).get("type", "")).upper())
        action = gp.get("real_game_action") or {}
        if street == "preflop" and action.get("code"):
            code = action["code"]
            if code == "RAI":
                code = f"AI{float(action.get('betsize') or 0):g}"
            preflop_before_fold.append(code)
        if action.get("position") == hero_pos and action.get("code") == "F":
            fold_street = street
            break
    if fold_street is None:
        return
    if fold_street == "preflop":
        hand["preflop_actions"] = "-".join(preflop_before_fold)
        hand["streets"] = []
        return
    fold_idx = STREETS.index(fold_street) - 1
    streets = list(hand.get("streets") or [])
    if fold_idx >= len(streets):
        return
    actions = streets[fold_idx].get("actions") or []
    for i, action in enumerate(actions):
        if action.get("position") == hero_pos and action.get("action") == "F":
            streets[fold_idx] = {**streets[fold_idx], "actions": actions[:i + 1]}
            break
    hand["streets"] = streets[:fold_idx + 1]


def gtow_decisions(detail: dict, hero_pos: str, *, solution_status: str | None = None) -> list[dict]:
    """Extract per-hero-decision GTOW truth keyed like ledger decisions."""
    counts: dict[str, int] = defaultdict(int)
    out: list[dict] = []
    analysis = detail.get("game_analysis") or {}
    gps = analysis.get("game_points") or []
    warning_status = analysis.get("warning_status")
    hand_exclusion_reasons = []
    if solution_status and solution_status != "OK":
        hand_exclusion_reasons.append(f"solution:{solution_status}")
    if warning_status and warning_status != "OK":
        hand_exclusion_reasons.append(f"warning:{warning_status}")
    for gp in gps:
        action = gp.get("real_game_action") or {}
        if action.get("position") != hero_pos:
            continue
        street_type = ((gp.get("real_game") or {}).get("current_street") or {}).get("type", "")
        street = STREET_FROM_GTOW.get(str(street_type).upper(), "preflop")
        idx = counts[street]
        counts[street] += 1
        available = (gp.get("analysis_solved") or {}).get("available_actions") or []
        selected = next((a for a in available if a.get("selected")), None)
        if selected is None:
            out.append({
                "street": street, "decision_idx": idx, "key": f"{street}:{idx}",
                "gtow_ungraded": True, "has_solution": bool(gp.get("has_solution")),
                "gtow_excluded": bool(hand_exclusion_reasons),
                "gtow_exclusion_reasons": list(hand_exclusion_reasons),
            })
            continue
        best = next((a for a in available if a.get("correctness") == "BEST_MOVE"), None)
        if best is None:
            numeric = [a for a in available if _float(a.get("ev")) is not None]
            best = max(numeric, key=lambda a: float(a["ev"])) if numeric else None
        acceptable = {
            normalize_code((a.get("action") or {}).get("code"))
            for a in available
            if a.get("correctness") in ("BEST_MOVE", "CORRECT_MOVE")
            and (_float(a.get("ev_loss")) or 0.0) <= 1e-9
        }
        seq = gp.get("solved_action_sequence") or {}
        real_game = gp.get("real_game") or {}
        selected_ev = _float(selected.get("ev"))
        best_ev = _float(best.get("ev")) if best else None
        solver_ev_loss = (
            max(0.0, best_ev - selected_ev)
            if selected_ev is not None and best_ev is not None else None
        )
        out.append({
            "street": street,
            "decision_idx": idx,
            "key": f"{street}:{idx}",
            "gametype": gp.get("gametype") or "",
            "depth": _float(gp.get("depth")),
            "board": _canonical_board_str(real_game.get("board") or "", street),
            "preflop_actions": _joined(seq.get("preflop_actions")),
            "flop_actions": _joined(seq.get("flop_actions")),
            "turn_actions": _joined(seq.get("turn_actions")),
            "river_actions": _joined(seq.get("river_actions")),
            "taken_code": normalize_code((selected.get("action") or {}).get("code")),
            "taken_pot_pct": _float((selected.get("action") or {}).get("betsize_by_pot")),
            "best_code": normalize_code((best.get("action") or {}).get("code")) if best else "",
            "acceptable_codes": sorted(c for c in acceptable if c),
            "correctness": selected.get("correctness"),
            "ev_loss_bb": _float(selected.get("ev_loss")),
            # Analyze's ``ev_loss`` is product-policy output and can be zero
            # even when available-action EVs differ (e.g. INACCURACY rows).
            # The local analyzer computes the raw best-minus-taken delta, so
            # parity must compare against the same raw GTOW solver values.
            "solver_ev_loss_bb": solver_ev_loss,
            "taken_freq": _float(selected.get("frequency")),
            "pot_bb": _float(real_game.get("pot")),
            "has_solution": bool(gp.get("has_solution")),
            "gtow_ungraded": (
                _float(selected.get("ev")) is None
                and _float(selected.get("ev_loss")) is None
                and selected.get("correctness") is None
            ),
            # Match ledger honesty: GTOW unknown/no-solution hands are useful
            # evidence that a fallback was needed, but they are not an oracle
            # against which the fallback's EV can be graded.
            "gtow_excluded": bool(hand_exclusion_reasons),
            "gtow_exclusion_reasons": list(hand_exclusion_reasons),
        })
    return out


def own_decisions(result: dict, hero_hand_raw: str) -> list[dict]:
    """Extract comparable metrics from structured ``analyze_hand_full`` output."""
    counts: dict[str, int] = defaultdict(int)
    out: list[dict] = []
    hero_hand = normalize_hand_name(hero_hand_raw)
    combo_idx = combo_index_for_hand(hero_hand_raw)
    spots = result.get("hero_spots") or []
    solutions = result.get("solutions") or []
    normalized_full_preflop = [
        p for p in str(result.get("preflop_actions") or "").split("-") if p
    ]
    for spot_i, spot in enumerate(spots):
        street = spot.get("street") or "preflop"
        idx = counts[street]
        counts[street] += 1
        key = f"{street}:{idx}"
        params = spot.get("params") or {}
        solution = solutions[spot_i] if spot_i < len(solutions) else None
        taken = normalize_code(spot.get("taken_code"))
        # analyze_hand historically omits taken_code on the initial preflop
        # spot.  Its params are the prefix immediately BEFORE hero acts, so the
        # next token in the normalized full line is the exact action evaluated.
        if not taken and street == "preflop":
            prefix_len = len([p for p in str(params.get("preflop_actions") or "").split("-") if p])
            if prefix_len < len(normalized_full_preflop):
                taken = normalize_code(normalized_full_preflop[prefix_len])
        hero_pos = spot.get("solver_hero_pos") or result.get("hero_position") or ""
        is_preflop = street == "preflop"
        action_evs = None
        frequencies = None
        in_range = None
        if solution:
            if is_preflop:
                action_evs = _get_action_evs_preflop(solution, hero_hand, hero_pos)
            else:
                in_range = (
                    _combo_idx_in_player_range(solution, hero_pos, combo_idx)
                    if combo_idx is not None else None
                )
                action_evs = _get_action_evs_postflop(
                    solution, hero_hand, hero_pos, combo_idx=combo_idx)
            frequencies = _get_action_strategy_frequencies(
                solution, hero_hand, hero_pos, is_preflop, None if is_preflop else combo_idx
            )
        action_evs = {normalize_code(k): float(v) for k, v in (action_evs or {}).items()}
        frequencies = {normalize_code(k): float(v) for k, v in (frequencies or {}).items()}
        # GTOW's displayed/recommended action is strategy-led. Raw EV arrays
        # on tiny/unused branches can contain solve noise (e.g. a 98.7%-check
        # combo whose rare bet EV is numerically 0.12bb higher). Match the
        # product's dominant strategy action; keep raw EVs only for loss math.
        best_code = max(frequencies, key=frequencies.get) if frequencies else ""
        taken_pot_pct = None
        if solution and taken:
            selected_solution = next(
                (a for a in solution.get("action_solutions", [])
                 if normalize_code((a.get("action") or {}).get("code")) == taken),
                None,
            )
            taken_pot_pct = _float(
                ((selected_solution or {}).get("action") or {}).get("betsize_by_pot"))
        ev_loss = None
        if taken and taken in action_evs and best_code:
            ev_loss = max(0.0, action_evs[best_code] - action_evs[taken])
        # A BB walk is N-1 folds and contains no hero decision. analyze_hand
        # can retain a root placeholder for presentation, but it is not a
        # comparable decision and GTOW correctly emits none.
        if not taken and solution is None:
            continue
        out.append({
            "street": street,
            "decision_idx": idx,
            "key": key,
            "gametype": params.get("gametype") or "",
            "depth": _float(params.get("depth")),
            "board": params.get("board") or "",
            "preflop_actions": params.get("preflop_actions") or "",
            "flop_actions": params.get("flop_actions") or "",
            "turn_actions": params.get("turn_actions") or "",
            "river_actions": params.get("river_actions") or "",
            "taken_code": taken,
            "taken_pot_pct": taken_pot_pct,
            "best_code": best_code,
            "ev_loss_bb": ev_loss,
            "taken_freq": frequencies.get(taken) if taken else None,
            "has_solution": solution is not None,
            "in_range": in_range,
            "spot_index": spot_i,
        })
    return out


def _node_differences(gtow: dict, own: dict, thresholds: Thresholds) -> list[str]:
    diffs: list[str] = []
    if gtow.get("gametype") != own.get("gametype"):
        diffs.append("gametype")
    gd, od = gtow.get("depth"), own.get("depth")
    gametype = str(gtow.get("gametype") or own.get("gametype") or "")
    if gd is None or od is None:
        diffs.append("depth")
    else:
        if gametype.startswith("Cash"):
            gkey, okey = nearest_cash_depth(gd), nearest_cash_depth(od)
        else:
            gkey, okey = nearest_depth(gd), nearest_depth(od)
        if abs(gkey - okey) > thresholds.depth_bb:
            diffs.append("depth")
    if (gtow.get("board") or "") != (own.get("board") or ""):
        diffs.append("board")
    for field in ("preflop_actions", "flop_actions", "turn_actions", "river_actions"):
        if not _action_lines_compatible(
                gtow.get(field), own.get(field), thresholds.sizing_relative):
            diffs.append(field)
    return diffs


def compare_decisions(gtow: list[dict], own: list[dict],
                      thresholds: Thresholds = Thresholds(), *,
                      gtow_hand_unknown: bool = False) -> list[dict]:
    """Compare decisions without treating different solver nodes as EV failures."""
    gtow_by = {d["key"]: d for d in gtow}
    own_by = {d["key"]: d for d in own}
    rows: list[dict] = []
    prior_zero_frequency_action = False
    for key in sorted(set(gtow_by) | set(own_by), key=_decision_sort_key):
        g, o = gtow_by.get(key), own_by.get(key)
        if g is None:
            status = "skipped_gtow_unknown" if gtow_hand_unknown else "extra_own_decision"
            rows.append({"key": key, "status": status, "gtow": None, "own": o})
            continue
        if o is None:
            if g.get("gtow_excluded"):
                status = "skipped_gtow_unknown"
            elif g.get("gtow_ungraded"):
                status = "skipped_gtow_ungraded"
            else:
                status = "missing_own_decision"
            rows.append({"key": key, "status": status, "gtow": g, "own": None})
            continue
        if g.get("gtow_excluded"):
            rows.append({
                "key": key, "status": "skipped_gtow_unknown",
                "gtow": g, "own": o,
            })
            continue
        if g.get("gtow_ungraded"):
            rows.append({
                "key": key, "status": "skipped_gtow_ungraded",
                "gtow": g, "own": o,
            })
            continue
        node_diffs = _node_differences(g, o, thresholds)
        taken_match = _codes_compatible(
            g.get("taken_code"), o.get("taken_code"), thresholds.sizing_relative)
        if not taken_match:
            gpct, opct = g.get("taken_pot_pct"), o.get("taken_pot_pct")
            taken_match = (
                gpct is not None and opct is not None
                and abs(float(gpct) - float(opct)) <= thresholds.sizing_relative
            )
        acceptable = set(g.get("acceptable_codes") or [])
        ev_delta = None
        gtow_solver_loss = g.get("solver_ev_loss_bb", g.get("ev_loss_bb"))
        if gtow_solver_loss is not None and o.get("ev_loss_bb") is not None:
            ev_delta = o["ev_loss_bb"] - gtow_solver_loss
        freq_delta = None
        if g.get("taken_freq") is not None and o.get("taken_freq") is not None:
            freq_delta = o["taken_freq"] - g["taken_freq"]
        best_compatible = (
            (bool(o.get("best_code")) and any(
                _codes_compatible(o.get("best_code"), code, thresholds.sizing_relative)
                for code in acceptable
            ))
            or (not g.get("best_code") and not acceptable)
            or (
                g.get("taken_code") in acceptable
                and o.get("ev_loss_bb") is not None
                and o["ev_loss_bb"] <= thresholds.ev_bb
            )
        )

        if g.get("gtow_excluded"):
            status = "skipped_gtow_unknown"
        elif not o.get("has_solution"):
            status = "missing_own_solution"
        elif not taken_match:
            status = "taken_action_mismatch"
        elif o.get("in_range") is False:
            status = (
                "skipped_own_offtree_continuation"
                if prior_zero_frequency_action else "own_combo_off_range"
            )
        elif (
            node_diffs
            and set(node_diffs) <= {
                "preflop_actions", "flop_actions", "turn_actions", "river_actions"
            }
            and _has_allin_numeric_drift(g, o)
        ):
            status = "skipped_solver_tree_semantic_drift"
        elif node_diffs:
            status = "node_mismatch"
        elif not best_compatible:
            status = "best_action_mismatch"
        elif ev_delta is None:
            status = "ev_unavailable"
        elif abs(ev_delta) > thresholds.ev_bb:
            status = "ev_mismatch"
        elif freq_delta is not None and abs(freq_delta) > thresholds.frequency:
            status = "frequency_mismatch"
        else:
            status = "match"
        rows.append({
            "key": key, "status": status, "node_differences": node_diffs,
            "taken_match": taken_match, "best_compatible": best_compatible,
            "ev_delta_bb": ev_delta, "frequency_delta": freq_delta,
            "gtow": g, "own": o,
        })
        taken_freq = o.get("taken_freq")
        if taken_freq is not None and float(taken_freq) <= 1e-12:
            prior_zero_frequency_action = True
    return rows


def _decision_sort_key(key: str) -> tuple[int, int]:
    street, _, idx = key.partition(":")
    return (STREETS.index(street) if street in STREETS else 99, int(idx or 0))


def compare_hand(hand_row: dict, detail: dict, analyze_fn=analyze_hand_full,
                 thresholds: Thresholds = Thresholds()) -> dict:
    """Pure orchestration boundary; ``analyze_fn`` is injectable for tests."""
    hand = reconstruct_analyze_hand(hand_row, detail)
    gtow = gtow_decisions(
        detail, hand["hero_position"], solution_status=hand_row.get("solution_status")
    )
    analysis = detail.get("game_analysis") or {}
    solution_status = hand_row.get("solution_status")
    gtow_hand_unknown = bool(
        (solution_status and solution_status != "OK")
        or (analysis.get("warning_status") and analysis.get("warning_status") != "OK")
    )
    # GTOW explicitly has no oracle here. Preserve the repository's multiway
    # fallback, but do not waste solver calls or grade that approximation as if
    # GTOW had produced a comparable answer.
    if gtow_hand_unknown:
        result = {"validation": None}
        own: list[dict] = []
    else:
        result = analyze_fn(hand)
        own = own_decisions(result, hand["hero_hand"])
    comparisons = compare_decisions(
        gtow, own, thresholds, gtow_hand_unknown=gtow_hand_unknown
    )
    counts = Counter(r["status"] for r in comparisons)
    return {
        "gtow_hand_id": hand["hand_id"],
        "played_at": _json_value(hand_row.get("played_at")),
        "position": hand["hero_position"],
        "hero_hand": hand["hero_hand"],
        "total_players": hand["num_players"],
        "pot_type": hand_row.get("pot_type"),
        "gtow_total_ev_loss_bb": _float(hand_row.get("total_ev_loss_bb")),
        "reconstructed_hand": hand,
        "validation": result.get("validation"),
        "summary": dict(counts),
        "decisions": comparisons,
    }


def classify_candidate(row: dict) -> list[str]:
    pot = str(row.get("pot_type") or "").lower()
    flags = set(row.get("flags") or [])
    labels: list[str] = []
    if pot == "5bet": labels.append("fivebet")
    if int(row.get("total_players") or 0) == 2: labels.append("heads_up")
    if int(row.get("total_players") or 0) == 9: labels.append("nine_max")
    if pot == "4bet": labels.append("fourbet")
    if pot == "squeeze": labels.append("squeeze")
    if row.get("has_allin"): labels.append("allin")
    if row.get("has_multi_decision"): labels.append("multi_decision")
    if "sizing_snap" in flags: labels.append("sizing_snap")
    if "depth_snap_gap" in flags: labels.append("depth_snap")
    if any("NO_GTO_SOLUTION" in f or "ZERO_PERCENT_ACTION" in f for f in flags):
        labels.append("no_solution")
    return labels


def select_sample(candidates: list[dict], size: int, seed: int) -> list[dict]:
    """Deterministic rare-first sample, then high-loss and baseline fill."""
    if size <= 0:
        return []
    rng = random.Random(seed)
    rows = [dict(r) for r in candidates]
    for row in rows:
        row["sample_strata"] = classify_candidate(row)
    chosen: list[dict] = []
    seen: set[str] = set()

    rare_target = min(len(rows), max(len(RARE_STRATA), size // 2), size)
    buckets = {s: [r for r in rows if s in r["sample_strata"]] for s in RARE_STRATA}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while len(chosen) < rare_target:
        progressed = False
        for stratum in RARE_STRATA:
            bucket = buckets[stratum]
            while bucket and str(bucket[0]["gtow_hand_id"]) in seen:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            row["sample_reason"] = stratum
            chosen.append(row); seen.add(str(row["gtow_hand_id"])); progressed = True
            if len(chosen) >= rare_target:
                break
        if not progressed:
            break

    high_target = min(size, len(chosen) + max(1, size // 5))
    for row in sorted(rows, key=lambda r: float(r.get("total_ev_loss_bb") or 0), reverse=True):
        hid = str(row["gtow_hand_id"])
        if hid in seen:
            continue
        row["sample_reason"] = "high_loss"
        chosen.append(row); seen.add(hid)
        if len(chosen) >= high_target:
            break

    remaining = [r for r in rows if str(r["gtow_hand_id"]) not in seen]
    rng.shuffle(remaining)
    for row in remaining:
        row["sample_reason"] = "baseline"
        chosen.append(row)
        if len(chosen) >= size:
            break
    return chosen[:size]


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    return value


def _json_default(value: Any) -> Any:
    converted = _json_value(value)
    if converted is not value:
        return converted
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def load_archived_detail(raw_path: str | None) -> dict | None:
    if not raw_path:
        return None
    path = ROOT / raw_path
    if not path.exists():
        return None
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            completed.add(str(json.loads(line)["gtow_hand_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(result, ensure_ascii=False, default=_json_default) + "\n")


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_report(results: Iterable[dict]) -> str:
    results = list(results)
    statuses = Counter()
    reasons = Counter()
    total_decisions = 0
    errors = [r for r in results if r.get("error")]
    for hand in results:
        reasons[hand.get("sample_reason") or "manual"] += 1
        for dec in hand.get("decisions") or []:
            statuses[dec["status"]] += 1
            total_decisions += 1
    matched = statuses.get("match", 0)
    skipped_unknown = statuses.get("skipped_gtow_unknown", 0)
    skipped_ungraded = statuses.get("skipped_gtow_ungraded", 0)
    skipped_offtree = statuses.get("skipped_own_offtree_continuation", 0)
    skipped_tree_drift = statuses.get("skipped_solver_tree_semantic_drift", 0)
    comparable = (
        total_decisions - skipped_unknown - skipped_ungraded
        - skipped_offtree - skipped_tree_drift
    )
    lines = [
        "# GTOW Analyzer vs analyze_hand fidelity",
        "",
        f"- hands: {len(results)}",
        f"- hand errors: {len(errors)}",
        f"- decisions: {total_decisions}",
        f"- GTOW-unknown decisions skipped: {skipped_unknown}",
        f"- GTOW-ungraded decisions skipped: {skipped_ungraded}",
        f"- local zero-frequency continuations skipped: {skipped_offtree}",
        f"- archived/current semantic tree drifts skipped: {skipped_tree_drift}",
        f"- exact comparable matches: {matched}/{comparable}",
        "",
        "## Statuses",
        "",
        "| status | n |",
        "|---|---:|",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in statuses.most_common())
    lines.extend(["", "## Sample reasons", "", "| reason | hands |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in reasons.most_common())
    if errors:
        lines.extend(["", "## Hand errors", "", "| hand | error |", "|---|---|"])
        for hand in errors:
            err = str(hand.get("error") or "").replace("|", "\\|")
            lines.append(f"| {str(hand.get('gtow_hand_id'))[:8]} | {err} |")
    lines.extend([
        "", "## Mismatches", "",
        "| hand | spot | status | node diffs | GTOW loss | own loss |",
        "|---|---|---|---|---:|---:|",
    ])
    for hand in results:
        for dec in hand.get("decisions") or []:
            if dec["status"] == "match":
                continue
            g, o = dec.get("gtow") or {}, dec.get("own") or {}
            lines.append(
                f"| {str(hand.get('gtow_hand_id'))[:8]} | {dec['key']} | {dec['status']} | "
                f"{','.join(dec.get('node_differences') or []) or '-'} | "
                f"{_fmt(g.get('ev_loss_bb'))} | {_fmt(o.get('ev_loss_bb'))} |"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


CANDIDATE_SQL = """
SELECT h.gtow_hand_id, h.played_at, h.position, h.hero_hand, h.total_players,
       h.pot_type, h.preflop_depth_bb, h.total_ev_loss_bb, h.solution_status, h.raw_path,
       COALESCE(bool_or(d.decision_idx > 0), false) AS has_multi_decision,
       COALESCE(bool_or(d.taken_code = 'AI' OR d.best_code = 'AI'), false) AS has_allin,
       COALESCE(array_agg(DISTINCT f.flag) FILTER (WHERE f.flag IS NOT NULL), '{}') AS flags
FROM ledger_hands h
LEFT JOIN ledger_decisions d ON d.gtow_hand_id=h.gtow_hand_id AND d.source='online'
LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(d.approx_flags, '[]'::jsonb)) f(flag) ON true
WHERE h.source='online' AND h.detail_fetched
GROUP BY h.gtow_hand_id, h.played_at, h.position, h.hero_hand, h.total_players,
         h.pot_type, h.preflop_depth_bb, h.total_ev_loss_bb, h.solution_status, h.raw_path
"""


async def fetch_candidates(conn, hand_ids: list[str] | None = None) -> list[dict]:
    sql = CANDIDATE_SQL
    args: list[Any] = []
    if hand_ids:
        sql += " HAVING h.gtow_hand_id = ANY($1::text[])"
        args.append(hand_ids)
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def amain(args) -> int:
    out_dir = Path(args.output_dir)
    jsonl = out_dir / "results.jsonl"
    report_path = out_dir / "report.md"
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        candidates = await fetch_candidates(conn, args.hand_id or None)
    finally:
        await conn.close()
    if args.hand_id:
        by_id = {str(r["gtow_hand_id"]): r for r in candidates}
        sample = [by_id[h] for h in args.hand_id if h in by_id]
        for row in sample:
            row["sample_reason"] = "manual"
            row["sample_strata"] = classify_candidate(row)
    else:
        sample = select_sample(candidates, args.sample_size, args.seed)
    if args.dry_run:
        print(json.dumps(sample, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    completed = load_completed(jsonl) if args.resume else set()
    if not args.resume and jsonl.exists():
        jsonl.unlink()
    for i, row in enumerate(sample, 1):
        hid = str(row["gtow_hand_id"])
        if hid in completed:
            print(f"[{i}/{len(sample)}] {hid[:8]} resume-skip")
            continue
        print(f"[{i}/{len(sample)}] {hid[:8]} {row.get('sample_reason')}", flush=True)
        detail = load_archived_detail(row.get("raw_path"))
        detail_source = "archive"
        if detail is None:
            detail = hand_detail(hid)
            detail_source = "api"
        if detail is None:
            result = {
                "gtow_hand_id": hid, "sample_reason": row.get("sample_reason"),
                "sample_strata": row.get("sample_strata"), "error": "detail_unavailable",
                "decisions": [],
            }
        else:
            try:
                result = compare_hand(row, detail)
                result.update({
                    "sample_reason": row.get("sample_reason"),
                    "sample_strata": row.get("sample_strata"),
                    "detail_source": detail_source,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as exc:
                result = {
                    "gtow_hand_id": hid, "sample_reason": row.get("sample_reason"),
                    "sample_strata": row.get("sample_strata"),
                    "detail_source": detail_source, "error": f"{type(exc).__name__}: {exc}",
                    "decisions": [],
                }
        append_result(jsonl, result)
    results = read_results(jsonl)
    report = render_report(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"JSONL: {jsonl}\nReport: {report_path}")
    return 1 if any(r.get("error") for r in results) else 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--hand-id", action="append", default=[])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(parse_args())))
