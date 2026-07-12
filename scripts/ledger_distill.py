#!/usr/bin/env python3
"""Distill raw GTOW Analyze JSON into ledger rows. Pure functions, re-runnable.

Input = (list_row, detail) as returned by gtow_analyze_api. Output rows use
exactly the ledger_hands / ledger_decisions column names. Raw stays on disk;
this module can always be re-run over the archive when taxonomy evolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Single official taxonomy (§4.2): action-line spot columns come from
# spot_taxonomy via backfill_spots; the legacy ~15-bucket family/texture is
# no longer written. Only the shared pot-type helper remains imported.
from spot_categorizer import compute_pot_type_from_preflop

STREET_ORDER = ["preflop", "flop", "turn", "river"]
CHIPEV_FLAG = "chipev_grading"
DEPTH_GAP_BB = 3.0
SIZING_SNAP_REL = 0.25


def depth_band(bb: float) -> str:
    if bb < 15: return "le15"
    if bb < 25: return "15_25"
    if bb < 40: return "25_40"
    return "40plus"


def _street_of(gp: dict) -> str:
    t = (gp.get("real_game") or {}).get("current_street", {}).get("type", "")
    t = (t or "").lower()
    return t if t in STREET_ORDER else "preflop"


def _norm_code(code: str) -> str:
    """Normalize GTOW action code to spot_categorizer token vocabulary."""
    if not code: return ""
    if code == "RAI": return "AI"
    if code.startswith("B"):          # defensive: bets appear as R{n} in solved codes
        return "R" + code[1:] if len(code) > 1 else "R2"
    return code                        # F / C / X / R{n} pass through


def _board_for_street(boards: str, street: str) -> str | None:
    if not boards: return None
    n = {"flop": 6, "turn": 8, "river": 10}.get(street)
    return boards[:n] if n else None


def _facing(street_actions: list[dict]) -> str:
    last_aggr = None
    saw_check = False
    for a in street_actions:
        c = a["action"]
        if c.startswith("R") or c.startswith("AI"): last_aggr = c
        elif c == "X": saw_check = True
    if last_aggr: return f"vs_{last_aggr}"
    return "checked_to" if saw_check else "unopened"


def _hand_flags(list_row: dict, ga: dict) -> tuple[list[str], bool]:
    """Hand-level honesty flags + hand-level exclusion."""
    flags, excluded = [], False
    ss = list_row.get("solution_status")
    if ss and ss != "OK":
        flags.append(f"solution:{ss}"); excluded = True
    ws = ga.get("warning_status")
    if ws and ws != "OK":
        flags.append(f"warning:{ws}"); excluded = True
    ar = ga.get("approximation_reason")
    if ar:
        flags.append(f"approx:{ar}")
    return flags, excluded


def distill_hand(list_row: dict, detail: dict) -> tuple[dict, list[dict]]:
    ga = detail.get("game_analysis") or {}
    gps = ga.get("game_points") or []
    hero_pos = list_row.get("player_position", "")
    boards = (list_row.get("boards") or [""])[0]
    real_depth = float(list_row.get("preflop_game_depth") or 0)

    hand_row = {
        "gtow_hand_id": list_row["hand_id"],
        "played_at": list_row["played_at"],
        "tournament_id": list_row.get("tournament_id"),
        "tournament_name": list_row.get("tournament_name"),
        "tournament_buyin": list_row.get("tournament_buyin"),
        "file_name": list_row.get("file_original_name"),
        "site": list_row.get("site"),
        "position": hero_pos,
        "hero_hand": list_row.get("hero_hand"),
        "boards": boards,
        "pot_type": list_row.get("pot_type"),
        "total_players": list_row.get("total_players"),
        "preflop_depth_bb": real_depth,
        "total_ev_loss_bb": float(list_row.get("total_ev_loss") or 0),
        "total_ev_loss_pct_pot": float(list_row.get("total_ev_loss_as_pot") or 0),
        "avg_gto_score": float(list_row["avg_gto_score"]) if list_row.get("avg_gto_score") is not None else None,
        "winloss_bb": float(list_row["player_winloss"]) if list_row.get("player_winloss") is not None else None,
        "hand_correctness": list_row.get("hand_correctness"),
        "solution_status": list_row.get("solution_status"),
    }

    hand_flags, hand_excluded = _hand_flags(list_row, ga)
    gtow_texture = "/".join(x for x in (list_row.get("board_flop_connectedness"),
                                        list_row.get("board_flop_pairedness")) if x) or None

    decisions: list[dict] = []
    preflop_tokens: list[str] = []               # action-order tokens (== seat order round 1)
    street_actions: dict[str, list[dict]] = {s: [] for s in STREET_ORDER}
    hero_count: dict[str, int] = {s: 0 for s in STREET_ORDER}

    for gp in gps:
        rga = gp.get("real_game_action") or {}
        sga = gp.get("solved_game_action") or rga
        pos = rga.get("position", "")
        street = _street_of(gp)
        code = _norm_code(sga.get("code") or rga.get("code") or "")

        sol = gp.get("analysis_solved") or {}
        avail = sol.get("available_actions") or []
        is_hero_decision = pos == hero_pos and any(a.get("selected") for a in avail)

        if is_hero_decision:
            sel = next(a for a in avail if a.get("selected"))
            best = next((a for a in avail if a.get("correctness") == "BEST_MOVE"), None)
            if best is None:
                best = max(avail, key=lambda a: float(a.get("ev") or 0))
            idx = hero_count[street]

            flags = list(hand_flags)
            excluded = hand_excluded
            corr = sel.get("correctness")
            if corr in (None, "UNSOLVED"):
                flags.append("unsolved"); excluded = True
            gp_depth = float(gp.get("depth") or 0)
            if real_depth and gp_depth and abs(real_depth - gp_depth) > DEPTH_GAP_BB:
                flags.append("depth_snap_gap")
            gametype = gp.get("gametype") or ""
            if "ICM" not in gametype.upper():
                flags.append(CHIPEV_FLAG)
            rb, sb_ = rga.get("betsize"), sga.get("betsize")
            try:
                rbf, sbf = float(rb or 0), float(sb_ or 0)
                if rbf > 0 and sbf > 0 and abs(rbf - sbf) / rbf > SIZING_SNAP_REL:
                    flags.append("sizing_snap")
            except (TypeError, ValueError):
                pass

            rg = gp.get("real_game") or {}
            decisions.append({
                "gtow_hand_id": list_row["hand_id"],
                "street": street, "decision_idx": idx,
                "source": "online", "grader": "gtow_analyzer",
                "gtow_texture": gtow_texture,
                "depth_band": depth_band(real_depth),
                "position": hero_pos,
                "pot_type": compute_pot_type_from_preflop(
                    "-".join(preflop_tokens), list_row.get("total_players") or 8),
                "facing": _facing(street_actions[street]),
                "taken_code": _norm_code((sel.get("action") or {}).get("code", "")),
                "best_code": _norm_code((best.get("action") or {}).get("code", "")),
                "correctness": corr,
                "ev_loss_bb": float(sel.get("ev_loss") or 0),
                "ev_loss_pct_pot": float(sel.get("ev_loss_as_pot") or 0),
                "taken_freq": float(sel.get("frequency") or 0),
                "freq_diff": float(sel.get("frequency_difference") or 0),
                "gto_score": float(sel.get("gto_score") or 0),
                "hand_eq": float(sol.get("hand_eq") or 0) or None,
                "pot_bb": float(rg.get("pot") or 0) or None,
                "gametype": gametype,
                "confidence": 1.0,
                "approx_flags": flags,
                "excluded": excluded,
                "played_at": list_row["played_at"],
            })
            hero_count[street] = idx + 1

        # record the action AFTER the decision so "before hero" lists are correct
        if street == "preflop":
            preflop_tokens.append(code)
        else:
            street_actions[street].append({"position": pos, "action": code})

    return hand_row, decisions
