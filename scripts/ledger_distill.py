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
MIN_STATS_CONFIDENCE = 0.8
LIST_ONLY_SOLVED_STATUSES = frozenset({"OK"})
ZERO_LOSS_THRESHOLD_BB = 0.0


class ListOnlyReconstructionError(ValueError):
    """The list row is not rich enough to reconstruct decisions faithfully."""


def depth_band(bb: float) -> str:
    if bb < 15: return "le15"
    if bb < 25: return "15_25"
    if bb < 40: return "25_40"
    return "40plus"


def decode_gtow_depth(value) -> float | None:
    """Decode GTOW's canonical ``bb + 0.125`` tree identifier to human bb.

    Decision-local effective depths can also be non-canonical decimals (for
    example 34.692); those are already real bb and must remain untouched.
    """
    try:
        depth = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if depth is None:
        return None
    whole = round(depth - 0.125)
    if abs(depth - (whole + 0.125)) < 1e-6:
        return float(whole)
    return depth


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


def _hand_flags(list_row: dict, ga: dict) -> list[str]:
    """Return hand-level audit flags without poisoning solved hero nodes.

    GTOW can label a hand ``NO_GTO_SOLUTION`` when only part of its action
    path is unsupported while still returning a fully solved hero game point.
    Eligibility is therefore decided per game point in ``distill_hand``; these
    hand-level statuses remain as provenance only.
    """
    flags = []
    ss = list_row.get("solution_status")
    if ss and ss != "OK":
        flags.append(f"solution:{ss}")
    ws = ga.get("warning_status")
    if ws and ws != "OK":
        flags.append(f"warning:{ws}")
    ar = ga.get("approximation_reason")
    if ar:
        flags.extend(["analyzer_approximation", f"approx:{ar}"])
    return flags


def _decision_depths(played_depth: float, gp: dict) -> tuple[float | None, float]:
    """Return (authoritative solver depth, depth used for classification).

    GTOW list rows describe the player's physical/preflop stack.  A game point
    carries the binding effective stack of the solution that actually graded
    the decision.  Those are legitimately different in covered/multiway pots;
    drills must follow the latter while retaining the former for audit.
    """
    solver_depth = decode_gtow_depth(gp.get("depth"))
    return solver_depth, solver_depth or played_depth


def _decision_confidence(flags: list[str], excluded: bool) -> float:
    """Eligibility tier for aggregate training statistics (not a probability).

    GTOW's selected action/EV is authoritative at its game point.  A physical
    stack difference alone is therefore not uncertainty.  Missing decision
    depth or a large size snap changes the represented node and is gated out;
    a declared Analyzer approximation remains visible at the confidence floor.
    """
    if excluded:
        return 0.0
    if "missing_solver_depth" in flags:
        return 0.6
    if "sizing_snap" in flags:
        return 0.75
    if any(f.startswith("approx:") for f in flags):
        return MIN_STATS_CONFIDENCE
    return 1.0


def distill_hand_row(list_row: dict) -> dict:
    """Fields shared by list-only and full-detail hand ingestion."""
    hero_pos = list_row.get("player_position", "")
    boards = (list_row.get("boards") or [""])[0]
    played_depth = float(list_row.get("preflop_game_depth") or 0)
    return {
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
        "preflop_depth_bb": played_depth,
        "total_ev_loss_bb": float(list_row.get("total_ev_loss") or 0),
        "total_ev_loss_pct_pot": float(list_row.get("total_ev_loss_as_pot") or 0),
        "avg_gto_score": float(list_row["avg_gto_score"]) if list_row.get("avg_gto_score") is not None else None,
        "winloss_bb": float(list_row["player_winloss"]) if list_row.get("player_winloss") is not None else None,
        "hand_correctness": list_row.get("hand_correctness"),
        "solution_status": list_row.get("solution_status"),
    }


def should_skip_zeroloss_detail(list_row: dict) -> bool:
    """Only solved, exactly-zero-loss hands are eligible for list-only ingest."""
    if list_row.get("solution_status") not in LIST_ONLY_SOLVED_STATUSES:
        return False
    try:
        return float(list_row.get("total_ev_loss")) == ZERO_LOSS_THRESHOLD_BB
    except (TypeError, ValueError):
        return False


def _list_actions(list_row: dict, street: str, *, required: bool) -> list[dict]:
    actions = list_row.get(f"actions_with_correctness_{street}")
    if not actions:
        if required:
            raise ListOnlyReconstructionError(f"missing {street} action array")
        return []
    if not isinstance(actions, list):
        raise ListOnlyReconstructionError(f"invalid {street} action array")
    for action in actions:
        if not isinstance(action, dict) or not action.get("action_code"):
            raise ListOnlyReconstructionError(f"blank {street} action")
    return actions


def _hero_code_is_exact(action_code: str) -> bool:
    """List numeric bet/raise sizes are rounded or generic versus detail.

    Those codes are structurally sufficient for taxonomy, but the design's
    fidelity tuple requires an exact taken_code. Fail closed and fetch detail.
    """
    return (action_code or "").upper() in {"F", "C", "X", "AI", "RAI"}


def _list_row_to_parsed(list_row: dict) -> tuple[dict, dict[tuple[str, int], dict]]:
    """Build walk_spots_from_parsed input and keyed hero-grade metadata."""
    from spot_taxonomy import _PREFLOP_ORDER, _postflop_rank, _preflop_seat_tokens

    hero = list_row.get("player_position") or ""
    npl = list_row.get("total_players")
    if npl not in _PREFLOP_ORDER or hero not in _PREFLOP_ORDER[npl]:
        raise ListOnlyReconstructionError("invalid table size or hero position")

    preflop = _list_actions(list_row, "preflop", required=True)
    tokens = [_norm_code(a["action_code"]) for a in preflop]
    seat_tokens = _preflop_seat_tokens(tokens, npl)
    if len(seat_tokens) != len(preflop):
        raise ListOnlyReconstructionError("preflop seat attribution incomplete")

    hero_meta: dict[tuple[str, int], dict] = {}
    hero_counts = {street: 0 for street in STREET_ORDER}

    def check_attribution(street: str, positioned: list[tuple[str, dict]]) -> None:
        for pos, action in positioned:
            graded = action.get("correctness") is not None
            if graded != (pos == hero):
                raise ListOnlyReconstructionError(
                    f"{street} hero correctness does not match reconstructed position")
            if graded:
                raw_code = action["action_code"]
                if action.get("correctness") == "UNSOLVED":
                    raise ListOnlyReconstructionError(f"{street} hero action is unsolved")
                if not _hero_code_is_exact(raw_code):
                    raise ListOnlyReconstructionError(
                        f"{street} hero action code is not detail-exact: {raw_code}")
                idx = hero_counts[street]
                hero_meta[(street, idx)] = {
                    "taken_code": _norm_code(raw_code),
                    "correctness": action.get("correctness"),
                }
                hero_counts[street] = idx + 1

    check_attribution("preflop", list(zip((p for p, _ in seat_tokens), preflop)))

    order = _PREFLOP_ORDER[npl]
    active = set(order)
    saw_preflop_allin = False
    for pos, code in seat_tokens:
        if code == "F":
            active.discard(pos)
        elif code == "AI":
            saw_preflop_allin = True

    boards = (list_row.get("boards") or [""])[0] or ""
    board_lengths = {"flop": 6, "turn": 8, "river": 10}
    streets = []
    for street in STREET_ORDER[1:]:
        has_board = len(boards) >= board_lengths[street]
        entries = _list_actions(list_row, street, required=has_board)
        if entries and not has_board:
            raise ListOnlyReconstructionError(f"{street} actions without board")
        if not entries:
            continue
        if saw_preflop_allin:
            raise ListOnlyReconstructionError("postflop after preflop all-in is ambiguous")
        if len(active) != 2:
            raise ListOnlyReconstructionError("multiway postflop position reconstruction")
        if npl == 2 and active == {"SB", "BB"}:
            street_order = ["BB", "SB"]
        else:
            ranked = [(p, _postflop_rank(p)) for p in active]
            if any(rank is None for _, rank in ranked):
                raise ListOnlyReconstructionError("unknown postflop position")
            street_order = [p for p, _ in sorted(ranked, key=lambda item: item[1])]

        positioned = []
        actions = []
        for index, entry in enumerate(entries):
            raw_code = entry["action_code"]
            code = _norm_code(raw_code)
            if code == "AI":
                raise ListOnlyReconstructionError("postflop all-in attribution is ambiguous")
            pos = street_order[index % len(street_order)]
            positioned.append((pos, entry))
            actions.append({"position": pos, "action": code})
            if code == "F":
                if index != len(entries) - 1:
                    raise ListOnlyReconstructionError("action follows a postflop fold")
                active.discard(pos)
        check_attribution(street, positioned)
        streets.append({"board": boards[:board_lengths[street]], "actions": actions})

    if not hero_meta:
        raise ListOnlyReconstructionError("no graded hero decisions")
    return {
        "hero_position": hero,
        "players_at_table": npl,
        "preflop_actions": "-".join(tokens),
        "effective_bb": float(list_row.get("preflop_game_depth") or 0),
        "streets": streets,
    }, hero_meta


def distill_hand_from_list(list_row: dict) -> tuple[dict, list[dict]]:
    """Distill a solved zero-loss list row without fetching hand detail.

    Raises ListOnlyReconstructionError whenever the list row cannot prove the
    same hero-decision structure the detail path would provide.
    """
    if not should_skip_zeroloss_detail(list_row):
        raise ListOnlyReconstructionError("hand is not solved exact-zero-loss")

    from spot_categorizer import compute_pot_type_from_preflop
    from spot_taxonomy import norm_pot_type, walk_spots_from_parsed

    parsed, hero_meta = _list_row_to_parsed(list_row)
    if parsed["streets"]:
        parsed_pot_type = compute_pot_type_from_preflop(
            parsed["preflop_actions"], parsed["players_at_table"])
        if norm_pot_type(parsed_pot_type) != norm_pot_type(list_row.get("pot_type")):
            raise ListOnlyReconstructionError(
                f"pot type mismatch: parsed={parsed_pot_type} list={list_row.get('pot_type')}")
    spots = list(walk_spots_from_parsed(parsed))
    if len(spots) != len(hero_meta):
        raise ListOnlyReconstructionError("walker decision count mismatch")

    played_depth = float(list_row.get("preflop_game_depth") or 0)
    gtow_texture = "/".join(x for x in (
        list_row.get("board_flop_connectedness"),
        list_row.get("board_flop_pairedness"),
    ) if x) or None
    decisions = []
    for spot in spots:
        key = (spot["street"], spot["decision_idx"])
        meta = hero_meta.get(key)
        if meta is None or _norm_code(spot.get("hero_action_raw")) != meta["taken_code"]:
            raise ListOnlyReconstructionError(f"walker hero action mismatch at {key}")
        tags = spot.get("tags") or {}
        before = spot.get("acts_before") or []
        pot_type = spot.get("pot_type")
        if spot["street"] == "preflop":
            # Category is more reliable than the legacy helper for shove
            # tokens (that helper historically ignored AI when naming pots).
            pot_type = {
                "RFI": "unopened", "vsOpen": "SRP", "vsRaiseCall": "SRP",
                "vsSqueeze": "squeezed", "vs3bet": "3bet",
                "vsCold3bet": "3bet", "vs4bet": "4bet", "vsCold4bet": "4bet",
                "discarded": "limp",
            }.get(spot.get("category"))
            if pot_type is None:
                pot_type = compute_pot_type_from_preflop(
                    "-".join(action for _pos, action in before), parsed["players_at_table"])
        decisions.append({
            "gtow_hand_id": list_row["hand_id"],
            "street": spot["street"], "decision_idx": spot["decision_idx"],
            "source": "online", "grader": "gtow_list",
            "gtow_texture": gtow_texture,
            "depth_band": tags.get("depth_band") or depth_band(played_depth),
            "played_depth_bb": played_depth or None, "solver_depth_bb": None,
            "position": list_row.get("player_position"),
            "pot_type": pot_type, "facing": spot.get("facing") or "unopened",
            "taken_code": meta["taken_code"], "best_code": meta["taken_code"],
            "correctness": meta["correctness"],
            "ev_loss_bb": 0.0, "ev_loss_pct_pot": 0.0,
            "taken_freq": None, "freq_diff": None, "gto_score": None,
            "hand_eq": None, "pot_bb": None, "gametype": None,
            "confidence": 1.0, "approx_flags": ["list_only"],
            "excluded": False, "played_at": list_row["played_at"],
            "spot_category": spot.get("category"), "spot_leaf": spot.get("leaf"),
            "spot_parent": spot.get("parent"), "spot_keys": spot.get("keys"),
            "hero_cat": spot.get("hero_cat"), "villain_cat": spot.get("villain_cat"),
            "ip_oop": spot.get("ip_oop"), "flop_seq": spot.get("flop_seq"),
            "turn_seq": spot.get("turn_seq"), "eff_stack": tags.get("eff_stack"),
            "board_suit": tags.get("board_suit"),
            "board_conn": list_row.get("board_flop_connectedness") or None,
            "board_paired": list_row.get("board_flop_pairedness") or None,
            "discarded": bool(spot.get("discarded")),
            "limp_origin": bool(spot.get("limp_origin")),
        })
    return distill_hand_row(list_row), decisions


def distill_hand(list_row: dict, detail: dict) -> tuple[dict, list[dict]]:
    ga = detail.get("game_analysis") or {}
    gps = ga.get("game_points") or []
    hero_pos = list_row.get("player_position", "")
    boards = (list_row.get("boards") or [""])[0]
    played_depth = float(list_row.get("preflop_game_depth") or 0)
    hand_row = distill_hand_row(list_row)

    hand_flags = _hand_flags(list_row, ga)
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
            excluded = gp.get("has_solution") is not True
            if excluded:
                flags.append("node:no_solution")
            elif any(f.startswith(("solution:", "warning:")) for f in hand_flags):
                flags.append("node:solved_partial_hand")
            corr = sel.get("correctness")
            if corr in (None, "UNSOLVED"):
                flags.append("unsolved"); excluded = True
            solver_depth, decision_depth = _decision_depths(played_depth, gp)
            if solver_depth is None:
                flags.append("missing_solver_depth")
            elif played_depth and abs(played_depth - solver_depth) > DEPTH_GAP_BB:
                # Audit-only: often a legitimate binding effective opponent,
                # not an approximation and not a reason to reject the grade.
                flags.append("played_solver_depth_gap")
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
                "depth_band": depth_band(decision_depth),
                "played_depth_bb": played_depth or None,
                "solver_depth_bb": solver_depth,
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
                "confidence": _decision_confidence(flags, excluded),
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
