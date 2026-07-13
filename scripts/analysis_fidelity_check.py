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

from analyze_hand import analyze_hand_full
from gto_formatter import (
    _combo_idx_in_player_range,
    _get_action_strategy_frequencies,
    combo_index_for_hand,
    normalize_hand_name,
)
from gtow_analyze_api import hand_detail
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


def normalize_code(code: str | None) -> str:
    """Normalize action aliases without hiding meaningful sizing differences."""
    code = (code or "").strip()
    if code == "RAI":
        return "AI"
    if code.startswith("B"):
        return "R" + code[1:] if len(code) > 1 else "R"
    return code


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
    _truncate_after_hero_fold(hand, detail, hero_pos)
    return hand


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
                code = f"R{float(action.get('betsize') or 0):g}"
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
        available = (gp.get("analysis_solved") or {}).get("available_actions") or []
        selected = next((a for a in available if a.get("selected")), None)
        if selected is None:
            continue
        street_type = ((gp.get("real_game") or {}).get("current_street") or {}).get("type", "")
        street = STREET_FROM_GTOW.get(str(street_type).upper(), "preflop")
        idx = counts[street]
        counts[street] += 1
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
            "best_code": normalize_code((best.get("action") or {}).get("code")) if best else "",
            "acceptable_codes": sorted(c for c in acceptable if c),
            "correctness": selected.get("correctness"),
            "ev_loss_bb": _float(selected.get("ev_loss")),
            "taken_freq": _float(selected.get("frequency")),
            "pot_bb": _float(real_game.get("pot")),
            "has_solution": bool(gp.get("has_solution")),
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
        best_code = max(action_evs, key=action_evs.get) if action_evs else ""
        ev_loss = None
        if taken and taken in action_evs and best_code:
            ev_loss = max(0.0, action_evs[best_code] - action_evs[taken])
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
    if gd is None or od is None or abs(gd - od) > thresholds.depth_bb:
        diffs.append("depth")
    for field in ("board", "preflop_actions", "flop_actions", "turn_actions", "river_actions"):
        if (gtow.get(field) or "") != (own.get(field) or ""):
            diffs.append(field)
    return diffs


def compare_decisions(gtow: list[dict], own: list[dict],
                      thresholds: Thresholds = Thresholds(), *,
                      gtow_hand_unknown: bool = False) -> list[dict]:
    """Compare decisions without treating different solver nodes as EV failures."""
    gtow_by = {d["key"]: d for d in gtow}
    own_by = {d["key"]: d for d in own}
    rows: list[dict] = []
    for key in sorted(set(gtow_by) | set(own_by), key=_decision_sort_key):
        g, o = gtow_by.get(key), own_by.get(key)
        if g is None:
            status = "skipped_gtow_unknown" if gtow_hand_unknown else "extra_own_decision"
            rows.append({"key": key, "status": status, "gtow": None, "own": o})
            continue
        if o is None:
            status = "skipped_gtow_unknown" if g.get("gtow_excluded") else "missing_own_decision"
            rows.append({"key": key, "status": status, "gtow": g, "own": None})
            continue
        node_diffs = _node_differences(g, o, thresholds)
        taken_match = g.get("taken_code") == o.get("taken_code")
        acceptable = set(g.get("acceptable_codes") or [])
        ev_delta = None
        if g.get("ev_loss_bb") is not None and o.get("ev_loss_bb") is not None:
            ev_delta = o["ev_loss_bb"] - g["ev_loss_bb"]
        freq_delta = None
        if g.get("taken_freq") is not None and o.get("taken_freq") is not None:
            freq_delta = o["taken_freq"] - g["taken_freq"]
        best_compatible = (
            (bool(o.get("best_code")) and o.get("best_code") in acceptable)
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
            status = "own_combo_off_range"
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
    skipped = statuses.get("skipped_gtow_unknown", 0)
    comparable = total_decisions - skipped
    lines = [
        "# GTOW Analyzer vs analyze_hand fidelity",
        "",
        f"- hands: {len(results)}",
        f"- hand errors: {len(errors)}",
        f"- decisions: {total_decisions}",
        f"- GTOW-unknown decisions skipped: {skipped}",
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
