#!/usr/bin/env python3
"""Analyze hand history deviations and format reports.

Reusable module for both CLI and Telegram bot.
"""

import time
from typing import Callable

from hh_deviation_check import check_hand
from gto_formatter import normalize_hand_name


_PHASE_LABELS = {
    "START": "起始",
    "PCT75": "75%",
    "PCT50": "50%",
    "PCT37": "37%",
    "PCT25": "25%",
    "PCT10": "10%",
    "PCT5": "5%",
    "BUBBLEEARLY": "泡沫前期",
    "BUBBLEMID": "泡沫中期",
    "BUBBLELATE": "泡沫後期",
    "FT": "決賽桌",
    "T2": "兩桌",
    "T3": "三桌",
}


def _extract_icm_phase_label(gametype: str) -> str:
    """Extract human-readable ICM phase from gametype like 'MTTGeneral_ICM8m1000PTPCT25'."""
    import re
    m = re.search(r"PT(.+)$", gametype)
    if not m:
        return "ICM"
    raw = m.group(1)
    # Handle bubble variants like "BUBBLE152PT" → "BUBBLELATE"
    # The raw value after PT is the phase code
    return _PHASE_LABELS.get(raw, raw)


def analyze_hands(
    hands: list[dict],
    delay: float = 0.3,
    on_progress: Callable[[int, int, str], None] | None = None,
    starting_stack: int = 0,
    tournament_size: int = 1000,
) -> list[dict]:
    """Run GTO deviation check on a list of parsed hands.

    Args:
        hands: list of parsed hand dicts from hh_parser
        delay: seconds between API calls
        on_progress: callback(current, total, status_msg) for progress updates
        starting_stack: tournament starting stack in chips (0 = chip EV only)
        tournament_size: 1000 or 200

    Returns list of result dicts, each with:
        hand_id, hero_position, hero_hand, hero_hand_normalized, effective_bb,
        num_players, preflop_actions, spots_checked, deviations, elapsed_s
    """
    results = []

    # Auto-detect starting_stack per tournament from earliest hand's hero chips
    # GGPoker HH files list newest first, so the last hand per tournament is the earliest
    starting_stack_by_tournament: dict[str, int] = {}
    if starting_stack == 0:
        for hand in reversed(hands):
            tid = hand.get("tournament_id", "")
            if tid and tid not in starting_stack_by_tournament and "hero_chips" in hand:
                starting_stack_by_tournament[tid] = hand["hero_chips"]

    # Track max ratio per tournament for monotonicity
    # (ratio only increases as tournament progresses → fewer players remain)
    max_ratio_by_tournament: dict[str, float] = {}

    for i, hand in enumerate(hands):
        hand_id = hand.get("hand_id", "?")
        hero_pos = hand["hero_position"]
        hero_hand = hand["hero_hand"]
        eff_bb = hand["effective_bb"]

        if on_progress:
            on_progress(i + 1, len(hands), f"{hero_pos} {hero_hand} ({eff_bb:.0f}bb)")

        # Compute ICM params with monotonicity enforcement
        icm_params = None
        tid = hand.get("tournament_id", "")
        effective_starting_stack = starting_stack or starting_stack_by_tournament.get(tid, 0)
        if effective_starting_stack > 0 and "avg_stack_chips" in hand and "stacks_bb" in hand:
            from icm_modes import infer_icm_phase, find_icm_params

            raw_ratio = hand["avg_stack_chips"] / effective_starting_stack

            # Enforce monotonicity: ratio can only increase (remaining decreases)
            if tid:
                prev_max = max_ratio_by_tournament.get(tid, 0)
                ratio = max(raw_ratio, prev_max)
                max_ratio_by_tournament[tid] = ratio
            else:
                ratio = raw_ratio

            # Use table_size (8) for ICM lookup during mid-tournament;
            # actual num_players only matters at final table.
            table_size = hand.get("table_size", 8)
            num_players = hand.get("num_players", table_size)
            estimated_remaining = tournament_size / ratio
            estimated_remaining = max(table_size, estimated_remaining)
            estimated_remaining = min(tournament_size, estimated_remaining)

            # Pad stacks to table_size if short-handed (mid-tournament reseating)
            stacks = list(hand["stacks_bb"])
            if len(stacks) < table_size:
                avg_bb = sum(stacks) / len(stacks) if stacks else 20
                stacks.extend([avg_bb] * (table_size - len(stacks)))

            icm_result = find_icm_params(
                player_stacks=stacks,
                tournament_size=tournament_size,
                players_remaining=int(round(estimated_remaining)),
            )
            if icm_result["gametype"] != "MTTGeneral":
                icm_params = icm_result

        # Extract ICM phase label for reporting
        icm_phase_label = ""
        if icm_params:
            icm_phase_label = _extract_icm_phase_label(icm_params["gametype"])

        t0 = time.time()
        try:
            devs = check_hand(hand, icm_params=icm_params)
            elapsed = time.time() - t0

            results.append({
                "hand_id": hand_id,
                "tournament_id": hand.get("tournament_id", ""),
                "file": hand.get("file", ""),
                "hero_position": hero_pos,
                "hero_hand": hero_hand,
                "hero_hand_normalized": normalize_hand_name(hero_hand),
                "effective_bb": eff_bb,
                "num_players": hand.get("num_players", 8),
                "preflop_actions": hand["preflop_actions"],
                "icm_phase": icm_phase_label,
                "spots_checked": len(devs),
                "deviations": devs,
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "hand_id": hand_id,
                "tournament_id": hand.get("tournament_id", ""),
                "file": hand.get("file", ""),
                "hero_position": hero_pos,
                "hero_hand": hero_hand,
                "hero_hand_normalized": normalize_hand_name(hero_hand),
                "effective_bb": eff_bb,
                "num_players": hand.get("num_players", 8),
                "preflop_actions": hand["preflop_actions"],
                "spots_checked": 0,
                "deviations": [],
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
            })

        if i < len(hands) - 1 and delay > 0:
            time.sleep(delay)

    return results


def format_deviation_report(results: list[dict], threshold_pct: float = 10) -> str:
    """Format deviation results into a Telegram-friendly text report.

    Args:
        results: list of result dicts from analyze_hands()
        threshold_pct: minimum deviation % to report (default 10 = flag if hero_freq < 90%)

    Returns formatted report string.
    """
    import re as _re
    from collections import Counter

    total_hands = len(results)
    hands_with_action = sum(1 for r in results if r["spots_checked"] > 0)
    errors = sum(1 for r in results if "error" in r)

    # Build tournament ID → short label mapping
    tournament_map: dict[str, str] = {}  # tid -> display name
    tid_short: dict[str, str] = {}       # tid -> short label like T1, T2
    for r in results:
        tid = r.get("tournament_id", "")
        if tid and tid not in tournament_map:
            fname = r.get("file", "")
            # Extract name after "GG... - " prefix, strip .txt
            m = _re.search(r" - (.+?)\.txt$", fname)
            tournament_map[tid] = m.group(1) if m else fname

    # Assign short labels (T1, T2, ...) in order of appearance
    for idx, tid in enumerate(tournament_map, 1):
        tid_short[tid] = f"T{idx}"

    # Collect significant deviations (skip moderate — hero_freq >= 25%)
    severe = []   # hero_freq == 0 (GTO never does this)
    major = []    # hero_freq < 25%

    cutoff = (100 - threshold_pct) / 100  # e.g., 0.9

    for r in results:
        for d in r.get("deviations", []):
            if d["hero_action"] == d["gto_action"]:
                continue
            if d["hero_freq"] >= cutoff:
                continue

            entry = {
                "hand_id": r["hand_id"],
                "tid_label": tid_short.get(r.get("tournament_id", ""), ""),
                "pos": r["hero_position"],
                "hand": r["hero_hand_normalized"],
                "raw_hand": r["hero_hand"],
                "ebb": r["effective_bb"],
                "np": r["num_players"],
                "pf": r["preflop_actions"],
                "icm_phase": r.get("icm_phase", ""),
                "street": d["street"],
                "spot": d["spot"],
                "hero_action": d["hero_action_label"],
                "hero_freq": d["hero_freq"],
                "gto_action": d["gto_action_label"],
                "gto_freq": d["gto_freq"],
                "all_freqs": d["all_freqs"],
                "ev_loss": d.get("ev_loss"),
            }

            if d["hero_freq"] < 0.005:
                severe.append(entry)
            elif d["hero_freq"] < 0.25:
                major.append(entry)

    severe.sort(key=lambda x: (-(x.get("ev_loss") or 0), x["hero_freq"]))
    major.sort(key=lambda x: (-(x.get("ev_loss") or 0), x["hero_freq"]))

    total_devs = len(severe) + len(major)
    total_ev_loss = sum(e.get("ev_loss") or 0 for e in severe + major)

    # Check if ICM mode was used
    icm_count = sum(1 for r in results if r.get("icm_phase"))
    icm_mode_str = ""
    if icm_count > 0:
        # Show the most common phase
        phases = Counter(r.get("icm_phase", "") for r in results if r.get("icm_phase"))
        phase_summary = "、".join(f"{p}({c}手)" for p, c in phases.most_common(3))
        icm_mode_str = f"\n📊 ICM 模式：{phase_summary}"

    lines = []
    lines.append("*GTO 偏差分析報告*")
    ev_loss_str = f"，累計 EV 損失 {total_ev_loss:.1f}bb" if total_ev_loss > 0.005 else ""
    lines.append(f"解析 {total_hands} 手，{hands_with_action} 手有行動，{total_devs} 處偏差{ev_loss_str}")
    if icm_mode_str:
        lines.append(icm_mode_str)
    if errors:
        lines.append(f"({errors} 手分析失敗)")

    # Tournament legend
    if len(tournament_map) > 1:
        lines.append("")
        for tid, name in tournament_map.items():
            lines.append(f"{tid_short[tid]}: {name}")
    elif len(tournament_map) == 1:
        tid, name = next(iter(tournament_map.items()))
        lines.append(f"錦標賽：{name}")

    lines.append("")

    if not total_devs:
        lines.append("沒有發現顯著偏差，打得不錯！")
        return "\n".join(lines)

    def _fmt_num(v: float) -> str:
        """Format number, stripping unnecessary trailing zeros."""
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s

    def _format_entry(e: dict) -> str:
        freq_str = f"{e['hero_freq']:.0%}" if e['hero_freq'] >= 0.005 else "0%"
        hand_id = e["hand_id"]
        tid_label = e.get("tid_label", "")
        tid_prefix = f"[{tid_label}] " if tid_label and len(tournament_map) > 1 else ""
        street_name = e["street"].capitalize()
        ebb = _fmt_num(e["ebb"])
        icm_tag = f" [{e['icm_phase']}]" if e.get("icm_phase") and e["street"] == "preflop" else ""
        ev_loss = e.get("ev_loss")
        ev_loss_tag = f" [-{ev_loss:.2f}bb]" if ev_loss is not None and ev_loss > 0.005 else ""
        # Show raw hand (with suits) for postflop to distinguish combos
        display_hand = e.get("raw_hand", e["hand"]) if e["street"] != "preflop" else e["hand"]
        parts = []
        parts.append(
            f"• {tid_prefix}`{hand_id}` {e['pos']} {display_hand} {ebb}bb"
            f" — {street_name}{icm_tag} {e['hero_action']} ({freq_str}){ev_loss_tag}"
        )
        parts.append(
            f"    建議：應 {e['gto_action']} ({e['gto_freq']:.0%})"
        )
        # Action line
        if e["street"] == "preflop":
            parts.append(f"    preflop: {e['pf']}")
        else:
            parts.append(f"    {e['spot']} | preflop: {e['pf']}")
        return "\n".join(parts)

    if severe:
        lines.append(f"*嚴重偏差（GTO 0%）— {len(severe)} 處*")
        for e in severe:
            lines.append(_format_entry(e))
        lines.append("")

    if major:
        lines.append(f"*較大偏差（GTO < 25%）— {len(major)} 處*")
        for e in major:
            lines.append(_format_entry(e))
        lines.append("")

    return "\n".join(lines)
