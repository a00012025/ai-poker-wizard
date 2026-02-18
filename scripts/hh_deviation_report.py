#!/usr/bin/env python3
"""Analyze hand history deviations and format reports.

Reusable module for both CLI and Telegram bot.
"""

import time
from typing import Callable

from hh_deviation_check import check_hand
from gto_formatter import normalize_hand_name


def analyze_hands(
    hands: list[dict],
    delay: float = 0.3,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Run GTO deviation check on a list of parsed hands.

    Args:
        hands: list of parsed hand dicts from hh_parser
        delay: seconds between API calls
        on_progress: callback(current, total, status_msg) for progress updates

    Returns list of result dicts, each with:
        hand_id, hero_position, hero_hand, hero_hand_normalized, effective_bb,
        num_players, preflop_actions, spots_checked, deviations, elapsed_s
    """
    results = []

    for i, hand in enumerate(hands):
        hand_id = hand.get("hand_id", "?")
        hero_pos = hand["hero_position"]
        hero_hand = hand["hero_hand"]
        eff_bb = hand["effective_bb"]

        if on_progress:
            on_progress(i + 1, len(hands), f"{hero_pos} {hero_hand} ({eff_bb:.0f}bb)")

        t0 = time.time()
        try:
            devs = check_hand(hand)
            elapsed = time.time() - t0

            results.append({
                "hand_id": hand_id,
                "file": hand.get("file", ""),
                "hero_position": hero_pos,
                "hero_hand": hero_hand,
                "hero_hand_normalized": normalize_hand_name(hero_hand),
                "effective_bb": eff_bb,
                "num_players": hand.get("num_players", 8),
                "preflop_actions": hand["preflop_actions"],
                "spots_checked": len(devs),
                "deviations": devs,
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "hand_id": hand_id,
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


def format_deviation_report(results: list[dict], threshold_pct: float = 10,
                            ev_threshold: float = 1.0) -> str:
    """Format deviation results into a Telegram-friendly text report.

    Args:
        results: list of result dicts from analyze_hands()
        threshold_pct: minimum deviation % to report (default 10 = flag if hero_freq < 90%)
        ev_threshold: EV below this (in bb) is considered marginal (default 1.0bb)

    Returns formatted report string.
    """
    total_hands = len(results)
    hands_with_action = sum(1 for r in results if r["spots_checked"] > 0)
    errors = sum(1 for r in results if "error" in r)

    # Collect significant deviations
    severe = []   # hero_freq == 0 (GTO never does this)
    major = []    # hero_freq < 25%
    moderate = [] # hero_freq < threshold cutoff
    marginal = [] # EV below threshold — deviation barely matters

    cutoff = (100 - threshold_pct) / 100  # e.g., 0.9

    for r in results:
        for d in r.get("deviations", []):
            if d["hero_action"] == d["gto_action"]:
                continue
            if d["hero_freq"] >= cutoff:
                continue

            hero_ev = d.get("hero_ev")
            entry = {
                "hand_id": r["hand_id"],
                "pos": r["hero_position"],
                "hand": r["hero_hand_normalized"],
                "raw_hand": r["hero_hand"],
                "ebb": r["effective_bb"],
                "np": r["num_players"],
                "pf": r["preflop_actions"],
                "street": d["street"],
                "spot": d["spot"],
                "hero_action": d["hero_action_label"],
                "hero_freq": d["hero_freq"],
                "gto_action": d["gto_action_label"],
                "gto_freq": d["gto_freq"],
                "all_freqs": d["all_freqs"],
                "hero_ev": hero_ev,
            }

            # If EV is available and below threshold, classify as marginal
            if hero_ev is not None and abs(hero_ev) < ev_threshold:
                marginal.append(entry)
            elif d["hero_freq"] < 0.005:
                severe.append(entry)
            elif d["hero_freq"] < 0.25:
                major.append(entry)
            else:
                moderate.append(entry)

    severe.sort(key=lambda x: x["hero_freq"])
    major.sort(key=lambda x: x["hero_freq"])
    moderate.sort(key=lambda x: x["hero_freq"])
    marginal.sort(key=lambda x: abs(x.get("hero_ev") or 0))

    total_devs = len(severe) + len(major) + len(moderate)

    lines = []
    lines.append("*GTO 偏差分析報告*")
    devs_label = f"{total_devs} 處偏差"
    if marginal:
        devs_label += f"，{len(marginal)} 處微小偏差"
    lines.append(f"解析 {total_hands} 手，{hands_with_action} 手有行動，{devs_label}")
    if errors:
        lines.append(f"({errors} 手分析失敗)")
    lines.append("")

    if not total_devs and not marginal:
        lines.append("沒有發現顯著偏差，打得不錯！")
        return "\n".join(lines)

    def _format_entry(e: dict, show_ev: bool = False) -> str:
        freq_str = f"{e['hero_freq']:.0%}" if e['hero_freq'] >= 0.005 else "0%"
        hand_id = e["hand_id"]
        parts = []
        ev_note = ""
        if show_ev and e.get("hero_ev") is not None:
            ev_note = f" (EV {e['hero_ev']:.2f}bb)"
        parts.append(
            f"• `{hand_id}` {e['pos']} {e['hand']} {e['ebb']:.0f}bb"
            f" — {e['hero_action']} ({freq_str})"
            f" → 應 {e['gto_action']} ({e['gto_freq']:.0%}){ev_note}"
        )
        # Action line
        if e["street"] == "preflop":
            parts.append(f"  preflop: {e['pf']}")
        else:
            parts.append(f"  {e['spot']} | preflop: {e['pf']}")
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

    if moderate:
        lines.append(f"*中等偏差 — {len(moderate)} 處*")
        for e in moderate:
            lines.append(_format_entry(e))
        lines.append("")

    if marginal:
        lines.append(f"*微小偏差（EV < {ev_threshold:.0f}bb）— {len(marginal)} 處*")
        lines.append("以下偏差 EV 影響極小，可忽略：")
        for e in marginal:
            lines.append(_format_entry(e, show_ev=True))
        lines.append("")

    if not total_devs and marginal:
        lines.insert(4, "沒有顯著偏差，只有微小 EV 差異。")

    return "\n".join(lines)
