#!/usr/bin/env python3
"""Batch-test hand history files against analyze_hand_full().

Usage:
    python scripts/hh_batch_test.py 2026-02-17/
    python scripts/hh_batch_test.py 2026-02-17/ --output results.json
    python scripts/hh_batch_test.py 2026-02-17/ --limit 5
"""

import json
import time
import traceback
from pathlib import Path

from hh_parser import parse_directory, parse_file
from analyze_hand import analyze_hand_full


def run_batch(hands: list[dict], delay: float = 0.5) -> list[dict]:
    """Run analyze_hand_full() on each hand, collecting results.

    Args:
        hands: List of parsed hand dicts from hh_parser
        delay: Seconds to wait between API calls
    """
    results = []
    total = len(hands)

    for i, hand in enumerate(hands):
        hand_id = hand.get("hand_id", "unknown")
        hero_pos = hand["hero_position"]
        hero_hand = hand["hero_hand"]
        eff_bb = hand["effective_bb"]
        num_streets = len(hand.get("streets", []))

        print(f"[{i+1}/{total}] {hand_id}: {hero_pos} {hero_hand} ({eff_bb:.1f}bb) "
              f"streets={num_streets}", end=" ... ", flush=True)

        result = {
            "hand_id": hand_id,
            "file": hand.get("file", ""),
            "hero_position": hero_pos,
            "hero_hand": hero_hand,
            "effective_bb": eff_bb,
            "table_size": hand.get("table_size", 8),
            "num_players": hand.get("num_players", 8),
            "preflop_actions": hand["preflop_actions"],
            "num_streets": num_streets,
        }

        t0 = time.time()
        try:
            # Remove parser metadata, keep only analyze_hand_full() input keys
            analysis_input = {
                "gametype": hand["gametype"],
                "effective_bb": hand["effective_bb"],
                "hero_position": hand["hero_position"],
                "hero_hand": hand["hero_hand"],
                "preflop_actions": hand["preflop_actions"],
            }
            if "streets" in hand:
                analysis_input["streets"] = hand["streets"]

            analysis = analyze_hand_full(analysis_input)
            elapsed = time.time() - t0

            text = analysis.get("text", "")
            solutions = analysis.get("solutions", [])
            hero_spots = analysis.get("hero_spots", [])

            has_any_solution = any(s is not None for s in solutions)
            has_all_solutions = all(s is not None for s in solutions) if solutions else False
            missing_count = sum(1 for s in solutions if s is None)

            if has_all_solutions:
                status = "success"
            elif has_any_solution:
                status = "partial"
            elif solutions:
                status = "no_data"
            else:
                status = "no_data"

            result["status"] = status
            result["analysis_text"] = text
            result["error"] = None
            result["has_solver_data"] = has_any_solution
            result["spots_total"] = len(hero_spots)
            result["spots_with_data"] = len(solutions) - missing_count
            result["elapsed_s"] = round(elapsed, 1)

            streets_analyzed = set()
            for spot, sol in zip(hero_spots, solutions):
                if sol is not None:
                    streets_analyzed.add(spot.get("street", ""))
            result["streets_analyzed"] = sorted(streets_analyzed)

            print(f"{status} ({elapsed:.1f}s, {len(solutions)-missing_count}/{len(solutions)} spots)")

        except Exception as e:
            elapsed = time.time() - t0
            result["status"] = "api_error"
            result["analysis_text"] = None
            result["error"] = f"{type(e).__name__}: {e}"
            result["has_solver_data"] = False
            result["elapsed_s"] = round(elapsed, 1)
            result["streets_analyzed"] = []
            tb = traceback.format_exc()
            print(f"ERROR ({elapsed:.1f}s): {e}")
            # Print traceback for debugging
            for line in tb.strip().split("\n")[-3:]:
                print(f"    {line}")

        results.append(result)

        # Rate limit between hands
        if i < total - 1:
            time.sleep(delay)

    return results


def generate_report(results: list[dict], all_hand_count: int) -> str:
    """Generate a summary report from batch results."""
    lines = []
    lines.append("=" * 70)
    lines.append("MTT Hand History Batch Analysis Report")
    lines.append("=" * 70)

    # Overall stats
    total = len(results)
    by_status = {}
    for r in results:
        s = r["status"]
        by_status[s] = by_status.get(s, 0) + 1

    lines.append(f"\nTotal hands in files:  ~{all_hand_count}")
    lines.append(f"Hero-played hands:     {total}")
    lines.append(f"\nResults by status:")
    for status in ["success", "partial", "no_data", "api_error", "parse_error"]:
        count = by_status.get(status, 0)
        pct = count / total * 100 if total else 0
        lines.append(f"  {status:12s}: {count:3d} ({pct:.1f}%)")

    # Success rate
    success_count = by_status.get("success", 0) + by_status.get("partial", 0)
    lines.append(f"\nOverall success rate:  {success_count}/{total} ({success_count/total*100:.1f}%)")

    # Breakdown by position
    lines.append("\n--- By Position ---")
    pos_stats = {}
    for r in results:
        pos = r["hero_position"]
        if pos not in pos_stats:
            pos_stats[pos] = {"total": 0, "success": 0, "partial": 0}
        pos_stats[pos]["total"] += 1
        if r["status"] == "success":
            pos_stats[pos]["success"] += 1
        elif r["status"] == "partial":
            pos_stats[pos]["partial"] += 1

    for pos in ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]:
        if pos in pos_stats:
            s = pos_stats[pos]
            ok = s["success"] + s["partial"]
            lines.append(f"  {pos:6s}: {ok}/{s['total']} success")

    # Breakdown by effective BB range
    lines.append("\n--- By Stack Depth ---")
    depth_buckets = {"0-10bb": [], "10-20bb": [], "20-40bb": [], "40-80bb": [], "80+bb": []}
    for r in results:
        ebb = r["effective_bb"]
        if ebb <= 10:
            depth_buckets["0-10bb"].append(r)
        elif ebb <= 20:
            depth_buckets["10-20bb"].append(r)
        elif ebb <= 40:
            depth_buckets["20-40bb"].append(r)
        elif ebb <= 80:
            depth_buckets["40-80bb"].append(r)
        else:
            depth_buckets["80+bb"].append(r)

    for label, bucket in depth_buckets.items():
        if bucket:
            ok = sum(1 for r in bucket if r["status"] in ("success", "partial"))
            lines.append(f"  {label:8s}: {ok}/{len(bucket)} success")

    # Failures detail
    failures = [r for r in results if r["status"] not in ("success", "partial")]
    if failures:
        lines.append(f"\n--- Failures ({len(failures)}) ---")
        for r in failures:
            lines.append(f"  {r['hand_id']}: {r['hero_position']} {r['hero_hand']} "
                        f"({r['effective_bb']:.1f}bb) pf={r['preflop_actions']}")
            if r.get("error"):
                lines.append(f"    Error: {r['error']}")
            lines.append(f"    Status: {r['status']}")

    # Partial successes detail
    partials = [r for r in results if r["status"] == "partial"]
    if partials:
        lines.append(f"\n--- Partial Results ({len(partials)}) ---")
        for r in partials:
            lines.append(f"  {r['hand_id']}: {r['hero_position']} {r['hero_hand']} "
                        f"({r['effective_bb']:.1f}bb) "
                        f"spots={r.get('spots_with_data',0)}/{r.get('spots_total',0)} "
                        f"streets={r.get('streets_analyzed', [])}")

    # Timing
    total_time = sum(r.get("elapsed_s", 0) for r in results)
    avg_time = total_time / total if total else 0
    lines.append(f"\n--- Timing ---")
    lines.append(f"  Total:   {total_time:.0f}s")
    lines.append(f"  Average: {avg_time:.1f}s per hand")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch test HH files against GTO analysis")
    parser.add_argument("path", help="Directory or file with hand histories")
    parser.add_argument("--output", "-o", default="hh_batch_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--limit", "-n", type=int, default=0,
                       help="Limit number of hands to analyze (0 = all)")
    parser.add_argument("--delay", "-d", type=float, default=0.5,
                       help="Delay between hands in seconds")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        hands = parse_directory(path)
        # Count total hands for report
        all_count = 0
        for f in path.glob("*.txt"):
            all_count += f.read_text().count("Poker Hand #")
    else:
        hands = parse_file(path)
        all_count = path.read_text().count("Poker Hand #")

    print(f"Parsed {len(hands)} hero-played hands from {all_count} total hands\n")

    if args.limit:
        hands = hands[:args.limit]
        print(f"Limiting to first {args.limit} hands\n")

    results = run_batch(hands, delay=args.delay)

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")

    # Generate and print report
    report = generate_report(results, all_count)
    print(f"\n{report}")

    # Also save report as text
    report_path = output_path.with_suffix(".txt")
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
