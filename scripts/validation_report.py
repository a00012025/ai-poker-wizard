#!/usr/bin/env python3
"""Validator backlog report — the parser-fix worklist.

The rules validator (scripts/hand_validator.py) is a deterministic function of a
stored parse, so we don't persist its output: we recompute it on demand over
every snapshot's parsed_json.  That makes this report **retroactive** (covers the
whole existing corpus) and **self-updating** (a hand drops off the moment a
parser fix is captured), which a stored column could not be.

Each flagged hand is one parser bug in the parse a user would get *today* (the
effective parse = corrected expected_json if present, else raw parsed_json — the
same thing analyze_hand_full sees at runtime).  Status:
  - UNREVIEWED      → only a raw parse exists; confirm the correct parse, fix, lock.
  - STALE_EXPECTED  → a reviewed expected_json EXISTS but itself breaks the rules
                      (the accepted snapshot is wrong — re-correct it first).

Usage:
    python scripts/validation_report.py                 # grouped summary + list
    python scripts/validation_report.py --code ORPHAN_CALL
    python scripts/validation_report.py --source image
    python scripts/validation_report.py --worklist      # ready-to-run fix commands
    python scripts/validation_report.py --json          # machine-readable (for skills)

Requires: SUPABASE_CONN.
"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from hand_validator import validate_hand  # noqa: E402


async def _scan() -> list[dict]:
    import asyncpg
    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        print("ERROR: SUPABASE_CONN not set", file=sys.stderr)
        sys.exit(2)
    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    try:
        rows = await conn.fetch(
            "SELECT hand_id, source_type, parsed_json, expected_json, is_regression "
            "FROM analysis_snapshots ORDER BY hand_id DESC")
    finally:
        await conn.close()

    findings: list[dict] = []
    for row in rows:
        # Effective parse = what a user gets today (expected when reviewed, else raw).
        raw = row["expected_json"] or row["parsed_json"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        rep = validate_hand(parsed)
        if rep.ok:
            continue
        codes = defaultdict(int)
        for i in rep.hard:
            codes[i.code] += 1
        findings.append({
            "hand_id": row["hand_id"],
            "source_type": row["source_type"] or "unknown",
            "codes": dict(codes),
            "status": "STALE_EXPECTED" if row["expected_json"] else "UNREVIEWED",
            "is_regression": bool(row["is_regression"]),
            "messages": [f"{i.code}@{i.street}: {i.message}" for i in rep.hard],
        })
    return findings


def _filter(findings, code, source):
    out = findings
    if code:
        out = [f for f in out if code.upper() in f["codes"]]
    if source:
        out = [f for f in out if f["source_type"] == source]
    return out


def main():
    ap = argparse.ArgumentParser(description="Validator backlog / parser-fix worklist")
    ap.add_argument("--code", help="filter by failure code (e.g. ORPHAN_CALL)")
    ap.add_argument("--source", help="filter by source_type (image/text/...)")
    ap.add_argument("--worklist", action="store_true",
                    help="print ready-to-run /fix-hand commands")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    findings = _filter(asyncio.run(_scan()), args.code, args.source)

    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
        return

    if args.worklist:
        print(f"# {len(findings)} hands to fix "
              f"(STALE_EXPECTED = the accepted snapshot itself breaks the rules)\n")
        for f in findings:
            codes = " ".join(f"{c}×{n}" if n > 1 else c for c, n in f["codes"].items())
            print(f"# [{f['status']:14}] {f['source_type']:5} {codes}")
            print(f"/fix-hand {f['hand_id']}  — {f['messages'][0]}")
        return

    # Default: grouped summary + per-hand list.
    by_code = defaultdict(lambda: defaultdict(int))
    for f in findings:
        for c in f["codes"]:
            by_code[c][f["source_type"]] += 1

    n_stale = sum(1 for f in findings if f['status'] == 'STALE_EXPECTED')
    print(f"Validator backlog — {len(findings)} hands flagged "
          f"({len(findings)-n_stale} UNREVIEWED, {n_stale} STALE_EXPECTED)\n")
    print("By failure mode × source (which parser is weakest):")
    for code in sorted(by_code, key=lambda c: -sum(by_code[c].values())):
        per_src = "  ".join(f"{s}:{n}" for s, n in sorted(by_code[code].items()))
        print(f"  {code:26} {sum(by_code[code].values()):3}   {per_src}")

    print("\nHands (newest first):")
    for f in findings:
        codes = " ".join(f"{c}×{n}" if n > 1 else c for c, n in f["codes"].items())
        reg = " [regression]" if f["is_regression"] else ""
        print(f"  {f['hand_id']:7} {f['source_type']:6} {f['status']:6} {codes}{reg}")

    print("\nNext: `--worklist` for fix commands, or `--code <CODE>` / `--source <src>` "
          "to focus the weakest parser.  Fix → re-capture → it drops off this list.")


if __name__ == "__main__":
    main()
