#!/usr/bin/env python3
"""Build an OCR ground-truth dataset from a corpus of GGPoker hand-history files.

PokerCraft "Download Game Histories" exports standard GGPoker HH text. Those
files are the unambiguous source of truth (`Dealt to Hero [9c Th]`,
`Seat 1: Hero (big blind)`, full action log). This script scans a directory of
those `.txt` files, parses every hand with hh_parser.parse_hand(), and emits:

  - ground_truth.jsonl : one JSON object per parsed hand, keyed by hand_id
  - coverage.json      : parse-success rate + categorized skip/failure reasons

The coverage report doubles as a parse-robustness test: a high `fail_parse`
count means hh_parser is missing a real GGPoker format and needs hardening
before the dataset can be trusted as a benchmark oracle.

Usage:
    python scripts/build_ground_truth.py <corpus_dir> [-o <out_dir>]
    python scripts/build_ground_truth.py ~/Downloads/0000019e-... -o data/gt
"""

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

from hh_parser import parse_hand, _split_hands  # noqa: E402

HAND_ID_RE = re.compile(r"#(TM\d+)")


def classify_skip(block: str) -> str:
    """Best-effort reason a hand block parsed to None (legit skip vs parser bug).

    Legit skips: hero never dealt in, or hero got a walk (no decision). A block
    that has hero cards + a button + a recognizable level but still fails is a
    parser gap worth investigating.
    """
    has_hero_cards = "Dealt to Hero [" in block
    has_button = "is the button" in block
    has_level = re.search(r"Level\d+\(", block) is not None
    if not has_hero_cards:
        return "skip_hero_not_dealt"
    if not has_level or not has_button:
        return "skip_malformed_header"
    # Hero was dealt and header looks fine but parse still failed: either a
    # legitimate walk (hero in BB, everyone folds) or a real parser gap.
    if re.search(r"Hero: (folds|checks)", block) is None and "Uncalled bet" in block:
        return "skip_walk_or_uncontested"
    return "fail_parse"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="Directory containing GG*.txt hand-history files")
    ap.add_argument("-o", "--out", default="data/ground_truth", help="Output directory")
    ap.add_argument("--glob", default="*.txt", help="Filename glob (default *.txt)")
    ap.add_argument(
        "--include-folds",
        action="store_true",
        default=True,
        help="Include hands where hero folded preflop (default on)",
    )
    args = ap.parse_args()

    corpus = Path(os.path.expanduser(args.corpus))
    files = sorted(corpus.glob(args.glob))
    if not files:
        print(f"No files matching {args.glob} in {corpus}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "ground_truth.jsonl"

    seen: set[str] = set()
    cat = Counter()
    per_file = {}
    fail_samples = []
    total_hands = 0

    with open(jsonl_path, "w", encoding="utf-8") as out:
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="replace")
            blocks = _split_hands(text)
            f_ok = f_skip = f_fail = 0
            for block in blocks:
                if "Poker Hand #" not in block:
                    continue
                total_hands += 1
                m = HAND_ID_RE.search(block)
                hand_id = m.group(1) if m else None
                if hand_id and hand_id in seen:
                    cat["dup"] += 1
                    continue
                try:
                    gt = parse_hand(block, include_folds=args.include_folds)
                except Exception as e:  # noqa: BLE001
                    cat["error"] += 1
                    f_fail += 1
                    if len(fail_samples) < 25:
                        fail_samples.append(
                            {"file": fp.name, "hand_id": hand_id,
                             "error": f"{type(e).__name__}: {e}",
                             "tb": traceback.format_exc().splitlines()[-1]}
                        )
                    continue
                if gt is None:
                    reason = classify_skip(block)
                    cat[reason] += 1
                    if reason == "fail_parse":
                        f_fail += 1
                        if len(fail_samples) < 25:
                            fail_samples.append(
                                {"file": fp.name, "hand_id": hand_id,
                                 "reason": reason,
                                 "header": block.splitlines()[0][:160]}
                            )
                    else:
                        f_skip += 1
                    continue
                if hand_id:
                    seen.add(hand_id)
                cat["ok"] += 1
                f_ok += 1
                out.write(json.dumps(
                    {"hand_id": gt.get("hand_id"),
                     "tournament_id": gt.get("tournament_id"),
                     "source_file": fp.name,
                     "ground_truth": gt},
                    ensure_ascii=False) + "\n")
            per_file[fp.name] = {"hands": f_ok + f_skip + f_fail,
                                 "ok": f_ok, "skip": f_skip, "fail": f_fail}

    parsed = cat["ok"]
    considered = total_hands - cat["dup"]
    legit_skip = sum(v for k, v in cat.items()
                     if k.startswith("skip_"))
    parser_fail = cat["fail_parse"] + cat["error"]
    # Parse rate among hands the parser *should* handle (exclude legit skips).
    eligible = parsed + parser_fail
    parse_rate = parsed / eligible if eligible else 1.0

    report = {
        "corpus": str(corpus),
        "files": len(files),
        "total_hand_blocks": total_hands,
        "duplicates": cat["dup"],
        "parsed_ok": parsed,
        "legit_skipped": legit_skip,
        "parser_failures": parser_fail,
        "parse_rate_on_eligible": round(parse_rate, 5),
        "categories": dict(cat),
        "fail_samples": fail_samples,
        "per_file": per_file,
    }
    (out_dir / "coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 64)
    print(f"corpus           : {corpus}")
    print(f"files            : {len(files)}")
    print(f"hand blocks      : {total_hands}  (dups {cat['dup']})")
    print(f"parsed OK        : {parsed}")
    print(f"legit skipped    : {legit_skip}  "
          f"({dict((k, v) for k, v in cat.items() if k.startswith('skip_'))})")
    print(f"parser failures  : {parser_fail}  "
          f"(fail_parse {cat['fail_parse']}, error {cat['error']})")
    print(f"PARSE RATE       : {parse_rate:.4%}  (parsed / (parsed+failures))")
    print(f"ground truth     : {jsonl_path}  ({parsed} hands)")
    print(f"coverage report  : {out_dir / 'coverage.json'}")
    if fail_samples:
        print("-" * 64)
        print("first failure samples:")
        for s in fail_samples[:8]:
            print(" ", s)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
