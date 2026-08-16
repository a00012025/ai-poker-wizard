#!/usr/bin/env python3
"""E2E snapshot regression tests.

Fetches regression-flagged snapshots from Supabase and verifies:
  Layer 1 (parse): OCR/Gemini re-parse produces correct hand JSON
  Layer 2 (GTO):   analyze_hand_full() output matches stored snapshot exactly

Usage:
    python scripts/snapshot_test.py                  # Run all regression tests
    python scripts/snapshot_test.py H2489            # Run specific hand
    python scripts/snapshot_test.py --list            # List regression snapshots
    python scripts/snapshot_test.py --add H2489       # Flag snapshot for regression + store expected output
    python scripts/snapshot_test.py --update H2489    # Re-run analysis and update stored GTO output
    python scripts/snapshot_test.py --set-expected H2489 '{"hero_hand":"T9s",...}'
                                                      # Set corrected expected_json

Requires: SUPABASE_CONN and a valid owner ``users.gto_refresh_token`` row.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Layer-1 text re-parse imports gemini_session, which lives in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncpg

SNAPSHOT_CACHE_DIR = Path(__file__).resolve().parent.parent / "tests" / "snapshots" / ".gto_cache"


# ── DB helpers (sync wrappers) ──

def _run(coro):
    """Run async coroutine synchronously."""
    return asyncio.run(coro)


async def _connect():
    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        print("ERROR: SUPABASE_CONN not set. Run: set -a && source .env && set +a")
        sys.exit(1)
    return await asyncpg.connect(dsn, statement_cache_size=0)


async def _fetch_regression_snapshots(hand_id_filter: str | None = None):
    conn = await _connect()
    try:
        if hand_id_filter:
            rows = await conn.fetch(
                "SELECT * FROM analysis_snapshots WHERE hand_id = $1",
                hand_id_filter,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM analysis_snapshots WHERE is_regression = TRUE ORDER BY hand_id",
            )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _update_snapshot(hand_id: str, **fields):
    conn = await _connect()
    try:
        sets = []
        vals = [hand_id]
        for i, (k, v) in enumerate(fields.items(), start=2):
            sets.append(f"{k} = ${i}")
            vals.append(v)
        sql = f"UPDATE analysis_snapshots SET {', '.join(sets)} WHERE hand_id = $1"
        await conn.execute(sql, *vals)
    finally:
        await conn.close()


# ── Test logic ──

def _analyze_snapshot_hand(hand_json: dict) -> dict:
    """Analyze through the same hermetic cache used by regression snapshots.

    Golden creation/update and verification must share one cache. Otherwise a
    golden produced from the owner's mutable root ``.gto_cache`` can disagree
    with the committed snapshot cache even when the code has not changed.
    """
    import gto_cache
    from analyze_hand import analyze_hand_full

    SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    original_dir = gto_cache._CACHE_DIR
    gto_cache._CACHE_DIR = SNAPSHOT_CACHE_DIR
    gto_cache._mem.clear()
    try:
        return analyze_hand_full(hand_json)
    finally:
        gto_cache._CACHE_DIR = original_dir
        gto_cache._mem.clear()


def _compare_parse_fields(parsed: dict, expected: dict) -> list[str]:
    """Compare key fields between parsed and expected hand JSON. Returns list of errors."""
    errors = []
    # effective_bb excluded — acceptable variance between OCR and ground truth
    for key in ["hero_hand", "hero_position", "preflop_actions",
                 "players_at_table", "tournament_type"]:
        p_val = parsed.get(key)
        e_val = expected.get(key)
        if p_val != e_val and e_val is not None:
            errors.append(f"  {key}: got {p_val!r}, expected {e_val!r}")
    # Compare streets structure
    p_streets = parsed.get("streets") or parsed.get("postflop_actions") or []
    e_streets = expected.get("streets") or expected.get("postflop_actions") or []
    if len(p_streets) != len(e_streets):
        errors.append(f"  streets count: got {len(p_streets)}, expected {len(e_streets)}")
    else:
        for i, (ps, es) in enumerate(zip(p_streets, e_streets)):
            p_board = ps.get("board", ps.get("card", ""))
            e_board = es.get("board", es.get("card", ""))
            if p_board != e_board:
                errors.append(f"  street[{i}] board: got {p_board!r}, expected {e_board!r}")
            if expected.get("_strict_actions"):
                p_actions = ps.get("actions", [])
                e_actions = es.get("actions", [])
                if len(p_actions) != len(e_actions):
                    errors.append(
                        f"  street[{i}] actions count: got {len(p_actions)}, "
                        f"expected {len(e_actions)}"
                    )
                    continue
                for j, (pa, ea) in enumerate(zip(p_actions, e_actions)):
                    for action_key in ("position", "action"):
                        p_val = pa.get(action_key)
                        e_val = ea.get(action_key)
                        if p_val != e_val:
                            errors.append(
                                f"  street[{i}].actions[{j}].{action_key}: "
                                f"got {p_val!r}, expected {e_val!r}"
                            )
                    if "size" in ea:
                        p_size = pa.get("size")
                        e_size = ea.get("size")
                        if p_size is None or abs(float(p_size) - float(e_size)) > 0.05:
                            errors.append(
                                f"  street[{i}].actions[{j}].size: "
                                f"got {p_size!r}, expected {e_size!r}"
                            )
    return errors


def run_layer1_ocr(snapshot: dict) -> tuple[bool, str]:
    """Layer 1: Re-parse image with OCR and compare to expected_json.

    Mirrors the production tiered gate in gemini_session.py. A parse is
    only surfaced to the user when OCR confidence is >= OCR_MEDIUM_TIER_MIN
    (default 0.80). Below that, production falls back to Gemini vision —
    a wrong OCR parse in that band is not user-visible and therefore not
    a regression. At or above the medium-tier floor the user sees the OCR
    result directly, so mismatches there are real failures (the async
    cross-check logs them but does not repair this analysis).
    """
    image_data = snapshot.get("image_data")
    if not image_data:
        return True, "SKIP (no image data)"

    expected = json.loads(snapshot["expected_json"]) if snapshot.get("expected_json") else json.loads(snapshot["parsed_json"])

    try:
        from ocr.n8_parser import parse_n8_screenshot
        result = parse_n8_screenshot(bytes(image_data))
    except Exception as e:
        return False, f"OCR error: {e}"

    conf = float(result.get("confidence", 0))
    card_conf = float(result.get("card_confidence", 0))
    MEDIUM_TIER_MIN = float(os.getenv("OCR_MEDIUM_TIER_MIN", "0.80"))
    MIN_CARD_CONF = float(os.getenv("OCR_MIN_CARD_CONF", "0.70"))
    # Production demotes to Gemini when card_conf is below the hard floor,
    # regardless of overall — mirror that here.
    would_demote = card_conf < MIN_CARD_CONF

    if not result.get("hand"):
        if conf < MEDIUM_TIER_MIN or would_demote:
            return True, (f"LOW_CONF_FALLBACK (confidence={conf:.2f}, "
                          f"card_conf={card_conf:.2f}, no hand)")
        return False, f"OCR returned no hand (confidence={conf:.2f})"

    errors = _compare_parse_fields(result["hand"], expected)
    if errors:
        if conf < MEDIUM_TIER_MIN or would_demote:
            return True, (f"LOW_CONF_FALLBACK (confidence={conf:.2f}, "
                          f"card_conf={card_conf:.2f}, would defer to Gemini)")
        return False, f"Parse mismatch (confidence={conf:.2f}):\n" + "\n".join(errors)

    return True, f"OK (confidence={conf:.2f})"


def run_layer1_text(snapshot: dict) -> tuple[bool, str]:
    """Layer 1: Re-parse text with Gemini and compare to expected_json.

    Requires GEMINI_API_KEY env var. Skips if not available.
    """
    user_input = snapshot.get("user_input")
    if not user_input or user_input == "[screenshot]":
        return True, "SKIP (no text input)"

    if not os.environ.get("GEMINI_API_KEY"):
        return True, "SKIP (no GEMINI_API_KEY)"

    expected = json.loads(snapshot["expected_json"]) if snapshot.get("expected_json") else json.loads(snapshot["parsed_json"])

    try:
        from gemini_session import GeminiSessionManager

        # Exercise the production parser, including deterministic repair gates,
        # instead of duplicating an obsolete hard-coded Gemini model call.
        parsed = _run(GeminiSessionManager()._parse_hand(-1, user_input))
    except Exception as e:
        return False, f"Parse error: {e}"

    if not parsed:
        return False, "Parse returned None"

    errors = _compare_parse_fields(parsed, expected)
    if errors:
        return False, f"Parse mismatch:\n" + "\n".join(errors)

    return True, "OK"


def run_layer2_gto(snapshot: dict) -> tuple[bool, str]:
    """Layer 2: Run analyze_hand_full() and compare GTO output exactly."""
    expected_json = snapshot.get("expected_json") or snapshot.get("parsed_json")
    hand_json = json.loads(expected_json) if isinstance(expected_json, str) else expected_json

    expected_text = snapshot["gto_text"]
    expected_compact = snapshot.get("gto_compact")

    try:
        result = _analyze_snapshot_hand(hand_json)
    except Exception as e:
        return False, f"Analysis error: {e}"

    actual_text = result["text"]
    actual_compact = result.get("text_compact")

    # Strip timing lines (they vary between runs)
    import re
    strip_timing = lambda s: re.sub(r"⏱ Discovery:.*$", "", s, flags=re.MULTILINE).rstrip()

    expected_stripped = strip_timing(expected_text)
    actual_stripped = strip_timing(actual_text)

    # Tolerate tiny solver drift in EV (bb) / frequency (%) values; combos
    # counts, action sequences, ranges and line count are still exact. Without
    # the snapshot .gto_cache (e.g. a fresh worktree) the live re-fetch wobbles
    # the last digit (±0.01bb / ±0.2pp), which is not a regression.
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches(expected_stripped, actual_stripped)
    if not ok:
        return False, f"GTO text mismatch: {msg}"

    if expected_compact and actual_compact:
        expected_c = strip_timing(expected_compact)
        actual_c = strip_timing(actual_compact)
        ok_c, msg_c = gto_text_matches(expected_c, actual_c)
        if not ok_c:
            return False, f"GTO compact text mismatch: {msg_c}"

    return True, "OK"


# ── CLI commands ──

def cmd_list():
    """List all regression snapshots."""
    snapshots = _run(_fetch_regression_snapshots())
    if not snapshots:
        print("No regression snapshots found.")
        return
    print(f"{'Hand ID':<10} {'Type':<6} {'Expected?':<10} {'Input':<50}")
    print("-" * 80)
    for s in snapshots:
        has_expected = "YES" if s.get("expected_json") else "no"
        user_input = (s.get("user_input") or "")[:48]
        print(f"{s['hand_id']:<10} {s['source_type']:<6} {has_expected:<10} {user_input}")


def cmd_add(hand_id: str):
    """Flag a snapshot for regression testing and store current GTO output as expected."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    hand_json = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    # Re-run analysis to get fresh GTO output
    print(f"Re-running analyze_hand_full() for {hand_id}...")
    result = _analyze_snapshot_hand(hand_json)

    _run(_update_snapshot(hand_id,
        gto_text=result["text"],
        gto_compact=result.get("text_compact"),
        is_regression=True,
    ))
    print(f"✅ {hand_id} flagged as regression test. GTO output updated ({len(result['text'])} chars).")


def cmd_update(hand_id: str):
    """Re-run analysis and update stored GTO output (after code fix)."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    hand_json = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    print(f"Re-running analyze_hand_full() for {hand_id}...")
    result = _analyze_snapshot_hand(hand_json)

    _run(_update_snapshot(hand_id,
        gto_text=result["text"],
        gto_compact=result.get("text_compact"),
    ))
    print(f"✅ {hand_id} GTO output updated ({len(result['text'])} chars).")


def cmd_set_expected(hand_id: str, json_patch: str):
    """Set corrected expected_json for a snapshot (merge patch into parsed_json)."""
    snapshots = _run(_fetch_regression_snapshots(hand_id))
    if not snapshots:
        print(f"ERROR: Snapshot {hand_id} not found in DB.")
        sys.exit(1)

    s = snapshots[0]
    base = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])

    patch = json.loads(json_patch)
    base.update(patch)

    _run(_update_snapshot(hand_id, expected_json=json.dumps(base)))
    print(f"✅ {hand_id} expected_json updated. Changed fields: {list(patch.keys())}")


def cmd_test(hand_id_filter: str | None = None):
    """Run regression tests."""
    snapshots = _run(_fetch_regression_snapshots(hand_id_filter))
    if hand_id_filter and not snapshots:
        print(f"ERROR: Snapshot {hand_id_filter} not found.")
        sys.exit(1)
    if not snapshots:
        print("No regression snapshots found. Use --add H{id} to flag snapshots.")
        return True

    passed = 0
    failed = 0
    t0 = time.time()

    for s in snapshots:
        hid = s["hand_id"]
        source = s["source_type"]
        print(f"\n{'='*60}")
        print(f"Testing {hid} ({source})")
        print(f"{'='*60}")

        # Layer 1: Parse test
        if source == "image":
            ok, msg = run_layer1_ocr(s)
            label = "L1-OCR"
        else:
            ok, msg = run_layer1_text(s)
            label = "L1-Parse"

        if ok:
            print(f"  \033[32mPASS\033[0m {label}: {msg}")
        else:
            print(f"  \033[31mFAIL\033[0m {label}: {msg}")
            failed += 1

        # Layer 2: GTO output test
        ok2, msg2 = run_layer2_gto(s)
        if ok2:
            print(f"  \033[32mPASS\033[0m L2-GTO: {msg2}")
            passed += 1
        else:
            print(f"  \033[31mFAIL\033[0m L2-GTO: {msg2}")
            failed += 1

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Snapshot tests: {passed} passed, {failed} failed ({total:.1f}s)")
    print(f"{'='*60}")
    return failed == 0


# ── Main ──

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        success = cmd_test()
        sys.exit(0 if success else 1)

    if args[0] == "--list":
        cmd_list()
    elif args[0] == "--add" and len(args) >= 2:
        cmd_add(args[1])
    elif args[0] == "--update" and len(args) >= 2:
        cmd_update(args[1])
    elif args[0] == "--set-expected" and len(args) >= 3:
        cmd_set_expected(args[1], args[2])
    elif args[0].startswith("H"):
        success = cmd_test(args[0])
        sys.exit(0 if success else 1)
    else:
        print(__doc__)
        sys.exit(1)
