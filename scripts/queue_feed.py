#!/usr/bin/env python3
"""Online-hand feed for the unified practice queue (North Star mid-loop:
diagnose → focus → drill; the Phase 1 → Phase 2 bridge).

Scans the rolling online-hand window and enqueues two shapes into
``drill_queue`` (design spec docs/superpowers/specs/2026-07-12-unified-drill-queue-design.md):

  kind='drill'  — a systematic leak (a spot_leaf with n>=3 lossy decisions
                  summing to >=3bb) → a GTOW Trainer drill link.
  kind='review' — a single-hand disaster (a decision >=5bb) aggregated per
                  hand → a GTOW Analyze review link.

This module also owns the SINGLE upsert policy for the whole queue: the
dedupe-aware ``enqueue`` (idempotent under the weekly rolling re-scan) lives
here and ``live_flow`` imports it — no second copy (§5.2, PR #92 dedup spirit).

CLI:
  python scripts/queue_feed.py --scan [--window-days 60] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import spot_leaderboard as lb
from action_bias import dominant_action_bias
from card_display import cards_to_emoji
from spot_leaderboard import analyze_table_url

TPE = ZoneInfo("Asia/Taipei")
log = logging.getLogger(__name__)

# ── scan thresholds (all tunable; owner-approved defaults) ────────────────────
QUEUE_SCAN_WINDOW_DAYS = 60     # rolling window (owner's "past two months"; kept
                                # DISTINCT from scorecard's 90d focus window on purpose)
QUEUE_DRILL_MIN_N = 3           # confidence floor: >=3 lossy decisions is a pattern
                                # (§2.1; n is only a gate, ranking stays pure EV §7.3)
QUEUE_DRILL_MIN_TOTAL_BB = 3.0  # worth one <=20min Trainer session (北極星 §5.6)
QUEUE_REVIEW_MIN_BB = 5.0       # single-decision disaster threshold
LOSSY_MIN_BB = 0.10             # a decision counts as "lossy" at/above this (== live QUEUE_EV_MIN)
REOPEN_MIN_NEW = 2              # cleared drill leaf revives only on >=this many post-clear lossy decisions
LOW_FREQUENCY_BRANCH = 0.05     # path hint only; NEVER used for EV ordering (§7.3)
QUEUE_SOURCE_HANDS_PER_LINK = 20
GTOW_ANALYZE_HANDS_URL = "https://app.gtowizard.com/analyze/v4/hands/table"

# The honest predicate — reused VERBATIM from spot_leaderboard so the queue and
# the leak board see the same population (NOT discarded strips discarded:* buckets).
_HONEST = ("NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL "
           "AND source='online' AND confidence >= 0.8")
_APPROX_KEYS = ("sizing_snap", "missing_solver_depth", "analyzer_approximation")


# ── pure helpers (unit-tested; no DB) ─────────────────────────────────────────
def _as_list(v):
    """asyncpg returns jsonb as a str unless a codec is set; normalize to list."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return []
    return list(v)


def entry_key(e: dict) -> tuple:
    """Dedupe identity of a source_hands entry (§5.2). Old live rows may lack
    decision_idx/src — .get keeps them distinct from new full-key entries
    rather than colliding."""
    ev = e.get("ev_loss_bb")
    return (e.get("hand_id"), e.get("street"), e.get("decision_idx"),
            round(float(ev), 4) if ev is not None else None, e.get("src"))


def dedupe_entries(entries: list[dict]) -> list[dict]:
    """Drop entries with a duplicate key, preserving order."""
    seen, out = set(), []
    for e in entries:
        k = entry_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def queue_source_hand_ids(entries: list[dict], ref_hand_id: str | None = None) -> list[str]:
    """Unique source hand ids in their persisted order, plus review fallback."""
    out = []
    seen = set()
    for entry in _as_list(entries):
        hand_id = entry.get("hand_id")
        if hand_id and hand_id not in seen:
            seen.add(hand_id)
            out.append(hand_id)
    if ref_hand_id and ref_hand_id not in seen:
        out.append(ref_hand_id)
    return out


def resolve_queue_source_hands(entries: list[dict], ledger_rows,
                               ref_hand_id: str | None = None) -> list[dict]:
    """Resolve queue provenance against ``ledger_hands`` and rank by EV loss.

    ``source_hands.src`` records how a queue entry was added; it is not a
    reliable hand-source discriminator (manual drills commonly point at online
    hands).  Only ``ledger_hands.source`` decides whether a hand is online or
    live.  Duplicate decision snapshots are ignored so a merged queue row
    cannot inflate the displayed per-hand EV.
    """
    ids = queue_source_hand_ids(entries, ref_hand_id)
    by_id = {dict(row).get("gtow_hand_id"): dict(row) for row in (ledger_rows or [])}
    aggregates = {hand_id: {"ev_loss_bb": 0.0, "order": index}
                  for index, hand_id in enumerate(ids)}
    seen_decisions = set()
    for entry in _as_list(entries):
        hand_id = entry.get("hand_id")
        if hand_id not in aggregates:
            continue
        ev = float(entry.get("ev_loss_bb") or 0.0)
        decision_key = (hand_id, entry.get("street"), entry.get("decision_idx"),
                        round(ev, 4))
        if decision_key in seen_decisions:
            continue
        seen_decisions.add(decision_key)
        aggregates[hand_id]["ev_loss_bb"] += ev

    resolved = []
    for hand_id in ids:
        meta = by_id.get(hand_id, {})
        source = meta.get("source")
        if source not in {"online", "live"}:
            source = "missing"
        row = dict(meta)
        row.update({
            "hand_id": hand_id,
            "source": source,
            "ev_loss_bb": round(aggregates[hand_id]["ev_loss_bb"], 4),
            "_source_order": aggregates[hand_id]["order"],
        })
        resolved.append(row)
    resolved.sort(key=lambda row: (-row["ev_loss_bb"], row["_source_order"]))
    for row in resolved:
        row.pop("_source_order", None)
    return resolved


def gtow_analyze_hands_urls(hand_ids: list[str],
                            chunk_size: int = QUEUE_SOURCE_HANDS_PER_LINK
                            ) -> list[tuple[str, list[str]]]:
    """Chunk ranked exact-hand filters into Telegram-safe URLs.

    ``hand_ids`` already arrive ordered by this queue item's attributed EV
    loss.  Do not ask GTOW to re-sort by whole-hand total EV: that would let
    unrelated later-street losses override the queue item's spot ranking.
    """
    unique_ids = list(dict.fromkeys(hand_id for hand_id in hand_ids if hand_id))
    urls = []
    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start:start + chunk_size]
        filters = json.dumps({"hand_id__in": chunk}, separators=(",", ":"))
        url = (f"{GTOW_ANALYZE_HANDS_URL}?filters={quote(filters)}"
               "&preselectGamemode=TOURNAMENT")
        urls.append((url, chunk))
    return urls


def diff_new_entries(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], float]:
    """Entries in ``incoming`` whose key is absent from ``existing``, plus the
    EV those fresh entries add. Python-side diff — never a blind ``|| append``
    (§5.2 idempotency: a rolling re-scan of overlapping windows must not
    re-inflate a pending row's totals)."""
    have = {entry_key(e) for e in existing}
    fresh, add_ev = [], 0.0
    for e in dedupe_entries(incoming):
        if entry_key(e) not in have:
            fresh.append(e)
            add_ev += float(e.get("ev_loss_bb") or 0.0)
    return fresh, round(add_ev, 4)


def reopen_decision(open_exists: bool, cleared_at, played_ats: list) -> str:
    """Route a scanned drill candidate: 'merge' into an open row, 'insert' a
    fresh pending row, or 'skip'. A cleared leaf does NOT blindly revive — it
    needs >=REOPEN_MIN_NEW lossy decisions played after the clear (§5.2)."""
    if open_exists:
        return "merge"
    if cleared_at is not None:
        new_ev = [p for p in played_ats if p is not None and p > cleared_at]
        return "insert" if len(new_ev) >= REOPEN_MIN_NEW else "skip"
    return "insert"


def _has_approx(flags) -> bool:
    return any(f in _APPROX_KEYS for f in _as_list(flags))


_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_STREET_TITLE = {"preflop": "Preflop", "flop": "Flop", "turn": "Turn", "river": "River"}


def low_frequency_anchor(decisions: list[dict], worst_street: str,
                         worst_idx: int) -> dict | None:
    """Earliest prior hero decision whose chosen action is <=5% in GTO.

    This is a review-order hint, not an EV severity score: the queue remains
    ranked by realized EV loss.  All ledger_decisions are hero decisions.
    """
    worst_key = (_STREET_ORDER.get(worst_street, 9), int(worst_idx or 0))
    ordered = sorted(decisions, key=lambda d: (
        _STREET_ORDER.get(d.get("street"), 9), int(d.get("decision_idx") or 0)))
    for dec in ordered:
        key = (_STREET_ORDER.get(dec.get("street"), 9), int(dec.get("decision_idx") or 0))
        freq = dec.get("taken_freq")
        if key < worst_key and freq is not None and float(freq) <= LOW_FREQUENCY_BRANCH:
            return dec
    return None


def pretty_hand(hero_hand: str | None) -> str:
    """Display exact combos with emoji suits; class/odd inputs pass through."""
    return cards_to_emoji(hero_hand)


def review_label(row: dict) -> str:
    """`復盤 {M/D} {hero combo w/ suits} {spot_desc_zh(worst decision)} −{ev:.1f}bb`
    (+⚠近似 when the worst decision leans on an off-tree approximation, §5.2).

    The exact combo is included so the owner knows which hand to read in the
    Study solution the link opens to."""
    from scorecard import spot_desc_zh
    played = row.get("played_at")
    md = played.astimezone(TPE).strftime("%-m/%-d") if played else "?"
    desc = spot_desc_zh({
        "spot_category": row.get("spot_category"), "spot_leaf": row.get("spot_leaf"),
        "hero_cat": row.get("hero_cat"), "villain_cat": row.get("villain_cat"),
        "ip_oop": row.get("ip_oop"), "hero_pos": row.get("hero_pos"),
        "street": row.get("spot_category")})
    hand = pretty_hand(row.get("hero_hand"))
    ev = float(row.get("max_ev") or 0.0)
    anchor_title = _STREET_TITLE.get(row.get("review_anchor_street"))
    hint = (f"（{anchor_title} 走了低頻分支，建議從 {anchor_title} 開始看）"
            if anchor_title else "")
    warn = " ⚠近似" if _has_approx(row.get("approx_flags")) else ""
    return f"復盤 {md} {hand + ' ' if hand else ''}{desc} −{ev:.1f}bb{hint}{warn}"


def drill_label(row: dict, action_bias: dict | None = None) -> str:
    """Compact online Drill name; action bias stays in Telegram metadata."""
    from spot_naming import compact_spot_name
    return compact_spot_name(row)


def _decision_effective_depth(row: dict) -> float | None:
    """Depth to use when replaying/navigating a specific ledger decision.

    ``ledger_hands.preflop_depth_bb`` is the hero/list-row stack for audit.  The
    solver tree for a decision is bounded by the opponent that can contest hero
    (effectively ``min(hero stack, max remaining opponent stack)`` preflop, and
    GTOW records that as ``ledger_decisions.solver_depth_bb``).  Study/custom
    links must use the decision depth, falling back to the played/list depth only
    when the game point did not expose one.
    """
    for key in ("solver_depth_bb", "played_depth_bb", "preflop_depth_bb"):
        value = row.get(key)
        try:
            depth = float(value)
        except (TypeError, ValueError):
            continue
        if depth > 0:
            return depth
    return None


def review_url(row: dict) -> str | None:
    """Review link. PREFERRED: the /solutions Study node for the worst decision,
    built from the archived hand detail (the owner reads their exact combo there).
    FALLBACK (archive missing / node has no solution): the Analyze table filtered
    to the hand's Taipei day."""
    study = _study_solution_url(row)
    if study:
        return study
    played = row.get("played_at")
    if not played:
        return None
    day = played.astimezone(TPE).strftime("%Y-%m-%d")
    return analyze_table_url(day, day)


def _study_solution_url(row: dict) -> str | None:
    """Build the /solutions Study URL for a review hand's worst decision from
    its archived GTOW detail JSON (`ledger_hands.raw_path`). Returns None on any
    failure so review_url falls back to the day-range Analyze table."""
    raw_path = row.get("raw_path")
    hero_pos = row.get("hero_pos")
    street = row.get("worst_street")
    idx = row.get("worst_idx")
    if not (raw_path and hero_pos and street and idx is not None):
        return None
    try:
        import gzip
        from gtow_solution_url import build_hand_solution_url
        p = Path(raw_path) if os.path.isabs(raw_path) else (ROOT / raw_path)
        if not p.exists():
            return None
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rb") as fh:
            detail = json.loads(fh.read())
        return build_hand_solution_url(
            detail, hero_pos, street, int(idx),
            preflop_depth_bb=_decision_effective_depth(row),
        )
    except Exception:
        return None


def mix_queue_quota(rows: list[dict], drill_slots: int, review_slots: int,
                    limit: int) -> list[dict]:
    """Fill `limit` plan slots from an EV-ordered, pending-first row list with a
    per-kind quota; a short kind is topped up from the other (§7)."""
    drills = [r for r in rows if r.get("kind", "drill") == "drill"]
    reviews = [r for r in rows if r.get("kind") == "review"]
    picked = drills[:drill_slots] + reviews[:review_slots]
    if len(picked) < limit:
        chosen = {id(r) for r in picked}
        for r in drills[drill_slots:] + reviews[review_slots:]:
            if len(picked) >= limit:
                break
            if id(r) not in chosen:
                picked.append(r)
    # keep the caller's pending-first / EV-desc order
    order = {id(r): i for i, r in enumerate(rows)}
    return sorted(picked[:limit], key=lambda r: order.get(id(r), 0))


def qex_submenu(decisions: list[dict], queue_id: int) -> list[dict]:
    """Sub-menu rows for a review item's decisions (§6.2).

    New rows carry the stable ledger decision identity
    ``gtow_hand_id/street/decision_idx``.  ``ledger_decisions.id`` is volatile
    because re-ingesting a hand deletes and re-inserts that hand's decisions.
    NEVER put the spot_leaf string in callback_data; it blows Telegram's
    64-byte limit.
    """
    from scorecard import spot_desc_zh
    st_order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    rows = []
    for d in sorted(decisions, key=lambda d: (st_order.get(d.get("street"), 9),
                                              d.get("decision_idx") or 0)):
        ev = float(d.get("ev_loss_bb") or 0.0)
        desc = spot_desc_zh({
            "spot_category": d.get("spot_category"), "spot_leaf": d.get("spot_leaf"),
            "hero_cat": d.get("hero_cat"), "villain_cat": d.get("villain_cat"),
            "ip_oop": d.get("ip_oop"), "hero_pos": d.get("position"),
            "street": d.get("spot_category")})
        prefix = f"➕ {d.get('street')} "
        suffix = f"｜損失 {ev:.1f}bb" if ev > 0 else ""
        # Preserve the only actionable annotation when a long action-line
        # description needs truncation. Frequency/BEST_MOVE details live in
        # GTOW Study; this picker only answers which line to add.
        desc_budget = max(0, 60 - len(prefix) - len(suffix))
        hid = d.get("gtow_hand_id")
        if hid and d.get("street") is not None and d.get("decision_idx") is not None:
            callback_data = f"qad2:{queue_id}:{hid}:{d['street']}:{int(d['decision_idx'])}"
        else:
            # Backward-compatible fallback for unit/dry-run callers that have
            # not supplied the stable identity.
            callback_data = f"qad:{queue_id}:{d['id']}"
        rows.append({"text": prefix + desc[:desc_budget] + suffix,
                     "callback_data": callback_data})
    return rows


_URL_UNSET = object()


def manual_drill_item(dec: dict, drill_url=_URL_UNSET) -> dict:
    """Build a manual drill queue item from a single ledger_decisions row (qad).

    Reuses the ONE drill-URL + label policy (§5.3); ev may be 0 (owner wants to
    drill a spot they played right but felt unsure about)."""
    from gtow_trainer_url import MTT_DEPTHS, DEPTH_BAND_DEPTHS, drill_url_for_spot
    depths = DEPTH_BAND_DEPTHS.get(dec.get("eff_stack") or "", list(MTT_DEPTHS))
    url = (drill_url_for_spot(
        dec.get("spot_category"), hero_pos=dec.get("position"),
        hero_cat=dec.get("hero_cat"), villain_cat=dec.get("villain_cat"),
        ip_oop=dec.get("ip_oop"), pot_type=dec.get("pot_type"), depths=depths)
        if drill_url is _URL_UNSET else drill_url)
    ev = dec.get("ev_loss_bb")
    return {
        "kind": "drill", "added_by": "manual", "source": "manual",
        "spot_leaf": dec.get("spot_leaf"), "spot_category": dec.get("spot_category"),
        "label": drill_label({
            "spot_category": dec.get("spot_category"), "spot_leaf": dec.get("spot_leaf"),
            "hero_cat": dec.get("hero_cat"), "villain_cat": dec.get("villain_cat"),
            "ip_oop": dec.get("ip_oop"), "hero_pos": dec.get("position")}),
        "drill_url": url, "ref_hand_id": dec.get("gtow_hand_id"),
        "total_ev_loss_bb": round(float(ev), 4) if ev is not None else 0.0,
        "source_hands": [{"hand_id": dec.get("gtow_hand_id"), "street": dec.get("street"),
                          "decision_idx": dec.get("decision_idx"),
                          "ev_loss_bb": float(ev) if ev is not None else 0.0,
                          "taken_code": dec.get("taken_code"),
                          "best_code": dec.get("best_code"),
                          "src": "manual"}],
    }


_QUEUE_DECISION_SQL = """
SELECT d.id, d.gtow_hand_id, d.street, d.decision_idx, d.spot_category,
       d.spot_leaf, d.hero_cat, d.villain_cat, d.ip_oop, d.position,
       d.pot_type, d.eff_stack, d.ev_loss_bb, d.taken_code, d.best_code, d.gametype,
       d.played_depth_bb, d.solver_depth_bb,
       h.source hand_source, h.raw_path, h.raw_text, h.parsed_json,
       h.position hand_position, h.preflop_depth_bb
FROM ledger_decisions d
JOIN ledger_hands h ON h.gtow_hand_id=d.gtow_hand_id
WHERE d.gtow_hand_id=$1 AND d.street=$2 AND d.decision_idx=$3
"""

_QUEUE_DECISIONS_BY_STREET_SQL = _QUEUE_DECISION_SQL.replace(
    " AND d.decision_idx=$3", " ORDER BY d.decision_idx")

_EXACT_SOURCE_CATEGORIES = {"flop", "turn", "river", "vsCold3bet", "vsCold4bet"}


def _load_source_hand(dec: dict) -> dict:
    """Reconstruct the parsed-hand shape required by the custom URL builder."""
    source = dec.get("hand_source")
    if source == "live":
        parsed = dec.get("parsed_json")
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            raise ValueError("live ledger hand has no parsed_json")
        parsed = dict(parsed)
        if not parsed.get("preflop_actions_for_pot") and dec.get("raw_text"):
            from live_flow import preflop_actions_for_pot_from_raw
            pot_line = preflop_actions_for_pot_from_raw(dec["raw_text"], parsed)
            if pot_line:
                parsed["preflop_actions_for_pot"] = pot_line
        return parsed

    raw_path = dec.get("raw_path")
    depth = _decision_effective_depth(dec)
    hero_pos = dec.get("hand_position") or dec.get("position")
    if not (raw_path and depth is not None and hero_pos):
        raise ValueError("online ledger hand is missing archive/depth/position")
    import gzip
    from gtow_solution_url import _parsed_hand_from_analyze
    p = Path(raw_path) if os.path.isabs(raw_path) else (ROOT / raw_path)
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rb") as fh:
        detail = json.loads(fh.read())
    return _parsed_hand_from_analyze(
        detail, hero_pos, float(depth), dec.get("gametype") or "MTTGeneral")


def _is_flat_vs_squeeze(dec: dict) -> bool:
    return "flat_vsSqueeze" in str(dec.get("spot_leaf") or "")


def _exact_pot_type(dec: dict) -> str:
    category = dec.get("spot_category")
    if _is_flat_vs_squeeze(dec):
        return "squeezed"
    if category == "vsCold3bet":  # legacy rows only; new taxonomy emits vs3bet
        return "3bet"
    if category == "vsCold4bet":
        return "4bet"
    return dec.get("pot_type") or ""


def queue_drill_url_for_decision(dec: dict, depths: list[int] | None = None) -> str | None:
    """Faithful queue Trainer URL for one joined ledger decision.

    Postflop and cold-raise spots require the source action history.  Failure is
    honest ``None`` — never a nearby shortcut.  Supported preflop categories
    continue to use CDP-verified shortcuts.
    """
    category = dec.get("spot_category")
    if category in _EXACT_SOURCE_CATEGORIES or _is_flat_vs_squeeze(dec):
        try:
            from gtow_custom_url import build_custom_spot_url
            hand = _load_source_hand(dec)
            return build_custom_spot_url(
                hand, dec.get("street") or "preflop",
                int(dec.get("decision_idx") or 0), _exact_pot_type(dec))
        except Exception as exc:
            log.warning(
                "queue Trainer exact URL failed (%s %s[%s]): %s",
                dec.get("gtow_hand_id"), dec.get("street"),
                dec.get("decision_idx"), exc)
            return None

    from gtow_trainer_url import MTT_DEPTHS, DEPTH_BAND_DEPTHS, drill_url_for_spot
    if depths is None:
        depths = DEPTH_BAND_DEPTHS.get(dec.get("eff_stack") or "", list(MTT_DEPTHS))
    return drill_url_for_spot(
        category, hero_pos=dec.get("position"), hero_cat=dec.get("hero_cat"),
        villain_cat=dec.get("villain_cat"), ip_oop=dec.get("ip_oop"),
        pot_type=dec.get("pot_type"), depths=depths)


async def _source_decisions(conn, entries: list[dict]) -> list[dict]:
    out = []
    for entry in entries:
        if not (entry.get("hand_id") and entry.get("street")):
            continue
        if entry.get("decision_idx") is not None:
            row = await conn.fetchrow(
                _QUEUE_DECISION_SQL, entry["hand_id"], entry["street"],
                int(entry["decision_idx"]))
        else:
            # Rows created before the unified queue migration omitted the
            # decision index.  Repair them by matching the stored EV on the
            # same hand/street (or the sole decision on that street).
            candidates = [dict(r) for r in await conn.fetch(
                _QUEUE_DECISIONS_BY_STREET_SQL, entry["hand_id"], entry["street"])]
            if len(candidates) == 1:
                row = candidates[0]
            elif candidates and entry.get("ev_loss_bb") is not None:
                target = float(entry["ev_loss_bb"])
                row = min(candidates, key=lambda d: abs(
                    float(d.get("ev_loss_bb") or 0.0) - target))
                if abs(float(row.get("ev_loss_bb") or 0.0) - target) > 0.001:
                    row = None
            else:
                row = None
        if row:
            out.append(dict(row))
    return out


async def normalize_source_entries(conn, entries: list[dict]) -> list[dict]:
    """Backfill queue-source identity and action facts from the ledger."""
    normalized = []
    for entry in entries:
        fixed = dict(entry)
        if (fixed.get("decision_idx") is None or fixed.get("taken_code") is None
                or fixed.get("best_code") is None):
            matches = await _source_decisions(conn, [fixed])
            if matches:
                fixed["decision_idx"] = int(matches[0]["decision_idx"])
                fixed.setdefault("src", matches[0].get("hand_source") or "online")
                fixed["taken_code"] = matches[0].get("taken_code")
                fixed["best_code"] = matches[0].get("best_code")
        normalized.append(fixed)
    return normalized


async def queue_drill_url_from_sources(conn, entries: list[dict],
                                       depths: list[int] | None = None) -> str | None:
    """Use the newest source decision that can produce a faithful URL.

    ``source_hands`` is persisted chronologically.  New evidence should own
    the representative depth and bet sizes; older hands are fallback only
    when the latest source cannot be reconstructed exactly.
    """
    decisions = await _source_decisions(conn, entries)
    for dec in reversed(decisions):
        if dec.get("spot_category") in _EXACT_SOURCE_CATEGORIES:
            # Exact resolution may query GTOW while snapping real bet sizes;
            # keep Telegram callbacks and weekly jobs off the event loop.
            url = await asyncio.to_thread(
                queue_drill_url_for_decision, dec, depths)
        else:
            url = queue_drill_url_for_decision(dec, depths=depths)
        if url:
            return url
    return None


# ── shared upsert policy (the ONE enqueue; live_flow imports this) ────────────
_OPEN_DRILL_SQL = """
SELECT id, source_hands, n_sources, total_ev_loss_bb
FROM drill_queue
WHERE spot_leaf = $1 AND kind = 'drill' AND status IN ('pending', 'prescribed')
ORDER BY (status = 'pending') DESC, last_added DESC LIMIT 1
"""

_MERGE_SQL = """
UPDATE drill_queue SET
  source_hands = $2::jsonb,
  n_sources = $3,
  total_ev_loss_bb = COALESCE(total_ev_loss_bb, 0) + $4,
  gtow_settings_hash = CASE
    WHEN $5::text IS NOT NULL AND drill_url IS DISTINCT FROM $5::text
      THEN NULL ELSE gtow_settings_hash END,
  gtow_drill_synced_at = CASE
    WHEN $5::text IS NOT NULL AND drill_url IS DISTINCT FROM $5::text
      THEN NULL ELSE gtow_drill_synced_at END,
  drill_url = COALESCE($5::text, drill_url),
  label = COALESCE($6, label),
  bias_key = CASE WHEN $7 THEN $8 ELSE bias_key END,
  bias_direction = CASE WHEN $7 THEN $9 ELSE bias_direction END,
  bias_n = CASE WHEN $7 THEN $10 ELSE bias_n END,
  bias_ev_loss_bb = CASE WHEN $7 THEN $11 ELSE bias_ev_loss_bb END,
  bias_share = CASE WHEN $7 THEN $12 ELSE bias_share END,
  last_added = NOW()
WHERE id = $1
"""

_INSERT_SQL = """
INSERT INTO drill_queue (spot_leaf, spot_category, label, drill_url,
                         review_anchor_url, review_anchor_street, source_hands,
                         n_sources, total_ev_loss_bb, kind, added_by, source,
                         ref_hand_id, bias_direction, bias_n, bias_ev_loss_bb,
                         bias_share, bias_key)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13,
        $14, $15, $16, $17, $18)
"""

_REFRESH_BIAS_SQL = """
UPDATE drill_queue SET label=$2, bias_key=$3, bias_direction=$4, bias_n=$5,
  bias_ev_loss_bb=$6, bias_share=$7, last_added=NOW()
WHERE id=$1
"""


async def enqueue_one(conn, it: dict) -> str:
    """Idempotent upsert of one queue item. Returns 'merged' | 'inserted' |
    'noop'. Drill items merge into an OPEN row of the same leaf (dedupe-aware);
    review items always insert (the scan guards ref_hand_id uniqueness)."""
    kind = it.get("kind", "drill")
    incoming = dedupe_entries(list(it.get("source_hands") or []))
    refresh_bias = "action_bias" in it
    bias = it.get("action_bias") or {}
    if kind == "drill":
        open_row = await conn.fetchrow(_OPEN_DRILL_SQL, it["spot_leaf"])
        if open_row:
            existing = _as_list(open_row["source_hands"])
            fresh, add_ev = diff_new_entries(existing, incoming)
            if not fresh:
                if refresh_bias:
                    await conn.execute(
                        _REFRESH_BIAS_SQL, open_row["id"], it.get("label"),
                        it.get("bias_key"), bias.get("direction"), bias.get("n"),
                        bias.get("ev_loss_bb"), bias.get("share"))
                return "noop"
            merged = existing + fresh
            await conn.execute(
                _MERGE_SQL, open_row["id"], json.dumps(merged),
                (open_row["n_sources"] or 0) + len(fresh), add_ev,
                it.get("drill_url"), it.get("label"), refresh_bias,
                it.get("bias_key"), bias.get("direction"), bias.get("n"),
                bias.get("ev_loss_bb"), bias.get("share"))
            return "merged"
    await conn.execute(
        _INSERT_SQL, it.get("spot_leaf"), it.get("spot_category"), it.get("label"),
        it.get("drill_url"), it.get("review_anchor_url"),
        it.get("review_anchor_street"), json.dumps(incoming), len(incoming),
        it.get("total_ev_loss_bb"), kind, it.get("added_by", "auto"),
        it.get("source", "online"), it.get("ref_hand_id"),
        bias.get("direction"), bias.get("n"), bias.get("ev_loss_bb"),
        bias.get("share"), it.get("bias_key"))
    return "inserted"


async def enqueue(conn, items: list[dict]) -> dict:
    """Upsert every item; returns a small {merged,inserted,noop} tally."""
    tally = {"merged": 0, "inserted": 0, "noop": 0}
    for it in items:
        tally[await enqueue_one(conn, it)] += 1
    return tally


async def remove_source_hand(conn, hand_id: str) -> None:
    """Strip a hand's contributions from every open drill queue row.

    For pending/prescribed rows whose ``source_hands`` includes ``hand_id``,
    drop those source entries, recompute EV/n, and clear empty auto/live drill
    rows as resend fallout.
    """
    rows = await conn.fetch(
        "SELECT id, source_hands, added_by, kind, drill_url FROM drill_queue "
        "WHERE status IN ('pending','prescribed') "
        "AND source_hands::text LIKE '%' || $1 || '%'", hand_id)
    for r in rows:
        srcs = _as_list(r["source_hands"])
        kept = [s for s in srcs if s.get("hand_id") != hand_id]
        if len(kept) == len(srcs):
            continue
        if (not kept and r["kind"] == "drill"
                and r["added_by"] in ("auto", "live")):
            await conn.execute(
                "UPDATE drill_queue SET status='cleared', cleared_at=NOW(), "
                "clear_reason='resend', source_hands='[]'::jsonb, "
                "total_ev_loss_bb=0, n_sources=0, drill_url=NULL, "
                "gtow_drill_id=NULL, gtow_drill_name=NULL, "
                "gtow_settings_hash=NULL, gtow_drill_synced_at=NULL, "
                "gtow_training_started_at=NULL, gtow_baseline_totals=NULL "
                "WHERE id=$1", r["id"])
            continue
        total = round(sum(float(s.get("ev_loss_bb") or 0) for s in kept), 4)
        old_url = r.get("drill_url") if hasattr(r, "get") else r["drill_url"]
        try:
            rebuilt_url = await queue_drill_url_from_sources(conn, kept)
        except Exception as exc:
            log.warning("queue source URL rebuild failed for row %s: %s",
                        r["id"], exc, exc_info=True)
            rebuilt_url = None
        if rebuilt_url is None:
            await conn.execute(
                "UPDATE drill_queue SET source_hands=$2::jsonb, "
                "total_ev_loss_bb=$3, n_sources=$4 WHERE id=$1",
                r["id"], json.dumps(kept), total, len(kept))
        elif rebuilt_url == old_url:
            await conn.execute(
                "UPDATE drill_queue SET source_hands=$2::jsonb, "
                "total_ev_loss_bb=$3, n_sources=$4, drill_url=$5 WHERE id=$1",
                r["id"], json.dumps(kept), total, len(kept), rebuilt_url)
        else:
            await conn.execute(
                "UPDATE drill_queue SET source_hands=$2::jsonb, "
                "total_ev_loss_bb=$3, n_sources=$4, drill_url=$5, "
                "gtow_drill_id=NULL, gtow_drill_name=NULL, "
                "gtow_settings_hash=NULL, gtow_drill_synced_at=NULL, "
                "gtow_training_started_at=NULL, gtow_baseline_totals=NULL "
                "WHERE id=$1",
                r["id"], json.dumps(kept), total, len(kept), rebuilt_url)


# ── online scan ──────────────────────────────────────────────────────────────
def _drill_scan_sql(win_col: str = "$1") -> str:
    return f"""
SELECT spot_leaf, spot_category,
       count(*) n, sum(ev_loss_bb) total_ev,
       mode() WITHIN GROUP (ORDER BY spot_parent)  spot_parent,
       mode() WITHIN GROUP (ORDER BY hero_cat)    hero_cat,
       mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
       mode() WITHIN GROUP (ORDER BY ip_oop)      ip_oop,
       mode() WITHIN GROUP (ORDER BY position)    hero_pos,
       jsonb_agg(jsonb_build_object(
           'hand_id', gtow_hand_id, 'street', street,
           'decision_idx', decision_idx, 'ev_loss_bb', ev_loss_bb,
           'taken_code', taken_code, 'best_code', best_code,
           'src', source) ORDER BY played_at) source_hands,
       array_agg(played_at ORDER BY played_at) played_ats
FROM ledger_decisions
WHERE {_HONEST} AND played_at >= {win_col} AND ev_loss_bb >= $2
GROUP BY spot_leaf, spot_category
HAVING count(*) >= $3 AND sum(ev_loss_bb) >= $4
ORDER BY sum(ev_loss_bb) DESC
"""


_REVIEW_SCAN_SQL = f"""
SELECT gtow_hand_id ref_hand_id,
       sum(ev_loss_bb) total_ev,
       (array_agg(spot_leaf     ORDER BY ev_loss_bb DESC))[1] spot_leaf,
       (array_agg(spot_category ORDER BY ev_loss_bb DESC))[1] spot_category,
       (array_agg(hero_cat      ORDER BY ev_loss_bb DESC))[1] hero_cat,
       (array_agg(villain_cat   ORDER BY ev_loss_bb DESC))[1] villain_cat,
       (array_agg(ip_oop        ORDER BY ev_loss_bb DESC))[1] ip_oop,
       (array_agg(position      ORDER BY ev_loss_bb DESC))[1] hero_pos,
       (array_agg(ev_loss_bb       ORDER BY ev_loss_bb DESC))[1] max_ev,
       (array_agg(approx_flags     ORDER BY ev_loss_bb DESC))[1] approx_flags,
       (array_agg(played_depth_bb  ORDER BY ev_loss_bb DESC))[1] played_depth_bb,
       (array_agg(solver_depth_bb  ORDER BY ev_loss_bb DESC))[1] solver_depth_bb,
       (array_agg(played_at        ORDER BY ev_loss_bb DESC))[1] played_at,
       (array_agg(street           ORDER BY ev_loss_bb DESC))[1] worst_street,
       (array_agg(decision_idx  ORDER BY ev_loss_bb DESC))[1] worst_idx,
       jsonb_agg(jsonb_build_object(
           'hand_id', gtow_hand_id, 'street', street,
           'decision_idx', decision_idx, 'ev_loss_bb', ev_loss_bb,
           'src', source) ORDER BY ev_loss_bb DESC) source_hands
FROM ledger_decisions
WHERE {_HONEST} AND played_at >= $1 AND ev_loss_bb >= $2
GROUP BY gtow_hand_id
ORDER BY sum(ev_loss_bb) DESC
"""

# per-hand meta for the Study link + combo (kept off the grouped scan to avoid a
# ledger_hands JOIN — both tables carry a `source` column, which would make the
# honesty predicate ambiguous).
_HAND_META_SQL = ("SELECT hero_hand, raw_path, position, preflop_depth_bb FROM ledger_hands "
                  "WHERE gtow_hand_id = $1")
_HAND_DECISIONS_SQL = """
SELECT street, decision_idx, taken_freq, spot_category, spot_leaf, hero_cat,
       villain_cat, ip_oop, position, ev_loss_bb, approx_flags,
       played_depth_bb, solver_depth_bb
FROM ledger_decisions
WHERE gtow_hand_id=$1 AND NOT excluded AND NOT discarded
  AND confidence >= 0.8
ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1
         WHEN 'turn' THEN 2 WHEN 'river' THEN 3 ELSE 9 END, decision_idx
"""

_CLEARED_SQL = ("SELECT max(cleared_at) c FROM drill_queue "
                "WHERE spot_leaf = $1 AND kind = 'drill' AND status = 'cleared'")
_OPEN_EXISTS_SQL = ("SELECT count(*) FROM drill_queue WHERE spot_leaf = $1 "
                    "AND kind = 'drill' AND status IN ('pending', 'prescribed')")
_REVIEW_EXISTS_SQL = ("SELECT count(*) FROM drill_queue "
                      "WHERE ref_hand_id = $1 AND kind = 'review'")


async def _build_drill_items(conn, since) -> list[dict]:
    rows = await conn.fetch(_drill_scan_sql(), since, LOSSY_MIN_BB,
                            QUEUE_DRILL_MIN_N, QUEUE_DRILL_MIN_TOTAL_BB)
    items: list[dict] = []
    for r in rows:
        leaf = r["spot_leaf"]
        open_exists = bool(await conn.fetchval(_OPEN_EXISTS_SQL, leaf))
        cleared_at = await conn.fetchval(_CLEARED_SQL, leaf)
        route = reopen_decision(open_exists, cleared_at, list(r["played_ats"]))
        if route == "skip":
            continue
        row = dict(r)
        bands = [dict(b) for b in await conn.fetch(lb.band_sql(since), leaf, since)]
        _restrict, depths = lb.choose_depths(bands)
        entries = _as_list(row["source_hands"])
        bias_key = row.get("spot_parent") or leaf
        if row.get("spot_parent"):
            bias_rows = await conn.fetch(
                lb.action_bias_sql("parent", since), bias_key, since)
            action_bias = dominant_action_bias(bias_rows)
        else:
            action_bias = dominant_action_bias(entries)
        url = await queue_drill_url_from_sources(conn, entries, depths=depths)
        items.append({
            "kind": "drill", "added_by": "auto", "source": "online",
            "spot_leaf": leaf, "spot_category": row["spot_category"],
            "label": drill_label(row, action_bias), "drill_url": url,
            "action_bias": action_bias, "bias_key": bias_key,
            "source_hands": entries,
            "total_ev_loss_bb": round(float(row["total_ev"]), 4),
        })
    return items


async def _build_review_items(conn, since) -> list[dict]:
    rows = await conn.fetch(_REVIEW_SCAN_SQL, since, QUEUE_REVIEW_MIN_BB)
    items: list[dict] = []
    for r in rows:
        if await conn.fetchval(_REVIEW_EXISTS_SQL, r["ref_hand_id"]):
            continue                       # 復盤過就是過了 (any status blocks)
        row = dict(r)
        meta = await conn.fetchrow(_HAND_META_SQL, r["ref_hand_id"])
        if meta:
            row["hero_hand"] = meta["hero_hand"]
            row["raw_path"] = meta["raw_path"]
            row["hero_pos"] = meta["position"] or row.get("hero_pos")
            row["preflop_depth_bb"] = meta["preflop_depth_bb"]
        decisions = [dict(d) for d in await conn.fetch(
            _HAND_DECISIONS_SQL, r["ref_hand_id"])]
        worst_dec = next((d for d in decisions
                          if d.get("street") == row.get("worst_street")
                          and int(d.get("decision_idx") or 0) == int(row.get("worst_idx") or 0)), None)
        anchor = low_frequency_anchor(
            decisions, row.get("worst_street"), row.get("worst_idx"))
        anchor_url = None
        if anchor:
            anchor_dec = next((d for d in decisions
                               if d.get("street") == anchor.get("street")
                               and int(d.get("decision_idx") or 0) == int(anchor.get("decision_idx") or 0)), None)
            anchor_url = _study_solution_url(dict(
                row, **(anchor_dec or {}), worst_street=anchor["street"],
                worst_idx=anchor["decision_idx"]))
        if anchor_url:
            row["review_anchor_street"] = anchor["street"]
        context = dict(row, **(worst_dec or {}),
                       worst_street=row.get("worst_street"),
                       worst_idx=row.get("worst_idx"),
                       max_ev=row.get("max_ev"),
                       hero_pos=row.get("hero_pos") or (worst_dec or {}).get("position"))
        items.append({
            "kind": "review", "added_by": "auto", "source": "online",
            "ref_hand_id": row["ref_hand_id"], "spot_leaf": row["spot_leaf"],
            "spot_category": row["spot_category"], "label": review_label(context),
            "drill_url": review_url(context), "review_anchor_url": anchor_url,
            "review_anchor_street": context.get("review_anchor_street"),
            "source_hands": _as_list(row["source_hands"]),
            "total_ev_loss_bb": round(float(row["total_ev"]), 4),
        })
    return items


_REVIEW_REFRESH_SQL = """
SELECT q.id, q.label, q.drill_url, q.review_anchor_url,
       q.review_anchor_street, q.ref_hand_id, q.source_hands,
       h.raw_path, h.position hero_pos, h.hero_hand,
       h.preflop_depth_bb, h.played_at
FROM drill_queue q
JOIN ledger_hands h ON h.gtow_hand_id = q.ref_hand_id
WHERE q.kind = 'review'
  AND ($1::boolean OR q.status IN ('pending', 'prescribed'))
ORDER BY q.id
"""


async def refresh_review_links(conn, include_all: bool = False) -> dict:
    """Rebuild persisted review URLs, path hints, and normalized labels.

    Weekly scans refresh open rows so old approximate links cannot remain in
    the visible queue.  The maintenance CLI can include cleared history too.
    """
    rows = await conn.fetch(_REVIEW_REFRESH_SQL, include_all)
    tally = {"checked": len(rows), "updated": 0, "unresolved": 0}
    for raw in rows:
        row = dict(raw)
        entries = _as_list(row.get("source_hands"))
        worst = entries[0] if entries else {}
        worst_street = worst.get("street")
        worst_idx = worst.get("decision_idx")
        decisions = [dict(d) for d in await conn.fetch(
            _HAND_DECISIONS_SQL, row["ref_hand_id"])]
        worst_dec = next((d for d in decisions
                          if d.get("street") == worst_street
                          and int(d.get("decision_idx") or 0) == int(worst_idx or 0)), None)
        if not worst_dec:
            # Preserve the existing button on a transient token/cache/archive
            # failure; the next weekly scan retries. New rows still use
            # review_url() and get the honest Analyze-table fallback.
            tally["unresolved"] += 1
            continue
        context = dict(row, **worst_dec, worst_street=worst_street,
                       worst_idx=worst_idx, max_ev=worst_dec.get("ev_loss_bb"),
                       hero_pos=row.get("hero_pos") or worst_dec.get("position"))
        loss_url = _study_solution_url(context)
        if not loss_url:
            tally["unresolved"] += 1
            loss_url = row.get("drill_url")

        anchor = low_frequency_anchor(decisions, worst_street, worst_idx)
        anchor_url = None
        anchor_street = None
        if anchor:
            anchor_url = _study_solution_url(dict(
                context, worst_street=anchor["street"],
                worst_idx=anchor["decision_idx"]))
            if anchor_url:
                anchor_street = anchor["street"]
        context["review_anchor_street"] = anchor_street
        label = review_label(context)
        current = (row.get("drill_url"), row.get("review_anchor_url"),
                   row.get("review_anchor_street"), row.get("label"))
        rebuilt = (loss_url, anchor_url, anchor_street, label)
        if rebuilt != current:
            await conn.execute(
                "UPDATE drill_queue SET drill_url=$2, review_anchor_url=$3, "
                "review_anchor_street=$4, label=$5 WHERE id=$1",
                row["id"], loss_url, anchor_url, anchor_street, label)
            tally["updated"] += 1
    return tally


async def refresh_trainer_links(conn, include_all: bool = False) -> dict:
    """Semantically rebuild persisted links from their ledger source decisions."""
    status = "" if include_all else "AND status IN ('pending', 'prescribed')"
    rows = await conn.fetch(
        f"SELECT id, drill_url, source_hands FROM drill_queue WHERE kind='drill' "
        f"{status} ORDER BY id")
    tally = {"checked": len(rows), "updated": 0, "unresolved": 0}
    for row in rows:
        entries = _as_list(row["source_hands"])
        normalized = await normalize_source_entries(conn, entries)
        rebuilt = await queue_drill_url_from_sources(conn, normalized)
        if not rebuilt:
            tally["unresolved"] += 1
        if rebuilt != row["drill_url"] or normalized != entries:
            await conn.execute(
                "UPDATE drill_queue SET drill_url=$2, source_hands=$3::jsonb, "
                "gtow_settings_hash=NULL, gtow_drill_synced_at=NULL WHERE id=$1",
                row["id"], rebuilt, json.dumps(normalized))
            tally["updated"] += 1
    return tally


async def backfill_live_preflop_pot_lines(conn) -> dict:
    """Persist real multiway preflop contributions into live parsed_json.

    Older rows only retained the HU-repaired solver line.  Rebuild the pot
    line from the raw live note so future queue URLs and GTOW Drill PATCHes
    remain correct even outside this process.
    """
    from live_flow import preflop_actions_for_pot_from_raw

    tally = {"scanned": 0, "updated": 0, "unresolved": 0}
    rows = await conn.fetch(
        "SELECT gtow_hand_id, raw_text, parsed_json FROM ledger_hands "
        "WHERE source='live' AND raw_text IS NOT NULL")
    for row in rows:
        tally["scanned"] += 1
        parsed = row["parsed_json"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            tally["unresolved"] += 1
            continue
        pot_line = preflop_actions_for_pot_from_raw(row["raw_text"], parsed)
        if not pot_line:
            tally["unresolved"] += 1
            continue
        if parsed.get("preflop_actions_for_pot") == pot_line:
            continue
        parsed["preflop_actions_for_pot"] = pot_line
        await conn.execute(
            "UPDATE ledger_hands SET parsed_json=$2::jsonb WHERE gtow_hand_id=$1",
            row["gtow_hand_id"], json.dumps(parsed, ensure_ascii=False))
        tally["updated"] += 1
    return tally


async def scan_online(conn, window_days: int = QUEUE_SCAN_WINDOW_DAYS,
                      dry_run: bool = False) -> dict:
    """Scan the rolling online window; enqueue drill + review items (idempotent).

    Called by scorecard's weekly job BEFORE it drains the queue (§5.4), and by
    the CLI's first run (which IS the owner's "past two months" backfill)."""
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    refresh = {"checked": 0, "updated": 0, "unresolved": 0}
    trainer_refresh = {"checked": 0, "updated": 0}
    if not dry_run:
        refresh = await refresh_review_links(conn)
        trainer_refresh = await refresh_trainer_links(conn)
    drill_items = await _build_drill_items(conn, since)
    review_items = await _build_review_items(conn, since)
    tally = {"merged": 0, "inserted": 0, "noop": 0}
    if not dry_run:
        tally = await enqueue(conn, drill_items + review_items)
    return {"since": since, "drill": drill_items, "review": review_items,
            "tally": tally, "refresh": refresh,
            "trainer_refresh": trainer_refresh}


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_scan(res: dict, dry_run: bool) -> None:
    print(f"\n== online scan since {res['since'].astimezone(TPE):%Y-%m-%d} "
          f"({'DRY-RUN' if dry_run else 'committed'}) ==")
    print(f"drill candidates: {len(res['drill'])}")
    for it in res["drill"]:
        print(f"  🎯 {it['spot_leaf']}  n={len(it['source_hands'])} "
              f"total={it['total_ev_loss_bb']:.1f}bb  "
              f"{'url' if it['drill_url'] else 'NO-URL'}  — {it['label']}")
    print(f"review candidates: {len(res['review'])}")
    for it in res["review"]:
        print(f"  🔍 {it['ref_hand_id']}  total={it['total_ev_loss_bb']:.1f}bb  "
              f"— {it['label']}")
    if not dry_run:
        print(f"enqueue tally: {res['tally']}")
        print(f"review-link refresh: {res['refresh']}")
        print(f"trainer-link refresh: {res['trainer_refresh']}")


async def _run(window_days: int, dry_run: bool) -> int:
    import asyncpg
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        res = await scan_online(conn, window_days=window_days, dry_run=dry_run)
        _print_scan(res, dry_run)
        return 0
    finally:
        await conn.close()


async def _run_refresh(include_all: bool) -> int:
    import asyncpg
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        tally = await refresh_review_links(conn, include_all=include_all)
        print(f"review-link refresh: {tally}")
        return 0
    finally:
        await conn.close()


async def _run_trainer_refresh(include_all: bool) -> int:
    import asyncpg
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        backfill = await backfill_live_preflop_pot_lines(conn)
        tally = await refresh_trainer_links(conn, include_all=include_all)
        print(f"live pot-line backfill: {backfill}")
        print(f"trainer-link refresh: {tally}")
        return 0
    finally:
        await conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="scan the online window and enqueue")
    ap.add_argument("--refresh-review-links", action="store_true",
                    help="rebuild persisted review links from real Analyze actions")
    ap.add_argument("--refresh-trainer-links", action="store_true",
                    help="rebuild persisted Trainer links from source decisions")
    ap.add_argument("--all-statuses", action="store_true",
                    help="with either refresh command, include cleared rows")
    ap.add_argument("--window-days", type=int, default=QUEUE_SCAN_WINDOW_DAYS)
    ap.add_argument("--dry-run", action="store_true", help="print candidates, no DB writes")
    a = ap.parse_args()
    if a.refresh_review_links:
        raise SystemExit(asyncio.run(_run_refresh(a.all_statuses)))
    if a.refresh_trainer_links:
        raise SystemExit(asyncio.run(_run_trainer_refresh(a.all_statuses)))
    if not a.scan:
        print("usage: queue_feed.py --scan [--window-days 60] [--dry-run] | "
              "--refresh-review-links [--all-statuses] | "
              "--refresh-trainer-links [--all-statuses]")
        raise SystemExit(1)
    raise SystemExit(asyncio.run(_run(a.window_days, a.dry_run)))


if __name__ == "__main__":
    main()
