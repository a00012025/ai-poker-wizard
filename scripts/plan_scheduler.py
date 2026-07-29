#!/usr/bin/env python3
"""Weekly-plan scheduling: what actually gets put in front of the owner.

Design: docs/superpowers/specs/2026-07-29-weekly-plan-freshness-design.md

The weekly plan used to replay itself. The 90d focus ranking barely moved
week to week, a prescription only left the queue via a manual ✔, and two
stale review rows owned the review quota indefinitely. This module owns the
one policy that decides the weekly slate:

  * freshness buckets  — fresh / relapse / backlog, backlog rotates
  * reserved tracks    — online 3 seats, live 2 seats
  * focus cooldown     — a treated spot returns only on fresh evidence
  * GTOW auto-close    — a drill whose bound attempt hit both targets retires

Two invariants shape it. §5.2 source isolation: live hands are a selectively
recorded (biased) sample, so live never enters an aggregate ranked against
online — "live weighs more" is implemented as reserved seats, never as a
multiplier. §7.3: ranking inside a bucket stays EV-weighted; freshness only
decides which bucket a row is in (the backlog's rotation order is scheduling
fairness, not an importance claim).

The pure functions carry the policy and are unit-tested without a DB; the
async helpers only fetch what those functions need.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── policy constants ─────────────────────────────────────────────────────────
QUEUE_SLOTS = 5
TRACK_SLOTS = {"online": 3, "live": 2}
REVIEW_RESERVE = {"online": 1, "live": 0}  # live produces no review rows today
BACKLOG_SLOTS_PER_TRACK = 1

# A re-surfaced drill earns its way back to a prime seat with this many new
# lossy decisions played after it was last put in front of the owner.
RELAPSE_MIN_N = 2
LOSSY_MIN_BB = 0.10  # == queue_feed.LOSSY_MIN_BB / live QUEUE_EV_MIN

# Focus (the 1-2 headline spots) is stickier than the queue: it is a whole
# week's training theme, so it needs both a cooldown and fresh evidence.
FOCUS_COOLDOWN_WEEKS = 2
FOCUS_RELAPSE_MIN_N = 10

_TRACKS = ("online", "live")


# ── pure policy ──────────────────────────────────────────────────────────────
def _ev(row: dict) -> float:
    try:
        return float(row.get("total_ev_loss_bb") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_entries(value):
    from queue_feed import _as_list

    return _as_list(value)


def resolve_track(row: dict, hand_sources: dict[str, str]) -> str:
    """Which track a queue row belongs to: 'online' or 'live'.

    ``drill_queue.source`` records how a row was ADDED, not where the hand was
    played (``queue_feed.resolve_queue_source_hands`` documents this: manual
    drills commonly point at online hands, and a line the owner added off a
    live hand is stored as 'manual'). Only ``ledger_hands.source`` is
    authoritative, so decide by majority over the row's source hands and fall
    back to the queue column only when the ledger says nothing.

    Ties go to live: the reserved live seats exist precisely because live
    evidence is scarce, and a tie means live evidence is present.
    """
    votes = {"online": 0, "live": 0}
    for entry in _as_entries(row.get("source_hands")):
        source = hand_sources.get(entry.get("hand_id"))
        if source in votes:
            votes[source] += 1
    ref = hand_sources.get(row.get("ref_hand_id"))
    if ref in votes:
        votes[ref] += 1
    if votes["live"] or votes["online"]:
        return "live" if votes["live"] >= votes["online"] else "online"
    return "live" if row.get("source") == "live" else "online"


def classify_freshness(row: dict) -> str:
    """'fresh' | 'relapse' | 'backlog' for one annotated queue row.

    ``new_evidence_n`` is the count of lossy decisions on this row's spot
    played AFTER it was last surfaced (see :func:`annotate_rows`).
    """
    if int(row.get("surfaced_count") or 0) == 0:
        return "fresh"
    # 端上桌一次就算送達: a review item is one specific hand, and a hand cannot
    # re-offend. Once shown it only ever competes for the rotating seat.
    if row.get("kind") == "review":
        return "backlog"
    if int(row.get("new_evidence_n") or 0) >= RELAPSE_MIN_N:
        return "relapse"
    return "backlog"


def _rotation_key(row: dict):
    """Backlog order: longest-unseen first. Never-stamped rows sort first."""
    stamped = row.get("last_surfaced_at")
    if stamped is None:
        return (0, 0.0)
    return (1, stamped.timestamp())


def _take_active(active: list[dict], slots: int, review_reserve: int) -> list[dict]:
    """Fill `slots` from the active rows, holding one seat for a review.

    Without the reserve a low-EV review can never win a seat against systematic
    drills, and single-hand disasters would stop being scheduled at all.
    """
    if slots <= 0:
        return []
    reviews = [r for r in active if r.get("kind") == "review"]
    picked = reviews[:max(0, review_reserve)][:slots]
    chosen = {id(r) for r in picked}
    rest = sorted((r for r in active if id(r) not in chosen),
                  key=_ev, reverse=True)
    picked.extend(rest[:slots - len(picked)])
    # Present highest EV first; the reserved review keeps its seat regardless.
    return sorted(picked, key=_ev, reverse=True)


def select_weekly_slate(rows: list[dict], slots: int = QUEUE_SLOTS) -> dict:
    """Pick this week's practice slate from the open queue rows.

    Returns ``{"picked": [...], "backlog_total": int}``. Each picked row is a
    copy carrying ``bucket`` and ``track``.

    Deliberately does NOT pad: when there is no fresh or relapsed work the
    plan comes back short and says so. Filling the gap with backlog is exactly
    the replay this module exists to stop, and "this week you have no new
    leaks" is a real signal, not an empty slot.
    """
    annotated = []
    for row in rows:
        item = dict(row)
        item.setdefault("track", "online")
        item["bucket"] = classify_freshness(item)
        annotated.append(item)

    picked: list[dict] = []
    for track in _TRACKS:
        trows = [r for r in annotated if r.get("track") == track]
        active = [r for r in trows if r["bucket"] != "backlog"]
        backlog = sorted([r for r in trows if r["bucket"] == "backlog"],
                         key=_rotation_key)
        track_slots = TRACK_SLOTS.get(track, 0)
        # The rotating seat is RESERVED, not leftover. If fresh work could
        # fill the track every week the backlog would never rotate at all, and
        # an unpracticed 20bb prescription would vanish behind a queue of small
        # new ones — which is the §14.2 failure this module has to avoid. When
        # there is no backlog to rotate the seat goes back to fresh work.
        reserved = min(BACKLOG_SLOTS_PER_TRACK, len(backlog), track_slots)
        taken = _take_active(active, track_slots - reserved,
                             REVIEW_RESERVE.get(track, 0))
        room = min(BACKLOG_SLOTS_PER_TRACK, track_slots - len(taken))
        taken.extend(backlog[:max(0, room)])
        picked.extend(taken)

    # A track with no candidates yields its unused seats to the other one, so a
    # week with no live session still gets a full plan. The per-track backlog
    # cap is absolute and is never relaxed by a yield.
    if len(picked) < slots:
        chosen = {id(r) for r in picked}
        spare = sorted((r for r in annotated
                        if r["bucket"] != "backlog" and id(r) not in chosen),
                       key=_ev, reverse=True)
        picked.extend(spare[:slots - len(picked)])

    return {"picked": picked[:slots],
            "backlog_total": sum(1 for r in annotated
                                 if r["bucket"] == "backlog")}


def focus_cooldown_blocked(diagnosis_key: str, prescribed_at, now,
                           post_n: int, post_per100, global_per100) -> bool:
    """Should this diagnosis key be kept out of the focus slot?

    A key that has never been prescribed is never blocked. Once prescribed it
    is blocked for :data:`FOCUS_COOLDOWN_WEEKS`, and after that it returns ONLY
    on fresh evidence: enough post-prescription decisions to judge, and a
    post-prescription EV loss still at or above the player's global average.

    Time alone never resurrects a treated spot. That is the fix for the frozen
    90d ranking: a spot whose damage lives entirely in old weeks stops being
    re-prescribed once treated, and comes back the moment it leaks again.
    """
    if prescribed_at is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if prescribed_at.tzinfo is None:
        prescribed_at = prescribed_at.replace(tzinfo=reference.tzinfo)
    if (reference - prescribed_at).days < FOCUS_COOLDOWN_WEEKS * 7:
        return True
    if int(post_n or 0) < FOCUS_RELAPSE_MIN_N:
        return True
    return float(post_per100 or 0.0) < float(global_per100 or 0.0)


def drill_attempt_passed(row: dict, attempt) -> bool:
    """Did the bound GTOW Drill attempt clear both targets?

    Identical predicate to the '✅ 本次 Drill 已達標' line on the Telegram drill
    card — the weekly job and the bot must never disagree about what passing is.
    """
    if attempt is None:
        return False
    target_hands = int(row.get("gtow_target_hands") or 30)
    target_score = float(row.get("gtow_target_score") or 0.90)
    return (int(getattr(attempt, "total_hands", 0) or 0) >= target_hands
            and float(getattr(attempt, "gto_score", 0.0) or 0.0) >= target_score)


# ── async helpers (thin: they only fetch what the pure policy needs) ──────────
_HAND_SOURCE_SQL = ("SELECT gtow_hand_id, source FROM ledger_hands "
                    "WHERE gtow_hand_id = ANY($1::text[])")

# Same honesty predicate as the leak board and the queue scan, minus the source
# clause: relapse is judged within the row's own track (§5.2 — a live row is
# never revived by online evidence, or vice versa).
_NEW_EVIDENCE_SQL = """
SELECT count(*) FROM ledger_decisions
WHERE spot_leaf = $1 AND source = $2 AND played_at > $3
  AND ev_loss_bb >= $4 AND NOT excluded AND NOT discarded
  AND spot_leaf IS NOT NULL AND confidence >= 0.8
"""


async def annotate_rows(conn, rows: list[dict]) -> list[dict]:
    """Attach ``track`` and ``new_evidence_n`` to open queue rows."""
    hand_ids = set()
    for row in rows:
        for entry in _as_entries(row.get("source_hands")):
            if entry.get("hand_id"):
                hand_ids.add(entry["hand_id"])
        if row.get("ref_hand_id"):
            hand_ids.add(row["ref_hand_id"])
    hand_sources = {}
    if hand_ids:
        hand_sources = {r["gtow_hand_id"]: r["source"]
                        for r in await conn.fetch(_HAND_SOURCE_SQL,
                                                  list(hand_ids))}
    out = []
    for row in rows:
        item = dict(row)
        item["track"] = resolve_track(item, hand_sources)
        stamped = item.get("last_surfaced_at")
        if stamped and item.get("spot_leaf") and item.get("kind") != "review":
            item["new_evidence_n"] = int(await conn.fetchval(
                _NEW_EVIDENCE_SQL, item["spot_leaf"], item["track"], stamped,
                LOSSY_MIN_BB) or 0)
        else:
            item["new_evidence_n"] = 0
        out.append(item)
    return out


_BOUND_DRILLS_SQL = """
SELECT id, gtow_drill_id, gtow_training_started_at, gtow_target_hands,
       gtow_target_score, label
FROM drill_queue
WHERE status IN ('pending', 'prescribed') AND kind = 'drill'
  AND gtow_drill_id IS NOT NULL AND gtow_training_started_at IS NOT NULL
"""


async def autoclose_passed_drills(conn, client) -> dict:
    """Retire drills whose bound GTOW attempt met both targets.

    Fails soft on purpose: an expired GTOW token or a flaky practice API must
    never take the weekly plan down with it — the worst case is that a passed
    drill lingers one more week and can still be cleared by hand.
    """
    import asyncio

    tally = {"checked": 0, "closed": 0, "skipped": 0}
    rows = [dict(r) for r in await conn.fetch(_BOUND_DRILLS_SQL)]
    tally["checked"] = len(rows)
    if not rows:
        return tally
    started_ats = {str(r["gtow_drill_id"]): r["gtow_training_started_at"]
                   for r in rows}
    try:
        attempts = await asyncio.to_thread(client.attempts_by_drill, started_ats)
    except Exception as exc:  # noqa: BLE001 - fail soft, see docstring
        log.warning("GTOW auto-close skipped (practice API unavailable): %s", exc)
        tally["skipped"] = len(rows)
        return tally
    for row in rows:
        if not drill_attempt_passed(row, attempts.get(str(row["gtow_drill_id"]))):
            continue
        await conn.execute(
            "UPDATE drill_queue SET status='cleared', cleared_at=NOW(), "
            "clear_reason='drill_passed' WHERE id=$1 "
            "AND status IN ('pending','prescribed')", row["id"])
        tally["closed"] += 1
        log.info("auto-closed drill queue row %s (%s): GTOW target met",
                 row["id"], row.get("label"))
    return tally


async def mark_surfaced(conn, queue_ids: list[int], week: str) -> None:
    """Record that these rows were put in front of the owner this week."""
    ids = [int(i) for i in (queue_ids or [])]
    if not ids:
        return
    await conn.execute(
        "UPDATE drill_queue SET surfaced_count = surfaced_count + 1, "
        "last_surfaced_at = NOW(), last_surfaced_week = $2 "
        "WHERE id = ANY($1::bigint[])", ids, week)


_FOCUS_POST_SQL = """
SELECT count(*) n, avg(ev_loss_bb)*100 per100
FROM ledger_decisions
WHERE {column} = $1 AND source='online' AND played_at >= $2
  AND NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL
  AND confidence >= 0.8
"""


async def focus_exclusions(conn, history: list[dict], now=None,
                           global_per100: float = 0.0) -> set[str]:
    """Diagnosis keys that must not be prescribed as this week's focus.

    ``history`` is the ``coach_focus`` rows (most recent first), each carrying
    the prescribed families and when they were prescribed.
    """
    reference = now or datetime.now(timezone.utc)
    seen: dict[str, tuple] = {}
    for entry in history or []:
        key = entry.get("diagnosis_key") or entry.get("spot_leaf")
        if key and key not in seen:
            seen[key] = (entry.get("prescribed_at"),
                         entry.get("diagnosis_level") or "leaf")
    blocked = set()
    for key, (prescribed_at, level) in seen.items():
        if prescribed_at is None:
            continue
        sql = _FOCUS_POST_SQL.format(
            column="spot_parent" if level == "parent" else "spot_leaf")
        row = await conn.fetchrow(sql, key, prescribed_at)
        if focus_cooldown_blocked(
                key, prescribed_at, reference, int(row["n"] or 0),
                row["per100"], global_per100):
            blocked.add(key)
    return blocked
