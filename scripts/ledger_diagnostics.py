#!/usr/bin/env python3
"""EV-weighted diagnostics over ledger rows. Pure functions + thin fetchers."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

TPE = ZoneInfo("Asia/Taipei")


def _week_label(dt: datetime) -> str:
    y, w, _ = dt.astimezone(TPE).isocalendar()
    return f"{y}-W{w:02d}"


def _included(decisions):
    return [d for d in decisions if not d.get("excluded")]


def weekly_series(decisions: list[dict]) -> list[dict]:
    by_week: dict[str, list[float]] = defaultdict(list)
    for d in _included(decisions):
        by_week[_week_label(d["played_at"])].append(d["ev_loss_bb"] or 0.0)
    out = []
    for wk in sorted(by_week):
        losses = by_week[wk]
        out.append({"week": wk, "n": len(losses), "total_bb": sum(losses),
                    "ev_loss_per_100": sum(losses) / len(losses) * 100})
    return out


def classify_leak(spot_decisions: list[dict]) -> tuple[str, str]:
    """Boundary vs knowledge split over the OFFICIAL taxonomy's dimensions
    (§4.2): depth_band + board_suit — the legacy `texture` is no longer
    written to the ledger."""
    total = sum(d["ev_loss_bb"] or 0 for d in spot_decisions) or 1e-9
    for dim in ("depth_band", "board_suit"):
        slices: dict[str, list[dict]] = defaultdict(list)
        for d in spot_decisions:
            if d.get(dim):
                slices[d[dim]].append(d)
        # A single populated slice carries no localization signal — the leak
        # can only be "boundary" (rule too coarse) if the loss concentrates in
        # one of several observed sub-slices.
        if len(slices) < 2:
            continue
        for key, ds in slices.items():
            share = sum(x["ev_loss_bb"] or 0 for x in ds) / total
            if share >= 0.7 and len(ds) >= 10:
                return "boundary", f"{dim}={key} ({share:.0%} of loss)"
    return "knowledge", "loss spread across slices"


def leak_board(decisions: list[dict], min_n: int = 25,
               weeks_window: int = 4) -> dict:
    """EV-ranked leak cells over the official action-line taxonomy:
    cell = spot_leaf × depth_band; leak type classified per spot_category."""
    inc = _included(decisions)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for d in inc:
        cells[(d["spot_leaf"], d["depth_band"])].append(d)
        by_cat[d.get("spot_category")].append(d)

    latest = max((d["played_at"] for d in inc), default=datetime.now(timezone.utc))
    cur_lo = latest - timedelta(weeks=weeks_window)
    prev_lo = latest - timedelta(weeks=2 * weeks_window)

    def per100(ds):
        return (sum(d["ev_loss_bb"] or 0 for d in ds) / len(ds) * 100) if ds else 0.0

    ranked, insufficient = [], []
    for (leaf, band), ds in cells.items():
        total = sum(d["ev_loss_bb"] or 0 for d in ds)
        cur = [d for d in ds if d["played_at"] >= cur_lo]
        prev = [d for d in ds if prev_lo <= d["played_at"] < cur_lo]
        ltype, sdesc = classify_leak(by_cat[ds[0].get("spot_category")])
        row = {"spot_leaf": leaf, "depth_band": band, "total_bb": total,
               "n": len(ds), "per100": per100(ds),
               "trend": per100(cur) - per100(prev),
               "trend_n": (len(cur), len(prev)),
               "leak_type": ltype, "slice_desc": sdesc}
        (ranked if len(ds) >= min_n else insufficient).append(row)
    ranked.sort(key=lambda r: -r["total_bb"])
    insufficient.sort(key=lambda r: -r["total_bb"])
    return {"cells": ranked, "insufficient": insufficient,
            "excluded_n": len(decisions) - len(inc)}


def most_expensive_hands(hands: list[dict], k: int = 3) -> list[dict]:
    return sorted(hands, key=lambda h: -(h.get("total_ev_loss_bb") or 0))[:k]


def pick_focus(board: dict, k: int = 2) -> list[dict]:
    return board["cells"][:k]


def session_correlations(decisions, hands, sessions) -> dict:
    inc = _included(decisions)
    hand_by_id = {h["gtow_hand_id"]: h for h in hands}
    sess_start = {s["id"]: s["started_at"] for s in sessions}

    def bucket(key_fn):
        b: dict = defaultdict(list)
        for d in inc:
            h = hand_by_id.get(d.get("gtow_hand_id"))
            if not h:
                continue
            key = key_fn(d, h)
            if key is not None:
                b[key].append(d["ev_loss_bb"] or 0)
        return [{"key": key, "n": len(v), "per100": sum(v) / len(v) * 100,
                 "low_n": len(v) < 20}
                for key, v in sorted(b.items())]

    by_hour = bucket(lambda d, h: (
        int((h["played_at"] - sess_start[h["session_id"]]).total_seconds() // 3600)
        if h.get("session_id") in sess_start else None))
    sess_tables = {s["id"]: s.get("max_concurrent_tables") for s in sessions}
    by_tables = bucket(lambda d, h: sess_tables.get(h.get("session_id")))

    bad_beats = sorted(h["played_at"] for h in hands if (h.get("winloss_bb") or 0) < -20)
    def in_window(t):
        import bisect
        i = bisect.bisect_left(bad_beats, t) - 1
        return i >= 0 and (t - bad_beats[i]).total_seconds() <= 900 and t != bad_beats[i]
    win = [d["ev_loss_bb"] or 0 for d in inc
           if in_window(hand_by_id.get(d.get("gtow_hand_id"), {}).get("played_at", d["played_at"]))]
    base = [d["ev_loss_bb"] or 0 for d in inc]
    post_bb = {"n": len(win),
               "per100": (sum(win) / len(win) * 100) if win else 0.0,
               "baseline_per100": (sum(base) / len(base) * 100) if base else 0.0,
               "low_n": len(win) < 20}
    return {"by_hour": by_hour, "by_tables": by_tables, "post_bad_beat": post_bb}


async def fetch_decisions(conn, since=None):
    # source isolation (§5.2): diagnostics/weekly series are online-only —
    # live hands are selectively recorded and would bias every aggregate.
    q = ("SELECT * FROM ledger_decisions WHERE source='online' AND confidence >= 0.8"
         + (" AND played_at >= $1" if since else ""))
    rows = await (conn.fetch(q, since) if since else conn.fetch(q))
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("approx_flags"), str):
            import json as _json
            d["approx_flags"] = _json.loads(d["approx_flags"])
        out.append(d)
    return out


async def fetch_hands(conn, since=None):
    q = ("SELECT * FROM ledger_hands WHERE source='online'"
         + (" AND played_at >= $1" if since else ""))
    return [dict(r) for r in await (conn.fetch(q, since) if since else conn.fetch(q))]


async def fetch_sessions(conn):
    return [dict(r) for r in await conn.fetch("SELECT * FROM ledger_sessions")]
