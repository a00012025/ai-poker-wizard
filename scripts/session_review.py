#!/usr/bin/env python3
"""Session 復盤 — 剛打完（同步後）的單場診斷摘要。

North Star §7 不變量 11（回饋延遲預算：session 復盤「能過夜不過週」）、§4.2（理想
每個 session 後入帳）、§5.9（session 級 EV loss）。把最近一個 online session 的決策，
用與週記分卡相同的 EV 加權口徑，聚合成一則 Telegram 訊息：

  • 共幾手、平均 EV loss（單場、**不作趨勢判斷**）
  • EV loss 加總最多的 spot（→ 現在練 / 排入佇列）
  • EV loss 最高的 8 個具體決策（→ 復盤 / 排入佇列；練習由 spot/queue 處方承接）

不變量：**只讀不改本週焦點 spot**（中圈穩定性 §3）。排入動作走既有 `drill_queue`
（`kind='drill'`/`'review'`，`added_by='session'`），與週掃描產生同構的 rows。口徑沿用
`queue_feed._HONEST`（source 隔離 §5.2 由 `source='online'` 保證）。
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import queue_feed as qf  # noqa: E402  (reuse URL/label/enqueue helpers + _HONEST)
from action_bias import bias_suffix  # noqa: E402
from queue_feed import TPE, LOSSY_MIN_BB, pretty_hand  # noqa: E402
from scorecard import spot_desc_zh  # noqa: E402

TOP_SPOTS = 2
TOP_DECISIONS = 8

# ── session-scoped SQL ─────────────────────────────────────────────────────────
# Membership by session_id (authoritative), honesty by queue_feed._HONEST on the
# decisions table only — no ledger_hands JOIN, so `source` stays unambiguous.
_IN_SESSION = ("gtow_hand_id IN (SELECT gtow_hand_id FROM ledger_hands "
               "WHERE session_id = $1)")

_LATEST_SESSION_SQL = """
SELECT id, started_at, ended_at, duration_min, tournaments,
       max_concurrent_tables, hands_count
FROM ledger_sessions ORDER BY ended_at DESC LIMIT 1
"""
_SESSION_BY_ID_SQL = _LATEST_SESSION_SQL.replace(
    "ORDER BY ended_at DESC LIMIT 1", "WHERE id = $1")

_OVERVIEW_SQL = f"""
SELECT count(*) n,
       avg(ev_loss_bb) * 100 per100,
       coalesce(sum(ev_loss_bb), 0) total_bb,
       count(*) FILTER (WHERE ev_loss_bb >= {LOSSY_MIN_BB}) n_lossy
FROM ledger_decisions
WHERE {qf._HONEST} AND {_IN_SESSION}
"""

_HONESTY_SQL = f"""
SELECT count(*) FILTER (WHERE discarded) discarded_n,
       count(*) FILTER (WHERE confidence < 0.8 AND NOT excluded
                        AND spot_leaf IS NOT NULL AND source='online') low_conf_n
FROM ledger_decisions
WHERE source='online' AND {_IN_SESSION}
"""

# Descriptive top-N (NO queue min-N gate — a session digest, not the auto-scan).
_TOP_SPOTS_SQL = f"""
SELECT spot_leaf, spot_category,
       count(*) n, sum(ev_loss_bb) total_ev, avg(ev_loss_bb) avg_ev,
       mode() WITHIN GROUP (ORDER BY hero_cat)    hero_cat,
       mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
       mode() WITHIN GROUP (ORDER BY ip_oop)      ip_oop,
       mode() WITHIN GROUP (ORDER BY position)    hero_pos,
       jsonb_agg(jsonb_build_object(
           'hand_id', gtow_hand_id, 'street', street,
           'decision_idx', decision_idx, 'ev_loss_bb', ev_loss_bb,
           'taken_code', taken_code, 'best_code', best_code,
           'src', source) ORDER BY played_at) source_hands
FROM ledger_decisions
WHERE {qf._HONEST} AND ev_loss_bb >= {LOSSY_MIN_BB} AND {_IN_SESSION}
GROUP BY spot_leaf, spot_category
ORDER BY sum(ev_loss_bb) DESC
LIMIT {TOP_SPOTS}
"""

_TOP_DECISIONS_SQL = f"""
SELECT gtow_hand_id ref_hand_id, street, decision_idx, spot_leaf, spot_category,
       hero_cat, villain_cat, ip_oop, position hero_pos, ev_loss_bb, approx_flags,
       played_at, taken_code, best_code, correctness, pot_type, eff_stack, gametype,
       jsonb_build_array(jsonb_build_object(
           'hand_id', gtow_hand_id, 'street', street,
           'decision_idx', decision_idx, 'ev_loss_bb', ev_loss_bb,
           'taken_code', taken_code, 'best_code', best_code, 'src', source)) source_hands
FROM ledger_decisions
WHERE {qf._HONEST} AND ev_loss_bb >= {LOSSY_MIN_BB} AND {_IN_SESSION}
ORDER BY ev_loss_bb DESC, played_at DESC
LIMIT {TOP_DECISIONS}
"""

_HAND_META_SQL = ("SELECT hero_hand, boards, raw_path, position, preflop_depth_bb "
                  "FROM ledger_hands WHERE gtow_hand_id = $1")

STREET_LABELS = {"preflop": "PF", "flop": "Flop", "turn": "Turn", "river": "River"}


# ── resolve + compute ──────────────────────────────────────────────────────────
async def resolve_session(conn, session_id: int | None = None) -> dict | None:
    row = (await conn.fetchrow(_SESSION_BY_ID_SQL, session_id) if session_id
           else await conn.fetchrow(_LATEST_SESSION_SQL))
    return dict(row) if row else None


async def _spot_items(conn, session_id: int) -> list[dict]:
    rows = await conn.fetch(_TOP_SPOTS_SQL, session_id)
    out = []
    for r in rows:
        row = dict(r)
        entries = qf._as_list(row["source_hands"])
        bias = qf.dominant_action_bias(entries)
        url = await qf.queue_drill_url_from_sources(conn, entries, depths=None)
        out.append({
            "desc": drill_desc(row, bias),
            "total_ev": round(float(row["total_ev"]), 2),
            "n": row["n"],
            "drill_url": url,
            # payload the Phase-2 callback persists + hands to enqueue_one:
            "enqueue_item": {
                "kind": "drill", "added_by": "session", "source": "online",
                "spot_leaf": row["spot_leaf"], "spot_category": row["spot_category"],
                "label": drill_desc(row, bias), "drill_url": url,
                "action_bias": bias, "bias_key": row["spot_leaf"],
                "source_hands": entries,
                "total_ev_loss_bb": round(float(row["total_ev"]), 4),
            },
        })
    return out


async def _decision_items(conn, session_id: int) -> list[dict]:
    rows = await conn.fetch(_TOP_DECISIONS_SQL, session_id)
    out = []
    for r in rows:
        row = dict(r)
        meta = await conn.fetchrow(_HAND_META_SQL, row["ref_hand_id"])
        if meta:
            row["hero_hand"] = meta["hero_hand"]
            row["boards"] = meta["boards"]
            row["raw_path"] = meta["raw_path"]
            row["hero_pos"] = meta["position"] or row.get("hero_pos")
            row["preflop_depth_bb"] = meta["preflop_depth_bb"]
        entries = qf._as_list(row["source_hands"])
        hand_urls = qf.gtow_analyze_hands_urls([row["ref_hand_id"]])
        exact_url = hand_urls[0][0] if hand_urls else None
        drill_url = await qf.queue_drill_url_from_sources(conn, entries, depths=None)
        action_ctx = decision_action_context(row)
        row["max_ev"] = row.get("ev_loss_bb")
        row["worst_street"] = row.get("street")
        row["worst_idx"] = row.get("decision_idx")
        out.append({
            "combo": pretty_hand(row.get("hero_hand")),
            "position": row.get("hero_pos"),
            "depth": float(row["preflop_depth_bb"]) if row.get("preflop_depth_bb") is not None else None,
            "boards": row.get("boards") or "",
            "desc": hand_desc(row),
            "street_line": action_ctx.get("street_line") or "",
            "street_lines": action_ctx.get("street_lines") or [],
            "action_line": (action_ctx.get("action_line")
                            or action_line(row.get("taken_code"), row.get("best_code"))),
            "ev_loss": round(float(row.get("ev_loss_bb") or 0.0), 2),
            "exact_url": exact_url,
            "drill_url": drill_url,
            "enqueue_item": {
                "kind": "review", "added_by": "session", "source": "online",
                "ref_hand_id": row["ref_hand_id"], "spot_leaf": row.get("spot_leaf"),
                "spot_category": row.get("spot_category"),
                "label": qf.review_label(row), "review_url": exact_url,
                "review_anchor_url": None, "review_anchor_street": None,
                "source_hands": entries,
                "total_ev_loss_bb": round(float(row.get("ev_loss_bb") or 0.0), 4),
            },
        })
    return out


def drill_desc(row: dict, bias: dict | None) -> str:
    """Short zh label for a session-leak spot (reuses scorecard.spot_desc_zh)."""
    base = spot_desc_zh({
        "spot_category": row.get("spot_category"), "spot_leaf": row.get("spot_leaf"),
        "hero_cat": row.get("hero_cat"), "villain_cat": row.get("villain_cat"),
        "ip_oop": row.get("ip_oop"), "hero_pos": row.get("hero_pos"),
        "street": row.get("spot_category")})
    return base + bias_suffix(bias)


def hand_desc(row: dict) -> str:
    return spot_desc_zh({
        "spot_category": row.get("spot_category"), "spot_leaf": row.get("spot_leaf"),
        "hero_cat": row.get("hero_cat"), "villain_cat": row.get("villain_cat"),
        "ip_oop": row.get("ip_oop"), "hero_pos": row.get("hero_pos"),
        "street": row.get("spot_category")})


_ACTION_ZH = {
    "F": "Fold", "C": "Call", "X": "Check", "RAI": "All-in", "AI": "All-in",
}


def _street_of_gp(gp: dict) -> str:
    t = ((gp.get("real_game") or {}).get("current_street") or {}).get("type", "")
    t = (t or "").lower()
    return t if t in {"preflop", "flop", "turn", "river"} else "preflop"


def _norm_code(code: str | None) -> str:
    if not code:
        return ""
    c = str(code).upper()
    if c == "RAI":
        return "AI"
    if c.startswith("B"):
        return "R" + c[1:] if len(c) > 1 else "R"
    return c


def _pct(action: dict | None) -> str:
    if not action:
        return ""
    raw = action.get("betsize_by_pot")
    if raw in (None, ""):
        return ""
    try:
        return f" {round(float(raw) * 100):.0f}%"
    except (TypeError, ValueError):
        return ""


def _action_obj_zh(action: dict | None, street: str) -> str:
    if not action:
        return "?"
    code = _norm_code(action.get("code"))
    display = (action.get("display_name") or "").upper()
    if code in {"F", "C", "X", "AI"}:
        return action_zh(code)
    if code.startswith("R"):
        # GTOW uses R* codes for both preflop raises and postflop bets/raises.
        # The display_name tells us whether this node is a first bet or a raise.
        verb = "Raise" if street == "preflop" or display == "RAISE" else "Bet"
        return verb + ("" if street == "preflop" else _pct(action))
    return action_zh(code)


def _format_history_action(action: dict, hero_pos: str, street: str) -> str:
    pos = action.get("position") or "?"
    who = "Hero" if pos == hero_pos else pos
    return f"{who} {_action_obj_zh(action, street)}"


def _street_board(boards: str, street: str) -> str:
    if not boards:
        return ""
    if street == "flop":
        return boards[:6] if len(boards) >= 6 else ""
    if street == "turn":
        return boards[6:8] if len(boards) >= 8 else ""
    if street == "river":
        return boards[8:10] if len(boards) >= 10 else ""
    return ""


def _load_detail(raw_path: str | None) -> dict | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def decision_action_context(row: dict) -> dict:
    """Return human-readable action history + taken/best action for one decision.

    Uses archived GTOW detail when present. Falls back to code-only labels when
    raw detail is unavailable, so rendering never depends on filesystem state.
    """
    detail = _load_detail(row.get("raw_path"))
    if not detail:
        return {}

    target_street = row.get("street") or ""
    target_idx = int(row.get("decision_idx") or 0)
    hero_pos = row.get("hero_pos") or row.get("position") or ""
    boards = row.get("boards") or ""
    gps = ((detail.get("game_analysis") or {}).get("game_points") or [])
    actions: dict[str, list[dict]] = {s: [] for s in ("preflop", "flop", "turn", "river")}
    hero_count: dict[str, int] = {s: 0 for s in ("preflop", "flop", "turn", "river")}

    for gp in gps:
        rga = gp.get("real_game_action") or {}
        sga = gp.get("solved_game_action") or rga
        street = _street_of_gp(gp)
        pos = rga.get("position", "")
        avail = (gp.get("analysis_solved") or {}).get("available_actions") or []
        is_hero = pos == hero_pos and any(a.get("selected") for a in avail)

        if is_hero:
            idx = hero_count[street]
            if street == target_street and idx == target_idx:
                sel = next((a for a in avail if a.get("selected")), None)
                best = next((a for a in avail if a.get("correctness") == "BEST_MOVE"), None)
                if best is None and avail:
                    best = max(avail, key=lambda a: float(a.get("ev") or 0))
                parts = []
                for s in ("preflop", "flop", "turn", "river"):
                    if s == "preflop" and street != "preflop":
                        continue
                    if s not in actions:
                        continue
                    acts = actions[s]
                    if s == street:
                        acts = list(acts)
                    elif not acts:
                        continue
                    if s == street and not acts:
                        break
                    board = _street_board(boards, s)
                    label = STREET_LABELS[s] + (f" {board}" if board else "")
                    parts.append(f"{label}: " + ", ".join(
                        _format_history_action(a, hero_pos, s) for a in acts))
                    if s == street:
                        break
                selected_action = (sel or {}).get("action") or sga
                best_action = (best or {}).get("action")
                return {
                    "street_lines": parts,
                    "street_line": " / ".join(parts),
                    "action_line": f"{_action_obj_zh(selected_action, street)}"
                                   f"→應{_action_obj_zh(best_action, street)}",
                }
            hero_count[street] = idx + 1

        actions[street].append(sga)
    return {}


def action_zh(code: str | None) -> str:
    if not code:
        return "?"
    c = _norm_code(code)
    if c.startswith("R"):
        return "Raise"
    return _ACTION_ZH.get(c, str(code))


def action_line(taken: str | None, best: str | None) -> str:
    return f"{action_zh(taken)}→應{action_zh(best)}"


def depth_label(depth: float | None) -> str:
    if depth is None:
        return ""
    d = float(depth)
    return f"{d:.0f}bb" if abs(d - round(d)) < 0.2 else f"{d:.1f}bb"


def decision_mark(i: int) -> str:
    keycaps = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    return keycaps[i] if i < len(keycaps) else f"{i+1}."


async def compute(conn, session: dict) -> dict:
    sid = session["id"]
    ov = await conn.fetchrow(_OVERVIEW_SQL, sid)
    hon = await conn.fetchrow(_HONESTY_SQL, sid)
    spots = await _spot_items(conn, sid)
    decisions = await _decision_items(conn, sid)
    return {
        "session_id": sid,
        "started_at": session["started_at"],
        "ended_at": session["ended_at"],
        "n_hands": session["hands_count"],
        "n_decisions": ov["n"] or 0,
        "per100": float(ov["per100"] or 0.0),
        "total_bb": float(ov["total_bb"] or 0.0),
        "n_lossy": ov["n_lossy"] or 0,
        "top_spots": spots,
        "top_decisions": decisions,
        "honesty": {"discarded_n": hon["discarded_n"] or 0,
                    "low_conf_n": hon["low_conf_n"] or 0},
        "empty": (ov["n_lossy"] or 0) == 0,
    }


# ── render ─────────────────────────────────────────────────────────────────────
def should_auto_send(data: dict) -> bool:
    """Auto-push a session review after a sync only when there's something worth
    reviewing — skip clean/empty sessions to respect the 依從 / time budget
    (§7-11). Manual /review still shows the clean digest on demand."""
    return not data.get("empty", True)


def _session_span(started, ended) -> str:
    # ledger_sessions timestamps are GTOW/PokerCraft local wall-clock values.
    # Do not convert timezone again, or a 19:00 session renders as 03:00.
    s = started
    e = ended
    if s.date() == e.date():
        return f"{s.strftime('%-m/%-d')} {s.strftime('%H:%M')}–{e.strftime('%H:%M')}"
    return f"{s.strftime('%-m/%-d %H:%M')} – {e.strftime('%-m/%-d %H:%M')}"


def render_tg(d: dict) -> dict:
    """Owner-facing session review (HTML parse_mode) + inline buttons.

    Pure function of ``compute()``'s output — no DB, no network. Returns
    ``{"html": str, "buttons": rows}`` where rows is list[list[button dict]].
    Buttons: URL (現在練/復盤) + callback (排入). callback_data stays
    <64B via `srd|srv:{session_id}:{i}` index keys.
    """
    sid = d["session_id"]
    span = _session_span(d["started_at"], d["ended_at"])
    L = [f"🔍 <b>這場復盤</b> · {escape(span)}", ""]

    if d["empty"]:
        L.append(f"這場 <b>{d['n_hands']}</b> 手，幾乎都打對了 — 沒有值得復盤的漏損 👍")
        return {"html": "\n".join(L), "buttons": []}

    L.append(f"這場 <b>{d['n_hands']}</b> 手，平均 <b>{d['per100']:.1f} bb/100 決策</b>，"
             f"本場合計漏 <b>{d['total_bb']:.1f} bb</b>。")

    buttons: list[list[dict]] = []

    if d["top_spots"]:
        L.append("")
        L.append("<b>EV Loss 最多的情境（加總）</b>")
        for i, s in enumerate(d["top_spots"], 1):
            L.append(f"{i}. {escape(s['desc'])} — <b>{s['total_ev']:.1f} bb</b>（{s['n']} 手）")
        for i, s in enumerate(d["top_spots"]):
            row = []
            if s.get("drill_url"):
                row.append({"text": f"🎯 現在練：{s['desc'][:16]}", "url": s["drill_url"]})
            row.append({"text": "📥 排入佇列", "callback_data": f"srd:{sid}:{i}"})
            buttons.append(row)

    top_decisions = d.get("top_decisions") or []
    if top_decisions:
        L.append("")
        L.append(f"<b>最值得回看的 {len(top_decisions)} 個決策</b>")
        for i, h in enumerate(top_decisions):
            m = decision_mark(i)
            combo = f"{escape(h['combo'])} " if h.get("combo") else ""
            pos = escape(str(h.get("position") or "?"))
            depth = depth_label(h.get("depth"))
            depth_part = f" {escape(depth)}" if depth else ""
            L.append(f"{m} {combo}{pos}{depth_part}｜{escape(h['desc'])}")
            street_lines = h.get("street_lines") or []
            if not street_lines and (h.get("street_line") or h.get("boards")):
                street_lines = [h.get("street_line") or h.get("boards")]
            for j, street_line in enumerate(street_lines):
                suffix = (f"｜<b>{escape(h['action_line'])}</b>｜−<b>{h['ev_loss']:.2f}bb</b>"
                          if j == len(street_lines) - 1 else "")
                L.append(f"{escape(street_line)}{suffix}")
            if not street_lines:
                L.append(f"<b>{escape(h['action_line'])}</b>｜−<b>{h['ev_loss']:.2f}bb</b>")
            L.append("")
        for i, h in enumerate(top_decisions):
            m = decision_mark(i)
            row = []
            if h.get("exact_url"):
                row.append({"text": f"{m} 📖 復盤", "url": h["exact_url"]})
            # Decision-level Trainer deep-links are very long; 8 of them can exceed
            # Telegram's inline-keyboard payload limit and make the whole review fail.
            # Keep concrete review links here; practice is still available via the
            # aggregated top-spot drill buttons and by enqueuing this decision.
            row.append({"text": f"{m} 📥 入 queue", "callback_data": f"srv:{sid}:{i}"})
            buttons.append(row)

    hon = d["honesty"]
    caveat = ["⚠️ 翻後以 chipEV 評分（泡沫/FT 有 ICM 誤差）"]
    if hon["discarded_n"]:
        caveat.append(f"limp {hon['discarded_n']} 手未計")
    if hon["low_conf_n"]:
        caveat.append(f"{hon['low_conf_n']} 個低信心未計")
    caveat.append("只含已上傳 Analyzer 的手")
    L.append("")
    L.append(" · ".join(caveat))

    return {"html": "\n".join(L), "buttons": buttons}


# ── CLI (dry-run: no send, no enqueue) ─────────────────────────────────────────
async def _cli(session_id: int | None, as_json: bool):
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        session = await resolve_session(conn, session_id)
        if not session:
            print("no session found (run ledger_sessions.py --rebuild first)")
            return
        data = await compute(conn, session)
    finally:
        await conn.close()

    if as_json:
        def _ser(o):
            return o.isoformat() if hasattr(o, "isoformat") else str(o)
        print(json.dumps(data, ensure_ascii=False, indent=2, default=_ser))
        return

    out = render_tg(data)
    print("=" * 60)
    print(f"session_id={data['session_id']}  hands={data['n_hands']}  "
          f"decisions={data['n_decisions']}  lossy={data['n_lossy']}")
    print("=" * 60)
    print("── TELEGRAM MESSAGE (HTML) ──")
    print(out["html"])
    print("\n── INLINE BUTTONS ──")
    for r, brow in enumerate(out["buttons"]):
        cells = []
        for b in brow:
            if "url" in b:
                cells.append(f"[{b['text']}] → {b['url'][:70]}…")
            else:
                cells.append(f"[{b['text']}] ⟳ {b['callback_data']}")
        print(f"  row {r}: " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser(description="Session 復盤 (dry-run digest)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true", help="most recent online session")
    g.add_argument("--session-id", type=int, help="specific ledger_sessions.id")
    ap.add_argument("--json", action="store_true", help="structured output")
    a = ap.parse_args()
    asyncio.run(_cli(a.session_id, a.json))


if __name__ == "__main__":
    main()
