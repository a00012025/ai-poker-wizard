#!/usr/bin/env python3
"""Session 復盤 — 剛打完（同步後）的單場診斷摘要。

North Star §7 不變量 11（回饋延遲預算：session 復盤「能過夜不過週」）、§4.2（理想
每個 session 後入帳）、§5.9（session 級 EV loss）。把最近一個 online session 的決策，
用與週記分卡相同的 EV 加權口徑，聚合成一則 Telegram 訊息：

  • 共幾手、平均 EV loss（單場、**不作趨勢判斷**）
  • EV loss 加總最多的 spot（→ 現在練 / 排入佇列）
  • EV loss 最高的 3 手（→ 手牌 history / 復盤 study / 排入佇列）

不變量：**只讀不改本週焦點 spot**（中圈穩定性 §3）。排入動作走既有 `drill_queue`
（`kind='drill'`/`'review'`，`added_by='session'`），與週掃描產生同構的 rows。口徑沿用
`queue_feed._HONEST`（source 隔離 §5.2 由 `source='online'` 保證）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import queue_feed as qf  # noqa: E402  (reuse URL/label/enqueue helpers + _HONEST)
from queue_feed import TPE, LOSSY_MIN_BB, pretty_hand  # noqa: E402
from scorecard import spot_desc_zh  # noqa: E402

TOP_SPOTS = 2
TOP_HANDS = 3

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

_TOP_HANDS_SQL = f"""
SELECT gtow_hand_id ref_hand_id, sum(ev_loss_bb) total_ev,
       (array_agg(spot_leaf     ORDER BY ev_loss_bb DESC))[1] spot_leaf,
       (array_agg(spot_category ORDER BY ev_loss_bb DESC))[1] spot_category,
       (array_agg(hero_cat      ORDER BY ev_loss_bb DESC))[1] hero_cat,
       (array_agg(villain_cat   ORDER BY ev_loss_bb DESC))[1] villain_cat,
       (array_agg(ip_oop        ORDER BY ev_loss_bb DESC))[1] ip_oop,
       (array_agg(position      ORDER BY ev_loss_bb DESC))[1] hero_pos,
       (array_agg(ev_loss_bb    ORDER BY ev_loss_bb DESC))[1] max_ev,
       (array_agg(approx_flags  ORDER BY ev_loss_bb DESC))[1] approx_flags,
       (array_agg(played_at     ORDER BY ev_loss_bb DESC))[1] played_at,
       (array_agg(street        ORDER BY ev_loss_bb DESC))[1] worst_street,
       (array_agg(decision_idx  ORDER BY ev_loss_bb DESC))[1] worst_idx,
       jsonb_agg(jsonb_build_object(
           'hand_id', gtow_hand_id, 'street', street,
           'decision_idx', decision_idx, 'ev_loss_bb', ev_loss_bb,
           'src', source) ORDER BY ev_loss_bb DESC) source_hands
FROM ledger_decisions
WHERE {qf._HONEST} AND ev_loss_bb >= {LOSSY_MIN_BB} AND {_IN_SESSION}
GROUP BY gtow_hand_id
ORDER BY sum(ev_loss_bb) DESC
LIMIT {TOP_HANDS}
"""

_HAND_META_SQL = ("SELECT hero_hand, boards, raw_path, position, preflop_depth_bb "
                  "FROM ledger_hands WHERE gtow_hand_id = $1")


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


async def _hand_items(conn, session_id: int) -> list[dict]:
    rows = await conn.fetch(_TOP_HANDS_SQL, session_id)
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
        # Exact-hand Analyze filter — the same reliable `hand_id__in` link the
        # queue's 📚 來源 sub-menu uses. Owner-preferred over the /solutions Study
        # node, which falls back to a day-range table when a low-loss hand has no
        # archived detail JSON (PR #122 skips detail for reconstructable hands).
        hand_urls = qf.gtow_analyze_hands_urls([row["ref_hand_id"]])
        exact_url = hand_urls[0][0] if hand_urls else None
        out.append({
            "combo": pretty_hand(row.get("hero_hand")),
            "boards": row.get("boards") or "",
            "desc": hand_desc(row),
            "max_ev": round(float(row.get("max_ev") or 0.0), 1),
            "total_ev": round(float(row["total_ev"]), 1),
            "exact_url": exact_url,
            "enqueue_item": {
                "kind": "review", "added_by": "session", "source": "online",
                "ref_hand_id": row["ref_hand_id"], "spot_leaf": row.get("spot_leaf"),
                "spot_category": row.get("spot_category"),
                "label": qf.review_label(row), "review_url": exact_url,
                "review_anchor_url": None, "review_anchor_street": None,
                "source_hands": qf._as_list(row["source_hands"]),
                "total_ev_loss_bb": round(float(row["total_ev"]), 4),
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
    return base + qf.bias_suffix(bias)


def hand_desc(row: dict) -> str:
    return spot_desc_zh({
        "spot_category": row.get("spot_category"), "spot_leaf": row.get("spot_leaf"),
        "hero_cat": row.get("hero_cat"), "villain_cat": row.get("villain_cat"),
        "ip_oop": row.get("ip_oop"), "hero_pos": row.get("hero_pos"),
        "street": row.get("spot_category")})


async def compute(conn, session: dict) -> dict:
    sid = session["id"]
    ov = await conn.fetchrow(_OVERVIEW_SQL, sid)
    hon = await conn.fetchrow(_HONESTY_SQL, sid)
    spots = await _spot_items(conn, sid)
    hands = await _hand_items(conn, sid)
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
        "top_hands": hands,
        "honesty": {"discarded_n": hon["discarded_n"] or 0,
                    "low_conf_n": hon["low_conf_n"] or 0},
        "empty": (ov["n_lossy"] or 0) == 0,
    }


# ── render ─────────────────────────────────────────────────────────────────────
def _session_span(started, ended) -> str:
    s = started.astimezone(TPE)
    e = ended.astimezone(TPE)
    if s.date() == e.date():
        return f"{s.strftime('%-m/%-d')} {s.strftime('%H:%M')}–{e.strftime('%H:%M')}"
    return f"{s.strftime('%-m/%-d %H:%M')} – {e.strftime('%-m/%-d %H:%M')}"


def render_tg(d: dict) -> dict:
    """Owner-facing session review (HTML parse_mode) + inline buttons.

    Pure function of ``compute()``'s output — no DB, no network. Returns
    ``{"html": str, "buttons": rows}`` where rows is list[list[button dict]].
    Buttons: URL (現在練/手牌/復盤) + callback (排入/略過). callback_data stays
    <64B via `srd|srv|srx:{session_id}:{i}` index keys.
    """
    sid = d["session_id"]
    span = _session_span(d["started_at"], d["ended_at"])
    L = [f"🔍 <b>這場復盤</b> · {escape(span)}", ""]

    if d["empty"]:
        L.append(f"這場 <b>{d['n_hands']}</b> 手，幾乎都打對了 — 沒有值得復盤的漏損 👍")
        L.append("（單場數字噪音大，這是「這場長怎樣」，不是進步/退步結論）")
        return {"html": "\n".join(L), "buttons": []}

    L.append(f"這場 <b>{d['n_hands']}</b> 手，平均 <b>{d['per100']:.1f} bb/100 決策</b>，"
             f"本場合計漏 <b>{d['total_bb']:.1f} bb</b>。")
    L.append("（單場數字噪音大，這是「這場長怎樣」，不是進步/退步結論）")

    buttons: list[list[dict]] = []

    if d["top_spots"]:
        L.append("")
        L.append("<b>漏最多的情境（EV 加總）</b>")
        for i, s in enumerate(d["top_spots"], 1):
            L.append(f"{i}. {escape(s['desc'])} — <b>{s['total_ev']:.1f} bb</b>（{s['n']} 手）")
        for i, s in enumerate(d["top_spots"]):
            row = []
            if s.get("drill_url"):
                row.append({"text": f"🎯 現在練：{s['desc'][:16]}", "url": s["drill_url"]})
            row.append({"text": "📥 排入佇列", "callback_data": f"srd:{sid}:{i}"})
            buttons.append(row)

    if d["top_hands"]:
        L.append("")
        L.append("<b>最貴 3 手</b>")
        marks = "①②③④⑤"
        for i, h in enumerate(d["top_hands"]):
            m = marks[i] if i < len(marks) else f"({i+1})"
            combo = f"{escape(h['combo'])} " if h["combo"] else ""
            board = f"{escape(h['boards'])} " if h["boards"] else ""
            L.append(f"{m} {combo}{board}— {escape(h['desc'])} −<b>{h['max_ev']:.1f} bb</b>")
        for i, h in enumerate(d["top_hands"]):
            m = marks[i] if i < len(marks) else f"#{i+1}"
            row = []
            if h.get("exact_url"):
                row.append({"text": f"{m} 📖 復盤", "url": h["exact_url"]})
            row.append({"text": "📥 排入", "callback_data": f"srv:{sid}:{i}"})
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
