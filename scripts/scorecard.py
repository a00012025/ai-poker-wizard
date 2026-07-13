#!/usr/bin/env python3
"""Weekly scorecard = training plan (action-line taxonomy, Version A loop).

Diagnose (action-line leak board by avg EV loss) -> prescribe 1-2 focus spots
with precise multi-depth GTOW Trainer drill links (the drill itself is the
retrieval practice) -> next-cycle EV-loss readback on the treated spot.

Windows (§2.2 歸因): weekly mode diagnoses over the trailing FOCUS_WINDOW_DAYS
(a cumulative all-history average would bury recent form), and the readback of
last week's focus spot is computed strictly over the post-prescription window —
that is the only window where the treatment effect is observable.

--preview   full-history window, no DB writes; emits data/scorecards/preview.*
--weekly    current ISO week, writes scorecards + coach_focus + readback backfill
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
import spot_leaderboard as lb
from spot_leaderboard import analyze_table_url

TPE = ZoneInfo("Asia/Taipei")
CAT_ZH = {
    "RFI": "開池", "vsOpen": "面對開池", "vsRaiseCall": "面對開池+跟注",
    "vsSqueeze": "被擠壓", "vs3bet": "被 3bet", "vsCold3bet": "冷面對 3bet",
    "vs4bet": "被 4bet", "vsCold4bet": "冷面對 4bet",
    "flop": "翻牌", "turn": "轉牌", "river": "河牌",
}
FACING_ZH = {"first_to_act": "首動", "vs_bet": "面對下注", "vs_check": "面對過牌",
             "vs_raise": "面對加注"}


# ── human-readable spot description (pure) ─────────────────────────────────
def spot_desc_zh(row: dict) -> str:
    cat = row["spot_category"]
    if row.get("diagnosis_level") == "parent":
        key = row.get("diagnosis_key") or "?"
        return f"{CAT_ZH.get(cat, cat)}能力族群 `{key}`"
    hc, vc, rel = row.get("hero_cat"), row.get("villain_cat"), row.get("ip_oop")
    if cat in ("flop", "turn", "river"):
        parts = row["spot_leaf"].split(":")
        pot = parts[1] if len(parts) > 1 else "?"
        facing = FACING_ZH.get(parts[-1], parts[-1])
        return f"{pot} 底池，你 {hc} 在 {rel or '?'}，{CAT_ZH[cat]}{facing}"
    if cat == "RFI":
        return f"{row.get('hero_pos') or hc} 開池"
    if cat == "vsOpen":
        return f"{row.get('hero_pos') or hc} 面對 {vc} 開池"
    if cat in ("vs3bet", "vsCold3bet", "vs4bet", "vsCold4bet", "vsSqueeze"):
        return f"{hc} {CAT_ZH[cat]}（對手 {vc}，你 {rel or '?'}）"
    return f"{hc} {CAT_ZH.get(cat, cat)}（對手 {vc}）"


# ── SVG sparkline ──────────────────────────────────────────────────────────
def _svg_sparkline(values, w=560, h=80) -> str:
    if not values:
        return "<svg/>"
    mx, mn = max(values) or 1, min(values)
    span = (mx - mn) or 1
    pts = " ".join(f"{i * w / max(len(values) - 1, 1):.1f},"
                   f"{h - (v - mn) / span * (h - 10) - 5:.1f}"
                   for i, v in enumerate(values))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{pts}"/></svg>')


# diagnosis window for --weekly: recent form, not all-history
FOCUS_WINDOW_DAYS = 90

TRAINING_READINESS_SQL = """
SELECT count(*) FILTER (
         WHERE NOT excluded AND NOT discarded AND confidence >= 0.8) eligible,
       count(*) FILTER (
         WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL
           AND confidence >= 0.8 AND spot_parent IS NOT NULL
           AND played_depth_bb IS NOT NULL) ready
FROM ledger_decisions WHERE source='online'
"""


def training_readiness(row) -> tuple[bool, str]:
    eligible, ready = int(row["eligible"] or 0), int(row["ready"] or 0)
    ok = eligible == ready
    return ok, f"ledger hierarchy readiness {ready}/{eligible}"


async def ensure_training_ready(conn) -> None:
    """Fail closed instead of publishing a partial migration/backfill cohort."""
    row = await conn.fetchrow(TRAINING_READINESS_SQL)
    ok, note = training_readiness(row)
    if not ok:
        raise RuntimeError(f"{note}; run python scripts/backfill_spots.py")


# ── pure assembly ──────────────────────────────────────────────────────────
def compute_training_plan(window_label, weekly_series, spots, top_hands,
                          readback, honesty, focus_k=2) -> dict:
    per100 = weekly_series[-1]["per100"] if weekly_series else 0.0
    delta = (weekly_series[-1]["per100"] - weekly_series[-2]["per100"]
             if len(weekly_series) >= 2 else 0.0)
    word = "較上週改善" if delta < 0 else ("較上週惡化" if delta > 0 else "持平")
    focus = []
    for it in spots[:focus_k]:
        r = it["row"]
        focus.append({
            "spot_leaf": r["spot_leaf"], "spot_category": r["spot_category"],
            "diagnosis_key": r.get("diagnosis_key", r["spot_leaf"]),
            "diagnosis_level": r.get("diagnosis_level", "leaf"),
            "representative_leaf": r.get("representative_leaf", r["spot_leaf"]),
            "desc": spot_desc_zh(r), "per100": r["avg_ev"] * 100, "n": r["n"],
            "shrunk_per100": r.get("shrunk_avg_ev", r["avg_ev"]) * 100,
            "hero_cat": r.get("hero_cat"), "villain_cat": r.get("villain_cat"),
            "ip_oop": r.get("ip_oop"), "drill_url": it["url"],
            "restrict": it.get("restrict"),
            "fragile": bool(it.get("fragile")),
            "samples": [dict(s) for s in it.get("samples", [])],
        })
    return {
        "window": window_label,
        "headline": f"本週 EV loss {per100:.2f} bb/100 決策，{word} {abs(delta):.2f}",
        "per100": per100, "delta": delta, "weekly_series": weekly_series,
        "leaderboard": [dict(it["row"], drill_url=it["url"], restrict=it.get("restrict"))
                        for it in spots],
        "focus": focus, "readback": readback,
        "top_hands": top_hands, "honesty": honesty,
    }


def prev_focus_readback(prev_focus, current_by_leaf):
    """Given last cycle's focus spot_leafs and their POST-PRESCRIPTION window
    stats ({leaf: {n, per100}}), build the readback rows. Never fed cumulative
    averages: adding one week of play to months of history dilutes any change
    to invisibility, which would blind the §2.2 attribution loop."""
    if not prev_focus:
        return None
    out = []
    for f in prev_focus:
        leaf = f.get("spot_leaf")
        cur = (current_by_leaf or {}).get(leaf)
        has_data = bool(cur and cur.get("n"))
        out.append({"spot_leaf": leaf, "prescribed_per100": f.get("per100"),
                    "current_per100": cur["per100"] if has_data else None,
                    "n": cur["n"] if cur else 0,
                    "note": "處方後實戰窗口讀數，連續 4 週才算數"})
    return out


# ── HTML render ────────────────────────────────────────────────────────────
_STYLE = """
body{font-family:-apple-system,'PingFang TC','Microsoft JhengHei',sans-serif;
max-width:720px;margin:0 auto;padding:24px;color:#1a202c;background:#fff}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px;color:#2b6cb0;
border-bottom:1px solid #e2e8f0;padding-bottom:4px}
.metric{font-size:32px;font-weight:700}.sub{color:#718096;font-size:13px}
.up{color:#c53030}.down{color:#2f855a}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;
padding:5px 8px;border-bottom:1px solid #edf2f7}th{color:#718096}
.card{border:1px solid #e2e8f0;border-radius:6px;padding:12px;margin:8px 0}
.btn{display:inline-block;padding:5px 12px;background:#2b6cb0;color:#fff;border-radius:4px;
text-decoration:none;font-size:12px;margin:4px 0}a{color:#2b6cb0}.note{color:#a0aec0;font-size:12px}
"""


def _trend(v):
    if v < -1e-9:
        return f'<span class="down">▼{abs(v):.2f}</span>'
    if v > 1e-9:
        return f'<span class="up">▲{v:.2f}</span>'
    return '<span class="sub">–</span>'


def render_html(d: dict) -> str:
    spark = _svg_sparkline([w["per100"] for w in d["weekly_series"]])
    lb_rows = "".join(
        f"<tr><td>{escape(r.get('diagnosis_key', r['spot_leaf']))}</td>"
        f"<td>{r['avg_ev']*100:.2f}</td>"
        f"<td>{r.get('shrunk_avg_ev', r['avg_ev'])*100:.2f}</td>"
        f"<td>{r['n']}</td></tr>" for r in d["leaderboard"][:8])
    focus_html = ""
    for f in d["focus"]:
        drill = (f'<a class="btn" href="{escape(f["drill_url"])}">🎯 進 GTOW Trainer 練這個 spot</a>'
                 if f["drill_url"] else '<span class="note">（此 spot 無 Trainer 捷徑，見 Analyze 樣本）</span>')
        band = f'<div class="note">⛏ 集中在 {lb.BAND_ZH.get(f["restrict"], f["restrict"])} 深度，drill 已鎖此帶</div>' if f.get("restrict") else '<div class="note">drill 涵蓋全部 stack depth</div>'
        samples = "".join(
            f'<div class="sub">· {escape(str(s.get("hero_hand") or "?"))} '
            f'{escape(str(s.get("boards") or ""))} 損失 {s.get("ev_loss_bb",0):.1f}bb '
            f'<a href="{analyze_table_url(s["played_at"].astimezone(TPE).strftime("%Y-%m-%d"), s["played_at"].astimezone(TPE).strftime("%Y-%m-%d"))}">Analyze</a></div>'
            for s in f.get("samples", []))
        frag = ('<div class="note">⚠️ fragile：排除樹外近似樣本後量級變動 &gt;30%，數字保守解讀</div>'
                if f.get("fragile") else '')
        focus_html += (f'<div class="card"><b>{escape(f["desc"])}</b>'
                       f'<div class="sub">觀察 {f["per100"]:.2f} · 收縮排序估計 '
                       f'{f.get("shrunk_per100", f["per100"]):.2f} bb/100 · n={f["n"]} · '
                       f'<code>{escape(f.get("diagnosis_key") or f["spot_leaf"])}</code></div>'
                       f'<div class="note">代表 action-line：<code>'
                       f'{escape(f.get("representative_leaf") or f["spot_leaf"])}</code></div>'
                       f'{drill}{band}{frag}{samples}</div>')
    rb = ""
    if d.get("readback"):
        for r in d["readback"]:
            cur = f'{r["current_per100"]:.2f}' if r["current_per100"] is not None else "—"
            pre = f'{r["prescribed_per100"]:.2f}' if r.get("prescribed_per100") is not None else "—"
            delta = (r["current_per100"] - r["prescribed_per100"]) if (r["current_per100"] is not None and r.get("prescribed_per100") is not None) else 0
            rb += (f'<div class="card">上週焦點 <code>{escape(str(r["spot_leaf"]))}</code>：'
                   f'處方時 {pre} → 本週 {cur} bb/100 (n={r["n"]}) {_trend(delta)}'
                   f'<div class="note">{escape(r["note"])}</div></div>')
    hon = d["honesty"]
    top = "".join(
        f'<div class="sub">· {escape(str(h.get("hero_hand") or "?"))} {escape(str(h.get("position") or ""))} '
        f'{escape(str(h.get("boards") or ""))} 損失 {(h.get("total_ev_loss_bb") or 0):.1f}bb '
        f'<a href="{analyze_table_url(h["played_at"].astimezone(TPE).strftime("%Y-%m-%d"), h["played_at"].astimezone(TPE).strftime("%Y-%m-%d"))}">Analyze</a></div>'
        for h in d["top_hands"][:3])
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>訓練計畫 {escape(d['window'])}</title><style>{_STYLE}</style></head><body>
<h1>週訓練計畫</h1><div class="sub">窗口：{escape(d['window'])}</div>
<h2>主指標：EV loss / 100 決策</h2>
<div class="metric">{d['per100']:.2f}<span class="sub"> bb/100 · 週變化 {_trend(d['delta'])}</span></div>{spark}
<h2>本週焦點 spot（先作答，再練）</h2>{focus_html or '<div class="sub">無足夠樣本的焦點</div>'}
{('<h2>上週焦點回讀</h2>'+rb) if rb else ''}
<h2>Leak 榜（收縮 EV-loss 估計排序）</h2>
<table><tr><th>family</th><th>觀察 bb/100</th><th>收縮估計</th><th>n</th></tr>{lb_rows}</table>
<h2>最貴 3 手</h2>{top}
<h2>誠實層</h2><div class="sub">excluded {hon['excluded_n']} · discarded(limp) {hon['discarded_n']} · low-confidence {hon.get('low_confidence_n', 0)} · chipEV 評分占比 {hon['chipev_share']*100:.0f}% · physical/effective depth 差異 {hon.get('depth_snap_n', 0)}</div>
<div class="note">低信心決策不進統計；physical/effective depth 差異保留作 audit，但 drill 使用 GTOW decision depth。</div>
</body></html>"""


def weekly_tg_html(week: str, d: dict) -> str:
    """End-user weekly coaching message for Telegram (HTML parse_mode).

    Concise, no jargon (no "北極星"/"迴圈"). Tells the player: what to drill,
    where they're leaking, the goal, and the honest data caveats (chipEV/ICM,
    dropped limps, coverage). Drill links are NOT embedded — they ride as URL
    buttons (weekly_tg_payload) under the message.
    """
    per100 = d.get("per100", 0.0)
    delta = d.get("delta", 0.0)
    if delta < -1e-9:
        trend = f"比上週少漏了 {abs(delta):.2f} 👍"
    elif delta > 1e-9:
        trend = f"比上週多漏了 {delta:.2f}"
    else:
        trend = "跟上週差不多"

    L = [f"🃏 <b>本週該練的地方</b>（{escape(week)}）", ""]
    L.append(f"這週你每 100 個決策平均漏掉 <b>{per100:.2f} bb</b>，{trend}。")

    focus = d.get("focus", [])
    if focus:
        L.append("")
        L.append("<b>最該補的洞：</b>")
        for i, f in enumerate(focus, 1):
            band = ""
            if f.get("restrict"):
                bz = lb.BAND_ZH.get(f["restrict"], f["restrict"])
                band = f"（{bz}特別弱，練習按鈕已鎖這個籌碼帶）"
            frag = "（⚠️ 這格的數字對樹外近似敏感，保守看）" if f.get("fragile") else ""
            rep = f.get("representative_leaf") or f.get("spot_leaf") or "?"
            L.append(f"{i}. {escape(f['desc'])} — 觀察 {f['per100']:.1f}、"
                     f"收縮排序估計 {f.get('shrunk_per100', f['per100']):.1f} bb/100，"
                     f"n={f['n']}；代表 action-line <code>{escape(rep)}</code>{band}{frag}")

    dq = d.get("drill_queue") or []
    if dq:
        lb_leafs = {r["spot_leaf"] for r in d.get("leaderboard", [])}
        L.append("")
        L.append("<b>📥 練習佇列（現場 + 線上）：</b>")
        for q in dq:
            aging = ""
            if q.get("status") == "prescribed":
                wk = q.get("prescribed_week")
                aging = f"（{wk} 已開過，還沒練 ⏰）" if wk else "（先前已開過，還沒練 ⏰）"
            lbl = q.get("label") or q.get("spot_leaf") or "?"
            if q.get("kind") == "review":
                # label already reads「復盤 M/D … −Xbb」
                L.append(f"• 🔍 {escape(lbl)}{aging}")
            else:
                cross = "（線上同一個情境也在漏 ⚠️）" if q.get("spot_leaf") in lb_leafs else ""
                ev = q.get("total_ev_loss_bb") or 0
                L.append(f"• 🎯 {escape(lbl)} — 來自 {q.get('n_sources', 1)} 手，"
                         f"累計損失 {ev:.1f}bb{cross}{aging}")

    focus_leafs = {f["spot_leaf"] for f in focus}
    others = [r for r in d.get("leaderboard", []) if r["spot_leaf"] not in focus_leafs][:3]
    if others:
        parts = [f"{spot_desc_zh(r)} {r['avg_ev'] * 100:.1f}" for r in others]
        L.append("")
        L.append("其他也在漏的（bb/100）：" + "、".join(escape(p) for p in parts))

    for r in (d.get("readback") or []):
        if r.get("current_per100") is not None and r.get("prescribed_per100") is not None:
            dv = r["current_per100"] - r["prescribed_per100"]
            arrow = "↓ 有進步" if dv < -1e-9 else ("↑ 還在漏" if dv > 1e-9 else "持平")
            L.append("")
            L.append(f"上週練的那個 spot：{r['prescribed_per100']:.1f} → "
                     f"{r['current_per100']:.1f} bb/100 {arrow}"
                     f"（{escape(r['note'])}）")

    L.append("")
    L.append("🎯 <b>目標</b>：把上面幾個 spot 練到接近 GTO，整體漏損往 &lt;2 bb/100 收。")

    hon = d.get("honesty", {})
    L.append("")
    L.append("⚠️ <b>老實說，這份數據有幾個誤差跟缺口：</b>")
    L.append(f"• 翻牌後全部用 chipEV 評分（占 {hon.get('chipev_share', 0) * 100:.0f}%），"
             "泡沫、單桌決賽的手會有 ICM 誤差，之後才會校正。")
    L.append(f"• 你的 limp 手（約 {hon.get('discarded_n', 0)} 個決策）直接沒算進去——"
             "GTOW 的 limp 範圍跟真人差太多，算了會誤導。")
    low_n = hon.get("low_confidence_n") or 0
    if low_n:
        L.append(f"• 有 {low_n} 個低信心決策未納入統計（例如尺寸樹外或缺 decision depth）。")
    gap_n = hon.get("depth_snap_n") or 0
    if gap_n:
        L.append(f"• 有 {gap_n} 個決策的牌桌 stack 與 GTOW binding effective depth 不同；"
                 "這是 audit 訊號，練習深度已改用實際評分的 decision depth。")
    L.append("• 只看得到你有上傳 GTOW Analyzer 的手 + 用 /live 記的現場手，其他不在裡面。")
    return "\n".join(L)


def weekly_tg_payload(week: str, d: dict) -> dict:
    """Weekly TG message + inline buttons: {"html": str, "buttons": rows}.

    Buttons: 🎯 focus-spot drills, then the practice-queue quota — drill items
    ride a 🎯 URL button; review items ride 🔗 復盤 (URL) + ✔ 完成 (qcl) + ➕ 加練
    (qex) callbacks (§7/§6.2). Rows may carry url OR callback_data entries.
    """
    buttons: list[list[dict]] = []
    for i, f in enumerate(d.get("focus", []), 1):
        if f.get("drill_url"):
            buttons.append([{"text": f"🎯 練 {i}：{f['desc'][:28]}", "url": f["drill_url"]}])
    for q in (d.get("drill_queue") or []):
        qid = q.get("id")
        lbl = (q.get("label") or q.get("spot_leaf") or "?")[:24]
        if q.get("kind") == "review":
            row: list[dict] = []
            anchor = q.get("review_anchor_url")
            anchor_street = q.get("review_anchor_street")
            if anchor:
                row.append({"text": f"↩ 先看 {(anchor_street or '上游').title()}",
                            "url": anchor})
            if q.get("drill_url"):
                text = "💥 再看損失" if anchor else f"🔗 復盤：{lbl}"
                row.append({"text": text, "url": q["drill_url"]})
            actions: list[dict] = []
            if qid is not None:
                actions.append({"text": "✔ 完成", "callback_data": f"qcl:{qid}"})
                actions.append({"text": "➕ 加練", "callback_data": f"qex:{qid}"})
            if anchor and row:
                buttons.append(row)
                row = actions
            else:
                row.extend(actions)
            if row:
                buttons.append(row)
        elif q.get("drill_url"):
            buttons.append([{"text": f"📥 佇列：{lbl}", "url": q["drill_url"]}])
    return {"html": weekly_tg_html(week, d), "buttons": buttons}


def preview_summary_md(d: dict) -> str:
    L = [f"# 訓練計畫預覽（{d['window']}）", "",
         f"## 主指標", f"- EV loss/100 決策：**{d['per100']:.2f} bb**（週變化 {d['delta']:+.2f}）",
         f"- 週序列：{len(d['weekly_series'])} 週", "", "## 本週焦點 spot"]
    for f in d["focus"]:
        L.append(f"### {f['desc']}  `{f.get('diagnosis_key') or f['spot_leaf']}`")
        L.append(f"- 觀察 {f['per100']:.2f} · 收縮排序估計 "
                 f"{f.get('shrunk_per100', f['per100']):.2f} bb/100 · n={f['n']}")
        if f["drill_url"]:
            L.append(f"- 🎯 drill：{f['drill_url']}")
        L.append("")
    L.append("## Leak 榜（收縮 EV-loss 估計排序）")
    L.append("| family | 觀察 bb/100 | 收縮估計 | n |")
    L.append("|---|---:|---:|---:|")
    for r in d["leaderboard"][:10]:
        L.append(f"| {r.get('diagnosis_key', r['spot_leaf'])} | {r['avg_ev']*100:.2f} | "
                 f"{r.get('shrunk_avg_ev', r['avg_ev'])*100:.2f} | {r['n']} |")
    L.append("")
    hon = d["honesty"]
    L.append(f"## 誠實層\n- excluded {hon['excluded_n']} · discarded(limp) {hon['discarded_n']} · "
             f"chipEV {hon['chipev_share']*100:.1f}%")
    return "\n".join(L)


# ── async fetch + build + CLI ──────────────────────────────────────────────
# All stats queries are source='online' only (§5.2 source isolation): live
# hands are selectively recorded — their averages are biased by design and
# surface in their own queue section instead.
WEEKLY_SQL = """
SELECT to_char((played_at AT TIME ZONE 'Asia/Taipei'), 'IYYY-"W"IW') week,
       count(*) n, avg(ev_loss_bb)*100 per100, sum(ev_loss_bb) total_bb
FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL AND source='online'
  AND confidence >= 0.8
GROUP BY 1 ORDER BY 1
"""
def top_hands_sql(since=None) -> str:
    win = " AND played_at >= $1" if since else ""
    return f"""
SELECT h.gtow_hand_id, h.played_at, h.hero_hand, h.position, h.boards,
       sum(d.ev_loss_bb) total_ev_loss_bb
FROM ledger_hands h JOIN ledger_decisions d ON d.gtow_hand_id=h.gtow_hand_id
WHERE h.source='online' AND d.source='online' AND NOT d.excluded AND NOT d.discarded
  AND d.confidence >= 0.8{win.replace('played_at', 'h.played_at')}
GROUP BY h.gtow_hand_id, h.played_at, h.hero_hand, h.position, h.boards
HAVING sum(d.ev_loss_bb) > 0
ORDER BY sum(d.ev_loss_bb) DESC LIMIT 3
"""


# post-prescription window stats for last cycle's focus leaf (§2.2 readback)
READBACK_WINDOW_SQL = """
SELECT count(*) n, avg(ev_loss_bb)*100 per100
FROM ledger_decisions
WHERE spot_leaf=$1 AND NOT excluded AND NOT discarded AND source='online'
  AND confidence >= 0.8 AND played_at >= $2
"""
READBACK_WINDOW_SQL_PARENT = READBACK_WINDOW_SQL.replace("spot_leaf=$1", "spot_parent=$1")

# prescribed-but-uncleared items KEEP re-surfacing in the plan (§14.2:
# silently dropping an unpracticed prescription degrades the coaching
# signal) — pending first, then open prescriptions by EV. The plan drains a
# per-kind quota (§7): QUEUE_DRILL_SLOTS drills + QUEUE_REVIEW_SLOTS reviews,
# one topping up the other to QUEUE_SLOTS. Fetch a wider window, mix in Python.
QUEUE_DRILL_SLOTS = 3
QUEUE_REVIEW_SLOTS = 2
QUEUE_SLOTS = QUEUE_DRILL_SLOTS + QUEUE_REVIEW_SLOTS
QUEUE_SQL = """
SELECT id, spot_leaf, spot_category, label, drill_url, review_anchor_url,
       review_anchor_street, n_sources, total_ev_loss_bb, source, status,
       prescribed_week, kind, ref_hand_id
FROM drill_queue WHERE status IN ('pending', 'prescribed')
ORDER BY (status = 'pending') DESC, total_ev_loss_bb DESC NULLS LAST LIMIT 40
"""


async def _honesty(conn) -> dict:
    src = "source='online'"
    tot = await conn.fetchval(f"SELECT count(*) FROM ledger_decisions WHERE {src}")
    exc = await conn.fetchval(f"SELECT count(*) FROM ledger_decisions WHERE excluded AND {src}")
    dis = await conn.fetchval(f"SELECT count(*) FROM ledger_decisions WHERE discarded AND {src}")
    inc = await conn.fetchval(
        f"SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded "
        f"AND confidence >= 0.8 AND {src}")
    low = await conn.fetchval(
        f"SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded "
        f"AND confidence < 0.8 AND {src}")
    chip = await conn.fetchval(
        f"SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded AND confidence >= 0.8 "
        f"AND approx_flags::text LIKE '%chipev_grading%' AND {src}")
    snap = await conn.fetchval(
        f"SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded "
        f"AND approx_flags ? 'sizing_snap' AND {src}")
    dgap = await conn.fetchval(
        f"SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded "
        f"AND approx_flags ? 'played_solver_depth_gap' AND {src}")
    return {"excluded_n": exc, "discarded_n": dis,
            "chipev_share": (chip / inc) if inc else 0.0, "total": tot,
            "sizing_snap_n": snap, "depth_snap_n": dgap, "low_confidence_n": low}


async def fetch_drill_queue(conn) -> list[dict]:
    """Top practice-queue items for the weekly plan (drill + review), mixed to
    the per-kind quota (§7). Pending-first / EV-desc order is preserved."""
    from queue_feed import mix_queue_quota
    rows = [dict(r) for r in await conn.fetch(QUEUE_SQL)]
    return mix_queue_quota(rows, QUEUE_DRILL_SLOTS, QUEUE_REVIEW_SLOTS, QUEUE_SLOTS)


async def fetch_readback(conn, prev_focus, prev_at) -> dict:
    """{leaf: {n, per100}} over the post-prescription window (played_at >= prev_at)."""
    out = {}
    for f in prev_focus or []:
        key = f.get("diagnosis_key") or f.get("spot_leaf")
        if not key:
            continue
        sql = (READBACK_WINDOW_SQL_PARENT
               if f.get("diagnosis_level") == "parent" else READBACK_WINDOW_SQL)
        r = await conn.fetchrow(sql, key, prev_at)
        out[f.get("spot_leaf") or key] = {"n": r["n"] or 0,
                     "per100": float(r["per100"]) if r["per100"] is not None else None}
    return out


async def build(conn, window_label, prev_focus, min_n=50, top=8,
                since=None, prev_at=None):
    weekly = [dict(r) for r in await conn.fetch(WEEKLY_SQL)]
    spots = await lb.hierarchical_leaderboard(
        conn, min_n=max(25, min_n // 2), top=top, since=since)
    top_hands = [dict(r) for r in await conn.fetch(
        top_hands_sql(since), *([since] if since else []))]
    honesty = await _honesty(conn)
    readback = None
    if prev_focus and prev_at:
        readback = prev_focus_readback(prev_focus, await fetch_readback(conn, prev_focus, prev_at))
    data = compute_training_plan(window_label, weekly, spots, top_hands, readback, honesty)
    data["drill_queue"] = await fetch_drill_queue(conn)
    return data


async def _run(mode: str, min_n: int):
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    outdir = ROOT / "data" / "scorecards"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        await ensure_training_ready(conn)
        if mode == "preview":
            data = await build(conn, "preview-all-history", None, min_n=min_n)
            (outdir / "preview.html").write_text(render_html(data))
            (outdir / "preview_summary.md").write_text(preview_summary_md(data))
            (outdir / "preview_data.json").write_text(json.dumps(data, default=str, ensure_ascii=False, indent=1))
            print(f"PREVIEW per100={data['per100']:.2f} focus={len(data['focus'])} "
                  f"leaderboard={len(data['leaderboard'])}")
            return 0
        now = datetime.now(TPE)
        y, wk, _ = now.isocalendar()
        week = f"{y}-W{wk:02d}"
        prev = await conn.fetchrow(
            "SELECT week, families, created_at FROM coach_focus ORDER BY created_at DESC LIMIT 1")
        prev_focus, prev_at = None, None
        if prev:
            fam = prev["families"]
            prev_focus = json.loads(fam) if isinstance(fam, str) else fam
            prev_at = prev["created_at"]
        # §5.4: scan the online window into the queue BEFORE building the plan,
        # so this week's fresh drill/review items are eligible to be prescribed
        # and drained. The scan uses its own 60d window (queue_feed constant),
        # deliberately distinct from the 90d focus window below.
        from queue_feed import scan_online
        scan = await scan_online(conn)
        print(f"queue scan: {len(scan['drill'])} drill + {len(scan['review'])} "
              f"review candidates, tally={scan['tally']}")
        since = datetime.now(timezone.utc) - timedelta(days=FOCUS_WINDOW_DAYS)
        data = await build(conn, week, prev_focus, min_n=min_n, since=since, prev_at=prev_at)
        html = render_html(data)
        (outdir / f"{week}.html").write_text(html)
        fam_payload = [{"spot_leaf": f["spot_leaf"],
                        "diagnosis_key": f.get("diagnosis_key"),
                        "diagnosis_level": f.get("diagnosis_level"),
                        "per100": f["per100"], "n": f["n"],
                        "drill_url": f["drill_url"]} for f in data["focus"]]
        presc = [{"label": f["desc"], "url": f["drill_url"]} for f in data["focus"] if f["drill_url"]]
        await conn.execute(
            "INSERT INTO scorecards (week, html, data_json) VALUES ($1,$2,$3) "
            "ON CONFLICT (week) DO UPDATE SET html=EXCLUDED.html, data_json=EXCLUDED.data_json",
            week, html, json.dumps(data, default=str))
        await conn.execute(
            "INSERT INTO coach_focus (week, families, rationale, prescriptions, readback) "
            "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (week) DO UPDATE SET "
            "families=EXCLUDED.families, prescriptions=EXCLUDED.prescriptions",
            week, json.dumps(fam_payload, default=str),
            json.dumps({"per100": data["per100"]}, default=str),
            json.dumps(presc, default=str), json.dumps(None))
        if prev and data.get("readback"):
            await conn.execute("UPDATE coach_focus SET readback=$2 WHERE week=$1",
                               prev["week"], json.dumps(data["readback"], default=str))
        dq_ids = [q["id"] for q in (data.get("drill_queue") or [])]
        if dq_ids:
            # surfaced in this week's plan -> prescribed (still visible in /queue
            # until the player marks them cleared)
            await conn.execute(
                "UPDATE drill_queue SET status='prescribed', prescribed_week=$1 "
                "WHERE id = ANY($2) AND status='pending'", week, dq_ids)
        print(f"WEEKLY week={week} per100={data['per100']:.2f} "
              f"queue={len(dq_ids)}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--min-n", type=int, default=50)
    a = ap.parse_args()
    mode = "preview" if a.preview else ("weekly" if a.weekly else None)
    if not mode:
        print("usage: scorecard.py --preview | --weekly [--min-n N]")
        raise SystemExit(1)
    raise SystemExit(asyncio.run(_run(mode, a.min_n)))
