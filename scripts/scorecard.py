#!/usr/bin/env python3
"""Weekly scorecard = training plan (action-line taxonomy, Version A loop).

Diagnose (action-line leak board by avg EV loss) -> prescribe 1-2 focus spots
with precise multi-depth GTOW Trainer drill links (the drill itself is the
retrieval practice) -> next-cycle EV-loss readback on the treated spot.

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
    "RFI": "首開", "vsOpen": "面對開池", "vsRaiseCall": "面對開池+跟注",
    "vsSqueeze": "被擠壓", "vs3bet": "被 3bet", "vsCold3bet": "冷面對 3bet",
    "vs4bet": "被 4bet", "vsCold4bet": "冷面對 4bet",
    "flop": "翻牌", "turn": "轉牌", "river": "河牌",
}
FACING_ZH = {"first_to_act": "首動", "vs_bet": "面對下注", "vs_check": "面對過牌",
             "vs_raise": "面對加注"}


# ── human-readable spot description (pure) ─────────────────────────────────
def spot_desc_zh(row: dict) -> str:
    cat = row["spot_category"]
    hc, vc, rel = row.get("hero_cat"), row.get("villain_cat"), row.get("ip_oop")
    if cat in ("flop", "turn", "river"):
        parts = row["spot_leaf"].split(":")
        pot = parts[1] if len(parts) > 1 else "?"
        facing = FACING_ZH.get(parts[-1], parts[-1])
        return f"{pot} 底池，你 {hc} 在 {rel or '?'}，{CAT_ZH[cat]} {facing}"
    if cat == "RFI":
        return f"{row.get('hero_pos') or hc} 首開（RFI）"
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


# ── pure assembly ──────────────────────────────────────────────────────────
def compute_training_plan(window_label, weekly_series, spots, top_hands,
                          prev_focus, honesty, focus_k=2) -> dict:
    per100 = weekly_series[-1]["per100"] if weekly_series else 0.0
    delta = (weekly_series[-1]["per100"] - weekly_series[-2]["per100"]
             if len(weekly_series) >= 2 else 0.0)
    word = "較上週改善" if delta < 0 else ("較上週惡化" if delta > 0 else "持平")
    focus = []
    for it in spots[:focus_k]:
        r = it["row"]
        focus.append({
            "spot_leaf": r["spot_leaf"], "spot_category": r["spot_category"],
            "desc": spot_desc_zh(r), "per100": r["avg_ev"] * 100, "n": r["n"],
            "hero_cat": r.get("hero_cat"), "villain_cat": r.get("villain_cat"),
            "ip_oop": r.get("ip_oop"), "drill_url": it["url"],
            "restrict": it.get("restrict"),
            "samples": [dict(s) for s in it.get("samples", [])],
        })
    return {
        "window": window_label,
        "headline": f"本週 EV loss {per100:.2f} bb/100 決策，{word} {abs(delta):.2f}",
        "per100": per100, "delta": delta, "weekly_series": weekly_series,
        "leaderboard": [dict(it["row"], drill_url=it["url"], restrict=it.get("restrict"))
                        for it in spots],
        "focus": focus, "readback": prev_focus_readback(prev_focus, spots),
        "top_hands": top_hands, "honesty": honesty,
    }


def prev_focus_readback(prev_focus, spots):
    """Given last cycle's focus spot_leafs, report this window's per100 for each."""
    if not prev_focus:
        return None
    by_leaf = {it["row"]["spot_leaf"]: it["row"] for it in spots}
    out = []
    for f in prev_focus:
        leaf = f.get("spot_leaf")
        cur = by_leaf.get(leaf)
        out.append({"spot_leaf": leaf, "prescribed_per100": f.get("per100"),
                    "current_per100": (cur["avg_ev"] * 100) if cur else None,
                    "n": cur["n"] if cur else 0,
                    "note": "單週讀數僅供參考，連續 4 週才算數"})
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
        f"<tr><td>{escape(r['spot_leaf'])}</td><td>{r['avg_ev']*100:.2f}</td>"
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
        focus_html += (f'<div class="card"><b>{escape(f["desc"])}</b>'
                       f'<div class="sub">{f["per100"]:.2f} bb/100 · n={f["n"]} · <code>{escape(f["spot_leaf"])}</code></div>'
                       f'{drill}{band}{samples}</div>')
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
<h2>Leak 榜（avg EV loss 排序）</h2>
<table><tr><th>spot</th><th>bb/100</th><th>n</th></tr>{lb_rows}</table>
<h2>最貴 3 手</h2>{top}
<h2>誠實層</h2><div class="sub">excluded {hon['excluded_n']} · discarded(limp) {hon['discarded_n']} · chipEV 評分占比 {hon['chipev_share']*100:.0f}%</div>
<div class="note">chipEV 評分：後期/泡沫手含 ICM 近似誤差（Phase 3 處理）；limp 相關 spot 已捨棄。</div>
</body></html>"""


def preview_summary_md(d: dict) -> str:
    L = [f"# 訓練計畫預覽（{d['window']}）", "",
         f"## 主指標", f"- EV loss/100 決策：**{d['per100']:.2f} bb**（週變化 {d['delta']:+.2f}）",
         f"- 週序列：{len(d['weekly_series'])} 週", "", "## 本週焦點 spot"]
    for f in d["focus"]:
        L.append(f"### {f['desc']}  `{f['spot_leaf']}`")
        L.append(f"- {f['per100']:.2f} bb/100 · n={f['n']}")
        if f["drill_url"]:
            L.append(f"- 🎯 drill：{f['drill_url']}")
        L.append("")
    L.append("## Leak 榜（avg EV loss）")
    L.append("| spot | bb/100 | n |")
    L.append("|---|---|---|")
    for r in d["leaderboard"][:10]:
        L.append(f"| {r['spot_leaf']} | {r['avg_ev']*100:.2f} | {r['n']} |")
    L.append("")
    hon = d["honesty"]
    L.append(f"## 誠實層\n- excluded {hon['excluded_n']} · discarded(limp) {hon['discarded_n']} · "
             f"chipEV {hon['chipev_share']*100:.1f}%")
    return "\n".join(L)


# ── async fetch + build + CLI ──────────────────────────────────────────────
WEEKLY_SQL = """
SELECT to_char((played_at AT TIME ZONE 'Asia/Taipei'), 'IYYY-"W"IW') week,
       count(*) n, avg(ev_loss_bb)*100 per100, sum(ev_loss_bb) total_bb
FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL
GROUP BY 1 ORDER BY 1
"""
TOP_HANDS_SQL = """
SELECT gtow_hand_id, played_at, hero_hand, position, boards, total_ev_loss_bb
FROM ledger_hands WHERE total_ev_loss_bb > 0 ORDER BY total_ev_loss_bb DESC LIMIT 3
"""


async def _honesty(conn) -> dict:
    tot = await conn.fetchval("SELECT count(*) FROM ledger_decisions")
    exc = await conn.fetchval("SELECT count(*) FROM ledger_decisions WHERE excluded")
    dis = await conn.fetchval("SELECT count(*) FROM ledger_decisions WHERE discarded")
    inc = await conn.fetchval("SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded")
    chip = await conn.fetchval(
        "SELECT count(*) FROM ledger_decisions WHERE NOT excluded AND NOT discarded "
        "AND approx_flags::text LIKE '%chipev_grading%'")
    return {"excluded_n": exc, "discarded_n": dis,
            "chipev_share": (chip / inc) if inc else 0.0, "total": tot}


async def build(conn, window_label, prev_focus, min_n=50, top=8):
    weekly = [dict(r) for r in await conn.fetch(WEEKLY_SQL)]
    spots = await lb.leaderboard(conn, min_n=min_n, top=top)
    top_hands = [dict(r) for r in await conn.fetch(TOP_HANDS_SQL)]
    honesty = await _honesty(conn)
    return compute_training_plan(window_label, weekly, spots, top_hands, prev_focus, honesty)


async def _run(mode: str, min_n: int):
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    outdir = ROOT / "data" / "scorecards"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
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
        prev = await conn.fetchrow("SELECT week, families FROM coach_focus ORDER BY created_at DESC LIMIT 1")
        prev_focus = None
        if prev:
            fam = prev["families"]
            prev_focus = json.loads(fam) if isinstance(fam, str) else fam
        data = await build(conn, week, prev_focus, min_n=min_n)
        html = render_html(data)
        (outdir / f"{week}.html").write_text(html)
        fam_payload = [{"spot_leaf": f["spot_leaf"], "per100": f["per100"], "n": f["n"],
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
        print(f"WEEKLY week={week} per100={data['per100']:.2f}")
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
