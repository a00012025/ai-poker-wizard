#!/usr/bin/env python3
"""Weekly scorecard: data compute (pure) + self-contained HTML + CLI.

--preview   full-history window, does NOT write coach_focus/scorecards;
            emits data/scorecards/{preview.html,preview_data.json,preview_summary.md}
--weekly    current ISO week label, writes scorecards + coach_focus, backfills
            last week's readback; emits data/scorecards/<week>.html
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gtow_trainer_url import build_trainer_url, SpotNotSupportedError
import ledger_diagnostics as diag

TPE = ZoneInfo("Asia/Taipei")
PREFLOP_FAMILIES = {"open_raise", "facing_open", "possible_squeeze", "hero_3bet",
                    "facing_3bet", "vs_squeeze", "squeeze", "facing_4bet", "limp_pot"}
BAND_MID = {"le15": 12, "15_25": 20, "25_40": 32, "40plus": 50}


def analyze_table_url(day_start_taipei: str, day_end_taipei: str) -> str:
    start = datetime.fromisoformat(day_start_taipei).replace(tzinfo=TPE)
    end = datetime.fromisoformat(day_end_taipei).replace(tzinfo=TPE) + timedelta(days=1)
    fmt = lambda d: d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filters = json.dumps({"played_at__range": [fmt(start), fmt(end)]})
    return (f"https://app.gtowizard.com/analyze/v4/hands/table"
            f"?filters={quote(filters)}&preselectGamemode=TOURNAMENT")


def _hand_review_url(h: dict) -> str:
    day = h["played_at"].astimezone(TPE).strftime("%Y-%m-%d")
    return analyze_table_url(day, day)


def _prescribe(cell: dict, decisions: list[dict]) -> list[dict]:
    fam, band = cell["family"], cell["depth_band"]
    street = "preflop" if fam in PREFLOP_FAMILIES else "flop"
    pot_types = [d.get("pot_type") for d in decisions
                 if d["family"] == fam and d.get("pot_type")]
    pot_type = max(set(pot_types), key=pot_types.count) if pot_types else "SRP"
    out = []
    try:
        out.append({"label": f"GTOW Trainer drill：{fam} @{band}",
                    "url": build_trainer_url(fam, street, BAND_MID[band],
                                             pot_type=None if street == "preflop" else pot_type)})
    except SpotNotSupportedError:
        # fallback: link to the Analyze table so the spot is still one click away
        today = datetime.now(TPE).strftime("%Y-%m-%d")
        out.append({"label": f"復盤 {fam} 手牌（Analyze）",
                    "url": analyze_table_url(today, today)})
    return out


def _family_top_hands(cell, decisions, hands, k=3):
    ids = {d.get("gtow_hand_id") for d in decisions
           if d["family"] == cell["family"] and (d["ev_loss_bb"] or 0) > 0}
    fam_hands = [h for h in hands if h["gtow_hand_id"] in ids]
    return diag.most_expensive_hands(fam_hands, k)


def _readback(prev_focus, decisions):
    if not prev_focus:
        return None
    out = []
    for f in prev_focus.get("families", []):
        fam = f["family"]
        fam_ds = [d for d in decisions
                  if d["family"] == fam and not d.get("excluded")]
        cur = (sum(d["ev_loss_bb"] or 0 for d in fam_ds) / len(fam_ds) * 100) if fam_ds else 0.0
        out.append({"family": fam, "prescribed_per100": f.get("per100"),
                    "current_per100": cur, "n": len(fam_ds),
                    "note": "單週讀數僅供參考，連續 4 週才算數"})
    return out


def compute_scorecard_data(decisions, hands, sessions, prev_focus, window_label) -> dict:
    series = diag.weekly_series(decisions)
    board = diag.leak_board(decisions)
    inc = [d for d in decisions if not d.get("excluded")]
    n = len(inc)
    per100 = (sum(d["ev_loss_bb"] or 0 for d in inc) / n * 100) if n else 0.0
    delta = (series[-1]["ev_loss_per_100"] - series[-2]["ev_loss_per_100"]
             if len(series) >= 2 else 0.0)
    word = "較上窗改善" if delta < 0 else ("較上窗惡化" if delta > 0 else "持平")

    focus_cells = diag.pick_focus(board)
    focus = {"families": [dict(c, prescriptions=_prescribe(c, inc),
                               top_hands=[dict(h, review_url=_hand_review_url(h))
                                          for h in _family_top_hands(c, inc, hands)])
                          for c in focus_cells],
             "readback": _readback(prev_focus, decisions)}

    unsolved = sum(1 for d in decisions
                   if any(f == "unsolved" for f in d.get("approx_flags", [])))
    chipev = sum(1 for d in inc
                 if any(f == "chipev_grading" for f in d.get("approx_flags", [])))
    return {
        "window": window_label,
        "headline": f"本窗 EV loss {per100:.2f}bb/100（n={n}），{word} {abs(delta):.2f}",
        "per100": per100, "n": n, "delta": delta,
        "weekly_series": series, "leak_board": board,
        "top_hands": [dict(h, review_url=_hand_review_url(h))
                      for h in diag.most_expensive_hands(hands)],
        "focus": focus,
        "session_obs": diag.session_correlations(decisions, hands, sessions),
        "honesty": {"excluded_n": board["excluded_n"], "unsolved_n": unsolved,
                    "chipev_share": (chipev / n) if n else 0.0,
                    "note": "chipEV 評分：後期/泡沫手的判定含 ICM 近似誤差（Phase 3 處理）；"
                            "單週讀數僅供參考，連續 4 週才算數"},
    }


# ── HTML rendering (self-contained: inline CSS + server-side SVG, no JS) ──

def _svg_sparkline(values: list[float], w=560, h=80) -> str:
    if not values:
        return "<svg/>"
    mx, mn = max(values) or 1, min(values)
    span = (mx - mn) or 1
    pts = " ".join(f"{i * w / max(len(values) - 1, 1):.1f},"
                   f"{h - (v - mn) / span * (h - 10) - 5:.1f}"
                   for i, v in enumerate(values))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{pts}"/></svg>')


_STYLE = """
body{font-family:-apple-system,'PingFang TC','Microsoft JhengHei',sans-serif;
max-width:680px;margin:0 auto;padding:24px;color:#1a202c;background:#fff}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:24px 0 8px;color:#2b6cb0;
border-bottom:1px solid #e2e8f0;padding-bottom:4px}
.metric{font-size:32px;font-weight:700} .sub{color:#718096;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{text-align:left;
padding:5px 8px;border-bottom:1px solid #edf2f7} th{color:#718096;font-weight:600}
.up{color:#c53030} .down{color:#2f855a} .tag{display:inline-block;font-size:11px;
padding:1px 6px;border-radius:3px;background:#edf2f7;color:#4a5568}
.card{border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px;margin:6px 0}
a{color:#2b6cb0} .note{color:#a0aec0;font-size:12px;margin-top:4px}
.btn{display:inline-block;padding:4px 10px;background:#2b6cb0;color:#fff;
border-radius:4px;text-decoration:none;font-size:12px;margin:2px 4px 2px 0}
"""


def _trend_span(v: float) -> str:
    if v < -1e-9:
        return f'<span class="down">▼{abs(v):.1f}</span>'
    if v > 1e-9:
        return f'<span class="up">▲{v:.1f}</span>'
    return '<span class="sub">–</span>'


def _leak_type_zh(t: str) -> str:
    return {"boundary": "邊界型", "knowledge": "知識型"}.get(t, t)


def render_scorecard_html(data: dict) -> str:
    d = data
    delta = d["delta"]
    delta_html = _trend_span(delta)
    spark = _svg_sparkline([w["ev_loss_per_100"] for w in d["weekly_series"]])

    # leak board
    rows = ""
    for c in d["leak_board"]["cells"][:5]:
        rows += (f"<tr><td>{escape(c['family'])}</td><td>{escape(c['depth_band'])}</td>"
                 f"<td>{c['total_bb']:.1f}</td><td>{c['n']}</td>"
                 f"<td>{c['per100']:.2f}</td><td>{_trend_span(c['trend'])}</td>"
                 f"<td><span class='tag'>{_leak_type_zh(c['leak_type'])}</span> "
                 f"{escape(c.get('slice_desc',''))}</td></tr>")
    insuff = d["leak_board"].get("insufficient", [])
    insuff_html = (f"<div class='note'>樣本不足未排名（n&lt;25）：{len(insuff)} 個 cell</div>"
                   if insuff else "")

    # most expensive hands
    hands_html = ""
    for h in d["top_hands"][:3]:
        hands_html += (f"<div class='card'><b>{escape(str(h.get('hero_hand') or '?'))}</b> "
                       f"{escape(str(h.get('position') or ''))} · "
                       f"{escape(str(h.get('boards') or ''))} · "
                       f"損失 {(h.get('total_ev_loss_bb') or 0):.1f}bb "
                       f"<a href='{escape(h['review_url'])}'>GTOW 復盤 →</a>"
                       f"<div class='sub'>{escape(str(h.get('tournament_name') or ''))}</div></div>")

    # focus + prescriptions
    focus_html = ""
    for f in d["focus"]["families"]:
        btns = "".join(f"<a class='btn' href='{escape(p['url'])}'>{escape(p['label'])}</a>"
                       for p in f.get("prescriptions", []))
        th = "".join(f"<div class='sub'>· {escape(str(h.get('hero_hand') or '?'))} "
                     f"損失 {(h.get('total_ev_loss_bb') or 0):.1f}bb "
                     f"<a href='{escape(h['review_url'])}'>復盤</a></div>"
                     for h in f.get("top_hands", []))
        focus_html += (f"<div class='card'><b>{escape(f['family'])} @{escape(f['depth_band'])}</b> "
                       f"<span class='tag'>{_leak_type_zh(f.get('leak_type',''))}</span>"
                       f"<div class='sub'>近窗 {f['total_bb']:.1f}bb / n={f['n']} · "
                       f"{escape(f.get('slice_desc',''))}</div>{th}<div>{btns}</div></div>")
    readback = d["focus"].get("readback")
    if readback:
        rb = "".join(f"<div class='card'>上週焦點 <b>{escape(r['family'])}</b>："
                     f"處方當時 {(_num(r.get('prescribed_per100')))} → 本窗 {r['current_per100']:.2f} bb/100 "
                     f"(n={r['n']}) {_trend_span(r['current_per100'] - (r.get('prescribed_per100') or 0))}"
                     f"<div class='note'>{escape(r['note'])}</div></div>" for r in readback)
        focus_html += f"<h2>上週焦點回讀</h2>{rb}"

    # session observations
    so = d["session_obs"]
    def _obs_rows(items, label):
        if not items:
            return ""
        r = "".join(f"<tr><td>{escape(str(i['key']))}</td><td>{i['n']}</td>"
                    f"<td>{i['per100']:.2f}</td>"
                    f"<td>{'⚠ n&lt;20' if i.get('low_n') else ''}</td></tr>" for i in items)
        return f"<div class='sub'>{label}</div><table><tr><th>{label}</th><th>n</th><th>bb/100</th><th></th></tr>{r}</table>"
    pbb = so.get("post_bad_beat", {})
    session_html = (_obs_rows(so.get("by_hour"), "session 第 N 小時")
                    + _obs_rows(so.get("by_tables"), "同時桌數")
                    + (f"<div class='sub'>大敗手後 15 分鐘窗口：per100 {pbb.get('per100',0):.2f} "
                       f"vs 全體 {pbb.get('baseline_per100',0):.2f}（n={pbb.get('n',0)}"
                       f"{'，樣本不足' if pbb.get('low_n') else ''}）</div>" if pbb else ""))

    hon = d["honesty"]
    honesty_html = (f"<div class='sub'>excluded 決策：{hon['excluded_n']} · "
                    f"unsolved：{hon['unsolved_n']} · "
                    f"chipEV 評分占比：{hon['chipev_share']*100:.0f}%</div>"
                    f"<div class='note'>{escape(hon['note'])}</div>")

    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ledger 記分卡 {escape(d['window'])}</title><style>{_STYLE}</style></head><body>
<h1>Ledger 記分卡</h1><div class="sub">窗口：{escape(d['window'])}</div>
<h2>主指標：EV loss / 100 決策</h2>
<div class="metric">{d['per100']:.2f}<span class="sub"> bb/100 · n={d['n']} · 週變化 {delta_html}</span></div>
{spark}
<h2>Leak 榜（EV 排序）</h2>
<table><tr><th>family</th><th>depth</th><th>總 bb</th><th>n</th><th>bb/100</th><th>趨勢</th><th>型別</th></tr>{rows}</table>{insuff_html}
<h2>最貴 3 手</h2>{hands_html or '<div class="sub">本窗無損失手</div>'}
<h2>本週焦點處方</h2>{focus_html or '<div class="sub">無足夠樣本的焦點</div>'}
<h2>Session 觀察</h2>{session_html or '<div class="sub">無 session 資料</div>'}
<h2>誠實層附註</h2>{honesty_html}
</body></html>"""


def _num(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


# ── preview summary markdown (Task 10 Step 2) ──

def _preview_summary_md(data: dict) -> str:
    d = data
    L = ["# 首份診斷預覽（全期）", ""]
    L.append(f"## 1. 全期主指標")
    trend_dir = "下降" if d["delta"] < 0 else ("上升" if d["delta"] > 0 else "持平")
    L.append(f"- EV loss/100 決策：**{d['per100']:.2f} bb**（n={d['n']}）")
    L.append(f"- 週趨勢方向（最近一週 vs 前一週）：{trend_dir} {abs(d['delta']):.2f} bb/100")
    L.append(f"- 週序列點數：{len(d['weekly_series'])} 週")
    L.append("")
    L.append("## 2. Top 5 leak lines（EV 排序，帶 n）")
    L.append("| # | family | depth | 總 bb | n | bb/100 | 型別 | slice |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(d["leak_board"]["cells"][:5], 1):
        L.append(f"| {i} | {c['family']} | {c['depth_band']} | {c['total_bb']:.1f} | "
                 f"{c['n']} | {c['per100']:.2f} | {_leak_type_zh(c['leak_type'])} | {c.get('slice_desc','')} |")
    L.append("")
    L.append("## 3. 每個 top-3 leak line 的最貴 2 手")
    for c in d["leak_board"]["cells"][:3]:
        L.append(f"### {c['family']} @{c['depth_band']}")
        th = c.get("top_hands") or []
        if not th:
            L.append("- （此預覽未附該 line 手牌明細，見焦點區）")
        for h in th[:2]:
            day = h["played_at"].astimezone(TPE).strftime("%Y-%m-%d") if hasattr(h["played_at"], "astimezone") else str(h["played_at"])
            L.append(f"- {day} · {h.get('hero_hand')} · 損失 {(h.get('total_ev_loss_bb') or 0):.1f}bb · [Analyze]({h.get('review_url','')})")
    L.append("")
    L.append("## 4. 全期最貴 5 手")
    for h in d["top_hands"][:5]:
        day = h["played_at"].astimezone(TPE).strftime("%Y-%m-%d") if hasattr(h["played_at"], "astimezone") else str(h["played_at"])
        L.append(f"- {day} · {h.get('hero_hand')} {h.get('position','')} · {h.get('boards','')} · "
                 f"損失 {(h.get('total_ev_loss_bb') or 0):.1f}bb · [Analyze]({h.get('review_url','')})")
    L.append("")
    L.append("## 5. Session 觀察（帶 n）")
    so = d["session_obs"]
    for i in (so.get("by_hour") or []):
        L.append(f"- 第 {i['key']} 小時：{i['per100']:.2f} bb/100（n={i['n']}{'，樣本不足' if i.get('low_n') else ''}）")
    for i in (so.get("by_tables") or []):
        L.append(f"- {i['key']} 桌：{i['per100']:.2f} bb/100（n={i['n']}{'，樣本不足' if i.get('low_n') else ''}）")
    pbb = so.get("post_bad_beat", {})
    if pbb:
        L.append(f"- 大敗手後 15 分鐘：{pbb.get('per100',0):.2f} vs 全體 {pbb.get('baseline_per100',0):.2f} bb/100（n={pbb.get('n',0)}）")
    L.append("")
    L.append("## 6. 誠實層統計")
    hon = d["honesty"]
    L.append(f"- excluded 決策：{hon['excluded_n']}")
    L.append(f"- unsolved 決策：{hon['unsolved_n']}")
    L.append(f"- chipEV 評分占比：{hon['chipev_share']*100:.1f}%")
    L.append("")
    L.append("## 7. 假設本週開處方（dry-run，未寫表）")
    for f in d["focus"]["families"]:
        L.append(f"- 焦點 **{f['family']} @{f['depth_band']}**（{_leak_type_zh(f.get('leak_type',''))}，"
                 f"近窗 {f['total_bb']:.1f}bb / n={f['n']}）")
        for p in f.get("prescriptions", []):
            L.append(f"  - {p['label']}: {p['url']}")
    L.append("")
    return "\n".join(L)


# ── CLI ──

async def _run(mode: str):
    import os
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    outdir = ROOT / "data" / "scorecards"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        # full history feeds the 18-week trend + adequate leak-board samples
        decisions = await diag.fetch_decisions(conn)
        hands = await diag.fetch_hands(conn)
        sessions = await diag.fetch_sessions(conn)

        if mode == "preview":
            data = compute_scorecard_data(decisions, hands, sessions,
                                          prev_focus=None, window_label="preview-all-history")
            (outdir / "preview.html").write_text(render_scorecard_html(data))
            (outdir / "preview_data.json").write_text(json.dumps(data, default=str, ensure_ascii=False, indent=1))
            (outdir / "preview_summary.md").write_text(_preview_summary_md(data))
            print(f"PREVIEW written: decisions={len(decisions)} hands={len(hands)} "
                  f"per100={data['per100']:.2f} leaks={len(data['leak_board']['cells'])}")
            return 0

        # weekly: current ISO week, write tables + readback backfill
        now = datetime.now(TPE)
        y, wk, _ = now.isocalendar()
        week_label = f"{y}-W{wk:02d}"
        prev = await conn.fetchrow(
            "SELECT week, families FROM coach_focus ORDER BY created_at DESC LIMIT 1")
        prev_focus = ({"families": json.loads(prev["families"]) if isinstance(prev["families"], str)
                       else prev["families"]} if prev else None)
        data = compute_scorecard_data(decisions, hands, sessions,
                                      prev_focus=prev_focus, window_label=week_label)
        html = render_scorecard_html(data)
        (outdir / f"{week_label}.html").write_text(html)
        # record per100 into each focus family so next week can read back
        fam_payload = [dict(f, per100=f.get("per100")) for f in data["focus"]["families"]]
        await conn.execute(
            "INSERT INTO scorecards (week, html, data_json) VALUES ($1,$2,$3) "
            "ON CONFLICT (week) DO UPDATE SET html=EXCLUDED.html, data_json=EXCLUDED.data_json",
            week_label, html, json.dumps(data, default=str))
        await conn.execute(
            "INSERT INTO coach_focus (week, families, rationale, prescriptions, readback) "
            "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (week) DO UPDATE SET "
            "families=EXCLUDED.families, prescriptions=EXCLUDED.prescriptions",
            week_label, json.dumps(fam_payload, default=str),
            json.dumps({"per100": data["per100"], "n": data["n"]}, default=str),
            json.dumps([p for f in data["focus"]["families"] for p in f.get("prescriptions", [])], default=str),
            json.dumps(None))
        if prev and data["focus"].get("readback"):
            await conn.execute("UPDATE coach_focus SET readback=$2 WHERE week=$1",
                               prev["week"], json.dumps(data["focus"]["readback"], default=str))
        print(f"WEEKLY written: week={week_label} per100={data['per100']:.2f}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    mode = "preview" if "--preview" in sys.argv else ("weekly" if "--weekly" in sys.argv else None)
    if mode is None:
        print("usage: scorecard.py --preview | --weekly")
        raise SystemExit(1)
    raise SystemExit(asyncio.run(_run(mode)))
