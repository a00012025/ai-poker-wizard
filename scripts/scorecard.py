#!/usr/bin/env python3
"""Weekly scorecard = training plan (action-line taxonomy, Version A loop).

Diagnose (action-line leak board by avg EV loss) -> prescribe 1-2 focus spots
with precise multi-depth GTOW Trainer drill links (the drill itself is the
retrieval practice) -> next-cycle EV-loss readback on the treated spot.

Windows (§2.2 歸因): weekly mode diagnoses only the current Taipei ISO week,
and the readback of last week's focus spot is computed strictly over the
post-prescription window — the only window where treatment can be observed.

--preview   full-history window, no DB writes; emits data/scorecards/preview.*
--weekly    current ISO week, writes scorecards + coach_focus + readback backfill
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
from plan_scheduler import QUEUE_SLOTS, TRACK_SLOTS
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

_ACTION_ZH = {"x": "過牌", "b": "下注", "c": "跟注", "r": "加注", "f": "棄牌"}
_BAND_WEAK_ZH = {"short": "≤20bb 特別弱", "medium": "20–50bb 特別弱",
                 "large": ">50bb 特別弱"}


def _actions_zh(raw: str) -> str:
    actions = []
    for token in str(raw or "").split("-"):
        key = token.strip().lower()[:1]
        actions.append(_ACTION_ZH.get(key, token))
    return "→".join(action for action in actions if action)


def action_line_zh(row: dict) -> str:
    """Translate one exact taxonomy leaf into a table-readable action line."""
    leaf = str(row.get("representative_leaf") or row.get("spot_leaf") or "")
    cat = str(row.get("spot_category") or "")
    if cat not in {"flop", "turn", "river"}:
        hero = row.get("hero_pos") or row.get("hero_cat") or "Hero"
        villain = row.get("villain_cat")
        rel = row.get("ip_oop")
        where = f"（{rel}）" if rel else ""
        target = f"{villain} " if villain else ""
        wording = {
            "RFI": "率先開池",
            "vsOpen": f"面對 {target}開池",
            "vsRaiseCall": f"面對 {target}開池及一人跟注",
            "vsSqueeze": f"面對 {target}squeeze",
            "vs3bet": f"面對 {target}3bet",
            "vsCold3bet": f"cold call 前面對 {target}3bet",
            "vs4bet": f"面對 {target}4bet",
            "vsCold4bet": f"cold call 前面對 {target}4bet",
        }.get(cat, CAT_ZH.get(cat, cat or "這個情境"))
        return f"Hero {hero}{where}，{wording}"

    parts = leaf.split(":")
    pot = parts[1] if len(parts) > 1 else "?"
    matchup = parts[2] if len(parts) > 2 else ""
    match_parts = matchup.split("v", 1)
    leaf_hero = match_parts[0] if match_parts else ""
    leaf_villain = match_parts[1] if len(match_parts) > 1 else ""
    hero = row.get("hero_pos") or row.get("hero_cat") or leaf_hero or "Hero"
    villain = row.get("villain_cat") or leaf_villain
    rel = row.get("ip_oop") or (parts[3] if len(parts) > 3 else "")
    chunks = [f"{pot}｜Hero {hero}" + (f" 對 {villain}" if villain else "")
              + (f"，{rel}" if rel else "")]
    match = re.search(r":\[([^\]]*)\]:", leaf)
    if match:
        histories = match.group(1).split("|")
        if histories and histories[0] not in {"", "-", "None", "null"}:
            chunks.append(f"翻牌 {_actions_zh(histories[0])}")
        if cat == "river" and len(histories) > 1 and histories[1] not in {
                "", "-", "None", "null"}:
            chunks.append(f"轉牌 {_actions_zh(histories[1])}")
    facing = parts[-1] if parts else ""
    node = {
        "first_to_act": f"{CAT_ZH[cat]}輪到 Hero 先行動",
        "vs_bet": f"{CAT_ZH[cat]} Hero 面對下注",
        "vs_check": f"{CAT_ZH[cat]}對手過牌到 Hero",
        "vs_raise": f"{CAT_ZH[cat]} Hero 面對加注",
    }.get(facing, f"{CAT_ZH[cat]} {FACING_ZH.get(facing, facing)}")
    chunks.append(node)
    return "；".join(chunks)


# ── human-readable spot description (pure) ─────────────────────────────────
def spot_desc_zh(row: dict) -> str:
    cat = row["spot_category"]
    if row.get("diagnosis_level") == "parent":
        key = row.get("diagnosis_key") or "?"
        parts = key.split(":")
        if cat in ("flop", "turn", "river"):
            return action_line_zh(row)
        hero = row.get("hero_pos") or row.get("hero_cat") or "?"
        villain = row.get("villain_cat")
        rel = row.get("ip_oop")
        if cat == "RFI":
            return f"Hero {hero} 開池"
        if cat == "vsOpen":
            return f"Hero {hero} 面對 {villain or '?'} 開池"
        if cat == "vsSqueeze" and "flat_vsSqueeze" in key:
            matchup = f"Hero {hero} flat 後對 {villain or '?'} squeeze"
            if rel:
                matchup += f"、處於 {rel}"
            return matchup
        if cat in ("vs3bet", "vsCold3bet", "vs4bet", "vsCold4bet", "vsSqueeze"):
            matchup = f"Hero {hero} 對 {villain or '?'}"
            if rel:
                matchup += f"、處於 {rel}"
            return f"{matchup}，{CAT_ZH[cat]}"
        return action_line_zh(row)
    hc, vc, rel = row.get("hero_cat"), row.get("villain_cat"), row.get("ip_oop")
    if cat in ("flop", "turn", "river"):
        return action_line_zh(row)
    if cat == "RFI":
        return f"{row.get('hero_pos') or hc} 開池"
    if cat == "vsOpen":
        return f"{row.get('hero_pos') or hc} 面對 {vc} 開池"
    if cat == "vsSqueeze" and "flat_vsSqueeze" in row["spot_leaf"]:
        return f"{hc} flat 後面對 squeeze（對手 {vc}，你 {rel or '?'}）"
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


# diagnosis window for --weekly: current Taipei ISO week, not historical debt
READBACK_MIN_N = 10  # early directional signal only; never a causal verdict
LIVE_FOCUS_MIN_TOTAL_BB = 0.10

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
                          readback, honesty, focus_k=2, focus_exclude=None) -> dict:
    """Assemble the weekly plan.

    ``focus_exclude`` holds diagnosis keys under focus cooldown (see
    ``plan_scheduler.focus_exclusions``). They are skipped when choosing this
    week's focus but stay on the leak board: hiding a spot from the ranking
    just because it was recently treated would misreport where EV is going.
    """
    current = weekly_series[-1] if weekly_series else None
    previous = weekly_series[-2] if len(weekly_series) >= 2 else None
    if re.fullmatch(r"\d{4}-W\d{2}", window_label):
        current = next((row for row in weekly_series
                        if row.get("week") == window_label), None)
        prior = [row for row in weekly_series
                 if str(row.get("week") or "") < window_label]
        previous = prior[-1] if prior else None
    has_current_week_data = current is not None
    per100 = float(current.get("per100") or 0.0) if current else 0.0
    n = int(current.get("n") or 0) if current else 0
    hands = int(current.get("hands") or n) if current else 0
    previous_hands = int(previous.get("hands") or previous.get("n") or 0) if previous else 0
    delta = (per100 - float(previous.get("per100") or 0.0)
             if current and previous else 0.0)
    change = (f"本週觀察值較上週少 {abs(delta):.2f}" if delta < 0 else
              (f"本週觀察值較上週多 {delta:.2f}" if delta > 0 else "與上週相同"))
    blocked = set(focus_exclude or ())
    focus_pool = [it for it in spots
                  if (it["row"].get("diagnosis_key")
                      or it["row"].get("spot_leaf")) not in blocked]
    focus = []
    for it in focus_pool[:focus_k]:
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
            "source": it.get("source", "online"),
            "action_bias": r.get("action_bias"),
            "samples": [dict(s) for s in it.get("samples", [])],
        })
    return {
        "window": window_label,
        "headline": f"本週平均 EV 損失 {per100:.2f} bb/100（共 {hands} 手），{change}",
        "per100": per100, "n": n, "hands": hands,
        "previous_hands": previous_hands,
        "has_current_week_data": has_current_week_data,
        "delta": delta, "weekly_series": weekly_series,
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
        out.append({"spot_leaf": leaf, "label": f.get("desc") or f.get("label") or leaf,
                    "prescribed_per100": f.get("per100"),
                    "current_per100": cur["per100"] if has_data else None,
                    "n": cur["n"] if cur else 0,
                    "note": "列入練習後的實戰讀數"})
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


def exact_analyze_url(hand_ids) -> str | None:
    """One GTOW Analyze table link filtered to the exact source hands."""
    from queue_feed import gtow_analyze_hands_urls
    urls = gtow_analyze_hands_urls([hand_id for hand_id in hand_ids if hand_id])
    return urls[0][0] if urls else None


def readback_signal(row: dict) -> str | None:
    """Small-sample-friendly direction without claiming causal improvement."""
    if row.get("current_per100") is None or int(row.get("n") or 0) < READBACK_MIN_N:
        return None
    before = row.get("prescribed_per100")
    if before is None:
        return "已達可觀察門檻，繼續追蹤"
    delta = float(row["current_per100"]) - float(before)
    if delta < -0.05:
        return f"初步訊號：比列入訓練時少 {abs(delta):.1f} bb/100"
    if delta > 0.05:
        return f"初步訊號：比列入訓練時多 {delta:.1f} bb/100"
    return "初步訊號：與列入訓練時相近"


def _sample_html(sample: dict, source: str) -> str:
    hand = escape(str(sample.get("hero_hand") or "?"))
    board = escape(str(sample.get("boards") or ""))
    loss = float(sample.get("ev_loss_bb") or 0.0)
    link = exact_analyze_url([sample.get("gtow_hand_id")]) if source == "online" else None
    action = (f'<a href="{escape(link)}">在 Hand Analyzer 看這手</a>'
              if link else "線下來源手")
    return f'<div class="sub">· {hand} {board} 損失 {loss:.1f}bb {action}</div>'


def render_html(d: dict) -> str:
    spark = _svg_sparkline([w["per100"] for w in d["weekly_series"]])
    lb_rows = "".join(
        f"<tr><td>{escape(spot_desc_zh(r))}"
        f"{('｜' + escape(r['action_bias']['label'])) if r.get('action_bias') else ''}</td>"
        f"<td>{r['avg_ev']*100:.2f}</td>"
        f"<td>{r['n']}</td></tr>" for r in d["leaderboard"][:8])
    focus_html = ""
    for f in d["focus"]:
        drill = (f'<a class="btn" href="{escape(f["drill_url"])}">🎯 進 GTOW Trainer 練這個情境</a>'
                 if f["drill_url"] else '<span class="note">（目前無法建立準確的 Trainer 連結，請從下方 Analyze 樣本複習）</span>')
        band = (f'<div class="note">⛏ {_BAND_WEAK_ZH.get(f["restrict"], f["restrict"] + " 特別弱")}</div>'
                if f.get("restrict") else '<div class="note">練習涵蓋各種籌碼深度</div>')
        samples = "".join(_sample_html(s, s.get("hand_source", f["source"]))
                          for s in f.get("samples", []))
        frag = ('<div class="note">⚠️ 部分下注尺寸不在 GTOW 標準樹上；排除這些手後結果差超過 30%，先不要過度解讀</div>'
                if f.get("fragile") else '')
        bias = f.get("action_bias")
        bias_html = (f'<div><b>明顯傾向：{escape(bias["label"])}</b>'
                     f'（{bias["n"]} 手，EV 損失合計 {bias["ev_loss_bb"]:.2f} bb）</div>'
                     if bias else '')
        source_zh = "線下" if f.get("source") == "live" else "線上"
        focus_html += (f'<div class="card"><b>{escape(f["desc"])}</b>'
                       f'<div class="sub">{source_zh}｜平均 EV 損失 {f["per100"]:.2f} bb/100'
                       f'（本週出現 {f["n"]} 次）</div>'
                       f'{bias_html}{drill}{band}{frag}{samples}</div>')
    rb = ""
    if d.get("readback"):
        for r in d["readback"]:
            label = escape(str(r.get("label") or r["spot_leaf"]))
            signal = readback_signal(r)
            if signal:
                body = f'目前 {r["current_per100"]:.2f} bb/100（{r["n"]} 次）'
                rb += (f'<div class="card">{label}：{body}'
                       f'<div class="note">{escape(signal)}；仍需更多實戰驗證。</div></div>')
    top = "".join(
        f'<div class="sub">· {escape(str(h.get("hero_hand") or "?"))} {escape(str(h.get("position") or ""))} '
        f'{escape(str(h.get("boards") or ""))} 損失 {(h.get("total_ev_loss_bb") or 0):.1f}bb '
        f'<a href="{exact_analyze_url([h.get("gtow_hand_id")])}">在 Hand Analyzer 看這手</a></div>'
        for h in d["top_hands"][:3])
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>訓練計畫 {escape(d['window'])}</title><style>{_STYLE}</style></head><body>
<h1>週訓練計畫</h1><div class="sub">統計期間：{escape(d['window'])}</div>
<h2>主指標：平均 EV 損失 / 100 決策</h2>
<div class="metric">{d['per100']:.2f}<span class="sub"> bb/100 · 共 {d.get('hands', d.get('n', 0))} 手 · 週變化 {_trend(d['delta'])}</span></div>{spark}
<h2>本週最該練的情境（先作答，再練）</h2>{focus_html or '<div class="sub">目前沒有樣本足夠的焦點</div>'}
{('<h2>列入練習後的實戰追蹤</h2>'+rb) if rb else ''}
<h2>EV 損失較高的情境</h2>
<div class="note">優先順序已考慮樣本數；表中顯示原始平均 EV 損失。</div>
<table><tr><th>情境類型</th><th>平均 EV 損失 bb/100</th><th>樣本決策</th></tr>{lb_rows}</table>
<h2>最貴 3 手</h2>{top}
<h2>資料口徑</h2><div class="sub">整體指標與其他節點僅納入高信心線上決策；焦點另看本週高信心線下牌，但與線上分開排序。翻牌後採 chipEV 評分，翻牌前涉及 limp 的決策不納入。</div>
</body></html>"""


TRACK_ZH = {"online": "🖥 線上", "live": "🎲 線下"}


def recommendation_ev(q: dict) -> float:
    """Current-window EV loss used by both recommendation text and buttons."""
    value = q.get("week_total_ev_loss_bb")
    if value is None:
        value = q.get("total_ev_loss_bb")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _focus_recommendation(focus: dict) -> dict | None:
    """Normalize an older scorecard focus row into the unified action list.

    Newly generated scorecards already put focus prescriptions through the
    queue scheduler. This adapter keeps the latest persisted pre-migration
    scorecard usable until the next weekly run, while dedupe below prevents a
    focus from appearing twice once both shapes are present.
    """
    qid = focus.get("queue_id")
    if qid is None:
        return None
    n = int(focus.get("n") or 0)
    avg = float(focus.get("per100") or 0.0) / 100
    shrunk = float(focus.get("shrunk_per100", focus.get("per100") or 0.0)) / 100
    samples = focus.get("samples") or []
    hand_ids = [s.get("gtow_hand_id") for s in samples if s.get("gtow_hand_id")]
    bias = focus.get("action_bias") or {}
    return {
        "id": int(qid), "kind": "drill",
        "track": focus.get("source", "online"),
        "spot_leaf": focus.get("spot_leaf"),
        "spot_category": focus.get("spot_category"),
        "label": focus.get("desc") or focus.get("spot_leaf") or "?",
        "drill_url": focus.get("drill_url"),
        "n_sources": n,
        "total_ev_loss_bb": avg * n,
        "priority_ev_loss_bb": shrunk * n,
        "week_analyze_url": (exact_analyze_url(hand_ids)
                             if focus.get("source", "online") == "online"
                             else None),
        "bias_direction": bias.get("direction"),
        "bias_n": bias.get("n"),
        "bias_ev_loss_bb": bias.get("ev_loss_bb"),
        "surfaced_count": 0,
    }


def _priority_ev(q: dict) -> float:
    value = q.get("priority_ev_loss_bb")
    return recommendation_ev(q) if value is None else float(value or 0.0)


def ordered_queue(d: dict) -> list[dict]:
    """One recommendation list, highest current-window EV loss first.

    Sample reliability and live priority are admission policy, not a hidden
    display multiplier: online family candidates are shrinkage-ranked and the
    scheduler reserves two of five seats for live evidence. Once admitted,
    every item is shown by the actual EV loss the player can inspect.
    """
    candidates = [dict(q) for q in (d.get("drill_queue") or [])]
    seen_ids = {int(q["id"]) for q in candidates if q.get("id") is not None}
    seen_spots = {q.get("spot_leaf") for q in candidates if q.get("spot_leaf")}
    for focus in d.get("focus") or []:
        item = _focus_recommendation(focus)
        if not item or item["id"] in seen_ids or item.get("spot_leaf") in seen_spots:
            continue
        candidates.append(item)
        seen_ids.add(item["id"])
        if item.get("spot_leaf"):
            seen_spots.add(item["spot_leaf"])

    rank = lambda q: (-_priority_ev(q),
                      0 if q.get("track") == "live" else 1,
                      int(q.get("id") or 0))
    if len(candidates) > QUEUE_SLOTS:
        picked = []
        for track in ("online", "live"):
            rows = sorted((q for q in candidates if q.get("track", "online") == track),
                          key=rank)
            picked.extend(rows[:TRACK_SLOTS[track]])
        chosen = {id(q) for q in picked}
        remainder = sorted((q for q in candidates if id(q) not in chosen), key=rank)
        picked.extend(remainder[:QUEUE_SLOTS - len(picked)])
        candidates = picked[:QUEUE_SLOTS]
    return sorted(candidates, key=rank)


def repeat_note(q: dict) -> str:
    """Say out loud that an item is coming round again, and why.

    §14.2 keeps unpracticed prescriptions alive; being honest that this is
    repeat number N is what stops that from reading as a broken plan.
    """
    times = int(q.get("surfaced_count") or 0)
    if times <= 0:
        return ""
    if q.get("bucket") == "relapse":
        return f"（🔁 這週又出現，第 {times + 1} 次排入）"
    return f"（之前排過但還沒完成，第 {times + 1} 次提醒）"


def backlog_remaining(d: dict) -> int:
    """Open backlog items NOT shown in this week's slate."""
    shown = sum(1 for q in (d.get("drill_queue") or [])
                if q.get("bucket") == "backlog")
    return max(0, int(d.get("queue_backlog_total") or 0) - shown)


def weekly_tg_html(week: str, d: dict) -> str:
    """End-user weekly coaching message for Telegram (HTML parse_mode).

    Concise, no jargon (no "北極星"/"迴圈"). Tells the player: what to drill,
    where they're leaking, the goal, and the honest data caveats (chipEV/ICM,
    dropped limps, coverage). Drill actions are NOT embedded — they ride as
    compact queue-detail buttons (weekly_tg_payload) under the message.
    """
    per100 = d.get("per100", 0.0)
    series = d.get("weekly_series") or []
    hands = int(d.get("hands") or
                (series[-1].get("hands") if series else 0) or d.get("n") or 0)
    previous_hands = int(d.get("previous_hands") or
                         (series[-2].get("hands") or series[-2].get("n")
                          if len(series) >= 2 else 0) or 0)
    delta = d.get("delta", 0.0)
    if delta < -1e-9:
        trend = f"本週觀察值較上週少了 <b>{abs(delta):.2f} bb/100</b>（上週 {previous_hands} 手）"
    elif delta > 1e-9:
        trend = f"本週觀察值較上週多 <b>{delta:.2f} bb/100</b>（上週 {previous_hands} 手）"
    else:
        trend = f"與上週相同（上週 {previous_hands} 手）"

    L = [f"🃏 <b>本週該練的地方</b>（{escape(week)}）", ""]
    if d.get("has_current_week_data", True):
        L.append(f"這週共 <b>{hands} 手</b>，平均 EV 損失為 "
                 f"<b>{per100:.2f} bb/100</b>；{trend}。")
    else:
        L.append(f"這週尚無符合統計口徑的線上手牌（<b>0 手</b>）；"
                 f"上週共 {previous_hands} 手。")

    dq = ordered_queue(d)
    if dq:
        L.append("")
        L.append("<b>📌 本週建議（EV 優先）：</b>")
        for qi, q in enumerate(dq, 1):
            track = q.get("track", "online")
            source = "線下" if track == "live" else "線上"
            if q.get("kind") == "review":
                lbl = q.get("label") or q.get("spot_leaf") or "?"
            else:
                from spot_naming import compact_spot_name
                lbl = compact_spot_name(q)
            note = repeat_note(q)
            if q.get("kind") == "review":
                # label already reads「復盤 M/D … −Xbb」
                L.append(f"{qi}. 🔍 [復盤·{source}] {escape(lbl)}{note}")
            else:
                ev = q.get("week_total_ev_loss_bb", q.get("total_ev_loss_bb")) or 0
                n_sources = q.get("week_n_sources", q.get("n_sources", 1))
                L.append(f"{qi}. 🎯 [訓練·{source}] {escape(lbl)} — {n_sources} 手，"
                         f"EV 損失合計 {ev:.1f} bb{note}")
                from spot_naming import telegram_bias_summary
                bias = telegram_bias_summary(q)
                if bias:
                    L.append(f"  ↳ {escape(bias)}")
    remaining = backlog_remaining(d)
    if remaining and len(dq) < QUEUE_SLOTS:
        L.append("")
        L.append(f"本週沒有更多新的漏洞 — 這是好消息。佇列裡還有 {remaining} 項"
                 "沒完成的舊處方，想加練用 /queue。")

    readback = d.get("readback") or []
    visible_readback = [r for r in readback if readback_signal(r)]
    if visible_readback:
        L.append("")
        L.append("<b>📈 列入練習後的實戰追蹤：</b>")
        for r in visible_readback:
            label = escape(str(r.get("label") or r.get("spot_leaf") or "?"))
            L.append(f"• {label} — 目前 {r['current_per100']:.1f} bb/100"
                     f"（{r['n']} 次）；{escape(readback_signal(r))}")
        L.append("  這是提早回饋，仍需更多實戰樣本確認是否真的學會。")

    L.append("")
    L.append("🎯 <b>本週目標</b>：完成以上訓練與復盤建議。")

    L.append("")
    L.append("⚠️ <b>統計口徑</b>：整體指標與線上排序只算高信心線上決策；"
             "線下牌保留較高訓練權重，但不和線上混算。翻牌後採 chipEV 評分，"
             "翻牌前涉及 limp 的決策不納入。")
    return "\n".join(L)


def weekly_tg_payload(week: str, d: dict) -> dict:
    """Weekly TG message + inline buttons: {"html": str, "buttons": rows}.

    Buttons follow the one EV-ordered recommendation list. Drill callbacks use
    the existing detail/provisioning flow and carry ``plan`` so the weekly
    message stays immutable. Every queue item exposes 📚 exact
    source hands (qsrc); review items additionally carry 🔗 復盤 + ✔ 完成
    (qcl) + ➕ 加練 (qex) (§7/§6.2).
    """
    buttons: list[list[dict]] = []
    for qi, q in enumerate(ordered_queue(d), 1):
        qid = q.get("id")
        if q.get("kind") == "review":
            lbl = q.get("label") or q.get("spot_leaf") or "?"
        else:
            from spot_naming import compact_spot_name
            lbl = compact_spot_name(q)
        lbl = lbl[:24]
        if q.get("kind") == "review":
            row: list[dict] = []
            anchor = q.get("review_anchor_url")
            anchor_street = q.get("review_anchor_street")
            if anchor:
                row.append({"text": f"↩ {qi} {(anchor_street or '上游').title()}",
                            "url": anchor})
            if q.get("drill_url"):
                is_solution = "/solutions" in str(q["drill_url"])
                text = f"🔍 解法 {qi}" if is_solution else f"🃏 原手 {qi}"
                row.append({"text": text,
                            "url": q["drill_url"]})
            actions: list[dict] = []
            if qid is not None:
                actions.append({"text": f"🃏 原手 {qi}", "callback_data": f"qsrc:{qid}"})
                actions.append({"text": f"✔ 完成 {qi}",
                                "callback_data": f"qcl:{qid}:0:completed:plan"})
                actions.append({"text": f"➕ 加練 {qi}", "callback_data": f"qex:{qid}"})
            if anchor and row:
                buttons.append(row)
                row = actions
            else:
                row.extend(actions)
            if row:
                buttons.append(row)
        else:
            row = []
            if qid is not None:
                row.append({"text": f"🎯 練 {qi}",
                            "callback_data": f"qdet:{qid}:0:plan"})
                if q.get("week_analyze_url"):
                    row.append({"text": f"🃏 看手 {qi}",
                                "url": q["week_analyze_url"]})
                else:
                    row.append({"text": f"🃏 看手 {qi}",
                                "callback_data": f"qsrc:{qid}"})
            if row:
                buttons.append(row)
    return {"html": weekly_tg_html(week, d), "buttons": buttons}


def preview_summary_md(d: dict) -> str:
    L = [f"# 訓練計畫預覽（{d['window']}）", "",
         f"## 主指標", f"- 平均 EV 損失：**{d['per100']:.2f} bb/100**"
         f"（共 {d.get('hands', d.get('n', 0))} 手；週變化 {d['delta']:+.2f}）",
         f"- 已累積：{len(d['weekly_series'])} 週資料", "", "## 本週最該練的情境",
         "優先順序已考慮樣本數；下方顯示原始平均 EV 損失。"]
    for f in d["focus"]:
        L.append(f"### {f['desc']}  `{f.get('diagnosis_key') or f['spot_leaf']}`")
        L.append(f"- 平均 EV 損失 {f['per100']:.2f} bb/100（出現 {f['n']} 次）")
        if f.get("action_bias"):
            b = f["action_bias"]
            L.append(f"- 明顯傾向：{b['label']}（{b['n']} 手，EV 損失合計 {b['ev_loss_bb']:.2f} bb）")
        if f["drill_url"]:
            L.append(f"- 🎯 drill：{f['drill_url']}")
        L.append("")
    L.append("## EV 損失較高的情境")
    L.append("")
    L.append("| 情境類型 | 平均 EV 損失 bb/100 | 樣本決策 |")
    L.append("|---|---:|---:|")
    for r in d["leaderboard"][:10]:
        label = r.get('diagnosis_key', r['spot_leaf'])
        if r.get("action_bias"):
            label += f"｜{r['action_bias']['label']}"
        L.append(f"| {label} | {r['avg_ev']*100:.2f} | {r['n']} |")
    L.append("")
    L.append("## 資料口徑\n- 整體指標與其他節點只算高信心線上決策；"
             "焦點另看高信心線下牌，但不和線上混算。翻牌後採 chipEV 評分，"
             "翻牌前涉及 limp 的決策不納入。")
    return "\n".join(L)


# ── async fetch + build + CLI ──────────────────────────────────────────────
# All stats queries are source='online' only (§5.2 source isolation): live
# hands are selectively recorded — their averages are biased by design and
# surface in their own queue section instead.
WEEKLY_SQL = """
SELECT to_char((played_at AT TIME ZONE 'Asia/Taipei'), 'IYYY-"W"IW') week,
       count(*) n, count(DISTINCT gtow_hand_id) hands,
       avg(ev_loss_bb)*100 per100, sum(ev_loss_bb) total_bb
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
WHERE spot_leaf=$1 AND NOT excluded AND NOT discarded AND source=$3
  AND confidence >= 0.8 AND played_at >= $2
"""
READBACK_WINDOW_SQL_PARENT = READBACK_WINDOW_SQL.replace("spot_leaf=$1", "spot_parent=$1")

# Prescribed-but-uncleared items are never DELETED (§14.2: silently dropping an
# unpracticed prescription degrades the coaching signal) — they stay in /queue
# forever. They just stop monopolising the weekly plan: plan_scheduler sorts
# open rows into fresh / relapsed / backlog and fills reserved online+live
# seats, rotating at most one backlog item per track. Fetch every open row so
# the backlog tally the message reports is the real one.
QUEUE_SQL = """
SELECT id, spot_leaf, spot_category, label, drill_url, review_anchor_url,
       review_anchor_street, n_sources, total_ev_loss_bb, source, status,
       prescribed_week, kind, ref_hand_id, bias_direction, bias_n,
       bias_ev_loss_bb, bias_share, source_hands, surfaced_count,
       last_surfaced_at, last_surfaced_week, depth_scope
FROM drill_queue WHERE status IN ('pending', 'prescribed')
ORDER BY (status = 'pending') DESC, total_ev_loss_bb DESC NULLS LAST LIMIT 200
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


_WEEK_DRILL_EVIDENCE_SQL = """
SELECT gtow_hand_id hand_id, street, decision_idx, ev_loss_bb
FROM ledger_decisions
WHERE spot_leaf=$1 AND source=$2 AND played_at >= $3
  AND ev_loss_bb >= 0.10 AND NOT excluded AND NOT discarded
  AND confidence >= 0.8
ORDER BY ev_loss_bb DESC
"""
_WEEK_REVIEW_EVIDENCE_SQL = """
SELECT count(*) FROM ledger_hands
WHERE gtow_hand_id=$1 AND source=$2 AND played_at >= $3
"""


async def fetch_drill_queue(conn, exclude_ids=None, since=None) -> dict:
    """This week's practice slate: ``{"picked": [...], "backlog_total": int}``.

    Freshness and the reserved online/live seats are decided by
    ``plan_scheduler.select_weekly_slate``; this only loads the open rows and
    normalizes their display labels.
    """
    from plan_scheduler import annotate_rows, select_weekly_slate
    rows = [dict(r) for r in await conn.fetch(QUEUE_SQL)]
    excluded = {int(i) for i in (exclude_ids or [])}
    rows = [row for row in rows if int(row["id"]) not in excluded]
    from spot_naming import compact_spot_name
    for row in rows:
        if row.get("kind") == "drill":
            row["label"] = compact_spot_name(row)
    rows = await annotate_rows(conn, rows)
    if since:
        current = []
        for row in rows:
            if row.get("kind") == "review":
                count = await conn.fetchval(
                    _WEEK_REVIEW_EVIDENCE_SQL, row.get("ref_hand_id"),
                    row.get("track", "online"), since)
            else:
                evidence = [dict(item) for item in await conn.fetch(
                    _WEEK_DRILL_EVIDENCE_SQL, row.get("spot_leaf"),
                    row.get("track", "online"), since)]
                count = len(evidence)
            if int(count or 0) > 0:
                item = dict(row)
                item["week_evidence_n"] = int(count)
                if row.get("kind") == "drill":
                    item["week_source_hands"] = [dict(source, src=row.get(
                        "track", "online")) for source in evidence]
                    item["week_n_sources"] = len({source["hand_id"]
                                                  for source in evidence})
                    item["week_total_ev_loss_bb"] = round(sum(
                        float(source.get("ev_loss_bb") or 0.0)
                        for source in evidence), 4)
                    if row.get("track") == "online":
                        item["week_analyze_url"] = exact_analyze_url(
                            [source["hand_id"] for source in evidence])
                current.append(item)
        rows = current
    return select_weekly_slate(rows)


def focus_queue_item(focus: dict) -> dict | None:
    """Build an idempotent drill_queue item for a weekly focus prescription."""
    if not focus.get("spot_leaf"):
        return None
    sources = [{
        "hand_id": s.get("gtow_hand_id"),
        "street": s.get("street"),
        "decision_idx": s.get("decision_idx"),
        "ev_loss_bb": float(s.get("ev_loss_bb") or 0.0),
        "src": focus.get("source", "online"),
    } for s in (focus.get("samples") or []) if s.get("gtow_hand_id")]
    if not sources:
        return None
    return {
        "kind": "drill", "added_by": "scorecard_focus",
        "source": focus.get("source", "online"),
        "spot_leaf": focus["spot_leaf"],
        "spot_category": focus.get("spot_category"),
        "label": focus.get("desc") or focus["spot_leaf"],
        "drill_url": focus["drill_url"],
        "total_ev_loss_bb": round(sum(s["ev_loss_bb"] for s in sources), 4),
        "source_hands": sources,
    }


async def bind_focus_queue_items(conn, focus: list[dict]) -> list[int]:
    """Ensure every actionable focus uses the existing Drill detail flow."""
    from queue_feed import enqueue_one
    from spot_naming import drill_depth_scope
    ids = []
    for prescription in focus:
        item = focus_queue_item(prescription)
        if not item:
            continue
        await enqueue_one(conn, item)
        row = await conn.fetchrow(
            "SELECT id FROM drill_queue WHERE spot_leaf=$1 AND depth_scope=$2 "
            "AND kind='drill' "
            "AND status IN ('pending','prescribed') "
            "ORDER BY (status='pending') DESC, last_added DESC LIMIT 1",
            item["spot_leaf"], drill_depth_scope(item))
        if row:
            queue_id = int(row["id"])
            prescription["queue_id"] = queue_id
            ids.append(queue_id)
    return ids


async def fetch_readback(conn, prev_focus, prev_at) -> dict:
    """{leaf: {n, per100}} over the post-prescription window (played_at >= prev_at)."""
    out = {}
    for f in prev_focus or []:
        key = f.get("diagnosis_key") or f.get("spot_leaf")
        if not key:
            continue
        sql = (READBACK_WINDOW_SQL_PARENT
               if f.get("diagnosis_level") == "parent" else READBACK_WINDOW_SQL)
        r = await conn.fetchrow(sql, key, prev_at, f.get("source", "online"))
        out[f.get("spot_leaf") or key] = {"n": r["n"] or 0,
                     "per100": float(r["per100"]) if r["per100"] is not None else None}
    return out


async def focus_history(conn) -> list[dict]:
    """All past focus prescriptions, newest first, for the cooldown.

    The diagnosis window is 90 days, so a fixed 12-week limit leaves a gap in
    which an 85–90-day-old prescription can be selected again without fresh
    evidence. The table grows by one small row per week; reading it all keeps
    the "time alone never resurrects a treated spot" contract exact.
    """
    out = []
    for row in await conn.fetch(
            "SELECT week, families, created_at FROM coach_focus "
            "ORDER BY created_at DESC"):
        fam = row["families"]
        fam = json.loads(fam) if isinstance(fam, str) else fam
        for entry in fam or []:
            if entry.get("source", "online") != "online":
                continue
            key = entry.get("diagnosis_key") or entry.get("spot_leaf")
            if key:
                out.append({"diagnosis_key": key,
                            "diagnosis_level": entry.get("diagnosis_level"),
                            "prescribed_at": row["created_at"],
                            "week": row["week"]})
    return out


def weekly_focus_candidates(online: list[dict], live: list[dict],
                            focus_k: int = 2) -> list[dict]:
    """Reserve one headline seat for selective live evidence.

    Sources remain statistically isolated: this composes already-ranked lists
    and never creates a mixed live/online average.  A live candidate gets the
    first seat because scarce live evidence deserves explicit coaching
    attention; online fills every unused seat.
    """
    picked = []
    seen = set()
    material_live = [item for item in live
                     if float(item["row"].get("total_ev") or 0.0)
                     >= LIVE_FOCUS_MIN_TOTAL_BB]
    if material_live:
        item = dict(material_live[0], source="live")
        picked.append(item)
        seen.add(item["row"].get("diagnosis_key") or item["row"].get("spot_leaf"))
    for item in online:
        if len(picked) >= focus_k:
            break
        key = item["row"].get("diagnosis_key") or item["row"].get("spot_leaf")
        if key in seen:
            continue
        picked.append(dict(item, source="online"))
        seen.add(key)
    return picked[:focus_k]


async def build(conn, window_label, prev_focus, min_n=50, top=8,
                since=None, prev_at=None, provision_focus=False,
                focus_exclude=None):
    weekly = [dict(r) for r in await conn.fetch(WEEKLY_SQL)]
    online_spots = await lb.hierarchical_leaderboard(
        conn, min_n=max(25, min_n // 2), top=top, since=since,
        source="online")
    # Live capture is sparse and some older imported decisions predate the
    # parent-family backfill. Rank exact action lines so every graded hand from
    # this week remains eligible; the material-loss floor below keeps tiny
    # solver noise from taking the reserved live focus seat.
    live_spots = await lb.leaderboard(
        conn, min_n=1, top=top, since=since, source="live")
    blocked = set(focus_exclude or ())
    eligible_online = [it for it in online_spots
                       if (it["row"].get("diagnosis_key")
                           or it["row"].get("spot_leaf")) not in blocked]
    spots = weekly_focus_candidates(eligible_online, live_spots)
    top_hands = [dict(r) for r in await conn.fetch(
        top_hands_sql(since), *([since] if since else []))]
    honesty = await _honesty(conn)
    readback = None
    if prev_focus and prev_at:
        readback = prev_focus_readback(prev_focus, await fetch_readback(conn, prev_focus, prev_at))
    data = compute_training_plan(window_label, weekly, spots, top_hands,
                                 readback, honesty)
    # The rest of the public leak board remains online-only.  Live is a
    # selectively recorded sample and must not be mixed into aggregate ranks.
    data["leaderboard"] = [dict(it["row"], drill_url=it["url"],
                                restrict=it.get("restrict"))
                           for it in online_spots]
    data["focus_excluded"] = sorted(focus_exclude or ())
    focus_queue_ids = (await bind_focus_queue_items(conn, data["focus"])
                       if provision_focus else [])
    # Focus is a diagnosis input, not a second user-facing list. Its queue rows
    # compete in the same five-seat slate as reviews and existing drills.
    slate = await fetch_drill_queue(conn, since=since)
    data["drill_queue"] = slate["picked"]
    selected_ids = {int(q["id"]) for q in slate["picked"]}
    if provision_focus:
        data["focus"] = [f for f in data["focus"]
                         if int(f.get("queue_id") or -1) in selected_ids]
        focus_queue_ids = [qid for qid in focus_queue_ids if qid in selected_ids]
    data["focus_queue_ids"] = focus_queue_ids
    data["queue_backlog_total"] = slate["backlog_total"]
    return data


async def _autoclose(conn) -> dict:
    """Close drills whose bound GTOW attempt met its targets (fail-soft).

    Needs the owner's GTOW session; without a token there is simply nothing to
    read, and the weekly plan must still go out.
    """
    from plan_scheduler import autoclose_passed_drills
    try:
        from gto_owner_token import resolve_owner_db_token
        from gtow_drill_service import GTOWDrillClient
        owner = resolve_owner_db_token()
        if not owner:
            return {"skipped": "no owner GTOW refresh token"}
        client = GTOWDrillClient(owner[0], owner[1])
    except Exception as exc:  # noqa: BLE001 - never block the weekly plan
        return {"skipped": f"no GTOW session ({exc})"}
    return await autoclose_passed_drills(conn, client)


async def _focus_exclusions(conn, since) -> set[str]:
    """Diagnosis keys held back from the focus slot by the cooldown."""
    from plan_scheduler import focus_exclusions
    extra = [since] if since else []
    global_avg = await conn.fetchval(lb.global_avg_sql(since), *extra)
    return await focus_exclusions(
        conn, await focus_history(conn),
        now=datetime.now(timezone.utc),
        global_per100=float(global_avg or 0.0) * 100)


async def _run(mode: str, min_n: int):
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    outdir = ROOT / "data" / "scorecards"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        await ensure_training_ready(conn)
        if mode == "preview":
            data = await build(conn, "歷史全部資料（預覽）", None, min_n=min_n,
                               focus_exclude=await _focus_exclusions(conn, None))
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
        # Retire drills whose bound GTOW attempt already met both targets, so a
        # passed drill cannot take a seat from work that still needs doing.
        # Runs before the scan: a re-scan would otherwise merge fresh evidence
        # into a row that is about to be closed.
        print(f"gtow auto-close: {await _autoclose(conn)}")
        # Keep the durable queue fully refreshed, then let build() show only
        # evidence played in this ISO week. Older unfinished work remains in
        # /queue but no longer occupies the new weekly plan.
        from queue_feed import scan_online
        scan = await scan_online(conn)
        print(f"queue scan: {len(scan['drill'])} drill + {len(scan['review'])} "
              f"review candidates, tally={scan['tally']}")
        week_start_tpe = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start_tpe -= timedelta(days=now.weekday())
        since = week_start_tpe.astimezone(timezone.utc)
        excluded = await _focus_exclusions(conn, since)
        if excluded:
            print(f"focus cooldown holds back: {sorted(excluded)}")
        data = await build(conn, week, prev_focus, min_n=min_n, since=since,
                           prev_at=prev_at, provision_focus=True,
                           focus_exclude=excluded)
        html = render_html(data)
        (outdir / f"{week}.html").write_text(html)
        fam_payload = [{"spot_leaf": f["spot_leaf"],
                        "diagnosis_key": f.get("diagnosis_key"),
                        "diagnosis_level": f.get("diagnosis_level"),
                        "source": f.get("source", "online"),
                        "desc": f.get("desc"),
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
        dq_ids = list(dict.fromkeys(
            list(data.get("focus_queue_ids") or [])
            + [q["id"] for q in (data.get("drill_queue") or [])]))
        if dq_ids:
            # surfaced in this week's plan -> prescribed (still visible in /queue
            # until the player marks them cleared). prescribed_week keeps the
            # FIRST prescription week; surfaced_count/last_surfaced_at track the
            # repeats that the next slate rotates on.
            await conn.execute(
                "UPDATE drill_queue SET status='prescribed', prescribed_week=$1 "
                "WHERE id = ANY($2) AND status='pending'", week, dq_ids)
            from plan_scheduler import mark_surfaced
            await mark_surfaced(conn, dq_ids, week)
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
