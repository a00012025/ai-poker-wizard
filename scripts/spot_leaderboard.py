#!/usr/bin/env python3
"""Action-line spot leaderboard: rank decision spots by avg EV loss (n>=min_n),
each with a precise GTOW Trainer drill link + sample hands. Pure query layer.

CLI: python scripts/spot_leaderboard.py [--min-n 50] [--top 5]
Emits data/scorecards/spot_leaderboard.md and prints it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
from gtow_trainer_url import (CAT_POSITIONS, MTT_DEPTHS, DEPTH_BAND_DEPTHS,
                              PREFLOP_CATS, drill_url_for_spot)  # noqa: F401 — PREFLOP_CATS re-exported
from action_bias import LOSSY_MIN_BB, dominant_action_bias

TPE = ZoneInfo("Asia/Taipei")


def analyze_table_url(day_start_taipei: str, day_end_taipei: str) -> str:
    """GTOW Analyze table filtered to a Taipei day range (fallback review link)."""
    start = datetime.fromisoformat(day_start_taipei).replace(tzinfo=TPE)
    end = datetime.fromisoformat(day_end_taipei).replace(tzinfo=TPE) + timedelta(days=1)
    fmt = lambda d: d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filters = json.dumps({"played_at__range": [fmt(start), fmt(end)]})
    return (f"https://app.gtowizard.com/analyze/v4/hands/table"
            f"?filters={quote(filters)}&preselectGamemode=TOURNAMENT")


BAND_MID = {"le15": 12, "15_25": 20, "25_40": 32, "40plus": 50}
BAND_ZH = {"short": "短籌(≤20)", "medium": "中籌(20-50)", "large": "深籌(>50)"}
MIN_TRAINING_CONFIDENCE = 0.8
FAMILY_PRIOR_N = 100

# All three queries take an optional time window (§2.2 歸因): an unwindowed
# leaderboard is a cumulative average — recent improvement/regression barely
# moves it, so focus picking and readback would be blind to change.
def leader_sql(since=None) -> str:
    win = " AND played_at >= $3" if since else ""
    # n_clean/avg_ev_clean re-aggregate WITHOUT the off-tree-approximated rows
    # (sizing_snap / depth_snap_gap) — the honesty layer's soft flags get a
    # consumer: a spot whose avg moves a lot without them is `fragile`
    # (§5.2 敏感度旗標), i.e. its magnitude leans on approximation error.
    return f"""
SELECT spot_leaf, spot_category,
       count(*) n, sum(ev_loss_bb) total_ev, avg(ev_loss_bb) avg_ev,
       count(*) FILTER (WHERE NOT (approx_flags ?| array['sizing_snap','missing_solver_depth','analyzer_approximation'])) n_clean,
       avg(ev_loss_bb) FILTER (WHERE NOT (approx_flags ?| array['sizing_snap','missing_solver_depth','analyzer_approximation'])) avg_ev_clean,
       mode() WITHIN GROUP (ORDER BY hero_cat)   hero_cat,
       mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
       mode() WITHIN GROUP (ORDER BY ip_oop)     ip_oop,
       mode() WITHIN GROUP (ORDER BY position)   hero_pos,
       mode() WITHIN GROUP (ORDER BY depth_band) depth_band,
       mode() WITHIN GROUP (ORDER BY street)     street
FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL AND source='online'
  AND confidence >= {MIN_TRAINING_CONFIDENCE}{win}
GROUP BY spot_leaf, spot_category
HAVING count(*) >= $1
ORDER BY avg(ev_loss_bb) DESC
LIMIT $2
"""


def sample_sql(since=None) -> str:
    win = " AND d.played_at >= $2" if since else ""
    return f"""
SELECT d.id, d.gtow_hand_id, d.street, d.decision_idx, d.spot_category,
       d.spot_leaf, d.hero_cat, d.villain_cat, d.ip_oop, d.position,
       d.pot_type, d.eff_stack, d.ev_loss_bb, d.correctness, d.gametype,
       d.played_depth_bb, d.solver_depth_bb,
       h.played_at, h.hero_hand, h.position hand_position, h.boards,
       h.total_ev_loss_bb, h.source hand_source, h.raw_path, h.parsed_json,
       h.preflop_depth_bb
FROM ledger_decisions d JOIN ledger_hands h ON h.gtow_hand_id = d.gtow_hand_id
WHERE d.spot_leaf = $1 AND d.ev_loss_bb > 0 AND NOT d.excluded AND d.source='online'{win}
  AND d.confidence >= {MIN_TRAINING_CONFIDENCE}
ORDER BY d.ev_loss_bb DESC LIMIT 2
"""


def band_sql(since=None) -> str:
    win = " AND played_at >= $2" if since else ""
    return f"""
SELECT eff_stack, count(*) n, avg(ev_loss_bb) avg_ev
FROM ledger_decisions
WHERE spot_leaf=$1 AND NOT excluded AND NOT discarded AND eff_stack IS NOT NULL
  AND source='online' AND confidence >= {MIN_TRAINING_CONFIDENCE}{win}
GROUP BY eff_stack
"""


def family_sql(since=None) -> str:
    """Stable parent-family diagnosis with one exact representative leaf.

    Parent aggregation recovers repeated skill errors split across exact action
    lines.  The representative is the member leaf with the greatest total EV
    loss, so the family diagnosis remains anchored to a concrete example/drill.
    """
    win = " AND played_at >= $2" if since else ""
    return f"""
WITH base AS (
  SELECT * FROM ledger_decisions
  WHERE NOT excluded AND NOT discarded AND spot_parent IS NOT NULL
    AND spot_leaf IS NOT NULL AND source='online'
    AND confidence >= {MIN_TRAINING_CONFIDENCE}{win}
), parent_stats AS (
  SELECT spot_parent diagnosis_key, spot_category,
         count(*) n, sum(ev_loss_bb) total_ev, avg(ev_loss_bb) avg_ev,
         count(*) FILTER (WHERE NOT (approx_flags ? 'analyzer_approximation')) n_clean,
         avg(ev_loss_bb) FILTER (WHERE NOT (approx_flags ? 'analyzer_approximation')) avg_ev_clean
  FROM base GROUP BY spot_parent, spot_category
  HAVING count(*) >= $1
), leaf_stats AS (
  SELECT spot_parent, spot_leaf representative_leaf, sum(ev_loss_bb) leaf_total_ev,
         count(*) leaf_n,
         mode() WITHIN GROUP (ORDER BY hero_cat) hero_cat,
         mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
         mode() WITHIN GROUP (ORDER BY ip_oop) ip_oop,
         mode() WITHIN GROUP (ORDER BY position) hero_pos,
         mode() WITHIN GROUP (ORDER BY depth_band) depth_band,
         mode() WITHIN GROUP (ORDER BY street) street,
         row_number() OVER (PARTITION BY spot_parent
                            ORDER BY sum(ev_loss_bb) DESC, count(*) DESC, spot_leaf) rn
  FROM base GROUP BY spot_parent, spot_leaf
)
SELECT p.*, l.representative_leaf, l.leaf_total_ev, l.leaf_n,
       l.hero_cat, l.villain_cat, l.ip_oop, l.hero_pos, l.depth_band, l.street
FROM parent_stats p JOIN leaf_stats l ON l.spot_parent=p.diagnosis_key AND l.rn=1
"""


def family_band_sql(since=None) -> str:
    win = " AND played_at >= $2" if since else ""
    return f"""
SELECT eff_stack, count(*) n, avg(ev_loss_bb) avg_ev
FROM ledger_decisions
WHERE spot_parent=$1 AND NOT excluded AND NOT discarded AND eff_stack IS NOT NULL
  AND source='online' AND confidence >= {MIN_TRAINING_CONFIDENCE}{win}
GROUP BY eff_stack
"""


def global_avg_sql(since=None) -> str:
    win = " AND played_at >= $1" if since else ""
    return f"""SELECT avg(ev_loss_bb) FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_parent IS NOT NULL
  AND source='online' AND confidence >= {MIN_TRAINING_CONFIDENCE}{win}"""


def action_bias_sql(level: str = "parent", since=None) -> str:
    """Material lossy decisions used only to explain an EV-ranked spot."""
    column = "spot_parent" if level == "parent" else "spot_leaf"
    win = " AND played_at >= $2" if since else ""
    return f"""SELECT taken_code, best_code, ev_loss_bb
FROM ledger_decisions
WHERE {column}=$1 AND NOT excluded AND NOT discarded AND source='online'
  AND confidence >= {MIN_TRAINING_CONFIDENCE} AND ev_loss_bb >= {LOSSY_MIN_BB}{win}
ORDER BY ev_loss_bb DESC"""


def rank_hierarchical_rows(rows, global_avg: float, prior_n: int = FAMILY_PRIOR_N):
    """Empirical-Bayes partial pooling toward the honest global EV-loss mean."""
    ranked = []
    for raw in rows:
        r = dict(raw)
        n, total = int(r.get("n") or 0), float(r.get("total_ev") or 0)
        r["shrunk_avg_ev"] = ((total + prior_n * float(global_avg or 0)) /
                              (n + prior_n))
        ranked.append(r)
    return sorted(ranked, key=lambda r: (-r["shrunk_avg_ev"], -float(r.get("total_ev") or 0),
                                         str(r.get("diagnosis_key") or "")))


def choose_depths(bands) -> tuple[str | None, list[int]]:
    """Default = all MTT depths. Restrict to one stack band ONLY when that band
    (n>=25) has avg EV loss >= 1.5x the next-highest band (n>=25) — i.e. the
    leak clearly concentrates in one stack depth."""
    sig = sorted([b for b in bands if b["n"] >= 25 and b["eff_stack"] in DEPTH_BAND_DEPTHS],
                 key=lambda b: -b["avg_ev"])
    if len(sig) >= 2 and sig[0]["avg_ev"] >= 1.5 * max(sig[1]["avg_ev"], 1e-9):
        return sig[0]["eff_stack"], DEPTH_BAND_DEPTHS[sig[0]["eff_stack"]]
    return None, list(MTT_DEPTHS)


def _drill_url(r, depths) -> str | None:
    if "flat_vsSqueeze" in str(r.get("spot_leaf") or ""):
        return None
    parts = r["spot_leaf"].split(":")
    return drill_url_for_spot(
        r["spot_category"], hero_pos=r.get("hero_pos"), hero_cat=r.get("hero_cat"),
        villain_cat=r.get("villain_cat"), ip_oop=r.get("ip_oop"),
        pot_type=parts[1] if len(parts) > 1 else None, depths=depths)


def drill_url_for_item(row: dict, depths: list[int], samples,
                       exact_builder=None) -> str | None:
    """Build the safest available Trainer link for one ranked item.

    Preflop families use the verified multi-depth shortcuts.  Postflop has no
    faithful shortcut, so use the highest-loss source hand from the exact
    representative action line to build a GTOW Custom Trainer URL.  Failure is
    still honest ``None``; never substitute a nearby but different spot.
    """
    url = _drill_url(row, depths)
    if url or not samples:
        return url
    if exact_builder is None:
        from queue_feed import queue_drill_url_for_decision
        exact_builder = queue_drill_url_for_decision
    try:
        return exact_builder(dict(samples[0]))
    except Exception:
        return None


def is_fragile(row: dict, rel_threshold: float = 0.30, min_clean_n: int = 10) -> bool:
    """§5.2 sensitivity flag: the spot's avg EV loss moves >30% once the
    off-tree-approximated samples (sizing_snap/depth_snap_gap) are removed —
    its magnitude leans on approximation error; read it conservatively."""
    n_clean = row.get("n_clean") or 0
    avg_clean, avg_all = row.get("avg_ev_clean"), row.get("avg_ev")
    if n_clean < min_clean_n or avg_clean is None or not avg_all or avg_all <= 1e-9:
        return False
    return abs(float(avg_clean) - float(avg_all)) / float(avg_all) > rel_threshold


async def leaderboard(conn, min_n=50, top=5, since=None):
    """since=None → all-history (CLI/preview); a datetime restricts every
    aggregate (ranking, bands, samples) to that window."""
    extra = [since] if since else []
    rows = await conn.fetch(leader_sql(since), min_n, top, *extra)
    out = []
    for r in rows:
        bands = [dict(b) for b in await conn.fetch(band_sql(since), r["spot_leaf"], *extra)]
        restrict, depths = choose_depths(bands)
        samples = await conn.fetch(sample_sql(since), r["spot_leaf"], *extra)
        row = dict(r)
        bias_rows = await conn.fetch(action_bias_sql("leaf", since), row["spot_leaf"], *extra)
        row["action_bias"] = dominant_action_bias(bias_rows)
        out.append({"row": row, "url": drill_url_for_item(row, depths, samples), "samples": samples,
                    "bands": bands, "restrict": restrict, "fragile": is_fragile(row)})
    return out


async def hierarchical_leaderboard(conn, min_n=25, top=5, since=None):
    """Parent-family focus ranking + exact representative evidence."""
    extra = [since] if since else []
    rows = await conn.fetch(family_sql(since), min_n, *extra)
    global_avg = await conn.fetchval(global_avg_sql(since), *extra)
    ranked = rank_hierarchical_rows(rows, float(global_avg or 0))[:top]
    out = []
    for row in ranked:
        key = row["diagnosis_key"]
        bias_rows = await conn.fetch(action_bias_sql("parent", since), key, *extra)
        row["action_bias"] = dominant_action_bias(bias_rows)
        family_bands = [dict(b) for b in await conn.fetch(
            family_band_sql(since), key, *extra)]
        # Prescription truth stays on the exact child being opened.  Family
        # bands diagnose the broader skill; they must never constrain a
        # representative leaf whose depth distribution may differ.
        prescription_bands = [dict(b) for b in await conn.fetch(
            band_sql(since), row["representative_leaf"], *extra)]
        restrict, depths = choose_depths(prescription_bands)
        samples = await conn.fetch(
            sample_sql(since), row["representative_leaf"], *extra)
        row["spot_leaf"] = row["representative_leaf"]
        row["diagnosis_level"] = "parent"
        out.append({"row": row, "url": drill_url_for_item(row, depths, samples), "samples": samples,
                    "bands": family_bands, "prescription_bands": prescription_bands,
                    "restrict": restrict, "fragile": is_fragile(row)})
    return out


def _render(items, min_n) -> str:
    L = [f"# 最燒錢情境排行（至少 {min_n} 個決策）", ""]
    L.append("每一列是一種牌局情境。漏損包含打對時的 0，所以代表你每遇到 100 次這個情境，"
             "平均會少贏多少 bb。")
    L.append("")
    for i, it in enumerate(items, 1):
        r = it["row"]
        L.append(f"## {i}. `{r['spot_leaf']}`")
        L.append(f"- **實戰漏損 = {r['avg_ev']*100:.2f} bb/100 決策**"
                 f"（每次平均 {r['avg_ev']:.3f}bb）· 樣本 {r['n']} 個決策 · 總損失 {r['total_ev']:.1f}bb")
        L.append(f"- 情境：{r['spot_category']} · hero_cat={r['hero_cat']} · "
                 f"villain_cat={r['villain_cat']} · {r['ip_oop'] or '-'}")
        if it.get("fragile"):
            ac = r.get("avg_ev_clean")
            L.append(f"- ⚠️ 部分下注尺寸不在 GTOW 標準樹上；排除後漏損變為 "
                     f"{(ac or 0)*100:.2f} bb/100（樣本 {r.get('n_clean')}），先不要過度解讀")
        # stack-band breakdown
        bands = sorted(it["bands"], key=lambda b: -b["avg_ev"])
        bd = " · ".join(f"{BAND_ZH.get(b['eff_stack'], b['eff_stack'])} "
                        f"{b['avg_ev']*100:.1f}bb/100 (n={b['n']})" for b in bands)
        L.append(f"- 不同籌碼深度：{bd}")
        if it["restrict"]:
            L.append(f"- ⛏ 此 spot 的 EV loss 明顯集中在 **{BAND_ZH.get(it['restrict'])}** → "
                     f"練習只選這個籌碼帶")
        else:
            L.append("- 練習**不鎖籌碼深度**，讓你遇到不同 stack 都能作答")
        if it["url"]:
            L.append(f"- 🎯 **GTOW Trainer 練習**：{it['url']}")
        else:
            L.append("- ⚠️ 目前無法建立準確的 GTOW Trainer 連結，請從樣本手複習")
        if it["samples"]:
            L.append("- 樣本手（點 Analyze 抽查）：")
            for s in it["samples"]:
                day = s["played_at"].strftime("%Y-%m-%d")
                L.append(f"    - {day} · {s['hero_hand']} {s['position']} · {s['boards'] or '-'} · "
                         f"該決策損失 {s['ev_loss_bb']:.2f}bb（{s['correctness']}）· "
                         f"[Analyze]({analyze_table_url(day, day)})")
        L.append("")
    return "\n".join(L)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        items = await leaderboard(conn, a.min_n, a.top)
        report = _render(items, a.min_n)
        (ROOT / "data/scorecards/spot_leaderboard.md").write_text(report)
        print(report)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
