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
from gtow_trainer_url import (build_drill_url, CAT_POSITIONS, SpotNotSupportedError,
                              MTT_DEPTHS, DEPTH_BAND_DEPTHS)

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
PREFLOP_CATS = {"RFI", "vsOpen", "vsRaiseCall", "vsSqueeze", "vs3bet", "vsCold3bet",
                "vs4bet", "vsCold4bet"}

# All three queries take an optional time window (§2.2 歸因): an unwindowed
# leaderboard is a cumulative average — recent improvement/regression barely
# moves it, so focus picking and readback would be blind to change.
def leader_sql(since=None) -> str:
    win = " AND played_at >= $3" if since else ""
    return f"""
SELECT spot_leaf, spot_category,
       count(*) n, sum(ev_loss_bb) total_ev, avg(ev_loss_bb) avg_ev,
       mode() WITHIN GROUP (ORDER BY hero_cat)   hero_cat,
       mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
       mode() WITHIN GROUP (ORDER BY ip_oop)     ip_oop,
       mode() WITHIN GROUP (ORDER BY position)   hero_pos,
       mode() WITHIN GROUP (ORDER BY depth_band) depth_band,
       mode() WITHIN GROUP (ORDER BY street)     street
FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL AND source='online'{win}
GROUP BY spot_leaf, spot_category
HAVING count(*) >= $1
ORDER BY avg(ev_loss_bb) DESC
LIMIT $2
"""


def sample_sql(since=None) -> str:
    win = " AND d.played_at >= $2" if since else ""
    return f"""
SELECT d.gtow_hand_id, h.played_at, h.hero_hand, h.position, h.boards, h.total_ev_loss_bb,
       d.ev_loss_bb, d.correctness
FROM ledger_decisions d JOIN ledger_hands h ON h.gtow_hand_id = d.gtow_hand_id
WHERE d.spot_leaf = $1 AND d.ev_loss_bb > 0 AND NOT d.excluded AND d.source='online'{win}
ORDER BY d.ev_loss_bb DESC LIMIT 2
"""


def band_sql(since=None) -> str:
    win = " AND played_at >= $2" if since else ""
    return f"""
SELECT eff_stack, count(*) n, avg(ev_loss_bb) avg_ev
FROM ledger_decisions
WHERE spot_leaf=$1 AND NOT excluded AND NOT discarded AND eff_stack IS NOT NULL
  AND source='online'{win}
GROUP BY eff_stack
"""


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
    cat = r["spot_category"]
    parts = r["spot_leaf"].split(":")
    try:
        if cat in PREFLOP_CATS:
            if cat in ("RFI", "vsOpen"):
                hero = [r["hero_pos"]] if r["hero_pos"] else CAT_POSITIONS.get(r["hero_cat"], [])
            else:
                hero = CAT_POSITIONS.get(r["hero_cat"], [])
            vc = r["villain_cat"]
            opp = CAT_POSITIONS.get(vc) if vc in CAT_POSITIONS else None
            return build_drill_url(cat, "preflop", 20, hero, opponent_positions=opp,
                                   rel_position=r["ip_oop"], depths=depths)
        pot_type = parts[1] if len(parts) > 1 else None
        hero = CAT_POSITIONS.get(r["hero_cat"], [])
        vc = r["villain_cat"]
        opp = CAT_POSITIONS.get(vc) if vc in CAT_POSITIONS else None
        return build_drill_url(cat, cat, 20, hero, opponent_positions=opp,
                               rel_position=r["ip_oop"], pot_type=pot_type, depths=depths)
    except (SpotNotSupportedError, ValueError):
        return None


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
        out.append({"row": r, "url": _drill_url(r, depths), "samples": samples,
                    "bands": bands, "restrict": restrict})
    return out


def _render(items, min_n) -> str:
    L = [f"# Action-line spot 排行榜：avg EV loss 最高（n≥{min_n}）", ""]
    L.append("每個 spot = 一種決策情境（行動線）。avg = 該情境每個決策的平均 EV loss（bb），"
             "含打對的 0 損失手，所以是「這個情境整體多貴」。")
    L.append("")
    for i, it in enumerate(items, 1):
        r = it["row"]
        L.append(f"## {i}. `{r['spot_leaf']}`")
        L.append(f"- **avg EV loss = {r['avg_ev']*100:.2f} bb/100 決策**"
                 f"（每決策 {r['avg_ev']:.3f}bb）· n={r['n']} · 總損失 {r['total_ev']:.1f}bb")
        L.append(f"- 情境：{r['spot_category']} · hero_cat={r['hero_cat']} · "
                 f"villain_cat={r['villain_cat']} · {r['ip_oop'] or '-'}")
        # stack-band breakdown
        bands = sorted(it["bands"], key=lambda b: -b["avg_ev"])
        bd = " · ".join(f"{BAND_ZH.get(b['eff_stack'], b['eff_stack'])} "
                        f"{b['avg_ev']*100:.1f}bb/100 (n={b['n']})" for b in bands)
        L.append(f"- stack 細分：{bd}")
        if it["restrict"]:
            L.append(f"- ⛏ 此 spot 的 EV loss 明顯集中在 **{BAND_ZH.get(it['restrict'])}** → "
                     f"drill 只勾這個 band（含該 band 全部深度）")
        else:
            L.append("- drill **不鎖深度**（涵蓋全部 stack depth，多元訓練）")
        if it["url"]:
            L.append(f"- 🎯 **GTOW Trainer drill**：{it['url']}")
        else:
            L.append("- ⚠️ 無對應 GTOW drill 捷徑（cold-3bet/4bet 或 postflop 特殊）")
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
