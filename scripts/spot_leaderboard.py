#!/usr/bin/env python3
"""Action-line spot leaderboard: rank decision spots by avg EV loss (n>=min_n),
each with a precise GTOW Trainer drill link + sample hands. Pure query layer.

CLI: python scripts/spot_leaderboard.py [--min-n 50] [--top 5]
Emits data/scorecards/spot_leaderboard.md and prints it.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
from gtow_trainer_url import build_drill_url, CAT_POSITIONS, SpotNotSupportedError
from scorecard import analyze_table_url

BAND_MID = {"le15": 12, "15_25": 20, "25_40": 32, "40plus": 50}
PREFLOP_CATS = {"RFI", "vsOpen", "vsRaiseCall", "vsSqueeze", "vs3bet", "vsCold3bet",
                "vs4bet", "vsCold4bet"}

LEADER_SQL = """
SELECT spot_leaf, spot_category,
       count(*) n, sum(ev_loss_bb) total_ev, avg(ev_loss_bb) avg_ev,
       mode() WITHIN GROUP (ORDER BY hero_cat)   hero_cat,
       mode() WITHIN GROUP (ORDER BY villain_cat) villain_cat,
       mode() WITHIN GROUP (ORDER BY ip_oop)     ip_oop,
       mode() WITHIN GROUP (ORDER BY position)   hero_pos,
       mode() WITHIN GROUP (ORDER BY depth_band) depth_band,
       mode() WITHIN GROUP (ORDER BY street)     street
FROM ledger_decisions
WHERE NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL
GROUP BY spot_leaf, spot_category
HAVING count(*) >= $1
ORDER BY avg(ev_loss_bb) DESC
LIMIT $2
"""

SAMPLE_SQL = """
SELECT d.gtow_hand_id, h.played_at, h.hero_hand, h.position, h.boards, h.total_ev_loss_bb,
       d.ev_loss_bb, d.correctness
FROM ledger_decisions d JOIN ledger_hands h ON h.gtow_hand_id = d.gtow_hand_id
WHERE d.spot_leaf = $1 AND d.ev_loss_bb > 0 AND NOT d.excluded
ORDER BY d.ev_loss_bb DESC LIMIT 2
"""


def _drill_url(r) -> str | None:
    cat = r["spot_category"]
    depth = BAND_MID.get(r["depth_band"], 20)
    parts = r["spot_leaf"].split(":")
    try:
        if cat in PREFLOP_CATS:
            if cat in ("RFI", "vsOpen"):
                hero = [r["hero_pos"]] if r["hero_pos"] else CAT_POSITIONS.get(r["hero_cat"], [])
            else:
                hero = CAT_POSITIONS.get(r["hero_cat"], [])
            vc = r["villain_cat"]
            opp = CAT_POSITIONS.get(vc) if vc in CAT_POSITIONS else None
            return build_drill_url(cat, "preflop", depth, hero,
                                   opponent_positions=opp, rel_position=r["ip_oop"])
        # postflop: leaf = street:pot_type:heroVvillain:ip:...
        pot_type = parts[1] if len(parts) > 1 else None
        hero = CAT_POSITIONS.get(r["hero_cat"], [])
        vc = r["villain_cat"]
        opp = CAT_POSITIONS.get(vc) if vc in CAT_POSITIONS else None
        return build_drill_url(cat, cat, depth, hero, opponent_positions=opp,
                               rel_position=r["ip_oop"], pot_type=pot_type)
    except (SpotNotSupportedError, ValueError):
        return None


async def leaderboard(conn, min_n=50, top=5):
    rows = await conn.fetch(LEADER_SQL, min_n, top)
    out = []
    for r in rows:
        samples = await conn.fetch(SAMPLE_SQL, r["spot_leaf"])
        out.append({"row": r, "url": _drill_url(r), "samples": samples})
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
                 f"villain_cat={r['villain_cat']} · {r['ip_oop'] or '-'} · 深度帶 {r['depth_band']}")
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
