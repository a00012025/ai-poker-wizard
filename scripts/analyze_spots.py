#!/usr/bin/env python3
"""Classify every hero decision in the archived raw into the action-line
taxonomy and emit a spot-count / EV-loss statistics report (md + html).

Reads only the local raw archive (data/gtow_raw) — no API, no DB writes.
This is the validation artifact for the taxonomy before any DB migration.
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from spot_taxonomy import walk_spots

RAW = ROOT / "data" / "gtow_raw"
OUT = ROOT / "data" / "scorecards"


def load_list_index() -> dict:
    idx = {}
    for f in glob.glob(str(RAW / "list" / "*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                idx[r["hand_id"]] = r
    return idx


def _bump(agg, key, ev, inc):
    a = agg[key]
    a["n"] += 1
    a["ev"] += ev
    if inc:
        a["n_inc"] += 1
        a["ev_inc"] += ev


def main():
    list_idx = load_list_index()
    print(f"list rows indexed: {len(list_idx)}")

    tree = defaultdict(lambda: {"n": 0, "ev": 0.0, "n_inc": 0, "ev_inc": 0.0})
    leaf = defaultdict(lambda: {"n": 0, "ev": 0.0, "n_inc": 0, "ev_inc": 0.0})
    cat = defaultdict(lambda: {"n": 0, "ev": 0.0, "n_inc": 0, "ev_inc": 0.0})
    other = Counter()
    other_ev = defaultdict(float)
    discarded = Counter()
    tag_dist = {k: Counter() for k in ("eff_stack", "board_suit", "board_conn", "board_paired")}
    n_spots = n_hands = n_excluded = n_missing_list = n_discarded = n_limp_origin = 0

    files = glob.glob(str(RAW / "detail" / "*" / "*.json.gz"))
    print(f"detail files: {len(files)}")
    for i, f in enumerate(files):
        hid = Path(f).stem.replace(".json", "")
        lr = list_idx.get(hid)
        if lr is None:
            n_missing_list += 1
            continue
        with gzip.open(f, "rt") as fh:
            det = json.load(fh)
        n_hands += 1
        for s in walk_spots(lr, det):
            n_spots += 1
            if s.get("discarded"):
                n_discarded += 1
                discarded[s.get("note") or s["leaf"]] += 1
                continue
            if s.get("limp_origin"):
                n_limp_origin += 1
            ev, inc = s["ev_loss_bb"], not s["excluded"]
            if not inc:
                n_excluded += 1
            _bump(cat, s["category"], ev, inc)
            _bump(leaf, s["leaf"], ev, inc)
            for k in s["keys"]:
                _bump(tree, k, ev, inc)
            if s["category"] == "other" or str(s["leaf"]).startswith("other:"):
                other[s.get("note") or s["leaf"]] += 1
                other_ev[s.get("note") or s["leaf"]] += ev
            for tk in tag_dist:
                tag_dist[tk][s["tags"].get(tk)] += 1
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{len(files)} files, {n_spots} spots", flush=True)

    report = _render_md(n_hands, n_spots, n_excluded, n_missing_list, n_discarded,
                        n_limp_origin, cat, tree, leaf, other, other_ev, discarded, tag_dist)
    (OUT / "spot_stats.md").write_text(report)
    (OUT / "spot_stats.html").write_text(_render_html(report))
    print(report[:2000])
    print(f"\n[full report] {OUT/'spot_stats.md'}  +  {OUT/'spot_stats.html'}")


PREFLOP_CATS = ["RFI", "vsOpen", "vsRaiseCall", "vsSqueeze", "vs3bet", "vsCold3bet",
                "vs4bet", "vsCold4bet"]


def _row(name, a):
    return (f"| {name} | {a['n']} | {a['n_inc']} | {a['ev_inc']:.1f} | "
            f"{(a['ev_inc']/a['n_inc']*100 if a['n_inc'] else 0):.2f} |")


def _render_md(n_hands, n_spots, n_excluded, n_missing, n_discarded, n_limp_origin,
               cat, tree, leaf, other, other_ev, discarded, tag_dist):
    scored = n_spots - n_discarded
    L = ["# Spot 分類統計（action-line taxonomy v2）", ""]
    L.append(f"- 分析手數：**{n_hands}**（含 detail 的手）")
    L.append(f"- 決策 spot 總數：**{n_spots}**（一手多 spot，故 > 手數）")
    L.append(f"- **捨棄（limp 相關，GTOW 範圍不可靠，不評量）：{n_discarded}**"
             f"（占 {n_discarded/max(n_spots,1)*100:.1f}%）")
    L.append(f"- 進入分類的 spot：**{scored}**")
    L.append(f"- excluded（unsolved/warning，不進 EV 統計但仍分類）：{n_excluded}")
    other_total = sum(other.values())
    L.append(f"- 標為 other（罕見結構，如 5bet+）：{other_total}"
             f"（占 scored {other_total/max(scored,1)*100:.1f}%）")
    L.append(f"- postflop limp/iso pot（保留但帶 limp_origin 注記）：{n_limp_origin}")
    if n_missing:
        L.append(f"- detail 找不到對應 list row（略過）：{n_missing}")
    L.append("")
    if discarded:
        L.append("### 捨棄明細（limp）")
        L.append("| 情況 | spots |")
        L.append("|---|---|")
        for note, n in discarded.most_common():
            L.append(f"| {escape(str(note))} | {n} |")
        L.append("")

    L.append("## 大類分佈（category）")
    L.append("| category | spots | 計入EV | 總EVloss(bb) | bb/100 |")
    L.append("|---|---|---|---|---|")
    for c in PREFLOP_CATS + ["flop", "turn", "river", "other"]:
        if c in cat:
            L.append(_row(c, cat[c]))
    L.append("")

    L.append("## Preflop 樹狀（category → L1 → L2，計入EV 排序）")
    # RFI by exact position
    L.append("### RFI（依 hero 精確位置）")
    L.append("| line | spots | 計入EV | 總EVloss | bb/100 |")
    L.append("|---|---|---|---|---|")
    for k in sorted([k for k in tree if k.endswith("_RFI")], key=lambda k: -tree[k]["ev_inc"]):
        L.append(_row(k, tree[k]))
    L.append("")
    for topcat in ["vsOpen", "vsRaiseCall", "vsSqueeze", "vs3bet", "vsCold3bet",
                   "vs4bet", "vsCold4bet"]:
        l1s = sorted([k for k in tree if k.endswith("_" + topcat) or ("_" + topcat + "_") in k],
                     key=lambda k: -tree[k]["ev_inc"])
        l1_only = [k for k in l1s if k.endswith("_" + topcat)]
        if not l1_only and topcat not in cat:
            continue
        L.append(f"### {topcat}（整體 {cat.get(topcat,{}).get('ev_inc',0):.1f}bb / "
                 f"{cat.get(topcat,{}).get('n_inc',0)} spots）")
        L.append("| line | spots | 計入EV | 總EVloss | bb/100 |")
        L.append("|---|---|---|---|---|")
        for k in l1s:
            indent = "&nbsp;&nbsp;↳ " if k.count("_") >= 2 and not k.endswith("_" + topcat) else ""
            L.append(_row(indent + k, tree[k]))
        L.append("")

    L.append("## Postflop 行動線（依 street → pot_type → facing；leaf 見下方 top 榜）")
    for st in ("flop", "turn", "river"):
        L.append(f"### {st}")
        L.append("| line | spots | 計入EV | 總EVloss | bb/100 |")
        L.append("|---|---|---|---|---|")
        ks = sorted([k for k in tree if k == st or k.startswith(st + ":")],
                    key=lambda k: (k.count(":"), -tree[k]["ev_inc"]))
        for k in ks:
            if k.count(":") <= 2:      # street / street:pot / street:pot:facing rollups
                L.append(_row(k, tree[k]))
        L.append("")

    L.append("## Top 30 漏 EV 的 leaf spot（計入EV、n≥20 排序）")
    L.append("| leaf line | spots | 計入EV | 總EVloss(bb) | bb/100 |")
    L.append("|---|---|---|---|---|")
    ranked = sorted([k for k in leaf if leaf[k]["n_inc"] >= 20],
                    key=lambda k: -leaf[k]["ev_inc"])[:30]
    for k in ranked:
        L.append(_row(k, leaf[k]))
    L.append("")

    L.append("## 無法分類 / other 明細（誠實層）")
    L.append("| 情況 | spots | 總EVloss(bb) |")
    L.append("|---|---|---|")
    for note, n in other.most_common():
        L.append(f"| {escape(str(note))} | {n} | {other_ev[note]:.1f} |")
    L.append("")

    L.append("## 標籤分佈")
    for tk, ctr in tag_dist.items():
        parts = ", ".join(f"{k}={v}" for k, v in ctr.most_common())
        L.append(f"- **{tk}**: {parts}")
    L.append("")
    return "\n".join(L)


def _render_html(md: str) -> str:
    # minimal md->html (headings, tables, lists) — self-contained
    import re
    lines = md.splitlines()
    html, in_tbl = [], False
    for ln in lines:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_tbl:
                html.append("<table>"); in_tbl = True
                html.append("<tr>" + "".join(f"<th>{escape(c)}</th>" for c in cells) + "</tr>")
            else:
                html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            html.append("</table>"); in_tbl = False
        if ln.startswith("### "):
            html.append(f"<h3>{escape(ln[4:])}</h3>")
        elif ln.startswith("## "):
            html.append(f"<h2>{escape(ln[3:])}</h2>")
        elif ln.startswith("# "):
            html.append(f"<h1>{escape(ln[2:])}</h1>")
        elif ln.startswith("- "):
            html.append(f"<div>{ln[2:]}</div>")
        elif ln.strip():
            html.append(f"<p>{ln}</p>")
    if in_tbl:
        html.append("</table>")
    body = "\n".join(html)
    return (f"<!doctype html><meta charset='utf-8'><title>Spot 分類統計</title>"
            f"<style>body{{font-family:-apple-system,'PingFang TC',sans-serif;max-width:900px;"
            f"margin:0 auto;padding:24px;font-size:14px}}table{{border-collapse:collapse;width:100%;"
            f"margin:8px 0}}th,td{{border:1px solid #e2e8f0;padding:3px 8px;text-align:left}}"
            f"th{{background:#f7fafc}}h2{{color:#2b6cb0;margin-top:24px}}h3{{color:#4a5568}}</style>{body}")


if __name__ == "__main__":
    main()
