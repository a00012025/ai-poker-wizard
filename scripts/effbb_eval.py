#!/usr/bin/env python3
"""Evaluate _compute_effective_bb over the input cache in seconds.

Replays cached inputs through the CURRENT _compute_effective_bb, scores at the
solver depth bucket level vs HH ground truth, splits hero-active vs hero-folded,
and prints the fault breakdown + a precision/coverage curve over confidence.

Usage: python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
"""
import argparse
import json


from effbb_metrics import bucket_match, hero_folded_preflop, classify_fault, depth_bucket
from ocr.n8_parser import _compute_effective_bb


def recompute(inp):
    """Replay one cached input tuple through _compute_effective_bb.
    Tolerates both the 2-tuple (legacy) and 3-tuple (rewritten) returns.

    NOTE: num_players is NOT passed — the function infers physical table size
    from seat geometry, exactly as it must in production (no GT leak)."""
    res = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"],
    )
    if isinstance(res, tuple) and len(res) == 3:
        return res                      # (eff, hero_start, confidence)
    eff, hero_start = res
    return eff, hero_start, 1.0


def _per_node_eval(cache_path, gt_path):
    """D1 per-decision-node depth accuracy (Phase A metric).

    For every hero-active hand whose HH ground truth carries node_effective_bb,
    compare two depth-assignment strategies against the exact-HH per-node depth:

      * SINGLE  — the legacy single hand-wide depth (nearest_depth(effective_bb))
                  applied to EVERY hero decision node (what the old global
                  override effectively did);
      * PERNODE — node_depth.resolve_preflop_nodes (the D1 resolver) run on the
                  GT play fields.

    Reports open-node and facing-node bucket precision for each strategy, so the
    open-node win (the user's "30bb open vs 17bb jam" complaint) is measured.
    """
    from gto_api import nearest_depth
    from node_depth import resolve_preflop_nodes
    from analyze_hand import _get_position_order

    gt_by_id = {}
    for line in open(gt_path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        gt_by_id[d["hand_id"]] = d.get("ground_truth") or {}

    # restrict to the same hero-active population the scalar metric uses
    active_ids = set()
    for line in open(cache_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        g = r.get("gt") or {}
        if g.get("effective_bb") and hero_folded_preflop(g) is False:
            active_ids.add(r["hand_id"])

    stats = {k: [0, 0] for k in ("single_open", "pernode_open",
                                 "single_facing", "pernode_facing")}
    hands = 0
    for hid in active_ids:
        gt = gt_by_id.get(hid) or {}
        nodes_gt = gt.get("node_effective_bb") or []
        if len(nodes_gt) < 1:
            continue
        eff = gt.get("effective_bb")
        stacks_bb = gt.get("stacks_bb") or []
        hero_pos = gt.get("hero_position")
        npl = gt.get("num_players")
        if not (eff and stacks_bb and hero_pos and npl):
            continue
        order = _get_position_order(npl)
        if hero_pos not in order or len(stacks_bb) != len(order):
            continue
        single_bucket = int(nearest_depth(eff))
        hero_start = stacks_bb[order.index(hero_pos)]
        stacks = {order[i]: stacks_bb[i] for i in range(len(order)) if stacks_bb[i]}
        pred = resolve_preflop_nodes(
            preflop_actions=gt.get("preflop_actions", ""),
            hero_position=hero_pos, position_order=order,
            hero_start=hero_start, stacks=stacks, is_icm=False,
        ) or []
        hands += 1
        # align GT nodes to predicted nodes by sequential order
        for idx, gnode in enumerate(nodes_gt):
            kind = "open" if gnode["node"] == "open" else "facing"
            gbucket = gnode["depth"]
            # SINGLE strategy: same hand-wide bucket for every node
            stats[f"single_{kind}"][1] += 1
            if single_bucket == gbucket:
                stats[f"single_{kind}"][0] += 1
            # PERNODE strategy: matched predicted node (by order)
            if idx < len(pred):
                stats[f"pernode_{kind}"][1] += 1
                if pred[idx]["depth_bucket"] == gbucket:
                    stats[f"pernode_{kind}"][0] += 1

    def pct(k):
        ok, n = stats[k]
        return f"{(100*ok/n if n else 0):.1f}% ({ok}/{n})"

    print(f"\n=== PER-NODE DEPTH ACCURACY (hero-active, GT node_effective_bb) ===")
    print(f"hands scored: {hands}")
    print(f"  OPEN node   — single-depth: {pct('single_open')}   "
          f"per-node: {pct('pernode_open')}")
    print(f"  FACING node — single-depth: {pct('single_facing')}   "
          f"per-node: {pct('pernode_facing')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--per-node", action="store_true",
                    help="D1 per-decision-node depth accuracy (needs rebuilt GT)")
    ap.add_argument("--gt",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl",
                    help="ground truth jsonl (must carry node_effective_bb)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="emit only when confidence >= this (precision/coverage knob)")
    args = ap.parse_args()

    if args.per_node:
        return _per_node_eval(args.cache, args.gt)

    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]
    active, folded = [], []
    for r in rows:
        gt = r.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in r:
            continue
        p_eff, hero_start, conf = recompute(r["inputs"])
        if conf < args.min_conf:
            p_eff = None
        rec = {"hid": r["hand_id"], "gt_eff": ge, "p_eff": p_eff,
               "hero_start": hero_start, "conf": conf,
               "gt_max": max(gt.get("stacks_bb") or [0]) or None}
        hf = hero_folded_preflop(gt)
        # CLEANUP: route by hero-folded status; skip when unknown (hf is None).
        if hf is True:
            folded.append(rec)
        elif hf is False:
            active.append(rec)

    def score(name, subset):
        emitted = [x for x in subset if x["p_eff"] is not None]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(subset) if subset else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"\n## {name}: n={len(subset)} emitted={len(emitted)} "
              f"coverage={cov:.1f}% bucket-precision={prec:.2f}% "
              f"({len(ok)}/{len(emitted)})")
        wrong = [x for x in emitted if not bucket_match(x["p_eff"], x["gt_eff"])]
        faults = {}
        for x in wrong:
            f = classify_fault(p_eff=x["p_eff"], gt_eff=x["gt_eff"],
                               hero_start=x["hero_start"] or x["gt_eff"],
                               gt_max=x["gt_max"])
            faults[f] = faults.get(f, 0) + 1
        if wrong:
            print("   faults:", faults)
        return prec, cov

    score("HERO ACTIVE (target population)", active)
    score("HERO FOLDED (context only)", folded)

    print("\n--- precision/coverage curve (hero-active, by confidence floor) ---")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        emitted = [x for x in active if x["p_eff"] is not None and x["conf"] >= thr]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(active) if active else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"  conf>={thr:.1f}: coverage={cov:5.1f}%  precision={prec:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
