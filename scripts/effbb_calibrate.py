#!/usr/bin/env python3
"""Phase 4 — calibrated abstain for effective_bb.

Over the hero-active cache hands, extract the per-hand abstain-signal features
(surfaced by ``_compute_effective_bb`` via ``_effbb_last_features``, no re-OCR),
then fit interpretable HARD GATES (abstain if any fires) that drop the
structurally error-prone hands the consensus signal is BLIND to (layout-
INDEPENDENT value errors: every reconstruction hypothesis agrees on the same
wrong bucket).

The point of the harness is the precision/coverage FRONTIER under honest 5-fold
pooled cross-validation (split by hand), NOT a vanity number. Gates are fit on
train folds and evaluated on held-out folds, so an over-tuned gate is exposed.

Usage:
  # build/refresh the feature table from the cache (slow path, ~minutes)
  python scripts/effbb_calibrate.py --build --cache data/effbb_cache/cache.jsonl
  # fit + CV from the cached feature table (fast)
  python scripts/effbb_calibrate.py
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ocr"))

from effbb_metrics import bucket_match, hero_folded_preflop

FEATURE_TABLE = Path("data/effbb_cache/effbb_features.jsonl")


# ---------------------------------------------------------------------------
# Feature extraction (the slow path — replays the cache through the parser).
# ---------------------------------------------------------------------------
def build_feature_table(cache_path: str, out_path: Path) -> int:
    """Replay hero-active cache hands, dump features + the correctness label."""
    from ocr.n8_parser import _compute_effective_bb, _effbb_last_features

    rows = [json.loads(l) for l in open(cache_path, encoding="utf-8") if l.strip()]
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            gt = r.get("gt") or {}
            ge = gt.get("effective_bb")
            if ge is None or ge < 1.0 or "inputs" not in r:
                continue
            if hero_folded_preflop(gt) is not False:
                continue  # hero-active only (the coached population)
            inp = r["inputs"]
            res = _compute_effective_bb(
                inp["columns"], inp["hero_stack"], inp["hero_position"],
                inp["stacks"], inp["named_stacks"],
            )
            p_eff = res[0]
            feat = _effbb_last_features()
            feat = {k: v for k, v in feat.items()
                    if k not in ("hero_position",)}  # keep it JSON-clean
            rec = {
                "hid": r["hand_id"],
                "gt_eff": ge,
                "p_eff": p_eff,
                # label: emitted AND bucket-correct. None p_eff = the parser
                # already abstained (not an emit; excluded from precision).
                "emitted": p_eff is not None,
                "correct": (p_eff is not None and bucket_match(p_eff, ge)),
                "feat": feat,
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def load_feature_table(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# Wilson 95% lower bound on a binomial proportion.
# ---------------------------------------------------------------------------
def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


# ---------------------------------------------------------------------------
# The abstain gates. Each is a pure predicate over the feature dict; it returns
# True to ABSTAIN (drop the emit). Interpretable, threshold-parameterized, and
# directly implementable in the parser. NO hand-ID literals.
# ---------------------------------------------------------------------------
def gate_predicates(p: dict):
    """Return the ordered list of (name, predicate) hard gates given params p.

    A hand is ABSTAINED if ANY predicate fires. Predicates only read the
    captured features (the parser can compute the identical signals at emit
    time). Each is motivated by a structural error mode:

      conf_floor       — the consensus confidence floor (existing lever).
      geometry_only    — base_conf<=0.86 binding (geometry/heuristic seat read):
                         only ~28% precise on its own; abstain unless the
                         engine vouches AND it is not boundary-fragile.
      engine_disagree  — independent betting-engine bucket dissents from emit:
                         the layout-independent value-error tell.
      boundary_frag    — emitted depth sits within X% of a bucket-cell edge: a
                         small OCR slip in the binding stack flips the bucket.
      method_straddle  — floors-on vs stack-only reconstructions land in
                         different buckets (a misread/mis-attributed all-in size).
      pot_residual     — preflop pot-conservation residual large: the betting
                         reconstruction does not reconcile with the pot header.
      hero_allin       — hero displayed ~0 (shoved/called-all-in) AND the
                         engine can't confirm the bucket: single-frame shove
                         size is unrecoverable (Phase-3 hard case).
    """
    g = []

    def add(name, fn):
        g.append((name, fn))

    if p.get("conf_floor") is not None:
        cf = p["conf_floor"]
        add("conf_floor", lambda f: (f.get("confidence") or 0.0) < cf)

    if p.get("geometry_only"):
        add("geometry_only", lambda f: bool(f.get("binding_geometry_only")))

    if p.get("engine_disagree"):
        add("engine_disagree", lambda f: bool(f.get("engine_disagrees")))

    bf = p.get("boundary_frag")
    if bf is not None:
        add("boundary_frag",
            lambda f: (f.get("boundary_dist") is not None
                       and f["boundary_dist"] < bf))

    if p.get("method_straddle"):
        add("method_straddle", lambda f: bool(f.get("method_straddle")))

    pr = p.get("pot_residual")
    if pr is not None:
        add("pot_residual",
            lambda f: (f.get("pot_residual") is not None
                       and f["pot_residual"] > pr))

    if p.get("hero_allin"):
        add("hero_allin",
            lambda f: (bool(f.get("hero_stack_near_zero"))
                       and not f.get("engine_agrees")))

    return g


def shipped_gate(f: dict, conf_floor: float = 0.7) -> bool:
    """EXACT mirror of the parser's shipped Phase-4 structural gate (n8_parser
    _compute_effective_bb, OCR_EFFBB_STRUCTURAL_GATE). Abstain (True) iff below
    the conf floor OR a structural signal fires, with the broad engine-disagree /
    method-straddle clauses SCOPED OFF the strong M1/M2 panel-read bindings
    (base_conf>=0.95) — those panel reads are reliable and an engine that reads
    a noisy seat must not abstain them. This is the operating point in prod."""
    if (f.get("confidence") or 0.0) < conf_floor:
        return True
    strong = (f.get("decision_class") in ("M1", "M2")
              and (f.get("base_conf") or 0.0) >= 0.95)
    return bool(
        (f.get("binding_geometry_only") and not f.get("engine_agrees"))
        or (f.get("hero_stack_near_zero") and not f.get("engine_agrees"))
        or (f.get("engine_disagrees") and not strong)
        or (f.get("method_straddle") and not strong)
    )


def abstains(feat: dict, gates) -> bool:
    return any(fn(feat) for _, fn in gates)


def evaluate(rows, gates):
    """Precision/coverage of emitted hands under the gate set, over ``rows``.

    Coverage denominator = ALL hero-active hands (emit + parser-abstain), so it
    is the honest production coverage. Emitted = parser emitted AND not gated.
    """
    n_total = len(rows)
    emit = [r for r in rows if r["emitted"] and not abstains(r["feat"], gates)]
    ok = [r for r in emit if r["correct"]]
    cov = len(emit) / n_total if n_total else 0.0
    prec = len(ok) / len(emit) if emit else 0.0
    lb = wilson_lower_bound(len(ok), len(emit))
    return {"n": n_total, "emit": len(emit), "ok": len(ok),
            "cov": cov, "prec": prec, "lb": lb}


# ---------------------------------------------------------------------------
# 5-fold pooled CV: fit params on train folds, evaluate on held-out, pool the
# held-out emits across folds, report the pooled frontier.
# ---------------------------------------------------------------------------
def make_folds(rows, k=5, seed=0):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    folds = [[] for _ in range(k)]
    for j, i in enumerate(idx):
        folds[j % k].append(rows[i])
    return folds


def fit_params_on_train(train, target_prec=0.995):
    """Greedily pick the gate operating point on TRAIN maximizing coverage s.t.
    train precision >= target. Sweeps a small interpretable grid; the chosen
    params are then applied UNCHANGED to the held-out fold.

    The gate family is fixed (geometry_only + engine_disagree always on — they
    isolate the layout-independent value errors); we sweep the continuous knobs
    (conf_floor, boundary_frag, pot_residual) + the hero_allin / method_straddle
    booleans for the operating point.

    NOTE (Phase-4 honest finding): target_prec=0.995 is UNREACHABLE on this cache
    at any usable coverage (the wrong emits are internally-consistent value
    errors no single-frame feature separates; the ceiling is ~86% @ ~10% cov). So
    when no grid point hits the target, we fall back to the highest-coverage
    point whose precision is within reach — and the caller reports that honestly.
    """
    best = None
    base = {"geometry_only": True, "engine_disagree": True}
    conf_grid = [0.7, 0.85, 0.9, 0.95, 0.98, 1.0]
    bf_grid = [None, 0.02, 0.03, 0.05]
    pr_grid = [None, 0.5, 0.3, 0.2]
    for allin in (False, True):
        for ms in (False, True):
            for cf in conf_grid:
                for bf in bf_grid:
                    for pr in pr_grid:
                        p = dict(base, conf_floor=cf, boundary_frag=bf,
                                 pot_residual=pr, hero_allin=allin,
                                 method_straddle=ms)
                        g = gate_predicates(p)
                        m = evaluate(train, g)
                        if m["emit"] < 20:
                            continue
                        if m["prec"] >= target_prec:
                            key = (m["cov"], m["prec"])
                            if best is None or key > best[0]:
                                best = (key, p)
    if best is None:
        # No train op-point hits target. Pick the MAX-PRECISION point with
        # non-trivial coverage (n>=80) — the honest "best we can do" operating
        # point, evaluated held-out by the caller.
        for allin in (True,):
            for ms in (True,):
                for cf in conf_grid:
                    for bf in bf_grid:
                        for pr in pr_grid:
                            p = dict(base, conf_floor=cf, boundary_frag=bf,
                                     pot_residual=pr, hero_allin=allin,
                                     method_straddle=ms)
                            m = evaluate(train, gate_predicates(p))
                            if m["emit"] < 80:
                                continue
                            if best is None or m["prec"] > best[0][1]:
                                best = ((m["cov"], m["prec"]), p)
    return best[1] if best else dict(base, conf_floor=1.0)


def cv(rows, k=5, seed=0, target_prec=0.995):
    folds = make_folds(rows, k, seed)
    pooled_emit, pooled_ok, pooled_n = 0, 0, 0
    chosen = []
    for i in range(k):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        p = fit_params_on_train(train, target_prec)
        chosen.append(p)
        m = evaluate(test, gate_predicates(p))
        pooled_emit += m["emit"]
        pooled_ok += m["ok"]
        pooled_n += m["n"]
    cov = pooled_emit / pooled_n if pooled_n else 0.0
    prec = pooled_ok / pooled_emit if pooled_emit else 0.0
    lb = wilson_lower_bound(pooled_ok, pooled_emit)
    return {"cov": cov, "prec": prec, "lb": lb, "emit": pooled_emit,
            "ok": pooled_ok, "n": pooled_n, "chosen": chosen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--build", action="store_true",
                    help="rebuild the feature table from the cache (slow)")
    ap.add_argument("--target", type=float, default=0.995)
    args = ap.parse_args()

    if args.build or not FEATURE_TABLE.exists():
        n = build_feature_table(args.cache, FEATURE_TABLE)
        print(f"built feature table: {n} hero-active hands -> {FEATURE_TABLE}")

    rows = load_feature_table(FEATURE_TABLE)
    print(f"\nloaded {len(rows)} hero-active hands")
    base_emit = [r for r in rows if r["emitted"]]
    base_ok = [r for r in base_emit if r["correct"]]
    print(f"parser baseline (no Phase-4 gates): "
          f"coverage={len(base_emit)/len(rows)*100:.1f}% "
          f"precision={len(base_ok)/len(base_emit)*100:.2f}%")

    # --- single-gate diagnostics: where does each gate isolate wrong hands? ---
    print("\n--- single-gate isolation (over emitted hands) ---")
    print("  gate                 abstained  wrong-in-abstained  kept-prec")
    single = {
        "geometry_only": {"geometry_only": True},
        "engine_disagree": {"engine_disagree": True},
        "boundary<0.03": {"boundary_frag": 0.03},
        "method_straddle": {"method_straddle": True},
        "pot_residual>0.3": {"pot_residual": 0.3},
        "hero_allin": {"hero_allin": True},
        "conf>=0.9": {"conf_floor": 0.9},
        "conf>=1.0": {"conf_floor": 1.0},
    }
    for name, p in single.items():
        g = gate_predicates(p)
        dropped = [r for r in base_emit if abstains(r["feat"], g)]
        wrong_dropped = [r for r in dropped if not r["correct"]]
        kept = [r for r in base_emit if not abstains(r["feat"], g)]
        kept_ok = [r for r in kept if r["correct"]]
        kp = len(kept_ok) / len(kept) * 100 if kept else 0
        print(f"  {name:20s} {len(dropped):8d}   "
              f"{len(wrong_dropped):4d}/{len(dropped):<4d}        {kp:6.2f}%")

    # --- the production structural gate frontier (conf-floor sweep) ---
    # This is the gate actually implemented in the parser: abstain if confidence
    # below floor OR any structural signal fires (geometry-only-unconfirmed,
    # engine-disagree, method-straddle, hero-all-in-unconfirmed).
    print("\n--- PRODUCTION structural-gate frontier (conf-floor sweep) ---")
    print("  conf_floor  cov     prec     wilson-lb   emit")
    base = {"geometry_only": True, "engine_disagree": True,
            "method_straddle": True, "hero_allin": True}
    for cf in (0.0, 0.7, 0.85, 0.9, 0.95, 0.98, 1.0):
        p = dict(base, conf_floor=cf if cf else None)
        m = evaluate(rows, gate_predicates(p))
        print(f"   {cf:.2f}       {m['cov']*100:5.1f}%  {m['prec']*100:6.2f}%  "
              f"{m['lb']*100:6.2f}%     {m['emit']}")

    # --- the ABSOLUTE precision ceiling (max-precision reachable slice) ---
    print("\n--- absolute precision ceiling (best reachable slice, any gate) ---")
    cieil = {"geometry_only": True, "engine_disagree": True,
             "method_straddle": True, "hero_allin": True,
             "conf_floor": 0.98, "boundary_frag": None}
    m = evaluate(rows, gate_predicates(cieil))
    print(f"  structural+conf0.98:  cov={m['cov']*100:5.1f}%  "
          f"prec={m['prec']*100:6.2f}%  lb={m['lb']*100:6.2f}%  emit={m['emit']}")

    # --- 5-fold pooled CV at the 99.5% target (DEMONSTRATES it is unreachable) ---
    print("\n=== 5-fold pooled CV @ point-precision target 99.5% (UNREACHABLE) ===")
    for seed in (0, 1, 2):
        res = cv(rows, k=5, seed=seed, target_prec=0.995)
        print(f"  seed={seed}: held-out coverage={res['cov']*100:5.2f}%  "
              f"precision={res['prec']*100:6.2f}%  "
              f"wilson-lb={res['lb']*100:6.2f}%  "
              f"({res['ok']}/{res['emit']} of {res['n']})")

    # --- the EXACT shipped parser gate (V4) frontier + held-out CV ---
    print("\n--- SHIPPED parser gate (V4, exact mirror) frontier ---")
    print("  conf_floor  cov     prec     wilson-lb   emit")
    for cf in (0.7, 0.9, 0.98, 1.0):
        emit = [r for r in rows if r["emitted"] and not shipped_gate(r["feat"], cf)]
        ok = [r for r in emit if r["correct"]]
        n = len(emit)
        prec = len(ok) / n if n else 0
        lb = wilson_lower_bound(len(ok), n)
        print(f"   {cf:.2f}       {n/len(rows)*100:5.1f}%  {prec*100:6.2f}%  "
              f"{lb*100:6.2f}%     {n}")

    print("\n=== 5-fold pooled CV — SHIPPED parser gate (V4, conf_floor=0.7) ===")
    for seed in (0, 1, 2):
        folds = make_folds(rows, 5, seed)
        pe = po = pn = 0
        for i in range(5):
            test = folds[i]
            emit = [r for r in test if r["emitted"] and not shipped_gate(r["feat"], 0.7)]
            ok = [r for r in emit if r["correct"]]
            pe += len(emit); po += len(ok); pn += len(test)
        cov = pe / pn if pn else 0
        prec = po / pe if pe else 0
        lb = wilson_lower_bound(po, pe)
        print(f"  seed={seed}: held-out coverage={cov*100:5.2f}%  "
              f"precision={prec*100:6.2f}%  wilson-lb={lb*100:6.2f}%  "
              f"({po}/{pe} of {pn})")

    print("\nHONEST SUMMARY: 99.5% point-precision is NOT reachable on this cache "
          "at usable coverage.\n  The wrong emits are internally-consistent "
          "single-frame VALUE errors (hero/villain stack misread,\n  "
          "start-vs-displayed) that NO surfaced feature separates from correct "
          "emits.\n  Absolute ceiling ~86% @ ~10% cov. The SHIPPED parser gate "
          "(V4, conf_floor=0.7)\n  lifts emitted precision to ~74% @ ~61% cov "
          "from the 70.9% @ 78.2% ungated baseline\n  (it scopes the broad "
          "engine-disagree/method-straddle clauses off strong M1/M2 panel\n  "
          "reads to keep those correct emits — slightly more coverage, slightly "
          "less\n  precision than the golden-agnostic V1 frontier above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
