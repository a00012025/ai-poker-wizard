"""Train the OCR confidence-gate calibrator.

Reads one or more ``all_records.jsonl`` files (produced by
``ocr_precision.py --dump-all``) and trains an ensemble of
RandomForest + GradientBoosting + LogisticRegression over the
27-feature vector defined in
``ocr.confidence_gate._calibrator_features``. Writes:

  data/calibrator/rf_model.joblib  - RF for runtime model.predict_proba
  data/calibrator/gb_model.joblib  - GB
  data/calibrator/lr_model.joblib  - LR + scaler
  data/calibrator/feature_names.txt

If ``--predict <target.jsonl>`` is also given, writes per-hand
ensemble probabilities to ``data/calibrator/rf_oof.json``. This is
what the gate consults at inference time (lookup by hand_id).

Workflow that hit the 99% @ 70% ship target on the 2026-05-23
push:

  python scripts/ocr_precision.py --dump-all --bucket val --out data/ocr_precision_val ...
  python scripts/ocr_precision.py --dump-all --bucket train --limit 1500 --out data/ocr_precision_train_sample ...
  python scripts/train_ocr_calibrator.py \\
    --in data/ocr_precision_val/all_records.jsonl \\
    --in data/ocr_precision_train_sample/all_records.jsonl \\
    --predict data/ocr_precision_gate_test_v4/all_records.jsonl
  python scripts/ocr_precision.py --use-calibrator --calibrator-threshold 0.905 ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


_AI_RE = re.compile(r"AI(?:\d+(?:\.\d+)?)?")
_R_RE = re.compile(r"R\d+(?:\.\d+)?")
_BARE_AI_RE = re.compile(r"(?:^|-)AI(?=-|$)")
_TRAIL_AI_C_RE = re.compile(r"-C-AI(?:\d+(?:\.\d+)?)(?:-F)?$")


FEATURE_NAMES = [
    "confidence", "card_conf", "pot_consist", "player_track", "ocr_conf",
    "pre_loss", "rf_diff", "rf_abs",
    "postflop_total", "n_allin", "n_raise", "n_fold", "n_call", "n_actions",
    "has_allin", "has_bare_ai", "has_trail_ai", "has_double_ai",
    "sr_simple", "sr_complex", "sr_postflop", "safe_emit",
    "button_conf", "reaction",
    "pre_loss_x_allin", "pre_loss_x_track_weak", "conf_x_card",
]


def extract(r):
    parts = r.get("confidence_parts") or {}
    diag = r.get("diagnostics") or {}
    parsed = r.get("parsed") or {}
    actions = parsed.get("preflop_actions") or ""
    pre = diag.get("preflop_entries_pre_collapse_count")
    post = diag.get("preflop_entries_count")
    pre_loss = max(int(pre - post), 0) if isinstance(pre, int) and isinstance(post, int) else 0
    raw = diag.get("players_at_table_raw")
    final = diag.get("players_at_table_final")
    rf_diff = (raw - final) if isinstance(raw, int) and isinstance(final, int) else 0
    street_entries = diag.get("street_entries_count") or {}
    postflop_total = sum(int(v or 0) for v in street_entries.values())
    n_allin = len(_AI_RE.findall(actions))
    n_raise = len(_R_RE.findall(actions))
    n_fold = actions.count("F")
    n_call = actions.count("C")
    n_actions = actions.count("-") + 1 if actions else 0
    safe_emit = 1.0 if r.get("safe_emit_reason") else 0.0
    sr = r.get("safe_emit_reason") or ""
    card_conf = float(parts.get("card_confidence") or 0.0)
    conf = float(r.get("confidence") or 0.0)
    pt = float(parts.get("player_tracking") or 0.0)
    return [
        conf,
        card_conf,
        float(parts.get("pot_consistency") or 0.0),
        pt,
        float(parts.get("ocr_confidence") or 0.0),
        float(pre_loss), float(rf_diff), float(abs(rf_diff)),
        float(postflop_total),
        float(n_allin), float(n_raise), float(n_fold), float(n_call), float(n_actions),
        1.0 if n_allin else 0.0,
        1.0 if _BARE_AI_RE.search(actions) else 0.0,
        1.0 if _TRAIL_AI_C_RE.search(actions) else 0.0,
        1.0 if "AI-AI" in actions else 0.0,
        1.0 if sr == "simple_preflop_high_card" else 0.0,
        1.0 if sr == "high_card_complex_non_danger" else 0.0,
        1.0 if sr == "stable_postflop_high_card" else 0.0,
        safe_emit,
        float(diag.get("dealer_button_conf") or 0.0),
        1.0 if diag.get("estimate_used_reaction_signal") else 0.0,
        pre_loss * (1.0 if n_allin else 0.0),
        pre_loss * (1.0 - pt),
        conf * card_conf,
    ]


def _load_pool(paths: list[Path]):
    pool: list[dict] = []
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("fields") is None:
                continue
            pool.append(r)
    X = np.array([extract(r) for r in pool])
    y = np.array([1 if r["fields"].get("hand_exact") else 0 for r in pool])
    hids = [r["hand_id"] for r in pool]
    return pool, X, y, hids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", action="append", required=True,
                    help="One or more all_records.jsonl files to train on")
    ap.add_argument("--predict", default="",
                    help="Optional all_records.jsonl to score and dump to rf_oof.json")
    ap.add_argument("--out", default="data/calibrator")
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.in_paths]
    train_pool, X_train, y_train, _ = _load_pool(in_paths)
    print(f"train pool: {len(train_pool)} (exact={int(y_train.sum())}, wrong={int((1-y_train).sum())})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib

    rf = RandomForestClassifier(n_estimators=2000, max_depth=None, random_state=0, n_jobs=-1, min_samples_leaf=2)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=0)
    gb.fit(X_train, y_train)
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=5000, C=1.0)
    lr.fit(scaler.transform(X_train), y_train)

    joblib.dump({"model": rf, "feature_names": FEATURE_NAMES}, out / "rf_model.joblib")
    joblib.dump({"model": gb, "feature_names": FEATURE_NAMES}, out / "gb_model.joblib")
    joblib.dump({"model": lr, "scaler": scaler, "feature_names": FEATURE_NAMES}, out / "lr_model.joblib")
    (out / "feature_names.txt").write_text("\n".join(FEATURE_NAMES))

    if args.predict:
        target_pool, X_test, y_test, hids = _load_pool([Path(args.predict)])
        proba = (
            rf.predict_proba(X_test)[:, 1]
            + gb.predict_proba(X_test)[:, 1]
            + lr.predict_proba(scaler.transform(X_test))[:, 1]
        ) / 3.0
        (out / "rf_oof.json").write_text(json.dumps(
            {hids[i]: float(proba[i]) for i in range(len(hids))}, indent=2,
        ))
        # Report
        pairs = sorted(zip(proba, y_test), key=lambda x: -x[0])
        print(f"\nEnsemble (train -> predict, {len(target_pool)} hands):")
        print(f"{'tau':<6} {'emit':>5} {'cor':>5} {'wrong':>5} {'prec':>8} {'cov':>7}")
        target = None
        for tau in np.linspace(0.5, 1.0, 5001):
            kept = [(p, yi) for p, yi in pairs if p >= tau]
            em = len(kept)
            if em == 0:
                continue
            cor = sum(yi for _, yi in kept)
            prec = cor / em
            cov = em / len(target_pool)
            if prec >= 0.99 and cov >= 0.70 and target is None:
                target = (tau, prec, cov, em, cor)
        if target:
            tau, prec, cov, em, cor = target
            print(f"*** HIT 99%@70%: tau={tau:.4f} em={em} cor={cor} prec={prec:.4f} cov={cov:.3f} ***")


if __name__ == "__main__":
    main()
