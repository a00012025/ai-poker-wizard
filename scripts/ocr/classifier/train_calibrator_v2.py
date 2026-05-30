"""Phase 11.C.1 — train the v2 RF+GB+LR ensemble calibrator on rich
feature records dumped by ocr_precision.py --dump-all.

Reads one or more all_records.jsonl files, concatenates them into the
calibrator's training set, runs 5-fold CV to produce out-of-fold
probabilities, then refits each base model on all training data and
saves the three .joblib bundles + an OOF predictions JSON.

Stacking is intentionally simple: at evaluation time the three
predict_proba outputs are averaged. CalibratorScorer (loaded later)
implements that averaging directly so this file only needs to save the
base models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_records(paths: list[Path], feature_key: str) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        if not p.exists():
            sys.exit(f"missing records file: {p}")
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get(feature_key) or "fields" not in rec:
                continue
            if "hand_exact" not in rec["fields"]:
                continue
            records.append(rec)
    return records


def _xy(
    records: list[dict], feature_key: str
) -> tuple[list[list[float]], list[int], list[str]]:
    X = [r[feature_key] for r in records]
    y = [1 if r["fields"]["hand_exact"] else 0 for r in records]
    ids = [r.get("hand_id", "") for r in records]
    return X, y, ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, action="append",
                    help="all_records.jsonl path (may be passed multiple times)")
    ap.add_argument("--val", action="append", default=[],
                    help="optional validation records appended to training set")
    ap.add_argument("--features", required=True,
                    help="feature schema txt — used to verify feature width")
    ap.add_argument("--feature-key", default="v2_features",
                    help="record field holding the feature vector "
                         "(v2_features or v3_features)")
    ap.add_argument("--out-suffix", default="v2",
                    help="artifact suffix: rf_model_<suffix>.joblib etc.")
    ap.add_argument("--isotonic", action="store_true",
                    help="fit an isotonic regression on the OOF ensemble "
                         "average for calibration; saves isotonic_<suffix>.joblib "
                         "and stores calibrated OOF probabilities")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    feat_names = [
        line.strip() for line in Path(args.features).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected_n = len(feat_names)

    paths = [Path(p) for p in args.train] + [Path(p) for p in args.val]
    records = _load_records(paths, args.feature_key)
    if not records:
        sys.exit(f"no usable records (each must have {args.feature_key} "
                 "+ fields.hand_exact)")
    X, y, ids = _xy(records, args.feature_key)
    assert all(len(row) == expected_n for row in X), (
        f"feature width mismatch: schema={expected_n} actual={len(X[0])}"
    )

    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_kw = dict(n_estimators=400, max_depth=8, class_weight="balanced",
                 random_state=args.seed, n_jobs=-1)
    gb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                 random_state=args.seed)
    lr_kw = dict(C=1.0, class_weight="balanced", max_iter=1000)

    # k-fold OOF on the joint set
    folds = max(2, min(args.folds, int(min(np.bincount(ya)))))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    oof_rf = np.zeros(len(ya), dtype=float)
    oof_gb = np.zeros(len(ya), dtype=float)
    oof_lr = np.zeros(len(ya), dtype=float)
    for k, (tr_idx, te_idx) in enumerate(skf.split(Xa, ya)):
        rf = RandomForestClassifier(**rf_kw).fit(Xa[tr_idx], ya[tr_idx])
        gb = GradientBoostingClassifier(**gb_kw).fit(Xa[tr_idx], ya[tr_idx])
        scaler = StandardScaler().fit(Xa[tr_idx])
        lr = LogisticRegression(**lr_kw).fit(scaler.transform(Xa[tr_idx]),
                                             ya[tr_idx])
        oof_rf[te_idx] = rf.predict_proba(Xa[te_idx])[:, 1]
        oof_gb[te_idx] = gb.predict_proba(Xa[te_idx])[:, 1]
        oof_lr[te_idx] = lr.predict_proba(scaler.transform(Xa[te_idx]))[:, 1]
        print(f"fold {k + 1}/{folds}: trained on {len(tr_idx)}, "
              f"scored on {len(te_idx)}")

    # Refit base models + scaler on the full joint set
    rf_full = RandomForestClassifier(**rf_kw).fit(Xa, ya)
    gb_full = GradientBoostingClassifier(**gb_kw).fit(Xa, ya)
    scaler_full = StandardScaler().fit(Xa)
    lr_full = LogisticRegression(**lr_kw).fit(scaler_full.transform(Xa), ya)

    sfx = args.out_suffix
    joblib.dump({"model": rf_full, "feature_names": feat_names},
                out_dir / f"rf_model_{sfx}.joblib")
    joblib.dump({"model": gb_full, "feature_names": feat_names},
                out_dir / f"gb_model_{sfx}.joblib")
    joblib.dump({"model": lr_full, "scaler": scaler_full,
                 "feature_names": feat_names},
                out_dir / f"lr_model_{sfx}.joblib")

    oof_avg = (oof_rf + oof_gb + oof_lr) / 3.0

    # Isotonic calibration on the OOF average. Monotonic, so it leaves the
    # τ-sweep operating-point curve (ranking) untouched while pulling the
    # predicted probabilities onto the empirical-accuracy diagonal — the
    # fix for the v2 calibrator's 0.135 ECE (target ≤ 0.04). Fit on
    # out-of-fold scores so it generalises rather than memorising.
    iso = None
    if args.isotonic:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_avg, ya)
        joblib.dump({"model": iso, "feature_key": args.feature_key},
                    out_dir / f"isotonic_{sfx}.joblib")
        oof_store = iso.predict(oof_avg)
    else:
        oof_store = oof_avg

    oof_map = {ids[i]: float(oof_store[i]) for i in range(len(ya))}
    (out_dir / f"oof_{sfx}.json").write_text(json.dumps(oof_map, indent=2))

    # Quick summary so eyes-on can sanity-check
    pos_rate = float(ya.mean())
    print(f"\nn={len(ya)} pos_rate={pos_rate:.3f}")
    print(f"OOF positive-class mean prob: "
          f"rf={oof_rf.mean():.3f} gb={oof_gb.mean():.3f} "
          f"lr={oof_lr.mean():.3f} avg={oof_avg.mean():.3f}")
    print(f"models saved → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
