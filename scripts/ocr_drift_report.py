"""Phase 11.E — daily OCR drift monitor.

Reads the last 24h of analysis_snapshots, runs the calibrated OCR
pipeline against each, and computes:

* ``emit_rate``       — fraction of parses where calibrator p ≥ τ
* ``card_conf_p50``   — median raw card_confidence
* ``card_conf_p10``   — 10th-percentile raw card_confidence
* ``calibrator_p50``  — median p(correct)
* ``calibrator_p10``  — 10th-percentile p(correct)
* ``ensemble_used_rate`` — fraction where ensemble routing fired

Compares against a rolling 7-day baseline in
``data/drift_baselines/<metric>.json``. Alerts (stdout + non-zero exit)
when any metric breaches > 2σ from baseline. Designed for a daily cron
or PTB JobQueue.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import asyncpg


METRICS = [
    "emit_rate", "card_conf_p50", "card_conf_p10",
    "calibrator_p50", "calibrator_p10", "ensemble_used_rate",
]


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


async def _fetch_window(conn, since: datetime) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT hand_id, image_data
        FROM analysis_snapshots
        WHERE source_type='image' AND image_data IS NOT NULL
          AND created_at >= $1
        ORDER BY created_at DESC
        """,
        since,
    )
    return [{"hand_id": r["hand_id"], "image_bytes": bytes(r["image_data"])}
            for r in rows]


def _compute_metrics(rows: list[dict], tau: float) -> dict:
    from ocr.n8_parser import parse_n8_screenshot
    from ocr.confidence_gate import CalibratorScorer
    scorer = CalibratorScorer()

    card_confs: list[float] = []
    cal_scores: list[float] = []
    emit_count = 0
    ensemble_used_count = 0
    parsed_count = 0
    for r in rows:
        try:
            result = parse_n8_screenshot(r["image_bytes"])
        except Exception:
            continue
        parsed = result.get("hand")
        if parsed is None:
            continue
        parsed_count += 1
        card_confs.append(float(result.get("card_confidence") or 0.0))
        score = scorer.score(result, hand_id=r["hand_id"])
        if score is not None:
            cal_scores.append(score)
            if score >= tau:
                emit_count += 1
        diag = result.get("diagnostics") or {}
        if diag.get("ensemble_used"):
            ensemble_used_count += 1

    if not parsed_count:
        return {m: 0.0 for m in METRICS} | {"n": 0}
    return {
        "n": parsed_count,
        "emit_rate": emit_count / parsed_count,
        "card_conf_p50": _quantile(card_confs, 0.5),
        "card_conf_p10": _quantile(card_confs, 0.1),
        "calibrator_p50": _quantile(cal_scores, 0.5) if cal_scores else 0.0,
        "calibrator_p10": _quantile(cal_scores, 0.1) if cal_scores else 0.0,
        "ensemble_used_rate": ensemble_used_count / parsed_count,
    }


def _update_baseline(
    baseline_dir: Path, day: str, metrics: dict, window: int = 7,
) -> dict:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    hist_path = baseline_dir / "history.jsonl"
    history: list[dict] = []
    if hist_path.exists():
        for line in hist_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            history.append(json.loads(line))
    # Drop any entry for the same day (replace).
    history = [h for h in history if h.get("day") != day]
    history.append({"day": day, **metrics})
    history = history[-30:]  # keep last 30 days
    hist_path.write_text("\n".join(json.dumps(h) for h in history) + "\n")

    # Rolling baseline = mean/stdev over the prior `window` days.
    rolling = [h for h in history[:-1] if h.get("n", 0) > 0][-window:]
    baseline = {}
    for m in METRICS:
        vals = [float(h.get(m) or 0.0) for h in rolling]
        if len(vals) >= 3:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            baseline[m] = {"mean": mean, "stdev": math.sqrt(var),
                           "n_days": len(vals)}
        else:
            baseline[m] = {"mean": None, "stdev": None,
                           "n_days": len(vals)}
    (baseline_dir / "rolling.json").write_text(json.dumps(baseline, indent=2))
    return baseline


def _check_alerts(metrics: dict, baseline: dict, sigma: float = 2.0) -> list[str]:
    alerts: list[str] = []
    for m in METRICS:
        b = baseline.get(m) or {}
        mean = b.get("mean")
        stdev = b.get("stdev")
        if mean is None or stdev is None or stdev < 1e-6:
            continue
        v = float(metrics.get(m) or 0.0)
        z = (v - mean) / stdev
        if abs(z) > sigma:
            alerts.append(
                f"{m}: {v:.4f} drifted {z:+.2f}σ from baseline "
                f"(mean={mean:.4f}, σ={stdev:.4f}, n={b['n_days']}d)"
            )
    return alerts


async def main_async(args) -> int:
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    conn = await asyncpg.connect(
        os.environ["SUPABASE_CONN"], statement_cache_size=0
    )
    rows = await _fetch_window(conn, since)
    await conn.close()

    print(f"window: last {args.hours}h since {since.isoformat()}")
    print(f"rows  : {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]

    metrics = _compute_metrics(rows, args.tau)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    baseline_dir = Path(args.baseline_dir)
    baseline = _update_baseline(baseline_dir, day, metrics)
    alerts = _check_alerts(metrics, baseline)

    summary = {
        "day": day,
        "window_hours": args.hours,
        "metrics": metrics,
        "baseline": baseline,
        "alerts": alerts,
    }
    print(json.dumps(summary, indent=2))
    return 1 if alerts else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.99,
                    help="Calibrator threshold for emit_rate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline-dir", default="data/drift_baselines")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
