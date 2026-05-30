"""Phase 11.D-a — the joint evaluator computes recall over the FULL
bucket (parsable + un-emittable parse_none), not just parsable. These
tests pin that denominator so the 95%-recall gate stays honest: true
parse_none can never be emitted, so it must count against coverage.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_calibrate_v2",
    Path(__file__).resolve().parents[2] / "scripts" / "_calibrate_v2.py",
)
cal = importlib.util.module_from_spec(_SPEC)
sys.modules["_calibrate_v2"] = cal
_SPEC.loader.exec_module(cal)


def _scored(pos: int, neg: int):
    """`pos` correct emits at high score, `neg` wrong emits at mid score."""
    return [(0.9, 1)] * pos + [(0.6, 0)] * neg


def test_prec_cov_uses_full_total_denominator():
    scored = _scored(90, 10)  # 100 parsable
    # 50 un-emittable → total 150. Emit everything above tau=0.
    prec, cov, em, cor = cal._prec_cov(scored, 0.0, total=150)
    assert em == 100
    assert cor == 90
    assert abs(prec - 0.90) < 1e-9
    assert abs(cov - 100 / 150) < 1e-9  # coverage over 150, not 100


def test_prec_cov_defaults_to_len_scored_when_no_total():
    scored = _scored(90, 10)
    _, cov, _, _ = cal._prec_cov(scored, 0.0)
    assert abs(cov - 1.0) < 1e-9


def test_precision_at_coverage_caps_when_target_exceeds_parsable():
    # 90 parsable, 10 un-emittable → total 100. Parsable ceiling = 0.90.
    scored = _scored(80, 10)  # 90 parsable
    row = cal._precision_at_coverage(scored, coverage=0.95, total=100)
    # Cannot reach 95% coverage: capped at 90/100.
    assert row["emitted"] == 90
    assert abs(row["coverage"] - 0.90) < 1e-9
    assert row["target_coverage"] == 0.95


def test_precision_at_coverage_reachable_target():
    scored = _scored(80, 20)  # 100 parsable, sorted: 80 correct then 20 wrong
    row = cal._precision_at_coverage(scored, coverage=0.80, total=125)
    # 80% of 125 = 100 emits = all parsable → precision 80/100.
    assert row["emitted"] == 100
    assert abs(row["precision"] - 0.80) < 1e-9


def test_sweep_ship_gate_unreachable_when_parsable_below_target():
    # Even a perfect calibrator can't hit 95% coverage if only 90% parsable.
    scored = _scored(90, 0)  # 90 perfect parsable
    gate = cal._sweep_for_ship_gate(scored, total=100)
    assert gate is None


def test_sweep_ship_gate_reachable_with_enough_parsable():
    scored = _scored(99, 1)  # 100 parsable, 99% precise
    gate = cal._sweep_for_ship_gate(scored, total=100)
    assert gate is not None
    assert gate["coverage"] >= 0.95
    assert gate["precision"] >= 0.99
