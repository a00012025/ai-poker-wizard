# Three-Day 99% Push — Non-Ship Handoff (2026-05-23)

## Executive summary

**The three-day target (precision ≥99% @ coverage ≥70% on the held-out test bucket) was not reached, and the calibrator analysis shows it cannot be reached with the current feature set.** The pre-push baseline was 95.588% precision at 85.237% coverage (612/718 emitted, 27 wrong). Trying to push precision up by gating on the existing diagnostics either: (a) abstains too many exact hands (gate too broad); or (b) catches too few wrong hands (gate too narrow). A logistic-regression calibrator trained out-of-fold on 12 diagnostic features peaks at **95.42% precision at 87.0% coverage** — i.e. with these features no threshold reaches the 99% target.

The push instead shipped:

1. A **wrong-hand audit** of the 27 emitted failures (`docs/superpowers/plans/artifacts/2026-05-23-three-day-99-audit.md`).
2. A **`confidence_gate.evaluate()`** module that wraps the existing `emit_threshold` + `safe_emit_reason` logic with a small set of selective hard-abstain rules (Rule A: severe collapse + player-count mismatch; Rule D: very low card confidence; Rule E: phantom doubled all-in; Rule G: trailing all-in after call; Rule H: pot-inconsistent postflop collapse). A wider soft-risk path is wired but disabled by default because the calibrator analysis shows it's net-harmful.
3. A **gate-eval harness** (`scripts/ocr_gate_eval.py`) that replays cached `ocr_precision` records through the gate without re-running OCR.
4. **`ocr_precision.py --dump-all`** so future calibrator work has full per-hand feature dumps.
5. **28 regression fixtures** covering the gate's selectivity claims and the documented residual cases.

## Final metrics (held-out test bucket)

| Run | emitted | wrong | precision | coverage | notes |
| --- | --- | --- | --- | --- | --- |
| pre-push baseline (`data/ocr_precision_current`) | 612 | 27 | 95.588% | 85.237% | run prior to push |
| v1 gate (broad rules) | 286 | 0 | 100.000% | 39.833% | catastrophic over-abstain — 357 exact hands lost |
| v2 gate (risky escalation) | 342 | 1 | 99.708% | 47.632% | still over-abstains (270 exact lost) |
| v3 gate (narrow rules) | 415 | 5 | 98.795% | 57.799% | best precision-coverage trade among gate variants but still < ship target and < non-ship bar |
| v4 gate (drop weak rules) | 590 | 19 | 96.780% | 82.173% | trade-off favours coverage but precision below non-ship bar |
| **v5 gate (final, minimal hard rules)** | **594** | **20** | **96.633%** | **82.730%** | shipped — selective hard rules only, soft-risk opt-in |

Ship target (precision ≥99% @ coverage ≥70%): **not reached**. Non-ship minimum bar (≥97.5% at ≥75% or ≥98.5% at ≥70%): **not reached**. The shipped v5 trades 2.5pp coverage for 1pp precision vs baseline — marginal utility.

## Calibrator analysis (the decisive evidence)

`scripts/_tmp.py` (preserved in the diff) fits a logistic regression on 12 features extracted from `all_records.jsonl` and evaluates out-of-fold with 5-fold CV.

```
pool size: 702, exact=643, wrong=59
Threshold sweep (OOF):
  tau=0.99: prec=0.9130 cov=0.033
  tau=0.98: prec=0.9788 cov=0.403
  tau=0.95: prec=0.9700 cov=0.570
  tau=0.92: prec=0.9507 cov=0.694
  tau=0.91: prec=0.9492 cov=0.756
  tau=0.85: prec=0.9522 cov=0.895
Best precision at coverage ≥70%: tau=0.865, prec=0.9542 cov=0.870
```

**No threshold reaches 99% precision at 70% coverage with these features.** This is the central finding of the push: the 27 wrong-emitted hands and the bulk of the 585 emitted-exact hands share the same diagnostic profile (collapse loss, weak player tracking, all-in tokens, weak button signal) so a calibrator cannot separate them cleanly.

Top calibrator coefficients (logistic, standardised):
```
has_allin     : -0.793   # AI in actions is the strongest wrong predictor
card_conf     : +0.571   # confirms card_confidence as a soft positive
confidence    : +0.407   # existing blended confidence is mildly useful
rf_diff       : -0.327   # raw-vs-final player mismatch is a weak negative
safe_emit     : +0.214   # safe_emit_reason is mildly positive
```

Strong-signal features (button_conf, reaction_signal, pre_loss) ended up with tiny coefficients — they're not separating the wrong/exact populations.

## Why the gate is still net-positive

Even though v5 misses the ship target, the rules retained were chosen by selectivity audit:

| Rule | hits (v3) | exact lost | wrong caught | wrong-rate | rationale |
| --- | --- | --- | --- | --- | --- |
| A — severe collapse + player-count mismatch | 16 | 7 | 9 | 56% | strongest specific signal; catches TM5913031183 (conf 1.000) and similar |
| D — `card_conf < 0.5` | ~18 | ~3 | ~15 | 80%+ | very low card confidence almost always equals wrong |
| E — `AI-AI` doubled all-in | 1 | 0 | 1 | 100% | phantom row collapse |
| G — trailing `-C-AI<n>[-F]` after a previous AI | 4 | 1 | 3 | 75% | duplicate all-in attribution |
| H — pot inconsistent (≤0.5) + collapse ≥3 + postflop entries | 7 | 4 | 3 | 43% | bordeline — kept because the wrong cases include board-street miscount |

Rules that were tried and removed for poor selectivity:
- Severe collapse alone (`pre_loss >= 5`): 357 exact lost, ~50 wrong caught — net harm.
- Severe collapse + weak tracking + AI: 108 exact lost, ~11 wrong caught — net harm.
- Reaction signal + severe collapse: 19 exact lost, 1 wrong caught — net harm.
- Sizeless AI mid-sequence: 4 exact lost, 2 wrong caught — net harm.
- Weak tracking + AI + any collapse (soft-risk, broad): ~225 exact demoted, ~17 wrong caught — net harm at default threshold.

## What blocked the 99% target

1. **Features overlap.** The 27 wrong-emitted hands and the bulk of the 585 emitted-exact hands share the same diagnostic profile. A calibrator cannot rank them apart.
2. **Two distinct wrong-hand modes.**
   - 3 high-confidence wrong (conf ≈ 0.997-1.000) need parser-level fixes (Rule A catches one of these via player-count mismatch).
   - 24 boundary-confidence wrong (conf ≈ 0.898-0.900) share features with many exact hands at the same conf level.
3. **Diagnostics are mostly outcome-correlated, not cause-correlated.** Pre-collapse loss happens to many hands — it doesn't itself cause wrong outcomes. The actual wrong-hand causes (re-action ambiguity, table-size estimation drift, hero-card crop generalisation) need new diagnostics that aren't currently exposed.

## Phase 8 recommendation

The roadmap explicitly contemplated this outcome in Phase 8:

> If after the learned calibrator (step 4) we cannot achieve coverage ≥ 70% at precision 99%, two recovery options:
>   - Loosen coverage to 60% (requires explicit user sign-off)
>   - Re-open Phases 2-7 on the residual failure clusters that the calibrator can't isolate

Concrete next-session work:

1. **Add new diagnostic features** for the wrong-hand causes the current feature set can't see:
   - All-in re-action attribution confidence (when the re-actor has no name)
   - Row-collision detection in `_group_by_y`
   - Per-row OCR confidence variance
   - Hero-card crop quality score (overlay coverage, brightness, mask consistency)
2. **Parser-fix candidates** for the residual cluster:
   - TM5913201917: pre_loss=6, no AI, all-clean parts but position wrong. Needs row-segmentation fix.
   - TM5879884236-style `-C-AI<n>-F` tail: extend Rule G to also catch with button strong (currently bypasses).
   - TM5963073078-style raw<final player over-count with no other signals.
3. **Train calibrator on train+val with new features** once the parser-fix candidates produce richer diagnostics. The current 12 features peak at 95.4% — any improvement to ≥98% in the calibrator would be a major signal that the new features are working.

## Acceptance battery (Day 3 results)

| Step | Status |
| --- | --- |
| `pytest tests/ocr -q` | PASS (52 + 28 gate tests) |
| H2894 snapshot | (Day 3 step — run as part of final commit) |
| `data/ocr_precision_gate_test_v5/summary.json` | written |

## Acceptance vs. shipping criteria

| Criterion | Required | v5 result | met? |
| --- | --- | --- | --- |
| `coverage` | ≥ 70.0% | 82.73% | ✅ |
| `hand_exact` | ≥ 99.0% | 96.63% | ❌ |
| `hero_hand` | ≥ 99.8% | 99.83% | ✅ |
| `critical_error` | ≤ 0.25% | 0.17% | ✅ |
| `ece_10bin` | ≤ 0.04 | (computed in calibration_summary.json) | TBD |

Ship gate **not met**. Non-ship minimum also not met. Recommendation: do not enable the gate in production by default until the Phase 8 work above produces a calibrator that can clear the precision bar.

## Files of interest in this push

- `scripts/ocr/confidence_gate.py` — gate module
- `scripts/ocr_gate_eval.py` — cached-record replay harness for fast iteration
- `scripts/ocr_precision.py` — wired in via `--disable-gate` and `--dump-all`
- `tests/ocr/test_confidence_gate.py` — fixture-based gate tests (12)
- `tests/ocr/test_confidence_gate_unit.py` — synthetic-input unit tests (16)
- `docs/superpowers/plans/artifacts/2026-05-23-three-day-99-audit.md` — the wrong-hand audit
- `data/ocr_precision_gate_test_v5/` — final benchmark output
- `data/ocr_precision_gate_test_v4/all_records.jsonl` — full-feature dump for future calibrator training
