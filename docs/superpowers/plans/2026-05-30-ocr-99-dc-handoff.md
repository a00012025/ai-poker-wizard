# OCR 99%@95% — Session Handoff (Phase 11.D-c done, next: hero-cards + recall)

> **Goal (unchanged): 99% precision @ 95% recall on the PokerCraft test set (718 hands).**
> This handoff continues `docs/superpowers/plans/2026-05-30-ocr-99-handoff.md`. That earlier
> plan's D-a/D-b/D-c framing is superseded by what was actually learned below — read this first.

---

## TL;DR — what happened this session

D-a (calibrator) was evaluated and found **insufficient for precision** (ECE fixed 0.135→0.034,
but precision capped ~95%@80% — the residual errors are *confident parser mistakes* with no
uncertainty signal). The table-size parser-fix path was exhausted (region pipeline at its OCR
noise floor). The breakthrough: a **holistic VLM** reads the table directly and doesn't share the
parser's failure mode. Shipped in **PR #31** (branch `worktree-ocr-99-calibrator-first`, commit
`dc0d56e`):

- **VLM structural re-check** (`gemini-3.5-flash`, seat count + hero position) — flag-gated
  override that fixes confident table-size/position errors.
- **All-in runout truncation** (deterministic) — fixes phantom turn/river board cards.

**Measured (test 718):** hand_exact 566→597 (+31), wrong 85→**40 (−53%)**, board_wrong 28→1,
position_wrong 23→4. **99% precision now holds to ~75% coverage (was ~67%).**

---

## Where the work lives

- **Worktree**: `/home/harry/ai-poker-wizard/.claude/worktrees/ocr-99-calibrator-first`
- **Branch**: `worktree-ocr-99-calibrator-first` — pushed, **PR #31** open against `main`.
- **Symlinks** (from worktree): `.env`, `.tokens.json`, `data/`, `.gto_cache/` → main repo.
- **Keys in `.env`**: `GEMINI_API_KEY`, `OPENAI_API_KEY` (no Anthropic/xAI).
- **GPU**: single 16GB Quadro RTX 5000, **shared with the live bot** (`python -m src.main_gemini`,
  ~0.7–1.6GB). Dumps need `OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  **4 workers** — 8 workers OOM; thread-limiting gave a 6.6× speedup (CPU was thrashing).

---

## The complete error diagnosis (test set, the map to 99%@95%)

After this PR: **597 correct, 40 wrong, 81 parse_none** (of 718).

| Error class | Count (post-fix) | Nature | Status / path |
|---|---|---|---|
| board | **1** (was 28) | phantom runout (cards read fine) | ✅ FIXED (deterministic truncation) |
| position | **4** (was 23) | table-size misestimation | ✅ mostly fixed (flash override) |
| hero_cards | **28** (was 30) | **rank misreads + localization** | ⬜ **CardCNN retrain** (next) |
| preflop_action_types | 7 | missing/garbled action codes | ⬜ minor |
| parse_none | **81** | assembly failed (64/81 have good cards) | ⬜ **recall** — Gemini fallback |

### Two binding constraints remain for 99%@95%

**(1) PRECISION above ~75% coverage — blocked by hero_cards (28).**
- Diagnosed: **0 suit-only errors**; all 30 involve rank (16 rank-only, 14 rank+suit). The ~8
  *confident* ones (card_conf≥0.7) are rank misreads — **"4" is over-predicted** (4/9 known-worst
  confusion) and `corner_ocr` is sometimes confidently wrong. 22/30 are already low-conf →
  abstained by the calibrator (so they hurt *recall* more than precision).
- **VLMs do NOT help here** (tested): gpt-5.5 full-img 67%/65%, flash hero-crop 53%/100%, gpt-5.5
  cropped noisy. Hero cards are tiny + WIN-overlay-occluded = fine-grained perception, not VLM's
  strength. → **CardCNN rank-head retrain** (`retrain-card-classifier` skill) + fix
  `_locate_hero_cards` duplicates/mislocations (e.g. parsed `9c9c`). This is the next big lever.

**(2) RECALL — deterministic ceiling 88.7% (parsable 637/718), need 95%.**
- parse_none = 81. `force_table_size` alone recovers only **15/81** (rest fail for missing-raise-
  size / no usable action rows). Full recovery needs a **full re-parse** = the **existing Gemini
  fallback** (`src/gemini_session.py::_parse_hand_from_image`, which already routes parse_none →
  `IMAGE_PARSE_PROMPT` full Gemini parse in production). So **production recall is already higher
  than this deterministic-only eval**; the eval undercounts it. Quantify production recall by
  running the bot's full path on parse_none hands, OR extend the eval to invoke the Gemini
  fallback on parse_none.

**Honest math:** 99%@95% needs hand_exact to rise 597→≥682 (fix ~85 more: 28 hero_cards + ~57
parse_none). It is a *very* aggressive target — even frontier VLMs err on the genuinely-ambiguous
hard hands. Realistic intermediate wins are clear (below).

---

## Recommended next steps (priority order)

### 1. Hero-card CardCNN retrain (precision lever) — **start here**
- Use the **`retrain-card-classifier`** skill (retrains rank/suit CNN on the `analysis_snapshots`
  corpus). Target the **rank head** — "4" over-prediction is the dominant confusion.
- Also fix **localization**: 3 of 30 are duplicate cards (`9c9c`) and ~13 are full mislocations →
  inspect `_locate_hero_cards` / `_trim_above_card_edge` in `table_parser.py`.
- Levers NOT yet tried (handoff §C4): hard-negative mining on the confident-wrong hero cards,
  focal loss on confused rank pairs, distillation from the larger board crops, test-time aug.
- **Don't** retry overlay-augmentation (regressed twice before — see parent plan §"Don't repeat").
- Validate: re-dump test, check hero_cards_wrong drops without regressing correct hands.

### 2. Recall via Gemini fallback for parse_none (recall lever)
- The fallback already exists in production. To make the **eval** honest, add a mode to
  `ocr_precision.py` (or a wrapper) that sends parse_none hands through
  `gemini_session._parse_hand_from_image` and counts recovered-correct.
- Measure true production recall; if <95%, improve the Gemini parse prompt for the hard
  (collapsed-row / multi-all-in) hands.

### 3. Broaden the structural trigger (small precision win) — task left open
- `OCR_VLM_RECHECK_TRIGGER=allin` catches 82% of structural errors (3–4 non-all-in slip through).
  Add `reaction` (estimate_used_reaction_signal) to the trigger, or use `all` for full coverage
  (override is *safe* on correct hands — 30/30 preserved — so trigger is a pure latency knob).
- Optional: implement `force_hero_position` re-anchoring to convert the **14 structural abstains**
  (currently → parse_none) into corrections (recovers ~14 recall, all correct). Fiddly: re-align
  action-entry→position offset. Deferred as risky; abstain is the safe default.

---

## How to run things (verified commands)

**Re-dump test bucket with the fixes + structural override (real flash, ~13 min):**
```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OCR_VLM_RECHECK=1 OCR_VLM_RECHECK_TRIGGER=allin
python scripts/ocr_precision.py --bucket test --split data/splits/card_classifier_v2.json \
  --dump-all --workers 4 --out data/ocr_precision_phase11X_test
```
Board fix is always-on (no flag). Re-dump train/val too (same, `--bucket train|val`) only if
retraining the calibrator. Full 7183-hand dump ≈ 55 min at 4 workers.

**Eval precision@coverage (honest recall over 718):**
```bash
python scripts/_calibrate_v2.py --calibrator-dir data/calibrator \
  --pokercraft-test data/ocr_precision_phase11X_test/all_records.jsonl \
  --pokercraft-dev  data/ocr_precision_phase11d_pokercraft_train/all_records.jsonl \
  --production-test data/ocr_precision_phase11d_production_test/all_records.jsonl \
  --features data/calibrator/v3_features.txt --feature-key v3_features --model-suffix v3 \
  --out /tmp/eval.json
```

**Tests** (33 new, all green; full suite has 2 pre-existing unrelated failures):
```bash
python -m pytest tests/ocr -q
```

---

## VLM model bench (decisive findings — don't re-litigate)

Tested on the same error+correct hands (position accuracy; latency at default config):

| model | err-fix | correct-safe | latency | verdict |
|---|---|---|---|---|
| **gemini-3.5-flash + focused prompt** | **100%** | **100%** | **~6–8s** | ✅ **PRODUCTION PICK** |
| gpt-5.5 (medium reasoning) | 100% | 100% | ~80–100s | accurate, too slow |
| gpt-5.5 (low) | 80% | 100% | ~28s | — |
| gpt-5.2 / gpt-5.4-mini | low | low | slow | ✗ |
| o4-mini | 70% | 30% | 57s | noisy ✗ |
| gemini-3.1-pro-preview | 70% | 40% | 62s | noisy ✗ |
| gemini-pro-latest | 91% | 75% | ~20s | noisy ✗ |
| gemini-2.5-flash (full prompt) | 2/10 | — | — | useless ✗ |

**Keys to the win:** (a) the newest **3.5-flash** (2.5-flash was useless), (b) a **FOCUSED prompt**
asking *only* for seat count + hero position (not a full re-parse) — cut latency 29s→8s AND raised
accuracy. Prompt text is `FOCUSED_PROMPT` in `scripts/ocr/vlm_recheck.py`.

---

## Gotchas / notes

- Calibrator schema (`data/calibrator/v2_features.txt`, `v3_features.txt`) and models
  (`*_v3.joblib`) live under the **gitignored `data/`** symlink — NOT in the PR. Tests read the
  schemas from local disk. Regenerate models via `train_calibrator_v2` (see parent plan).
- `_calibrate_v2.py` (eval) and the many `scripts/_*.py` (probes) are local/scratch, uncommitted.
- The VLM re-check is **flag-gated OFF by default** — turning it on in production adds ~6–8s on
  the all-in subset (~36% of hands). Consider async (fast OCR reply + flash correction follow-up)
  if synchronous latency matters; user flagged TG-bot latency as a constraint.
- Memory: `~/.claude/projects/-home-harry-ai-poker-wizard/memory/ocr-99-d-a-calibrator-wall.md`
  has the full investigation trail.
