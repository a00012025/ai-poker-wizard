# Post-PR Investigation — Can we push effbb past 78.2%? (No.)

Branch: `feat/effbb-precision`. Measured 2026-06-13 over
`data/effbb_cache/cache.jsonl`, hero-active emitted population (n=1270, baseline
**78.19% @ 70.4%**, 993/1270). All experiments are strict cache-only A/B
(no re-OCR) via `scripts/_tmp.py` — each isolates a single change and reports
`fixed / broke` vs the production reconstruction.

## TL;DR

The effbb hero-active frontier is **input-bound, not logic-bound.** Eight
distinct interventions — including the plan's chip-constraint-solver idea taken
all the way to a per-street conservation *repair* — were each **net-negative**.
Decision: **stop.** effbb stays at 78.19%; no behavior change ships from this
investigation. The real win from this line of work was Phase A per-node depth
(open-node 63.7%→98.6%, facing 70.7%→98.9%) — a coaching-correctness gain, not a
scalar-effbb gain.

## Fault decomposition (the 277 emitted-wrong hero-active hands)

| fault | n | nature | conservation can repair? |
|---|---|---|---|
| near | 113 | adjacent depth bucket | no (needs soft-bucketing) |
| selection | 85 | wrong/missed binding villain (**84/85 multiway**) | no (displayed-stack side, rank-deficient) |
| undershoot | 76 | emitted far below GT (dropped/under-read investment) | partially (investment side) |
| impossible_over | 3 | over-added contribution | yes (investment side) |

Key structural fact: **`collapse-to-hero` (emitted == hero_start bucket) is the
DOMINANT CORRECT pattern** — 425 of 545 multiway-collapse emissions are correct,
because in a multiway pot hero genuinely is the effective-short most of the time.
So "collapse" cannot be used as an error signal, and the often-quoted "+6.7pp
selection ceiling" is an **oracle** ceiling: it requires GT to know which 85 of
the 545 are wrong. Nothing in the current parse separates them.

## The eight experiments

| # | intervention | fixed | broke | net |
|---|---|---|---|---|
| 1 | engine relevant-set as PRIMARY selector (not advisory) | — | — | recovers only **4/85** selection |
| 2 | min over all visible non-hero stickers | +28 | −731 | **−703** |
| 3 | min over engine relevant-set raw stickers | +37 | −255 | **−218** |
| 4 | targeted abstain (multiway & collapse & shorter-visible, several variants) | — | — | drops **3–5 correct per 1 wrong** |
| 5 | chip-conservation as a discriminator | — | — | **zero separation** (flags 99% of BOTH correct & wrong "inconsistent") |
| 6 | Phase B single-field contribution repair | +0 | −1 | **−1** |
| 7 | per-street conservation repair (both directions) | +24 | −96 | **−72** |
| 8 | per-street conservation repair (upward-only, undershoot-targeted) | +20 | −41 | **−21** |

## Why the chip-constraint solver cannot fix effbb

The plan's D3 instinct — "use the conservation constraint to correct uncertain
numbers" — was tested to its conclusion with the **correct granularity** (the
shipped Phase B compared END-of-hand contributions to a preflop-START header; we
fixed that to per-street equations `Σ street-contrib == pot_delta`). It still
fails, for two reasons:

1. **Rank-deficiency (selection).** Selection errors live on the *displayed
   remaining stack* side (which sticker is the binding short villain, vs a felt
   chip phantom). A single snapshot gives one pot equation against many unknown
   stacks — the conservation system cannot pin an individual villain's start.
   Empirically: promoting the engine's (reliable) action-order opponent
   selection to primary recovers only 4/85.

2. **Pot-header OCR noise (undershoot).** 58 of 76 undershoot hands DO carry a
   conservation residual >1bb — but the per-street repair fixed only **2** of
   them, because the residual reflects a *pot misread*, not a localizable
   contribution error. Repairing on that noisy signal corrupts more correct
   hands than it fixes (experiments 7–8). This is the same data limitation that
   gave Phase B's `pot_residual` feature "poor discrimination at 46.7%
   precision" and kept the Phase B gate from firing.

## The only remaining lever (not pursued)

Adding **new information**, gated to fire ONLY on low-confidence hands so it
never touches the ~78% already correct: a targeted VLM / re-read adjudication of
the binding multiway seat ("what is the shortest opponent's stack?"). This is
the one path that can raise precision without sacrificing coverage. It is more
expensive (re-reads the image) and was deliberately **not** taken — per the
decision to stop, effbb ROI is now below the cost. Note that *uniform* cleaner
OCR does not help (Phase C avatar-anchored reads regressed effbb −1.14pp); the
information must be targeted at the specific multiway phantom/short-villain
ambiguity, not the read quality in general.

## Net

Zero behavior change. This investigation's value is the **negative result
itself**: it closes the chip-solver-as-repair direction with measured evidence
(preventing a future re-attempt) and reframes the 78.2% frontier as input-bound.
The shippable wins of PR #68 are unchanged — Phase A per-node depth (the real
improvement), with Phase B (chip solver, feature-only) and Phase C
(avatar-anchored OCR, flag OFF) landing as tested-but-inactive infrastructure.
