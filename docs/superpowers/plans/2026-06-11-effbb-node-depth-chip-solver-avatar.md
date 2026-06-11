# Effective-Stack Precision Roadmap: Per-Node Depth + Chip Constraint Solver + Avatar-Anchored Seats

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push effective-stack correctness past the current 78.2% @ 70.4% frontier by (A) querying the GTO solver at the *right depth per hero decision node* with honest caveats, (B) replacing investment guessing with a deterministic poker-rules chip solver that exploits pot-equation redundancy, and (C) replacing scan-everything seat OCR with avatar-anchored per-seat reading.

**Architecture:** Three independently landable phases, in dependency order. Phase A changes *what we ask the solver* (analysis layer, `analyze_hand.py`); Phase B changes *how invested chips are reconstructed* (pure logic over already-cached inputs, `scripts/ocr/`); Phase C changes *how seat values enter the pipeline* (vision layer, `table_parser.py`). A and B iterate on the existing `data/effbb_cache/cache.jsonl` in seconds; only C requires a cache rebuild.

**Tech Stack:** Python 3, existing GTO Wizard API client (`scripts/gto_api.py`), existing betting engine (`scripts/ocr/effbb_engine.py`), OpenCV (already a dependency via the OCR stack), EasyOCR (existing), no new external dependencies.

---

## 0. Locked Architecture Decisions (read before implementing anything)

These decisions are **fixed**. Implementation must not re-litigate them. Each was
measured or derived during the 2026-06-11 dissection session (see
`docs/superpowers/plans/2026-06-11-effbb-99-5-architecture.md` Phase 6 and the
memory note `effbb-86-ceiling-was-wrong`).

### D1. Per-node effective stack semantics (Phase A)

A hand no longer has ONE effective depth. Each **hero decision node** gets its own:

| Node type | Effective stack definition | Why |
|---|---|---|
| Open / first voluntary action | `min(hero_start, max(live_cover))` where `live_cover` = stacks of opponents not yet folded at hero's action (incl. players still to act) | Hero's open targets everyone behind; you can't play deeper than the deepest live opponent. CO 30bb opens with BTN 30bb behind → 30bb tree, even if SB has 17bb. |
| Facing a raise / 3-bet / 4-bet / jam (continuation node) | `min(hero_start, aggressor_total_commitment)` for the LAST aggressor hero faces. For an all-in, total commitment = the jam size + the jammer's prior street investment (already computed by `effbb_engine` M1 logic). | The call/fold decision is against exactly that stack. SB jams 17bb over hero's 30bb open → this node queries the 17bb tree. |
| Postflop streets | UNCHANGED — existing SPR-matched compression (`analyze_hand.py:959-990`) stays as-is. | Already shipped and tested (PR #45 lineage). |

**Range-mismatch caveat (D1a):** whenever two consecutive hero nodes resolve to
*different solver depth buckets*, the deeper-context node carries a caveat that
must reach the user: the queried tree assumes hero arrived with the
*shallow-tree* range, but hero actually arrived with the *deep-tree* range.
Caveat text (exact, Chinese-canonical per project terminology rules):

```
⚠ 此節點以 {node_bb}bb 樹查詢（你前一個決策是在 {prev_bb}bb 樹做的）；
solver 假設你帶著 {node_bb}bb 的範圍到達此節點，範圍銜接會有偏差，數據供參考。
```

**No-data honesty (D1b):** query the node at `nearest_depth(node_eff)`. If the
API returns no node (existing 204/empty-`available_actions` handling), emit the
existing `（無 solver 數據）` marker for that node. NEVER silently substitute a
different depth for a *facing* node. (Depth escalation stays allowed for
postflop streets only — that is existing behavior.)

**Scope guard (D1c):** Phase A applies to **chip-EV gametypes only**
(`MTTGeneral`, `Cash*`). ICM hands keep their current single
`find_icm_params()` depth — ICM depth is a stack-config lookup, not a free
parameter, and per-node ICM trees mostly don't exist. The per-node resolver
returns `None` for ICM hands and all call sites fall through to current
behavior.

**What D1 replaces (D1d):** the global all-in override at
`scripts/analyze_hand.py:1250-1254` (`_preflop_allin_effective_bb` overwriting
`hand["effective_bb"]` + `depth` for the WHOLE hand) is **deleted** and replaced
by per-node resolution. The open-node hero-stack-depth special case at
`analyze_hand.py:1404-1431` (`if not is_icm and not allin_effective:` ...) loses
its `not allin_effective` condition — the open node ALWAYS uses its own D1
definition now. `_preflop_allin_effective_bb` itself is kept (renamed into the
resolver) because its jam-size extraction is correct — only its *application*
was wrong.

### D2. Per-node ground truth (Phase A)

`scripts/hh_parser.py` gains a pure function `node_effectives(...)` that derives,
from the HH (exact chips), the D1-defined effective for every hero preflop
decision node. `scripts/build_ground_truth.py` writes it as a new GT field:

```json
"node_effective_bb": [
  {"node": "open",   "eff": 30.2, "depth": 30},
  {"node": "facing_allin", "aggressor": "SB", "eff": 17.4, "depth": 17}
]
```

The existing scalar `effective_bb` field is UNCHANGED (downstream consumers and
the 78.2% metric keep working). Per-node accuracy becomes a new, additional
metric in `effbb_eval.py`.

### D3. Chip constraint solver shape (Phase B)

A new pure module `scripts/ocr/chip_solver.py`. It does **not** replace
`effbb_engine.py` — it consumes the engine's `assigned` actions + contributions
and adds the redundancy check the engine lacks:

- **Inputs:** engine `EngineResult` (positions, per-position contributions, sb/bb/ante),
  per-street pot headers from the panel, hero displayed stack, seat reads.
- **Equations:** for each street boundary with a pot header `P_s`:
  `P_s = Σ contributions(through street s-1) + blinds + antes` (tolerance ±0.25bb
  plus 2% relative for unit jitter).
- **Output:** `ChipCheck` dataclass:
  - `consistent: bool` — all available equations within tolerance
  - `residuals: dict[street, float]` — signed residual per equation
  - `repair: dict | None` — if exactly ONE single-field change (one action size,
    one ante value, or one missing blind) makes ALL equations consistent, name
    it: `{"field": "preflop[3].size", "from": 1.0, "to": 10.0}`. If zero or
    multiple repairs exist → `None`. **Single-field repair only — this is not a
    general CSP** (YAGNI; multi-field search explodes and was the failure mode
    of every "clever" lever rejected in Phase 6).
- **Integration rule (D3a):** the solver NEVER directly changes the emitted
  value. It feeds `_compute_effective_bb` two things only:
  1. `consistent=True` → confidence nudge `+0.03` (mirrors the existing
     `x_agree` nudge);
  2. `consistent=False and repair is None` → structural-abstain input (a new
     gate clause, measured before enabling — see Task B6's marginal-precision
     gate audit, same methodology as `scripts/_tmp_gate.py`).
  A found `repair` is logged into the features (`_LAST_EFFBB_FEATURES`) for
  calibration but NOT auto-applied in this phase. Auto-apply is a separate
  future decision after corpus measurement.
- **Env flag:** `OCR_EFFBB_CHIP_SOLVER` (default ON after the Task B6 audit
  passes; `=0` reverts cleanly).

### D4. Avatar-anchored seat reading (Phase C)

New detection path: **find the people first, then read their numbers.**

- `scripts/ocr/seat_detector.py` — finds avatar anchors. Decision: **classical
  CV first** (N8 avatars are fixed-size circular discs at table-size-specific
  layout positions; Hough circle transform + the known per-table-size layout
  prior), NO neural detector in this phase. A CNN detector is a fallback
  decision only if classical recall measures < 97% on the auto-label harness.
- `scripts/ocr/seat_reader.py` — for each anchor, crops the nameplate ROI and
  stack ROI at **fixed offsets relative to the avatar** (N8 renders name above
  stack in a fixed-size plate directly under the avatar disc), runs EasyOCR
  with `allowlist="0123456789.,KM"` on the stack ROI. Bounty badges (the `$`
  pill left/right of the avatar) are excluded *by construction* — their ROI is
  never read. No new digit model in this phase (decision: constrained-ROI
  EasyOCR first; a dedicated digit CNN is a follow-up only if the harness shows
  digit errors > 2% inside correctly-cropped ROIs).
- **Auto-label harness (D4a)** — the key de-risking idea: for every corpus image
  we have the HH. The HH yields each seat's **expected displayed stack** at
  screenshot time (`starting − invested`, both exact from the HH). The harness
  matches detector output against expected values + player names → per-seat
  labels and a pipeline scorecard with zero manual labeling.
- **Panel⊆table invariant (D4c):** every actor in the action panel sits at a
  detected seat, but a detected seat may have NO panel action (sit-out /
  not-yet-dealt players). Therefore: avatar count is the authoritative
  physical-seat count (improves `_infer_num_players`' ring fallback), and the
  seat→position mapping must remain injective panel→seats — never force every
  seat to receive a position.
- **Output contract:** `seat_reader` produces the SAME `named_stacks` schema
  (`{"name": str, "stack": float, "x": float, "y": float}`) plus
  `"anchor_conf": float` per row, behind env flag `OCR_SEAT_ANCHORED` (default
  OFF until Task C7's corpus A/B). Downstream (`_seat_ring`,
  `_enumerate_layouts`, `_compute_effective_bb`) is UNTOUCHED in this phase —
  the win comes purely from cleaner inputs.
- **Cache contract (D4b):** flipping `OCR_SEAT_ANCHORED=1` changes
  `_compute_effective_bb` inputs ⇒ requires a full `EFFBB_CAPTURE=1` corpus
  re-parse (~3h) to rebuild `data/effbb_cache/cache.jsonl` before quoting any
  effbb numbers. Phases A and B must NOT rebuild the cache.

### D5. Phase ordering, landing, and metrics

- Each phase is its own worktree + branch + PR: `feat/node-depth-analysis`,
  `feat/chip-solver`, `feat/seat-anchored-ocr`. A later phase must not start
  until the previous PR is merged (they touch adjacent code).
- Frozen metrics, measured before/after every phase:
  1. scalar effbb: `python scripts/effbb_eval.py` — hero-active emitted
     precision/coverage (baseline at plan time: **78.19% @ 70.4%**, correct=993)
  2. per-node (after A): new `--per-node` mode of the same script
  3. regression suite: `python scripts/regression_test.py` — 626 passed at plan
     time; the suite must stay green and every fix adds a test (project rule)
- Hard gates: a phase may not land if scalar effbb precision drops > 0.3pp or
  coverage drops > 1pp without an explicitly measured, documented win
  elsewhere. (This caught and killed 3 of 6 levers in Phase 6 — keep the
  discipline: **measure net fixed/broken with an A/B script before committing
  any behavior change**; `scripts/_tmp_breaks.py` from the session shows the
  pattern: save a `--save ref.json` snapshot under the old code, diff under the
  new.)

### D6. Known traps (encode these, do not rediscover them)

1. **Matched preflop jam = board run-out** ⇒ ground truth switches to the
   postflop definition (active players only). Any "seats behind hero bind"
   logic must check for this (see `effbb_engine.analyze` `_any_pf_jam` guard).
2. **Engine-correlation kills the dissent gate.** Any change that makes the
   legacy estimate consume the same engine read that `_engine_relevant_bucket`
   uses will inflate agreement and emit garbage (measured: +44 abstain→wrong).
   The chip solver is safe because it consumes pot headers (a signal the engine
   bucket does not use), but verify independence in the Task B6 audit.
3. **Panel call sizes are increments, not totals**; raise/bet/all-in sizes are
   "to" amounts. Hero rows are unnamed and must be treated as one actor stream.
   Blinds column is often empty — BB/SB posted chips must be credited.
4. **`hero_folded_preflop` mis-indexes when preflop_parts is shorter than the
   position order** (non-acting seats). Per-node GT (Task A2) must replay
   actions the way `hh_parser` builds them (first round position-ordered,
   continuation rounds cycling non-folded players).
5. Snapshot suite has 16 pre-existing stale-expected failures (Call→Limp
   renames etc.) — unrelated to this work; do not "fix" them in these PRs.

---

## Phase A — Per-Node Depth Analysis (`feat/node-depth-analysis`)

**Files:**
- Create: `scripts/node_depth.py`
- Modify: `scripts/hh_parser.py` (add `node_effectives`)
- Modify: `scripts/build_ground_truth.py` (emit `node_effective_bb`)
- Modify: `scripts/analyze_hand.py:1250-1254, 1404-1431, 1470-1560` (consume resolver)
- Modify: `scripts/gto_formatter.py` (render caveat line)
- Modify: `scripts/effbb_eval.py` (`--per-node` metric)
- Test: `scripts/regression_test.py` (new `@test` functions)

### Task A1: `scripts/node_depth.py` — the pure resolver

- [ ] **Step 1: Write the failing tests** (append to `scripts/regression_test.py`)

```python
@test
def test_node_depth_open_uses_max_live_cover():
    """D1: the open node plays vs the deepest live opponent, not the shortest
    seat behind. CO 30bb opens, BTN 30bb behind, SB 17bb behind -> open node
    is 30bb; the 17bb stack does NOT shallow the open."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        # position -> starting stack where known (None = unknown)
        stacks={"UTG": 42.0, "HJ": 25.0, "CO": 30.0, "BTN": 30.0,
                "SB": 17.0, "BB": 51.0},
        is_icm=False,
    )
    open_node = nodes[0]
    assert_eq(open_node["node"], "open")
    assert_eq(open_node["eff"], 30.0)
    assert_eq(open_node["depth_bucket"], 30)


@test
def test_node_depth_facing_allin_uses_jammer_commitment():
    """D1: the facing-all-in node queries min(hero, jam total). SB jams 17
    over hero CO's 30bb open -> facing node is 17bb with a range-mismatch
    caveat naming both depths."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "BTN": 30.0, "SB": 17.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"]
    assert_eq(len(facing), 1)
    assert_eq(facing[0]["eff"], 17.0)
    assert_eq(facing[0]["depth_bucket"], 17)
    assert_true(facing[0]["caveat"] is not None
                and "17" in facing[0]["caveat"] and "30" in facing[0]["caveat"],
                f"caveat must name both depths: {facing[0]['caveat']}")


@test
def test_node_depth_same_bucket_no_caveat():
    """No caveat when consecutive nodes land in the SAME depth bucket —
    don't spam the user with a meaningless warning."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI29.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "SB": 29.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"][0]
    assert_eq(facing["depth_bucket"], 30)
    assert_true(facing["caveat"] is None, "same-bucket node must carry no caveat")


@test
def test_node_depth_icm_returns_none():
    """D1c: ICM hands keep the single find_icm_params depth — resolver opts out."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0, stacks={}, is_icm=True,
    )
    assert_true(nodes is None, "ICM must opt out of per-node depths")
```

- [ ] **Step 2: Run tests, verify they fail** —
`python scripts/regression_test.py 2>&1 | grep -E "node_depth|FAIL"` →
expect 4 FAILs with `No module named 'node_depth'`.

- [ ] **Step 3: Implement `scripts/node_depth.py`**

```python
"""Per-hero-decision-node effective stack resolution (chip-EV only).

D1 semantics (docs/superpowers/plans/2026-06-11-effbb-node-depth-chip-solver-avatar.md):
  * open node          : min(hero_start, max live cover at hero's action)
  * facing-raise/jam   : min(hero_start, last aggressor's total commitment)
ICM hands opt out (return None) — their depth is a stack-config lookup.

Pure module: no OCR, no API, no analyze_hand imports (gto_api only for the
depth tables). preflop_actions uses the project's position-ordered code string
("F-R2.0-AI17.0-...", continuation codes appended after the first N).
"""
from gto_api import nearest_depth

_CAVEAT = ("⚠ 此節點以 {node}bb 樹查詢（你前一個決策是在 {prev}bb 樹做的）；"
           "solver 假設你帶著 {node}bb 的範圍到達此節點，範圍銜接會有偏差，"
           "數據供參考。")


def _replay(parts, order):
    """Yield (actor_index, code, prefix_parts) in action order, replaying the
    project's encoding: first len(order) parts position-ordered, continuation
    parts cycling the non-folded actors."""
    n = len(order)
    active = [i for i in range(min(n, len(parts))) if parts[i] not in ("F", "")]
    for i, code in enumerate(parts[:n]):
        yield i, code, parts[:i]
    ci = 0
    prefix = list(parts[:n])
    for code in parts[n:]:
        if not active:
            break
        ci %= len(active)
        actor = active[ci]
        yield actor, code, list(prefix)
        prefix.append(code)
        if code == "F":
            active.pop(ci)
        else:
            ci += 1


def _jam_total(code, stacks, order, actor_idx):
    """A preflop AI's total commitment ~= its size (panel/HH sizes are
    cumulative 'to' amounts preflop). Falls back to the actor's stack."""
    raw = code[2:]
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def resolve_preflop_nodes(*, preflop_actions, hero_position, position_order,
                          hero_start, stacks, is_icm):
    """Return [ {node, eff, depth_bucket, caveat, aggressor_code} ] for every
    hero preflop decision node, or None when the resolver opts out (ICM /
    hero unknown / no hero action)."""
    if is_icm or not hero_start or hero_position not in position_order:
        return None
    parts = (preflop_actions or "").split("-")
    hidx = position_order.index(hero_position)
    if hidx >= len(parts) or parts[hidx] in ("", "F"):
        return None

    nodes = []
    # ---- open node ----
    live_cover = [
        (stacks or {}).get(p)
        for i, p in enumerate(position_order)
        if p != hero_position
        and (i > hidx or (i < len(parts) and parts[i] not in ("F", "")))
        and (stacks or {}).get(p)
    ]
    open_eff = min(hero_start, max(live_cover)) if live_cover else hero_start
    nodes.append({
        "node": "open", "eff": round(open_eff, 1),
        "depth_bucket": int(nearest_depth(open_eff)),
        "caveat": None, "aggressor_code": None,
    })

    # ---- facing nodes: every action by another player AFTER hero's first
    # voluntary action that raises the level (R/AI) creates a hero decision ----
    hero_acted = False
    pending_aggr = None     # (code, actor_idx) of the latest raise hero faces
    for actor, code, prefix in _replay(parts, position_order):
        if actor == hidx:
            if hero_acted and pending_aggr is not None:
                a_code, a_idx = pending_aggr
                if a_code.startswith("AI"):
                    total = _jam_total(a_code, stacks, position_order, a_idx)
                    kind = "facing_allin"
                else:
                    total = (stacks or {}).get(position_order[a_idx])
                    kind = "facing_raise"
                if total:
                    eff = round(min(hero_start, total), 1)
                    bucket = int(nearest_depth(eff))
                    prev_bucket = nodes[-1]["depth_bucket"]
                    caveat = (_CAVEAT.format(node=bucket, prev=prev_bucket)
                              if bucket != prev_bucket else None)
                    nodes.append({
                        "node": kind, "eff": eff, "depth_bucket": bucket,
                        "caveat": caveat, "aggressor_code": a_code,
                    })
                pending_aggr = None
            hero_acted = True
        elif hero_acted and (code.startswith("R") or code.startswith("AI")):
            pending_aggr = (code, actor)
    # An aggression hero never answered (parser dropped hero's response) still
    # defines a facing node — analyze_hand surfaces those (H3428 behavior).
    if pending_aggr is not None:
        a_code, a_idx = pending_aggr
        if a_code.startswith("AI"):
            total = _jam_total(a_code, stacks, position_order, a_idx)
            if total:
                eff = round(min(hero_start, total), 1)
                bucket = int(nearest_depth(eff))
                prev_bucket = nodes[-1]["depth_bucket"]
                nodes.append({
                    "node": "facing_allin", "eff": eff, "depth_bucket": bucket,
                    "caveat": (_CAVEAT.format(node=bucket, prev=prev_bucket)
                               if bucket != prev_bucket else None),
                    "aggressor_code": a_code,
                })
    return nodes
```

- [ ] **Step 4: Run the 4 tests, verify they pass** —
`python scripts/regression_test.py 2>&1 | grep -E "node_depth"` → 4 PASS.

- [ ] **Step 5: Commit** — `git add scripts/node_depth.py scripts/regression_test.py && git commit -m "feat(node-depth): pure per-node preflop effective resolver (D1 semantics)"`

### Task A2: per-node ground truth from the HH

- [ ] **Step 1: Failing test** (append to `scripts/regression_test.py`)

```python
@test
def test_hh_node_effectives_open_vs_facing():
    """D2: hh_parser.node_effectives derives per-node effectives from exact HH
    chips. Build a synthetic HH where hero CO (9000 chips, bb=300 -> 30bb)
    opens, BTN (9000) folds, SB (5100 -> 17bb) jams, hero calls: open node
    30bb, facing node 17bb."""
    from hh_parser import node_effectives
    nodes = node_effectives(
        positions=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        pos_to_chips={"UTG": 12000, "HJ": 8000, "CO": 9000, "BTN": 9000,
                      "SB": 5100, "BB": 15000},
        preflop_actions_ordered=[("UTG", "F"), ("HJ", "F"), ("CO", "R2.0"),
                                 ("BTN", "F"), ("SB", "AI17.0"), ("BB", "F"),
                                 ("CO", "C")],
        hero_position="CO", bb_size=300,
    )
    assert_eq(nodes[0]["node"], "open")
    assert_eq(nodes[0]["eff"], 30.0)
    facing = [n for n in nodes if n["node"].startswith("facing")][0]
    assert_eq(facing["eff"], 17.0)
```

- [ ] **Step 2: Run, verify FAIL** (`node_effectives` not defined).

- [ ] **Step 3: Implement `node_effectives` in `scripts/hh_parser.py`**

Place directly above `parse_hand`. It re-uses D1 semantics but from
`preflop_actions_ordered` (the HH's exact play-order list, already built at
`hh_parser.py:213-236`), so trap D6.4 (index drift) cannot occur:

```python
def node_effectives(*, positions, pos_to_chips, preflop_actions_ordered,
                    hero_position, bb_size):
    """Per-hero-decision-node effective stacks (D1 semantics), exact from HH.

    open node   : min(hero, max stack of opponents live at hero's first action)
    facing node : min(hero, the facing aggressor's total commitment); for an
                  AI code the size IS the total (HH 'raises X to Y' all-in).
    Returns a list of {node, eff, depth} dicts (eff in bb, 1 decimal).
    """
    from gto_api import nearest_depth
    hero = pos_to_chips.get(hero_position)
    if not hero or not bb_size:
        return []
    folded = set()
    hero_acted = False
    pending = None  # (code, pos)
    nodes = []
    for pos, code in preflop_actions_ordered:
        if pos == hero_position:
            if not hero_acted and code != "F":
                live = [pos_to_chips[p] for i, p in enumerate(positions)
                        if p != hero_position and p not in folded
                        and pos_to_chips.get(p)]
                eff = min(hero, max(live)) / bb_size if live else hero / bb_size
                nodes.append({"node": "open", "eff": round(eff, 1),
                              "depth": int(nearest_depth(eff))})
            elif hero_acted and pending is not None:
                a_code, a_pos = pending
                if a_code.startswith("AI") and len(a_code) > 2:
                    total_bb = float(a_code[2:])
                else:
                    total_bb = (pos_to_chips.get(a_pos) or 0) / bb_size
                if total_bb > 0:
                    eff = min(hero / bb_size, total_bb)
                    nodes.append({
                        "node": ("facing_allin" if a_code.startswith("AI")
                                 else "facing_raise"),
                        "eff": round(eff, 1), "depth": int(nearest_depth(eff)),
                    })
                pending = None
            hero_acted = True
        else:
            if code == "F":
                folded.add(pos)
            elif hero_acted and (code.startswith("R") or code.startswith("AI")):
                pending = (code, pos)
    return nodes
```

- [ ] **Step 4: Run test → PASS. Commit** — `git commit -m "feat(node-depth): exact per-node effectives from HH (D2 ground truth)"`

- [ ] **Step 5: Wire into `build_ground_truth.py`** — inside `parse_hand`
(`scripts/hh_parser.py`, after `effective_bb` is computed ~line 290), call
`node_effectives(...)` with the already-built locals and add
`"node_effective_bb": nodes` to the returned dict (~line 364). Rebuild GT:
`python scripts/build_ground_truth.py data/pokercraft_corpus -o data/pokercraft_corpus/ground_truth`
then spot-check:
`python -c "import json; r=[json.loads(l) for l in open('data/pokercraft_corpus/ground_truth/ground_truth.jsonl')]; w=[x for x in r if len((x['ground_truth'].get('node_effective_bb') or []))>1]; print(len(w), w[0]['ground_truth']['node_effective_bb'])"`
Expected: hundreds of multi-node hands; first sample shows distinct open/facing depths.

- [ ] **Step 6: Commit** — `git commit -m "feat(node-depth): emit node_effective_bb in ground truth"`

### Task A3: consume the resolver in `analyze_hand.py`

- [ ] **Step 1: Failing test** — cache-driven, the user's exact scenario:

```python
@test
def test_analyze_per_node_depths_split():
    """The open spot queries the deep tree; the facing-all-in spot queries the
    jam-depth tree with a range-mismatch caveat — replacing the old global
    allin_effective override that dragged the WHOLE hand to jam depth."""
    from analyze_hand import _build_hero_spot_depths   # new pure helper
    hand = {
        "effective_bb": 30.0, "hero_starting_stack": 30.0,
        "hero_position": "CO", "players_at_table": 6,
        "preflop_actions": "F-F-R2.0-F-AI17.0-F-C",
        "player_stacks": [42.0, 25.0, 30.0, 30.0, 17.0, 51.0],
    }
    spots = _build_hero_spot_depths(hand, is_icm=False, is_cash=False)
    assert_eq(spots["open"]["depth"], "30.125")
    assert_eq(spots["facing"]["depth"], "17.125")
    assert_true(spots["facing"]["caveat"] is not None)
```

- [ ] **Step 2: Implement.** In `scripts/analyze_hand.py`:
  1. Add `_build_hero_spot_depths(hand, *, is_icm, is_cash)` — a thin adapter
     that maps `hand` fields to `node_depth.resolve_preflop_nodes(...)` args
     (position order from `_get_position_order(num_players)`, stacks from
     `player_stacks` zipped with the order when lengths match) and converts
     each node's `depth_bucket` to the API string format via the existing
     `f"{bucket}.125"` convention (use `nearest_cash_depth` for cash).
  2. Delete the global override at `:1250-1254`; keep the
     `allin_effective` computation only as the resolver's fallback when
     `player_stacks` is unavailable.
  3. Open spot (`hero_spots[0]`, built at `:1459-1467`): set
     `params["depth"]` from the resolver's open node; drop the
     `not allin_effective` condition at `:1412`.
  4. Continuation/facing spots (`:1518-1531` and the H3428 block): set
     `params["depth"]` from the matching facing node; re-normalize the
     prefix for that depth (raise codes are depth-specific — call
     `_normalize_preflop_actions(raw_prefix, gametype, node_depth)` exactly as
     the open node does at `:1425-1430`); attach `spot["depth_caveat"] =
     node["caveat"]`.
  5. When the per-node query 204s, the existing no-data path renders
     `（無 solver 數據）` — verify, do not re-implement (D1b).
- [ ] **Step 3: Render the caveat.** In `scripts/gto_formatter.py`, where a
  hero spot's header/`action_desc` is rendered, append `spot["depth_caveat"]`
  on its own line when present. Add a regression test asserting the caveat
  line appears in the formatted output for the Task A3 hand and does NOT
  appear when buckets match.
- [ ] **Step 4: Full suite** — `python scripts/regression_test.py` → must stay
  green (626+new). Several existing tests assert the OLD global-override
  behavior (e.g. any test asserting whole-hand depth == jam depth) — update
  those tests to the per-node expectation and say so in the commit message.
- [ ] **Step 5: E2E sanity** —
  `python scripts/e2e_test.py "有效 30bb，CO open 2bb，BTN fold，SB 17bb all-in，hero call，hero hand AhKh"`
  Expected: output contains an open section at 30bb, a facing-all-in section at
  17bb, and the caveat line.
- [ ] **Step 6: Commit** — `git commit -m "feat(node-depth): per-node solver depths in analyze_hand + caveat rendering (replaces global allin override)"`

### Task A4: per-node metric + land

- [ ] **Step 1:** Add `--per-node` to `scripts/effbb_eval.py`: for hero-active
  cache rows whose GT has `node_effective_bb`, run the parser's
  `_compute_effective_bb` → `hand` fields → `node_depth.resolve_preflop_nodes`
  and score `depth_bucket` agreement per node type. Print
  `open-node precision`, `facing-node precision`, coverage of each.
- [ ] **Step 2:** Run it; record the numbers in the plan-results doc
  (`docs/superpowers/plans/effbb-node-depth-results.md` — create it; this file
  is the phase's evidence).
- [ ] **Step 3:** Scalar-metric gate (D5): `python scripts/effbb_eval.py` —
  precision must stay ≥ 77.9% (the scalar metric does not consume node depths,
  so any drift means an accidental coupling — investigate before landing).
- [ ] **Step 4:** Push, PR titled
  `feat(node-depth): per-decision-node solver depths with range-mismatch caveats`,
  body includes before/after per-node numbers and the D1 table.

---

## Phase B — Chip Constraint Solver (`feat/chip-solver`)

**Files:**
- Create: `scripts/ocr/chip_solver.py`
- Modify: `scripts/ocr/n8_parser.py` (`_compute_effective_bb` integration, ~line 2557 feature block)
- Modify: `scripts/effbb_calibrate.py` (new feature columns)
- Test: `scripts/regression_test.py`

### Task B1: `ChipCheck` core — equations + residuals

- [ ] **Step 1: Failing tests**

```python
@test
def test_chip_solver_consistent_hand():
    """Pot headers that match the engine contributions -> consistent, ~0 residuals.
    6-max, blinds 0.5/1.0, no ante: UTG opens to 2.0, BB calls (1.0 more) ->
    flop pot = 2.0 + 2.0 + 0.5(SB fold) = 4.5."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0, "SB": 0.5},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 4.5},
    )
    assert_true(res.consistent, f"residuals={res.residuals}")
    assert_true(abs(res.residuals["flop"]) < 0.01)
    assert_true(res.repair is None)


@test
def test_chip_solver_single_field_repair():
    """A garbled call size (1.0 read for 10.0) leaves a 9.0 residual that
    exactly ONE field change explains -> repair names that field; nothing is
    auto-applied (D3a)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 11.0, "BB": 2.0, "SB": 0.5},   # BB call misread
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 22.5},                          # truth: BB called 11
        candidates={"BB": [2.0]},   # repairable fields: BB's contribution
    )
    assert_true(not res.consistent)
    assert_true(res.repair is not None and res.repair["field"] == "BB",
                f"repair={res.repair}")
    assert_true(abs(res.repair["to"] - 11.0) < 0.01)


@test
def test_chip_solver_ambiguous_repair_returns_none():
    """Two fields could each explain the residual -> repair=None (never guess)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 9.5},          # 5.0 unexplained
        candidates={"UTG": [2.0], "BB": [2.0]},
    )
    assert_true(not res.consistent and res.repair is None)
```

- [ ] **Step 2: Run → FAIL** (module missing).
- [ ] **Step 3: Implement `scripts/ocr/chip_solver.py`**

```python
"""Chip-conservation check over engine contributions vs panel pot headers.

D3: single-field repair only, never auto-applied; output feeds confidence /
structural-abstain features in _compute_effective_bb. Pure module."""
from dataclasses import dataclass, field


@dataclass
class ChipCheck:
    consistent: bool
    residuals: dict = field(default_factory=dict)
    repair: dict | None = None


def _tol(p):
    return max(0.25, 0.02 * abs(p))


def check_chips(*, contributions, sb, bb, ante_total, pot_headers,
                candidates=None):
    """contributions: position -> permanent chips in (engine units, bb).
    pot_headers: street -> pot shown at street START (so it sums everything
    permanently invested BEFORE that street; the project's panel headers are
    street-start values — see analyze docstrings around pot_bound).
    candidates: position -> [current values] eligible for single-field repair.
    """
    total = sum(contributions.values()) + (ante_total or 0.0)
    residuals = {}
    ok = True
    for street, p in (pot_headers or {}).items():
        if not isinstance(p, (int, float)) or p <= 0:
            continue
        r = total - p
        residuals[street] = round(r, 2)
        if abs(r) > _tol(p):
            ok = False
    if ok or not residuals:
        return ChipCheck(consistent=bool(residuals) and ok,
                         residuals=residuals)
    # single-field repair: one position's contribution shifted by the COMMON
    # residual fixes every inconsistent equation simultaneously.
    rs = [r for r in residuals.values()]
    common = rs[0]
    if any(abs(r - common) > 0.26 for r in rs):
        return ChipCheck(False, residuals)        # residuals disagree — multi-field
    fixes = []
    for pos, vals in (candidates or {}).items():
        cur = contributions.get(pos)
        if cur is None:
            continue
        to = cur - common
        if to >= 0:
            fixes.append({"field": pos, "from": cur, "to": round(to, 2)})
    repair = fixes[0] if len(fixes) == 1 else None
    return ChipCheck(False, residuals, repair)
```

NOTE the street-start semantics: build `pot_headers` from panel columns the
same way `_effective_bb_for_layout` builds `pot_by_street` (Flop header =
everything invested preflop). The caller (Task B2) is responsible for passing
contributions *as of the right street*; for v1 pass FINAL contributions and
only the LAST street's header + the matched-last-street top-up exactly as the
existing `pot_bound` block does (`n8_parser.py` "Pot-bounded over-compute
guard") — reuse that computed `pot_bound`, do not duplicate its logic.

- [ ] **Step 4: Run → PASS. Commit** — `git commit -m "feat(chip-solver): chip-conservation check with single-field repair detection"`

### Task B2: integration into `_compute_effective_bb` (feature-only)

- [ ] **Step 1:** In the orchestrator's feature block
  (`n8_parser.py` ~`:2596`, the `try: from . import effbb_engine as _eng_f`
  block that already computes `pot_residual`), call `check_chips` with the
  engine result (`_er.contribution`, `_er.sb/_er.bb/_er.ante_total`) and the
  street pots from `_engine_streets`; store
  `feat["chip_consistent"]`, `feat["chip_repair_found"]` (bool),
  `feat["chip_residual"]` (the common residual or None). NO behavior change yet.
- [ ] **Step 2:** Test: cache-driven `@test` asserting the three new keys exist
  in `_effbb_last_features()` after running a clean hand (mirror
  `test_phase4_features_surfaced_per_hand`, hand `TM5862908042`).
- [ ] **Step 3:** Run full suite → green. Commit —
  `git commit -m "feat(chip-solver): surface chip-conservation features per hand (no behavior change)"`

### Task B3: marginal-precision audit → enable as confidence input

- [ ] **Step 1:** Write `scripts/_tmp_chipgate.py` (gitignored, pattern of
  `scripts/_tmp_gate.py` from the Phase-6 session): over hero-active cache
  rows, bucket emitted hands by
  (`chip_consistent` / `~consistent & repair` / `~consistent & no repair`) and
  print n + marginal precision of each bucket.
- [ ] **Step 2:** Run it. Decision rule (locked):
  - the `~consistent & no repair` bucket abstains **only if** its marginal
    precision measures < 60% on ≥ 20 hands; otherwise the clause must NOT ship
    (this is exactly how the herozero clause was caught costing coverage).
  - `consistent` earns the `+0.03` confidence nudge only if its bucket measures
    above the overall emitted precision.
- [ ] **Step 3:** Implement whatever the audit licenses, behind
  `OCR_EFFBB_CHIP_SOLVER` (default per audit outcome), in the structural-gate
  block (`n8_parser.py` `_EFFBB_STRUCTURAL_GATE` section). Mirror the clause in
  `effbb_calibrate.py:shipped_gate` (keep them in sync — there's a drift
  comment there from 2026-06-11).
- [ ] **Step 4:** A/B with `scripts/_tmp_breaks.py` (save ref with flag off,
  diff with on): record fixed/broken/wrong→abstain in the commit message.
  D5 gates apply.
- [ ] **Step 5:** Full suite + commit —
  `git commit -m "feat(chip-solver): gate/confidence integration (audit numbers in message)"`

### Task B4: regression goldens + land

- [ ] **Step 1:** Add goldens: pick 2 corpus hands the audit shows being
  correctly abstained (or confidence-nudged) — assert via `_effbb_run`-style
  helpers. Every shipped clause needs a hand-level test (project rule).
- [ ] **Step 2:** `python scripts/effbb_eval.py` before/after numbers into the
  PR body. Push, PR `feat(chip-solver): chip-conservation redundancy check for effective_bb`.

---

## Phase C — Avatar-Anchored Seat Reading (`feat/seat-anchored-ocr`)

**Files:**
- Create: `scripts/ocr/seat_detector.py`
- Create: `scripts/ocr/seat_reader.py`
- Create: `scripts/seat_autolabel.py` (harness, committed — not `_tmp`)
- Modify: `scripts/ocr/table_parser.py:1052-1111` (`parse_table` — alternate `named_stacks` source behind `OCR_SEAT_ANCHORED`)
- Test: `scripts/regression_test.py` + harness scorecard

### Task C1: auto-label harness FIRST (it defines success before any detector exists)

- [ ] **Step 1:** Implement `scripts/seat_autolabel.py`:
  - For each corpus image with GT (`data/hand_images/img/*.png` ×
    `ground_truth.jsonl`): compute each seat's **expected displayed stack** =
    `starting_chips − invested_chips(through end of HH)` ÷ bb. Invested comes
    from replaying the HH actions per player (reuse
    `hh_parser.parse_hand` internals — extract its per-player chip walk into a
    helper `player_invested(hand_text) -> {position: chips}` if not already
    separable). Names: HH seats are anonymized IDs except Hero — match by
    VALUE + relative seat order, not by name.
  - CLI: `python scripts/seat_autolabel.py --score <detector_jsonl>` →
    per-image: how many expected seats matched a detected (value within
    max(0.3bb, 3%)), how many detections were phantom (no expected match),
    plus corpus totals: `seat_recall`, `seat_precision`, `value_accuracy`.
  - Baseline mode: `--score-current` runs the EXISTING `parse_table`
    `named_stacks` through the same scorer → this number is the bar Phase C
    must beat.
- [ ] **Step 2:** Run `--score-current` on a 300-image stride
  (`--stride 24`), record baseline `seat_recall/precision/value_accuracy` in
  `docs/superpowers/plans/effbb-seat-anchored-results.md`.
- [ ] **Step 3:** Commit harness + baseline doc.

### Task C2: `seat_detector.py` — classical avatar detection

- [ ] **Step 1:** Implement:
  - `detect_avatars(image: np.ndarray, table_size_hint: int | None) -> list[dict]`
    returning `{"cx": float, "cy": float, "r": float, "conf": float}`.
  - Method (locked, D4): `cv2.HoughCircles` on the table region
    (`region_detector.detect_regions` already isolates it) with radius bounds
    from the known N8 avatar size (measure once from 3 sample images; the
    corpus is fixed-resolution), THEN snap candidates to the table-size layout
    prior (the per-table-size canonical seat anglesalready implicit in
    `_seat_ring`'s hero-bottom-center assumption): a candidate > 8% of image
    diagonal away from every canonical slot is dropped; a canonical slot with
    no candidate is emitted with `conf=0` (lets the scorer count misses).
- [ ] **Step 2:** Score with the harness on the same 300-image stride:
  `python scripts/seat_autolabel.py --detector avatars --stride 24`.
  Gate: avatar recall ≥ 97% (D4). If below: ONE iteration of parameter tuning,
  then stop and report — choosing a CNN detector is a user decision, not an
  implementer decision.
- [ ] **Step 3:** Commit with the scorecard numbers in the message.

### Task C3: `seat_reader.py` — anchored ROI reads

- [ ] **Step 1:** Implement
  `read_seats(image, avatars) -> list[dict]` producing the `named_stacks`
  schema + `anchor_conf`:
  - nameplate ROI: fixed offset box under the avatar disc (measure the offset
    once from samples; constants at module top with a comment showing the
    measurement images);
  - stack ROI: lower half of the nameplate; EasyOCR with
    `allowlist="0123456789.,KM"`, parse `K/M` suffixes to bb-units the same
    way `table_parser` does today (reuse its number-normalization helper —
    grep `def _parse_stack` / equivalent in `table_parser.py` and import it,
    do not copy);
  - bounty/`$`-pill region adjacent to the avatar is **never** included in
    either ROI (D4) — no dollar-sign post-filtering needed.
- [ ] **Step 2:** Harness score: `value_accuracy` within correctly-detected
  seats. Gate: ≥ 98% on the stride (vs baseline from C1). If digit errors > 2%
  inside correct ROIs → stop, report (dedicated digit CNN is the follow-up
  decision, D4).
- [ ] **Step 3:** Commit with numbers.

### Task C4: wire into `parse_table` behind `OCR_SEAT_ANCHORED`

- [ ] **Step 1:** In `table_parser.parse_table` (~`:1052`), when
  `os.getenv("OCR_SEAT_ANCHORED") == "1"`: run detector+reader; if avatar
  count ≥ 2, REPLACE `all_stacks_named` with the anchored result; else fall
  back to the legacy scan (and tag `diagnostics["seat_anchored_fallback"]=True`).
- [ ] **Step 2:** `@test`: feed one checked-in fixture image
  (`scripts/ocr/test_data/` already exists for this purpose) through
  `parse_table` with the flag on; assert the named_stacks rows carry
  `anchor_conf` and contain no row whose value equals the pot.
- [ ] **Step 3:** Full suite green (flag defaults OFF — zero behavior change).
  Commit.

### Task C5: corpus A/B + cache rebuild + land

- [ ] **Step 1:** Full-corpus harness run, flag on vs off →
  `seat_recall/precision/value_accuracy` table into the results doc.
- [ ] **Step 2:** Rebuild the effbb cache with the flag on (D4b):
  `OCR_SEAT_ANCHORED=1 EFFBB_CAPTURE=1 python scripts/effbb_cache.py` (~3h).
  Save to `data/effbb_cache/cache_anchored.jsonl` — do NOT overwrite the
  baseline cache.
- [ ] **Step 3:** `python scripts/effbb_eval.py --cache data/effbb_cache/cache_anchored.jsonl`
  vs baseline. D5 gates decide the default flag value. Record both frontiers.
- [ ] **Step 4:** Push, PR with both scorecards. Flag default flips to ON only
  if the effbb frontier improves on BOTH precision and coverage; otherwise it
  lands OFF with the numbers documented and the decision escalated to the user.

---

## Self-check before each PR (all phases)

1. `python scripts/regression_test.py` → 0 new failures (the 1 known H2494
   worktree-environment failure and 16 stale snapshot expecteds are
   pre-existing — do not chase them here).
2. `python scripts/effbb_eval.py` scalar frontier vs 78.19% @ 70.4% — D5 gates.
3. Every behavior change has (a) an A/B fixed/broken count from a
   `_tmp_breaks.py`-style diff in the commit message, and (b) a hand-level
   regression golden.
4. Ad-hoc analysis scripts go to `scripts/_tmp*.py` (gitignored); harnesses
   that future phases need (`seat_autolabel.py`) are committed.
