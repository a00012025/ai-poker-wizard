# Coach Follow-up Grounding (P0 + P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the Telegram poker coach from hallucinating villain ranges/equity on "why / strategic" follow-ups by routing P0/P1 intents through a deterministic fact-fetch → grounded-narrate → hard-verify pipeline backed by GTO Wizard spot-solution data.

**Architecture:** New module `scripts/coach_facts.py` sits between the follow-up handler and the narrator. A tiny Flash intent classifier maps a question to a `QuestionType`; that type's `fetch()` pulls the right spot-solution node(s) from the cached hand context (and, for villain response nodes, one extra cached `get_spot_solution` call) and returns a compact `Facts` card plus an `allowed_claims` whitelist. A Flash narrator writes prose from ONLY the card; a hard verifier scans the prose for poker-combo tokens outside the whitelist and regenerates once, then falls back to a deterministic template. Unknown/`other` intents fall back to today's existing tool-calling path (no regression).

**Tech Stack:** Python 3, google-genai (`gemini-2.5-flash`, `thinking_budget=0`), existing `scripts/gto_api.py` + `scripts/gto_cache.py` + `scripts/gto_formatter.py` helpers, `scripts/regression_test.py` harness (`@test`, `assert_eq`, `assert_in`, `assert_true`).

---

## Background: confirmed data shapes (from live cache inspection)

`analyze_hand_full(hand)` → context dict (cached at `self.hand_contexts[chat_id]`) with keys:
`text, text_compact, hand, gametype, depth, stacks, is_icm, hero_position, hero_hand,
no_hero_hand, preflop_actions, street_states, final_actions{flop_actions,turn_actions,river_actions},
hero_spots[], solutions[], deeplink_raw_preflop, deeplink_raw_players`.

- `hero_spots[i]` = `{street, header, params{gametype,depth,stacks,preflop_actions,board,flop_actions,turn_actions,river_actions}, solver_hero_pos, action_desc, taken_code}`.
- `solutions[i]` = raw spot-solution JSON for that hero node (parallel to `hero_spots`), or `None`.

Spot-solution JSON (confirmed keys):
- `game{active_position, board, pot, pot_odds, current_street{type}}`
- `players_info[2]`, each: `player{position}`, `range[1326]`, `hand_eqs[1326]`, `hand_evs[1326]`,
  `hand_eqrs[1326]`, `eq_percentile[1326]`, `equity_buckets_range[1326]`,
  `hand_categories[]` (range composition; **for the acting player each entry has
  `actions_total_frequencies{code:freq}`**), `draw_categories[]`, `equity_buckets[]`
  (acting player: each has `actions_total_frequencies`), `simple_hand_counters{169 classes}`
  (each: `name,total_combos_available,total_combos,total_frequency,actions_total_combos,
  actions_total_frequencies,hand_ev,hand_eq,hand_eqr`), `total_eq,total_ev,total_eqr,pot_share`.
- `action_solutions[]`, each: `action{code,betsize,betsize_by_pot,allin,display_name}`,
  `strategy[1326]`, `evs`, `total_frequency`, `total_combos`, `hand_categories[]`, `draw_categories[]`.
- top-level: `hand_categories_range[1326]`, `draw_categories_range[1326]`.

**Key rule:** read per-action splits from the player whose `position == game.active_position`
(the actor). For "what does hero's bet fold out" we must fetch the **villain response node**
(hero's node + hero's bet code appended → villain becomes the actor). For "what's in villain's
bet range" we use the node where **hero faces villain's bet** (already a hero spot; villain is the
non-acting player there and `players_info[villain].hand_categories` is the betting-range
composition, while `players_info[hero].hand_eqs` is hero equity vs that range).

Reusable helpers in `scripts/gto_formatter.py`: `_COMBO_INDEX` (1326 list of `(card,card)`),
`_combo_to_hand_name(c1,c2)`, `combo_index_for_hand(raw)`, `normalize_hand_name(raw)`.

Follow-up routing today: `GeminiSessionManager.send_message` → if message isn't a hand →
`self._chat(...)` → `_chat_with_tools(...)` (model `self.model` = `gemini-2.5-pro`, tool loop,
`_needs_solver_grounding` forces `tool_config mode=ANY`). Context at `self.hand_contexts[chat_id]`.

---

## File structure

- **Create** `scripts/coach_facts.py` — `Ctx`, `Facts`, `QuestionType`, registry, node-resolution
  helpers, per-type `fetch` functions, intent classifier, narrator, hard verifier, public entry
  `answer_followup(...)`. One responsibility: turn a follow-up question + cached hand context into
  a grounded answer string (or signal `other` → caller falls back).
- **Create** `scripts/test_fixtures/coach_facts/` — committed JSON fixtures (real spot-solution
  nodes + a synthetic `Ctx`) so fetch/verifier tests run with no network.
- **Modify** `src/gemini_session.py` — in `_chat_with_tools`, before the tool loop, try
  `coach_facts.answer_followup`; on a non-`other` grounded answer, return it; else fall through.
- **Modify** `scripts/regression_test.py` — add `coach_facts` unit/golden tests.

---

## Type / intent labels (stable across tasks)

Classifier emits exactly one of: `why_action` (B), `fold_equity` (C), `villain_range` (D),
`hand_strength` (E), `range_shift` (F, P1), `hypothetical` (G, P1), `sizing` (H, P1),
`node_url` (I, P1), `range_lookup` (A → existing path), `other`.

`Facts` dataclass (final shape, used by every task):

```python
@dataclass
class Facts:
    intent: str                  # QuestionType.id, e.g. "fold_equity"
    title: str                   # one-line node header for the card
    lines: list[str]             # human-readable card body lines
    allowed_claims: set[str]     # normalized combo/class tokens the narrator may name
    numbers: set[int]            # integer percentages present in the card (for numeric audit)
    meta: dict                   # structured data for the deterministic template fallback
    note: str | None = None      # fragility flag (off-tree / low-freq / missing node)

    def render(self) -> str:
        head = self.title
        body = "\n".join(self.lines)
        tail = f"\n⚠ {self.note}" if self.note else ""
        return f"{head}\n{body}{tail}"
```

`Ctx` dataclass:

```python
@dataclass
class Ctx:
    question: str
    hand_context: dict           # the analyze_hand_full dict
    user_id: int | None = None
    refresh_token: str | None = None
```

`QuestionType` dataclass:

```python
@dataclass
class QuestionType:
    id: str
    matches: list[str]                       # intent labels the classifier may emit
    fetch: Callable[["Ctx"], "Facts | None"] # builds the card; None → cannot ground → other
```

---

### Task 1: Module skeleton + shared dataclasses + combo/class utilities

**Files:**
- Create: `scripts/coach_facts.py`
- Test: add to `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests** (append near other unit tests in `scripts/regression_test.py`)

```python
@test("coach_facts: class->combo-index grouping covers all 1326 and 169 classes")
def test_coach_facts_class_groups():
    import coach_facts
    groups = coach_facts._class_to_combo_indices()
    assert_eq(len(groups), 169, "169 classes")
    assert_eq(sum(len(v) for v in groups.values()), 1326, "all combos grouped")
    assert_eq(len(groups["AA"]), 6, "AA has 6 combos")
    assert_eq(len(groups["AKs"]), 4, "AKs has 4 combos")
    assert_eq(len(groups["AKo"]), 12, "AKo has 12 combos")

@test("coach_facts: extract_combo_tokens finds classes and specific combos in Chinese prose")
def test_coach_facts_extract_tokens():
    import coach_facts
    toks = coach_facts.extract_combo_tokens("對手 AJo 會棄牌，但 66 和 AhKh 跟注，頂對價值。")
    assert_in("AJo", toks)
    assert_in("66", toks)
    assert_in("AhKh", toks)
    # position/term false-positives excluded
    none = coach_facts.extract_combo_tokens("BB 在 BTN 的 EV 很高")
    assert_true("BB" not in none and "EV" not in none, "positions/terms not combos")
```

- [ ] **Step 2: Run, expect ImportError/FAIL**

Run: `python scripts/regression_test.py 2>&1 | grep -i coach_facts`
Expected: failures (module missing).

- [ ] **Step 3: Implement skeleton + utilities in `scripts/coach_facts.py`**

```python
#!/usr/bin/env python3
"""Grounded coach follow-up answers.

Routes a follow-up question + cached hand context through:
  classify_intent -> registry[type].fetch -> narrate -> verify -> (regen|template).
Unknown/other intents return None so the caller keeps the existing tool path.
"""
from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import gto_formatter as gf
from gto_api import get_spot_solution

logger = logging.getLogger(__name__)

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_POSITIONS = {"UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB", "MP", "EP", "UTG1", "UTG2"}
# Latin tokens that look like combos but are jargon to ignore.
_TOKEN_STOPWORDS = {"EV", "GTO", "EQ", "IP", "OOP", "AI", "VS", "OK"}


@dataclass
class Ctx:
    question: str
    hand_context: dict
    user_id: int | None = None
    refresh_token: str | None = None


@dataclass
class Facts:
    intent: str
    title: str
    lines: list[str]
    allowed_claims: set[str] = field(default_factory=set)
    numbers: set[int] = field(default_factory=set)
    meta: dict = field(default_factory=dict)
    note: str | None = None

    def render(self) -> str:
        head = self.title
        body = "\n".join(self.lines)
        tail = f"\n⚠ {self.note}" if self.note else ""
        return f"{head}\n{body}{tail}"


@dataclass
class QuestionType:
    id: str
    matches: list[str]
    fetch: Callable[["Ctx"], "Facts | None"]


# ── combo / class utilities ──────────────────────────────────────────────
_CLASS_GROUPS_CACHE: dict[str, list[int]] | None = None


def _class_to_combo_indices() -> dict[str, list[int]]:
    """Map each 169-class name -> list of 1326 combo indices belonging to it."""
    global _CLASS_GROUPS_CACHE
    if _CLASS_GROUPS_CACHE is None:
        groups: dict[str, list[int]] = {}
        for idx, (c1, c2) in enumerate(gf._COMBO_INDEX):
            name = gf._combo_to_hand_name(c1, c2)
            groups.setdefault(name, []).append(idx)
        _CLASS_GROUPS_CACHE = groups
    return _CLASS_GROUPS_CACHE


# class token: two ranks + optional s/o  (AKs, AJo, 66, KT)
_RE_CLASS = re.compile(r"\b([2-9TJQKA])([2-9TJQKA])([so])?\b")
# specific combo: two full cards  (AhKh, 9d8d)
_RE_COMBO = re.compile(r"\b([2-9TJQKA][cdhs])([2-9TJQKA][cdhs])\b")


def extract_combo_tokens(text: str) -> set[str]:
    """Extract poker hand tokens (classes + specific combos) from prose."""
    out: set[str] = set()
    if not text:
        return out
    for m in _RE_COMBO.finditer(text):
        out.add(m.group(1) + m.group(2))
    for m in _RE_CLASS.finditer(text):
        tok = m.group(0)
        if tok in _TOKEN_STOPWORDS or tok in _POSITIONS:
            continue
        out.add(tok)
    return out


def canonical_forms(token: str) -> set[str]:
    """All written forms of a hand token for whitelist matching.

    Normalizes rank order so 'TK'->'KT'; a specific combo also yields its class.
    """
    forms: set[str] = set()
    t = token.strip()
    cm = _RE_COMBO.fullmatch(t)
    if cm:
        c1, c2 = cm.group(1), cm.group(2)
        forms.add(c1 + c2)
        forms.add(c2 + c1)
        forms.add(gf._combo_to_hand_name(c1, c2))
        return forms
    clm = _RE_CLASS.fullmatch(t)
    if clm:
        r1, r2, suit = clm.group(1), clm.group(2), clm.group(3) or ""
        order = "AKQJT98765432"
        if order.index(r1) > order.index(r2):
            r1, r2 = r2, r1
        forms.add(r1 + r2 + suit)
        forms.add(r1 + r2)        # bare class always allowed if suited/offsuit allowed
        return forms
    forms.add(t)
    return forms
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python scripts/regression_test.py 2>&1 | grep -i "coach_facts"`
Expected: both new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): coach_facts skeleton — dataclasses + combo/class utils"
```

---

### Task 2: Capture deterministic fixtures (hero node + villain response node)

**Files:**
- Create: `scripts/test_fixtures/coach_facts/hero_node.json`,
  `scripts/test_fixtures/coach_facts/villain_response_node.json`,
  `scripts/test_fixtures/coach_facts/ctx.json`
- Create (throwaway, gitignored): `scripts/_tmp.py`

- [ ] **Step 1: Write capture script `scripts/_tmp.py`**

Capture from a real analyzed hand so fixtures match production shapes. Use a postflop hand
where hero bets the flop (gives both a hero acting node and a villain response node).

```python
import json, os
from pathlib import Path
import analyze_hand

HAND = {
    "game_format": "mtt", "effective_bb": 40, "players_at_table": 6,
    "hero_position": "HJ", "hero_hand": "KsJh",
    "preflop_actions": "F-F-R2.2-F-C-F",   # HJ open, BB call (adjust if parser differs)
    "streets": {"flop": {"board": "Ks9s2h", "actions": "X-R1.4-C"}},
}
ctx = analyze_hand.analyze_hand_full(HAND)
out = Path("scripts/test_fixtures/coach_facts"); out.mkdir(parents=True, exist_ok=True)

# hero node = first postflop solution that is non-null
hero_sol = next((s for s in ctx["solutions"] if s and (s.get("game") or {}).get("board")), None)
(out / "hero_node.json").write_text(json.dumps(hero_sol))

# villain response node: hero flop spot params + hero bet appended
flop_spot = next(sp for sp, s in zip(ctx["hero_spots"], ctx["solutions"])
                 if sp["street"] == "flop" and s)
p = dict(flop_spot["params"])
# append hero's taken bet code to the flop action string
p["flop_actions"] = (p["flop_actions"] + "-" + flop_spot["taken_code"]).lstrip("-")
vr = get_spot_solution(**p) if False else __import__("gto_api").get_spot_solution(**p)
(out / "villain_response_node.json").write_text(json.dumps(vr))

# minimal ctx for fetch tests (strip giant arrays we don't need duplicated)
slim = {k: ctx[k] for k in ("hero_hand","hero_position","gametype","depth","stacks",
        "final_actions","street_states","hand")}
slim["hero_spots"] = [{k: sp[k] for k in ("street","params","solver_hero_pos","taken_code")}
                      for sp in ctx["hero_spots"]]
(out / "ctx.json").write_text(json.dumps(slim))
print("acting hero:", (hero_sol.get("game") or {}).get("active_position"))
print("acting villain node:", (vr.get("game") or {}).get("active_position") if vr else None)
```

- [ ] **Step 2: Run capture (uses owner DB token); verify acting players**

Run: `cd ~/ai-poker-wizard-coach-impl && python scripts/_tmp.py`
Expected: prints hero acting position == "HJ" and villain node acting == "BB".
If the villain node is `None` or the parser rejects the preflop string, adjust `HAND`
(`preflop_actions`/positions) until both nodes resolve. The board/positions may differ from
the spec's example boards — that's fine; tests assert structure, not historical boards.

- [ ] **Step 3: Commit fixtures**

```bash
git add scripts/test_fixtures/coach_facts/*.json
git commit -m "test(coach): capture real spot-solution fixtures for coach_facts"
```

---

### Task 3: Node-resolution + digest helpers

**Files:**
- Modify: `scripts/coach_facts.py`
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests**

```python
def _load_coach_ctx():
    import coach_facts, json, os
    base = os.path.join(os.path.dirname(__file__), "test_fixtures", "coach_facts")
    hctx = json.load(open(os.path.join(base, "ctx.json")))
    hero = json.load(open(os.path.join(base, "hero_node.json")))
    villain = json.load(open(os.path.join(base, "villain_response_node.json")))
    return coach_facts, hctx, hero, villain

@test("coach_facts: acting_position + category_action_table")
def test_coach_facts_digest_helpers():
    cf, hctx, hero, villain = _load_coach_ctx()
    assert_eq(cf._acting_position(hero), hctx["hero_position"], "hero acts at hero node")
    table = cf._category_action_table(hero, top_n=4)
    assert_true(len(table) >= 1, "at least one category")
    name, freq, actions = table[0]
    assert_true(0.0 <= freq <= 1.0, "category freq is a fraction")
    assert_true(abs(sum(actions.values()) - 1.0) < 0.05, "per-category actions sum ~1")

@test("coach_facts: rep_classes_for_category returns in-range classes mapped to category")
def test_coach_facts_rep_classes():
    cf, hctx, hero, villain = _load_coach_ctx()
    table = cf._category_action_table(hero, top_n=6)
    cat_name = table[0][0]
    reps = cf._rep_classes_for_category(hero, cat_name, top_k=2)
    assert_true(len(reps) >= 1, "at least one representative class")
    cls, freq, actions = reps[0]
    assert_true(cls in cf._class_to_combo_indices(), "rep is a real 169 class")
```

- [ ] **Step 2: Run, expect FAIL** (helpers undefined)

Run: `python scripts/regression_test.py 2>&1 | grep -i "digest_helpers\|rep_classes"`

- [ ] **Step 3: Implement helpers in `scripts/coach_facts.py`**

```python
def _players(sol: dict) -> dict[str, dict]:
    return {pi["player"]["position"]: pi for pi in (sol.get("players_info") or [])}


def _acting_position(sol: dict) -> str | None:
    return (sol.get("game") or {}).get("active_position")


def _pct(x: float) -> int:
    return int(round(100 * (x or 0.0)))


def _category_action_table(sol: dict, top_n: int = 4) -> list[tuple[str, float, dict]]:
    """Acting player's hand categories with per-action freq splits, by frequency.

    Returns [(category_name, range_frequency, {action_code: freq}), ...].
    """
    pos = _acting_position(sol)
    pi = _players(sol).get(pos) or {}
    rows = []
    for hc in pi.get("hand_categories") or []:
        freq = hc.get("total_frequency") or 0.0
        if freq <= 0.005:
            continue
        actions = {k: v for k, v in (hc.get("actions_total_frequencies") or {}).items()
                   if v and v > 0.005}
        if not actions:
            continue
        rows.append((hc["name"], freq, actions))
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]


def _class_category_map(sol: dict, pos: str) -> dict[str, str]:
    """Map each in-range 169-class -> its dominant hand category at this board.

    Uses top-level hand_categories_range[1326] (category index per combo) + the
    category index->name map from the acting player's hand_categories list.
    """
    idx_to_name = {}
    pi = _players(sol).get(pos) or {}
    for hc in pi.get("hand_categories") or []:
        if "index" in hc:
            idx_to_name[hc["index"]] = hc["name"]
    hcr = sol.get("hand_categories_range") or []
    if len(hcr) != 1326:
        return {}
    rng = pi.get("range") or []
    groups = _class_to_combo_indices()
    out: dict[str, str] = {}
    for cls, idxs in groups.items():
        counts: dict[int, float] = {}
        for i in idxs:
            w = rng[i] if i < len(rng) else 0.0
            if w and w > 0:
                cat = hcr[i]
                counts[cat] = counts.get(cat, 0.0) + w
        if counts:
            best = max(counts, key=counts.get)
            out[cls] = idx_to_name.get(best, str(best))
    return out


def _rep_classes_for_category(sol: dict, category_name: str,
                              top_k: int = 2) -> list[tuple[str, float, dict]]:
    """Top-k in-range 169-classes whose dominant category == category_name.

    Returns [(class, total_frequency, {action_code: freq}), ...] from
    the acting player's simple_hand_counters (grounded, real numbers).
    """
    pos = _acting_position(sol)
    pi = _players(sol).get(pos) or {}
    cmap = _class_category_map(sol, pos)
    shc = pi.get("simple_hand_counters") or {}
    cand = []
    for cls, cat in cmap.items():
        if cat != category_name:
            continue
        hd = shc.get(cls)
        if not hd:
            continue
        f = hd.get("total_frequency") or 0.0
        if f <= 0.01:
            continue
        actions = {k: v for k, v in (hd.get("actions_total_frequencies") or {}).items()
                   if v and v > 0.01}
        cand.append((cls, f, actions))
    cand.sort(key=lambda r: -r[1])
    return cand[:top_k]


def _hero_combo_facts(sol: dict, hero_pos: str, hero_hand: str) -> dict | None:
    """Hero combo's eq / eqr / percentile / per-action freq at this node."""
    pi = _players(sol).get(hero_pos)
    if not pi:
        return None
    idx = gf.combo_index_for_hand(hero_hand) if hero_hand else None
    cls = gf.normalize_hand_name(hero_hand) if hero_hand else None
    out: dict = {"class": cls}
    if idx is not None:
        for k_src, k_dst in (("hand_eqs", "eq"), ("hand_eqrs", "eqr"),
                             ("eq_percentile", "percentile")):
            arr = pi.get(k_src) or []
            if idx < len(arr):
                out[k_dst] = arr[idx]
    shc = (pi.get("simple_hand_counters") or {}).get(cls or "")
    if shc:
        out["actions"] = {k: v for k, v in (shc.get("actions_total_frequencies") or {}).items()
                          if v and v > 0.01}
        out.setdefault("eq", shc.get("hand_eq"))
    return out


def _hero_eq_vs_range(sol: dict, hero_pos: str, hero_hand: str) -> tuple[int, int] | None:
    """(equity%, percentile%) for hero's combo vs the opponent range at this node."""
    f = _hero_combo_facts(sol, hero_pos, hero_hand)
    if not f or f.get("eq") is None:
        return None
    return _pct(f["eq"]), _pct(f.get("percentile") or 0.0)


def _hero_villain(ctx: Ctx) -> tuple[str, str | None]:
    """Resolve hero position + villain position from context (postflop 2-way)."""
    hc = ctx.hand_context
    hero = hc.get("hero_position")
    villain = None
    for sol in (hc.get("solutions") or []):
        if not sol:
            continue
        for p in _players(sol):
            if p != hero:
                villain = p
                break
        if villain:
            break
    return hero, villain


def _street_from_question(q: str) -> str | None:
    ql = (q or "").lower()
    if any(w in ql for w in ("river", "河牌", "河")):
        return "river"
    if any(w in ql for w in ("turn", "轉牌", "转牌", "轉")):
        return "turn"
    if any(w in ql for w in ("flop", "翻牌", "翻")):
        return "flop"
    return None


def _hero_spot_and_sol(ctx: Ctx, street: str | None):
    """Pick (hero_spot, solution) for the requested street, else the last postflop spot."""
    hc = ctx.hand_context
    spots = hc.get("hero_spots") or []
    sols = hc.get("solutions") or []
    pairs = [(sp, s) for sp, s in zip(spots, sols) if s]
    if not pairs:
        return None, None
    if street:
        for sp, s in pairs:
            if sp.get("street") == street:
                return sp, s
    post = [(sp, s) for sp, s in pairs if sp.get("street") in ("flop", "turn", "river")]
    return (post[-1] if post else pairs[-1])
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python scripts/regression_test.py 2>&1 | grep -i "digest_helpers\|rep_classes"`

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): node-resolution + digest helpers for coach_facts"
```

---

### Task 4: Type B (why hero hand acts) + Type E (hand strength) fetch

**Files:**
- Modify: `scripts/coach_facts.py`
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests**

```python
@test("coach_facts: fetch_why_action builds grounded card with hero combo facts")
def test_coach_facts_fetch_b():
    cf, hctx, hero, villain = _load_coach_ctx()
    ctx = cf.Ctx(question="為什麼這手牌要下注？", hand_context=hctx)
    facts = cf.fetch_why_action(ctx)
    assert_true(facts is not None, "B fetch returns facts")
    assert_eq(facts.intent, "why_action")
    # hero hand is always an allowed claim
    assert_in(hctx["hero_hand"], {c for c in facts.allowed_claims})
    assert_true(any("%" in ln for ln in facts.lines), "card has numbers")

@test("coach_facts: fetch_hand_strength reports equity + percentile bucket")
def test_coach_facts_fetch_e():
    cf, hctx, hero, villain = _load_coach_ctx()
    ctx = cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx)
    facts = cf.fetch_hand_strength(ctx)
    assert_true(facts is not None, "E fetch returns facts")
    assert_eq(facts.intent, "hand_strength")
    assert_true(len(facts.numbers) >= 1, "numbers captured for audit")
```

- [ ] **Step 2: Run, expect FAIL**

Run: `python scripts/regression_test.py 2>&1 | grep -i "fetch_b\|fetch_e"`

- [ ] **Step 3: Implement in `scripts/coach_facts.py`**

```python
# ── category display (Chinese) ──
_CAT_ZH = {
    "no_made_hand": "無對子", "king_high": "K高", "ace_high": "A高", "low_pair": "小對",
    "third_pair": "三對", "second_pair": "中對", "top_pair": "頂對", "over_pair": "超對",
    "two_pair": "兩對", "set": "三條", "trips": "三條", "straight": "順子", "flush": "同花",
    "full_house": "葫蘆", "quads": "四條", "straight_flush": "同花順",
}


def _cat_zh(name: str) -> str:
    return _CAT_ZH.get(name, name)


def _fmt_actions(actions: dict) -> str:
    parts = []
    for code, fr in sorted(actions.items(), key=lambda kv: -kv[1]):
        label = "過牌" if code == "X" else ("跟注" if code == "C" else
                ("棄牌" if code == "F" else ("全下" if code in ("RAI",) else f"下注{code[1:]}")))
        parts.append(f"{label} {_pct(fr)}%")
    return " | ".join(parts)


def _record_numbers(facts: Facts, *vals):
    for v in vals:
        if isinstance(v, (int, float)):
            facts.numbers.add(_pct(v) if v <= 1.0 else int(round(v)))


def fetch_why_action(ctx: Ctx) -> Facts | None:
    street = _street_from_question(ctx.question)
    spot, sol = _hero_spot_and_sol(ctx, street)
    if not sol:
        return None
    hero = ctx.hand_context.get("hero_position")
    hero_hand = ctx.hand_context.get("hero_hand")
    board = (sol.get("game") or {}).get("board") or ""
    hf = _hero_combo_facts(sol, hero, hero_hand)
    if not hf:
        return None
    facts = Facts(intent="why_action",
                  title=f"{hero} {hero_hand} 在 {board} 的決策數據：",
                  lines=[])
    cls = hf.get("class") or hero_hand
    facts.allowed_claims |= canonical_forms(hero_hand or cls)
    if hf.get("eq") is not None:
        facts.lines.append(f"  本手 equity {_pct(hf['eq'])}%"
                           + (f"、強度 percentile {_pct(hf['percentile'])}%"
                              if hf.get("percentile") is not None else ""))
        facts.numbers.add(_pct(hf["eq"]))
    if hf.get("actions"):
        facts.lines.append(f"  solver 動作頻率：{_fmt_actions(hf['actions'])}")
    # per-action EV from action_solutions for hero combo
    idx = gf.combo_index_for_hand(hero_hand) if hero_hand else None
    if idx is not None:
        for asol in sol.get("action_solutions") or []:
            evs = asol.get("evs") or []
            strat = asol.get("strategy") or []
            if len(evs) == 1326 and len(strat) == 1326 and strat[idx] > 0.01:
                code = asol["action"]["code"]
                facts.lines.append(f"  若 {code}: EV {evs[idx]:.2f}bb，頻率 {_pct(strat[idx])}%")
    facts.meta = {"hero_hand": hero_hand, "board": board, "facts": hf}
    return facts


def fetch_hand_strength(ctx: Ctx) -> Facts | None:
    street = _street_from_question(ctx.question)
    spot, sol = _hero_spot_and_sol(ctx, street)
    if not sol:
        return None
    hero = ctx.hand_context.get("hero_position")
    hero_hand = ctx.hand_context.get("hero_hand")
    board = (sol.get("game") or {}).get("board") or ""
    eqp = _hero_eq_vs_range(sol, hero, hero_hand)
    if not eqp:
        return None
    eq, pct = eqp
    hf = _hero_combo_facts(sol, hero, hero_hand) or {}
    pos = _acting_position(sol)
    # which equity bucket the hero combo sits in
    bucket_name = None
    pi = _players(sol).get(hero) or {}
    ebr = pi.get("equity_buckets_range") or []
    idx = gf.combo_index_for_hand(hero_hand) if hero_hand else None
    if idx is not None and idx < len(ebr):
        bidx = ebr[idx]
        for eb in pi.get("equity_buckets") or []:
            if eb.get("index") == bidx:
                bucket_name = eb.get("name")
                break
    facts = Facts(intent="hand_strength",
                  title=f"{hero} {hero_hand} 在 {board} 的牌力：",
                  lines=[f"  對上對手範圍 equity {eq}%、強度 percentile {pct}%"
                         + (f"、區間「{bucket_name}」" if bucket_name else "")])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    facts.numbers |= {eq, pct}
    facts.meta = {"hero_hand": hero_hand, "board": board, "eq": eq, "percentile": pct,
                  "bucket": bucket_name}
    return facts
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python scripts/regression_test.py 2>&1 | grep -i "fetch_b\|fetch_e"`

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): type B (why-action) + E (hand-strength) fetch"
```

---

### Task 5: Type C (fold equity) + Type D (villain range) fetch

**Files:**
- Modify: `scripts/coach_facts.py`
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests**

```python
@test("coach_facts: fetch_fold_equity uses villain response node category splits")
def test_coach_facts_fetch_c():
    cf, hctx, hero, villain = _load_coach_ctx()
    # stub the villain-response fetch with our fixture (no network)
    ctx = cf.Ctx(question="我這樣下注能讓對手棄掉哪些牌？", hand_context=hctx)
    facts = cf._fetch_fold_equity_from(villain, hctx)
    assert_true(facts is not None, "C fetch returns facts")
    assert_eq(facts.intent, "fold_equity")
    assert_true(any("棄牌" in ln for ln in facts.lines), "shows fold split")
    # every example class named is whitelisted
    for ln in facts.lines:
        for tok in cf.extract_combo_tokens(ln):
            assert_in(tok, {c for c in facts.allowed_claims}, f"{tok} grounded")

@test("coach_facts: fetch_villain_range composes villain bet range + hero equity")
def test_coach_facts_fetch_d():
    cf, hctx, hero, villain = _load_coach_ctx()
    ctx = cf.Ctx(question="對手下注的範圍有哪些牌？", hand_context=hctx)
    # hero faces villain bet node == the hero node fixture has villain as non-actor
    facts = cf._fetch_villain_range_from(hero, hctx)
    assert_true(facts is not None, "D fetch returns facts")
    assert_eq(facts.intent, "villain_range")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement in `scripts/coach_facts.py`**

```python
def _resolve_villain_response_node(ctx: Ctx) -> dict | None:
    """Node where villain faces hero's bet (villain is the actor).

    hero's bet node params + hero's bet code appended to that street's action string,
    fetched via get_spot_solution (cached after first call).
    """
    street = _street_from_question(ctx.question)
    spot, sol = _hero_spot_and_sol(ctx, street)
    if not spot:
        return None
    taken = spot.get("taken_code") or ""
    if not taken.startswith("R"):
        return None  # hero didn't bet/raise on this street → no fold-equity question
    p = dict(spot["params"])
    skey = {"flop": "flop_actions", "turn": "turn_actions",
            "river": "river_actions"}[spot["street"]]
    p[skey] = (p.get(skey, "") + "-" + taken).lstrip("-")
    try:
        return get_spot_solution(**p)
    except Exception as e:
        logger.warning(f"coach_facts: villain response fetch failed: {e}")
        return None


def _fetch_fold_equity_from(vsol: dict, hand_context: dict) -> Facts | None:
    if not vsol:
        return None
    board = (vsol.get("game") or {}).get("board") or ""
    villain = _acting_position(vsol)
    hero = hand_context.get("hero_position")
    hero_hand = hand_context.get("hero_hand")
    table = _category_action_table(vsol, top_n=5)
    if not table:
        return None
    facts = Facts(intent="fold_equity",
                  title=f"對手 {villain} 面對你的下注（{board}）的反應：",
                  lines=[])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    for name, freq, actions in table:
        zh = _cat_zh(name)
        line = f"  {zh} 佔 {_pct(freq)}% — {_fmt_actions(actions)}"
        reps = _rep_classes_for_category(vsol, name, top_k=2)
        if reps:
            ex = "，".join(f"{c}({_fmt_actions(a)})" for c, a, in
                          [(c, a) for c, f, a in reps])
            line += f"   例：{ex}"
            for c, f, a in reps:
                facts.allowed_claims |= canonical_forms(c)
        facts.lines.append(line)
        facts.numbers |= {_pct(v) for v in actions.values()}
    eqp = _hero_eq_vs_range(vsol, hero, hero_hand)
    if eqp:
        facts.lines.append(f"  你的 {hero_hand} 對上對手續打範圍 equity {eqp[0]}%")
        facts.numbers.add(eqp[0])
    facts.meta = {"villain": villain, "board": board, "table": table}
    return facts


def _fetch_villain_range_from(hsol: dict, hand_context: dict) -> Facts | None:
    """Villain bet-range composition at the node where hero faces villain's bet."""
    if not hsol:
        return None
    hero = hand_context.get("hero_position")
    hero_hand = hand_context.get("hero_hand")
    board = (hsol.get("game") or {}).get("board") or ""
    # villain is the non-acting player here; its range == the betting range
    acting = _acting_position(hsol)
    villain = next((p for p in _players(hsol) if p != acting and p != hero), None)
    villain = villain or next((p for p in _players(hsol) if p != hero), None)
    pi = _players(hsol).get(villain) or {}
    rows = []
    for hc in pi.get("hand_categories") or []:
        f = hc.get("total_frequency") or 0.0
        if f > 0.005:
            rows.append((hc["name"], f))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return None
    facts = Facts(intent="villain_range",
                  title=f"對手 {villain} 在 {board} 的下注範圍組成：",
                  lines=[])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    cmap = _class_category_map(hsol, villain)
    shc = pi.get("simple_hand_counters") or {}
    for name, f in rows[:5]:
        zh = _cat_zh(name)
        line = f"  {zh} 佔 {_pct(f)}%"
        reps = sorted(((c, shc.get(c, {}).get("total_frequency", 0.0))
                       for c, cat in cmap.items() if cat == name
                       and shc.get(c, {}).get("total_frequency", 0.0) > 0.01),
                      key=lambda kv: -kv[1])[:2]
        if reps:
            line += "   例：" + "、".join(c for c, _ in reps)
            for c, _ in reps:
                facts.allowed_claims |= canonical_forms(c)
        facts.lines.append(line)
        facts.numbers.add(_pct(f))
    eqp = _hero_eq_vs_range(hsol, hero, hero_hand)
    if eqp:
        facts.lines.append(f"  你的 {hero_hand} 對上此範圍 equity {eqp[0]}%、percentile {eqp[1]}%")
        facts.numbers |= {eqp[0], eqp[1]}
    facts.meta = {"villain": villain, "board": board, "rows": rows}
    return facts


def fetch_fold_equity(ctx: Ctx) -> Facts | None:
    vsol = _resolve_villain_response_node(ctx)
    return _fetch_fold_equity_from(vsol, ctx.hand_context)


def fetch_villain_range(ctx: Ctx) -> Facts | None:
    # node where hero faces villain's bet: a hero spot whose taken context faces a bet
    street = _street_from_question(ctx.question)
    spot, sol = _hero_spot_and_sol(ctx, street)
    return _fetch_villain_range_from(sol, ctx.hand_context)
```

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): type C (fold-equity) + D (villain-range) fetch"
```

---

### Task 6: Hard verifier (extract → whitelist → verdict)

**Files:**
- Modify: `scripts/coach_facts.py`
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests**

```python
@test("coach_facts: verifier passes grounded prose, flags ungrounded combo")
def test_coach_facts_verifier():
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t",
                     lines=["A高 棄牌 80%，例：AJo"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KsJh"),
                     meta={})
    board = "Ks9s2h"
    ok = cf.verify_claims("對手 A高如 AJo 會棄牌，你的 KsJh 領先。", facts, board)
    assert_true(ok.ok, "grounded prose passes")
    bad = cf.verify_claims("對手 AQo 和 KTs 會棄牌。", facts, board)
    assert_true(not bad.ok, "ungrounded AQo/KTs flagged")
    assert_in("AQo", bad.violations)

@test("coach_facts: verifier whitelists board cards and hero hand forms")
def test_coach_facts_verifier_board():
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t", lines=["equity 37%"],
                     allowed_claims=cf.canonical_forms("KsJh"), meta={})
    ok = cf.verify_claims("KsJh 在 Ks9s2h 上是頂對。", facts, "Ks9s2h")
    assert_true(ok.ok, "board cards + hero combo allowed")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement in `scripts/coach_facts.py`**

```python
@dataclass
class Verdict:
    ok: bool
    violations: list[str] = field(default_factory=list)
    number_violations: list[int] = field(default_factory=list)


def _board_card_tokens(board: str) -> set[str]:
    out: set[str] = set()
    for k in range(0, len(board or ""), 2):
        c = board[k:k + 2]
        if len(c) == 2:
            out.add(c)
            out.add(c[0] + c[0])  # 'Ks' -> allow 'KK' references to the board pair? no:
    # only the literal card tokens; remove the accidental pair form
    return {c for k in range(0, len(board or ""), 2) for c in [board[k:k+2]] if len(c) == 2}


def _whitelist(facts: Facts, board: str) -> set[str]:
    wl: set[str] = set()
    for tok in facts.allowed_claims:
        wl |= canonical_forms(tok)
    wl |= _board_card_tokens(board)
    return wl


def verify_claims(prose: str, facts: Facts, board: str,
                  audit_numbers: bool = False) -> Verdict:
    """Flag any poker-combo token in prose not present in the whitelist.

    Numeric audit (P1): when audit_numbers, flag integer %s far from any fact number.
    """
    wl = _whitelist(facts, board)
    violations = []
    for tok in extract_combo_tokens(prose):
        forms = canonical_forms(tok)
        if not (forms & wl):
            violations.append(tok)
    num_viol = []
    if audit_numbers and facts.numbers:
        for m in re.finditer(r"(\d{1,3})\s*%", prose):
            n = int(m.group(1))
            if all(abs(n - f) > 8 for f in facts.numbers):
                num_viol.append(n)
    return Verdict(ok=not violations and not num_viol,
                   violations=violations, number_violations=num_viol)
```

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): hard verifier — combo whitelist + numeric audit hook"
```

---

### Task 7: Intent classifier + narrator + template fallback + public entry

**Files:**
- Modify: `scripts/coach_facts.py`
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests** (deterministic parts only — no live model)

```python
@test("coach_facts: registry covers all P0/P1 intent labels")
def test_coach_facts_registry():
    import coach_facts as cf
    ids = {qt.id for qt in cf.REGISTRY}
    for need in ("why_action", "fold_equity", "villain_range", "hand_strength",
                 "range_shift", "sizing", "hypothetical", "node_url"):
        assert_in(need, ids)

@test("coach_facts: deterministic template builds grounded answer from facts")
def test_coach_facts_template():
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對你的下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo"), meta={})
    out = cf.render_template(facts)
    assert_in("A高", out)
    # template only contains grounded combos
    for tok in cf.extract_combo_tokens(out):
        assert_in(tok, {c for c in facts.allowed_claims})

@test("coach_facts: answer_followup returns None for 'other' intent (fallback)")
def test_coach_facts_other_fallback():
    import coach_facts as cf
    cf._set_intent_classifier(lambda q, c: "other")  # test hook
    out = cf.answer_followup(cf.Ctx(question="天氣如何", hand_context={}))
    assert_true(out is None, "other → None so caller keeps existing path")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement in `scripts/coach_facts.py`**

```python
REGISTRY: list[QuestionType] = [
    QuestionType("why_action", ["why_action"], fetch_why_action),
    QuestionType("fold_equity", ["fold_equity"], fetch_fold_equity),
    QuestionType("villain_range", ["villain_range"], fetch_villain_range),
    QuestionType("hand_strength", ["hand_strength"], fetch_hand_strength),
    # P1 (filled in later tasks; placeholders raise if hit before implemented)
    QuestionType("range_shift", ["range_shift"], lambda c: fetch_range_shift(c)),
    QuestionType("sizing", ["sizing"], lambda c: fetch_sizing(c)),
    QuestionType("hypothetical", ["hypothetical"], lambda c: fetch_hypothetical(c)),
    QuestionType("node_url", ["node_url"], lambda c: fetch_node_url(c)),
]
_BY_INTENT = {qt.id: qt for qt in REGISTRY}

# P1 fetchers default to None until their task lands (registry-ready, no crash).
def fetch_range_shift(ctx): return None
def fetch_sizing(ctx): return None
def fetch_hypothetical(ctx): return None
def fetch_node_url(ctx): return None

_CLASSIFIER = None  # injectable for tests


def _set_intent_classifier(fn):
    global _CLASSIFIER
    _CLASSIFIER = fn


_INTENT_PROMPT = (
    "你是撲克教練問題分類器。讀使用者的 follow-up 問題，輸出『一個』分類標籤，"
    "只能是下列其中之一（只輸出標籤本身，不要解釋）：\n"
    "why_action: 問某手牌為什麼採取某動作（下注/過牌/跟注/棄牌）\n"
    "fold_equity: 問我方下注能讓對手棄掉/跟注哪些牌、棄牌率\n"
    "villain_range: 問對手下注/加注/全下的範圍有哪些牌\n"
    "hand_strength: 問我這手牌的牌力/強弱/equity\n"
    "range_shift: 問某張牌（轉牌/河牌）出現後範圍或牌力如何變化\n"
    "sizing: 問為什麼用這個下注尺寸、該用多大\n"
    "hypothetical: 問『如果…會怎樣』的假設情境\n"
    "node_url: 訊息含 GTO Wizard 連結，要求解釋該節點\n"
    "range_lookup: 單純查詢某位置某街的開牌/範圍頻率\n"
    "other: 以上皆非\n"
)


def classify_intent(question: str, hand_context: dict) -> str:
    if _CLASSIFIER is not None:
        return _CLASSIFIER(question, hand_context)
    if "gtowizard.com" in (question or "") or "gto wizard" in (question or "").lower():
        return "node_url"
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=question)])],
            config=types.GenerateContentConfig(
                system_instruction=_INTENT_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.0, max_output_tokens=8,
            ),
        )
        label = (resp.text or "other").strip().split()[0].lower()
        valid = {qt.id for qt in REGISTRY} | {"range_lookup", "other"}
        return label if label in valid else "other"
    except Exception as e:
        logger.warning(f"coach_facts: intent classify failed: {e}")
        return "other"


_NARRATOR_SYSTEM = (
    "你是繁體中文撲克教練。只能根據下面提供的『事實卡』內容回答，"
    "嚴禁提到事實卡與英雄手牌、公牌以外的任何具體牌（如 AJo、66、AhKh）。"
    "可以用牌型類別（頂對、A高、同花聽）與通用概念（價值、詐唬、阻斷牌、equity）。"
    "用 2-4 句白話、口語、不要列 JSON 或原始數據。"
)


def _narrate(facts: Facts, question: str, extra_vocab: str = "") -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    sys = _NARRATOR_SYSTEM + (f"\n允許提到的具體牌僅限：{extra_vocab}" if extra_vocab else "")
    prompt = f"使用者問題：{question}\n\n事實卡：\n{facts.render()}"
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=sys,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2, max_output_tokens=400,
        ),
    )
    return (resp.text or "").strip()


def render_template(facts: Facts) -> str:
    """Deterministic, fully-grounded prose built from the card (no model)."""
    return facts.title + "\n" + "\n".join(facts.lines) + (
        f"\n（{facts.note}）" if facts.note else "")


def answer_followup(ctx: Ctx) -> str | None:
    """Public entry. Returns a grounded answer, or None for 'other' (caller falls back)."""
    intent = classify_intent(ctx.question, ctx.hand_context)
    if intent in ("other", "range_lookup"):
        return None
    qt = _BY_INTENT.get(intent)
    if not qt:
        return None
    try:
        facts = qt.fetch(ctx)
    except Exception as e:
        logger.warning(f"coach_facts: fetch({intent}) failed: {e}")
        return None
    if not facts or not facts.lines:
        return None
    board = facts.meta.get("board", "") or _question_board(ctx)
    # narrate → verify → regen once → template
    try:
        draft = _narrate(facts, ctx.question)
    except Exception as e:
        logger.warning(f"coach_facts: narrate failed: {e}")
        return render_template(facts)
    v = verify_claims(draft, facts, board, audit_numbers=True)
    if v.ok:
        return _finalize(draft)
    vocab = "、".join(sorted(facts.allowed_claims))
    try:
        draft2 = _narrate(facts, ctx.question, extra_vocab=vocab)
        if verify_claims(draft2, facts, board, audit_numbers=True).ok:
            return _finalize(draft2)
    except Exception:
        pass
    logger.info(f"coach_facts: verifier fallback to template (intent={intent}, "
                f"violations={v.violations})")
    return render_template(facts)


def _question_board(ctx: Ctx) -> str:
    for sol in (ctx.hand_context.get("solutions") or []):
        if sol and (sol.get("game") or {}).get("board"):
            return sol["game"]["board"]
    return ""


def _finalize(text: str) -> str:
    return text.strip()
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `python scripts/regression_test.py 2>&1 | grep -i "registry\|template\|other_fallback"`

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): intent classifier + narrator + verifier loop + answer_followup"
```

---

### Task 8: Wire `coach_facts` into the follow-up path

**Files:**
- Modify: `src/gemini_session.py` (inside `_chat_with_tools`, before the tool loop)
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing test** (route + fallback behavior, classifier stubbed)

```python
@test("gemini_session: follow-up routes through coach_facts when grounded")
def test_session_routes_coach_facts():
    import sys, types as _t
    import coach_facts as cf
    # force a grounded answer
    called = {}
    def fake_answer(ctx):
        called["q"] = ctx.question
        return "GROUNDED_ANSWER"
    orig = cf.answer_followup
    cf.answer_followup = fake_answer
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager()
        mgr.hand_contexts[1] = {"hero_position": "HJ", "hero_hand": "KsJh",
                                "solutions": [], "hero_spots": []}
        out = mgr._try_coach_facts(1, "為什麼這手牌下注？")
        assert_eq(out, "GROUNDED_ANSWER")
        assert_in("下注", called["q"])
    finally:
        cf.answer_followup = orig

@test("gemini_session: _try_coach_facts returns None without cached hand")
def test_session_coach_facts_no_ctx():
    from gemini_session import GeminiSessionManager
    mgr = GeminiSessionManager()
    assert_true(mgr._try_coach_facts(999, "為什麼下注") is None)
```

- [ ] **Step 2: Run, expect FAIL** (`_try_coach_facts` undefined)

- [ ] **Step 3: Add `_try_coach_facts` and call it in `_chat_with_tools`**

In `src/gemini_session.py`, near the top with other imports inside the function or module
(follow existing import style; the repo imports `analyze_hand` lazily — match it):

```python
    def _try_coach_facts(self, chat_id: int, user_text: str,
                         user_id: int | None = None,
                         refresh_token: str | None = None) -> str | None:
        """Deterministic grounded answer for P0/P1 follow-up intents.

        Returns the answer string, or None to fall back to the tool-calling path.
        """
        ctx = self.hand_contexts.get(chat_id)
        if not ctx or not ctx.get("solutions"):
            return None
        try:
            import coach_facts
            answer = coach_facts.answer_followup(coach_facts.Ctx(
                question=user_text, hand_context=ctx,
                user_id=user_id, refresh_token=refresh_token,
            ))
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] coach_facts failed: {e}")
            return None
        if answer:
            self._logger.info(f"[chat={chat_id}] coach_facts grounded answer "
                              f"({len(answer)} chars)")
        return answer
```

Then in `_chat_with_tools`, immediately after `force_tools` is computed and before the
generation `for round_num in range(max_rounds):` loop, insert:

```python
        # Deterministic grounded path for P0/P1 follow-up intents (coach_facts).
        # Only when we have a cached analyzed hand and the grounding gate matched;
        # 'other'/unknown intents return None → keep the existing tool loop below.
        if force_tool_eligible and not disable_tools:
            grounded = self._try_coach_facts(
                chat_id, user_text, user_id=user_id, refresh_token=refresh_token)
            if grounded:
                grounded = _normalize_terms(grounded)
                history = self.histories.get(chat_id, [])
                history.append(types.Content(role="user",
                                             parts=[types.Part(text=user_text)]))
                history.append(types.Content(role="model",
                                             parts=[types.Part(text=grounded)]))
                self.histories[chat_id] = history[-20:]
                return grounded
```

Note: verify the exact local variable names (`force_tool_eligible`, `disable_tools`,
`user_id`, `refresh_token`, `self._logger`) against the real `_chat_with_tools` signature
before editing; adapt to match. Keep the insertion strictly before the first model call.

- [ ] **Step 4: Run tests, expect PASS**

Run: `python scripts/regression_test.py 2>&1 | grep -i "routes_coach_facts\|no_ctx"`

- [ ] **Step 5: Commit**

```bash
git add src/gemini_session.py scripts/regression_test.py
git commit -m "feat(coach): route P0/P1 follow-ups through coach_facts, tool path as fallback"
```

---

### Task 9: Golden cases from the 3 real failures

**Files:**
- Modify: `scripts/regression_test.py`

These assert the anti-hallucination invariant deterministically: given a fetched `Facts`
card, (a) the verifier passes a draft that only uses allowed vocabulary, and (b) flags a draft
that injects the exact invented combos from the real logs (AJo/AQo/ATo for KTo-bet; KJs/KTs/QJs
for A3-vs-Q9; QJs/QTs/JTs/AQs/AJs for JJ-vs-66) UNLESS those combos are in the card.

- [ ] **Step 1: Write tests**

```python
@test("coach_facts golden: invented combos are flagged unless grounded")
def test_coach_facts_golden_invented():
    import coach_facts as cf
    # simulate a fold-equity card that does NOT mention AQo/ATo
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KdTc"),
                     meta={"board": "9h8s2s"})
    board = "9h8s2s"
    # the historical hallucination named AJo/AQo/ATo — only AJo is grounded
    invented = "BB 範圍裡有大量 AJo、AQo、ATo 會棄牌。"
    v = cf.verify_claims(invented, facts, board)
    assert_true(not v.ok, "ungrounded AQo/ATo flagged")
    assert_in("AQo", v.violations)
    assert_in("ATo", v.violations)
    assert_true("AJo" not in v.violations, "grounded AJo allowed")
    # a grounded rewrite passes
    good = "對手 A高（例如 AJo）大多會棄牌，整體棄牌率約 80%。"
    assert_true(cf.verify_claims(good, facts, board).ok, "category-level + grounded example passes")

@test("coach_facts golden: A3-vs-Q9 outs invention flagged")
def test_coach_facts_golden_outs():
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t",
                     lines=["equity 41%"], allowed_claims=cf.canonical_forms("Q9s"),
                     meta={"board": "Kc7h2d"})
    bad = "對手 KJs、KTs、QJs、JTs 有 6 outs。"
    v = cf.verify_claims(bad, facts, "Kc7h2d")
    assert_true(not v.ok and "KJs" in v.violations, "invented draws flagged")
```

- [ ] **Step 2: Run, expect PASS** (verifier already implemented)

Run: `python scripts/regression_test.py 2>&1 | grep -i golden

- [ ] **Step 3: Commit**

```bash
git add scripts/regression_test.py
git commit -m "test(coach): golden anti-hallucination cases from 3 real failures"
```

---

### Task 10 (P1): Type H sizing + Type F range-shift fetch

**Files:**
- Modify: `scripts/coach_facts.py` (replace `fetch_sizing`, `fetch_range_shift` stubs)
- Test: `scripts/regression_test.py`

- [ ] **Step 1: Write failing tests**

```python
@test("coach_facts P1: fetch_sizing lists solver bet sizes + frequencies")
def test_coach_facts_sizing():
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_sizing_from(hero, hctx)
    assert_true(facts is not None and facts.intent == "sizing")
    assert_true(any("%" in ln for ln in facts.lines), "sizes have freqs")

@test("coach_facts P1: fetch_range_shift compares two streets' hero equity")
def test_coach_facts_range_shift():
    cf, hctx, hero, villain = _load_coach_ctx()
    # single-street fixture: range_shift returns None gracefully (needs >=2 streets)
    out = cf.fetch_range_shift(cf.Ctx(question="轉牌 As 之後牌力怎麼變", hand_context=hctx))
    assert_true(out is None or out.intent == "range_shift")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement (replace stubs)**

```python
def _fetch_sizing_from(hsol: dict, hand_context: dict) -> Facts | None:
    if not hsol:
        return None
    board = (hsol.get("game") or {}).get("board") or ""
    actor = _acting_position(hsol)
    hero_hand = hand_context.get("hero_hand")
    rows = []
    for asol in hsol.get("action_solutions") or []:
        act = asol["action"]
        code = act["code"]
        if code in ("X", "C", "F"):
            continue
        rows.append((code, act.get("betsize_by_pot"), asol.get("total_frequency") or 0.0))
    rows = [r for r in rows if r[2] > 0.005]
    if not rows:
        return None
    facts = Facts(intent="sizing", title=f"{actor} 在 {board} 的下注尺寸選擇：", lines=[])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    for code, bypot, fr in rows:
        sz = f"{_pct(bypot)}% 底池" if bypot else f"{code[1:]}bb"
        facts.lines.append(f"  {sz}：頻率 {_pct(fr)}%")
        facts.numbers.add(_pct(fr))
    facts.meta = {"board": board, "rows": rows}
    return facts


def fetch_sizing(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    return _fetch_sizing_from(sol, ctx.hand_context)


def fetch_range_shift(ctx: Ctx) -> Facts | None:
    """Compare hero equity/percentile across the two most recent postflop streets."""
    hc = ctx.hand_context
    pairs = [(sp, s) for sp, s in zip(hc.get("hero_spots") or [], hc.get("solutions") or [])
             if s and sp.get("street") in ("flop", "turn", "river")]
    if len(pairs) < 2:
        return None
    hero = hc.get("hero_position")
    hero_hand = hc.get("hero_hand")
    (sp0, s0), (sp1, s1) = pairs[-2], pairs[-1]
    e0 = _hero_eq_vs_range(s0, hero, hero_hand)
    e1 = _hero_eq_vs_range(s1, hero, hero_hand)
    if not e0 or not e1:
        return None
    b0 = (s0.get("game") or {}).get("board") or ""
    b1 = (s1.get("game") or {}).get("board") or ""
    facts = Facts(intent="range_shift",
                  title=f"{hero} {hero_hand} 牌力變化：",
                  lines=[f"  {sp0['street']} ({b0})：equity {e0[0]}%、percentile {e0[1]}%",
                         f"  {sp1['street']} ({b1})：equity {e1[0]}%、percentile {e1[1]}%",
                         f"  變化：equity {e1[0]-e0[0]:+d}pp"])
    facts.allowed_claims |= canonical_forms(hero_hand or "")
    facts.numbers |= {e0[0], e0[1], e1[0], e1[1]}
    facts.meta = {"board": b1, "from": e0, "to": e1}
    return facts
```

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): P1 type H (sizing) + F (range-shift) fetch"
```

---

### Task 11 (P1): Type G hypothetical (+ off-tree size rejection) + Type I node-URL

**Files:**
- Modify: `scripts/coach_facts.py` (replace `fetch_hypothetical`, `fetch_node_url`)
- Test: `scripts/regression_test.py`

**G design:** A hypothetical that changes a *size* ("如果下注更大") is answered by mapping the
requested size to the nearest on-tree action; if the requested pot-ratio is off-tree (no action
within tolerance) we REJECT with a grounded note rather than invent. We reuse the current node's
`action_solutions` (already grounded) and report the closest size's data. Hypotheticals that
change cards reuse `fetch_range_shift`; ones that change the line are out of P1 scope → return a
note that solver tree doesn't cover it (still grounded, no invention).

**I design:** A pasted GTO Wizard URL is parsed for board + action params; if it matches the
analyzed hand's tree we explain that node via the existing fetchers; otherwise we return a
grounded note. URL parsing reuses the patterns from `docs/.../gtow-custom-spot-urls` conventions
(query params: `gametype, depth, board, preflop_actions, flop_actions, ...`).

- [ ] **Step 1: Write failing tests**

```python
@test("coach_facts P1: hypothetical maps requested size to nearest on-tree, rejects off-tree")
def test_coach_facts_hypothetical():
    cf, hctx, hero, villain = _load_coach_ctx()
    # ask for a 50% pot bet — find nearest on-tree size
    facts = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=0.5)
    assert_true(facts is not None and facts.intent == "hypothetical")
    # absurd size off-tree → note set, no invented combos
    far = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=9.9)
    assert_true(far is None or far.note, "off-tree flagged")

@test("coach_facts P1: node_url parses GTO Wizard link params")
def test_coach_facts_node_url():
    import coach_facts as cf
    p = cf._parse_gtow_url(
        "https://app.gtowizard.com/solutions?gametype=MTTGeneral&depth=40.125"
        "&board=Ks9s2h&preflop_actions=F-F-R2-F-C-F&flop_actions=X-R1.4")
    assert_eq(p["board"], "Ks9s2h")
    assert_eq(p["flop_actions"], "X-R1.4")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement (replace stubs)**

```python
import urllib.parse as _urlparse


def _parse_gtow_url(url: str) -> dict:
    try:
        q = _urlparse.urlparse(url).query
        params = {k: v[0] for k, v in _urlparse.parse_qs(q).items()}
    except Exception:
        return {}
    keep = ("gametype", "depth", "stacks", "board", "preflop_actions",
            "flop_actions", "turn_actions", "river_actions")
    return {k: params[k] for k in keep if k in params}


def _fetch_hypothetical_size_from(hsol: dict, hand_context: dict,
                                  target_pot_ratio: float) -> Facts | None:
    if not hsol:
        return None
    board = (hsol.get("game") or {}).get("board") or ""
    hero_hand = hand_context.get("hero_hand")
    best = None
    for asol in hsol.get("action_solutions") or []:
        bp = asol["action"].get("betsize_by_pot")
        if bp is None:
            continue
        d = abs(bp - target_pot_ratio)
        if best is None or d < best[0]:
            best = (d, asol)
    if not best or best[0] > 0.25:  # off-tree tolerance
        f = Facts(intent="hypothetical", title=f"假設下注尺寸（{board}）：",
                  lines=["  此尺寸不在 solver 樹中，無法提供可靠數據。"],
                  note="off-tree 尺寸，已避免臆測")
        f.allowed_claims |= canonical_forms(hero_hand or "")
        return f
    asol = best[1]
    code = asol["action"]["code"]
    f = Facts(intent="hypothetical",
              title=f"最接近 {_pct(target_pot_ratio)}% 底池的 solver 尺寸（{board}）：",
              lines=[f"  {code}：整體頻率 {_pct(asol.get('total_frequency') or 0)}%"])
    f.allowed_claims |= canonical_forms(hero_hand or "")
    f.numbers.add(_pct(asol.get("total_frequency") or 0))
    f.meta = {"board": board, "code": code}
    return f


_RE_POT_RATIO = re.compile(r"(\d{1,3})\s*%")


def fetch_hypothetical(ctx: Ctx) -> Facts | None:
    spot, sol = _hero_spot_and_sol(ctx, _street_from_question(ctx.question))
    if not sol:
        return None
    m = _RE_POT_RATIO.search(ctx.question or "")
    if m:
        return _fetch_hypothetical_size_from(sol, ctx.hand_context, int(m.group(1)) / 100.0)
    # card-change hypothetical → reuse range_shift if it applies
    rs = fetch_range_shift(ctx)
    if rs:
        rs.intent = "hypothetical"
        return rs
    f = Facts(intent="hypothetical", title="假設情境：",
              lines=["  此假設情境超出目前 solver 樹涵蓋範圍，無法提供可靠數據。"],
              note="超出 solver 樹")
    f.allowed_claims |= canonical_forms(ctx.hand_context.get("hero_hand") or "")
    return f


def fetch_node_url(ctx: Ctx) -> Facts | None:
    import re as _re
    m = _re.search(r"https?://\S*gtowizard\.com\S*", ctx.question or "")
    if not m:
        return None
    params = _parse_gtow_url(m.group(0))
    if not params.get("board"):
        return None
    try:
        sol = get_spot_solution(
            gametype=params.get("gametype", "MTTGeneral"),
            depth=float(params.get("depth", 30.125)),
            stacks=params.get("stacks", ""),
            preflop_actions=params.get("preflop_actions", ""),
            board=params.get("board", ""),
            flop_actions=params.get("flop_actions", ""),
            turn_actions=params.get("turn_actions", ""),
            river_actions=params.get("river_actions", ""))
    except Exception as e:
        logger.warning(f"coach_facts: node_url fetch failed: {e}")
        return None
    if not sol:
        return None
    actor = _acting_position(sol)
    board = (sol.get("game") or {}).get("board") or params["board"]
    table = _category_action_table(sol, top_n=5)
    facts = Facts(intent="node_url",
                  title=f"連結節點 {actor} 在 {board} 的策略：", lines=[])
    for name, freq, actions in table:
        line = f"  {_cat_zh(name)} 佔 {_pct(freq)}% — {_fmt_actions(actions)}"
        reps = _rep_classes_for_category(sol, name, top_k=2)
        for c, fr, a in reps:
            facts.allowed_claims |= canonical_forms(c)
        if reps:
            line += "   例：" + "、".join(c for c, _, _ in reps)
        facts.lines.append(line)
        facts.numbers |= {_pct(v) for v in actions.values()}
    facts.meta = {"board": board}
    return facts
```

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/coach_facts.py scripts/regression_test.py
git commit -m "feat(coach): P1 type G (hypothetical+off-tree reject) + I (node-URL explain)"
```

---

### Task 12: Full numeric-claim audit (P1) — already wired, add tests + tighten

**Files:**
- Modify: `scripts/regression_test.py` (and `scripts/coach_facts.py` if a tolerance tweak is needed)

- [ ] **Step 1: Write tests**

```python
@test("coach_facts P1: numeric audit flags grossly wrong percentages")
def test_coach_facts_numeric_audit():
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t",
                     lines=["A高 棄牌 80%"], allowed_claims=cf.canonical_forms("KsJh"),
                     numbers={80, 20}, meta={"board": "Ks9s2h"})
    ok = cf.verify_claims("對手約 80% 會棄牌。", facts, "Ks9s2h", audit_numbers=True)
    assert_true(ok.ok, "matching number passes")
    bad = cf.verify_claims("對手約 35% 會棄牌。", facts, "Ks9s2h", audit_numbers=True)
    assert_true(not bad.ok and 35 in bad.number_violations, "gross-mismatch flagged")
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit**

```bash
git add scripts/regression_test.py scripts/coach_facts.py
git commit -m "test(coach): P1 numeric-claim audit coverage"
```

---

### Task 13: Live smoke test (optional, gated on API key) + full suite + docs

**Files:**
- Modify: `scripts/regression_test.py` (skip-if-no-key live narrator smoke)
- Modify: `docs/superpowers/specs/2026-06-07-coach-followup-grounding-design.md` (mark P0/P1 status)

- [ ] **Step 1: Add a guarded live smoke test**

```python
@test("coach_facts live: narrated answer is grounded (skips without GEMINI_API_KEY)")
def test_coach_facts_live_smoke():
    import os
    if not os.getenv("GEMINI_API_KEY"):
        return  # skip
    cf, hctx, hero, villain = _load_coach_ctx()
    cf._set_intent_classifier(lambda q, c: "hand_strength")
    out = cf.answer_followup(cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx))
    cf._set_intent_classifier(None)
    assert_true(out and len(out) > 5, "produced an answer")
    facts = cf.fetch_hand_strength(cf.Ctx(question="x", hand_context=hctx))
    board = facts.meta.get("board", "")
    v = cf.verify_claims(out, facts, board)
    assert_true(v.ok, f"live answer grounded (violations={v.violations})")
```

- [ ] **Step 2: Run full regression suite**

Run: `python scripts/regression_test.py`
Expected: all coach_facts tests pass; the 5 pre-existing environmental failures
(missing `data/pokercraft_corpus/...` fixtures + button-detector fixture + H2494 `.gto_cache`
drift) remain unchanged. No NEW failures.

- [ ] **Step 3: Update spec status + commit**

Edit the spec header `Status:` to `P0 + P1 implemented` and append an "Implemented" note
listing the registry types now live. Then:

```bash
git add scripts/regression_test.py docs/superpowers/specs/2026-06-07-coach-followup-grounding-design.md
git commit -m "test(coach): live narrator smoke + mark spec P0/P1 implemented"
```

---

### Task 14: Open PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/coach-followup-grounding-impl
```

- [ ] **Step 2: Create PR** (base `main`) summarizing: new `scripts/coach_facts.py`
deterministic routing, P0 types B/C/D/E, P1 types F/G/H/I + numeric audit, hard verifier,
`gemini_session` integration with existing tool path as `other` fallback, fixtures + golden +
unit tests. Note latency win and the pre-existing environmental test failures.

---

## Self-review notes

- **Spec coverage:** B→Task4, C→Task5, D→Task5, E→Task4; verifier→Task6; classifier/narrator/
  template→Task7; integration→Task8; golden→Task9; P1 F/H→Task10, G/I→Task11, numeric audit→
  Task12; testing/docs→Task13; PR→Task14. Registry extensibility realized via `REGISTRY` list.
- **Type consistency:** `Facts(intent,title,lines,allowed_claims,numbers,meta,note)`,
  `Verdict(ok,violations,number_violations)`, `Ctx(question,hand_context,user_id,refresh_token)`,
  `QuestionType(id,matches,fetch)` used identically across tasks. Helper names
  (`_acting_position`, `_category_action_table`, `_rep_classes_for_category`,
  `_class_category_map`, `_hero_combo_facts`, `_hero_eq_vs_range`, `_hero_spot_and_sol`,
  `canonical_forms`, `extract_combo_tokens`, `verify_claims`, `render_template`,
  `answer_followup`, `classify_intent`, `_set_intent_classifier`) are consistent.
- **Risk (node resolution):** C fetches the villain response node live (cached after first
  call); D/sizing use existing hero spots. Off-tree/low-freq → `note` flag, never invention.
- **Decision:** `allowed_claims` is stored on `Facts` (built inside each `fetch`) rather than a
  separate `QuestionType.allowed_claims` callable as sketched in the spec — simpler, keeps the
  verifier type-agnostic, same guarantee.
```
