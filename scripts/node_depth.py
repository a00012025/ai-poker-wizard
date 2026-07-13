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
                          hero_start, stacks, is_icm, default_effective=None):
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
    # ---- first hero node ----
    # A first voluntary hero action may already face an open/jam. Resolve the
    # aggressor instead of assigning hero/list-row depth to every first node.
    first_aggressor = None
    involved_before_hero = set()
    for actor, code, _prefix in _replay(parts, position_order):
        if actor == hidx:
            break
        if code not in ("F", ""):
            involved_before_hero.add(actor)
        if code.startswith("R") or code.startswith("AI"):
            first_aggressor = (code, actor)
    if first_aggressor is not None:
        a_code, a_idx = first_aggressor
        if a_code.startswith("AI"):
            total = _jam_total(a_code, stacks, position_order, a_idx)
            kind = "facing_allin"
        else:
            total = (stacks or {}).get(position_order[a_idx])
            kind = "facing_raise"
        # In a side-pot/multiway node a short all-in is not necessarily the
        # solver avatar binding hero's decision (cd23771b: opener + 2.5bb AI +
        # caller, hero squeezes 16bb on the 17bb tree). Keep the known played
        # effective when multiple opponents already entered the pot.
        if len(involved_before_hero) > 1 and default_effective:
            total = None
        fallback = default_effective or hero_start
        eff = round(min(hero_start, total), 1) if total else round(fallback, 1)
        nodes.append({
            "node": kind, "eff": eff,
            "depth_bucket": int(nearest_depth(eff)),
            "caveat": None, "aggressor_code": a_code,
        })
    else:
        live_cover = [
            (stacks or {}).get(p)
            for i, p in enumerate(position_order)
            if p != hero_position
            and (i > hidx or (i < len(parts) and parts[i] not in ("F", "")))
            and (stacks or {}).get(p)
        ]
        # ``effective_bb`` supplied by the parser/importer already identifies
        # the binding played opponent. Prefer it when available; max-live-cover
        # remains the fallback for callers that only know physical stacks.
        open_eff = (
            default_effective
            if default_effective
            else (min(hero_start, max(live_cover)) if live_cover else hero_start)
        )
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
