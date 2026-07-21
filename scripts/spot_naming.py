"""Compact, shared names for practice-queue spots and GTOW Drills."""
from __future__ import annotations

import re
from typing import Mapping

from action_bias import BIAS_LABELS


_STREET_ZH = {"flop": "翻牌", "turn": "轉牌", "river": "河牌"}
_FACING = {
    "first_to_act": "首動",
    "vs_check": "vs X",
    "vs_bet": "vs Bet",
    "vs_raise": "vs XR",
}


def _postflop_name(row: Mapping, leaf: str, category: str) -> str | None:
    parts = leaf.split(":")
    if len(parts) < 5:
        return None
    street = category if category in _STREET_ZH else parts[0]
    if street not in _STREET_ZH:
        return None
    pot = parts[1]
    positions = parts[2]
    hero_leaf = positions.split("v", 1)[0] if "v" in positions else positions
    hero = str(row.get("hero_cat") or hero_leaf or "").strip()
    rel = str(row.get("ip_oop") or parts[3] or "").strip()
    facing = _FACING.get(parts[-1], parts[-1])
    actor = " ".join(part for part in (hero, rel) if part and part != "?")
    name = f"{pot}｜{actor}｜{_STREET_ZH[street]} {facing}"

    match = re.search(r":\[([^\]]*)\]:", leaf)
    if match:
        seqs = match.group(1).split("|", 1)
        prior = []
        if seqs and seqs[0] not in {"", "-", "None", "null"}:
            prior.append(f"翻牌 {seqs[0]}")
        if (street == "river" and len(seqs) > 1
                and seqs[1] not in {"", "-", "None", "null"}):
            prior.append(f"轉牌 {seqs[1]}")
        if prior:
            name += "｜" + "／".join(prior)
    return name


def _preflop_parts(row: Mapping, leaf: str, marker: str) -> tuple[str, str, str]:
    token = f"_{marker}"
    if token not in leaf:
        return "", "", ""
    hero_leaf, suffix = leaf.split(token, 1)
    tail = [part for part in suffix.strip("_").split("_") if part]
    rel_leaf = tail.pop() if tail and tail[-1] in {"IP", "OOP"} else ""
    villain_leaf = tail[0].removeprefix("v") if tail else ""
    hero = str(row.get("hero_cat") or row.get("hero_pos") or hero_leaf or "")
    villain = str(row.get("villain_cat") or villain_leaf or "")
    rel = str(row.get("ip_oop") or rel_leaf or "")
    return hero, villain, rel


def _preflop_name(row: Mapping, leaf: str, category: str) -> str | None:
    if category == "RFI" or leaf.endswith("_RFI"):
        hero = str(row.get("hero_pos") or leaf.removesuffix("_RFI") or "")
        return f"{hero} RFI" if hero else None

    specs = {
        "vsOpen": ("vsOpen", "Open", False),
        "vsRaiseCall": ("vsRaiseCall", "Open+Call", False),
        "vsSqueeze": ("vsSqueeze", "Squeeze", False),
        "vs3bet": ("vs3bet", "3bet", False),
        "vsCold3bet": ("vsCold3bet", "3bet", True),
        "vs4bet": ("vs4bet", "4bet", False),
        "vsCold4bet": ("vsCold4bet", "4bet", True),
    }
    spec = specs.get(category)
    if not spec:
        return None
    marker, action, cold = spec
    hero, villain, rel = _preflop_parts(row, leaf, marker)
    flat = False
    if category == "vsSqueeze" and "flat_vsSqueeze" in leaf:
        m = re.match(r"([^_]+)flat_vsSqueeze(?:_v([^_]+))?(?:_([^_]+))?", leaf)
        if m:
            hero = str(row.get("hero_cat") or row.get("hero_pos") or m.group(1) or "")
            villain = str(row.get("villain_cat") or m.group(2) or "")
            rel = str(row.get("ip_oop") or m.group(3) or "")
        else:
            hero, villain, rel = _preflop_parts(row, leaf, "flat_vsSqueeze")
        flat = True
    actor = " ".join(part for part in (hero, rel) if part and part != "?")
    target = " ".join(part for part in (villain, action) if part)
    if flat:
        return f"{actor} flat vs {target}" if actor and target else None
    if cold:
        return f"{actor}｜Cold vs {target}" if actor and target else None
    return f"{actor} vs {target}" if actor and target else None


def compact_spot_name(row: Mapping) -> str:
    """Return one terse spot name shared by queue, detail, and GTOW Drill."""
    leaf = str(row.get("spot_leaf") or "").strip()
    category = str(row.get("spot_category") or row.get("street") or "").strip()
    name = (_postflop_name(row, leaf, category)
            if category in _STREET_ZH or leaf.split(":", 1)[0] in _STREET_ZH
            else _preflop_name(row, leaf, category))
    if name:
        return name
    legacy = str(row.get("label") or leaf or "?").strip()
    return legacy.split("｜", 1)[0].strip() or "?"


def telegram_bias_summary(row: Mapping) -> str:
    """Bias is Telegram coaching context, never part of the Drill name."""
    direction = row.get("bias_direction")
    label = BIAS_LABELS.get(direction)
    if not label:
        return ""
    n = int(row.get("bias_n") or 0)
    ev = float(row.get("bias_ev_loss_bb") or 0.0)
    if n and ev:
        return f"明顯傾向：{label}（{n} 手，共損失 {ev:.2f}bb）"
    return f"明顯傾向：{label}"
