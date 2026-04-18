"""Build GTO Wizard custom-spot practice URLs from parsed hand data.

This module produces the precise `fh_start_spot=custom_spot` deep-links used
by the weekly leak report, replacing the coarse bucket shortcuts emitted by
gtow_trainer_url.build_trainer_url.

Fallback contract: any exception raised from build_custom_spot_url is the
caller's cue to fall back to the bucket URL. The builder itself never catches;
callers that want soft failure wrap in try/except.
"""
from __future__ import annotations

from typing import Literal


def _split_board(board: str) -> list[tuple[str, str]]:
    """'4c6h8h' → [('4','c'),('6','h'),('8','h')]. Empty → []."""
    board = (board or "").strip()
    if not board:
        return []
    if len(board) % 2 != 0:
        raise ValueError(f"board length must be even, got {board!r}")
    return [(board[i], board[i + 1]) for i in range(0, len(board), 2)]


def _paired_flag(cards: list[tuple[str, str]]) -> str:
    """'not_paired' | 'paired' | 'tripled' — matches GTOW flop_paired vocab.

    Turn/river only get 'not_paired' | 'paired' (no 'tripled' there per spec),
    but the same trips detection on a 4/5-card board that contains a three-of-
    a-kind subset is still treated as 'paired' by GTOW's turn_paired/river_paired.
    Caller decides which street key to emit — we just do the detection.
    """
    ranks = [r for r, _ in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    max_count = max(counts.values()) if counts else 0
    if max_count >= 3:
        return "tripled"
    if max_count == 2:
        return "paired"
    return "not_paired"


def _suit_flag_flop(cards: list[tuple[str, str]]) -> str:
    """flop_suits: 'rainbow' | 'flush_draw' | 'monotone'."""
    suits = [s for _, s in cards]
    max_count = max(suits.count(s) for s in set(suits)) if suits else 0
    if max_count >= 3:
        return "monotone"
    if max_count == 2:
        return "flush_draw"
    return "rainbow"


def _suit_flag_turn_river(cards: list[tuple[str, str]]) -> str:
    """turn_suit / river_suit: 'rainbow' | 'backdoor' | 'flush'.

    Different vocab from flop. 'backdoor' = max suit count is exactly 2.
    'flush' = max suit count >= 3 (flush possible on this board state).
    On river, 'rainbow' is unreachable (5 cards into 4 suits, pigeonhole
    guarantees max >= 2) — but we still emit it for correctness.
    """
    suits = [s for _, s in cards]
    max_count = max(suits.count(s) for s in set(suits)) if suits else 0
    if max_count >= 3:
        return "flush"
    if max_count == 2:
        return "backdoor"
    return "rainbow"


_RANK_ORDER = "23456789TJQKA"


def _connectedness_flag(cards: list[tuple[str, str]]) -> str:
    """flop_connectedness: 'connected' | 'oesd_possible' | 'disconnected'.

    Only meaningful on the 3-card flop (GTOW exposes no turn/river
    connectedness filter). Classification by gaps between sorted ranks:
        [1,1] → connected (e.g., 789)
        any gap of 1 but not [1,1] → oesd_possible (e.g., 78T, 67T, 89J with gap)
        no gap of 1 → disconnected (e.g., 468, 259)

    A is treated as high (rank index 12). A-low wheel boards (A-2-3) may be
    classified as oesd_possible instead of connected — documented edge case,
    low impact, URL still loads.
    """
    if len(cards) < 3:
        return ""
    ranks = sorted(_RANK_ORDER.index(r) for r, _ in cards[:3])
    gaps = [ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)]
    if gaps == [1, 1]:
        return "connected"
    if 1 in gaps:
        return "oesd_possible"
    return "disconnected"


def classify_board(board: str) -> dict[str, str]:
    """Classify a board string into GTOW custom-spot board-texture flags.

    Returns keys for whatever streets are present on the board:
      - flop_paired  (not_paired|paired|tripled) / flop_suits  (rainbow|flush_draw|monotone)   — if ≥3 cards
      - turn_paired  (not_paired|paired)         / turn_suit   (rainbow|backdoor|flush)        — if ≥4 cards
      - river_paired (not_paired|paired)         / river_suit  (rainbow|backdoor|flush)        — if ≥5 cards

    Flag names and value vocab are authoritative per GTOW frontend JS:
        flop_suits:   ["rainbow", "flush_draw", "monotone"]
        flop_paired:  ["not_paired", "paired", "tripled"]
        turn_suit:    ["rainbow", "backdoor", "flush"]
        turn_paired:  ["not_paired", "paired"]
        river_suit:   ["rainbow", "backdoor", "flush"]
        river_paired: ["not_paired", "paired"]

    The turn/river paired vocab does not include "tripled" per the spec, so
    even on a board like 7h7d7s2c the turn_paired value is "paired". (The
    flop_paired for that same board IS "tripled" because that's flop state.)
    """
    cards = _split_board(board)
    out: dict[str, str] = {}
    if len(cards) >= 3:
        flop = cards[:3]
        out["flop_paired"]        = _paired_flag(flop)
        out["flop_suits"]         = _suit_flag_flop(flop)
        out["flop_connectedness"] = _connectedness_flag(flop)
    if len(cards) >= 4:
        turn = cards[:4]
        # Collapse 'tripled' → 'paired' for turn/river per GTOW vocab
        p = _paired_flag(turn)
        out["turn_paired"] = "paired" if p == "tripled" else p
        out["turn_suit"]   = _suit_flag_turn_river(turn)
    if len(cards) >= 5:
        river = cards[:5]
        p = _paired_flag(river)
        out["river_paired"] = "paired" if p == "tripled" else p
        out["river_suit"]   = _suit_flag_turn_river(river)
    return out
