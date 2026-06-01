"""Generate 13x13 poker range grid images from solver data.

Each cell represents one of the 169 starting hands (AA top-left, 22 bottom-right).
Colors encode actions: fold=gray, call=green, raise=blue, big raise/all-in=red.
Mixed strategies shown as proportional color splits within each cell.
"""
from __future__ import annotations

import io
from PIL import Image, ImageDraw, ImageFont

# Standard poker grid layout
RANKS = "AKQJT98765432"
CELL = 52          # cell size in pixels
PAD = 2            # padding between cells
HEADER = 20        # top/left header for rank labels
FONT_SIZE = 14
HAND_FONT_SIZE = 11


def _hand_name(row: int, col: int) -> str:
    """Return hand name for grid position."""
    r1, r2 = RANKS[row], RANKS[col]
    if row == col:
        return f"{r1}{r2}"
    elif row < col:
        return f"{r1}{r2}s"
    else:
        return f"{r2}{r1}o"


# Action colors (RGB)
COLOR_FOLD = (40, 60, 100)      # dark blue
COLOR_NOT_IN_RANGE = (50, 50, 50)  # dark gray — hand not in range at all
COLOR_CALL = (34, 139, 34)      # forest green
COLOR_RAISE = (220, 140, 30)    # orange
COLOR_BIG_RAISE = (200, 40, 40) # red
COLOR_BG = (25, 25, 25)         # background
COLOR_TEXT = (255, 255, 255)     # white text
COLOR_TEXT_DIM = (180, 180, 180) # dimmed text for fold
COLOR_HEADER = (200, 200, 200)  # header text
COLOR_GRID = (50, 50, 50)       # grid lines


def _classify_actions(actions_freq: dict, action_solutions: list | None = None
                      ) -> list[tuple[str, float, tuple]]:
    """Classify actions into at most 4 categories with colors.

    Returns list of (label, freq, color) sorted by freq descending.
    Groups raise sizes into at most 2 buckets: raise and big_raise/all-in.
    """
    fold_f = actions_freq.get("F", 0)
    call_f = actions_freq.get("C", 0) + actions_freq.get("X", 0)

    # Collect all raise actions
    raise_actions = []
    for code, freq in actions_freq.items():
        if code in ("F", "C", "X"):
            continue
        if freq < 0.005:
            continue
        # Determine if this is a big raise (≥75% pot or all-in)
        is_big = code == "RAI"
        if not is_big and action_solutions:
            for asol in action_solutions:
                if asol["action"]["code"] == code:
                    pct = float(asol["action"].get("betsize_by_pot") or 0)
                    if pct >= 0.70 or asol["action"].get("allin"):
                        is_big = True
                    break
        raise_actions.append((code, freq, is_big))

    # Group into small raise and big raise
    small_raise_f = sum(f for _, f, big in raise_actions if not big)
    big_raise_f = sum(f for _, f, big in raise_actions if big)

    result = []
    if fold_f >= 0.005:
        result.append(("fold", fold_f, COLOR_FOLD))
    if call_f >= 0.005:
        result.append(("call", call_f, COLOR_CALL))
    if small_raise_f >= 0.005:
        result.append(("raise", small_raise_f, COLOR_RAISE))
    if big_raise_f >= 0.005:
        result.append(("big", big_raise_f, COLOR_BIG_RAISE))

    result.sort(key=lambda x: -x[1])
    return result


def _legend_labels(action_solutions: list | None, game: dict | None
                   ) -> tuple[str, str, bool]:
    """Pick legend labels (passive, aggressive, show_fold) for the node.

    A first-to-act spot (Check available, no Call) must read Check/Bet, while a
    spot facing a bet reads Call/Raise. Preflop aggression is conventionally a
    "raise" (over the forced blind), so it stays Raise even when first-in.
    Returns (passive_label, aggressive_label, show_fold).
    """
    codes = {
        a["action"]["code"]
        for a in (action_solutions or [])
        if a.get("action", {}).get("code")
    }
    can_check = "X" in codes
    has_fold = "F" in codes
    street = ((game or {}).get("current_street", {}) or {}).get("type", "") or ""
    is_preflop = street.lower() == "preflop"

    passive_label = "Check" if can_check else "Call"
    aggressive_label = "Bet" if (can_check and not is_preflop) else "Raise"
    return passive_label, aggressive_label, has_fold


def generate_range_grid(spot_solution: dict, position: str,
                        title: str = "") -> bytes:
    """Generate a 13x13 range grid image from solver data.

    Args:
        spot_solution: solver spot_solution dict with players_info
        position: position to show (e.g., "BB", "CO")
        title: optional title text above the grid

    Returns:
        PNG image as bytes
    """
    # Extract hand data
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break
    if not player_info:
        raise ValueError(f"Position {position} not found in solution")

    shc = player_info.get("simple_hand_counters", {})
    action_solutions = spot_solution.get("action_solutions", [])

    # Image dimensions
    grid_size = 13 * CELL + 12 * PAD
    title_h = 30 if title else 0
    legend_h = 30
    img_w = HEADER + grid_size + 10
    img_h = title_h + HEADER + grid_size + legend_h + 5

    img = Image.new("RGB", (img_w, img_h), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Try to load a good font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", HAND_FONT_SIZE)
    except (OSError, IOError):
        font = ImageFont.load_default()
        small_font = font

    # Title
    if title:
        draw.text((img_w // 2, 5), title, fill=COLOR_HEADER, font=font, anchor="mt")

    # Draw header labels
    for i, rank in enumerate(RANKS):
        x = HEADER + i * (CELL + PAD) + CELL // 2
        draw.text((x, title_h + 2), rank, fill=COLOR_HEADER, font=small_font, anchor="mt")
        y = title_h + HEADER + i * (CELL + PAD) + CELL // 2
        draw.text((HEADER // 2, y), rank, fill=COLOR_HEADER, font=small_font, anchor="mm")

    # Draw cells
    for row in range(13):
        for col in range(13):
            hand = _hand_name(row, col)
            x0 = HEADER + col * (CELL + PAD)
            y0 = title_h + HEADER + row * (CELL + PAD)

            hand_data = shc.get(hand)
            if not hand_data:
                # Not in range at all — gray cell
                draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], fill=COLOR_NOT_IN_RANGE)
                draw.text((x0 + CELL // 2, y0 + CELL // 2), hand,
                          fill=COLOR_TEXT_DIM, font=small_font, anchor="mm")
                continue

            actions_freq = hand_data.get("actions_total_frequencies", {})
            categories = _classify_actions(actions_freq, action_solutions)

            if not categories:
                draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], fill=COLOR_NOT_IN_RANGE)
                draw.text((x0 + CELL // 2, y0 + CELL // 2), hand,
                          fill=COLOR_TEXT_DIM, font=small_font, anchor="mm")
                continue

            # Draw proportional vertical color bars (left to right).
            # Order: raise/big_raise first (left), then call, then fold (right).
            priority = {"big": 0, "raise": 1, "call": 2, "fold": 3}
            categories.sort(key=lambda x: priority.get(x[0], 9))

            drawn_w = 0
            for j, (label, freq, color) in enumerate(categories):
                bar_w = round(freq * CELL)
                if j == len(categories) - 1:
                    bar_w = CELL - drawn_w
                if bar_w <= 0:
                    continue
                draw.rectangle([x0 + drawn_w, y0, x0 + drawn_w + bar_w, y0 + CELL],
                               fill=color)
                drawn_w += bar_w

            # Hand name text — top-left corner with shadow
            tx, ty = x0 + 4, y0 + 4
            draw.text((tx + 1, ty + 1), hand, fill=(0, 0, 0), font=small_font, anchor="lt")
            draw.text((tx, ty), hand, fill=COLOR_TEXT, font=small_font, anchor="lt")

    # Legend — labels adapt to the node type so a first-to-act (check/bet)
    # spot is not mislabeled as call/raise. The passive bucket lumps Check (X)
    # with Call (C) and the aggressive buckets lump bets with raises (see
    # _classify_actions), so the legend text is the only thing that tells the
    # two apart. Derive the node type from the actually-available action codes.
    pas_label, agg_label, show_fold = _legend_labels(
        action_solutions, spot_solution.get("game", {}))
    ly = title_h + HEADER + grid_size + 8
    legend_items = []
    if show_fold:
        legend_items.append(("Fold", COLOR_FOLD))
    legend_items += [
        (pas_label, COLOR_CALL),
        (agg_label, COLOR_RAISE),
        (f"Big {agg_label}/AI", COLOR_BIG_RAISE),
        ("N/A", COLOR_NOT_IN_RANGE),
    ]
    lx = HEADER
    for label, color in legend_items:
        draw.rectangle([lx, ly, lx + 12, ly + 12], fill=color)
        draw.text((lx + 16, ly + 6), label, fill=COLOR_HEADER, font=small_font, anchor="lm")
        lx += len(label) * 8 + 30

    # Export as PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
