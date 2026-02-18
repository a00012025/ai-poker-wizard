#!/usr/bin/env python3
"""Parse GGPoker hand history files into analyze_hand_full() input JSON.

Usage:
    python scripts/hh_parser.py 2026-02-17/
    python scripts/hh_parser.py 2026-02-17/file.txt
"""

import re
from pathlib import Path

# GTO Wizard position orders (matching analyze_hand.py)
POSITION_ORDERS = {
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"],
}


def _clockwise_from_button(button_seat: int, occupied_seats: set[int], max_seats: int) -> list[int]:
    """Return occupied seats in clockwise order starting from button."""
    order = []
    for i in range(max_seats):
        seat = (button_seat - 1 + i) % max_seats + 1
        if seat in occupied_seats:
            order.append(seat)
    return order


def _parse_amount(s: str) -> int:
    """Parse chip amount like '2,557' or '500' to int."""
    return int(s.replace(",", ""))


def _split_hands(text: str) -> list[str]:
    """Split a hand history file into individual hand blocks."""
    hands = []
    current = []
    for line in text.split("\n"):
        if line.startswith("Poker Hand #") and current:
            hands.append("\n".join(current))
            current = []
        current.append(line)
    if current and any(l.strip() for l in current):
        hands.append("\n".join(current))
    return hands


def parse_hand(text: str, include_folds: bool = False) -> dict | None:
    """Parse a single hand history block into analyze_hand_full() input dict.

    Returns dict with keys matching analyze_hand_full() input, plus metadata:
        hand_id, file, table_size, gametype, effective_bb, hero_position,
        hero_hand, preflop_actions, streets, num_players

    Returns None if hand should be skipped (hero not present, hero won uncontested, etc.)
    If include_folds=True, includes hands where hero folded preflop.
    """
    lines = text.strip().split("\n")
    if len(lines) < 5:
        return None

    # ── Header ──
    header = lines[0]
    hand_id_m = re.search(r"#(TM\d+)", header)
    hand_id = hand_id_m.group(1) if hand_id_m else "unknown"

    level_m = re.search(r"Level\d+\(([\d,]+)/([\d,]+)\)", header)
    if not level_m:
        return None
    bb_size = _parse_amount(level_m.group(2))

    # ── Table info ──
    table_line = lines[1]
    max_seats_m = re.search(r"(\d+)-max", table_line)
    max_seats = int(max_seats_m.group(1)) if max_seats_m else 8

    button_m = re.search(r"Seat #(\d+) is the button", table_line)
    if not button_m:
        return None
    button_seat = int(button_m.group(1))

    # ── Parse seats ──
    seats = {}  # seat_num -> {name, chips}
    hero_seat = None
    for line in lines[2:]:
        m = re.match(r"Seat (\d+): (.+?) \(([\d,]+) in chips\)", line)
        if m:
            seat_num = int(m.group(1))
            name = m.group(2)
            chips = _parse_amount(m.group(3))
            seats[seat_num] = {"name": name, "chips": chips}
            if name == "Hero":
                hero_seat = seat_num
        elif not line.startswith("Seat "):
            break

    if hero_seat is None:
        return None

    num_players = len(seats)
    positions = POSITION_ORDERS.get(num_players)
    if positions is None:
        return None

    # ── Assign positions ──
    # Clockwise from button: BTN, SB, BB, UTG, UTG+1, ...
    clockwise = _clockwise_from_button(button_seat, set(seats.keys()), max_seats)

    if num_players == 2:
        # Heads-up: button = SB
        clockwise_positions = ["SB", "BB"]
    else:
        btn_idx = positions.index("BTN")
        clockwise_positions = positions[btn_idx:] + positions[:btn_idx]

    seat_to_pos = {}
    for i, seat in enumerate(clockwise):
        seat_to_pos[seat] = clockwise_positions[i]

    hero_position = seat_to_pos[hero_seat]
    hero_chips = seats[hero_seat]["chips"]
    effective_bb = hero_chips / bb_size

    # name -> position mapping
    name_to_pos = {}
    for seat, info in seats.items():
        name_to_pos[info["name"]] = seat_to_pos[seat]

    # ── Parse hero's hand ──
    hero_hand = None
    for line in lines:
        m = re.match(r"Dealt to Hero \[(.+?)\]", line)
        if m:
            cards = m.group(1).split()
            hero_hand = cards[0] + cards[1]
            break

    if hero_hand is None:
        return None

    # ── Split into sections by *** markers ──
    sections = {"preflop": [], "flop": [], "turn": [], "river": []}
    current_section = None
    board_flop = None
    turn_card = None
    river_card = None

    for line in lines:
        if "*** HOLE CARDS ***" in line:
            current_section = "preflop"
            continue
        elif "*** FLOP ***" in line:
            current_section = "flop"
            m = re.search(r"\[(.+?)\]", line)
            if m:
                board_flop = m.group(1).replace(" ", "")
            continue
        elif "*** TURN ***" in line:
            current_section = "turn"
            brackets = re.findall(r"\[(.+?)\]", line)
            if len(brackets) >= 2:
                turn_card = brackets[1].replace(" ", "")
            continue
        elif "*** RIVER ***" in line:
            current_section = "river"
            brackets = re.findall(r"\[(.+?)\]", line)
            if len(brackets) >= 2:
                river_card = brackets[1].replace(" ", "")
            continue
        elif "*** SHOWDOWN ***" in line or "*** SUMMARY ***" in line:
            current_section = None
            continue

        if current_section:
            sections[current_section].append(line)

    # ── Parse preflop actions ──
    preflop_actions_ordered = []  # list of (position, action_code) in play order

    for line in sections["preflop"]:
        if line.startswith("Dealt to"):
            continue
        if "Uncalled bet" in line:
            continue
        if ": posts " in line:
            continue

        m = re.match(r"(.+?): (.+)", line)
        if not m:
            continue
        player = m.group(1).strip()
        action_text = m.group(2).strip()

        if player not in name_to_pos:
            continue
        pos = name_to_pos[player]

        action_code = _parse_action_preflop(action_text, bb_size)
        if action_code:
            preflop_actions_ordered.append((pos, action_code))

    # Build preflop_actions string ordered by position
    # First round: one action per position in POSITION_ORDERS order
    # Continuation: re-raises in play order
    pos_actions = {}  # position -> list of action codes (in order)
    for pos, code in preflop_actions_ordered:
        pos_actions.setdefault(pos, []).append(code)

    preflop_parts = []
    # First round
    for pos in positions:
        if pos in pos_actions and pos_actions[pos]:
            preflop_parts.append(pos_actions[pos].pop(0))

    # Continuation rounds — actions remaining after first round, in play order
    # Replay in position order for each remaining round
    has_more = True
    while has_more:
        has_more = False
        for pos in positions:
            if pos in pos_actions and pos_actions[pos]:
                preflop_parts.append(pos_actions[pos].pop(0))
                has_more = True

    preflop_actions = "-".join(preflop_parts)

    # ── Check if hero folded preflop ──
    hero_preflop_idx = positions.index(hero_position)
    if hero_preflop_idx < len(preflop_parts) and preflop_parts[hero_preflop_idx] == "F":
        if not include_folds:
            return None

    # Check if hero had no decision (walk — everyone folded before hero acted)
    if hero_preflop_idx >= len(preflop_parts):
        # Hero's position was never reached (everyone folded)
        return None

    # ── Parse postflop streets ──
    streets = []
    hero_has_postflop_action = False

    for street_name, cards_value in [("flop", board_flop), ("turn", turn_card), ("river", river_card)]:
        if not sections[street_name]:
            break

        street_actions = []
        for line in sections[street_name]:
            if "Uncalled bet" in line:
                continue
            if ": shows " in line or ": mucks " in line:
                continue

            m = re.match(r"(.+?): (.+)", line)
            if not m:
                continue
            player = m.group(1).strip()
            action_text = m.group(2).strip()

            if player not in name_to_pos:
                continue
            pos = name_to_pos[player]

            action_obj = _parse_action_postflop(action_text, bb_size)
            if action_obj:
                action_obj["position"] = pos
                street_actions.append(action_obj)
                if pos == hero_position:
                    hero_has_postflop_action = True

        if not street_actions:
            break

        street_dict = {"actions": street_actions}
        if street_name == "flop":
            street_dict["board"] = cards_value
        else:
            street_dict["card"] = cards_value
        streets.append(street_dict)

    # ── Build result ──
    result = {
        "hand_id": hand_id,
        "table_size": max_seats,
        "num_players": num_players,
        "gametype": "MTTGeneral",
        "effective_bb": round(effective_bb, 1),
        "hero_position": hero_position,
        "hero_hand": hero_hand,
        "preflop_actions": preflop_actions,
    }

    if streets:
        result["streets"] = streets

    return result


def _parse_action_preflop(action_text: str, bb_size: int) -> str | None:
    """Convert HH action text to preflop action code."""
    if action_text == "folds":
        return "F"
    if action_text == "checks":
        return "X"
    if action_text.startswith("calls"):
        return "C"

    # "raises X to Y and is all-in"
    if "raises" in action_text and "all-in" in action_text:
        m = re.search(r"to ([\d,]+)", action_text)
        if m:
            to_amount = _parse_amount(m.group(1))
            size_bb = round(to_amount / bb_size, 1)
            return f"AI{size_bb}"
        return "AI"

    # "raises X to Y"
    if "raises" in action_text:
        m = re.search(r"to ([\d,]+)", action_text)
        if m:
            to_amount = _parse_amount(m.group(1))
            size_bb = round(to_amount / bb_size, 1)
            return f"R{size_bb}"
        return None

    # "bets X and is all-in" (rare preflop but handle it)
    if "all-in" in action_text:
        m = re.search(r"([\d,]+)", action_text)
        if m:
            amount = _parse_amount(m.group(1))
            size_bb = round(amount / bb_size, 1)
            return f"AI{size_bb}"
        return "AI"

    return None


def _parse_action_postflop(action_text: str, bb_size: int) -> dict | None:
    """Convert HH action text to postflop action dict."""
    if action_text == "checks":
        return {"action": "X"}
    if action_text == "folds":
        return {"action": "F"}

    # "calls X"
    if action_text.startswith("calls"):
        return {"action": "C"}

    # "bets X and is all-in"
    if action_text.startswith("bets") and "all-in" in action_text:
        m = re.search(r"bets ([\d,]+)", action_text)
        if m:
            amount = _parse_amount(m.group(1))
            size_bb = round(amount / bb_size, 1)
            return {"action": "AI", "size": size_bb}
        return {"action": "AI"}

    # "raises X to Y and is all-in"
    if "raises" in action_text and "all-in" in action_text:
        m = re.search(r"to ([\d,]+)", action_text)
        if m:
            to_amount = _parse_amount(m.group(1))
            size_bb = round(to_amount / bb_size, 1)
            return {"action": "AI", "size": size_bb}
        return {"action": "AI"}

    # "bets X"
    if action_text.startswith("bets"):
        m = re.search(r"bets ([\d,]+)", action_text)
        if m:
            amount = _parse_amount(m.group(1))
            size_bb = round(amount / bb_size, 1)
            return {"action": f"R{size_bb}", "size": size_bb}
        return None

    # "raises X to Y"
    if "raises" in action_text:
        m = re.search(r"to ([\d,]+)", action_text)
        if m:
            to_amount = _parse_amount(m.group(1))
            size_bb = round(to_amount / bb_size, 1)
            return {"action": f"R{size_bb}", "size": size_bb}
        return None

    return None


def parse_file(filepath: str | Path, include_folds: bool = False) -> list[dict]:
    """Parse all hands from a single HH file. Returns list of parsed hand dicts."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    blocks = _split_hands(text)

    results = []
    for block in blocks:
        parsed = parse_hand(block, include_folds=include_folds)
        if parsed:
            parsed["file"] = filepath.name
            results.append(parsed)
    return results


def parse_directory(dirpath: str | Path, include_folds: bool = False) -> list[dict]:
    """Parse all .txt HH files in a directory. Returns list of parsed hand dicts."""
    dirpath = Path(dirpath)
    all_hands = []
    for filepath in sorted(dirpath.glob("*.txt")):
        hands = parse_file(filepath, include_folds=include_folds)
        all_hands.extend(hands)
    return all_hands


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/hh_parser.py <directory_or_file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        hands = parse_directory(path)
    else:
        hands = parse_file(path)

    print(f"Parsed {len(hands)} hero-played hands")
    for h in hands:
        pos = h["hero_position"]
        hand = h["hero_hand"]
        ebb = h["effective_bb"]
        pf = h["preflop_actions"]
        streets = len(h.get("streets", []))
        print(f"  {h['hand_id']}: {pos} {hand} ({ebb:.1f}bb) pf={pf} streets={streets}")

    if "--json" in sys.argv:
        print("\n" + json.dumps(hands, indent=2))
