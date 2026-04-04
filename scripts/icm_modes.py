#!/usr/bin/env python3
"""ICM game mode discovery and stack matching.

Loads game modes from GTO Wizard API and finds the nearest matching
ICM gametype and stack configuration for a given tournament scenario.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_api import _session, _ensure_auth, API_BASE

# Cache game modes in memory
_game_modes_cache: list[dict] | None = None

# Persist to local file to avoid repeated API calls
_CACHE_FILE = Path(__file__).resolve().parent.parent / ".game_modes_cache.json"


def _load_game_modes() -> list[dict]:
    """Load game modes from cache or API."""
    global _game_modes_cache
    if _game_modes_cache is not None:
        return _game_modes_cache

    # Try local file cache first
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE) as f:
            _game_modes_cache = json.load(f)
        return _game_modes_cache

    # Fetch from API
    _ensure_auth()
    r = _session.get(f"{API_BASE}/v4/game-modes/", timeout=30)
    r.raise_for_status()
    _game_modes_cache = r.json()

    # Persist
    with open(_CACHE_FILE, "w") as f:
        json.dump(_game_modes_cache, f)

    return _game_modes_cache


def _parse_stacks(stacks: list[str]) -> list[float]:
    """Convert stacks list ['50.125', '30.125'] to bb values [50, 30]."""
    return [float(s) - 0.125 for s in stacks]


def find_gametype(
    players_at_table: int = 8,
    pko: bool = False,
    tournament_size: int = 1000,
    players_remaining: int | None = None,
    phase: str | None = None,
) -> str:
    """Find the best matching ICM gametype.

    Args:
        players_at_table: Players at table (2-9), determines FT/T2/T3 modes
        pko: True for PKO (progressive knockout) tournaments
        tournament_size: 1000 or 200
        players_remaining: Approximate players remaining (optional if phase given)
        phase: Direct phase name override (START, PCT75, BUBBLE, FT, etc.)

    Returns:
        Gametype string like 'MTTGeneral_ICM8m1000PTPCT25'
    """
    modes = _load_game_modes()
    pko_str = "PKO" if pko else ""

    # Filter to relevant ICM modes (MTTGeneral only, skip Test/SimpleTest)
    candidates = []
    for mode in modes:
        name = mode["name"]
        if not name.startswith("MTTGeneral_ICM"):
            continue
        if f"ICM{pko_str}" not in name:
            continue
        if pko and "PKO" not in name:
            continue
        if not pko and "PKO" in name:
            continue

        info = mode.get("info", {})
        mp = mode.get("players", 0)

        # Must have visible configs
        visible = sum(
            1 for gm in mode["game_modes"]
            if not gm.get("info", {}).get("hidden", False)
        )
        if visible == 0:
            continue

        candidates.append({
            "name": name,
            "players": mp,
            "remaining": info.get("players_remaining", 0),
            "phase": info.get("tournament_phase", ""),
            "tournament_players": info.get("tournament_players", 0),
        })

    if not candidates:
        return "MTTGeneral"  # fallback to chip EV

    # Phase-based matching
    if phase:
        phase_upper = phase.upper().replace(" ", "")
        # Map common user inputs to phase names
        phase_map = {
            "START": "START", "EARLY": "START",
            "PCT75": "PCT75", "75%": "PCT75",
            "PCT50": "PCT50", "50%": "PCT50",
            "PCT25": "PCT25", "25%": "PCT25",
            "PCT10": "PCT10", "10%": "PCT10",
            "PCT5": "PCT5", "5%": "PCT5",
            "BUBBLE": "BUBBLEMID", "BUBBLEEARLY": "BUBBLEEARLY",
            "BUBBLEMID": "BUBBLEMID", "BUBBLELATE": "BUBBLELATE",
            "FT": "FT", "FINALTABLE": "FT",
            "T2": "T2", "T3": "T3",
        }
        target_phase = phase_map.get(phase_upper, phase_upper)

        # Filter by matching table size and tournament size
        phase_matches = [
            c for c in candidates
            if c["phase"] == target_phase
            and c["players"] == players_at_table
            and c["tournament_players"] == tournament_size
        ]
        if phase_matches:
            return phase_matches[0]["name"]

    # Remaining-based matching: find the mode with closest players_remaining
    if players_remaining is not None:
        # Filter by table size and tournament size
        size_matches = [
            c for c in candidates
            if c["players"] == players_at_table
            and c["tournament_players"] == tournament_size
        ]
        if not size_matches:
            # Try other tournament size
            size_matches = [
                c for c in candidates
                if c["players"] == players_at_table
            ]
        if size_matches:
            best = min(size_matches, key=lambda c: abs(c["remaining"] - players_remaining))
            return best["name"]

    # Default: match by table players only
    table_matches = [
        c for c in candidates
        if c["players"] == players_at_table
        and c["tournament_players"] == tournament_size
    ]
    if table_matches:
        # Default to a middle phase
        for preferred_phase in ["PCT25", "PCT50", "BUBBLEMID", "FT"]:
            for c in table_matches:
                if c["phase"] == preferred_phase:
                    return c["name"]
        return table_matches[0]["name"]

    return "MTTGeneral"


def find_stacks(
    gametype: str,
    player_stacks: list[float],
    preflop_actions: str = "",
    empty_seats: set[int] | None = None,
) -> tuple[str, str]:
    """Find the nearest stack configuration for an ICM gametype.

    Uses a three-component distance metric:
    1. Log-ratio distance  — scale-invariant proportion matching
    2. Rank inversion penalty — preserves who covers whom (ICM critical)
    3. Min stack penalty   — shortest stack depth drives push/fold ranges

    Active positions (haven't folded yet) are weighted 3× more heavily
    than already-folded positions.

    Args:
        gametype: e.g., 'MTTGeneral_ICM8m1000PTPCT25'
        player_stacks: Stack sizes in bb ordered [UTG, UTG+1, ..., BB]
        preflop_actions: e.g., 'F-F-F-F-F-RAI' to identify folded positions

    Returns:
        (depth_str, stacks_str) e.g., ('50.125', '50.125-25.125-...')
    """
    modes = _load_game_modes()

    # Find the mode
    mode = None
    for m in modes:
        if m["name"] == gametype:
            mode = m
            break
    if not mode:
        # Fallback: symmetric stacks at average depth
        avg = sum(player_stacks) / len(player_stacks)
        depth_str = f"{avg:.3f}"
        stacks_str = "-".join(f"{s + 0.125:.3f}" for s in player_stacks)
        return depth_str, stacks_str

    # Get visible configs
    configs = []
    for gm in mode["game_modes"]:
        info = gm.get("info", {})
        if info.get("hidden", False):
            continue
        stacks = gm.get("stacks")
        if not stacks:
            continue
        if len(stacks) != len(player_stacks):
            continue
        configs.append({
            "depth": gm["depth"],
            "stacks": stacks,
            "stacks_bb": _parse_stacks(stacks),
            "type": info.get("stacks_type", ""),
        })

    if not configs:
        avg = sum(player_stacks) / len(player_stacks)
        depth_str = f"{avg + 0.125:.3f}"
        stacks_str = "-".join(f"{s + 0.125:.3f}" for s in player_stacks)
        return depth_str, stacks_str

    # --- Identify folded positions from preflop_actions ---
    # "F-F-F-F-F-RAI" → first 5 positions folded
    folded = set()
    if preflop_actions:
        parts = preflop_actions.split("-")
        for i, p in enumerate(parts):
            if i >= len(player_stacks):
                break
            if p.upper() == "F":
                folded.add(i)
            else:
                break  # first non-fold ends the fold prefix

    active_weight = 3.0
    inversion_penalty = 0.5
    min_stack_weight = 3.0

    if empty_seats is None:
        empty_seats = set()

    # The shortest REAL folded stack creates ICM pressure regardless
    # of which seat it's in.  Match it position-independently against
    # the shortest non-active stack in the config.
    folded_min_stack: float | None = None
    real_folded = [player_stacks[i] for i in folded
                   if player_stacks[i] > 0 and i not in empty_seats]
    if real_folded:
        folded_min_stack = min(real_folded)

    def _icm_distance(config_stacks: list[float]) -> float:
        n = len(player_stacks)

        # --- ① Log-ratio distance (active positions only) ---
        log_dist = 0.0
        for i in range(n):
            a, c = player_stacks[i], config_stacks[i]
            if a <= 0 or c <= 0:
                continue
            if i in folded or i in empty_seats:
                continue  # folded/empty handled by ③ instead
            log_dist += active_weight * abs(
                math.log(a) - math.log(c))

        # --- ② Rank inversion penalty (active positions only) ---
        # Count pairwise inversions: if actual[i] < actual[j] but
        # config[i] > config[j], the cover relationship is flipped.
        active_idx = [i for i in range(n)
                      if i not in folded and player_stacks[i] > 0]
        inversions = 0
        for ii in range(len(active_idx)):
            for jj in range(ii + 1, len(active_idx)):
                ai, aj = active_idx[ii], active_idx[jj]
                actual_cmp = (player_stacks[ai] > player_stacks[aj]) \
                    - (player_stacks[ai] < player_stacks[aj])
                config_cmp = (config_stacks[ai] > config_stacks[aj]) \
                    - (config_stacks[ai] < config_stacks[aj])
                if actual_cmp != 0 and config_cmp != 0 \
                        and actual_cmp != config_cmp:
                    inversions += 1

        # --- ③ Short stack penalties (position-independent) ---
        # a) Folded shortest stack: match against the smallest
        #    non-active config stack.  The ICM pressure comes from
        #    *having* a short stack, not which seat it occupies.
        short_penalty = 0.0
        if folded_min_stack is not None:
            non_active_cfg = [config_stacks[i] for i in range(n)
                              if i not in active_idx
                              and config_stacks[i] > 0]
            if non_active_cfg:
                cfg_min = min(non_active_cfg)
                short_penalty = abs(
                    math.log(folded_min_stack) - math.log(cfg_min))

        # b) Active shortest stack: depth drives push/fold ranges.
        min_penalty = 0.0
        if active_idx:
            min_actual = min(player_stacks[i] for i in active_idx)
            min_config = min(config_stacks[i] for i in active_idx)
            if min_actual > 0 and min_config > 0:
                min_penalty = abs(
                    math.log(min_actual) - math.log(min_config))

        return (log_dist
                + inversion_penalty * inversions
                + min_stack_weight * short_penalty
                + min_stack_weight * min_penalty)

    best = min(configs, key=lambda c: _icm_distance(c["stacks_bb"]))

    depth_str = best["depth"]
    stacks_str = "-".join(best["stacks"])

    return depth_str, stacks_str


def find_icm_params(
    player_stacks: list[float],
    pko: bool = False,
    tournament_size: int = 1000,
    players_remaining: int | None = None,
    phase: str | None = None,
    players_at_table: int | None = None,
    preflop_actions: str = "",
) -> dict:
    """High-level: find gametype + stacks for an ICM scenario.

    Args:
        player_stacks: Stack sizes in bb ordered [UTG, UTG+1, ..., BB]
        pko: PKO tournament?
        tournament_size: 1000 or 200
        players_remaining: Players remaining in tournament
        phase: Direct phase name
        players_at_table: Override for table size (e.g., 8 for 8-max FT even if
            only 5 players have stacks). Zero-stack positions are filled with
            average of remaining stacks for better solver matching.
        preflop_actions: e.g., 'F-F-F-F-F-RAI' — used to identify folded
            positions for smarter stack matching.

    Returns:
        Dict with keys: gametype, depth, stacks, approximation_note
    """
    if players_at_table is None:
        players_at_table = len(player_stacks)

    # Track empty seats (zero-stack positions from 8-max padding).
    # These are filled with small values for config length matching but
    # should be ignored in the ICM distance calculation.
    empty_seats = {i for i, s in enumerate(player_stacks) if s == 0}
    non_zero = [s for s in player_stacks if s > 0]
    if non_zero and empty_seats:
        fill_value = min(non_zero) * 0.5
        player_stacks = [s if s > 0 else fill_value for s in player_stacks]

    gametype = find_gametype(
        players_at_table=players_at_table,
        pko=pko,
        tournament_size=tournament_size,
        players_remaining=players_remaining,
        phase=phase,
    )

    if gametype == "MTTGeneral":
        # Fallback to chip EV
        from gto_api import nearest_depth
        avg = sum(player_stacks) / len(player_stacks)
        return {
            "gametype": "MTTGeneral",
            "depth": nearest_depth(avg),
            "stacks": "",
            "approximation_note": "找不到匹配的 ICM 模式，使用 Chip EV 替代",
        }

    depth_str, stacks_str = find_stacks(gametype, player_stacks,
                                        preflop_actions=preflop_actions,
                                        empty_seats=empty_seats)
    actual_stacks = _parse_stacks(stacks_str.split("-"))

    # Build approximation note with clear user stacks vs solver stacks comparison
    notes = []
    notes.append(f"ICM 模式: {gametype}")
    notes.append(f"用戶籌碼: {' / '.join(f'{s:.0f}' for s in player_stacks)}")
    notes.append(f"Solver 籌碼: {' / '.join(f'{s:.0f}' for s in actual_stacks)}")

    # Show stack differences
    diffs = [abs(a - b) for a, b in zip(player_stacks, actual_stacks)]
    max_diff = max(diffs)
    if max_diff > 1:
        notes.append(f"最大差異: {max_diff:.0f}bb")

    return {
        "gametype": gametype,
        "depth": depth_str,
        "stacks": stacks_str,
        "approximation_note": "\n".join(notes),
    }


def infer_icm_phase(
    avg_stack_chips: float,
    starting_stack_chips: float,
    players_at_table: int = 8,
    pko: bool = False,
    tournament_size: int = 1000,
) -> dict:
    """Infer ICM params from avg stack at table vs starting stack.

    The ratio avg_stack/starting_stack approximates total_players/players_remaining.
    This lets us estimate the tournament phase without knowing exact players_remaining.

    Args:
        avg_stack_chips: Average chip stack at the table
        starting_stack_chips: Tournament starting stack in chips
        players_at_table: Number of players at the table
        pko: PKO tournament?
        tournament_size: 1000 or 200

    Returns:
        Dict with keys: gametype, depth, stacks, approximation_note, phase_info
    """
    if starting_stack_chips <= 0:
        return {"gametype": "MTTGeneral", "depth": None, "stacks": "",
                "approximation_note": "無效的起始籌碼", "phase_info": "chip_ev"}

    ratio = avg_stack_chips / starting_stack_chips
    estimated_remaining = tournament_size / ratio
    # Can't have fewer remaining than players at the table
    estimated_remaining = max(players_at_table, estimated_remaining)
    estimated_remaining = min(tournament_size, estimated_remaining)

    gametype = find_gametype(
        players_at_table=players_at_table,
        pko=pko,
        tournament_size=tournament_size,
        players_remaining=int(round(estimated_remaining)),
    )

    if gametype == "MTTGeneral":
        from gto_api import nearest_depth
        return {
            "gametype": "MTTGeneral",
            "depth": None,
            "stacks": "",
            "approximation_note": "找不到匹配的 ICM 模式，使用 Chip EV",
            "phase_info": "chip_ev",
        }

    # Extract phase from gametype name for logging
    phase_info = f"ratio={ratio:.1f}x, est_remaining={int(round(estimated_remaining))}"

    return {
        "gametype": gametype,
        "estimated_remaining": int(round(estimated_remaining)),
        "ratio": round(ratio, 2),
        "phase_info": phase_info,
    }


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", help="Comma-separated stacks in bb, e.g., '50,30,45,20,35,25,15,40'")
    parser.add_argument("--pko", action="store_true")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--remaining", type=int)
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    if args.stacks:
        stacks = [float(x) for x in args.stacks.split(",")]
    else:
        stacks = [50, 30, 45, 20, 35, 25, 15, 40]

    result = find_icm_params(
        player_stacks=stacks,
        pko=args.pko,
        tournament_size=args.size,
        players_remaining=args.remaining,
        phase=args.phase,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
