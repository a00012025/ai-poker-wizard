#!/usr/bin/env python3
"""ICM game mode discovery and stack matching.

Loads game modes from GTO Wizard API and finds the nearest matching
ICM gametype and stack configuration for a given tournament scenario.
"""
import json
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
) -> tuple[str, str]:
    """Find the nearest stack configuration for an ICM gametype.

    Args:
        gametype: e.g., 'MTTGeneral_ICM8m1000PTPCT25'
        player_stacks: Stack sizes in bb ordered [UTG, UTG+1, ..., BB]

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

    # Find nearest config by L1 distance on bb values
    def stack_distance(config_stacks: list[float]) -> float:
        return sum(abs(a - b) for a, b in zip(player_stacks, config_stacks))

    best = min(configs, key=lambda c: stack_distance(c["stacks_bb"]))

    depth_str = best["depth"]
    stacks_str = "-".join(best["stacks"])

    return depth_str, stacks_str


def find_icm_params(
    player_stacks: list[float],
    pko: bool = False,
    tournament_size: int = 1000,
    players_remaining: int | None = None,
    phase: str | None = None,
) -> dict:
    """High-level: find gametype + stacks for an ICM scenario.

    Args:
        player_stacks: Stack sizes in bb ordered [UTG, UTG+1, ..., BB]
        pko: PKO tournament?
        tournament_size: 1000 or 200
        players_remaining: Players remaining in tournament
        phase: Direct phase name

    Returns:
        Dict with keys: gametype, depth, stacks, approximation_note
    """
    players_at_table = len(player_stacks)

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

    depth_str, stacks_str = find_stacks(gametype, player_stacks)
    actual_stacks = _parse_stacks(stacks_str.split("-"))

    # Build approximation note
    notes = []
    notes.append(f"ICM 模式: {gametype}")
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
