"""GTO API cache backed by process memory and persistent local files.

The cache is deliberately independent of Supabase: solver responses are large,
immutable by cache key, and are part of the owner's portable personal solve
library.  Docker bind-mounts ``.gto_cache`` so entries survive deploys.
"""
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path


def _sanitize_json(obj):
    """Recursively replace NaN/Inf floats so persisted JSON stays portable."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


logger = logging.getLogger("poker_bot")

SENTINEL = object()  # distinguishes "not in cache" from "cached None"

# L1: hot entries for this Python process.
_mem: dict[str, dict | None] = {}

# L2: durable host storage.  An override is useful for tests and maintenance
# tools; production uses the bind-mounted repository directory.
_CACHE_DIR = Path(
    os.getenv("GTO_CACHE_DIR", Path(__file__).resolve().parent.parent / ".gto_cache")
)

_PARAM_KEYS = [
    "gametype", "depth", "stacks", "preflop_actions",
    "board", "flop_actions", "turn_actions", "river_actions",
]


def _cache_key(function: str, params: dict) -> str:
    canonical = function + "|" + "|".join(
        str(params.get(k, "")) for k in _PARAM_KEYS
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_local(key: str, response: dict | None) -> bool:
    """Atomically persist one sanitized response beside its destination."""
    temp_path: Path | None = None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{key}.json"
        data = ({"is_null": True} if response is None else
                {"is_null": False, "response": response})
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=_CACHE_DIR,
            prefix=f".{key}.", suffix=".tmp", delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, allow_nan=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, cache_file)
        return True
    except Exception as e:
        logger.warning(f"gto_cache: local write failed: {e}")
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _read_local(key: str):
    """Return a local cached value, or ``SENTINEL`` on miss/corruption."""
    cache_file = _CACHE_DIR / f"{key}.json"
    try:
        if not cache_file.exists():
            return SENTINEL
        data = json.loads(cache_file.read_text())
        return None if data["is_null"] else data["response"]
    except Exception as e:
        # The caller will fetch the solver response again and atomically repair
        # the file.  Keep the failure visible rather than silently trusting it.
        logger.warning(f"gto_cache: local read failed: {e}")
        return SENTINEL


def get(function: str, params: dict):
    """Look up a response. Returns ``SENTINEL`` on miss, including corruption."""
    key = _cache_key(function, params)
    if key in _mem:
        return _mem[key]

    local = _read_local(key)
    if local is not SENTINEL:
        _mem[key] = local
    return local


def put(function: str, params: dict, response: dict | None):
    """Store a response in memory and durable local storage."""
    key = _cache_key(function, params)
    sanitized = _sanitize_json(response) if response is not None else None
    _mem[key] = sanitized
    _write_local(key, sanitized)


def entry_count() -> int:
    """Count complete cache entries without opening their large JSON payloads."""
    try:
        return sum(
            1 for entry in os.scandir(_CACHE_DIR)
            if entry.is_file() and len(entry.name) == 69
            and entry.name.endswith(".json")
        )
    except FileNotFoundError:
        return 0
    except OSError as e:
        logger.warning(f"gto_cache: count failed: {e}")
        return 0
