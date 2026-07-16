"""GTO API cache — L1 memory + L2 persistent local files + L3 PostgreSQL.

Sync interface (psycopg2) because gto_api.py is synchronous.
"""
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from pathlib import Path

import psycopg2


def _sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None so Postgres JSONB accepts it.

    Python's json.dumps emits 'NaN'/'Infinity' by default (non-standard JSON)
    which Postgres rejects. Some GTO Wizard responses contain NaN total_ev
    when a hand combo has 0 range weight on a given street. We null those
    out before serializing.
    """
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

# L1: in-memory cache
_mem: dict[str, dict | None] = {}

# L3: DB connection (lazy, thread-safe)
_db_conn = None
_db_lock = threading.Lock()

# L2: persistent local file cache. Docker bind-mounts this directory so cache
# hits survive deploys and do not create Shared Pooler egress after restarts.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".gto_cache"

_PARAM_KEYS = [
    "gametype", "depth", "stacks", "preflop_actions",
    "board", "flop_actions", "turn_actions", "river_actions",
]


def _cache_key(function: str, params: dict) -> str:
    canonical = function + "|" + "|".join(
        str(params.get(k, "")) for k in _PARAM_KEYS
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _get_conn():
    """Lazy-init psycopg2 connection from SUPABASE_CONN. Auto-reconnect."""
    global _db_conn
    dsn = os.getenv("SUPABASE_CONN")
    if not dsn:
        return None
    try:
        if _db_conn is not None and _db_conn.closed == 0:
            return _db_conn
    except Exception:
        pass
    try:
        _db_conn = psycopg2.connect(dsn)
        _db_conn.autocommit = True
        return _db_conn
    except Exception as e:
        logger.warning(f"gto_cache: DB connect failed: {e}")
        _db_conn = None
        return None


def _write_local(key: str, response: dict | None) -> bool:
    """Atomically persist one sanitized cache response.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic even when ``.gto_cache`` is a Docker bind mount. Concurrent writers
    may replace one another, but readers can never observe a partial JSON file.
    """
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
            json.dump(data, f, allow_nan=False)
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
        # Corrupt or interrupted legacy files are soft misses. The DB fallback
        # below repairs them with a fresh atomic local copy.
        logger.warning(f"gto_cache: local read failed: {e}")
        return SENTINEL


def get(function: str, params: dict):
    """Look up cache. Returns SENTINEL on miss, else dict or None."""
    key = _cache_key(function, params)

    # L1
    if key in _mem:
        return _mem[key]

    # L2: persistent local cache first. This is intentionally before the DB:
    # a new Python process or bot deploy should not download the same large
    # solver JSON through Supavisor when the host already has it.
    local = _read_local(key)
    if local is not SENTINEL:
        _mem[key] = local
        return local

    # L3: PostgreSQL disaster-recovery/cold-miss fallback.
    with _db_lock:
        conn = _get_conn()
        if conn is not None:
            cur = None
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT response, is_null FROM gto_api_cache WHERE cache_key = %s",
                    (key,),
                )
                row = cur.fetchone()
                if row is not None:
                    response, is_null = row
                    result = None if is_null else response
                    _mem[key] = result
                    # Hydrate L2 so later processes and container restarts no
                    # longer need the DB for this key.
                    _write_local(key, result)
                    return result
            except Exception as e:
                logger.warning(f"gto_cache: DB read failed: {e}")
            finally:
                if cur is not None:
                    cur.close()

    return SENTINEL


def put(function: str, params: dict, response: dict | None):
    """Store result in memory, local persistence, then best-effort DB."""
    key = _cache_key(function, params)
    _mem[key] = response

    # Sanitize once up-front so local JSON and PostgreSQL JSONB stay strict.
    sanitized = _sanitize_json(response) if response is not None else None

    # L2 first: DB latency/failure must not prevent durable local caching.
    _write_local(key, sanitized)

    # L3: PostgreSQL remains a best-effort fallback for a lost/new host.
    with _db_lock:
        conn = _get_conn()
        if conn is not None:
            cur = None
            try:
                cur = conn.cursor()
                if sanitized is None:
                    cur.execute(
                        "INSERT INTO gto_api_cache (cache_key, response, is_null) "
                        "VALUES (%s, NULL, TRUE) "
                        "ON CONFLICT (cache_key) DO UPDATE SET response = NULL, is_null = TRUE",
                        (key,),
                    )
                else:
                    cur.execute(
                        "INSERT INTO gto_api_cache (cache_key, response, is_null) "
                        "VALUES (%s, %s, FALSE) "
                        "ON CONFLICT (cache_key) DO UPDATE SET response = EXCLUDED.response, is_null = FALSE",
                        (key, json.dumps(sanitized, allow_nan=False)),
                    )
            except Exception as e:
                logger.warning(f"gto_cache: DB write failed: {e}")
            finally:
                if cur is not None:
                    cur.close()
