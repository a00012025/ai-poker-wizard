"""GTO API cache — L1 in-memory dict + L2 PostgreSQL + L3 local files.

Sync interface (psycopg2) because gto_api.py is synchronous.
"""
import hashlib
import json
import logging
import math
import os
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

# L2: DB connection (lazy, thread-safe)
_db_conn = None
_db_lock = threading.Lock()

# L3: local file cache
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


def get(function: str, params: dict):
    """Look up cache. Returns SENTINEL on miss, else dict or None."""
    key = _cache_key(function, params)

    # L1
    if key in _mem:
        return _mem[key]

    # L2
    with _db_lock:
        conn = _get_conn()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT response, is_null FROM gto_api_cache WHERE cache_key = %s",
                    (key,),
                )
                row = cur.fetchone()
                cur.close()
                if row is not None:
                    response, is_null = row
                    result = None if is_null else response
                    _mem[key] = result
                    return result
            except Exception as e:
                logger.warning(f"gto_cache: DB read failed: {e}")

    # L3: file cache
    cache_file = _CACHE_DIR / f"{key}.json"
    try:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            result = None if data["is_null"] else data["response"]
            _mem[key] = result
            return result
    except Exception as e:
        logger.warning(f"gto_cache: file read failed: {e}")

    return SENTINEL


def put(function: str, params: dict, response: dict | None):
    """Store result in L1 + L2 + L3."""
    key = _cache_key(function, params)
    _mem[key] = response

    # Sanitize once up-front: replace NaN/Inf floats with None so both L2
    # (Postgres JSONB) and L3 (file cache reparsed elsewhere) stay strict-JSON.
    sanitized = _sanitize_json(response) if response is not None else None

    # L2: PostgreSQL
    with _db_lock:
        conn = _get_conn()
        if conn is not None:
            try:
                cur = conn.cursor()
                if sanitized is None:
                    cur.execute(
                        "INSERT INTO gto_api_cache (cache_key, response, is_null) "
                        "VALUES (%s, NULL, TRUE) ON CONFLICT DO NOTHING",
                        (key,),
                    )
                else:
                    cur.execute(
                        "INSERT INTO gto_api_cache (cache_key, response, is_null) "
                        "VALUES (%s, %s, FALSE) ON CONFLICT DO NOTHING",
                        (key, json.dumps(sanitized, allow_nan=False)),
                    )
                cur.close()
            except Exception as e:
                logger.warning(f"gto_cache: DB write failed: {e}")

    # L3: file cache
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        cache_file = _CACHE_DIR / f"{key}.json"
        if sanitized is None:
            data = {"is_null": True}
        else:
            data = {"is_null": False, "response": sanitized}
        cache_file.write_text(json.dumps(data, allow_nan=False))
    except Exception as e:
        logger.warning(f"gto_cache: file write failed: {e}")
