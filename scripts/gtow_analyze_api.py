#!/usr/bin/env python3
"""GTO Wizard Analyze API client (hand-history list + detail).

Auth = Bearer access token (gto_token) + GWCLIENTID header. Global
throttle ~2.5 rps with jitter; exponential backoff on 429/5xx; one token
re-mint retry on 401. All probing that discovered this contract lives in
docs/superpowers/specs/2026-07-07-phase1-ledger-design.md §3.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_token import TokenExpiredError

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"
_CLIENT_ID_PATH = Path(__file__).resolve().parent.parent / ".gtow_client_id"
_MIN_INTERVAL = 0.4          # ~2.5 rps
_MAX_RETRIES = 5
_TIMEOUT = 30
_last_request_ts = 0.0

LIST_FIELDS = [
    "played_at", "total_ev_loss", "total_ev_loss_as_pot", "avg_gto_score",
    "avg_frequency_difference", "player_winloss", "player_position",
    "pot_type", "hero_hand", "boards", "hand_correctness",
    "preflop_game_depth", "blinds", "game_format", "file_original_name",
    "site", "solution_status", "total_players",
    "tournament_id", "tournament_name", "tournament_buyin", "total_pot",
    "board_flop_connectedness", "board_flop_pairedness",
    "actions_with_correctness_preflop", "actions_with_correctness_flop",
    "actions_with_correctness_turn", "actions_with_correctness_river",
]


def get_client_id(path: Path | str = _CLIENT_ID_PATH) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    cid = str(uuid.uuid4())
    p.write_text(cid)
    return cid


# Per-request token mode: extension-sync passes the requesting user's token.
# Owner-run CLI tools resolve the owner token from DB; bot requests fail closed
# rather than silently borrowing owner credentials.
_ENV_TOKEN_USER = -1     # sentinel user id for the env-provided token


def _get_token(force_remint: bool = False) -> str:
    if os.environ.get("POKER_BOT_PROCESS") == "1":
        raise TokenExpiredError(
            "Bot Analyze request 缺少 per-user GTO token；拒絕改用 owner token。"
        )
    refresh = os.environ.get("GTOW_REFRESH_TOKEN")
    if not refresh:
        from gto_owner_token import bootstrap_owner_db_token
        if not bootstrap_owner_db_token(verbose=False):
            raise TokenExpiredError(
                "找不到 owner DB GTO token；請先綁定 owner token，或設定 "
                "GTOW_REFRESH_TOKEN。"
            )
        refresh = os.environ["GTOW_REFRESH_TOKEN"]
    from gto_token import get_user_access_token, invalidate_user_token
    if force_remint:
        invalidate_user_token(_ENV_TOKEN_USER)
    return get_user_access_token(_ENV_TOKEN_USER, refresh)


def _headers(force_remint: bool = False) -> dict:
    return {
        "authorization": f"Bearer {_get_token(force_remint)}",
        "gwclientid": get_client_id(),
        "origin": ORIGIN,
        "content-type": "application/json",
    }


def _backoff_delay(attempt: int) -> int:
    return min(2 ** (attempt + 1), 60)


def _throttle(_sleep=time.sleep):
    global _last_request_ts
    wait = _MIN_INTERVAL + random.uniform(0, 0.2) - (time.monotonic() - _last_request_ts)
    if wait > 0:
        _sleep(wait)
    _last_request_ts = time.monotonic()


def _request(method: str, url: str, request_fn=None, _sleep=time.sleep,
             soft_statuses=(), **kw):
    """Central request with throttle/backoff/401-retry. request_fn injectable for tests.

    soft_statuses: HTTP codes that mean "no data / not ready for this resource"
    rather than a fatal error — return None so the caller can skip and retry
    later (e.g. a single hand whose GTOW upload is still processing -> 404,
    204 no-solution, 403 forbidden config). Everything else >= 400 raises.
    """
    fn = request_fn or requests.request
    reminted = False
    remint_next = False
    for attempt in range(_MAX_RETRIES + 1):
        if request_fn is None:
            _throttle(_sleep)
            kw["headers"] = _headers(force_remint=remint_next)
            remint_next = False      # consume the remint exactly once
            kw["timeout"] = _TIMEOUT
        r = fn(method, url, **kw)
        if r.status_code == 401 and not reminted:
            reminted = True          # token may have just expired; re-mint once
            remint_next = True
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
            _sleep(_backoff_delay(attempt))
            continue
        if r.status_code in soft_statuses:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"GTOW Analyze API {r.status_code} for {url}: {r.content[:300]!r}")
        return r.json()
    raise RuntimeError(f"GTOW Analyze API retries exhausted for {url}")


def list_hands(since_iso: str, until_iso: str | None = None, offset: int = 0,
               limit: int = 100, ordering: list[str] | None = None,
               request_fn=None, _sleep=time.sleep) -> dict:
    body = {
        "filters": {
            "played_at__range": [since_iso, until_iso],
            "analyzer_game_format": "TOURNAMENT",
        },
        "pagination": {"limit": limit, "offset": offset,
                       "ordering": ordering or ["played_at"]},
        "response_fields": LIST_FIELDS,
    }
    return _request("POST", f"{API_BASE}/v4/hand-history/hands/",
                    request_fn=request_fn, _sleep=_sleep, json=body)


def iter_all_hands(since_iso: str, until_iso: str | None = None,
                   page_size: int = 100, request_fn=None) -> Iterator[dict]:
    offset = 0
    while True:
        page = list_hands(since_iso, until_iso, offset=offset, limit=page_size,
                          request_fn=request_fn)
        items = page.get("items", [])
        yield from items
        offset += len(items)
        if offset >= page.get("total", 0) or not items:
            return


def hand_detail(gtow_hand_id: str, request_fn=None) -> dict | None:
    """Return the hand detail dict, or None if the hand has no retrievable
    analysis yet (204 no-solution / 403 forbidden / 404 upload still
    processing). Callers should skip None hands and let a later run retry."""
    return _request("GET", f"{API_BASE}/v4/hand-history/hands/{gtow_hand_id}/",
                    request_fn=request_fn, soft_statuses=(204, 403, 404))
