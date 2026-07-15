#!/usr/bin/env python3
"""Per-user GTO Wizard access-token minting and in-memory caching."""
import hashlib
import json
import logging
import time

import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"


class TokenExpiredError(Exception):
    """Raised when GTO Wizard tokens are expired and cannot be refreshed automatically."""
    pass


def _jwt_exp(token: str) -> int:
    """Extract exp from JWT without verification."""
    import base64
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["exp"]


def _import_signing():
    try:
        from scripts.gto_signing import generate_keypair_jwk, sign_refresh_request
    except ImportError:
        from gto_signing import generate_keypair_jwk, sign_refresh_request
    return generate_keypair_jwk, sign_refresh_request


def _get_or_create_keypair(tokens: dict | None = None) -> dict:
    """Get signing keypair from tokens dict, or generate a new one."""
    if tokens and "signing_keypair" in tokens:
        return tokens["signing_keypair"]
    generate_keypair_jwk, _ = _import_signing()
    return generate_keypair_jwk()


def _refresh_access(refresh_token: str, signing_keypair: dict | None = None) -> str | None:
    """Exchange refresh token for a new access token.

    Uses google-anal-id signed header required by GTO Wizard API.
    """
    try:
        _, sign_refresh_request = _import_signing()

        if signing_keypair is None:
            signing_keypair = _get_or_create_keypair()

        extra_headers = sign_refresh_request(refresh_token, signing_keypair)
        r = requests.post(
            f"{API_BASE}/v1/token/refresh/",
            data=json.dumps({"refresh": refresh_token}, separators=(",", ":")),
            headers=extra_headers,
            timeout=15,
        )
        if r.ok:
            return r.json()["access"]
        log.warning("Token refresh failed: HTTP %s — %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Token refresh error: %s", e)
    return None


# ── Per-user token management ──

_user_token_cache: dict[int, tuple[str, float, str]] = {}


_user_keypair_cache: dict[int, dict] = {}  # user_id -> signing_keypair


def get_user_access_token(user_id: int, refresh_token: str) -> str:
    """Get a valid access token for a specific user, refreshing if needed.

    Raises TokenExpiredError if the refresh token is invalid/expired.
    """
    refresh_fingerprint = hashlib.sha256(refresh_token.encode()).hexdigest()
    cached = _user_token_cache.get(user_id)
    if cached and cached[1] > time.time() + 60 and cached[2] == refresh_fingerprint:
        return cached[0]

    keypair = _user_keypair_cache.get(user_id)
    if keypair is None:
        keypair = _get_or_create_keypair()
        _user_keypair_cache[user_id] = keypair

    access = _refresh_access(refresh_token, keypair)
    if not access:
        # Evict stale cache entry
        _user_token_cache.pop(user_id, None)
        raise TokenExpiredError(
            "GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
        )

    _user_token_cache[user_id] = (access, _jwt_exp(access), refresh_fingerprint)
    return access


def invalidate_user_token(user_id: int):
    """Remove cached access token for a user (e.g. on logout)."""
    _user_token_cache.pop(user_id, None)


if __name__ == "__main__":
    import os
    from gto_owner_token import bootstrap_owner_db_token

    if not bootstrap_owner_db_token():
        raise SystemExit(1)
    access = get_user_access_token(-1, os.environ["GTOW_REFRESH_TOKEN"])
    print(f"OK: owner DB token minted access (length={len(access)})")
