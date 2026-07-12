"""Shared primitives for pairing Chrome devices with Telegram users."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

_CROCKFORD = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
PAIR_CODE_LENGTH = 12
PAIR_TTL_MINUTES = 5


def get_sync_pepper() -> str:
    """Return the secret shared by the bot and Supabase Edge Function."""
    pepper = os.getenv("GTOW_SYNC_PEPPER", "")
    if len(pepper) < 32:
        raise RuntimeError("GTOW_SYNC_PEPPER must be at least 32 characters")
    return pepper


def normalize_pair_code(code: str) -> str:
    """Normalize a human-entered pairing code to its canonical form."""
    return "".join(ch for ch in code.upper() if ch.isalnum())


def generate_pair_code() -> str:
    """Generate a 60-bit, typo-resistant, human-readable pairing code."""
    raw = "".join(secrets.choice(_CROCKFORD) for _ in range(PAIR_CODE_LENGTH))
    return "-".join(raw[index : index + 4] for index in range(0, len(raw), 4))


def hash_pair_code(code: str, pepper: str | None = None) -> str:
    """HMAC a pairing code before it is stored in Postgres."""
    key = (pepper or get_sync_pepper()).encode()
    normalized = ("pair:v1:" + normalize_pair_code(code)).encode()
    return hmac.new(key, normalized, hashlib.sha256).hexdigest()


def short_device_id(device_id: object) -> str:
    """Return the stable prefix shown by /devices and accepted by /revoke."""
    return str(device_id).split("-", 1)[0]
