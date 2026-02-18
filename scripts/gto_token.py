#!/usr/bin/env python3
"""GTO Wizard token management.

Stores refresh token locally, auto-refreshes access tokens.
Falls back to browser login if refresh fails.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_FILE = _PROJECT_ROOT / ".tokens.json"

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"


class TokenExpiredError(Exception):
    """Raised when GTO Wizard tokens are expired and cannot be refreshed automatically."""
    pass


def _load_tokens() -> dict:
    if _TOKEN_FILE.exists():
        return json.loads(_TOKEN_FILE.read_text())
    return {}


def _save_tokens(data: dict):
    _TOKEN_FILE.write_text(json.dumps(data, indent=2))


def _jwt_exp(token: str) -> int:
    """Extract exp from JWT without verification."""
    import base64
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["exp"]


def _refresh_access(refresh_token: str) -> str | None:
    """Exchange refresh token for a new access token."""
    try:
        r = requests.post(
            f"{API_BASE}/v1/token/refresh/",
            json={"refresh": refresh_token},
            headers={"origin": ORIGIN, "content-type": "application/json"},
            timeout=10,
        )
        if r.ok:
            return r.json()["access"]
    except Exception:
        pass
    return None


def _browser_login() -> str | None:
    """Open browser for manual login, extract refresh token from localStorage."""
    print("需要登入 GTO Wizard，正在開啟瀏覽器...", file=sys.stderr)
    subprocess.run(["agent-browser", "close"], capture_output=True)
    subprocess.run(
        ["agent-browser", "--session-name", "gto-wizard", "--headed",
         "open", f"{ORIGIN}/solutions"],
        capture_output=True, timeout=30,
    )
    print("請在瀏覽器中登入 GTO Wizard，完成後按 Enter...", file=sys.stderr)
    input()
    result = subprocess.run(
        ["agent-browser", "--session-name", "gto-wizard",
         "eval", "localStorage.getItem('user_refresh')"],
        capture_output=True, text=True, timeout=15,
    )
    token = result.stdout.strip().strip('"')
    if token and token.startswith("eyJ"):
        return token
    return None


def ensure_session() -> bool:
    """Check if GTO Wizard session can be refreshed. Opens browser if needed.

    Returns True if session is valid.
    Returns False if browser login is needed (browser has been opened).
    """
    tokens = _load_tokens()

    # Check access token
    access = tokens.get("access")
    if access:
        try:
            if _jwt_exp(access) > time.time() + 60:
                return True
        except Exception:
            pass

    # Try refresh
    refresh = tokens.get("refresh")
    if refresh:
        try:
            if _jwt_exp(refresh) > time.time():
                access = _refresh_access(refresh)
                if access:
                    tokens["access"] = access
                    _save_tokens(tokens)
                    return True
        except Exception:
            pass

    # Refresh token expired
    return False


def capture_browser_token() -> bool:
    """Try to capture refresh token from browser after user logs in.

    Returns True if token was captured and session is now valid.
    """
    try:
        result = subprocess.run(
            ["agent-browser", "--session-name", "gto-wizard",
             "eval", "localStorage.getItem('user_refresh')"],
            capture_output=True, text=True, timeout=15,
        )
        token = result.stdout.strip().strip('"')
        if token and token.startswith("eyJ"):
            access = _refresh_access(token)
            if access:
                _save_tokens({"refresh": token, "access": access})
                return True
    except Exception:
        pass
    return False


def get_access_token() -> str:
    """Get a valid access token, refreshing or re-logging in as needed."""
    tokens = _load_tokens()

    # Check if current access token is still valid (with 60s buffer)
    access = tokens.get("access")
    if access:
        try:
            if _jwt_exp(access) > time.time() + 60:
                return access
        except Exception:
            pass

    # Try refresh
    refresh = tokens.get("refresh")
    if refresh:
        try:
            if _jwt_exp(refresh) < time.time():
                refresh = None  # expired
        except Exception:
            refresh = None

    if refresh:
        access = _refresh_access(refresh)
        if access:
            tokens["access"] = access
            _save_tokens(tokens)
            return access

    # Refresh failed — raise error instead of browser login (container-safe)
    raise TokenExpiredError(
        "GTO Wizard token 過期，無法自動刷新。請手動更新 .tokens.json"
    )


if __name__ == "__main__":
    token = get_access_token()
    print(token)
