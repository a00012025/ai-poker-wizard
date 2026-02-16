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

    # Refresh failed — need browser login
    refresh = _browser_login()
    if not refresh:
        print("登入失敗", file=sys.stderr)
        sys.exit(1)

    access = _refresh_access(refresh)
    if not access:
        print("Token refresh 失敗", file=sys.stderr)
        sys.exit(1)

    _save_tokens({"refresh": refresh, "access": access})
    return access


if __name__ == "__main__":
    token = get_access_token()
    print(token)
