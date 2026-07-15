#!/usr/bin/env python3
"""Regression test suite for core analysis logic.

Usage:
    python scripts/regression_test.py          # Run all tests
    python scripts/regression_test.py -v       # Verbose output
    python scripts/regression_test.py -k chip  # Run tests matching "chip"

Requires a valid GTO Wizard token and network access for API-backed cases.
The test cases live in ``scripts/regression_tests/`` and remain executable
through this compatibility entry point.

Token source: API-backed cases run on the OWNER's DB-synced GTOW refresh token
(``users.gto_refresh_token``), injected via ``GTOW_REFRESH_TOKEN`` before the
run. This shares the owner's live GTOW session (the browser/extension keeps that
row fresh) instead of minting a SECOND session from ``.tokens.json`` — a second
session trips GTOW's "too many sessions" FORCED_LOGOUT and kicks the owner's own
browser login. Set ``GTOW_REFRESH_TOKEN`` yourself to override; the bootstrap is
a no-op then. Falls back to ``.tokens.json`` only if the DB token is unavailable.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_owner_db_token() -> None:
    """Inject the owner's DB-synced GTOW token as GTOW_REFRESH_TOKEN.

    No-op when GTOW_REFRESH_TOKEN is already set (e.g. the ingest subprocess) or
    when the DB / owner token cannot be resolved — in which case API-backed
    tests fall back to the legacy .tokens.json path. Owner resolution mirrors
    ledger_service.resolve_owner_chat_id (OWNER_CHAT_ID env, else sole active
    user).
    """
    if os.environ.get("GTOW_REFRESH_TOKEN"):
        return
    conn_str = os.environ.get("SUPABASE_CONN")
    if not conn_str:
        print("[regression] SUPABASE_CONN unset — API tests use .tokens.json",
              file=sys.stderr)
        return
    try:
        import asyncio
        import asyncpg

        async def _fetch():
            conn = await asyncpg.connect(conn_str, statement_cache_size=0)
            try:
                owner_env = os.environ.get("OWNER_CHAT_ID")
                if owner_env:
                    uid = int(owner_env)
                else:
                    rows = await conn.fetch(
                        "SELECT user_id FROM users WHERE is_active")
                    if len(rows) != 1:
                        return None
                    uid = rows[0]["user_id"]
                row = await conn.fetchrow(
                    "SELECT gto_refresh_token FROM users WHERE user_id = $1", uid)
                if row and row["gto_refresh_token"]:
                    return uid, row["gto_refresh_token"]
                return None
            finally:
                await conn.close()

        result = asyncio.run(_fetch())
    except Exception as exc:  # DB down / driver missing — never block the suite
        print(f"[regression] DB token bootstrap failed ({exc}) — "
              "API tests use .tokens.json", file=sys.stderr)
        return
    if not result:
        print("[regression] no owner DB GTOW token — API tests use .tokens.json",
              file=sys.stderr)
        return
    uid, token = result
    os.environ["GTOW_REFRESH_TOKEN"] = token
    print(f"[regression] GTOW token: owner {uid} DB row "
          "(shared session, .tokens.json untouched)", file=sys.stderr)


_bootstrap_owner_db_token()

from regression_tests import load_all  # noqa: E402
from regression_tests.harness import run_tests  # noqa: E402

load_all()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
