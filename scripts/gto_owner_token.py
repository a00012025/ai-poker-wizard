#!/usr/bin/env python3
"""Resolve the owner's DB-synced GTO Wizard refresh token for local tooling.

Telegram requests carry their own user's token through thread-local auth.  This
module is only the fallback for owner-run CLI tools and regression tests.  It
uses a synchronous DB client deliberately: callers include synchronous solver
functions that may be invoked while an asyncio event loop is already running.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def resolve_owner_db_token() -> tuple[int, str] | None:
    """Return ``(owner_user_id, refresh_token)`` from ``users`` when resolvable."""
    conn_str = os.environ.get("SUPABASE_CONN")
    if not conn_str:
        return None

    owner_env = os.environ.get("OWNER_CHAT_ID")
    with psycopg2.connect(conn_str, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            if owner_env:
                owner_id = int(owner_env)
            else:
                cur.execute("SELECT user_id FROM users WHERE is_active")
                rows = cur.fetchall()
                if len(rows) != 1:
                    return None
                owner_id = int(rows[0][0])
            cur.execute(
                "SELECT gto_refresh_token FROM users WHERE user_id = %s",
                (owner_id,),
            )
            row = cur.fetchone()
    if not row or not row[0]:
        return None
    return owner_id, str(row[0])


def bootstrap_owner_db_token(*, verbose: bool = True) -> bool:
    """Populate ``GTOW_REFRESH_TOKEN`` from the owner's DB row.

    An explicitly provided environment token always wins.  Returns whether a
    usable token is available; failures remain secret-safe and never fall back
    to local files.
    """
    if os.environ.get("GTOW_REFRESH_TOKEN"):
        return True
    try:
        result = resolve_owner_db_token()
    except Exception as exc:
        if verbose:
            print(f"[gto-token] owner DB token bootstrap failed ({exc})",
                  file=sys.stderr)
        return False
    if not result:
        if verbose:
            print("[gto-token] no owner DB GTO token", file=sys.stderr)
        return False
    owner_id, refresh_token = result
    os.environ["GTOW_REFRESH_TOKEN"] = refresh_token
    if verbose:
        print(f"[gto-token] using owner {owner_id} DB token (shared GTO session)",
              file=sys.stderr)
    return True
