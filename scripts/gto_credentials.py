#!/usr/bin/env python3
"""Browser-first GTO Wizard credential provider.

The extension-synced browser access token is authoritative until its actual
JWT expiry.  Only then may the backend exchange the stored refresh token, using
one persistent signing keypair and a PostgreSQL advisory lock so concurrent
processes cannot create duplicate GTOW sessions.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Protocol

import psycopg2

from gto_signing import generate_keypair_jwk
from gto_token import TokenExpiredError, _jwt_exp, _refresh_access


class CredentialUnavailable(TokenExpiredError):
    """No usable synchronized credential exists for this user."""


class SessionBundleMissing(CredentialUnavailable):
    """The rollout-compatible synchronized row/columns are not available."""


def _jwt_claims(token: str | None) -> dict:
    if not token:
        return {}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def access_is_expired(
    token: str | None,
    *,
    access_exp: float | datetime | None = None,
    now: float | None = None,
) -> bool:
    """Return True only at/after the access token's real expiry.

    No proactive 60-second refresh buffer is used: avoiding unnecessary GTOW
    refresh sessions is more important than hiding a single retry's latency.
    """
    current = time.time() if now is None else float(now)
    claims_exp = _jwt_claims(token).get("exp")
    if claims_exp is not None:
        try:
            return float(claims_exp) <= current
        except (TypeError, ValueError):
            pass
    if isinstance(access_exp, datetime):
        expiry = access_exp.timestamp()
    elif access_exp is not None:
        expiry = float(access_exp)
    else:
        return True
    return expiry <= current


@dataclass(frozen=True)
class SessionBundle:
    user_id: int
    refresh_token: str | None
    access_token: str | None
    access_exp: float
    client_id: str | None
    signing_keypair: dict | None
    access_source: str | None

    def with_backend_access(
        self,
        *,
        access_token: str,
        access_exp: float,
        signing_keypair: dict,
    ) -> "SessionBundle":
        return replace(
            self,
            access_token=access_token,
            access_exp=access_exp,
            signing_keypair=signing_keypair,
            access_source="backend_refresh",
        )


class CredentialStore(Protocol):
    def load(self, user_id: int) -> SessionBundle | None: ...

    def refresh_if_expired(
        self,
        user_id: int,
        *,
        now: float,
        refresh_fn: Callable[[str, dict], str | None],
        keypair_factory: Callable[[], dict],
    ) -> SessionBundle: ...


_SESSION_COLUMNS = """
    user_id,
    gto_refresh_token,
    gto_access_token,
    gto_access_token_exp,
    gto_client_id,
    gto_backend_signing_keypair,
    gto_access_token_source
"""


def _timestamp(value) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if value is None:
        return 0.0
    return float(value)


def _bundle_from_row(row) -> SessionBundle | None:
    if not row:
        return None
    return SessionBundle(
        user_id=int(row[0]),
        refresh_token=str(row[1]) if row[1] else None,
        access_token=str(row[2]) if row[2] else None,
        access_exp=_timestamp(row[3]),
        client_id=str(row[4]) if row[4] else None,
        signing_keypair=row[5],
        access_source=str(row[6]) if row[6] else None,
    )


class PostgresCredentialStore:
    """Synchronous store for solver code that already runs off the event loop."""

    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    def load(self, user_id: int) -> SessionBundle | None:
        with psycopg2.connect(self.conn_str, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM users WHERE user_id = %s",
                    (int(user_id),),
                )
                return _bundle_from_row(cur.fetchone())

    def refresh_if_expired(
        self,
        user_id: int,
        *,
        now: float,
        refresh_fn: Callable[[str, dict], str | None],
        keypair_factory: Callable[[], dict],
    ) -> SessionBundle:
        with psycopg2.connect(self.conn_str, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                      hashtextextended('gtow-session:' || %s::text, 0)
                    )
                    """,
                    (int(user_id),),
                )
                cur.execute(
                    f"""
                    SELECT {_SESSION_COLUMNS}
                    FROM users
                    WHERE user_id = %s
                    FOR UPDATE
                    """,
                    (int(user_id),),
                )
                bundle = _bundle_from_row(cur.fetchone())
                if not bundle or not bundle.refresh_token:
                    raise SessionBundleMissing(
                        "找不到有效的 GTOW session；請開啟 GTOW 並讓 Extension 同步。"
                    )
                if not access_is_expired(
                    bundle.access_token,
                    access_exp=bundle.access_exp,
                    now=now,
                ):
                    return bundle

                keypair = bundle.signing_keypair or keypair_factory()
                access = refresh_fn(bundle.refresh_token, keypair)
                if not access:
                    raise CredentialUnavailable(
                        "GTOW access refresh 失敗；請重新登入 GTOW 並讓 Extension 同步。"
                    )
                access_exp = float(_jwt_exp(access))
                claims = _jwt_claims(access)
                access_iat = claims.get("iat")
                refreshed = bundle.with_backend_access(
                    access_token=access,
                    access_exp=access_exp,
                    signing_keypair=keypair,
                )
                cur.execute(
                    """
                    UPDATE users
                    SET gto_access_token = %s,
                        gto_access_token_fingerprint = %s,
                        gto_access_token_iat = %s,
                        gto_access_token_exp = %s,
                        gto_access_token_source = 'backend_refresh',
                        gto_backend_signing_keypair = %s::jsonb,
                        gto_session_observed_at = NOW()
                    WHERE user_id = %s
                    """,
                    (
                        access,
                        hashlib.sha256(access.encode()).hexdigest(),
                        (
                            datetime.fromtimestamp(float(access_iat), timezone.utc)
                            if access_iat is not None else None
                        ),
                        datetime.fromtimestamp(access_exp, timezone.utc),
                        json.dumps(keypair, separators=(",", ":")),
                        int(user_id),
                    ),
                )
                return refreshed


class CredentialProvider:
    """Process-local cache backed by a cross-process single-flight store."""

    def __init__(
        self,
        *,
        store: CredentialStore,
        now: Callable[[], float] = time.time,
        refresh_fn: Callable[[str, dict], str | None] = _refresh_access,
        keypair_factory: Callable[[], dict] = generate_keypair_jwk,
    ):
        self.store = store
        self.now = now
        self.refresh_fn = refresh_fn
        self.keypair_factory = keypair_factory
        self._cache: dict[int, SessionBundle] = {}
        self._lock = threading.RLock()

    def get(self, user_id: int) -> SessionBundle:
        uid = int(user_id)
        current = self.now()
        with self._lock:
            cached = self._cache.get(uid)
            if cached and not access_is_expired(
                cached.access_token,
                access_exp=cached.access_exp,
                now=current,
            ):
                return cached
            latest = self.store.load(uid)
            if latest and not access_is_expired(
                latest.access_token,
                access_exp=latest.access_exp,
                now=current,
            ):
                self._cache[uid] = latest
                return latest
            refreshed = self.store.refresh_if_expired(
                uid,
                now=current,
                refresh_fn=self.refresh_fn,
                keypair_factory=self.keypair_factory,
            )
            self._cache[uid] = refreshed
            return refreshed

    def invalidate(self, user_id: int) -> None:
        with self._lock:
            self._cache.pop(int(user_id), None)

    def reload_if_changed(
        self,
        user_id: int,
        *,
        previous_access: str,
    ) -> SessionBundle | None:
        """Reload an extension update without ever minting a new session.

        This is the recovery path for a still-unexpired token returning 401:
        another browser request may already have synchronized a replacement.
        """
        uid = int(user_id)
        current = self.now()
        with self._lock:
            latest = self.store.load(uid)
            if (
                latest
                and latest.access_token != previous_access
                and not access_is_expired(
                    latest.access_token,
                    access_exp=latest.access_exp,
                    now=current,
                )
            ):
                self._cache[uid] = latest
                return latest
            return None


_provider: CredentialProvider | None = None
_provider_lock = threading.Lock()


def _default_provider() -> CredentialProvider:
    global _provider
    with _provider_lock:
        if _provider is None:
            conn_str = os.environ.get("SUPABASE_CONN")
            if not conn_str:
                raise CredentialUnavailable("SUPABASE_CONN 未設定，無法讀取 GTOW session。")
            _provider = CredentialProvider(
                store=PostgresCredentialStore(conn_str),
            )
        return _provider


def get_synced_credentials(user_id: int) -> SessionBundle:
    return _default_provider().get(int(user_id))


def invalidate_synced_credentials(user_id: int) -> None:
    if _provider is not None:
        _provider.invalidate(int(user_id))


def reload_synced_credentials_if_changed(
    user_id: int,
    previous_access: str,
) -> SessionBundle | None:
    """Read a newer browser session from DB without invoking refresh."""
    return _default_provider().reload_if_changed(
        int(user_id),
        previous_access=previous_access,
    )


def _legacy_credentials(
    refresh: str,
    *,
    user_id: int = -1,
    client_id: str | None = None,
) -> SessionBundle:
    from gto_token import get_user_access_token

    token = get_user_access_token(int(user_id), refresh)
    return SessionBundle(
        user_id=int(user_id),
        refresh_token=refresh,
        access_token=token,
        access_exp=float(_jwt_exp(token)),
        client_id=client_id,
        signing_keypair=None,
        access_source="legacy_refresh",
    )


def get_user_credentials(
    user_id: int,
    *,
    fallback_refresh: str | None = None,
) -> SessionBundle:
    """Use the synchronized bundle, with a rollout/test compatibility fallback."""
    try:
        return get_synced_credentials(int(user_id))
    except (SessionBundleMissing, psycopg2.errors.UndefinedColumn):
        if fallback_refresh:
            return _legacy_credentials(fallback_refresh, user_id=int(user_id))
        raise


def request_credentials(
    *,
    user_id: int | None = None,
    fallback_refresh: str | None = None,
    fallback_client_id: str | None = None,
) -> SessionBundle:
    """Resolve credentials for bot, subprocess, and owner CLI request paths."""
    env_user = os.environ.get("GTOW_USER_ID")
    resolved_user = int(env_user) if env_user else user_id
    if resolved_user is not None:
        return get_synced_credentials(int(resolved_user))

    access = os.environ.get("GTOW_ACCESS_TOKEN")
    access_exp_raw = os.environ.get("GTOW_ACCESS_TOKEN_EXP")
    access_exp = float(access_exp_raw) if access_exp_raw else float(
        _jwt_claims(access).get("exp") or 0
    )
    if access and not access_is_expired(access, access_exp=access_exp):
        return SessionBundle(
            user_id=-1,
            refresh_token=None,
            access_token=access,
            access_exp=access_exp,
            client_id=os.environ.get("GTOW_CLIENT_ID") or fallback_client_id,
            signing_keypair=None,
            access_source="explicit_access",
        )

    refresh = fallback_refresh or os.environ.get("GTOW_REFRESH_TOKEN")
    if refresh:
        # Compatibility path for local tests and explicit operator tooling.
        return _legacy_credentials(
            refresh,
            client_id=os.environ.get("GTOW_CLIENT_ID") or fallback_client_id,
        )
    raise CredentialUnavailable(
        "找不到 GTOW session；請開啟 GTOW 並讓 Extension 同步。"
    )
