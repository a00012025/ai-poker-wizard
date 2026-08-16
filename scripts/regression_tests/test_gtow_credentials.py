"""Regression coverage for browser-first GTOW credential handling."""

from __future__ import annotations

import base64
import json
import threading
import time

from regression_tests.harness import (
    REPO_ROOT,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
)


def _jwt(*, exp: int, iat: int | None = None, marker: str = "token") -> str:
    payload = {"exp": exp, "marker": marker}
    if iat is not None:
        payload["iat"] = iat
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    return f"eyJhbGciOiJub25lIn0.{encoded}.signature"


class _FakeStore:
    def __init__(self, bundle):
        self.bundle = bundle
        self.lock = threading.Lock()
        self.refresh_calls = 0
        self.generated_keypairs = 0

    def load(self, user_id):
        assert_eq(user_id, 42)
        return self.bundle

    def refresh_if_expired(self, user_id, *, now, refresh_fn, keypair_factory):
        with self.lock:
            if self.bundle.access_token and self.bundle.access_exp > now:
                return self.bundle
            self.refresh_calls += 1
            keypair = self.bundle.signing_keypair
            if keypair is None:
                self.generated_keypairs += 1
                keypair = keypair_factory()
            access = refresh_fn(self.bundle.refresh_token, keypair)
            self.bundle = self.bundle.with_backend_access(
                access_token=access,
                access_exp=float(json.loads(base64.urlsafe_b64decode(
                    access.split(".")[1] + "=="))["exp"]),
                signing_keypair=keypair,
            )
            return self.bundle


def test_browser_access_is_used_until_actual_expiry_without_refreshing():
    """A browser access token remains authoritative until its real JWT exp."""
    from gto_credentials import CredentialProvider, SessionBundle

    now = 1_000.0
    access = _jwt(exp=1001, iat=900, marker="browser")
    store = _FakeStore(SessionBundle(
        user_id=42,
        refresh_token="refresh",
        access_token=access,
        access_exp=1001,
        client_id="browser-client",
        signing_keypair=None,
        access_source="extension",
    ))
    refreshed = []
    provider = CredentialProvider(
        store=store,
        now=lambda: now,
        refresh_fn=lambda *_args: refreshed.append(True),
        keypair_factory=lambda: {"privateJwk": {}, "publicJwk": {}},
    )

    credentials = provider.get(42)

    assert_eq(credentials.access_token, access)
    assert_eq(credentials.client_id, "browser-client")
    assert_eq(refreshed, [])
    assert_eq(store.refresh_calls, 0)


def test_expired_access_refreshes_once_with_one_stable_keypair():
    """Concurrent expired readers single-flight and persist one backend keypair."""
    from gto_credentials import CredentialProvider, SessionBundle

    now = 2_000.0
    expired = _jwt(exp=1999, marker="expired")
    minted = _jwt(exp=2600, iat=2000, marker="backend")
    store = _FakeStore(SessionBundle(
        user_id=42,
        refresh_token="refresh",
        access_token=expired,
        access_exp=1999,
        client_id="browser-client",
        signing_keypair=None,
        access_source="extension",
    ))
    provider = CredentialProvider(
        store=store,
        now=lambda: now,
        refresh_fn=lambda refresh, keypair: minted,
        keypair_factory=lambda: {
            "privateJwk": {"d": "stable"},
            "publicJwk": {"x": "stable"},
        },
    )
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(provider.get(42)))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert_eq({result.access_token for result in results}, {minted})
    assert_eq(store.refresh_calls, 1)
    assert_eq(store.generated_keypairs, 1)
    assert_eq(store.bundle.signing_keypair["privateJwk"]["d"], "stable")
    assert_eq(store.bundle.access_source, "backend_refresh")


def test_unexpired_401_is_not_allowed_to_force_refresh():
    """A 401 cannot manufacture a second session while JWT exp is still valid."""
    from gto_credentials import access_is_expired

    token = _jwt(exp=5000, marker="revoked")
    assert_true(not access_is_expired(token, access_exp=5000, now=4999.9))
    assert_true(access_is_expired(token, access_exp=5000, now=5000.0))


def test_unexpired_401_can_reload_new_browser_access_without_refresh():
    """A replacement browser bundle recovers 401 without creating a session."""
    from gto_credentials import CredentialProvider, SessionBundle

    old = _jwt(exp=5000, iat=4000, marker="old")
    new = _jwt(exp=5100, iat=4100, marker="new")
    store = _FakeStore(SessionBundle(
        user_id=42,
        refresh_token="refresh",
        access_token=new,
        access_exp=5100,
        client_id="new-browser-client",
        signing_keypair=None,
        access_source="extension",
    ))
    provider = CredentialProvider(store=store, now=lambda: 4500)

    replacement = provider.reload_if_changed(42, previous_access=old)

    assert_eq(replacement.access_token, new)
    assert_eq(replacement.client_id, "new-browser-client")
    assert_eq(store.refresh_calls, 0)


def test_subprocess_auth_uses_user_identity_not_refresh_secret():
    """Live and ingest children receive GTOW_USER_ID, never a refresh token."""
    bot = (REPO_ROOT / "src/telegram_bot/bot.py").read_text()
    ingest = (REPO_ROOT / "src/ingest_runner.py").read_text()

    live_body = bot[
        bot.index("async def _process_live_batch"):
        bot.index("async def _apply_live_resend")
    ]
    pipeline_body = ingest[
        ingest.index("async def run_pipeline"):
        ingest.index("async def _claim_next")
    ]
    assert_in('"GTOW_USER_ID"', live_body)
    assert_not_in('"GTOW_REFRESH_TOKEN":', live_body)
    assert_in('"GTOW_USER_ID"', pipeline_body)
    assert_not_in('"GTOW_REFRESH_TOKEN":', pipeline_body)


def test_session_bundle_migration_is_private_and_single_flight():
    """The DB contract stores the bundle privately and serializes refreshes."""
    migration = (
        REPO_ROOT / "supabase/migrations/20260731000000_gtow_session_bundle.sql"
    ).read_text()

    for column in (
        "gto_access_token",
        "gto_access_token_exp",
        "gto_client_id",
        "gto_backend_signing_keypair",
        "gto_access_token_source",
    ):
        assert_in(column, migration)
    assert_in("pg_advisory_xact_lock", migration)
    assert_in("sync_gtow_session_bundle", migration)
    assert_in("REVOKE ALL", migration)
    assert_not_in("GRANT SELECT ON public.users TO anon", migration)
