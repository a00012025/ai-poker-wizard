"""GTO Wizard request signing (google-anal-id header).

Generates ECDSA P-256 signed headers required by the /v1/token/refresh/ endpoint.
Keypair is generated once per installation and persisted alongside tokens.
"""

import base64
import json
import logging
import time

import requests
from email.utils import parsedate_to_datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils as ec_utils
from cryptography.hazmat.backends import default_backend

log = logging.getLogger(__name__)

_VERSION = "v1"
_ORIGIN = "https://app.gtowizard.com"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
_BUILD_VERSION = "2026-03-23.2e78d975"

# Server time offset (ms), synced lazily
_server_time_offset: int | None = None


# ── Key helpers ──

def _b64url_decode(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    return base64.b64decode(s)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def generate_keypair_jwk() -> dict:
    """Generate a new ECDSA P-256 keypair, return as JWK dict for storage."""
    key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    priv = key.private_numbers()
    pub = priv.public_numbers
    return {
        "publicJwk": {
            "crv": "P-256", "kty": "EC",
            "x": _b64url_encode(pub.x.to_bytes(32, "big")),
            "y": _b64url_encode(pub.y.to_bytes(32, "big")),
        },
        "privateJwk": {
            "crv": "P-256", "kty": "EC",
            "d": _b64url_encode(priv.private_value.to_bytes(32, "big")),
            "x": _b64url_encode(pub.x.to_bytes(32, "big")),
            "y": _b64url_encode(pub.y.to_bytes(32, "big")),
        },
    }


def _load_private_key(jwk: dict):
    d = int.from_bytes(_b64url_decode(jwk["d"]), "big")
    x = int.from_bytes(_b64url_decode(jwk["x"]), "big")
    y = int.from_bytes(_b64url_decode(jwk["y"]), "big")
    pub = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1())
    return ec.EllipticCurvePrivateNumbers(d, pub).private_key(default_backend())


def _export_spki_b64(private_key) -> str:
    der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(der).decode()


def _sign_raw_b64(private_key, data: str) -> str:
    """ECDSA P-256 SHA-256, return base64 of raw r||s (64 bytes)."""
    der_sig = private_key.sign(data.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
    r, s = ec_utils.decode_dss_signature(der_sig)
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return base64.b64encode(raw).decode()


# ── Server time sync ──

def _sync_server_time():
    global _server_time_offset
    try:
        r = requests.get(
            "https://api.gtowizard.com/v4/core/server-time/", timeout=10
        )
        dt = parsedate_to_datetime(r.headers["date"])
        server_ms = int(dt.timestamp() * 1000)
        local_ms = int(time.time() * 1000)
        _server_time_offset = server_ms - local_ms
    except Exception as e:
        log.warning("Server time sync failed: %s", e)
        _server_time_offset = 0


def _get_server_time_ms() -> int:
    if _server_time_offset is None:
        _sync_server_time()
    return int(time.time() * 1000) + (_server_time_offset or 0)


# ── Public API ──

def sign_refresh_request(
    refresh_token: str,
    keypair_jwk: dict,
    app_uid: str = "00000000-0000-0000-0000-000000000000",
) -> dict:
    """Build signed headers for POST /v1/token/refresh/.

    Args:
        refresh_token: The JWT refresh token.
        keypair_jwk: Dict with "privateJwk" and "publicJwk" keys.
        app_uid: Client identifier (gwclientid header).

    Returns:
        Dict of extra headers to include in the request.
    """
    private_key = _load_private_key(keypair_jwk["privateJwk"])
    timestamp = _get_server_time_ms()

    body_str = json.dumps({"refresh": refresh_token}, separators=(",", ":"))

    # createSignaturePayload: METHOD|PATH|TIMESTAMP|BODY|ORIGIN|UA|APPUID|BUILDVER
    parts = [
        "POST",
        "/v1/token/refresh/",
        str(timestamp),
        body_str,
        _ORIGIN,
        _USER_AGENT,
        app_uid,
        _BUILD_VERSION,
    ]
    sign_payload = "|".join(parts)

    sig_b64 = _sign_raw_b64(private_key, sign_payload)
    spki_b64 = _export_spki_b64(private_key)

    headers_obj = {
        "origin": _ORIGIN,
        "userAgent": _USER_AGENT,
        "appUid": app_uid,
        "buildVersion": _BUILD_VERSION,
    }
    headers_b64 = base64.b64encode(
        json.dumps(headers_obj, separators=(",", ":")).encode()
    ).decode()

    google_anal_id = ".".join([
        sig_b64, spki_b64, str(timestamp), _VERSION, headers_b64
    ])

    return {
        "google-anal-id": google_anal_id,
        "origin": _ORIGIN,
        "user-agent": _USER_AGENT,
        "content-type": "application/json",
    }
