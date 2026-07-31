# GTOW browser-first session bundle

## Decision

The logged-in GTO Wizard browser session is authoritative. The extension
captures the access token, refresh token, `GWCLIENTID`, and access expiry and
sends the bundle through the existing paired-device endpoint.

All production solver and Analyze API calls resolve credentials by user ID:

1. Use the synchronized browser access token until its actual JWT `exp`.
2. Do not refresh merely because GTOW returned 401 while that token is valid.
3. Once expired, acquire a per-user PostgreSQL advisory lock, re-read the row,
   and refresh at most once with the stored signing keypair.
4. Persist the resulting access token for all processes to reuse.

## Security and lifecycle

- Raw GTOW credentials are never written to Chrome storage.
- The database RPC is `SECURITY DEFINER`, device-authenticated, and unavailable
  to public, anonymous, or authenticated client roles.
- Access-token ordering uses the signed JWT `iat`; client wall-clock
  `observed_at` is only a fallback when the stored row has no access `iat`.
- `/logout` atomically clears refresh, access, client ID, and backend signing
  keypair.
- `GTOW_REFRESH_TOKEN` remains only as a rollout-compatible owner CLI/test
  fallback. Production child processes receive `GTOW_USER_ID` only.

## Operational note

Deploy the database migration before the Edge Function and extension update.
Refresh-only extension sync requests are rejected with
`GTOW_SESSION_BUNDLE_REQUIRED`; they never validate by calling GTOW's refresh
endpoint. Upgrade the extension to v2.3 before using manual sync or ingest.
