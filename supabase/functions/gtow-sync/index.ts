import { createClient } from "npm:@supabase/supabase-js@2";

const encoder = new TextEncoder();
const MAX_BODY_BYTES = 8192;

class ServiceError extends Error {
  constructor(message: string, readonly status = 500) {
    super(message);
  }
}

function corsHeaders(req: Request): Record<string, string> {
  const origin = req.headers.get("origin") || "";
  return {
    "Access-Control-Allow-Origin": origin.startsWith("chrome-extension://")
      ? origin
      : "https://app.gtowizard.com",
    "Access-Control-Allow-Headers": "authorization, content-type, apikey",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}

function json(req: Request, body: unknown, status = 200): Response {
  return Response.json(body, { status, headers: corsHeaders(req) });
}

function base64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(
    /=+$/,
    "",
  );
}

async function hmacHex(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = new Uint8Array(
    await crypto.subtle.sign("HMAC", key, encoder.encode(value)),
  );
  return [...signature].map((byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", encoder.encode(value)),
  );
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function normalizePairCode(value: unknown): string {
  return typeof value === "string"
    ? value.toUpperCase().replace(/[^A-Z0-9]/g, "")
    : "";
}

function deviceSecret(req: Request): string | null {
  const match = (req.headers.get("authorization") || "").match(
    /^Device\s+([A-Za-z0-9_-]{32,})$/,
  );
  return match?.[1] || null;
}

function parseRefreshToken(
  token: unknown,
): { token: string; iat: number; exp: number } | null {
  if (typeof token !== "string" || token.length < 100 || token.length > 4096) {
    return null;
  }
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[0].startsWith("eyJ")) return null;
  try {
    const padded = parts[1].replaceAll("-", "+").replaceAll("_", "/")
      .padEnd(Math.ceil(parts[1].length / 4) * 4, "=");
    const claims = JSON.parse(atob(padded));
    const iat = Number(claims.iat);
    const exp = Number(claims.exp);
    if (
      !Number.isFinite(iat) || !Number.isFinite(exp) ||
      iat > Math.floor(Date.now() / 1000) + 300 ||
      exp <= Math.floor(Date.now() / 1000) + 60
    ) {
      return null;
    }
    return { token, iat, exp };
  } catch {
    return null;
  }
}

function parseJwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[0].startsWith("eyJ")) return null;
  try {
    const padded = parts[1].replaceAll("-", "+").replaceAll("_", "/")
      .padEnd(Math.ceil(parts[1].length / 4) * 4, "=");
    const claims = JSON.parse(atob(padded));
    return claims && typeof claims === "object" && !Array.isArray(claims)
      ? claims as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function parseAccessExp(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && /^\d{10}$/.test(value)) return Number(value);
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return Math.floor(parsed / 1000);
  }
  return null;
}

function parseObservedAt(value: unknown): string | null {
  if (typeof value !== "string" || value.length < 20 || value.length > 40) {
    return null;
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return null;
  const now = Date.now();
  if (parsed > now + 5 * 60 * 1000 || parsed < now - 24 * 60 * 60 * 1000) {
    return null;
  }
  return new Date(parsed).toISOString();
}

function parseClientId(value: unknown): string | null {
  if (value === undefined || value === null || value === "") return "";
  if (typeof value !== "string") return null;
  const clientId = value.trim();
  if (clientId.length > 128 || !/^[A-Za-z0-9._:-]+$/.test(clientId)) {
    return null;
  }
  return clientId;
}

function parseAccessToken(
  token: unknown,
  accessExpInput: unknown,
): { token: string; iat: number; exp: number } | null {
  if (typeof token !== "string" || token.length < 100 || token.length > 8192) {
    return null;
  }
  const claims = parseJwtPayload(token);
  if (!claims) return null;
  const iat = Number(claims.iat);
  const tokenExp = Number(claims.exp);
  const suppliedExp = parseAccessExp(accessExpInput);
  const exp = Number.isFinite(tokenExp) ? tokenExp : suppliedExp;
  const now = Math.floor(Date.now() / 1000);
  if (
    !Number.isFinite(iat) || exp === null || !Number.isFinite(exp) ||
    iat > now + 300 ||
    exp <= now + 5
  ) {
    return null;
  }
  if (suppliedExp !== null && Math.abs(suppliedExp - exp) > 5) return null;
  return { token, iat, exp };
}

async function readBody(req: Request): Promise<Record<string, unknown>> {
  if (Number(req.headers.get("content-length") || "0") > MAX_BODY_BYTES) {
    throw new Error("BODY_TOO_LARGE");
  }
  const text = await req.text();
  if (encoder.encode(text).byteLength > MAX_BODY_BYTES) {
    throw new Error("BODY_TOO_LARGE");
  }
  const parsed = JSON.parse(text || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("BODY_INVALID");
  }
  return parsed as Record<string, unknown>;
}

function adminClient() {
  const url = Deno.env.get("SUPABASE_URL");
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ||
    Deno.env.get("SUPABASE_SECRET_KEY");
  if (!url || !key) throw new Error("SUPABASE_CONFIGURATION_MISSING");
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

async function pair(req: Request, pepper: string): Promise<Response> {
  const body = await readBody(req);
  const code = normalizePairCode(body.code);
  const name = typeof body.device_name === "string"
    ? body.device_name.trim()
    : "";
  if (code.length !== 12 || name.length < 1 || name.length > 80) {
    return json(req, { error: "PAIRING_INPUT_INVALID" }, 400);
  }
  const rawCredential = base64Url(crypto.getRandomValues(new Uint8Array(32)));
  const [codeHash, credentialHash] = await Promise.all([
    hmacHex(pepper, `pair:v1:${code}`),
    hmacHex(pepper, `device:v1:${rawCredential}`),
  ]);
  const { data, error } = await adminClient().rpc("claim_gtow_device_pairing", {
    p_code_hash: codeHash,
    p_credential_hash: credentialHash,
    p_device_name: name,
  });
  if (error || !data?.[0]) {
    if (error && !error.message.includes("PAIRING_INVALID_OR_EXPIRED")) {
      throw new ServiceError("PAIRING_BACKEND_FAILED");
    }
    return json(req, { error: "PAIRING_INVALID_OR_EXPIRED" }, 400);
  }
  const { error: cleanupError } = await adminClient().rpc(
    "cleanup_gtow_token_sync_metadata",
  );
  if (cleanupError) console.error("GTOW sync metadata cleanup failed");
  return json(req, {
    device_id: data[0].device_id,
    device_secret: rawCredential,
    telegram_label: data[0].telegram_label,
  });
}

async function authenticateDevice(req: Request, pepper: string) {
  const secret = deviceSecret(req);
  if (!secret) return null;
  const credentialHash = await hmacHex(pepper, `device:v1:${secret}`);
  const { data, error } = await adminClient()
    .from("gtow_sync_devices")
    .select("id,user_id,name,last_seen_at,last_sync_at,created_at")
    .eq("credential_hash", credentialHash)
    .is("revoked_at", null)
    .maybeSingle();
  if (error) throw new ServiceError("DEVICE_LOOKUP_FAILED");
  return data ? { credentialHash, device: data } : null;
}

async function syncToken(req: Request, pepper: string): Promise<Response> {
  const authenticated = await authenticateDevice(req, pepper);
  if (!authenticated) return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
  const lastSync = authenticated.device.last_sync_at
    ? Date.parse(authenticated.device.last_sync_at)
    : 0;
  if (Date.now() - lastSync < 3000) {
    return json(req, { error: "SYNC_RATE_LIMITED" }, 429);
  }

  const body = await readBody(req);
  const parsed = parseRefreshToken(body.refresh_token);
  if (!parsed) return json(req, { error: "REFRESH_TOKEN_INVALID" }, 400);
  const hasAccessBundle = body.access_token !== undefined ||
    body.access_exp !== undefined ||
    body.gwclientid !== undefined;
  if (hasAccessBundle) {
    const access = parseAccessToken(body.access_token, body.access_exp);
    const observedAt = parseObservedAt(body.observed_at);
    const clientId = parseClientId(body.gwclientid);
    if (!access) return json(req, { error: "ACCESS_TOKEN_INVALID" }, 400);
    if (!observedAt) return json(req, { error: "OBSERVED_AT_INVALID" }, 400);
    if (clientId === null) return json(req, { error: "GWCLIENTID_INVALID" }, 400);
    const [refreshFingerprint, accessFingerprint] = await Promise.all([
      sha256Hex(parsed.token),
      sha256Hex(access.token),
    ]);
    const { data, error } = await adminClient().rpc(
      "sync_gtow_session_bundle",
      {
        p_credential_hash: authenticated.credentialHash,
        p_refresh_token: parsed.token,
        p_refresh_token_fingerprint: refreshFingerprint,
        p_refresh_token_iat: new Date(parsed.iat * 1000).toISOString(),
        p_refresh_token_exp: new Date(parsed.exp * 1000).toISOString(),
        p_access_token: access.token,
        p_access_token_fingerprint: accessFingerprint,
        p_access_token_iat: new Date(access.iat * 1000).toISOString(),
        p_access_token_exp: new Date(access.exp * 1000).toISOString(),
        p_client_id: clientId,
        p_observed_at: observedAt,
        p_force: body.force === true,
      },
    );
    if (error || !data?.[0]) {
      if (error?.message.includes("DEVICE_UNAUTHORIZED")) {
        return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
      }
      if (
        error?.message.includes("STALE_REFRESH_TOKEN") ||
        error?.message.includes("CONFLICTING_REFRESH_TOKEN")
      ) {
        return json(req, { error: "STALE_REFRESH_TOKEN" }, 409);
      }
      if (
        error?.message.includes("STALE_ACCESS_TOKEN") ||
        error?.message.includes("CONFLICTING_ACCESS_TOKEN")
      ) {
        return json(req, { error: "STALE_ACCESS_TOKEN" }, 409);
      }
      return json(req, { error: "TOKEN_SYNC_FAILED" }, 500);
    }
    return json(req, {
      status: data[0].sync_result,
      fingerprint: refreshFingerprint.slice(0, 12),
      access_fingerprint: accessFingerprint.slice(0, 12),
      access_expires_at: new Date(access.exp * 1000).toISOString(),
      expires_at: new Date(parsed.exp * 1000).toISOString(),
      synced_at: new Date().toISOString(),
    });
  }
  return json(req, { error: "GTOW_SESSION_BUNDLE_REQUIRED" }, 400);
}

async function status(req: Request, pepper: string): Promise<Response> {
  const authenticated = await authenticateDevice(req, pepper);
  if (!authenticated) return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
  return json(req, {
    paired: true,
    device_id: authenticated.device.id,
    device_name: authenticated.device.name,
    last_seen_at: authenticated.device.last_seen_at,
    last_sync_at: authenticated.device.last_sync_at,
  });
}

async function triggerIngest(req: Request, pepper: string): Promise<Response> {
  const authenticated = await authenticateDevice(req, pepper);
  if (!authenticated) return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
  const { data, error } = await adminClient().rpc(
    "enqueue_gtow_ingest_request",
    { p_credential_hash: authenticated.credentialHash },
  );
  if (error || !data?.[0]) {
    if (error?.message.includes("DEVICE_UNAUTHORIZED")) {
      return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
    }
    return json(req, { error: "INGEST_ENQUEUE_FAILED" }, 500);
  }
  return json(req, {
    request_id: data[0].request_id,
    status: data[0].request_status,
    reused: data[0].reused,
  });
}

async function ingestStatus(req: Request, pepper: string): Promise<Response> {
  const authenticated = await authenticateDevice(req, pepper);
  if (!authenticated) return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
  const id = new URL(req.url).searchParams.get("id") || "";
  if (!/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(id)) {
    return json(req, { error: "REQUEST_ID_INVALID" }, 400);
  }
  const { data, error } = await adminClient().rpc(
    "get_gtow_ingest_request",
    { p_credential_hash: authenticated.credentialHash, p_request_id: id },
  );
  if (error) {
    if (error.message.includes("DEVICE_UNAUTHORIZED")) {
      return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
    }
    return json(req, { error: "INGEST_STATUS_FAILED" }, 500);
  }
  if (!data?.[0]) return json(req, { error: "REQUEST_NOT_FOUND" }, 404);
  return json(req, {
    status: data[0].request_status,
    progress: data[0].progress,
    result: data[0].result,
    requested_at: data[0].requested_at,
    finished_at: data[0].finished_at,
  });
}

async function revoke(req: Request, pepper: string): Promise<Response> {
  const authenticated = await authenticateDevice(req, pepper);
  if (!authenticated) return json(req, { error: "DEVICE_UNAUTHORIZED" }, 401);
  const { error } = await adminClient()
    .from("gtow_sync_devices")
    .update({ revoked_at: new Date().toISOString() })
    .eq("id", authenticated.device.id);
  return error
    ? json(req, { error: "DEVICE_REVOKE_FAILED" }, 500)
    : json(req, { revoked: true });
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders(req) });
  }
  const pepper = Deno.env.get("GTOW_SYNC_PEPPER") || "";
  if (pepper.length < 32) {
    return json(req, { error: "SERVICE_NOT_CONFIGURED" }, 503);
  }
  const path = new URL(req.url).pathname.replace(/\/+$/, "");
  try {
    if (req.method === "GET" && path.endsWith("/health")) {
      const { error } = await adminClient().from("gtow_sync_devices")
        .select("id", { head: true, count: "exact" }).limit(1);
      return error
        ? json(req, { ok: false, error: "DEPENDENCY_NOT_READY" }, 503)
        : json(req, { ok: true });
    }
    if (req.method === "POST" && path.endsWith("/pair/exchange")) {
      return await pair(req, pepper);
    }
    if (req.method === "POST" && path.endsWith("/token")) {
      return await syncToken(req, pepper);
    }
    if (req.method === "GET" && path.endsWith("/ingest/status")) {
      return await ingestStatus(req, pepper);
    }
    if (req.method === "GET" && path.endsWith("/status")) {
      return await status(req, pepper);
    }
    if (req.method === "POST" && path.endsWith("/ingest")) {
      return await triggerIngest(req, pepper);
    }
    if (req.method === "DELETE" && path.endsWith("/device")) {
      return await revoke(req, pepper);
    }
    return json(req, { error: "NOT_FOUND" }, 404);
  } catch (error) {
    const code = error instanceof Error ? error.message : "REQUEST_FAILED";
    if (error instanceof ServiceError) {
      return json(req, { error: code }, error.status);
    }
    return json(
      req,
      { error: code === "BODY_INVALID" ? code : "REQUEST_INVALID" },
      code === "BODY_TOO_LARGE" ? 413 : 400,
    );
  }
});
