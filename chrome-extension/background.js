importScripts("config.js");

const CONFIG = globalThis.GTOW_SYNC_CONFIG;
const STORAGE_KEYS = [
  "deviceId",
  "deviceSecret",
  "telegramLabel",
  "deviceName",
  "lastFingerprint",
  "lastSessionFingerprint",
  "lastSyncAt",
  "lastStatus",
  "lastError",
  "tokenDetected",
];

async function storageGet() {
  return chrome.storage.local.get(STORAGE_KEYS);
}

async function setStatus(patch) {
  await chrome.storage.local.set(patch);
  chrome.runtime.sendMessage({ type: "STATE_CHANGED" }).catch(() => {});
}

async function api(path, options = {}) {
  const response = await fetch(`${CONFIG.apiBase}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `HTTP_${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

async function sha256(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

const sessionSyncsInFlight = new Map();

async function syncSession(session, force = false) {
  if (!session?.refreshToken?.startsWith("eyJ")) throw new Error("GTOW_TOKEN_NOT_FOUND");
  if (!session.accessToken?.startsWith("eyJ")) throw new Error("GTOW_ACCESS_TOKEN_INVALID");
  if (!Number.isFinite(session.accessExp)) throw new Error("GTOW_ACCESS_EXP_INVALID");
  const inFlightKey = [
    session.refreshToken,
    session.accessToken,
    session.accessExp,
    session.clientId || "",
  ].join("\u0000");
  const existing = sessionSyncsInFlight.get(inFlightKey);
  if (existing) return existing;
  const work = (async () => {
    const refreshFingerprint = await sha256(session.refreshToken);
    const sessionFingerprint = await sha256([
      refreshFingerprint,
      session.accessToken,
      session.accessExp,
      session.clientId || "",
    ].join(":"));
    const state = await storageGet();
    if (!state.deviceSecret) {
      await setStatus({ tokenDetected: true, lastStatus: "waiting_pair" });
      return { status: "waiting_pair" };
    }
    if (!force && sessionFingerprint === state.lastSessionFingerprint) {
      const remote = await remoteStatus();
      if (!remote.paired) {
        await setStatus({ lastStatus: "waiting_pair", lastError: "DEVICE_UNAUTHORIZED" });
        return { status: "waiting_pair" };
      }
      await setStatus({ tokenDetected: true, lastStatus: "up_to_date", lastError: null });
      return { status: "unchanged_local" };
    }
    await setStatus({ tokenDetected: true, lastStatus: "syncing", lastError: null });
    try {
      const result = await api("/token", {
        method: "POST",
        headers: { Authorization: `Device ${state.deviceSecret}` },
        body: JSON.stringify({
          refresh_token: session.refreshToken,
          access_token: session.accessToken,
          access_exp: session.accessExp,
          gwclientid: session.clientId || "",
          observed_at: session.observedAt || new Date().toISOString(),
          force,
        }),
      });
      await setStatus({
        lastFingerprint: refreshFingerprint,
        lastSessionFingerprint: sessionFingerprint,
        lastSyncAt: result.synced_at || new Date().toISOString(),
        lastStatus: result.status === "updated" ? "synced" : "up_to_date",
        lastError: null,
      });
      return result;
    } catch (error) {
      if (error.status === 401) {
        await chrome.storage.local.remove(["deviceId", "deviceSecret", "telegramLabel"]);
      }
      await setStatus({ lastStatus: "error", lastError: error.message });
      throw error;
    }
  })();
  sessionSyncsInFlight.set(inFlightKey, work);
  try {
    return await work;
  } finally {
    if (sessionSyncsInFlight.get(inFlightKey) === work) {
      sessionSyncsInFlight.delete(inFlightKey);
    }
  }
}

async function activeGtowTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url?.startsWith("https://app.gtowizard.com/")) {
    throw new Error("OPEN_GTOW_FIRST");
  }
  return tab;
}

function validSessionBundle(session) {
  return Boolean(
    session?.refreshToken?.startsWith("eyJ")
    && session.accessToken?.startsWith("eyJ")
    && Number.isFinite(session.accessExp),
  );
}

async function sessionFromActiveTab() {
  const tab = await activeGtowTab();
  if (!chrome.tabs.sendMessage) return null;
  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_SESSION_BUNDLE" });
    return validSessionBundle(response?.session) ? response.session : null;
  } catch {
    return null;
  }
}

async function tokenFromActiveTab() {
  const tab = await activeGtowTab();
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => localStorage.getItem("user_refresh") || "",
  });
  if (!result?.startsWith("eyJ")) throw new Error("GTOW_TOKEN_NOT_FOUND");
  return result;
}

async function settokenCommandFromActiveTab() {
  const token = await tokenFromActiveTab();
  return `/settoken ${token}`;
}

async function pairDevice(code, deviceName) {
  const result = await api("/pair/exchange", {
    method: "POST",
    body: JSON.stringify({ code, device_name: deviceName }),
  });
  await chrome.storage.local.set({
    deviceId: result.device_id,
    deviceSecret: result.device_secret,
    telegramLabel: result.telegram_label,
    deviceName,
    lastStatus: "paired",
    lastError: null,
  });
  try {
    const session = await sessionFromActiveTab();
    if (session) {
      await syncSession(session, true);
    }
  } catch (error) {
    if (error.message !== "OPEN_GTOW_FIRST") throw error;
  }
  return result;
}

async function unpairDevice() {
  const state = await storageGet();
  if (state.deviceSecret) {
    try {
      await api("/device", {
        method: "DELETE",
        headers: { Authorization: `Device ${state.deviceSecret}` },
      });
    } catch (error) {
      if (error.status !== 401) throw error;
    }
  }
  await chrome.storage.local.remove(STORAGE_KEYS);
  return { revoked: true };
}

async function deviceHeaders() {
  const state = await storageGet();
  if (!state.deviceSecret) throw new Error("DEVICE_UNAUTHORIZED");
  return { Authorization: `Device ${state.deviceSecret}` };
}

async function triggerIngest() {
  const headers = await deviceHeaders();
  const session = await sessionFromActiveTab();
  if (!session) throw new Error("GTOW_SESSION_BUNDLE_NOT_READY");
  await syncSession(session, true);
  return api("/ingest", { method: "POST", headers, body: "{}" });
}

async function ingestStatus(requestId) {
  if (typeof requestId !== "string" || !requestId) {
    throw new Error("REQUEST_ID_INVALID");
  }
  const headers = await deviceHeaders();
  return api(`/ingest/status?id=${encodeURIComponent(requestId)}`, { headers });
}

async function remoteStatus() {
  const state = await storageGet();
  if (!state.deviceSecret) return { paired: false };
  try {
    return await api("/status", {
      headers: { Authorization: `Device ${state.deviceSecret}` },
    });
  } catch (error) {
    if (error.status === 401) {
      await chrome.storage.local.remove(STORAGE_KEYS);
      return { paired: false, error: error.message };
    }
    throw error;
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    switch (message.type) {
      case "TOKEN_DETECTED":
        throw new Error("GTOW_SESSION_BUNDLE_NOT_READY");
      case "SESSION_DETECTED":
        return syncSession(message);
      case "PAIR_DEVICE":
        return pairDevice(message.code, message.deviceName);
      case "SYNC_ACTIVE_TAB": {
        const session = await sessionFromActiveTab();
        if (!session) throw new Error("GTOW_SESSION_BUNDLE_NOT_READY");
        return syncSession(session, true);
      }
      case "GET_SETTOKEN_COMMAND":
        return { command: await settokenCommandFromActiveTab() };
      case "UNPAIR_DEVICE":
        return unpairDevice();
      case "INGEST_TRIGGER":
        return triggerIngest();
      case "INGEST_STATUS":
        return ingestStatus(message.requestId);
      case "GET_STATE":
        return { ...(await storageGet()), config: CONFIG };
      case "REMOTE_STATUS":
        return remoteStatus();
      case "OPEN_GTOW":
        await chrome.tabs.create({ url: CONFIG.gtowUrl });
        return { opened: true };
      case "OPEN_TELEGRAM":
        await chrome.tabs.create({ url: CONFIG.telegramBotUrl });
        return { opened: true };
      case "STATE_CHANGED":
        return null;
      default:
        throw new Error("UNKNOWN_MESSAGE");
    }
  })().then(
    (result) => sendResponse({ ok: true, result }),
    (error) => sendResponse({ ok: false, error: error.message || "REQUEST_FAILED" }),
  );
  return true;
});
