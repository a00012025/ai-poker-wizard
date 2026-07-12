importScripts("config.js");

const CONFIG = globalThis.GTOW_SYNC_CONFIG;
const STORAGE_KEYS = [
  "deviceId",
  "deviceSecret",
  "telegramLabel",
  "deviceName",
  "lastFingerprint",
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

async function syncToken(refreshToken, suppliedFingerprint, force = false) {
  if (!refreshToken?.startsWith("eyJ")) throw new Error("GTOW_TOKEN_NOT_FOUND");
  const state = await storageGet();
  if (!state.deviceSecret) {
    await setStatus({ tokenDetected: true, lastStatus: "waiting_pair" });
    return { status: "waiting_pair" };
  }
  const fingerprint = suppliedFingerprint || await sha256(refreshToken);
  if (!force && fingerprint === state.lastFingerprint) {
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
        refresh_token: refreshToken,
        observed_at: new Date().toISOString(),
      }),
    });
    await setStatus({
      lastFingerprint: fingerprint,
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
}

async function tokenFromActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url?.startsWith("https://app.gtowizard.com/")) {
    throw new Error("OPEN_GTOW_FIRST");
  }
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => localStorage.getItem("user_refresh") || "",
  });
  if (!result?.startsWith("eyJ")) throw new Error("GTOW_TOKEN_NOT_FOUND");
  return result;
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
    const token = await tokenFromActiveTab();
    await syncToken(token, null, true);
  } catch (error) {
    if (!['OPEN_GTOW_FIRST', 'GTOW_TOKEN_NOT_FOUND'].includes(error.message)) throw error;
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
        return syncToken(message.refreshToken, message.fingerprint);
      case "PAIR_DEVICE":
        return pairDevice(message.code, message.deviceName);
      case "SYNC_ACTIVE_TAB": {
        const token = await tokenFromActiveTab();
        return syncToken(token, null, true);
      }
      case "UNPAIR_DEVICE":
        return unpairDevice();
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
