let latestSessionBundle = null;
const SESSION_SOURCE = "ai-poker-wizard-gtow-session";
const SESSION_MESSAGE_TYPE = "GTOW_SESSION_OBSERVED";

function decodeBase64UrlJson(value) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
  return JSON.parse(atob(padded));
}

function jwtExp(token) {
  if (typeof token !== "string") return null;
  const parts = token.split(".");
  if (parts.length !== 3 || !parts.every(Boolean)) return null;
  try {
    const payload = decodeBase64UrlJson(parts[1]);
    return Number.isFinite(payload.exp) ? payload.exp : null;
  } catch {
    return null;
  }
}

function detectToken() {
  const refreshToken = localStorage.getItem("user_refresh") || "";
  if (!refreshToken.startsWith("eyJ")) latestSessionBundle = null;
}

function detectSession(event) {
  if (event.source !== window) return;
  const data = event.data;
  if (data?.source !== SESSION_SOURCE || data.type !== SESSION_MESSAGE_TYPE) return;
  const accessExp = jwtExp(data.accessToken);
  if (!accessExp) return;
  const refreshToken = localStorage.getItem("user_refresh") || "";
  if (!refreshToken.startsWith("eyJ")) return;
  latestSessionBundle = {
    refreshToken,
    accessToken: data.accessToken,
    accessExp,
    clientId: typeof data.clientId === "string" ? data.clientId : "",
    observedAt: typeof data.observedAt === "string" ? data.observedAt : new Date().toISOString(),
  };
  chrome.runtime.sendMessage({
    type: "SESSION_DETECTED",
    ...latestSessionBundle,
  }).catch(() => {});
}

detectToken();
window.addEventListener("message", detectSession);
window.addEventListener("pageshow", detectToken);
window.addEventListener("storage", (event) => {
  if (event.key === "user_refresh") {
    latestSessionBundle = null;
    detectToken();
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) detectToken();
});
setInterval(() => {
  if (!document.hidden) detectToken();
}, 60_000);

chrome.runtime.onMessage?.addListener((message, _sender, sendResponse) => {
  if (message.type !== "GET_SESSION_BUNDLE") return false;
  sendResponse({ session: latestSessionBundle });
  return false;
});
