let lastObservedFingerprint = "";

async function fingerprint(value) {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function detectToken() {
  const token = localStorage.getItem("user_refresh") || "";
  if (!token.startsWith("eyJ")) return;
  const currentFingerprint = await fingerprint(token);
  if (currentFingerprint === lastObservedFingerprint) return;
  lastObservedFingerprint = currentFingerprint;
  chrome.runtime.sendMessage({
    type: "TOKEN_DETECTED",
    refreshToken: token,
    fingerprint: currentFingerprint,
  }).catch(() => {
    lastObservedFingerprint = "";
  });
}

detectToken();
window.addEventListener("pageshow", detectToken);
window.addEventListener("storage", (event) => {
  if (event.key === "user_refresh") detectToken();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) detectToken();
});
setInterval(() => {
  if (!document.hidden) detectToken();
}, 60_000);

// ── One-click hand ingest (floating button + progress toast) ────────────────

const INGEST_POLL_MS = 2000;
const INGEST_POLL_MAX_MS = 30 * 60 * 1000;

function sendMessage(payload) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(payload, (response) => {
      if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
      response?.ok
        ? resolve(response.result)
        : reject(new Error(response?.error || "REQUEST_FAILED"));
    });
  });
}

function ingestErrorText(code) {
  return {
    DEVICE_UNAUTHORIZED: "尚未配對：請點 extension 圖示完成 Telegram 配對。",
    GTOW_TOKEN_NOT_FOUND: "找不到 GTOW token，請先登入 GTO Wizard。",
    REQUEST_NOT_FOUND: "找不到這筆同步請求，請重試。",
    INGEST_ENQUEUE_FAILED: "無法排入同步佇列，請稍後重試。",
    "Failed to fetch": "無法連線同步服務，請檢查網路。",
  }[code] || code;
}

function ingestUi() {
  let root = document.getElementById("apw-ingest-root");
  if (root) return root;
  root = document.createElement("div");
  root.id = "apw-ingest-root";
  root.style.cssText =
    "position:fixed;right:18px;bottom:18px;z-index:2147483647;display:flex;" +
    "flex-direction:column;align-items:flex-end;gap:8px;font-family:Inter,system-ui,sans-serif;";
  const toast = document.createElement("div");
  toast.id = "apw-ingest-toast";
  toast.style.cssText =
    "display:none;max-width:340px;padding:10px 14px;border-radius:12px;" +
    "background:#151b24;color:#f2f4f7;border:1px solid #293140;font-size:13px;" +
    "line-height:1.5;white-space:pre-wrap;box-shadow:0 6px 24px rgba(0,0,0,.45);";
  const button = document.createElement("button");
  button.id = "apw-ingest-button";
  button.textContent = "♠ 同步手牌到 DB";
  button.style.cssText =
    "padding:10px 16px;border:0;border-radius:999px;background:#95f4aa;" +
    "color:#102516;font-size:13px;font-weight:700;cursor:pointer;" +
    "box-shadow:0 6px 24px rgba(0,0,0,.45);";
  button.addEventListener("click", runIngest);
  root.append(toast, button);
  document.documentElement.appendChild(root);
  return root;
}

function showToast(text, { error = false, autoHideMs = 0 } = {}) {
  const toast = ingestUi().querySelector("#apw-ingest-toast");
  toast.style.display = "block";
  toast.style.borderColor = error ? "#ff6b6b" : "#293140";
  toast.textContent = text;
  clearTimeout(showToast._timer);
  if (autoHideMs) {
    showToast._timer = setTimeout(() => { toast.style.display = "none"; }, autoHideMs);
  }
}

async function runIngest() {
  const button = ingestUi().querySelector("#apw-ingest-button");
  if (button.disabled) return;
  button.disabled = true;
  button.style.opacity = "0.55";
  try {
    showToast("排隊中…");
    const trigger = await sendMessage({
      type: "INGEST_TRIGGER",
      refreshToken: localStorage.getItem("user_refresh") || "",
    });
    const startedAt = Date.now();
    while (Date.now() - startedAt < INGEST_POLL_MAX_MS) {
      await new Promise((resolve) => setTimeout(resolve, INGEST_POLL_MS));
      const state = await sendMessage({
        type: "INGEST_STATUS",
        requestId: trigger.request_id,
      });
      if (state.status === "done") {
        showToast(`✅ ${state.result || "同步完成"}`, { autoHideMs: 20_000 });
        return;
      }
      if (state.status === "error") {
        showToast(`❌ ${state.result || "同步失敗"}`, { error: true, autoHideMs: 20_000 });
        return;
      }
      showToast(state.status === "running"
        ? `⏳ ${state.progress || "攝取中…"}`
        : "⏳ 排隊中…（bot 會在幾秒內接手）");
    }
    showToast("⌛ 等待逾時 — 完成後 Telegram 仍會通知你。", { error: true, autoHideMs: 20_000 });
  } catch (error) {
    showToast(`❌ ${ingestErrorText(error.message)}`, { error: true, autoHideMs: 20_000 });
  } finally {
    button.disabled = false;
    button.style.opacity = "1";
  }
}

ingestUi();
