const $ = (selector) => document.querySelector(selector);

function send(type, payload = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, ...payload }, (response) => {
      if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
      response?.ok ? resolve(response.result) : reject(new Error(response?.error || "REQUEST_FAILED"));
    });
  });
}

function formatTime(value) {
  if (!value) return "尚未同步";
  return new Intl.DateTimeFormat("zh-TW", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function errorText(code) {
  return {
    OPEN_GTOW_FIRST: "請先在目前分頁開啟 GTO Wizard。",
    GTOW_TOKEN_NOT_FOUND: "找不到 GTOW token，請先登入 GTO Wizard。",
    PAIRING_INVALID_OR_EXPIRED: "配對碼錯誤或已過期，請回 Telegram 重新輸入 /pair。",
    DEVICE_UNAUTHORIZED: "裝置配對已失效，請重新配對。",
    REFRESH_TOKEN_REJECTED: "GTOW 拒絕這組 token，請重新登入後再同步。",
    STALE_REFRESH_TOKEN: "這組 token 比伺服器版本舊；請重新整理 GTOW 後再同步。",
    SYNC_RATE_LIMITED: "同步太頻繁，請幾秒後再試。",
    INGEST_ENQUEUE_FAILED: "無法排入同步佇列，請稍後重試。",
    REQUEST_NOT_FOUND: "找不到這筆同步請求，請重試。",
    "Failed to fetch": "無法連線同步服務，請檢查網路。",
  }[code] || code;
}

function setBusy(button, busy) { button.disabled = busy; }
function message(text = "", isError = false) {
  $("#message").textContent = text;
  $("#message").classList.toggle("error", isError);
}

async function render() {
  const local = await send("GET_STATE");
  let remoteError = null;
  if (local.deviceSecret) {
    try { await send("REMOTE_STATUS"); }
    catch (error) { remoteError = error.message; }
  }
  const state = await send("GET_STATE");
  const paired = Boolean(state.deviceSecret);
  $("#pair-panel").classList.toggle("hidden", paired);
  $("#paired-panel").classList.toggle("hidden", !paired);
  $("#status-dot").className = `dot ${state.lastStatus === "error" ? "error" : paired ? "ok" : ""}`;
  $("#status-title").textContent = paired ? "已連結自動同步" : "尚未配對 Telegram";
  $("#status-detail").textContent = paired
    ? remoteError ? `暫時無法確認遠端狀態：${errorText(remoteError)}`
      : state.lastStatus === "error" ? errorText(state.lastError) : "登入 GTOW 後會自動同步 token。"
    : state.tokenDetected ? "已偵測 GTOW 登入，完成配對即可同步。" : "取得 /pair 配對碼後只需設定一次。";
  $("#telegram-label").textContent = state.telegramLabel || "Telegram user";
  $("#device-name").textContent = state.deviceName || "Chrome";
  $("#last-sync").textContent = formatTime(state.lastSyncAt);
}

$("#pair-code").addEventListener("input", (event) => {
  const raw = event.target.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 12);
  event.target.value = raw.match(/.{1,4}/g)?.join("-") || "";
});

$("#pair-button").addEventListener("click", async () => {
  const button = $("#pair-button");
  const code = $("#pair-code").value;
  if (code.replaceAll("-", "").length !== 12) return message("請輸入完整的 12 碼配對碼。", true);
  setBusy(button, true); message("配對中…");
  try {
    await send("PAIR_DEVICE", { code, deviceName: `Chrome · ${navigator.platform || "Desktop"}` });
    message("配對成功；若 GTOW 已登入，token 也已同步。");
    await render();
  } catch (error) { message(errorText(error.message), true); }
  finally { setBusy(button, false); }
});

$("#ingest-button").addEventListener("click", async () => {
  const button = $("#ingest-button");
  setBusy(button, true);
  message("排隊中…");
  try {
    const trigger = await send("INGEST_TRIGGER");
    const startedAt = Date.now();
    while (Date.now() - startedAt < 30 * 60 * 1000) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const state = await send("INGEST_STATUS", { requestId: trigger.request_id });
      if (state.status === "done") return message(`✅ ${state.result || "同步完成"}`);
      if (state.status === "error") return message(`❌ ${state.result || "同步失敗"}`, true);
      message(state.status === "running" ? `⏳ ${state.progress || "攝取中…"}` : "⏳ 排隊中…");
    }
    message("⌛ 等待逾時 — 完成後 Telegram 仍會通知你。", true);
  } catch (error) { message(errorText(error.message), true); }
  finally { setBusy(button, false); await render(); }
});

$("#sync-button").addEventListener("click", async () => {
  const button = $("#sync-button"); setBusy(button, true); message("同步中…");
  try { await send("SYNC_ACTIVE_TAB"); message("同步完成。"); await render(); }
  catch (error) { message(errorText(error.message), true); }
  finally { setBusy(button, false); }
});

$("#copy-settoken-button").addEventListener("click", async () => {
  const button = $("#copy-settoken-button");
  setBusy(button, true);
  message("正在讀取目前的 GTOW token…");
  try {
    const { command } = await send("GET_SETTOKEN_COMMAND");
    await navigator.clipboard.writeText(command);
    message("✅ 已複製 /settoken 指令，請直接貼到 Telegram bot。");
  } catch (error) { message(errorText(error.message), true); }
  finally { setBusy(button, false); }
});

$("#unpair-button").addEventListener("click", async () => {
  if (!confirm("確定解除這台 Chrome 的同步配對？")) return;
  try {
    await send("UNPAIR_DEVICE");
    message("已解除這台裝置。");
    await render();
  } catch (error) {
    message(`解除失敗，配對資料仍保留，請稍後重試：${errorText(error.message)}`, true);
  }
});
$("#gtow-button").addEventListener("click", () => send("OPEN_GTOW"));
$("#telegram-button").addEventListener("click", () => send("OPEN_TELEGRAM"));
chrome.runtime.onMessage.addListener((event) => { if (event.type === "STATE_CHANGED") render(); });
render().catch((error) => message(errorText(error.message), true));
