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
