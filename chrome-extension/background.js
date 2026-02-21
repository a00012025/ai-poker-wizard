chrome.action.onClicked.addListener(async (tab) => {
  // Must be on GTO Wizard
  if (!tab.url || !tab.url.includes("gtowizard.com")) {
    chrome.tabs.create({ url: "https://app.gtowizard.com" });
    return;
  }

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const token = localStorage.getItem("user_refresh");
      if (!token || !token.startsWith("eyJ")) {
        alert("找不到 GTO Wizard token，請先登入。");
        return;
      }
      const cmd = "/settoken " + token;
      navigator.clipboard.writeText(cmd).then(() => {
        alert("已複製！請回 Telegram 貼上。");
      }).catch(() => {
        // Fallback for clipboard API failure
        const ta = document.createElement("textarea");
        ta.value = cmd;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        alert("已複製！請回 Telegram 貼上。");
      });
    },
  });
});
