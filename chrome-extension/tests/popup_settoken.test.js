const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function element() {
  return {
    classList: { toggle() {} },
    addEventListener(type, callback) { this.listeners[type] = callback; },
    listeners: {},
    textContent: "",
    value: "",
  };
}

test("copy fallback writes the complete /settoken command to the clipboard", async () => {
  const elements = new Map();
  const getElement = (selector) => {
    if (!elements.has(selector)) elements.set(selector, element());
    return elements.get(selector);
  };
  let copied = "";
  const token = `eyJ${"x".repeat(40)}`;
  const context = {
    chrome: {
      runtime: {
        lastError: null,
        onMessage: { addListener() {} },
        sendMessage(message, callback) {
          if (message.type === "GET_STATE") {
            callback({ ok: true, result: {} });
          } else if (message.type === "GET_SETTOKEN_COMMAND") {
            callback({ ok: true, result: { command: `/settoken ${token}` } });
          } else {
            callback({ ok: true, result: {} });
          }
        },
      },
    },
    confirm() { return true; },
    document: { querySelector: getElement },
    Intl,
    navigator: {
      clipboard: { async writeText(value) { copied = value; } },
      platform: "Test",
    },
    setTimeout,
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "..", "popup.js"), "utf8"),
    context,
  );

  await getElement("#copy-settoken-button").listeners.click();

  assert.equal(copied, `/settoken ${token}`);
  assert.match(getElement("#message").textContent, /已複製 \/settoken 指令/);
});
