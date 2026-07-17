const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadBackground({ tabUrl, token }) {
  let listener;
  const context = {
    chrome: {
      runtime: {
        onMessage: { addListener(callback) { listener = callback; } },
        sendMessage() { return Promise.resolve(); },
      },
      scripting: {
        async executeScript() { return [{ result: token }]; },
      },
      storage: {
        local: {
          async get() { return {}; },
          async remove() {},
          async set() {},
        },
      },
      tabs: {
        async create() {},
        async query() { return [{ id: 17, url: tabUrl }]; },
      },
    },
    crypto: require("node:crypto").webcrypto,
    fetch,
    globalThis: null,
    importScripts() {
      context.GTOW_SYNC_CONFIG = {
        apiBase: "https://example.invalid",
        gtowUrl: "https://app.gtowizard.com/",
        telegramBotUrl: "https://t.me/example",
      };
    },
    TextEncoder,
    Uint8Array,
  };
  context.globalThis = context;
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8"),
    context,
  );
  return listener;
}

function dispatch(listener, type) {
  return new Promise((resolve) => {
    listener({ type }, {}, resolve);
  });
}

test("manual fallback returns a complete /settoken command", async () => {
  const token = `eyJ${"x".repeat(40)}`;
  const listener = loadBackground({
    tabUrl: "https://app.gtowizard.com/solutions",
    token,
  });

  const response = await dispatch(listener, "GET_SETTOKEN_COMMAND");

  assert.equal(response.ok, true);
  assert.equal(response.result.command, `/settoken ${token}`);
});

test("manual fallback requires the active GTO Wizard tab", async () => {
  const listener = loadBackground({
    tabUrl: "https://example.com/",
    token: `eyJ${"x".repeat(40)}`,
  });

  const response = await dispatch(listener, "GET_SETTOKEN_COMMAND");

  assert.equal(response.ok, false);
  assert.equal(response.error, "OPEN_GTOW_FIRST");
});
