const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function jwt(payload) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(payload)}.sig`;
}

function loadContent({ refreshToken = "" } = {}) {
  const listeners = {};
  const messages = [];
  const document = {
    hidden: false,
    addEventListener() {},
  };
  const window = {
    addEventListener(type, callback) {
      listeners[type] = callback;
    },
  };
  const context = {
    atob(value) {
      return Buffer.from(value, "base64").toString("binary");
    },
    chrome: { runtime: {
      onMessage: { addListener(callback) { listeners.runtimeMessage = callback; } },
      sendMessage: (message) => {
        messages.push(message);
        return Promise.resolve();
      },
    } },
    crypto: require("node:crypto").webcrypto,
    document,
    localStorage: { getItem: () => refreshToken },
    setInterval() {},
    TextEncoder,
    Uint8Array,
    window,
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8"),
    context,
  );
  return { listeners, messages, window };
}

test("content script validates same-window session messages and sends bundle", async () => {
  const refreshToken = `eyJ${"r".repeat(40)}`;
  const accessToken = jwt({ exp: 1_900_000_000 });
  const { listeners, messages, window } = loadContent({ refreshToken });

  listeners.message({
    source: window,
    data: {
      source: "ai-poker-wizard-gtow-session",
      type: "GTOW_SESSION_OBSERVED",
      accessToken,
      clientId: "client-2",
      observedAt: "2026-07-31T00:00:00.000Z",
    },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  const sessionMessage = messages.find((message) => message.type === "SESSION_DETECTED");
  assert.equal(sessionMessage.refreshToken, refreshToken);
  assert.equal(sessionMessage.accessToken, accessToken);
  assert.equal(sessionMessage.accessExp, 1_900_000_000);
  assert.equal(sessionMessage.clientId, "client-2");
  assert.equal(sessionMessage.observedAt, "2026-07-31T00:00:00.000Z");
});

test("content script never auto-syncs a refresh-only token", async () => {
  const { messages } = loadContent({ refreshToken: `eyJ${"r".repeat(40)}` });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(messages.some((message) => message.type === "TOKEN_DETECTED"), false);
});

test("content script answers latest session bundle from memory", async () => {
  const refreshToken = `eyJ${"r".repeat(40)}`;
  const accessToken = jwt({ exp: 1_900_000_000 });
  const { listeners, window } = loadContent({ refreshToken });

  listeners.message({
    source: window,
    data: {
      source: "ai-poker-wizard-gtow-session",
      type: "GTOW_SESSION_OBSERVED",
      accessToken,
      clientId: "client-2",
      observedAt: "2026-07-31T00:00:00.000Z",
    },
  });

  let response;
  const handled = listeners.runtimeMessage({ type: "GET_SESSION_BUNDLE" }, {}, (value) => {
    response = value;
  });

  assert.equal(handled, false);
  assert.deepEqual(JSON.parse(JSON.stringify(response.session)), {
    refreshToken,
    accessToken,
    accessExp: 1_900_000_000,
    clientId: "client-2",
    observedAt: "2026-07-31T00:00:00.000Z",
  });
});

test("content script rejects invalid source and invalid access JWT", async () => {
  const { listeners, messages, window } = loadContent({ refreshToken: `eyJ${"r".repeat(40)}` });

  listeners.message({
    source: {},
    data: {
      source: "ai-poker-wizard-gtow-session",
      type: "GTOW_SESSION_OBSERVED",
      accessToken: jwt({ exp: 1_900_000_000 }),
    },
  });
  listeners.message({
    source: window,
    data: {
      source: "ai-poker-wizard-gtow-session",
      type: "GTOW_SESSION_OBSERVED",
      accessToken: "not-a-jwt",
    },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(messages.some((message) => message.type === "SESSION_DETECTED"), false);
});
