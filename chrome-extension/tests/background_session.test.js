const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadBackground({ storage = {}, fetchImpl, sessionBundle = null, tabUrl = "" } = {}) {
  let listener;
  const stored = { ...storage };
  const counts = { executeScript: 0, sendMessage: 0 };
  const context = {
    chrome: {
      runtime: {
        onMessage: { addListener(callback) { listener = callback; } },
        sendMessage() { return Promise.resolve(); },
      },
      scripting: {
        async executeScript() {
          counts.executeScript += 1;
          return [{ result: `eyJ${"legacy".repeat(8)}` }];
        },
      },
      storage: {
        local: {
          async get() { return { ...stored }; },
          async remove(keys) {
            (Array.isArray(keys) ? keys : [keys]).forEach((key) => delete stored[key]);
          },
          async set(patch) { Object.assign(stored, patch); },
        },
      },
      tabs: {
        async create() {},
        async query() {
          return tabUrl ? [{ id: 17, url: tabUrl }] : [];
        },
        async sendMessage(_tabId, message) {
          counts.sendMessage += 1;
          if (message.type !== "GET_SESSION_BUNDLE") throw new Error("UNKNOWN_MESSAGE");
          return { session: sessionBundle };
        },
      },
    },
    crypto: require("node:crypto").webcrypto,
    fetch: fetchImpl,
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
  return { counts, listener, stored };
}

function dispatch(listener, message) {
  return new Promise((resolve) => {
    listener(message, {}, resolve);
  });
}

test("background syncs session bundle fields to /token", async () => {
  const calls = [];
  const { listener, stored } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options, body: JSON.parse(options.body) });
      return new Response(JSON.stringify({ status: "updated", synced_at: "2026-07-31T01:00:00.000Z" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });

  const response = await dispatch(listener, {
    type: "SESSION_DETECTED",
    refreshToken: `eyJ${"r".repeat(40)}`,
    accessToken: `eyJ${"a".repeat(40)}`,
    accessExp: 1_900_000_000,
    clientId: "client-3",
    observedAt: "2026-07-31T00:00:00.000Z",
  });

  assert.equal(response.ok, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://example.invalid/token");
  assert.equal(calls[0].options.headers.Authorization, "Device device-secret");
  assert.deepEqual(calls[0].body, {
    refresh_token: `eyJ${"r".repeat(40)}`,
    access_token: `eyJ${"a".repeat(40)}`,
    access_exp: 1_900_000_000,
    gwclientid: "client-3",
    observed_at: "2026-07-31T00:00:00.000Z",
    force: false,
  });
  assert.equal(stored.lastStatus, "synced");
  assert.ok(stored.lastSessionFingerprint);
});

test("background dedupes sessions by access/client/exp fingerprint", async () => {
  const calls = [];
  const { listener } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options });
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify({ paired: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(JSON.stringify({ status: "updated" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });
  const message = {
    type: "SESSION_DETECTED",
    refreshToken: `eyJ${"r".repeat(40)}`,
    accessToken: `eyJ${"a".repeat(40)}`,
    accessExp: 1_900_000_000,
    clientId: "client-3",
    observedAt: "2026-07-31T00:00:00.000Z",
  };

  const first = await dispatch(listener, message);
  const second = await dispatch(listener, message);

  assert.equal(first.ok, true);
  assert.equal(second.ok, true);
  assert.equal(second.result.status, "unchanged_local");
  assert.equal(calls.filter((call) => call.url.endsWith("/token")).length, 1);
  assert.equal(calls.filter((call) => call.url.endsWith("/status")).length, 1);
});

test("manual sync uses active tab session bundle instead of legacy token validation", async () => {
  const calls = [];
  const sessionBundle = {
    refreshToken: `eyJ${"r".repeat(40)}`,
    accessToken: `eyJ${"a".repeat(40)}`,
    accessExp: 1_900_000_000,
    clientId: "client-manual",
    observedAt: "2026-07-31T02:00:00.000Z",
  };
  const { counts, listener } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    sessionBundle,
    tabUrl: "https://app.gtowizard.com/solutions",
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options, body: options.body ? JSON.parse(options.body) : null });
      return new Response(JSON.stringify({ status: "updated" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });

  const response = await dispatch(listener, { type: "SYNC_ACTIVE_TAB" });

  assert.equal(response.ok, true);
  assert.equal(counts.sendMessage, 1);
  assert.equal(counts.executeScript, 0);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://example.invalid/token");
  assert.equal(calls[0].body.access_token, sessionBundle.accessToken);
  assert.equal(calls[0].body.access_exp, sessionBundle.accessExp);
  assert.equal(calls[0].body.gwclientid, sessionBundle.clientId);
  assert.equal(calls[0].body.force, true);
});

test("ingest trigger syncs active tab session bundle before ingest without legacy token validation", async () => {
  const calls = [];
  const sessionBundle = {
    refreshToken: `eyJ${"r".repeat(40)}`,
    accessToken: `eyJ${"a".repeat(40)}`,
    accessExp: 1_900_000_000,
    clientId: "client-ingest",
    observedAt: "2026-07-31T02:05:00.000Z",
  };
  const { counts, listener } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    sessionBundle,
    tabUrl: "https://app.gtowizard.com/solutions",
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options, body: options.body ? JSON.parse(options.body) : null });
      return new Response(JSON.stringify(url.endsWith("/ingest") ? { request_id: "req-1" } : { status: "updated" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });

  const response = await dispatch(listener, { type: "INGEST_TRIGGER" });

  assert.equal(response.ok, true);
  assert.equal(counts.sendMessage, 1);
  assert.equal(counts.executeScript, 0);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, "https://example.invalid/token");
  assert.equal(calls[0].body.access_token, sessionBundle.accessToken);
  assert.equal(calls[0].body.gwclientid, sessionBundle.clientId);
  assert.equal(calls[0].body.force, true);
  assert.equal(calls[1].url, "https://example.invalid/ingest");
  assert.deepEqual(calls[1].body, {});
});

test("manual sync never falls back to refresh-only token validation", async () => {
  const calls = [];
  const { counts, listener } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    sessionBundle: null,
    tabUrl: "https://app.gtowizard.com/solutions",
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options });
      return new Response("{}", { status: 200 });
    },
  });

  const response = await dispatch(listener, { type: "SYNC_ACTIVE_TAB" });

  assert.equal(response.ok, false);
  assert.equal(response.error, "GTOW_SESSION_BUNDLE_NOT_READY");
  assert.equal(counts.executeScript, 0);
  assert.equal(calls.length, 0);
});

test("ingest refuses to enqueue without an observed session bundle", async () => {
  const calls = [];
  const { counts, listener } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    sessionBundle: null,
    tabUrl: "https://app.gtowizard.com/solutions",
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options });
      return new Response("{}", { status: 200 });
    },
  });

  const response = await dispatch(listener, { type: "INGEST_TRIGGER" });

  assert.equal(response.ok, false);
  assert.equal(response.error, "GTOW_SESSION_BUNDLE_NOT_READY");
  assert.equal(counts.executeScript, 0);
  assert.equal(calls.length, 0);
});

test("concurrent identical observations share one session sync", async () => {
  const calls = [];
  let release;
  let markStarted;
  const gate = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { markStarted = resolve; });
  const { listener, stored } = loadBackground({
    storage: { deviceSecret: "device-secret" },
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options });
      markStarted();
      await gate;
      return new Response(JSON.stringify({ status: "updated" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });
  const message = {
    type: "SESSION_DETECTED",
    refreshToken: `eyJ${"r".repeat(40)}`,
    accessToken: `eyJ${"a".repeat(40)}`,
    accessExp: 1_900_000_000,
    clientId: "client-race",
    observedAt: "2026-07-31T02:05:00.000Z",
  };

  const first = dispatch(listener, message);
  const second = dispatch(listener, message);
  await started;
  assert.equal(calls.length, 1);
  release();
  const [firstResult, secondResult] = await Promise.all([first, second]);

  assert.equal(firstResult.ok, true);
  assert.equal(secondResult.ok, true);
  assert.equal(calls.length, 1);
  assert.equal(stored.lastStatus, "synced");
});
