const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadPageSession({ fetchImpl } = {}) {
  const posted = [];
  class FakeXhr {
    constructor() {
      this.listeners = {};
      this.responseText = "{}";
    }
    open(_method, url) {
      this.url = url;
    }
    setRequestHeader(name, value) {
      this.headers = { ...(this.headers || {}), [name]: value };
    }
    addEventListener(type, callback) {
      this.listeners[type] = callback;
    }
    send() {
      this.listeners.load?.();
    }
  }
  const context = {
    fetch: fetchImpl || (async () => new Response("{}")),
    Headers,
    Response,
    URL,
    XMLHttpRequest: FakeXhr,
    window: null,
  };
  context.window = {
    fetch: context.fetch,
    location: { origin: "https://app.gtowizard.com" },
    postMessage(message) { posted.push(message); },
  };
  context.window.XMLHttpRequest = FakeXhr;
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "..", "page_session.js"), "utf8"),
    context,
  );
  return { context, posted };
}

test("page session script observes fetch authorization and refresh access response", async () => {
  const access = "eyJ.fetch.access";
  const { context, posted } = loadPageSession({
    fetchImpl: async () => new Response(JSON.stringify({ access }), {
      headers: { "content-type": "application/json" },
    }),
  });

  await context.window.fetch("https://api.gtowizard.com/v1/token/refresh/", {
    headers: {
      authorization: "Bearer eyJ.request.access",
      gwclientid: "client-1",
    },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(posted.map((message) => message.accessToken), [
    "eyJ.request.access",
    access,
  ]);
  assert.equal(posted[0].clientId, "client-1");
  assert.equal(posted[0].source, "ai-poker-wizard-gtow-session");
  assert.equal(posted[0].type, "GTOW_SESSION_OBSERVED");
});

test("page session script observes XMLHttpRequest authorization", () => {
  const { context, posted } = loadPageSession();

  const xhr = new context.XMLHttpRequest();
  xhr.open("GET", "https://api.gtowizard.com/v1/poker/next-actions/");
  xhr.setRequestHeader("Authorization", "Bearer eyJ.xhr.access");
  xhr.setRequestHeader("GWCLIENTID", "client-xhr");
  xhr.send();

  assert.equal(posted.length, 1);
  assert.equal(posted[0].accessToken, "eyJ.xhr.access");
  assert.equal(posted[0].clientId, "client-xhr");
});

test("page session script ignores bearer tokens sent outside GTOW API", async () => {
  const { context, posted } = loadPageSession();

  await context.window.fetch("https://telemetry.example.com/events", {
    headers: {
      authorization: "Bearer eyJ.not.gtow",
      gwclientid: "not-gtow",
    },
  });
  const xhr = new context.XMLHttpRequest();
  xhr.open("GET", "https://support.example.com/profile");
  xhr.setRequestHeader("Authorization", "Bearer eyJ.also.not.gtow");
  xhr.send();

  assert.deepEqual(posted, []);
});
