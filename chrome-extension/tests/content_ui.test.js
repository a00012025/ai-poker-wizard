const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("content script does not inject a floating ingest control", () => {
  const document = {
    hidden: false,
    addEventListener() {},
    createElement() {
      throw new Error("content script must not inject page UI");
    },
  };
  const source = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
  const context = {
    chrome: { runtime: { sendMessage: () => Promise.resolve() } },
    crypto: require("node:crypto").webcrypto,
    document,
    localStorage: { getItem: () => "" },
    setInterval() {},
    TextEncoder,
    Uint8Array,
    window: { addEventListener() {} },
  };

  vm.runInNewContext(source, context);
});
