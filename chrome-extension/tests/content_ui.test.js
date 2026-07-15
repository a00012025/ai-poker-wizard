const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tagName, document) {
    this.tagName = tagName;
    this.document = document;
    this.children = [];
    this.style = { cssText: "" };
    this.attributes = {};
  }

  append(...children) {
    this.children.push(...children);
    children.forEach((child) => this.document.register(child));
  }

  appendChild(child) {
    this.append(child);
  }

  addEventListener() {}

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  querySelector(selector) {
    return this.document.getElementById(selector.replace(/^#/, ""));
  }
}

function makeDocument() {
  const elements = new Map();
  const document = {
    hidden: false,
    createElement: (tagName) => new Element(tagName, document),
    getElementById: (id) => elements.get(id) || null,
    addEventListener() {},
    register(element) {
      if (element.id) elements.set(element.id, element);
      element.children.forEach((child) => document.register(child));
    },
  };
  document.documentElement = new Element("html", document);
  return document;
}

test("floating ingest control stays compact and accessible", () => {
  const document = makeDocument();
  const source = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
  const context = {
    chrome: { runtime: { sendMessage: () => Promise.resolve() } },
    crypto: require("node:crypto").webcrypto,
    document,
    localStorage: { getItem: () => "" },
    setInterval() {},
    setTimeout() {},
    clearTimeout() {},
    TextEncoder,
    Uint8Array,
    window: { addEventListener() {} },
  };

  vm.runInNewContext(source, context);

  const button = document.getElementById("apw-ingest-button");
  assert.equal(button.textContent, "♠");
  assert.equal(button.title, "同步手牌到 DB");
  assert.equal(button.attributes["aria-label"], "同步手牌到 DB");
  assert.match(button.style.cssText, /width:42px;height:42px/);
  assert.doesNotMatch(button.textContent, /同步手牌到 DB/);
});
