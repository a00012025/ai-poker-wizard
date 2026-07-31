const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");

const repoRoot = path.resolve(__dirname, "..", "..");

test("extension package includes every manifest content script", () => {
  const archive = execFileSync(
    "bash",
    [path.join(repoRoot, "scripts", "package_extension.sh")],
    { cwd: repoRoot, encoding: "utf8" },
  ).trim();
  const packagedFiles = execFileSync(
    "unzip",
    ["-Z1", archive],
    { encoding: "utf8" },
  ).trim().split("\n");

  assert.ok(packagedFiles.includes("content.js"));
  assert.ok(packagedFiles.includes("page_session.js"));
});
