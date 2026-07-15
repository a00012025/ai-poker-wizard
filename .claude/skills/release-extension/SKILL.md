---
name: release-extension
description: Use when the Chrome extension (chrome-extension/) changed and needs to ship — packaging a new build and publishing it as a GitHub release for users to download or reload. Triggers on "發 extension release", "extension 新版本", "publish extension", "package extension", "ext-v", "extension release", "更新 extension".
user_invocable: true
---

# Release the Chrome Extension

The extension is distributed **as a GitHub release** (tag `ext-vX.Y.Z`), NOT via the
Chrome Web Store. `manifest.json` has **no `update_url`**, so users do not auto-update —
they download the zip from the release (or reload the unpacked folder). Every user-facing
extension change therefore needs a fresh release, or users stay on the old build.

This is separate from `bash scripts/deploy.sh` (bot + edge function + DB). Deploying the
backend does NOT publish the extension. If a change touches both, do the backend deploy
first (so the edge-function routes the new extension calls exist), then release the extension.

## When to Use

- After merging any change under `chrome-extension/` that users should get.
- Right after `deploy` when the extension side of a feature also changed (e.g. the ingest
  trigger button — backend route + extension button ship together).
- **Not** for backend-only changes (bot, edge function, migrations) — use `deploy`.

## Steps

### 1. Bump the manifest version (if not already bumped in the merged PR)

`chrome-extension/manifest.json` → `"version"` must be a **new** semver higher than the
latest `ext-v*` release. Package + release use this value verbatim for the zip name and tag.

```bash
grep '"version"' chrome-extension/manifest.json
gh release list | head -3          # confirm it's higher than the latest ext-v*
```

If it still matches the latest release, bump it (patch for fixes, minor for features),
commit, and push before releasing — a duplicate tag will fail.

### 2. Package the zip

```bash
bash scripts/package_extension.sh
```

Prints the output path: `dist/ai-poker-wizard-gtow-sync-v${VERSION}.zip`. The script
validates that every required file exists (manifest, JS, popup, README, all four icons)
and fails loudly if one is missing — do not hand-zip around a failure.

### 3. Publish the GitHub release

**Publishing to the public repo is an outward-facing action — confirm the user wants THIS
release published before running `gh release create`.** ("Should we release?" is a question,
not authorization.)

```bash
gh release create ext-v${VERSION} \
  "dist/ai-poker-wizard-gtow-sync-v${VERSION}.zip" \
  --title "AI Poker Wizard GTOW 自動同步 Extension v${VERSION}" \
  --notes "$NOTES"
```

Tag convention: **`ext-vX.Y.Z`** (matches the manifest version). Asset: the packaged zip.

Release notes follow the prior release's shape (`gh release view ext-v2.0.0` for the
template) — Traditional Chinese, sections: `## 新功能` (what changed, derived from
`git diff --stat ext-v<prev>..HEAD -- chrome-extension/` + the popup/manifest text) and
`## 安裝方式` (download → `chrome://extensions` → 開發人員模式 → 載入未封裝項目; existing
users can just hit reload). Close with the token-privacy line.

### 4. Verify

```bash
gh release view ext-v${VERSION}          # asset attached, notes render
```

Then tell the user how to update: reload the unpacked extension at `chrome://extensions`
(fastest for the owner) or download+unzip the new asset.

## Quick Reference

| Item | Value |
|------|-------|
| Tag format | `ext-vX.Y.Z` (= manifest `version`) |
| Zip | `dist/ai-poker-wizard-gtow-sync-v${VERSION}.zip` |
| Package script | `scripts/package_extension.sh` |
| Prior release (notes template) | `gh release view ext-v2.0.0` |
| Auto-update? | No `update_url` — manual download/reload |

## Common Mistakes

- **Deploying the backend and forgetting the extension.** `deploy.sh` never touches the
  release. Users keep the old popup until you publish `ext-v*`.
- **Reusing the current version.** `gh release create` fails on a duplicate tag; bump
  `manifest.json` first.
- **Publishing without confirmation.** The repo is public — get an explicit go-ahead for
  the specific release before `gh release create`.
- **Hand-editing the zip.** Let `package_extension.sh` build it so the required-file check
  runs; a missing icon or JS file ships a broken extension.
