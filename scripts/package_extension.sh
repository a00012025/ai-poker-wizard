#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXT="$ROOT/chrome-extension"
cd "$ROOT"
VERSION=$(python -c 'import json; print(json.load(open("chrome-extension/manifest.json"))["version"])')
OUT_DIR="$ROOT/dist"
OUT="$OUT_DIR/ai-poker-wizard-gtow-sync-v${VERSION}.zip"

mkdir -p "$OUT_DIR"
rm -f "$OUT"

required=(manifest.json config.js background.js content.js popup.html popup.css popup.js README.md
          icons/icon16.png icons/icon32.png icons/icon48.png icons/icon128.png)
for file in "${required[@]}"; do
  [[ -f "$EXT/$file" ]] || { echo "missing extension file: $file" >&2; exit 1; }
done

(
  cd "$EXT"
  zip -q "$OUT" "${required[@]}"
)

echo "$OUT"
