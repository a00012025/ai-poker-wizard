#!/usr/bin/env bash
# PreToolUse hook: block `source .env` / `set -a && source .env` in Bash commands.
# Scripts should use python-dotenv (load_dotenv) instead.
#
# Exit 2 → block the tool call; stderr is fed back to Claude as guidance.

set -euo pipefail

# Read hook input JSON from stdin
input="$(cat)"

# Extract the bash command. Use python to parse JSON safely (jq may not exist).
cmd="$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))')"

# Normalize whitespace for matching
norm="$(printf '%s' "$cmd" | tr -s '[:space:]' ' ')"

# Match patterns like:
#   source .env
#   . .env
#   set -a && source .env
#   set -a; source .env; set +a
# but allow sourcing other files (e.g. virtualenv activate, ~/.zshrc, etc.)
if printf '%s' "$norm" | grep -Eq '(^|[^[:alnum:]_/.-])(source|\.)[[:space:]]+([^[:space:]&;|]*/)?\.env([[:space:]&;|]|$)'; then
  cat >&2 <<'EOF'
Blocked: `source .env` is disallowed in this project.

Use python-dotenv instead. In scripts/_tmp.py and other ad-hoc scripts:

    from dotenv import load_dotenv
    load_dotenv()  # loads .env from cwd/parents

Then just run:

    python scripts/_tmp.py

(no `set -a && source .env && set +a` wrapper needed)
EOF
  exit 2
fi

# Allow the command
exit 0
