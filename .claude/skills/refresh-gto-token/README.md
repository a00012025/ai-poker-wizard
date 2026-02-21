# Refresh GTO Wizard Token

Use when the bot reports "GTO Wizard session 過期" or the user asks to update GTO Wizard tokens.

## Trigger Phrases

- "更新 GTO token" / "refresh gto token"
- "GTO Wizard 過期" / "token expired"
- "登入 GTO Wizard" / "login to gto wizard"

## Process

### Step 1: Open Chrome for login

Launch the system Chrome (NOT agent-browser — session restore doesn't work with GTO Wizard's SPA auth).

```bash
DISPLAY=:24 google-chrome --window-size=1426,1052 --window-position=0,0 "https://app.gtowizard.com/login" &>/dev/null &
disown
```

Tell the user to log in via Chrome Remote Desktop (Google/Facebook/Apple sign-in).

### Step 2: Extract refresh token from Chrome's localStorage on disk

After the user confirms login, extract the token from Chrome's LevelDB storage:

```bash
strings "/home/harry/.config/google-chrome/Default/Local Storage/leveldb/000003.log" 2>/dev/null \
  | grep -o 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*' \
  | while read token; do
      TYPE=$(echo "$token" | cut -d. -f2 | python3 -c "import sys,base64,json; p=sys.stdin.read().strip(); p+='='*(-len(p)%4); print(json.loads(base64.urlsafe_b64decode(p)).get('token_type',''))" 2>/dev/null)
      if [ "$TYPE" = "refresh" ]; then
        echo "$token"
      fi
    done | tail -1
```

### Step 3: Update .tokens.json on host AND in container

**CRITICAL: Docker bind mount inode issue**

The `.tokens.json` file is bind-mounted into the Docker container. If you create a new file (new inode), the container still sees the old file. You MUST update BOTH:

1. **Host file** (for future deploys):
```bash
python3 -c "
import json
with open('/home/harry/ai-poker-wizard/.tokens.json', 'r+') as f:
    f.seek(0)
    json.dump({'refresh': '<NEW_TOKEN>', 'access': ''}, f, indent=2)
    f.truncate()
"
```

2. **Container file** (for immediate effect):
```bash
docker compose exec bot sh -c 'cat > /app/.tokens.json << '"'"'EOF'"'"'
{
  "refresh": "<NEW_TOKEN>",
  "access": ""
}
EOF'
```

Do NOT use the Write tool for `.tokens.json` — it creates a new file with a new inode, breaking the Docker bind mount.

### Step 4: Verify

```bash
# Verify container sees new token
docker compose exec bot python3 -c "
from scripts.gto_token import get_access_token
token = get_access_token()
print(f'OK: access token length={len(token)}')
"
```

## Important Notes

- The `access` token field can be empty — `gto_token.py` auto-refreshes it from the refresh token
- Refresh tokens expire in ~5 years, so this is rarely needed
- The LevelDB file path may vary — if `000003.log` doesn't have it, try other `.log` files in the same directory
- Chrome must remain open during token extraction (LevelDB is locked while Chrome runs, but `strings` still works)
