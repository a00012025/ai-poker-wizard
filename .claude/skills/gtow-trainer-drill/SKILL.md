---
name: gtow-trainer-drill
description: >
  Navigate the GTO Wizard SPA over a live CDP session and build precise GTOW
  Trainer drill deep-links. Use when you need to (a) reverse-engineer what URL
  params a GTOW page uses, (b) drive the Practice/Trainer UI from the terminal,
  or (c) construct a `/practice/trainer?...` deep-link that lands the player on
  an exact spot (position + action + IP/OOP + street). Triggers: "gtow trainer
  參數", "drill 連結", "how does the trainer URL encode X", "reverse engineer
  gtow drill", "build a precise practice link".
---

# GTO Wizard Trainer drill deep-links (CDP navigation)

The companion to `gtow-cdp-session` (which handles *attaching*). This skill is
the hard-won map of **how the SPA behaves** and **the Trainer drill URL
contract**, so you never have to trial-and-error the UI again.

## 0. Attach first

Reuse the dedicated debug profile from `gtow-cdp-session`. A CDP Chrome usually
persists on port 9222 already:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9222/json/version   # 200 == up
agent-browser --cdp 9222 tab list          # authed when NOT on /login
```

If it's down or logged out, follow `gtow-cdp-session` to relaunch + log in.

## 1. The Trainer drill URL contract (VERIFIED 2026-07-09)

When you configure a drill and click **START TRAINING**, the SPA navigates to
`https://app.gtowizard.com/practice/trainer?...`. Two classes of params:

**Config params (the drill filter — this is what you build to pin a spot):**

| Param | Meaning | Values |
|---|---|---|
| `fh_hero` | hero position(s) | `UTG,UTG+1,LJ,HJ,CO,BTN,SB,BB` — comma = "any of" |
| `fh_opponent` | opponent position(s) | same set, comma-separated |
| `fh_rel_positions` | relative position | `IP` / `OOP` |
| `fh_actions` | preflop action(s) | `RFI`, `vsSRP` (= "vs Open"), `possibleSqueeze` (= open + call), `vs3bet`, `vs4bet`, `vsSqueeze`, `vsLimp`, `vsIso`, `StartOfHand` (= "From start") — comma-separated |
| `fh_start_spot` | where the drill starts | `preflop` / `flop` / `custom_spot` |
| `preflop_actions` | exact preflop line (custom_spot) | e.g. `F-F-F-R2-F-F-F-C` |
| `flop_actions` | exact flop line (custom_spot) | e.g. `X-R1.8-R4.8` |
| `depth` / `depth_list` | stack depth (bb+0.125) | `20.125`; list is CSV of allowed depths |
| `gametype` | solution set | `MTTGeneral` (MTT ChipEV), `MTTMRonlyGeneral`, ... |
| `fh_groups` / `fh_groups_selection` | pinned hero hand range | `manual` + CSV of `AKs,22,...` |

**Instance params (the *dealt hand*, NOT filter — ignore when building links):**
`fh_hero_tmp`, `fh_opponent_tmp`, `fh_rel_position_tmp`, `fh_action_tmp`,
`hand_tmp`, `board_tmp`, `history_spot`, and the *realized* `preflop_actions` /
`flop_actions` for that one hand. The `_tmp` suffix always = "this hand".

**Label → value gotchas:** the UI label "vs Open" encodes as `fh_actions=vsSRP`;
"From start" = `StartOfHand`. Open + call uses `possibleSqueeze` (the repo's
taxonomy name `vsRaiseCall` is not accepted by the Trainer SPA). `vsIso`/
`vsLimp`/`vs3bet`/`vs4bet`/`vsSqueeze`/`RFI` match their labels. A flop drill with a specific action line
uses `fh_start_spot=custom_spot` + `flop_actions=...`, not `fh_start_spot=flop`.

Baseline UI flags (ranges visible, etc.) live in `scripts/gtow_trainer_url.py`
`_TRAINER_UI_DEFAULTS`. That builder currently pins only `fh_actions` + depth +
`fh_start_spot` ("Option Y"); the params above are the deferred "Option Z" —
extend it with `fh_hero`/`fh_opponent`/`fh_rel_positions` to pin the exact spot.

## 2. Position category → GTOW positions

Repo taxonomy (`scripts/spot_taxonomy.py`) → `fh_hero`/`fh_opponent` value:
`EP=UTG,UTG+1` · `MP=LJ,HJ` · `LP=CO,BTN` · `SB=SB` · `BB=BB`. For an exact-hero
line (RFI/vsOpen) pass the single position; for a category line pass the CSV.

## 3. How to drive the SPA (what actually works)

The Practice UI is a React SPA. Lessons that save loops:

- **Click by text via `eval`, not by coordinates.** Icon buttons are SVGs;
  `document.elementFromPoint(x,y).click()` hits a wrapper `div` and does nothing.
  Match visible text instead:
  ```bash
  agent-browser --cdp 9222 eval '(()=>{const e=[...document.querySelectorAll("*")]
    .find(x=>x.textContent.trim()==="START TRAINING"&&x.offsetWidth>0&&x.children.length<=2);
    if(e){e.click();return"ok"}return"miss"})()'
  ```
  Constrain with `children.length<=1/2` so you get the leaf element, not an
  ancestor wrapping the whole page.
- **The "New" button's label is an 8px `<span>`; click its `BUTTON` ancestor**
  (walk up ~4 parents to `tagName==="BUTTON"`).
- **Reaching the drill config page:** from `/practice/drills` → click **New** →
  ensure **Hold'em** (not PLO) → the config page (visual table + Starting spot +
  Preflop action + ALL SETTINGS + START TRAINING) appears. Clicking a **drill
  name in the left sidebar** loads that saved drill's full pinned config — the
  fastest way to see how a precise spot is encoded. ("Configure Training" text
  match is flaky; clicking a sidebar drill name works reliably.)
- **Read the resulting URL** after START with `agent-browser --cdp 9222 get url`,
  then `grep -oE 'fh_[a-z_]*=[^&]*'` to isolate the config params.
- **`open <url>` forces fresh state**; `history.back()` on the SPA often lands on
  the running trainer, not the config page — prefer `open` + re-drive.
- **Screenshot to orient** (`agent-browser --cdp 9222 screenshot /tmp/x.png` then
  Read it) whenever a text-match misses — the header Hold'em/PLO toggle and the
  current route are easy to lose track of.
- Saved drills that encode useful configs to inspect: "Mastering the BTN"
  (`fh_hero=BTN`, `fh_actions=vsSRP,RFI`), "IP Flop C-Bet raised"
  (`fh_rel_positions=IP`, `fh_start_spot=custom_spot`, `flop_actions=X-R1.8-R4.8`),
  "Blind vs Blind" (`fh_hero=SB,BB`, `fh_opponent=BB,SB`).

## 4. Example — build a precise deep-link

Spot: hero **BB** defending **OOP** vs a **late-position open**, ~20bb, preflop.
```
https://app.gtowizard.com/practice/trainer?solution_type=gwiz&gametype=MTTGeneral
  &depth=20.125&depth_list=20.125&fh_start_spot=preflop&fh_actions=vsSRP
  &fh_hero=BB&fh_opponent=CO,BTN&fh_rel_positions=OOP
  &fh_groups_selection=manual&...(_TRAINER_UI_DEFAULTS)...&dialogs=
```
Opening it drops the player straight into that drill; GTOW deals hands within
the filter (each dealt hand adds `*_tmp` params).

## 5. Teardown

Leave Chrome running for the next session (profile + login persist). To stop:
`pkill -f "user-data-dir=$HOME/.config/gtow-cdp"`.
