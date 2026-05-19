#!/usr/bin/env python3
"""Scrape every hand's PokerCraft replay image in BB view, keyed by hand_id.

Proven mechanism (verified against my.pokercraft.com):

  - The tournament-list tab (pin with --tab) shows 165 tournaments for the
    active date filter. Trusted-click a tournament's date cell to select it,
    then trusted-click the "Game History" tab → a paginated hand table whose
    rows each carry the TMxxxx hand id.
  - Trusted-click a hand-id cell → a MAT-DIALOG modal with the replay as
    <img id="hand-scene" src="data:image/png;base64,...">. The hand id is
    NOT in the modal, so it must be read from the row before clicking.
  - The "BB" header button only reacts to a REAL pointer click
    (move→down→up); Playwright's element click is ignored. After the raw
    click it gets active="true" and #hand-scene re-renders in BB units.
  - Read the data URL, decode, save <hand_id>.png. Close via the OK button.

Resumable: existing <hand_id>.png are skipped. A manifest maps
hand_id → tournament. Pair with ground_truth.jsonl (also keyed by hand_id)
for the OCR benchmark.

Usage:
  python scripts/scrape_hand_images.py --tab t4 --out data/hand_images
  python scripts/scrape_hand_images.py --tab t4 --out data/hand_images --limit-tourneys 1
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AB = "agent-browser"


def ab(*args: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run([AB, *args], capture_output=True, text=True,
                            timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def ev(js: str, timeout: int = 60):
    out = ab("eval", js, timeout=timeout)
    if out.startswith('"') and out.endswith('"'):
        try:
            return json.loads(out)
        except Exception:
            return out
    return out


def snap_ref(pattern: str) -> str | None:
    """Return the first snapshot ref whose line matches pattern (regex)."""
    for line in ab("snapshot", "-i", "-c").splitlines():
        if re.search(pattern, line):
            m = re.search(r"ref=(e\d+)", line)
            if m:
                return m.group(1)
    return None


def click_ref(ref: str) -> bool:
    if not ref:
        return False
    ab("click", f"@{ref}")
    return True


def raw_mouse_click(x: int, y: int) -> None:
    ab("mouse", "move", str(x), str(y))
    time.sleep(0.12)
    ab("mouse", "down", "left")
    time.sleep(0.1)
    ab("mouse", "up", "left")


def pin_tab(tab: str) -> None:
    ab("tab", tab)
    time.sleep(1)
    n = ev("document.querySelectorAll('table')[0]"
           "?.querySelectorAll('tbody tr').length||0")
    if not str(n).isdigit() or int(n) <= 0:
        sys.exit(f"--tab {tab}: no tournament list. Open the filtered list there.")
    print(f"[tab {tab}] {n} tournament rows")


def tournament_rows() -> list[dict]:
    return json.loads(ev(
        "(()=>{const t=document.querySelectorAll('table')[0];"
        "return JSON.stringify([...t.querySelectorAll('tbody tr')]"
        ".map((r,i)=>{const c=[...r.children].map(td=>td.innerText.trim());"
        "return {i,date:c[1]||'',name:c[2]||''};}));})()"))


def _dismiss() -> None:
    """Close the PokerCraft replay modal.

    The hand-modal is a CDK overlay (.cdk-overlay-pane.hand-modal). Its OK
    button is inert/off-viewport and Escape does NOT close it — but a click
    on the .cdk-overlay-backdrop does (verified). A stranded backdrop is
    fatal: agent-browser snapshots then only see the overlay, so every
    subsequent table ref click fails. This is the single reliable close.
    """
    ev("(()=>{const b=document.querySelector('.cdk-overlay-backdrop');"
       "if(b){b.click();return 1;}return 0;})()")
    ab("press", "Escape")  # belt-and-braces for any non-backdrop popup


def clear_overlay() -> bool:
    for _ in range(10):
        if _overlay_gone():
            return True
        _dismiss()
        time.sleep(0.6)
    return _overlay_gone()


def _gh_open() -> bool:
    return ev(_HAND_TABLE_JS) and json.loads(ev(_HAND_TABLE_JS))["k"] >= 0


def open_game_history(row_i: int) -> bool:
    """Ensure tournament row_i is selected and its Game History tab is open.

    Idempotent: a tournament row toggles selection, so blindly clicking can
    DEselect an already-open tournament. Clear any selection first, then
    select row_i and open Game History, verifying the hand table appears.
    """
    clear_overlay()
    # Deselect anything currently selected so the row click reliably selects.
    ev("""(()=>{for(const tr of (document.querySelectorAll(
      'table')[0]?.querySelectorAll('tbody tr')||[])){
      const cb=tr.querySelector('input[type=checkbox]');
      if(cb&&cb.checked) cb.click();}})()""")
    time.sleep(0.6)
    cell = ev(f"""(()=>{{const t=document.querySelectorAll('table')[0];
      const r=[...t.querySelectorAll('tbody tr')][{row_i}];
      if(!r) return ''; r.scrollIntoView({{block:'center'}});
      return (r.children[1]?.innerText||'').trim();}})()""")
    if not cell:
        return False
    for _ in range(3):
        ref = snap_ref(re.escape(cell) + r'.*\[ref=e\d+')
        if not ref or not click_ref(ref):
            time.sleep(1)
            continue
        time.sleep(2.5)
        gh = snap_ref(r'"Game History".*\[ref=e\d+')
        if gh:
            click_ref(gh)
            time.sleep(3)
            if _gh_open():
                return True
        time.sleep(1)
    return _gh_open()


_HAND_TABLE_JS = """(()=>{const ts=[...document.querySelectorAll('table')];
  for(let k=0;k<ts.length;k++){const rs=[...ts[k].querySelectorAll('tbody tr')];
   if(rs.some(r=>/TM\\d{6,}/.test(r.innerText)))
     return JSON.stringify({k, ids:rs.map(r=>{const m=r.innerText.match(/TM\\d{6,}/);
       return m?m[0]:null;})});}
  return JSON.stringify({k:-1,ids:[]});})()"""


def hand_table() -> dict:
    return json.loads(ev(_HAND_TABLE_JS))


def _overlay_gone() -> bool:
    return ev("(()=>{return !document.getElementById('hand-scene') && "
              "!document.querySelector('.cdk-overlay-pane.hand-modal') && "
              "!document.querySelector('.cdk-overlay-backdrop');})()") in (
        "true", True)


def open_hand_modal(hand_id: str) -> bool:
    for attempt in range(3):
        # Make sure no leftover overlay is intercepting clicks.
        for _ in range(10):
            if _overlay_gone():
                break
            time.sleep(0.4)
        # Scroll the target row into view so it appears in the a11y snapshot
        # and is clickable.
        ev(f"""(()=>{{const t=[...document.querySelectorAll('table')];
          for(const tb of t){{for(const r of tb.querySelectorAll('tbody tr')){{
            if(r.innerText.includes('{hand_id}')){{
              r.scrollIntoView({{block:'center'}}); return 1;}}}}}}
          return 0;}})()""")
        time.sleep(0.6)
        ref = snap_ref(rf'cell "{hand_id}".*\[ref=e\d+')
        if ref and click_ref(ref):
            for _ in range(15):
                if ev("!!document.getElementById('hand-scene')") in (
                        "true", True):
                    return True
                time.sleep(0.5)
        time.sleep(1)
    return False


def set_bb_view() -> bool:
    """Raw-click the BB header button until #hand-scene re-renders in BB.

    Fast path: 1 click then poll every 0.25s; the re-render lands in <1s.
    """
    pos = ev("""(()=>{const m=document.querySelector('app-hand-scene-modal');
      if(!m) return ''; const b=[...m.querySelectorAll('button')][2];
      if(!b) return ''; const i=document.getElementById('hand-scene');
      window.__s=i?i.src:''; const r=b.getBoundingClientRect();
      return JSON.stringify([Math.round(r.x+r.width/2),
                             Math.round(r.y+r.height/2)]);})()""")
    if not pos:
        return False
    x, y = json.loads(pos)
    for _ in range(3):
        raw_mouse_click(x, y)
        for _ in range(8):
            time.sleep(0.25)
            s = json.loads(ev(
                "(()=>{const m=document.querySelector('app-hand-scene-modal');"
                "const b=m?[...m.querySelectorAll('button')][2]:null;"
                "const i=document.getElementById('hand-scene');"
                "return JSON.stringify({act:b?b.getAttribute('active'):null,"
                "changed:i?(i.src!==window.__s):false});})()"))
            if s["act"] == "true" and s["changed"]:
                return True
    return False


def bb_active() -> bool:
    """True if the modal's BB toggle is currently on (image is BB-view)."""
    return ev("(()=>{const m=document.querySelector('.cdk-overlay-pane"
              ".hand-modal,app-hand-scene-modal');const b=m?"
              "[...m.querySelectorAll('button')][2]:null;"
              "return !!b && b.getAttribute('active')==='true';})()") in (
        "true", True)


def set_page_size_max() -> int:
    """Pick the largest rows-per-page so there are far fewer pages to walk.

    The trigger is a normal button ("10 Rows") and the menu items are
    buttons/menuitems — all honour Playwright ref clicks (like the pager),
    so use snap_ref+click_ref, not raw mouse (which stranded the overlay).
    Returns the chosen size, or 0 if unchanged (caller falls back to 10).
    """
    cur = ev("""(()=>{const b=document.querySelector(
      '.pagination .mat-menu-trigger'); if(!b) return '';
      b.scrollIntoView({block:'center'});
      b.setAttribute('aria-label','SCRAPE_PGSIZE');
      return (b.innerText||'').trim();})()""")
    if not cur:
        return 0
    ref = snap_ref(r'SCRAPE_PGSIZE.*\[ref=e\d+')
    if not ref or not click_ref(ref):
        return 0
    time.sleep(1.0)
    # Tag each numeric menu option, then ref-click the largest.
    opts = json.loads(ev("""(()=>{const it=[...document.querySelectorAll(
      '.cdk-overlay-container [role=menuitem],.cdk-overlay-container button,'
      +'.mat-menu-panel button')]
      .map(e=>{const m=(e.textContent||'').match(/(\\d+)/);
        return m?{n:+m[1],e}:null;}).filter(Boolean);
      it.forEach((o,i)=>o.e.setAttribute('aria-label','SCRAPE_OPT_'+o.n));
      return JSON.stringify(it.map(o=>o.n));})()""") or "[]")
    if not opts:
        ab("press", "Escape")
        return 0
    best = max(opts)
    ref = snap_ref(rf'SCRAPE_OPT_{best}\b.*\[ref=e\d+')
    if ref:
        click_ref(ref)
        time.sleep(1.5)
        return best
    ab("press", "Escape")
    return 0


def collect_list_ids() -> list[str]:
    """Read every Game-History hand id in display order across all pages.

    Pure DOM reads + pager clicks — no modals — so this is cheap. The
    in-modal right arrow walks hands in this same order, so image #k maps
    to ids[k] (per the tournament's hand sequence)."""
    ids: list[str] = []
    seen_pages = 0
    while True:
        page = json.loads(ev(_HAND_TABLE_JS))["ids"]
        ids.extend(h for h in page if h)
        seen_pages += 1
        if seen_pages > 60 or not hand_pager_next():
            break
    return ids


def _cur_page() -> str:
    return str(ev("(()=>{const c=document.querySelector('.pagination "
                  ".current-page');return c?c.textContent.trim():'';})()"))


def _pager_click(arrow: str) -> str:
    """Tag the pager prev/next button with an aria-label, then Playwright-
    click it by ref. The pager (unlike BB/OK) DOES honour ref clicks, so
    this is reliable; raw-mouse on these tiny off-viewport buttons was not.

    Returns 'last' (button absent/disabled) or the page number before click.
    """
    icon = "fa-angle-right" if arrow == "next" else "fa-angle-left"
    o = json.loads(ev(f"""(()=>{{const p=document.querySelector('.pagination');
      if(!p) return JSON.stringify({{last:true}});
      const cur=p.querySelector('.current-page');
      const b=[...p.querySelectorAll('button')].find(x=>
        x.querySelector('i.{icon}'));
      if(!b || b.disabled || b.getAttribute('disabled')!==null)
        return JSON.stringify({{last:true}});
      b.setAttribute('aria-label','SCRAPE_PAGER');
      b.scrollIntoView({{block:'center'}});
      return JSON.stringify({{cur:(cur?cur.textContent.trim():'')}});}})()"""))
    if o.get("last"):
        return "last"
    cur = o["cur"]
    ref = snap_ref(r'SCRAPE_PAGER.*\[ref=e\d+')
    if ref:
        click_ref(ref)
    return cur


def pager_to_first() -> None:
    """Page LEFT until back on page 1 (never re-selects the tournament)."""
    for _ in range(80):
        if _cur_page() in ("1", ""):
            return
        cur = _pager_click("prev")
        if cur == "last":
            return
        for _ in range(10):
            time.sleep(0.3)
            if _cur_page() != cur:
                break


def open_hand_by_id(hand_id: str) -> bool:
    """Open the replay modal for hand_id (anchor; once per tournament).

    Uses the proven path: scroll the row into view so it lands in the a11y
    snapshot, then a Playwright click on the cell ref (raw-mouse coords on
    this Angular cell were unreliable; the ref click opens it every time).
    """
    for _ in range(4):
        if not _overlay_gone():
            clear_overlay()
        ev(f"""(()=>{{for(const tb of document.querySelectorAll('table')){{
          for(const r of tb.querySelectorAll('tbody tr')){{
            if(r.innerText.includes('{hand_id}')){{
              r.scrollIntoView({{block:'center'}}); return 1;}}}}}}
          return 0;}})()""")
        time.sleep(0.5)
        ref = snap_ref(rf'cell "{hand_id}".*\[ref=e\d+')
        if ref and click_ref(ref):
            # The modal opens immediately as a .hand-modal CDK pane, but the
            # replay <img id=hand-scene> is rendered asynchronously and on
            # the first hand can take well over 5s. As long as the pane is
            # up, keep waiting (don't clear/retry, which races the render).
            for _ in range(60):
                time.sleep(0.5)
                st = json.loads(ev(
                    "(()=>JSON.stringify({s:!!document.getElementById("
                    "'hand-scene'),m:!!document.querySelector("
                    "'.cdk-overlay-pane.hand-modal')}))()"))
                if st["s"]:
                    return True
                if not st["m"] and _ > 6:
                    break  # pane never appeared / vanished — retry
        time.sleep(0.6)
    return False


def nav_right() -> bool:
    """Raw-click the in-modal right arrow; True once #hand-scene advances."""
    o = json.loads(ev("""(()=>{const n=document.querySelector('.navigator-right');
      const i=document.getElementById('hand-scene');
      if(!n||!i) return JSON.stringify({err:1});
      window.__p=i.src; const r=n.getBoundingClientRect();
      return JSON.stringify({x:Math.round(r.x+r.width/2),
        y:Math.round(r.y+r.height/2)});})()"""))
    if o.get("err"):
        return False
    raw_mouse_click(o["x"], o["y"])
    for _ in range(20):
        time.sleep(0.25)
        if ev("(()=>{const i=document.getElementById('hand-scene');"
              "return i&&i.src!==window.__p;})()") in ("true", True):
            return True
    return False


def grab_scene(dest: Path) -> bool:
    ev("window.__g=document.getElementById('hand-scene').src")
    out = ab("eval", "window.__g")
    if out.startswith('"'):
        out = json.loads(out)
    if "," not in out:
        return False
    dest.write_bytes(base64.b64decode(out.split(",", 1)[1]))
    return dest.stat().st_size > 1000


def close_modal() -> None:
    for _ in range(10):
        if _overlay_gone():
            return
        _dismiss()
        time.sleep(0.5)


def hand_pager_next() -> bool:
    """Advance the hand list to the next page (Playwright ref click)."""
    cur = _pager_click("next")
    if cur == "last":
        return False
    for _ in range(14):
        time.sleep(0.4)
        if _cur_page() != cur:
            time.sleep(1.0)  # let the new page's rows render
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True, help="agent-browser tab id (e.g. t4)")
    ap.add_argument("--out", default="data/hand_images")
    ap.add_argument("--limit-tourneys", type=int, default=0)
    ap.add_argument("--gt", default="",
                    help="ground_truth.jsonl: only scrape hands present here "
                         "(every image is then benchmarkable)")
    args = ap.parse_args()

    gt_ids: set[str] | None = None
    if args.gt:
        gt_ids = set()
        with open(args.gt, encoding="utf-8") as fh:
            for line in fh:
                try:
                    gt_ids.add(json.loads(line)["hand_id"])
                except Exception:
                    pass
        print(f"[gt] restricting to {len(gt_ids)} ground-truth hand ids")

    out = Path(args.out)
    imgs = out / "img"
    imgs.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"
    done = {p.stem for p in imgs.glob("*.png")}
    print(f"[resume] {len(done)} images already scraped")

    pin_tab(args.tab)
    if not clear_overlay():
        print("[warn] could not clear a pre-existing overlay")
    tours = tournament_rows()
    if args.limit_tourneys:
        tours = tours[: args.limit_tourneys]
    print(f"[plan] {len(tours)} tournaments")

    man = man_path.open("a", encoding="utf-8")
    total = 0
    for ti, tr in enumerate(tours):
        # Re-pin the tab each tournament so any drift (a stray tab switch
        # between tournaments) self-corrects without per-hand cost.
        ab("tab", args.tab)
        time.sleep(0.3)
        if not open_game_history(tr["i"]):
            print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']}: "
                  f"could not open Game History; skip")
            continue
        # Fast flow: max page size → record every hand id in time order
        # across pages → page back to 1 → open hand #0, BB once → then just
        # press the right arrow per hand. image #k == ids[k] (the modal has
        # no TM id, but the in-modal arrow walks the same order the list
        # shows). BB is re-applied only if it didn't persist across the
        # arrow — checked once, then trusted, with a cheap per-hand guard.
        size = set_page_size_max()
        ids = collect_list_ids()
        if not ids:
            print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']}: no hands")
            continue
        wanted = [h for h in ids
                  if h not in done and (gt_ids is None or h in gt_ids)]
        print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']} — "
              f"{len(ids)} hands, {len(wanted)} to grab (page size {size or 10})")
        if not wanted:
            continue  # whole tournament already scraped
        pager_to_first()
        if not open_hand_by_id(ids[0]):
            print(f"   could not open first hand {ids[0]}; skip")
            clear_overlay()
            continue
        set_bb_view()
        bb_persists: bool | None = None
        for k, hid in enumerate(ids):
            if k > 0:
                if not nav_right():
                    print(f"   arrow stalled at k={k}; "
                          f"{len(ids)-k} hands unreached")
                    break
                if bb_persists is None:        # learn once per tournament
                    bb_persists = bb_active()
                if not bb_persists and not bb_active():
                    set_bb_view()
            if hid in done or (gt_ids is not None and hid not in gt_ids):
                continue
            if not bb_active():                # cheap guarantee of BB view
                set_bb_view()
            if grab_scene(imgs / f"{hid}.png"):
                done.add(hid)
                total += 1
                man.write(json.dumps(
                    {"hand_id": hid, "tournament": tr["name"],
                     "date": tr["date"], "order_index": k},
                    ensure_ascii=False) + "\n")
                man.flush()
                if total % 25 == 0:
                    print(f"   ... {total} new ({len(done)} total)")
        close_modal()
    man.close()
    print(f"[done] {total} new images this run; "
          f"{len(list(imgs.glob('*.png')))} total in {imgs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
