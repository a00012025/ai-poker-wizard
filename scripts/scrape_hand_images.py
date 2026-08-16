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

from title_ocr import read_title_id  # noqa: E402

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
    # Tag the target row's date cell with a stable aria-label and click it
    # by ref. Matching on the cell's innerText is unreliable: many rows
    # render a garbled multi-cell concatenation ("Apr 19, 20:00Apr 1920:00
    # $35 Mega…Buy-in…") that never equals the clean a11y label, so those
    # tournaments were silently skipped (the 0%-coverage older events).
    ok = ev(f"""(()=>{{const t=document.querySelectorAll('table')[0];
      const r=[...t.querySelectorAll('tbody tr')][{row_i}];
      if(!r) return 0; r.scrollIntoView({{block:'center'}});
      const c=r.children[1]||r.children[0];
      c.setAttribute('aria-label','SCRAPE_ROW'); return 1;}})()""")
    if ok not in (1, "1"):
        return False
    for _ in range(3):
        ref = snap_ref(r'SCRAPE_ROW.*\[ref=e\d+')
        if not ref or not click_ref(ref):
            time.sleep(1)
            continue
        time.sleep(2.5)
        gh = snap_ref(r'"Game History".*\[ref=e\d+')
        if gh:
            click_ref(gh)
            time.sleep(3)
            if _gh_open():
                ev("(()=>{const e=document.querySelector("
                   "'[aria-label=SCRAPE_ROW]');if(e)"
                   "e.removeAttribute('aria-label');})()")
                return True
        time.sleep(1)
    ev("(()=>{const e=document.querySelector('[aria-label=SCRAPE_ROW]');"
       "if(e)e.removeAttribute('aria-label');})()")
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


def _scene_src() -> str:
    s = ev("(()=>{const i=document.getElementById('hand-scene');"
           "return i&&i.src?i.src:'';})()")
    return s if isinstance(s, str) else ""


def open_hand_by_id(hand_id: str) -> bool:
    """Open the replay modal for hand_id by clicking ITS row.

    Scroll the row into view (so it lands in the a11y snapshot), then a
    Playwright click on the cell ref (raw-mouse on this Angular cell was
    unreliable; the ref click opens it every time).

    Crucially, this waits until #hand-scene shows a *fresh, settled* image,
    not merely that the element exists. The modal reuses one <img>, so a
    click can briefly leave the PREVIOUS hand's frame in place — capturing
    then is exactly the stale-anchor bug that shifted all of Daily Hyper 1.
    Clicking a different row always loads a different image, so "src left
    its pre-click value and then stopped changing" pins the right hand.
    This makes the row click correct by construction with no title needed
    — the only id source for long-name events whose title is truncated.
    """
    for _ in range(4):
        if not _overlay_gone():
            clear_overlay()
        pre = _scene_src()
        ev(f"""(()=>{{for(const tb of document.querySelectorAll('table')){{
          for(const r of tb.querySelectorAll('tbody tr')){{
            if(r.innerText.includes('{hand_id}')){{
              r.scrollIntoView({{block:'center'}}); return 1;}}}}}}
          return 0;}})()""")
        time.sleep(0.5)
        ref = snap_ref(rf'cell "{hand_id}".*\[ref=e\d+')
        if ref and click_ref(ref):
            last, stable = None, 0
            for i in range(60):
                time.sleep(0.5)
                st = json.loads(ev(
                    "(()=>JSON.stringify({s:!!document.getElementById("
                    "'hand-scene'),m:!!document.querySelector("
                    "'.cdk-overlay-pane.hand-modal')}))()"))
                if st["s"]:
                    cur = _scene_src()
                    # must be non-empty, different from the pre-click frame,
                    # and unchanged across two reads (render finished)
                    if cur and cur != pre:
                        if cur == last:
                            stable += 1
                            if stable >= 2:
                                return True
                        else:
                            stable = 0
                        last = cur
                    continue
                if not st["m"] and i > 6:
                    break  # pane never appeared / vanished — retry
        time.sleep(0.6)
    return False


def nav_right() -> bool:
    """Raw-click the in-modal right arrow; True once #hand-scene advances.

    A single click occasionally doesn't register (the run stalled mid-
    tournament, leaving the rest unreached). Retry up to 4 times, re-reading
    the arrow's position each time, before giving up.
    """
    for _ in range(4):
        o = json.loads(ev(
            "(()=>{const n=document.querySelector('.navigator-right');"
            "const i=document.getElementById('hand-scene');"
            "if(!n||!i) return JSON.stringify({err:1});"
            "n.scrollIntoView({block:'center'});"
            "window.__p=i.src; const r=n.getBoundingClientRect();"
            "return JSON.stringify({x:Math.round(r.x+r.width/2),"
            "y:Math.round(r.y+r.height/2)});})()"))
        if o.get("err"):
            return False
        raw_mouse_click(o["x"], o["y"])
        for _ in range(16):
            time.sleep(0.25)
            if ev("(()=>{const i=document.getElementById('hand-scene');"
                  "return i&&i.src!==window.__p;})()") in ("true", True):
                return True
    return False


def grab_scene_bytes() -> bytes | None:
    """Return the current #hand-scene PNG bytes (or None if not ready)."""
    ev("window.__g=document.getElementById('hand-scene').src")
    out = ab("eval", "window.__g")
    if out.startswith('"'):
        out = json.loads(out)
    if "," not in out:
        return None
    raw = base64.b64decode(out.split(",", 1)[1])
    return raw if len(raw) > 1000 else None


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
    ap.add_argument("--only-date", default="",
                    help="only process tournaments whose row date cell "
                         "contains this substring (e.g. 'May 03'). Filters on "
                         "the cheap list cell — no Game History open — so a "
                         "targeted backfill skips ~150 irrelevant events fast.")
    ap.add_argument("--ignore-done-tours", action="store_true",
                    help="do not skip tournaments recorded in "
                         "done_tournaments.json. Use when a needed hand sits "
                         "in a row whose tkey was wrongly marked complete "
                         "(name/time collisions): every row is opened and the "
                         "hand-id intersection finds the right one.")
    ap.add_argument("--by-row-match", default="",
                    help="for tournaments whose name contains this substring, "
                         "open EACH hand by its own row click instead of the "
                         "arrow walk. Needed for long-name live events whose "
                         "replay title is truncated (no #TM id in the image), "
                         "so title-OCR can't name the file — the row click is "
                         "correct by construction.")
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
    # Tournaments confirmed fully covered (every GT hand already imaged) are
    # recorded so a resume skips them WITHOUT opening Game History or paging
    # — otherwise each token window is wasted re-walking finished events.
    dt_path = out / "done_tournaments.json"
    done_tours: set[str] = set(
        json.loads(dt_path.read_text())) if dt_path.exists() else set()
    print(f"[resume] {len(done)} images, "
          f"{len(done_tours)} tournaments already complete")

    pin_tab(args.tab)
    if not clear_overlay():
        print("[warn] could not clear a pre-existing overlay")
    tours = tournament_rows()
    if args.only_date:
        tours = [t for t in tours if args.only_date in t["date"]]
        print(f"[only-date] '{args.only_date}': {len(tours)} matching rows")
    if args.limit_tourneys:
        tours = tours[: args.limit_tourneys]
    print(f"[plan] {len(tours)} tournaments")

    man = man_path.open("a", encoding="utf-8")
    total = 0
    for ti, tr in enumerate(tours):
        tkey = f"{tr['date']}|{tr['name']}"
        if tkey in done_tours and not args.ignore_done_tours:
            continue  # already fully covered — skip without opening it
        # Re-pin the tab each tournament so any drift (a stray tab switch
        # between tournaments) self-corrects without per-hand cost.
        ab("tab", args.tab)
        time.sleep(0.3)
        if not open_game_history(tr["i"]):
            print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']}: "
                  f"could not open Game History; skip")
            continue
        # Per-PAGE anchor + arrow. The in-modal right arrow only walks within
        # the loaded page window and hard-stalls at ~page_size steps, so a
        # single anchor can't cover >100-hand tournaments. Instead: max the
        # page size, then for each list page open its first hand, BB once,
        # and arrow-walk just that page's <=100 hands (image #k == that
        # page's ids[k] — verified correct), then page on. Labels stay
        # correct; coverage no longer caps at 100.
        size = set_page_size_max() or 10
        by_row = bool(args.by_row_match) and args.by_row_match in tr["name"]
        print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']} "
              f"(page size {size}{'; by-row' if by_row else ''})")
        pageno = 0
        want_seen = 0
        errored = False
        while True:
            pageno += 1
            page_ids = [h for h in json.loads(ev(_HAND_TABLE_JS))["ids"] if h]
            if not page_ids:
                break
            # Fast skip for targeted runs (--gt = a small id set, often with
            # --ignore-done-tours): if this tournament's first page shares no
            # id with the target set, it holds none of them — don't page
            # through it. Harmless on full-GT runs (every hand is in gt_ids,
            # so page 1 always intersects and this never trips).
            if pageno == 1 and gt_ids is not None and not (
                    set(page_ids) & gt_ids):
                break
            want = [h for h in page_ids
                    if h not in done and (gt_ids is None or h in gt_ids)]
            want_seen += len(want)
            if want and by_row:
                # No usable title in the image: open each wanted hand by its
                # OWN row click. open_hand_by_id waits for a fresh, settled
                # scene, so the file we save IS the hand we clicked — correct
                # by construction, no arrow order or OCR involved.
                for hid in want:
                    if not open_hand_by_id(hid):
                        print(f"   p{pageno}: could not open {hid}")
                        errored = True
                        clear_overlay()
                        continue
                    if not bb_active():
                        set_bb_view()
                    png = grab_scene_bytes()
                    if png is None:
                        man.write(json.dumps(
                            {"hand_id": hid, "tournament": tr["name"],
                             "date": tr["date"], "page": pageno,
                             "by_row": True, "grab_failed": True},
                            ensure_ascii=False) + "\n")
                        man.flush()
                        close_modal()
                        continue
                    (imgs / f"{hid}.png").write_bytes(png)
                    done.add(hid)
                    total += 1
                    man.write(json.dumps(
                        {"hand_id": hid, "tournament": tr["name"],
                         "date": tr["date"], "page": pageno,
                         "by_row": True},
                        ensure_ascii=False) + "\n")
                    man.flush()
                    if total % 25 == 0:
                        print(f"   ... {total} new ({len(done)} total)")
                    close_modal()
            elif want:
                if not open_hand_by_id(page_ids[0]):
                    print(f"   p{pageno}: could not open first hand "
                          f"{page_ids[0]}")
                    errored = True
                    clear_overlay()
                else:
                    set_bb_view()
                    bb_persists: bool | None = None
                    prev_oid: str | None = None
                    # The arrow walks the page fast, but the in-modal order
                    # is NOT guaranteed to equal page_ids and the anchor's
                    # first frame can be stale (this caused Daily Hyper 1's
                    # whole-tournament off-by-one). So never trust k for the
                    # name: OCR each scene's own title bar — the only place
                    # the true id exists — and key the file by THAT.
                    for k in range(len(page_ids)):
                        if k > 0:
                            if not nav_right():
                                print(f"   p{pageno}: arrow stalled at k={k};"
                                      f" {len(page_ids)-k} unreached")
                                errored = True
                                break
                            if bb_persists is None:
                                bb_persists = bb_active()
                            if not bb_persists and not bb_active():
                                set_bb_view()
                        if not bb_active():
                            set_bb_view()
                        png = grab_scene_bytes()
                        if png is None:
                            continue
                        oid, _, _ = read_title_id(png, valid=gt_ids)
                        # Scene must advance every step; an unchanged id
                        # means a stale frame — wait a beat and re-read.
                        if oid is not None and oid == prev_oid:
                            time.sleep(0.7)
                            p2 = grab_scene_bytes()
                            if p2:
                                o2, _, _ = read_title_id(p2, valid=gt_ids)
                                if o2 and o2 != prev_oid:
                                    png, oid = p2, o2
                        if oid is None:  # one retry before giving up
                            time.sleep(0.5)
                            p2 = grab_scene_bytes()
                            if p2:
                                o2, _, _ = read_title_id(p2, valid=gt_ids)
                                if o2:
                                    png, oid = p2, o2
                        if oid is None:
                            man.write(json.dumps(
                                {"hand_id": page_ids[k],
                                 "tournament": tr["name"],
                                 "date": tr["date"], "page": pageno,
                                 "k": k, "ocr_failed": True},
                                ensure_ascii=False) + "\n")
                            man.flush()
                            continue
                        prev_oid = oid
                        if oid in done or (
                                gt_ids is not None and oid not in gt_ids):
                            continue
                        (imgs / f"{oid}.png").write_bytes(png)
                        done.add(oid)
                        total += 1
                        man.write(json.dumps(
                            {"hand_id": oid, "tournament": tr["name"],
                             "date": tr["date"], "page": pageno, "k": k,
                             "list_id": page_ids[k]},
                            ensure_ascii=False) + "\n")
                        man.flush()
                        if total % 25 == 0:
                            print(f"   ... {total} new "
                                  f"({len(done)} total)")
                    close_modal()
            if not hand_pager_next():
                break
        # Nothing was left to grab anywhere in this tournament and no step
        # failed → it is fully covered; record it so future resumes skip it
        # outright (no Game History open, no paging).
        if want_seen == 0 and not errored:
            done_tours.add(tkey)
            dt_path.write_text(json.dumps(sorted(done_tours),
                                          ensure_ascii=False))
    man.close()
    print(f"[done] {total} new images this run; "
          f"{len(list(imgs.glob('*.png')))} total in {imgs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
