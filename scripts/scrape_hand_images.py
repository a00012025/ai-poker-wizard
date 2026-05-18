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


def clear_overlay() -> bool:
    """Dismiss any open replay modal so the page is clickable.

    The Material dialog's OK button ignores synthesized element clicks (same
    as the BB button) and sits off the viewport bottom, but the dialog closes
    on Escape — so just press Escape until the overlay is gone.
    """
    for _ in range(10):
        if _overlay_gone():
            return True
        ab("press", "Escape")
        time.sleep(0.8)
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
              "!document.querySelector('app-hand-scene-modal') && "
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


def set_page_size_max() -> int:
    """Best-effort: pick the largest rows-per-page (fewer pages to walk).

    Never fatal — any leftover Material menu overlay is force-closed so a
    failed attempt can't block the run; the scraper then just paginates the
    default 10/page.
    """
    try:
        pos = ev("""(()=>{const b=document.querySelector(
          '.pagination .mat-menu-trigger'); if(!b) return '';
          b.scrollIntoView({block:'center'}); const r=b.getBoundingClientRect();
          return JSON.stringify({x:Math.round(r.x+r.width/2),
            y:Math.round(r.y+r.height/2)});})()""")
        if not pos:
            return 0
        o = json.loads(pos)
        raw_mouse_click(o["x"], o["y"])
        time.sleep(1)
        opts = ev("""(()=>{const it=[...document.querySelectorAll(
          '.cdk-overlay-container button,.mat-menu-panel button,'
          +'.mat-menu-item,[role=menuitem]')]
          .map(i=>{const r=i.getBoundingClientRect();
            const m=(i.innerText||'').match(/^\\s*(\\d+)\\s*$/);
            return (m&&r.width)?{n:+m[1],x:Math.round(r.x+r.width/2),
              y:Math.round(r.y+r.height/2)}:null;}).filter(Boolean);
          return JSON.stringify(it);})()""")
        items = json.loads(opts) if opts else []
        if not items:
            return 0
        best = max(items, key=lambda d: d["n"])
        raw_mouse_click(best["x"], best["y"])
        time.sleep(1.5)
        return best["n"]
    except Exception:
        return 0
    finally:
        # Force any leftover Material menu shut (Escape closes mat-menu)
        # so a failed attempt can't block subsequent clicks.
        ab("press", "Escape")
        time.sleep(0.3)
        ab("press", "Escape")
        time.sleep(0.3)


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


def pager_to_first() -> None:
    """Click the pager's prev arrow until back on page 1."""
    for _ in range(60):
        o = json.loads(ev("""(()=>{const p=document.querySelector('.pagination');
          if(!p) return JSON.stringify({done:true});
          const cur=p.querySelector('.current-page');
          if(cur && cur.textContent.trim()==='1')
            return JSON.stringify({done:true});
          const prev=[...p.querySelectorAll('button')].find(b=>
            b.querySelector('i.fa-angle-left'));
          if(!prev||prev.disabled) return JSON.stringify({done:true});
          prev.scrollIntoView({block:'center'});
          const r=prev.getBoundingClientRect();
          return JSON.stringify({x:Math.round(r.x+r.width/2),
            y:Math.round(r.y+r.height/2)});})()"""))
        if o.get("done"):
            return
        raw_mouse_click(o["x"], o["y"])
        time.sleep(0.6)


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
            for _ in range(20):
                time.sleep(0.25)
                if ev("!!document.getElementById('hand-scene')") in (
                        "true", True):
                    return True
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
        ab("press", "Escape")
        time.sleep(0.7)


def hand_pager_next() -> bool:
    """Advance the Game-History hand table to the next page.

    The pager is <div.pagination> with a prev button (i.fa-angle-left,
    disabled on page 1), a <span.current-page>, and a next button
    (i.fa-angle-right, disabled on the last page). Like the other PokerCraft
    controls it ignores synthesized clicks, so raw-mouse-click the next
    button's centre and confirm <span.current-page> incremented.
    """
    info = ev("""(()=>{const p=document.querySelector('.pagination');
      if(!p) return JSON.stringify({err:'no pager'});
      const cur=p.querySelector('.current-page');
      const next=[...p.querySelectorAll('button')].find(b=>
        b.querySelector('i.fa-angle-right'));
      if(!next) return JSON.stringify({err:'no next'});
      if(next.disabled || next.getAttribute('disabled')!==null)
        return JSON.stringify({last:true});
      next.scrollIntoView({block:'center'});
      const r=next.getBoundingClientRect();
      return JSON.stringify({cur:(cur?cur.textContent.trim():''),
        x:Math.round(r.x+r.width/2), y:Math.round(r.y+r.height/2)});})()""")
    o = json.loads(info)
    if o.get("err") or o.get("last"):
        return False
    raw_mouse_click(o["x"], o["y"])
    for _ in range(12):
        time.sleep(0.5)
        now = ev("(()=>{const c=document.querySelector('.pagination "
                 ".current-page');return c?c.textContent.trim():'';})()")
        if str(now) != str(o["cur"]):
            time.sleep(1.5)  # let the new page's rows render
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
        if not open_game_history(tr["i"]):
            print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']}: "
                  f"could not open Game History; skip")
            continue
        # 1. Maximise rows-per-page so there are far fewer pages to walk.
        size = set_page_size_max()
        # 2. Read the tournament's full hand-id order from the list (cheap:
        #    pure DOM + pager, no modals). Ends on the last page.
        ids = collect_list_ids()
        if not ids:
            print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']}: no hands")
            continue
        print(f"[t{ti+1}/{len(tours)}] {tr['date']} {tr['name']} "
              f"— {len(ids)} hands (page size {size or 10})")
        # 3. Page the list LEFT back to page 1 (re-selecting the tournament
        #    would toggle it OFF), then anchor the modal on hand #0.
        pager_to_first()
        if not open_hand_by_id(ids[0]):
            print(f"   could not open first hand {ids[0]}; skip tournament")
            clear_overlay()
            continue
        # 3. Walk every hand with the right arrow. image at step k == ids[k]
        #    (the tournament's hand order). Skipped hands are still stepped
        #    past so the image↔id alignment never drifts.
        for k, hid in enumerate(ids):
            if k > 0 and not nav_right():
                print(f"   arrow stalled at k={k}; {len(ids)-k} unreached")
                break
            if hid in done or (gt_ids is not None and hid not in gt_ids):
                continue
            if not set_bb_view():
                print(f"   {hid}: BB toggle failed (saving chip view)")
            if grab_scene(imgs / f"{hid}.png"):
                done.add(hid)
                total += 1
                man.write(json.dumps(
                    {"hand_id": hid, "tournament": tr["name"],
                     "date": tr["date"], "order_index": k},
                    ensure_ascii=False) + "\n")
                man.flush()
                if total % 25 == 0:
                    print(f"   ... {total} new images ({len(done)} total)")
        close_modal()
    man.close()
    print(f"[done] {total} new images this run; "
          f"{len(list(imgs.glob('*.png')))} total in {imgs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
