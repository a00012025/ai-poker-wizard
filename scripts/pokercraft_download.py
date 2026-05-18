#!/usr/bin/env python3
"""Bulk-download GGPoker Game Histories from PokerCraft for ground truth.

Mechanism (all steps verified against my.pokercraft.com):

  PokerCraft's REST API encrypts JSON responses, but the user-facing
  "Download Game Histories" button produces a plain ZIP of standard GGPoker
  HH text. Clicking it opens an /embedded/download/hand/<job> page that
  assembles the ZIP as an in-page **blob: URL**. We fetch that blob from
  inside the page (cookies + decryption already handled by the SPA),
  base64-exfiltrate it via `agent-browser eval`, and unzip locally. This
  sidesteps the encrypted API, request signing, and Chrome's random
  download directory entirely.

Preconditions:
  - `agent-browser` is connected to a Chrome with an authenticated
    PokerCraft session, and a tab is open at my.pokercraft.com/tournament
    with the desired date filter already applied (e.g. Custom Range = last
    30 days) and the list shown.

Usage:
  python scripts/pokercraft_download.py --out data/pokercraft_corpus --batch 25
  python scripts/pokercraft_download.py --out data/pc --limit 2   # smoke test
  python scripts/pokercraft_download.py --out data/pc --build      # + build GT
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

AB = "agent-browser"


def ab(*args: str, timeout: int = 60) -> str:
    """Run an agent-browser command, return stdout (stripped)."""
    r = subprocess.run([AB, *args], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def ab_eval(js: str, timeout: int = 60) -> str:
    out = ab("eval", js, timeout=timeout)
    if out.startswith('"') and out.endswith('"'):
        try:
            return json.loads(out)
        except Exception:
            return out
    return out


def list_tabs() -> list[dict]:
    raw = ab("tab")
    tabs = []
    for line in raw.splitlines():
        line = line.strip()
        # "→ [t5] PokerCraft - https://my.pokercraft.com/..."
        if "[t" not in line:
            continue
        cur = line.startswith("→")
        tid = line[line.index("[t") + 1 : line.index("]")]
        url = line.split(" - ")[-1].strip()
        tabs.append({"id": tid, "url": url, "current": cur})
    return tabs


def _rows_on_current() -> int:
    try:
        return int(ab_eval(
            "document.querySelectorAll('table')[0]"
            "?.querySelectorAll('tbody tr').length || 0"))
    except Exception:
        return 0


def pick_list_tab(explicit: str | None = None) -> str:
    """Resolve the tournament-list tab.

    With --tab, pin to it (no iteration → can't race the user closing tabs).
    Otherwise scan PokerCraft tabs and take the one with the most rows, then
    re-verify after switching back so a mid-scan tab change can't slip through.
    """
    if explicit:
        tid = explicit if explicit.startswith("t") else f"t{explicit}"
        ab("tab", tid)
        time.sleep(1)
        n = _rows_on_current()
        if n <= 0:
            sys.exit(f"--tab {tid}: no tournament rows (wrong tab or list not "
                     f"shown). Open the filtered list there first.")
        print(f"[list tab] {tid} (pinned) with {n} tournament rows")
        return tid

    best, best_rows = None, -1
    for t in list_tabs():
        if "my.pokercraft.com/tournament" not in t["url"]:
            continue
        ab("tab", t["id"])
        time.sleep(1)
        n = _rows_on_current()
        if n > best_rows:
            best, best_rows = t["id"], n
    if best is None or best_rows <= 0:
        sys.exit("No my.pokercraft.com/tournament tab with a populated list. "
                 "Open the filtered tournament list, or pass --tab.")
    ab("tab", best)
    time.sleep(1)
    confirm = _rows_on_current()
    if confirm <= 0:
        sys.exit(f"Picked {best} but it now has no rows (tab changed mid-scan). "
                 f"Re-run with explicit --tab.")
    print(f"[list tab] {best} with {confirm} tournament rows")
    return best


# Read the tournament list: index, date, name, current selection state.
_ROWS_JS = """(()=>{
  const tbl=document.querySelectorAll('table')[0];
  const rows=[...tbl.querySelectorAll('tbody tr')];
  return JSON.stringify(rows.map((tr,i)=>{
    const c=[...tr.children].map(td=>td.innerText.trim());
    const cb=tr.querySelector('input[type=checkbox]');
    return {i, date:c[1]||'', name:c[2]||'', checked: cb? cb.checked:false};
  }));
})()"""


def get_rows() -> list[dict]:
    return json.loads(ab_eval(_ROWS_JS))


def set_selection(indices: set[int]) -> int:
    """Tick exactly the given row checkboxes (untick the rest). Returns count."""
    js = """(idx=>{
      const want=new Set(idx);
      const tbl=document.querySelectorAll('table')[0];
      const rows=[...tbl.querySelectorAll('tbody tr')];
      let n=0;
      rows.forEach((tr,i)=>{
        const cb=tr.querySelector('input[type=checkbox]');
        if(!cb) return;
        if(want.has(i)!==cb.checked) cb.click();
        if(want.has(i)) n++;
      });
      return n;
    })(%s)""" % json.dumps(sorted(indices))
    return int(ab_eval(js))


# Button innerText is "Download\n47 Game Histories" — normalize whitespace
# before matching (JS "." does not cross newlines).
_FIND_DL_BTN = """[...document.querySelectorAll('button')].find(x=>{
  const t=(x.innerText||'').replace(/\\s+/g,' ').trim();
  return /Download\\s+[\\d,]+\\s+Game Histor/i.test(t);})"""


def download_button_count() -> int:
    js = ("(()=>{const b=%s; if(!b) return -1;"
          "const t=b.innerText.replace(/\\s+/g,' ');"
          "const m=t.match(/Download\\s+([\\d,]+)/i);"
          "return m? parseInt(m[1].replace(/,/g,'')):0;})()") % _FIND_DL_BTN
    return int(ab_eval(js))


import re as _re


def click_download() -> bool:
    """Click the download button with a *trusted* synthesized click.

    The embedded download opens via a user-gesture popup; a programmatic
    element.click() from eval is ignored. agent-browser's click performs a
    real Playwright pointer click, so resolve the button's snapshot ref and
    click that. The a11y tree flattens the button's newline to a space, so
    it shows as: button "Download 47 Game Histories" [ref=eNN].
    """
    snap = ab("snapshot", "-i", "-c")
    ref = None
    for line in snap.splitlines():
        if _re.search(r"Download\s+[\d,]+\s+Game Histor", line, _re.I):
            m = _re.search(r"ref=(e\d+)", line)
            if m:
                ref = m.group(1)
                break
    if not ref:
        print("  ! download button ref not found in snapshot")
        return False
    ab("click", f"@{ref}")
    return True


def grab_blob_zip(list_tab: str, dest_zip: Path, wait: int = 90) -> bool:
    """Wait for the embedded download tab, fetch its blob ZIP, save it."""
    dl_tab = None
    for _ in range(wait):
        tabs = list_tabs()
        for t in tabs:
            if "/embedded/download/hand/" in t["url"]:
                dl_tab = t["id"]
                break
        if dl_tab:
            break
        # Fallback: same-tab navigation, or an in-page blob dialog.
        if "/embedded/download/hand/" in (ab_eval("location.href") or ""):
            dl_tab = list_tab
            break
        if ab_eval("[...document.querySelectorAll('a')]"
                   ".some(a=>/^blob:/.test(a.href))") in ("true", True):
            dl_tab = list_tab
            break
        time.sleep(1)
    if not dl_tab:
        print("  ! no embedded download tab/blob appeared")
        return False

    if dl_tab != list_tab:
        ab("tab", dl_tab)
    b64 = ""
    for _ in range(wait):
        b64 = ab_eval("""(async()=>{
          const a=[...document.querySelectorAll('a')].find(x=>/^blob:/.test(x.href));
          if(!a) return '';
          const r=await fetch(a.href); const u=new Uint8Array(await r.arrayBuffer());
          let s=''; for(let i=0;i<u.length;i++) s+=String.fromCharCode(u[i]);
          return btoa(s);
        })()""", timeout=120)
        if b64:
            break
        time.sleep(1)

    ok = False
    if b64:
        data = base64.b64decode(b64)
        dest_zip.write_bytes(data)
        ok = zipfile.is_zipfile(dest_zip)
        print(f"  ZIP {len(data)} bytes -> {dest_zip.name} ({'valid' if ok else 'INVALID'})")
    else:
        print("  ! blob never materialized")

    if dl_tab != list_tab:
        # Separate download tab: close it, return to the list.
        ab("tab", dl_tab)
        ab("tab", "close")
        ab("tab", list_tab)
        time.sleep(1)
    else:
        # Same-tab navigation: go back to restore the tournament list.
        ab("back")
        time.sleep(2)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pokercraft_corpus")
    ap.add_argument("--batch", type=int, default=25, help="Tournaments per ZIP")
    ap.add_argument("--limit", type=int, default=0, help="Max tournaments (0=all)")
    ap.add_argument("--build", action="store_true",
                    help="Run build_ground_truth.py over the corpus when done")
    ap.add_argument("--tab", default=None,
                    help="Pin to this agent-browser tab id (e.g. t4); skips "
                         "tab scanning so it can't race tab changes")
    args = ap.parse_args()

    out = Path(args.out)
    corpus = out / "hh"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    list_tab = pick_list_tab(args.tab)
    rows = get_rows()
    print(f"[rows] {len(rows)} tournaments in filtered list")

    todo = [r for r in rows if f"{r['date']}|{r['name']}" not in manifest]
    already = len(rows) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[todo] {len(todo)} to download this run "
          f"({already} already in manifest, limit={args.limit or 'none'})")

    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    for bi, batch in enumerate(batches):
        idx = {r["i"] for r in batch}
        sel = set_selection(idx)
        cnt = download_button_count()
        print(f"[batch {bi + 1}/{len(batches)}] selected {sel} tournaments, "
              f"button says {cnt} hands")
        if sel == 0:
            continue
        if not click_download():
            print("  ! could not click download button; skipping batch")
            continue
        zip_path = out / f"batch_{int(time.time())}_{bi}.zip"
        if not grab_blob_zip(list_tab, zip_path):
            print("  ! batch failed; leaving in manifest untouched, continuing")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(corpus)
            extracted = zf.namelist()
        for r in batch:
            manifest[f"{r['date']}|{r['name']}"] = {
                "zip": zip_path.name, "ts": time.time()}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"  extracted {len(extracted)} files; manifest now "
              f"{len(manifest)} tournaments")
        # Clear selection before the next batch.
        set_selection(set())
        time.sleep(1)

    txts = list(corpus.glob("*.txt"))
    print(f"\n[done] corpus has {len(txts)} HH files in {corpus}")

    if args.build and txts:
        print("[build] running build_ground_truth.py ...")
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("build_ground_truth.py")),
             str(corpus), "-o", str(out / "ground_truth")],
            check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
