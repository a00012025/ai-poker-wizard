#!/usr/bin/env python3
"""Phase-3 de-risking probe: does re-reading stacks from the screenshot recover
the correct solver-depth bucket?

MEASUREMENT ONLY — no production wiring. Sends each sampled screenshot to a
strong vision model, asks for displayed stacks (bb) per seat + hero stack +
still-in seats + any all-in size, then scores two recovery methods vs GT:

  (a) substitution      — swap VLM stacks into inputs, re-run _compute_effective_bb
  (b) min-of-contesting — eff = min(contesting starting stacks, capped by shove)

Also: a cheap NON-VLM baseline (does a +-decimal / x10 correction of an existing
OCR stack reach the GT bucket?), Gemini-vs-GPT agreement on an overlap set, and
cost/latency.

Self-contained: builds the WRONG/CORRECT sample from the cache on first run,
caches it + raw VLM reads under data/effbb_reread/ (gitignored via data/).

Usage:
  python scripts/effbb_reread_probe.py --build             # build/refresh sample
  python scripts/effbb_reread_probe.py --gemini            # primary read
  python scripts/effbb_reread_probe.py --gpt-overlap 15    # GPT second opinion
  python scripts/effbb_reread_probe.py --score             # score from cached out
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()
import os

from effbb_metrics import bucket_match, depth_bucket
from ocr.n8_parser import _compute_effective_bb

IMGDIR = Path("data/hand_images/img")
CACHE = "data/effbb_cache/cache.jsonl"
ARTDIR = Path("data/effbb_reread")        # gitignored under data/
SAMPLE = str(ARTDIR / "sample.json")
OUT = str(ARTDIR / "reads.json")

VLM_PROMPT = """You are reading a poker tournament screenshot (GGPoker style). All stacks are shown in BIG BLINDS (bb). The table is at the top; a hand-history panel may be below it.

Return STRICT JSON only, no prose, with this exact schema:
{
  "seat_stacks_bb": [<number>, ...],   // every visible player's DISPLAYED stack in bb, as printed under/near each avatar
  "hero_stack_bb": <number>,           // the hero's own displayed stack in bb (hero seat is bottom-center, usually highlighted)
  "still_in_seats": [                  // players still contesting at the LAST action shown (not folded)
    {"stack_bb": <number>, "is_hero": <true|false>, "all_in_size_bb": <number|null>}
  ]
}

Rules:
- Report NUMBERS ONLY in bb (e.g. 31.5). Do NOT include chip-count text, names, or units.
- If a player is all-in and a shove size is shown, put it in all_in_size_bb, else null.
- still_in_seats must include the hero if the hero has not folded.
- Be precise about the decimal point and digit count (e.g. 9.13 not 91.3, 27.32 not 273.2).
JSON only."""


def _img_b64(hid):
    return base64.b64encode((IMGDIR / f"{hid}.png").read_bytes()).decode()


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # strip trailing commas
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return None


# ---------------- VLM callers ----------------
def call_gemini(hid, model):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    data = (IMGDIR / f"{hid}.png").read_bytes()
    t0 = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type="image/png"),
            types.Part(text=VLM_PROMPT),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        ),
    )
    dt = time.time() - t0
    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    toks = (usage.total_token_count if usage else None)
    return _extract_json(text), {"latency_s": round(dt, 2), "tokens": toks, "raw": text[:1200]}


def call_gpt(hid, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = _img_b64(hid)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    dt = time.time() - t0
    text = resp.choices[0].message.content or ""
    toks = resp.usage.total_tokens if resp.usage else None
    return _extract_json(text), {"latency_s": round(dt, 2), "tokens": toks, "raw": text[:1200]}


# ---------------- recompute helpers ----------------
def recompute(inp):
    res = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"],
    )
    if isinstance(res, tuple) and len(res) == 3:
        return res
    eff, hs = res
    return eff, hs, 1.0


def substitute_inputs(rec, vlm):
    """Build a new inputs dict with VLM stacks swapped in.

    Matches VLM seat_stacks_bb onto the existing named_stacks order (the OCR
    geometry stays; only the NUMBERS change). If counts differ, fall back to
    nearest-by-value replacement so geometry/positions are preserved."""
    inp = json.loads(json.dumps(rec["_inputs"]))
    vs = [float(x) for x in (vlm.get("seat_stacks_bb") or []) if _isnum(x)]
    hero_v = vlm.get("hero_stack_bb")
    if _isnum(hero_v):
        inp["hero_stack"] = float(hero_v)
    ns = inp.get("named_stacks") or []
    if vs and ns and len(vs) == len(ns):
        # positional swap (named_stacks is x/y ordered same as VLM left-right? not
        # guaranteed) -> instead match each named seat to nearest VLM value.
        used = [False] * len(vs)
        new_stacks = []
        for seat in ns:
            old = seat.get("stack")
            best, bi = None, None
            for i, v in enumerate(vs):
                if used[i]:
                    continue
                d = abs(v - (old or 0))
                if best is None or d < best:
                    best, bi = d, i
            if bi is not None:
                used[bi] = True
                seat["stack"] = vs[bi]
                new_stacks.append(vs[bi])
        if new_stacks:
            inp["stacks"] = new_stacks
    elif vs:
        inp["stacks"] = vs
    return inp


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def min_of_contesting(rec, vlm):
    """eff = min over still-in seats of (starting stack, capped by any shove)."""
    sins = vlm.get("still_in_seats") or []
    vals = []
    for s in sins:
        if not isinstance(s, dict):
            continue
        st = s.get("stack_bb")
        ai = s.get("all_in_size_bb")
        cand = None
        if _isnum(st):
            cand = float(st)
        if _isnum(ai):
            cand = float(ai) if cand is None else min(cand, float(ai))
        if cand is not None and cand >= 1.0:
            vals.append(cand)
    if len(vals) < 2:
        # need at least hero + 1 villain; fall back to hero vs smallest seat stack
        hero_v = vlm.get("hero_stack_bb")
        seats = [float(x) for x in (vlm.get("seat_stacks_bb") or []) if _isnum(x) and x >= 1.0]
        if _isnum(hero_v) and seats:
            return min(float(hero_v), min(seats))
        return None
    return min(vals)


# ---------------- non-VLM cheap baseline ----------------
def cheap_correction_recoverable(rec):
    """Could a +-decimal / x10 / /10 correction of SOME existing OCR stack land
    in the GT bucket? (No re-read, pure arithmetic on cached OCR numbers.)"""
    gb = depth_bucket(rec["gt_eff"])
    cands = list(rec["_inputs"]["stacks"] or [])
    h = rec["_inputs"]["hero_stack"]
    if _isnum(h):
        cands.append(h)
    for s in cands:
        if not _isnum(s) or s <= 0:
            continue
        # digit-slip family: x10, /10, +-10, +-decimal-point shift
        for corrected in (s * 10, s / 10, s + 10, s - 10,
                          s + 1, s - 1):
            if corrected > 0 and depth_bucket(corrected) == gb:
                return True
    return False


# ---------------- sample build ----------------
def build_sample(n_wrong=60, n_correct=20):
    """Split hero-active hands (w/ image) into WRONG/CORRECT by current bucket
    match, then stride-sample for diversity. Writes SAMPLE."""
    from effbb_metrics import hero_folded_preflop
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    wrong, correct = [], []
    for r in rows:
        gt = r.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in r:
            continue
        if hero_folded_preflop(gt) is not False:
            continue
        hid = r["hand_id"]
        if not (IMGDIR / f"{hid}.png").exists():
            continue
        p_eff, _, _ = recompute(r["inputs"])
        rec = {"hid": hid, "gt_eff": ge, "p_eff": p_eff,
               "gt_stacks": gt.get("stacks_bb"), "preflop": gt.get("preflop_actions"),
               "num_players": gt.get("num_players"),
               "hero_position": r["inputs"]["hero_position"],
               "hero_stack": r["inputs"]["hero_stack"],
               "ocr_stacks": r["inputs"]["stacks"]}
        (correct if bucket_match(p_eff, ge) else wrong).append(rec)

    def stride(lst, n):
        if len(lst) <= n:
            return lst
        step = len(lst) / n
        return [lst[int(i * step)] for i in range(n)]

    ARTDIR.mkdir(parents=True, exist_ok=True)
    sample = {"wrong": stride(wrong, n_wrong), "correct": stride(correct, n_correct)}
    json.dump(sample, open(SAMPLE, "w"), ensure_ascii=False, indent=1)
    print(f"hero-active w/ image: wrong={len(wrong)} correct={len(correct)}; "
          f"sampled wrong={len(sample['wrong'])} correct={len(sample['correct'])}",
          file=sys.stderr)
    return sample


# ---------------- driver ----------------
def load_sample():
    if not Path(SAMPLE).exists():
        return build_sample()
    return json.load(open(SAMPLE))


def attach_inputs(recs):
    """Re-attach raw inputs (the sample json dropped them to stay small? no — it
    kept ocr_stacks/named_stacks but not columns). Reload from cache by hid."""
    by_hid = {}
    for l in open(CACHE, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        by_hid[r["hand_id"]] = r.get("inputs")
    for rec in recs:
        rec["_inputs"] = by_hid[rec["hid"]]


def run_reads(reader, model, recs, label):
    out = {}
    if Path(OUT).exists():
        out = json.load(open(OUT))
    for i, rec in enumerate(recs):
        hid = rec["hid"]
        key = f"{label}:{hid}"
        if key in out and out[key].get("vlm") is not None:
            continue
        try:
            vlm, meta = reader(hid, model)
        except Exception as e:
            vlm, meta = None, {"error": str(e)[:300]}
        out[key] = {"vlm": vlm, "meta": meta}
        json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
        print(f"[{label} {i+1}/{len(recs)}] {hid} "
              f"lat={meta.get('latency_s')}s tok={meta.get('tokens')} "
              f"ok={'Y' if vlm else 'N'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the sample")
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--gpt-overlap", type=int, default=0)
    ap.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--gpt-model", default="gpt-5.4")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.build:
        build_sample()
    sample = load_sample()
    allrecs = sample["wrong"] + sample["correct"]
    attach_inputs(allrecs)

    if args.gemini:
        run_reads(call_gemini, args.gemini_model, allrecs, "gemini")
    if args.gpt_overlap:
        # overlap on first N WRONG hands (most informative)
        run_reads(call_gpt, args.gpt_model, sample["wrong"][:args.gpt_overlap], "gpt")
    if args.score:
        score(sample, allrecs)


def score(sample, allrecs):
    out = json.load(open(OUT))
    by_hid = {r["hid"]: r for r in allrecs}

    def vlm_for(label, hid):
        e = out.get(f"{label}:{hid}")
        return (e or {}).get("vlm")

    # ---- main recovery on WRONG, damage on CORRECT (gemini) ----
    for setname, recs in (("WRONG", sample["wrong"]), ("CORRECT", sample["correct"])):
        n = sub_ok = moc_ok = union_ok = vlm_miss = cheap_ok = 0
        broke_sub = broke_moc = 0
        details = []
        for rec in recs:
            n += 1
            if cheap_correction_recoverable(rec):
                cheap_ok += 1
            vlm = vlm_for("gemini", rec["hid"])
            if not vlm:
                vlm_miss += 1
                details.append({"hid": rec["hid"], "gt": rec["gt_eff"],
                                "cur": rec["p_eff"], "sub": None, "moc": None,
                                "s_ok": None, "m_ok": None})
                continue
            sub_inp = substitute_inputs(rec, vlm)
            try:
                s_eff, _, _ = recompute(sub_inp)
            except Exception:
                s_eff = None
            m_eff = min_of_contesting(rec, vlm)
            s_ok = s_eff is not None and bucket_match(s_eff, rec["gt_eff"])
            m_ok = m_eff is not None and bucket_match(m_eff, rec["gt_eff"])
            sub_ok += s_ok
            moc_ok += m_ok
            union_ok += (s_ok or m_ok)
            # damage = a method emitted a value but it is in the WRONG bucket
            broke_sub += (s_eff is not None and not s_ok)
            broke_moc += (m_eff is not None and not m_ok)
            details.append({"hid": rec["hid"], "gt": rec["gt_eff"],
                            "cur": rec["p_eff"],
                            "sub": round(s_eff, 1) if s_eff else None,
                            "moc": round(m_eff, 1) if m_eff else None,
                            "s_ok": s_ok, "m_ok": m_ok})
        print(f"\n=== {setname} (n={n}) ===")
        print(f"  VLM read failures (no JSON): {vlm_miss}")
        scored = n - vlm_miss
        if setname == "WRONG":
            print(f"  (a) substitution recovered:      {sub_ok}/{scored} = {pct(sub_ok, scored)}")
            print(f"  (b) min-of-contesting recovered: {moc_ok}/{scored} = {pct(moc_ok, scored)}")
            print(f"  (a OR b) union recovered:        {union_ok}/{scored} = {pct(union_ok, scored)}")
            print(f"  cheap +-decimal/x10 recoverable: {cheap_ok}/{n} = {pct(cheap_ok, n)}")
        else:
            print(f"  (a) substitution BROKE:      {broke_sub}/{scored} = {pct(broke_sub, scored)}")
            print(f"  (b) min-of-contesting BROKE: {broke_moc}/{scored} = {pct(broke_moc, scored)}")
        # a few examples
        print("  examples (hid, gt, cur_pred, sub_eff[ok], moc_eff[ok]):")
        for d in details[:8]:
            print(f"    {d['hid']} gt={d['gt']} cur={d['cur']} "
                  f"sub={d['sub']}[{d['s_ok']}] moc={d['moc']}[{d['m_ok']}]")

    # ---- cost / latency ----
    lats, toks = [], []
    for k, v in out.items():
        if not k.startswith("gemini:"):
            continue
        m = v.get("meta") or {}
        if m.get("latency_s"):
            lats.append(m["latency_s"])
        if m.get("tokens"):
            toks.append(m["tokens"])
    if lats:
        print(f"\n=== Gemini cost/latency ===")
        print(f"  latency: mean={sum(lats)/len(lats):.2f}s p50={sorted(lats)[len(lats)//2]:.2f}s "
              f"max={max(lats):.2f}s  (n={len(lats)})")
        if toks:
            print(f"  tokens:  mean={sum(toks)/len(toks):.0f} total={sum(toks)}")

    # ---- Gemini vs GPT agreement (overlap) ----
    overlap = [k.split(":", 1)[1] for k in out if k.startswith("gpt:")]
    if overlap:
        print(f"\n=== Gemini vs GPT agreement (overlap n={len(overlap)}) ===")
        agree_sub_bucket = agree_moc_bucket = both_recover = 0
        glats, gtoks = [], []
        for hid in overlap:
            rec = by_hid[hid]
            gv = vlm_for("gemini", hid)
            pv = vlm_for("gpt", hid)
            gm = (out.get(f"gpt:{hid}") or {}).get("meta") or {}
            if gm.get("latency_s"):
                glats.append(gm["latency_s"])
            if gm.get("tokens"):
                gtoks.append(gm["tokens"])
            if not gv or not pv:
                continue
            ge, _, _ = recompute(substitute_inputs(rec, gv))
            pe, _, _ = recompute(substitute_inputs(rec, pv))
            if ge is not None and pe is not None and bucket_match(ge, pe):
                agree_sub_bucket += 1
            gmoc, pmoc = min_of_contesting(rec, gv), min_of_contesting(rec, pv)
            if gmoc and pmoc and bucket_match(gmoc, pmoc):
                agree_moc_bucket += 1
            g_ok = (ge and bucket_match(ge, rec["gt_eff"])) or (gmoc and bucket_match(gmoc, rec["gt_eff"]))
            p_ok = (pe and bucket_match(pe, rec["gt_eff"])) or (pmoc and bucket_match(pmoc, rec["gt_eff"]))
            if g_ok and p_ok:
                both_recover += 1
        print(f"  substitution-bucket agreement: {agree_sub_bucket}/{len(overlap)}")
        print(f"  moc-bucket agreement:          {agree_moc_bucket}/{len(overlap)}")
        print(f"  both recover GT (a OR b):      {both_recover}/{len(overlap)}")
        if glats:
            print(f"  GPT latency mean={sum(glats)/len(glats):.2f}s tokens mean={sum(gtoks)/len(gtoks):.0f}" if gtoks else f"  GPT latency mean={sum(glats)/len(glats):.2f}s")


def pct(a, b):
    return f"{(100*a/b):.1f}%" if b else "n/a"


if __name__ == "__main__":
    main()
