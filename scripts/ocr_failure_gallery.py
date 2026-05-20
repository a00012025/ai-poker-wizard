#!/usr/bin/env python3
"""Build a per-failure-mode image gallery from ocr_precision diffs.

For each failure bucket (hero_cards_wrong, hero_cards_missing,
preflop_action_types_wrong, position_wrong, board_wrong, parse_none),
sample up to N hands and emit a montage PNG with the original image
annotated with the OCR vs GT diff. Harry can eyeball each montage to
classify root causes.

Usage:
    python scripts/ocr_failure_gallery.py \
        --diffs data/ocr_precision_full/diffs.jsonl \
        --out data/ocr_precision_full/gallery \
        --per-mode 6
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(img_path: Path, rec: dict, width: int) -> Image.Image:
    """Return img scaled to `width`, with an annotation strip below."""
    img = Image.open(img_path).convert("RGB")
    scale = width / img.width
    img = img.resize((width, int(img.height * scale)), Image.LANCZOS)

    parsed = rec.get("parsed") or {}
    gt = rec.get("gt") or {}
    conf = rec.get("confidence", 0.0)
    card_conf = rec.get("card_confidence", 0.0)
    mode = rec.get("failure_mode") or ("parse_none" if rec.get("parsed_none")
                                       else "error" if rec.get("error") else "?")
    fields = rec.get("fields") or {}

    def mark(k: str, v) -> str:
        ok = fields.get(k)
        if ok is True:
            return f"OK  {k}: {v}"
        if ok is False:
            return f"X   {k}: {v}"
        return f"-   {k}: {v}"

    lines = [
        f"{rec['hand_id']}    mode={mode}   conf={conf:.2f}  card_conf={card_conf:.2f}",
        f"OCR  hero={parsed.get('hero_hand')}  pos={parsed.get('hero_position')}  "
        f"players={parsed.get('players_at_table')}  eff_bb={parsed.get('effective_bb')}",
        f"     pre={parsed.get('preflop_actions')}",
        f"     streets={rec.get('parsed_streets')}",
        f"GT   hero={gt.get('hero_hand')}  pos={gt.get('hero_position')}  "
        f"players={gt.get('num_players')}/{gt.get('table_size')}  eff_bb={gt.get('effective_bb')}",
        f"     pre={gt.get('preflop_actions')}",
        f"     streets={rec.get('gt_streets')}",
    ]
    font = _font(14)
    line_h = 18
    pad = 6
    strip_h = line_h * len(lines) + pad * 2
    out = Image.new("RGB", (width, img.height + strip_h), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, img.height, width, img.height + strip_h), fill="black")
    for i, line in enumerate(lines):
        color = "white"
        if line.startswith("X   "):
            color = "#ff6b6b"
        if line.startswith("OK  "):
            color = "#7bed9f"
        draw.text((pad, img.height + pad + i * line_h), line[:160],
                  fill=color, font=font)
    return out


def montage(images: list[Image.Image], cols: int = 3, gap: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (10, 10), "white")
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (w * cols + gap * (cols - 1),
                             h * rows + gap * (rows - 1)), "white")
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        out.paste(im, (c * (w + gap), r * (h + gap)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-mode", type=int, default=6)
    ap.add_argument("--tile-width", type=int, default=440)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_mode: dict[str, list[dict]] = defaultdict(list)
    with open(args.diffs, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            mode = (rec.get("failure_mode")
                    or ("parse_none" if rec.get("parsed_none") else None)
                    or ("error" if rec.get("error") else "other"))
            by_mode[mode].append(rec)

    for mode, recs in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        picks = recs[: args.per_mode]
        tiles = []
        for rec in picks:
            img_path = Path(rec.get("image") or "")
            if not img_path.exists():
                continue
            tiles.append(annotate(img_path, rec, args.tile_width))
        if not tiles:
            continue
        m = montage(tiles, cols=args.cols)
        out_path = out_dir / f"{mode}.png"
        m.save(out_path, "PNG", optimize=True)
        print(f"  {mode:32}  {len(recs):4} fails  -> {out_path}  "
              f"(showing {len(tiles)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
