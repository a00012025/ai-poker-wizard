from __future__ import annotations

from pathlib import Path


from ocr.diagnostics_dump import dump_hand


def test_dump_hand_writes_overlays(tmp_path):
    img = Path("data/hand_images/img/TM5846885824.png")
    out = tmp_path / "TM5846885824"
    dump_hand(img.read_bytes(), out_dir=out, hand_id="TM5846885824")

    assert (out / "original.png").exists()
    assert (out / "table_with_button.png").exists()
    assert any(p.name.startswith("col_") for p in out.glob("col_*.png"))
