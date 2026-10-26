"""The production bot must keep OCR available when host GPU memory is full."""

from pathlib import Path

import yaml


def test_bot_disables_cuda_for_ocr():
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )
    assert compose["services"]["bot"]["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
