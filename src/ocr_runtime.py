"""Choose a usable OCR device for each screenshot."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


_log = logging.getLogger(__name__)
_CPU_WORKER = Path(__file__).resolve().parents[1] / "scripts" / "ocr_cpu_worker.py"


def select_ocr_mode(
    *, cuda_available: bool, free_mib: float | None, min_free_mib: float = 2048
) -> str:
    """Return GPU, local CPU, or isolated CPU according to current headroom."""
    if not cuda_available:
        return "local_cpu"
    if free_mib is None or free_mib < min_free_mib:
        return "cpu_worker"
    return "gpu"


def current_ocr_mode() -> str:
    """Inspect current GPU headroom; failures select the isolated CPU path."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "local_cpu"
    except Exception as error:
        _log.warning("Cannot read CUDA headroom; using CPU OCR: %s", error)
        return "cpu_worker"

    # NVML inspection through nvidia-smi avoids creating a CUDA context just
    # to decide whether there is enough room to create one for EasyOCR.
    try:
        status = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5,
        )
        free_mib = float(status.stdout.decode().strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        _log.warning("Cannot read GPU free memory; using CPU OCR: %s", error)
        return "cpu_worker"

    min_free_mib = float(os.getenv("OCR_GPU_MIN_FREE_MIB", "2048"))
    return select_ocr_mode(
        cuda_available=True, free_mib=free_mib, min_free_mib=min_free_mib
    )


def _is_cuda_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def parse_with_device_fallback(
    image_bytes: bytes,
    mode: str,
    local_parse: Callable[[bytes], dict],
    cpu_parse: Callable[[bytes], dict],
) -> dict:
    """Run GPU when ready; retry the whole image on isolated CPU after OOM."""
    if mode == "cpu_worker":
        return cpu_parse(image_bytes)
    try:
        return local_parse(image_bytes)
    except Exception as error:
        if mode != "gpu" or not _is_cuda_oom(error):
            raise
        _log.warning("GPU OCR ran out of memory; retrying screenshot on CPU")
        return cpu_parse(image_bytes)


def run_cpu_worker(image_bytes: bytes, *, runner=subprocess.run) -> dict:
    """Parse in a fresh process where PyTorch cannot see CUDA.

    EasyOCR pins DataLoader memory whenever CUDA is visible, even with a CPU
    reader. A separate process with CUDA hidden avoids that hidden GPU use.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = runner(
        [sys.executable, str(_CPU_WORKER)],
        input=image_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=240,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"CPU OCR worker failed: {detail}")
    return json.loads(result.stdout)


def parse_n8_adaptive(image_bytes: bytes) -> dict:
    """Production entry point used for each Telegram N8 screenshot."""
    from ocr.n8_parser import parse_n8_screenshot

    mode = current_ocr_mode()
    _log.info("N8 OCR device: %s", mode)
    return parse_with_device_fallback(
        image_bytes, mode, parse_n8_screenshot, run_cpu_worker
    )
