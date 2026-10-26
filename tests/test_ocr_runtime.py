"""Image OCR chooses GPU when usable and completes on CPU under contention."""

from src.ocr_runtime import (
    current_ocr_mode,
    parse_with_device_fallback,
    run_cpu_worker,
    select_ocr_mode,
)


def test_select_ocr_mode_uses_gpu_only_with_enough_free_memory():
    assert select_ocr_mode(cuda_available=False, free_mib=None) == "local_cpu"
    assert select_ocr_mode(cuda_available=True, free_mib=128) == "cpu_worker"
    assert select_ocr_mode(cuda_available=True, free_mib=4096) == "gpu"


def test_device_selection_rechecks_free_memory_for_each_image(monkeypatch):
    from types import SimpleNamespace
    import torch
    from src import ocr_runtime

    available_mib = iter((4096, 128, 4096))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ocr_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{next(available_mib)}\n".encode()
        ),
    )

    assert [current_ocr_mode() for _ in range(3)] == [
        "gpu", "cpu_worker", "gpu"
    ]


def test_gpu_parse_is_used_when_available():
    calls = []

    def local_parse(image):
        calls.append("gpu")
        return {"hand": {"hero_hand": "TsTc"}}

    def cpu_parse(image):
        calls.append("cpu")
        return {"hand": {"hero_hand": "AdQd"}}

    result = parse_with_device_fallback(b"image", "gpu", local_parse, cpu_parse)

    assert result["hand"]["hero_hand"] == "TsTc"
    assert calls == ["gpu"]


def test_low_gpu_memory_uses_cpu_without_attempting_gpu():
    calls = []

    def local_parse(image):
        calls.append("gpu")
        raise AssertionError("GPU should not run")

    def cpu_parse(image):
        calls.append("cpu")
        return {"hand": {"hero_hand": "AdQd"}}

    result = parse_with_device_fallback(b"image", "cpu_worker", local_parse, cpu_parse)

    assert result["hand"]["hero_hand"] == "AdQd"
    assert calls == ["cpu"]


def test_cuda_oom_retries_whole_image_in_isolated_cpu_process():
    calls = []

    def local_parse(image):
        calls.append("gpu")
        raise RuntimeError("CUDA error: out of memory")

    def cpu_parse(image):
        calls.append("cpu")
        return {"hand": {"hero_hand": "9h8h", "hero_position": "HJ"}}

    result = parse_with_device_fallback(b"image", "gpu", local_parse, cpu_parse)

    assert result["hand"] == {"hero_hand": "9h8h", "hero_position": "HJ"}
    assert calls == ["gpu", "cpu"]


def test_non_gpu_error_is_not_hidden():
    def local_parse(image):
        raise ValueError("bad image")

    try:
        parse_with_device_fallback(b"image", "gpu", local_parse, lambda image: {})
    except ValueError as error:
        assert str(error) == "bad image"
    else:
        raise AssertionError("non-GPU errors must propagate")


def test_cpu_worker_starts_with_cuda_hidden(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b'{"hand":{"hero_hand":"AdQd"}}', stderr=b"")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = run_cpu_worker(b"image", runner=fake_run)

    assert result["hand"]["hero_hand"] == "AdQd"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert seen["input"] == b"image"
