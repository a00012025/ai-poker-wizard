"""One screenshot per process, with CUDA hidden before PyTorch is imported."""

from contextlib import redirect_stdout
import json
import sys


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    from ocr.n8_parser import parse_n8_screenshot

    # EasyOCR may print model-download progress to stdout. Keep the protocol
    # output as one JSON object while retaining diagnostics on stderr.
    with redirect_stdout(sys.stderr):
        result = parse_n8_screenshot(sys.stdin.buffer.read())
    sys.stdout.write(json.dumps(result, default=_json_default))


if __name__ == "__main__":
    main()
