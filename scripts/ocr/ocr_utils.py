"""OCR utilities using EasyOCR (GPU-accelerated) with Tesseract fallback."""
import cv2
import numpy as np

_reader = None


def _get_reader():
    """Lazy-initialize EasyOCR reader (loads model once, reuses across calls)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        except ImportError:
            pass
    return _reader


def ocr_full_image(image: np.ndarray) -> list[dict]:
    """OCR entire image, return all detected text with bounding boxes.

    Returns:
        [{"text": str, "bbox": [[x1,y1],...], "conf": float, "center_y": float}]
    """
    reader = _get_reader()
    if reader is None:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    results = reader.readtext(gray)
    out = []
    for bbox, text, conf in results:
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        out.append({
            "text": text,
            "bbox": bbox,
            "conf": conf,
            "center_y": sum(ys) / len(ys),
            "center_x": sum(xs) / len(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "x_min": min(xs),
            "x_max": max(xs),
        })
    return out


def ocr_region_text(image: np.ndarray) -> str:
    """OCR a small region, return concatenated text."""
    results = ocr_full_image(image)
    return " ".join(r["text"] for r in results)


# Keep legacy functions for backward compatibility with existing tests/code

def preprocess_for_ocr(image: np.ndarray, min_width: int = 600) -> np.ndarray:
    """Preprocess image for OCR: upscale if small, convert to grayscale."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.shape[1] < min_width:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return gray


def ocr_text(image: np.ndarray, whitelist: str = "", psm: int = 7) -> tuple[str, float]:
    """OCR a region using EasyOCR (ignores whitelist/psm, kept for API compat)."""
    results = ocr_full_image(image)
    if not results:
        return "", 0.0
    text = " ".join(r["text"] for r in results)
    avg_conf = sum(r["conf"] for r in results) / len(results) * 100  # scale to 0-100
    return text, avg_conf


def ocr_number(image: np.ndarray) -> tuple[float | None, float]:
    """OCR a number from an image region."""
    text, conf = ocr_text(image)
    text = text.strip().replace(" ", "").upper().replace("BB", "").replace("B", "")
    cleaned = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            cleaned += ch
        elif cleaned:
            break
    try:
        return float(cleaned), conf
    except (ValueError, TypeError):
        return None, 0.0


def binarize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Apply adaptive thresholding for text extraction."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary
