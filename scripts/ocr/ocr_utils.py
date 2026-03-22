"""OCR utility functions: Tesseract wrapper and image preprocessing."""
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


def preprocess_for_ocr(image: np.ndarray, min_width: int = 600) -> np.ndarray:
    """Preprocess image for OCR: upscale if small, convert to grayscale.

    Args:
        image: Input image (grayscale or BGR)
        min_width: Minimum width; images smaller than this are upscaled 2x
    Returns:
        Preprocessed grayscale image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.shape[1] < min_width:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return gray


def ocr_text(image: np.ndarray, whitelist: str = "", psm: int = 7) -> tuple[str, float]:
    """Run Tesseract OCR on an image region.
    Returns (text, confidence 0-100)."""
    if pytesseract is None:
        return "", 0.0
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        texts, confs = [], []
        for i, conf in enumerate(data["conf"]):
            conf = int(conf)
            if conf > 0 and data["text"][i].strip():
                texts.append(data["text"][i].strip())
                confs.append(conf)
        return " ".join(texts), (sum(confs) / len(confs) if confs else 0.0)
    except Exception:
        return "", 0.0


def ocr_number(image: np.ndarray) -> tuple[float | None, float]:
    """OCR a number from an image region. Returns (number, confidence)."""
    text, conf = ocr_text(image, whitelist="0123456789.", psm=7)
    text = text.strip().replace(" ", "")
    try:
        return float(text), conf
    except (ValueError, TypeError):
        return None, 0.0


def binarize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Apply adaptive thresholding for text extraction."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary
