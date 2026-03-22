"""Region detector for Natural8 replay screenshots.

Splits a full N8 screenshot into table (upper) and action panel (lower)
regions by detecting the dark horizontal divider band between them.
"""

import cv2
import numpy as np


def detect_regions(image: np.ndarray) -> dict | None:
    """Detect table and action panel regions in an N8 replay screenshot.

    Scans for a dark horizontal band in the 35-55% height range, then
    validates by checking for the column header row just below it.

    Args:
        image: BGR image (full screenshot)

    Returns:
        {"table": ndarray, "panel": ndarray, "divider_y": int} or None
    """
    if image is None or len(image.shape) != 3:
        return None

    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Scan rows in the 35-55% height range for dark horizontal bands
    scan_start = int(h * 0.35)
    scan_end = int(h * 0.55)

    # Compute mean brightness per row in the scan range
    row_means = np.mean(gray[scan_start:scan_end, :], axis=1)

    # Find runs of dark rows (mean brightness < threshold)
    dark_threshold = 60
    dark_rows = row_means < dark_threshold

    # Find the best dark band: longest contiguous run of dark rows
    best_start = -1
    best_len = 0
    cur_start = -1
    cur_len = 0

    for i in range(len(dark_rows)):
        if dark_rows[i]:
            if cur_start == -1:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
            cur_start = -1
            cur_len = 0

    # Check final run
    if cur_len > best_len:
        best_start = cur_start
        best_len = cur_len

    if best_len < 2:
        return None

    # The divider_y is the bottom of the dark band (absolute coords)
    divider_y = scan_start + best_start + best_len

    # Validate: check that below the divider there's a header-like region
    # The header row should have relatively dark background with lighter text
    # Check a strip just below the divider band
    header_start = divider_y
    header_end = min(divider_y + int(h * 0.06), h)

    if header_end - header_start < 5:
        return None

    header_strip = gray[header_start:header_end, :]

    # The header should have a mix: dark background with some bright text pixels
    header_mean = np.mean(header_strip)
    header_std = np.std(header_strip)

    # N8 header has dark bg (low mean) with text (moderate std)
    # Reject if the header area is too bright (not N8) or too uniform
    if header_mean > 160 or header_std < 15:
        return None

    # Additional validation: check that the dark band spans most of the width
    # (not just a small dark area)
    band_row = gray[scan_start + best_start:divider_y, :]
    # Check what fraction of pixels in the band are dark
    dark_pixel_ratio = np.mean(band_row < dark_threshold + 20)
    if dark_pixel_ratio < 0.5:
        return None

    table = image[:divider_y, :]
    panel = image[divider_y:, :]

    return {
        "table": table,
        "panel": panel,
        "divider_y": divider_y,
    }
