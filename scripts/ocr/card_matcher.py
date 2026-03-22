"""Card matcher using template matching for N8 replay screenshots.

Identifies rank and suit of a card image by comparing cropped regions
against stored templates using cv2.matchTemplate.
"""

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]


class CardMatcher:
    """Match card images against rank and suit templates."""

    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.rank_templates: dict[str, np.ndarray] = {}
        self.suit_templates: dict[str, np.ndarray] = {}

        for rank in RANKS:
            path = template_dir / f"rank_{rank}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.rank_templates[rank] = img

        for suit in SUITS:
            path = template_dir / f"suit_{suit}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.suit_templates[suit] = img

    def _crop_rank_region(self, card: np.ndarray) -> np.ndarray:
        """Crop the rank region from a card image (top ~35%, left ~50%)."""
        h, w = card.shape[:2]
        region = card[0:int(h * 0.35), 0:int(w * 0.50)]
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return region

    def _crop_suit_region(self, card: np.ndarray) -> np.ndarray:
        """Crop the suit region from a card image (~30-65% height, left ~55%)."""
        h, w = card.shape[:2]
        region = card[int(h * 0.30):int(h * 0.65), 0:int(w * 0.55)]
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return region

    def _match_template(
        self, region: np.ndarray, templates: dict[str, np.ndarray]
    ) -> tuple[str | None, float]:
        """Match a region against templates, return (best_label, confidence).

        Resizes templates to match the region dimensions before matching.
        """
        best_label = None
        best_score = -1.0

        for label, tmpl in templates.items():
            # Resize template to match region size
            resized = cv2.resize(tmpl, (region.shape[1], region.shape[0]))
            result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_label = label

        return best_label, best_score

    def _detect_suit_by_color(self, card: np.ndarray) -> str | None:
        """Detect suit color (red or black) from card image.

        Returns 'red' for hearts/diamonds, 'black' for spades/clubs, or None.
        """
        if len(card.shape) != 3:
            return None

        h, w = card.shape[:2]
        # Look at the suit region for color
        suit_area = card[int(h * 0.30):int(h * 0.65), 0:int(w * 0.55)]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)

        # Red mask (hue ~0-10 and ~170-180)
        mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

        # Count total non-white pixels (the symbol itself)
        gray = cv2.cvtColor(suit_area, cv2.COLOR_BGR2GRAY)
        _, mask_dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        dark_pixels = cv2.countNonZero(mask_dark)

        if dark_pixels < 10:
            return None

        red_ratio = red_pixels / dark_pixels if dark_pixels > 0 else 0
        return "red" if red_ratio > 0.3 else "black"

    def match(self, card_image: np.ndarray) -> tuple[str | None, str | None, float]:
        """Identify rank and suit of a card image.

        Args:
            card_image: BGR or grayscale image of a single card.

        Returns:
            (rank, suit, confidence) e.g. ("K", "s", 0.92).
            rank/suit are None if identification fails.
            confidence is the minimum of rank and suit match scores.
        """
        if card_image is None or card_image.size == 0:
            return None, None, 0.0

        # Ensure we have grayscale for template matching
        if len(card_image.shape) == 3:
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_image

        rank_region = self._crop_rank_region(gray)
        suit_region = self._crop_suit_region(gray)

        rank, rank_conf = self._match_template(rank_region, self.rank_templates)
        suit, suit_conf = self._match_template(suit_region, self.suit_templates)

        # Color-based fallback/refinement for suit
        if suit_conf < 0.7 or (suit is not None and suit_conf < 0.85):
            color = self._detect_suit_by_color(card_image)
            if color is not None:
                if suit is not None:
                    # Validate: if template says heart but color is black, reconsider
                    is_red_suit = suit in ("h", "d")
                    is_red_color = color == "red"
                    if is_red_suit != is_red_color:
                        # Template match disagrees with color — pick best of
                        # correct-color suits
                        if is_red_color:
                            candidates = {k: v for k, v in self.suit_templates.items()
                                          if k in ("h", "d")}
                        else:
                            candidates = {k: v for k, v in self.suit_templates.items()
                                          if k in ("s", "c")}
                        if candidates:
                            suit, suit_conf = self._match_template(
                                suit_region, candidates
                            )

        confidence = min(rank_conf, suit_conf) if rank is not None and suit is not None else 0.0
        return rank, suit, confidence
