"""User-facing card display helpers.

Keep solver/API/internal values in ASCII card notation (AcKd, Qh8c).  Only call
these helpers at presentation boundaries: Telegram messages, CLI summaries,
coach/context text, and report labels.
"""
from __future__ import annotations

import re

SUIT_EMOJI = {"c": "♣️", "d": "♦️", "h": "♥️", "s": "♠️"}
_CARD_RE = re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])([cdhs])(?![A-Za-z0-9])", re.IGNORECASE)
_CONCAT_CARDS_RE = re.compile(r"^(?:[2-9TJQKA][cdhs])+$", re.IGNORECASE)


def card_to_emoji(card: str | None) -> str:
    """Display one exact card: ``Ac`` -> ``A♣️``.

    Unknown/partial inputs pass through unchanged; this is display-only and
    deliberately not a parser.
    """
    s = (card or "").strip()
    if len(s) != 2:
        return s
    rank, suit = s[0].upper(), s[1].lower()
    if rank not in "23456789TJQKA" or suit not in SUIT_EMOJI:
        return s
    return rank + SUIT_EMOJI[suit]


def cards_to_emoji(cards: str | None) -> str:
    """Display concatenated exact cards: ``AcKdQs`` -> ``A♣️K♦️Q♠️``.

    Class hands (``T9s``, ``K2o``, ``AA``) and malformed values pass through.
    """
    s = (cards or "").strip()
    if not s or len(s) % 2 != 0 or not _CONCAT_CARDS_RE.fullmatch(s):
        return s
    return "".join(card_to_emoji(s[i:i + 2]) for i in range(0, len(s), 2))


def card_tokens_to_emoji(text: str | None) -> str:
    """Replace standalone ASCII exact-card tokens inside user-facing text.

    Examples: ``"Board Ac Kd"`` -> ``"Board A♣️ K♦️"``.  Concatenated boards
    like ``AcKdQs`` should use :func:`cards_to_emoji` directly so we do not
    accidentally rewrite machine URL/query parameters embedded in prose.
    """
    if text is None:
        return ""
    return _CARD_RE.sub(lambda m: card_to_emoji(m.group(1) + m.group(2)), str(text))
