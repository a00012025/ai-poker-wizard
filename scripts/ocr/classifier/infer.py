"""CardClassifier — lazy load, batched inference, graceful missing checkpoint."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_V1_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
_V2_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v2.pt"
_DEFAULT_CKPT = _V2_CKPT if _V2_CKPT.exists() else _V1_CKPT

_log = logging.getLogger(__name__)
_LOGGED_MISSING = False


class CardClassifier:
    """Thread-unsafe lazy singleton. Instantiate once at startup (or per call — cheap)."""

    _instance: "CardClassifier | None" = None

    def __new__(cls, ckpt_path: Path | str | None = None):
        # Test-time instantiation with custom ckpt_path bypasses the singleton
        if ckpt_path is not None:
            return super().__new__(cls)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ckpt_path: Path | str | None = None):
        if getattr(self, "_initialized", False) and ckpt_path is None:
            return
        self._ckpt_path = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
        self._net = None
        self._meta = {}
        self._torch = None
        self._letterbox = None
        self._to_tensor = None
        self._load_failed = False
        self._initialized = True

    def _ensure_loaded(self) -> bool:
        global _LOGGED_MISSING
        if self._net is not None:
            return True
        if self._load_failed:
            return False
        if not self._ckpt_path.exists():
            if not _LOGGED_MISSING:
                _log.error("CLASSIFIER_CHECKPOINT_UNAVAILABLE: %s", self._ckpt_path)
                _LOGGED_MISSING = True
            self._load_failed = True
            return False
        try:
            import torch
            from .model import CardCNN
            from .dataset import _letterbox, _to_tensor
            meta_path = self._ckpt_path.with_suffix(".json")
            if meta_path.exists():
                self._meta = json.loads(meta_path.read_text())
            version = self._meta.get("version", "v1")
            if version == "mobilenet_v3_small":
                from .model import CardMobileNetV3Small
                net = CardMobileNetV3Small()
            elif version == "v2":
                from .model import CardCNNv2
                net = CardCNNv2()
            else:
                net = CardCNN()
            net.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"))
            net.eval()
            self._net = net
            self._torch = torch
            self._letterbox = _letterbox
            self._to_tensor = _to_tensor
            return True
        except Exception as e:
            _log.error("CLASSIFIER_LOAD_FAILED: %s", e, exc_info=True)
            self._load_failed = True
            return False

    def _warm(self) -> None:
        """Force checkpoint load + one dummy forward pass."""
        if not self._ensure_loaded():
            return
        dummy = np.zeros((48, 64, 3), dtype=np.uint8)
        self.classify_batch([dummy])

    def classify(self, crop: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        results = self.classify_batch([crop])
        return results[0] if results else (None, None, 0.0)

    def classify_batch(
        self, crops: list[np.ndarray]
    ) -> list[tuple[Optional[str], Optional[str], float]]:
        return [(d["rank"], d["suit"], d["conf"])
                for d in self.classify_batch_detailed(crops)]

    def classify_batch_detailed(
        self, crops: list[np.ndarray]
    ) -> list[dict]:
        """Same as classify_batch but exposes rank_conf and suit_conf separately.

        Returned dict keys: rank, rank_conf, suit, suit_conf, conf
        (conf = min(rank_conf, suit_conf)). When the checkpoint is missing,
        returns dicts with None ranks/suits and 0.0 confidences.
        """
        if not crops:
            return []
        if not self._ensure_loaded():
            return [{"rank": None, "rank_conf": 0.0,
                     "suit": None, "suit_conf": 0.0, "conf": 0.0}] * len(crops)
        from .model import RANK_CLASSES, SUIT_CLASSES
        x = self._torch.stack([self._to_tensor(self._letterbox(c)) for c in crops])
        with self._torch.no_grad():
            rl, sl = self._net(x)
            temp_rank = float(self._meta.get("temperature_rank", self._meta.get("temperature", 1.0)) or 1.0)
            temp_suit = float(self._meta.get("temperature_suit", self._meta.get("temperature", 1.0)) or 1.0)
            rl = rl / temp_rank
            sl = sl / temp_suit
            r_probs = self._torch.softmax(rl, dim=1)
            s_probs = self._torch.softmax(sl, dim=1)
        out = []
        for i in range(x.shape[0]):
            r_idx = int(r_probs[i].argmax()); r_c = float(r_probs[i, r_idx])
            s_idx = int(s_probs[i].argmax()); s_c = float(s_probs[i, s_idx])
            rank_top = self._torch.topk(r_probs[i], k=min(2, len(RANK_CLASSES)))
            suit_top = self._torch.topk(s_probs[i], k=min(2, len(SUIT_CLASSES)))
            out.append({
                "rank": RANK_CLASSES[r_idx], "rank_conf": r_c,
                "suit": SUIT_CLASSES[s_idx], "suit_conf": s_c,
                "rank_top2": [
                    (RANK_CLASSES[int(idx)], float(prob))
                    for prob, idx in zip(rank_top.values, rank_top.indices)
                ],
                "suit_top2": [
                    (SUIT_CLASSES[int(idx)], float(prob))
                    for prob, idx in zip(suit_top.values, suit_top.indices)
                ],
                "conf": min(r_c, s_c),
            })
        return out
