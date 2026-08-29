"""
Thin wrapper around PaddleOCR.

Why wrap it instead of calling PaddleOCR directly from routes:
  1. PaddleOCR is expensive to initialize (loads detection + recognition +
     classification models). We want exactly one instance, created once,
     reused across requests.
  2. PaddleOCR's raw output is a nested list-of-lists shape that's easy to
     get wrong. This module converts it into our own TextBlock schema
     immediately, so nothing downstream needs to know PaddleOCR exists.
  3. Language selection: Indian labels mix English with regional scripts.
     PaddleOCR loads one language model at a time, so Phase 1 runs English
     by default and exposes `lang` so a Hindi/Devanagari pass can be added
     as a second OCR call on the same preprocessed image (see NOTE below).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from paddleocr import PaddleOCR

from app.schemas import BoundingBox, TextBlock

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_engine(lang: str = "en") -> PaddleOCR:
    """
    Cached engine construction, keyed by language. lru_cache means we pay
    the model-load cost once per language, not once per request.

    use_angle_cls=True lets PaddleOCR correct individual text lines that
    are rotated 180 degrees (common when a label is photographed upside
    down) independently of the whole-image deskew we already did.
    """
    logger.info("Loading PaddleOCR engine (lang=%s) -- first call downloads models", lang)
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def run_ocr(image: np.ndarray, lang: str = "en") -> list[TextBlock]:
    """
    Run OCR on a preprocessed BGR image and return structured TextBlocks.

    PaddleOCR's raw result shape is: result[0] = list of
    [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence) ]
    i.e. a 4-point polygon plus a (text, confidence) tuple. We collapse
    the polygon to an axis-aligned bounding box since our downstream field
    classifier only needs rough position, not exact rotation.
    """
    engine = _get_engine(lang)
    result = engine.ocr(image, cls=True)

    blocks: list[TextBlock] = []
    if not result or result[0] is None:
        return blocks

    for line in result[0]:
        polygon, (text, confidence) = line
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        blocks.append(TextBlock(
            text=text,
            confidence=float(confidence),
            bbox=BoundingBox(
                x_min=min(xs), y_min=min(ys),
                x_max=max(xs), y_max=max(ys),
            ),
        ))
    return blocks


# --- Roadmap note ---------------------------------------------------------
# Multilingual handling for Phase 1: call run_ocr() once with lang="en" and
# once with the appropriate Indic language code (PaddleOCR supports "hi"
# for Hindi/Devanagari; other regional scripts need their own PaddleOCR
# language packs, not all of which exist yet in the mature 2.x line -- this
# is worth checking against your target states before committing to a
# specific set of regional languages). Merge both result sets and let the
# field classifier match against text in either script. Running two passes
# roughly doubles OCR latency per image, which is an acceptable Phase 1
# tradeoff but worth revisiting if scan volume grows.
