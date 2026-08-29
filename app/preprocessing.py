"""
Image preprocessing for label photos.

Real label photos are rarely flat, well-lit scans -- they're phone photos
of curved bottles, jars, and boxes under mixed lighting. OCR accuracy drops
sharply on such images, so we run a short pipeline before handing anything
to the OCR engine:

    1. Resize (very large images slow OCR down for no accuracy benefit)
    2. Denoise
    3. Deskew (rotate so the dominant text angle is horizontal)
    4. Glare / contrast correction (CLAHE)

Each step is deliberately simple and fast rather than state-of-the-art --
this is the Phase 1 MVP. Perspective correction for strongly curved
surfaces (bottles, jars) is flagged as a Phase 1.5 improvement, not
implemented here (see module docstring at bottom).
"""
from __future__ import annotations

import cv2
import numpy as np


MAX_DIMENSION = 2000  # px, longest side


def load_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (as received over HTTP) into a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image -- unsupported format or corrupt file")
    return img


def resize_if_needed(img: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


def correct_glare_and_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) applied on the
    lightness channel in LAB space. This evens out glare hotspots and dim
    regions without blowing out color, which plain histogram equalization
    tends to do.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Estimate the dominant text angle from edges and rotate to correct it.
    Falls back to returning the image unchanged if no reliable angle is
    found (e.g. very little text, or a busy background) -- a bad guess is
    worse than no correction here.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=img.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only trust near-horizontal lines as text baselines
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img  # not worth rotating for sub-degree skew

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h),
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Run the full preprocessing pipeline and return an OCR-ready image."""
    img = load_image(image_bytes)
    img = resize_if_needed(img)
    img = denoise(img)
    img = correct_glare_and_contrast(img)
    img = deskew(img)
    return img


# --- Roadmap note ---------------------------------------------------------
# Perspective correction for curved surfaces (bottles/jars) is a known gap
# in this MVP. A reasonable Phase 1.5 approach: detect the label's four
# corners (contour detection assuming a roughly rectangular label region)
# and apply cv2.getPerspectiveTransform + warpPerspective. This was left
# out of the MVP because corner detection is unreliable on cluttered
# backgrounds without a training pass -- it needs its own validation step
# rather than being bolted on silently.
