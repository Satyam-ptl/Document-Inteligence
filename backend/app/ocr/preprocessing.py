"""
OpenCV preprocessing applied to page images before they go into PaddleOCR-VL:
grayscale + denoise + deskew. PaddleOCR-VL is reasonably robust on its own,
so this is deliberately light-touch — it corrects the two failure modes that
hurt OCR the most (skewed scans, salt-and-pepper scanner noise) and stops
there, rather than trying to binarize/enhance aggressively and risk
destroying faint text.

Runs in place: overwrites the input image with the processed version and
returns the same path, so callers (the rasterizer output, or an uploaded
image file) don't need to track two paths.
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def preprocess_image(image_path: Path) -> Path:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"OpenCV could not read {image_path}; leaving un-preprocessed.")
            return image_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        deskewed = _deskew(denoised)

        cv2.imwrite(str(image_path), deskewed)
    except Exception:  # noqa: BLE001 — preprocessing must never block ingestion
        logger.warning(f"Preprocessing failed for {image_path}; using original image.")
    return image_path


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotational skew via the minimum-area
    bounding rect of thresholded foreground pixels. Returns the input
    unchanged if the estimated skew is negligible or detection is unstable
    (near-blank pages, photos with little text) — an unnecessary rotation on
    a page that wasn't actually skewed does more harm than good."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns an angle in (-90, 0]; normalize to a small
    # signed rotation rather than a near-90-degree "correction".
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        # Too small to matter, or too large to trust as scan skew (more
        # likely an actual rotated/landscape page or a bad estimate).
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
