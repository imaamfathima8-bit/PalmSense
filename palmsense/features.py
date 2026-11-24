"""
features.py - STEP 3 & 4: Preprocessing (enhancement) and feature extraction.

Preprocessing chain (same for enrollment, verification and palm reading):
  colour ROI -> grayscale -> fill background -> Gaussian denoise -> CLAHE contrast
  -> normalisation (zero mean, unit variance)

Two complementary feature methods:
  A. PALM-LINE features  : black-hat morphology highlights dark creases/lines.
                           "Where are the lines?"   -> 1024 numbers
  B. LBP TEXTURE features: Local Binary Patterns describe the fine skin texture.
                           "What does the skin surface look like?" -> 160 numbers
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

LINE_GRID = 32          # line map is reduced to 32x32 cells
LBP_POINTS, LBP_RADIUS = 8, 2
LBP_GRID = 4            # 4x4 cells -> 16 local histograms
LBP_BINS = LBP_POINTS + 2  # 'uniform' LBP gives P+2 different codes


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(roi_color: np.ndarray, roi_mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    inside = roi_mask > 0
    # Fill the background corners with the median skin value so they create no fake edges.
    filled = gray.copy()
    if inside.any():
        filled[~inside] = int(np.median(gray[inside]))
    denoised = cv2.GaussianBlur(filled, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)
    f = clahe.astype(np.float32)
    normalized = (f - f.mean()) / (f.std() + 1e-6)
    norm_vis = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {"gray": gray, "denoised": denoised, "clahe": clahe,
            "normalized": normalized, "normalized_vis": norm_vis, "mask": inside}


def sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian inside the palm. Low value = blurry photo."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask].var()) if mask.any() else 0.0


# --------------------------------------------------------------------------- #
# Method A: palm lines (black-hat morphology)
# --------------------------------------------------------------------------- #
def line_response(clahe: np.ndarray) -> np.ndarray:
    """Black-hat = closing(image) - image. Dark thin lines become bright."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel)


def line_map(response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Binary map of the strongest lines: threshold + morphology + remove small blobs."""
    vals = response[mask] if mask.any() else response.ravel()
    t = max(float(np.percentile(vals, 90)), 8.0)
    binary = ((response > t) & mask).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            clean[labels == i] = 255
    return clean


def line_features(response: np.ndarray) -> np.ndarray:
    f = cv2.GaussianBlur(response.astype(np.float32), (0, 0), 3)   # small shift tolerance
    f = cv2.resize(f, (LINE_GRID, LINE_GRID), interpolation=cv2.INTER_AREA).ravel()
    f -= f.mean()
    return (f / (np.linalg.norm(f) + 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Method B: LBP texture
# --------------------------------------------------------------------------- #
def lbp_image(clahe: np.ndarray) -> np.ndarray:
    small = cv2.resize(clahe, (128, 128), interpolation=cv2.INTER_AREA)
    return local_binary_pattern(small, LBP_POINTS, LBP_RADIUS, method="uniform")


def lbp_features(lbp: np.ndarray) -> np.ndarray:
    cell = lbp.shape[0] // LBP_GRID
    hists = []
    for r in range(LBP_GRID):
        for c in range(LBP_GRID):
            block = lbp[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            h, _ = np.histogram(block, bins=LBP_BINS, range=(0, LBP_BINS))
            hists.append(h / max(h.sum(), 1))
    return np.concatenate(hists).astype(np.float32)


# --------------------------------------------------------------------------- #
# One call that does everything (used by enrollment, verification, reading)
# --------------------------------------------------------------------------- #
def extract_all(roi_color: np.ndarray, roi_mask: np.ndarray) -> dict:
    pre = preprocess(roi_color, roi_mask)
    resp = line_response(pre["clahe"])
    lmap = line_map(resp, pre["mask"])
    lbp = lbp_image(pre["clahe"])
    edges = cv2.Canny(pre["denoised"], 40, 110)
    edges[~pre["mask"]] = 0
    return {
        "pre": pre,
        "line_response": resp,
        "line_map": lmap,
        "canny": edges,
        "lbp_image": lbp,
        "vectors": {"lines": line_features(resp), "lbp": lbp_features(lbp)},
        "sharpness": sharpness(pre["gray"], pre["mask"]),
    }
