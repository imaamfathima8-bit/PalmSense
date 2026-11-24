"""
pipeline.py - ONE function that runs the full image-processing pipeline.

    analyze_palm(image, hand)  ->  PalmAnalysis

Enrollment, verification and palm reading ALL call this same function,
so the preprocessing and features are always identical (no mismatch).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import features as F
from .hand import InvalidImage, check_exposure, load_image, locate_palm


@dataclass
class PalmAnalysis:
    ok: bool
    reason: str = ""
    hand: str = "right"
    stages: dict = field(default_factory=dict)      # name -> RGB/gray image (for display)
    vectors: dict | None = None                      # feature vectors for matching
    info: dict = field(default_factory=dict)         # numbers shown in the UI
    roi_clahe: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    line_map: np.ndarray | None = None
    line_response: np.ndarray | None = None
    duration_ms: float = 0.0


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze_palm(image, hand: str, settings: dict) -> PalmAnalysis:
    t0 = time.perf_counter()
    res = PalmAnalysis(ok=False, hand=hand)
    try:
        bgr = load_image(image, settings)
        res.stages["1. Original (resized)"] = _rgb(bgr)
        check_exposure(bgr, settings["quality"])          # extreme dark/bright only

        loc = locate_palm(bgr, settings, hand)
        res.stages["2. Hand mask (Cr + Otsu)"] = loc["mask"]
        res.stages["3. Detection & alignment"] = _rgb(loc["detection_vis"])
        res.stages["4. Palm ROI"] = _rgb(loc["roi_color"])

        feats = F.extract_all(loc["roi_color"], loc["roi_mask"])
        pre = feats["pre"]
        res.stages["5. Grayscale"] = pre["gray"]
        res.stages["6. Gaussian denoise"] = pre["denoised"]
        res.stages["7. CLAHE contrast"] = pre["clahe"]
        res.stages["8. Normalised"] = pre["normalized_vis"]
        res.stages["9. Black-hat line response"] = cv2.normalize(
            feats["line_response"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        res.stages["10. Line map (threshold + morphology)"] = feats["line_map"]
        res.stages["11. Canny edges"] = feats["canny"]
        res.stages["12. LBP texture"] = cv2.normalize(
            feats["lbp_image"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        sharp = feats["sharpness"]
        res.info.update({
            "exposure": loc["exposure"],
            "sharpness": round(sharp, 1),
            "rotation_deg": loc["rotation_deg"],
            "palm_radius_px": loc["palm_radius"],
            "finger_valleys": loc["finger_valleys"],
            "thumb_side": loc["thumb_side"],
            "thumb_source": loc["thumb_source"],
            "mirrored_to_canonical": loc["mirrored"],
            "hand_fraction": loc["hand_fraction"],
            "solidity": loc["solidity"],
            "geometry": loc["geometry"],
        })
        if sharp < settings["quality"]["min_sharpness"]:
            raise InvalidImage("The image is too blurry. Hold the camera steady and focus on the palm.")

        res.vectors = feats["vectors"]
        res.roi_clahe = pre["clahe"]
        res.roi_mask = pre["mask"]
        res.line_map = feats["line_map"]
        res.line_response = feats["line_response"]
        res.ok = True
    except InvalidImage as e:
        res.reason = str(e)
    except Exception as e:  # never crash the UI on a strange photo
        res.reason = f"Could not process this image ({type(e).__name__}). Please try another photo."
    res.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res
