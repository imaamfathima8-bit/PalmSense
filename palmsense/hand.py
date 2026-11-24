"""
hand.py - STEP 1 & 2: Image acquisition, quality control, hand detection,
alignment and palm ROI (Region Of Interest) extraction.

Simple idea to explain:
  1. Read the photo safely and make it a standard size.
  2. Find the skin-coloured hand using the Cr colour channel + Otsu threshold.
  3. Find the "valleys" between the fingers (convexity defects).
  4. Rotate the hand so the finger valleys are level  -> alignment.
  5. Find the biggest circle that fits inside the palm (distance transform).
     Its centre = palm centre, its radius = palm size.
  6. Cut a square around that circle -> the palm ROI. Same region every time.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}  # MPO = JPEG variant some phones produce


class InvalidImage(Exception):
    """Raised when the image cannot be used. The message is shown to the user."""


# --------------------------------------------------------------------------- #
# 1. Acquisition
# --------------------------------------------------------------------------- #
def load_image(data: bytes | np.ndarray, settings: dict) -> np.ndarray:
    """Decode bytes -> BGR image, fix phone rotation (EXIF) and resize."""
    if isinstance(data, np.ndarray):
        bgr = data.copy()
    else:
        try:
            pil = Image.open(io.BytesIO(data))
            fmt = pil.format
            pil.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise InvalidImage("The file is corrupted or is not an image. Please upload a JPG, JPEG or PNG photo.")
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImage(f"Unsupported format '{fmt}'. Only JPG, JPEG and PNG are accepted.")
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    if min(h, w) < settings["image"]["min_side"]:
        raise InvalidImage(
            f"Image resolution is too low ({w}x{h}). Please use at least "
            f"{settings['image']['min_side']} px on the shorter side."
        )
    # Resizing: keep aspect ratio, longest side = max_side -> same scale for every photo.
    max_side = settings["image"]["max_side"]
    scale = max_side / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def check_exposure(bgr: np.ndarray, q: dict, mask: np.ndarray | None = None) -> dict:
    """Brightness check. Measured on the HAND pixels when a mask is given,
    so a dark or white background (recommended) is not mistaken for bad exposure."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[mask > 0] if mask is not None and np.count_nonzero(mask) else gray.ravel()
    mean = float(vals.mean())
    dark = float((vals < 15).mean())
    bright = float((vals > 245).mean())
    if mask is None:  # whole-image test: only reject extreme cases
        if mean < 18:
            raise InvalidImage("The image is too dark. Please take the photo in better lighting.")
        if mean > 248:
            raise InvalidImage("The image is over-exposed (too bright). Avoid direct flash or strong light.")
    else:
        if mean < q["min_brightness"] or dark > 0.3:
            raise InvalidImage("The palm is too dark. Please take the photo in better lighting.")
        if mean > q["max_brightness"] or bright > q["max_clipped_fraction"]:
            raise InvalidImage("The palm is over-exposed (too bright). Avoid direct flash or strong light.")
    return {"palm_brightness": round(mean, 1), "dark_fraction": round(dark, 3), "bright_fraction": round(bright, 3)}

# --------------------------------------------------------------------------- #
# 2. Hand segmentation
# --------------------------------------------------------------------------- #
def segment_hand(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (255 = hand) of the largest skin-coloured object.

    Skin has a high 'Cr' (red-difference) value in the YCrCb colour space for
    every skin tone, so Otsu's method automatically picks the best threshold
    between skin and background.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = cv2.GaussianBlur(ycrcb[:, :, 1], (7, 7), 0)
    _, otsu = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Loose skin range removes objects that are clearly not skin.
    skin_range = cv2.inRange(ycrcb, (20, 133, 70), (255, 185, 135))
    mask = cv2.bitwise_and(otsu, skin_range)

    # Morphology: opening removes small noise, closing fills small holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clean = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
    return clean


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def palm_circle(mask: np.ndarray):
    """Largest circle inside the hand = the palm. Uses the distance transform."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, centre = cv2.minMaxLoc(dist)
    return (float(centre[0]), float(centre[1])), float(radius)


def finger_valleys(contour, radius: float):
    """Convexity defects = the deep 'valleys' between fingers."""
    eps = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 5:
        return []
    try:
        hull = cv2.convexHull(approx, returnPoints=False)
        hull = np.sort(hull.ravel())[:, None].astype(np.int32)
        defects = cv2.convexityDefects(approx, hull)
    except cv2.error:
        return []
    if defects is None:
        return []
    valleys = []
    for s, e, f, d in np.asarray(defects).reshape(-1, 4):
        depth = d / 256.0
        if depth > 0.25 * radius:
            valleys.append({"point": tuple(int(v) for v in approx[f][0]), "depth": float(depth)})
    return valleys


def rotate_bound(img: np.ndarray, angle_deg: float, interp=cv2.INTER_LINEAR, border_value=0):
    """Rotate counter-clockwise by angle_deg without cutting off corners."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=interp, borderValue=border_value)


def _coarse_angle(mask: np.ndarray) -> float:
    """Angle (deg, CCW) that makes the hand's long axis vertical (image moments / PCA)."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    theta = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))  # axis vs x-axis (y down)
    # Rotating CCW by (theta + 90) brings the axis to vertical; choose the smaller of the two options
    angle = theta + 90.0
    if angle > 90:
        angle -= 180
    return angle


def _split_valleys(valleys, centre, radius):
    cx, cy = centre
    fingers = [v for v in valleys if v["point"][1] < cy - 0.75 * radius]
    others = [v for v in valleys if cy - 0.75 * radius <= v["point"][1] < cy + 0.9 * radius
              and abs(v["point"][0] - cx) > 0.4 * radius]
    thumb = max(others, key=lambda v: v["depth"]) if others else None
    return fingers, thumb


# --------------------------------------------------------------------------- #
# 3. Full localisation: mask -> alignment -> ROI
# --------------------------------------------------------------------------- #
def locate_palm(bgr: np.ndarray, settings: dict, declared_hand: str) -> dict:
    q = settings["quality"]
    h, w = bgr.shape[:2]
    mask = segment_hand(bgr)
    contour = largest_contour(mask)
    if contour is None:
        raise InvalidImage("No palm detected. Place your open hand clearly in the frame on a plain background.")

    frac = cv2.contourArea(contour) / float(h * w)
    if frac < q["min_hand_fraction"]:
        raise InvalidImage("No palm detected (hand too small or missing). Move the hand closer to the camera.")
    if frac > q["max_hand_fraction"]:
        raise InvalidImage("The hand fills the whole frame. Move it a little further from the camera.")

    exposure = check_exposure(bgr, q, mask)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / max(hull_area, 1)

    # ---- Coarse alignment (long axis vertical, fingers up) ----
    angle1 = _coarse_angle(mask)
    mask_r = rotate_bound(mask, angle1, cv2.INTER_NEAREST)
    centre, radius = palm_circle(mask_r)
    valleys = finger_valleys(largest_contour(mask_r), radius)
    fingers, _ = _split_valleys(valleys, centre, radius)

    # ---- Fine alignment: make the finger valleys level ----
    angle2 = 0.0
    if len(fingers) >= 2:
        pts = np.array([v["point"] for v in fingers], dtype=np.float64)
        slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
        angle2 = math.degrees(math.atan(slope))  # positive slope (y down) -> rotate CCW
        angle2 = float(np.clip(angle2, -30, 30))
    total_angle = angle1 + angle2

    # Apply ONE rotation to the original colour image + mask
    img_al = rotate_bound(bgr, total_angle, cv2.INTER_LINEAR, border_value=(0, 0, 0))
    mask_al = rotate_bound(mask, total_angle, cv2.INTER_NEAREST)
    contour_al = largest_contour(mask_al)
    centre, radius = palm_circle(mask_al)
    valleys = finger_valleys(contour_al, radius)
    fingers, thumb = _split_valleys(valleys, centre, radius)
    cx, cy = centre

    if radius < q["min_palm_radius_px"]:
        raise InvalidImage("The palm is too small in the photo. Move the hand closer to the camera.")
    if solidity > 0.93 and len(fingers) < 2:
        raise InvalidImage("The object does not look like an open hand. Show your palm with fingers slightly spread.")

    # Palm cut by image border? (check in original image coordinates via the aligned mask border)
    hh, ww = mask_al.shape
    half = 0.95 * radius
    if cx - half < 0 or cy - half < 0 or cx + half > ww or cy + half > hh:
        raise InvalidImage("The palm is cut off by the edge of the photo. Keep the whole hand inside the frame.")

    # Thumb side: detected from the thumb valley if possible, otherwise from the declared hand.
    # (Palm facing camera: right hand -> thumb on image right, left hand -> thumb on image left.)
    if thumb is not None:
        thumb_side = "right" if thumb["point"][0] > cx else "left"
        thumb_source = "detected"
    else:
        thumb_side = "right" if declared_hand == "right" else "left"
        thumb_source = "assumed from selected hand"

    # ---- ROI: square around the palm circle ----
    side = int(round(2 * half))
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    roi = img_al[y0:y0 + side, x0:x0 + side].copy()
    roi_mask = mask_al[y0:y0 + side, x0:x0 + side].copy()
    size = settings["image"]["roi_size"]
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(roi_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Canonical orientation: thumb always on the RIGHT so every palm is compared the same way.
    mirrored = thumb_side == "left"
    if mirrored:
        roi, roi_mask = cv2.flip(roi, 1), cv2.flip(roi_mask, 1)

    # ---- Hand geometry (approximate, used for palm-shape reading) ----
    top_y = int(contour_al[:, 0, 1].min())
    valley_y = float(np.mean([v["point"][1] for v in fingers])) if fingers else cy - 1.1 * radius
    finger_len = max(valley_y - top_y, 0.0)
    palm_len = (cy + 1.05 * radius) - valley_y
    row = mask_al[int(cy)]
    palm_width = float(np.count_nonzero(row)) if np.count_nonzero(row) else 2 * radius
    geometry = {
        "palm_width_px": round(palm_width, 1),
        "palm_length_px": round(palm_len, 1),
        "finger_length_px": round(finger_len, 1),
        "palm_aspect": round(palm_len / max(palm_width, 1), 2),
        "finger_to_palm": round(finger_len / max(palm_len, 1), 2),
        "fingers_separated": len(fingers),
    }

    # ---- Visualisations for the UI ----
    vis_detect = img_al.copy()
    cv2.drawContours(vis_detect, [contour_al], -1, (80, 220, 120), 2)
    cv2.circle(vis_detect, (int(cx), int(cy)), int(radius), (255, 200, 60), 2)
    cv2.circle(vis_detect, (int(cx), int(cy)), 5, (255, 200, 60), -1)
    for v in fingers:
        cv2.circle(vis_detect, v["point"], 8, (60, 60, 255), -1)
    if thumb:
        cv2.circle(vis_detect, thumb["point"], 8, (0, 220, 255), -1)
    cv2.rectangle(vis_detect, (x0, y0), (x0 + side, y0 + side), (255, 80, 200), 3)

    return {
        "mask": mask,
        "exposure": exposure,
        "aligned": img_al,
        "detection_vis": vis_detect,
        "roi_color": roi,
        "roi_mask": roi_mask,
        "rotation_deg": round(total_angle, 1),
        "palm_centre": (round(cx, 1), round(cy, 1)),
        "palm_radius": round(radius, 1),
        "solidity": round(solidity, 3),
        "hand_fraction": round(frac, 3),
        "finger_valleys": len(fingers),
        "thumb_side": thumb_side,
        "thumb_source": thumb_source,
        "mirrored": mirrored,
        "geometry": geometry,
    }
