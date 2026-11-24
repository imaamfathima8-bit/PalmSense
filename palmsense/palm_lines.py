"""
palm_lines.py - STEP 7: Identify the major palm lines for the palm reading.

Input : the aligned palm ROI (fingers at the top, thumb on the RIGHT) and its
        binary line map from features.py.
Method: 1. Thin the line map to 1-pixel lines (skeletonisation).
        2. Each major line is searched for only in the zone where it normally lies
           (heart = top, head = middle, life = around the thumb, fate = centre).
        3. RANSAC curve fitting picks the longest smooth curve in that zone and
           ignores small random creases.
        4. Every line gets a confidence score -> Detected / Uncertain / Not visible.

Everything in this file is OBSERVED by computer vision. Interpretations are
written later by the language model and are labelled separately.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
from skimage.morphology import skeletonize

warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)

COLORS = {"heart": (80, 80, 255), "head": (255, 170, 60), "life": (90, 220, 90), "fate": (40, 210, 255)}
NAMES = {"heart": "Heart line", "head": "Head line", "life": "Life line", "fate": "Fate line"}

# zone = (x_min, x_max, y_min, y_max) in 0..1 of the ROI; along = axis the line runs along
ZONES = {
    "heart": {"zone": (0.00, 0.88, 0.02, 0.42), "along": "x", "max_slope": 0.7, "expected": 0.60},
    "head": {"zone": (0.08, 1.00, 0.25, 0.66), "along": "x", "max_slope": 0.6, "expected": 0.55},
    "life": {"zone": (0.42, 1.00, 0.22, 1.00), "along": "y", "max_slope": 1.3, "expected": 0.55},
    "fate": {"zone": (0.30, 0.70, 0.40, 1.00), "along": "y", "max_slope": 0.35, "expected": 0.40, "degree": 1},
}


def _ransac(u, v, degree, max_slope, size, rng, iters=500, tol=3.0):
    """Fit v = poly(u). Returns (coeffs, inlier mask) of the curve covering most distinct u values."""
    n = len(u)
    if n < 15:
        return None, None
    best, best_cov = None, 0
    k = degree + 1
    for _ in range(iters):
        idx = rng.choice(n, k, replace=False)
        if np.ptp(u[idx]) < 0.12 * size:
            continue
        try:
            c = np.polyfit(u[idx], v[idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue
        um = (u.min() + u.max()) / 2
        slope = np.polyval(np.polyder(c), um) if degree else 0
        if abs(slope) > max_slope or (degree == 2 and abs(c[0]) > 4.0 / size):
            continue
        inl = np.abs(np.polyval(c, u) - v) < tol
        cov = len(np.unique(u[inl]))
        if cov > best_cov:
            best, best_cov = inl, cov
    if best is None or best.sum() < 12:
        return None, None
    c = np.polyfit(u[best], v[best], degree)
    inl = np.abs(np.polyval(c, u) - v) < tol + 1
    return c, inl


def _longest_run(values, gap):
    vals = np.unique(values)
    if len(vals) == 0:
        return None
    runs, start = [], vals[0]
    for a, b in zip(vals, vals[1:]):
        if b - a > gap:
            runs.append((start, a))
            start = b
    runs.append((start, vals[-1]))
    return max(runs, key=lambda r: r[1] - r[0])


def _describe(key, pts, size, strength):
    """Turn the fitted curve into simple observed attributes."""
    seg = np.diff(pts, axis=0)
    length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))) / size
    chord = np.linalg.norm(pts[-1] - pts[0]) + 1e-6
    ab, ap = pts[-1] - pts[0], pts - pts[0]
    d = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / chord   # distance of each point from the chord
    sagitta = float(d.max() / chord)
    attrs = {
        "length": "short" if length < 0.40 else "medium" if length < 0.62 else "long",
        "shape": "straight" if sagitta < 0.04 else "gently curved" if sagitta < 0.10 else "strongly curved",
        "clarity": "faint" if strength < 0.35 else "clear" if strength < 0.60 else "deep",
        "length_ratio": round(length, 2),
    }
    if key == "heart":
        left, right = pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]
        attrs["ending"] = ("curves up toward the fingers" if (left[1] - right[1]) / size > 0.07
                           else "runs fairly straight across the palm")
    if key == "head":
        left, right = pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]
        drop = (left[1] - right[1]) / size
        attrs["slope"] = "slopes down toward the wrist" if drop > 0.07 else "runs fairly straight across"
    if key == "life":
        attrs["arc"] = "wide arc into the palm" if pts[:, 0].min() / size < 0.62 else "close arc around the thumb"
    return attrs


def detect_lines(line_map: np.ndarray, response: np.ndarray, mask: np.ndarray, seed: int = 7) -> dict:
    size = line_map.shape[0]
    skel = skeletonize(line_map > 0)
    ys, xs = np.nonzero(skel)
    available = np.ones(len(xs), bool)
    p99 = float(np.percentile(response[mask], 99)) if mask.any() else float(response.max() or 1)
    rng = np.random.default_rng(seed)
    out = {}

    for key, cfg in ZONES.items():
        x0, x1, y0, y1 = (np.array(cfg["zone"]) * size)
        sel = available & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        u, v = (xs[sel], ys[sel]) if cfg["along"] == "x" else (ys[sel], xs[sel])
        deg = cfg.get("degree", 2)
        coeffs, inl = _ransac(u.astype(float), v.astype(float), deg, cfg["max_slope"], size, rng)
        result = {"name": NAMES[key], "status": "Not clearly visible", "confidence": 0.0, "points": None}
        if coeffs is not None:
            run = _longest_run(u[inl], gap=0.10 * size)
            if run is not None and run[1] - run[0] > 0.12 * size:
                uu = np.linspace(run[0], run[1], 60)
                vv = np.polyval(coeffs, uu)
                pts = np.stack([uu, vv], 1) if cfg["along"] == "x" else np.stack([vv, uu], 1)
                pts = np.clip(pts, 0, size - 1)
                on = inl & (u >= run[0]) & (u <= run[1])
                px, py = (u[on], v[on]) if cfg["along"] == "x" else (v[on], u[on])
                strength = float(np.clip(response[py.astype(int), px.astype(int)].mean() / (p99 + 1e-6), 0, 1))
                coverage = len(np.unique(u[on])) / (cfg["expected"] * size)
                conf = round(float(np.clip(0.7 * min(coverage, 1.0) + 0.3 * strength, 0, 1)), 2)
                status = "Detected" if conf >= 0.55 else "Uncertain" if conf >= 0.35 else "Not clearly visible"
                result.update({"status": status, "confidence": conf})
                if status != "Not clearly visible":
                    result["points"] = pts
                    result["attributes"] = _describe(key, pts, size, strength)
                    # remove these pixels so the next line cannot reuse them
                    idx = np.nonzero(sel)[0][on]
                    available[idx] = False
                    near = np.min(np.hypot(xs[:, None] - pts[None, :, 0], ys[:, None] - pts[None, :, 1]), axis=1) < 6
                    available &= ~near
        out[key] = result

    # Joined head and life line (a classic palmistry observation)
    if out["head"].get("points") is not None and out["life"].get("points") is not None:
        hp, lp = out["head"]["points"], out["life"]["points"]
        head_start = hp[np.argmax(hp[:, 0])]
        life_start = lp[np.argmin(lp[:, 1])]
        out["head"]["attributes"]["joined_with_life_line"] = bool(np.linalg.norm(head_start - life_start) < 0.10 * size)

    # Other prominent creases left over
    rest = np.zeros_like(line_map)
    rest[ys[available], xs[available]] = 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(rest, np.ones((3, 3), np.uint8)))
    out["minor_creases"] = int(sum(stats[i, cv2.CC_STAT_AREA] > 0.12 * size for i in range(1, n)))
    return out


def palm_shape(geometry: dict) -> dict:
    """Approximate palm & finger proportions -> traditional 'element' hand type."""
    square = geometry["palm_aspect"] < 1.05
    long_fingers = geometry["finger_to_palm"] >= 0.85
    element = {(True, False): "Earth", (True, True): "Air", (False, True): "Water", (False, False): "Fire"}[
        (square, long_fingers)]
    return {
        "palm": "square" if square else "rectangular (long)",
        "fingers": "long" if long_fingers else "short",
        "element_type": element,
        "palm_aspect": geometry["palm_aspect"],
        "finger_to_palm": geometry["finger_to_palm"],
        "note": "approximate - measured from the hand outline",
    }


def annotate(roi_color: np.ndarray, lines: dict, mirrored: bool) -> np.ndarray:
    """Draw detected lines (solid) and uncertain lines (dashed) on the palm ROI."""
    vis = roi_color.copy()
    big = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_CUBIC)
    s = 512 / roi_color.shape[0]
    for key in ("heart", "head", "life", "fate"):
        info = lines[key]
        if info.get("points") is None:
            continue
        pts = (info["points"] * s).astype(np.int32)
        col = COLORS[key]
        if info["status"] == "Detected":
            cv2.polylines(big, [pts], False, (255, 255, 255), 7, cv2.LINE_AA)
            cv2.polylines(big, [pts], False, col, 4, cv2.LINE_AA)
        else:
            for i in range(0, len(pts) - 1, 4):
                cv2.line(big, tuple(pts[i]), tuple(pts[min(i + 2, len(pts) - 1)]), col, 3, cv2.LINE_AA)
    if mirrored:
        big = cv2.flip(big, 1)
    for key in ("heart", "head", "life", "fate"):  # labels after flipping so text is readable
        info = lines[key]
        if info.get("points") is None:
            continue
        p = (info["points"][len(info["points"]) // 2] * s).astype(int)
        if mirrored:
            p[0] = 511 - p[0]
        label = NAMES[key].split()[0] + ("" if info["status"] == "Detected" else "?")
        org = (int(np.clip(p[0] - 25, 2, 440)), int(np.clip(p[1] - 10, 15, 505)))
        cv2.putText(big, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(big, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[key], 2, cv2.LINE_AA)
    return big


def structured_features(lines: dict, shape: dict, hand: str) -> dict:
    """Clean JSON-ready summary: this is what the language model receives."""
    feats = {"hand": hand, "palm_shape": shape, "minor_creases": lines["minor_creases"], "lines": {}}
    for key in ("heart", "head", "life", "fate"):
        info = lines[key]
        feats["lines"][key] = {"status": info["status"], "confidence": info["confidence"],
                               **({"attributes": info["attributes"]} if info.get("attributes") else {})}
    return feats


def compare_hands(left: dict, right: dict) -> list[str]:
    """Observed differences between the two palms (computed, not invented)."""
    diffs = []
    for key in ("heart", "head", "life", "fate"):
        a, b = left["lines"][key], right["lines"][key]
        name = NAMES[key]
        if a["status"] != b["status"]:
            diffs.append(f"{name}: {a['status'].lower()} on the left palm, {b['status'].lower()} on the right palm.")
            continue
        for attr in ("length", "shape", "clarity", "ending", "slope", "arc"):
            va, vb = a.get("attributes", {}).get(attr), b.get("attributes", {}).get(attr)
            if va and vb and va != vb:
                diffs.append(f"{name} {attr}: left = {va}, right = {vb}.")
    if left["palm_shape"]["element_type"] != right["palm_shape"]["element_type"]:
        diffs.append(f"Hand shape: left looks {left['palm_shape']['element_type']}, "
                     f"right looks {right['palm_shape']['element_type']}.")
    return diffs or ["No clear differences were detected between the left and right palms."]


def read_palm(analysis) -> dict:
    """Run line detection on a PalmAnalysis from pipeline.analyze_palm()."""
    lines = detect_lines(analysis.line_map, analysis.line_response, analysis.roi_mask)
    shape = palm_shape(analysis.info["geometry"])
    roi_color = cv2.cvtColor(analysis.stages["4. Palm ROI"], cv2.COLOR_RGB2BGR)
    ann = annotate(roi_color, lines, analysis.info["mirrored_to_canonical"])
    return {"lines": lines, "shape": shape,
            "annotated": cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
            "features": structured_features(lines, shape, analysis.hand)}
