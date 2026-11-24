"""
matching.py - STEP 5: Templates, similarity scores and the accept/reject decision.

Convention used everywhere in this app:
    SIMILARITY score in [0, 1]  ->  HIGHER = more similar
    ACCEPT when score >= threshold
"""
from __future__ import annotations

import numpy as np

METHODS = ("lines", "lbp")


def build_template(vectors: list[dict]) -> dict:
    """Average the enrollment samples into one template per method."""
    lines = np.mean([v["lines"] for v in vectors], axis=0)
    lines -= lines.mean()
    lines /= np.linalg.norm(lines) + 1e-9
    lbp = np.mean([v["lbp"] for v in vectors], axis=0)
    return {"lines": lines.astype(np.float32), "lbp": lbp.astype(np.float32)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Method A score: cosine similarity of the line patterns (negative -> 0)."""
    s = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
    return max(0.0, s)


def chi_square_similarity(a: np.ndarray, b: np.ndarray, n_cells: int = 16) -> float:
    """Method B score: 1 - average chi-square distance between LBP histograms."""
    a = a.reshape(n_cells, -1)
    b = b.reshape(n_cells, -1)
    chi = 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9), axis=1)   # each in [0, 1]
    return float(1.0 - chi.mean())


def normalise(score: float, lo: float, hi: float) -> float:
    return float(np.clip((score - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def compare(probe: dict, template: dict, matching_cfg: dict) -> dict:
    """Compare a probe's feature vectors with a stored template."""
    s_lines = cosine_similarity(probe["lines"], template["lines"])
    s_lbp = chi_square_similarity(probe["lbp"], template["lbp"])
    norm = matching_cfg["norm"]
    w = matching_cfg.get("fusion_weight_lines", 0.5)
    s_fusion = w * normalise(s_lines, *norm["lines"]) + (1 - w) * normalise(s_lbp, *norm["lbp"])
    return {"lines": round(s_lines, 4), "lbp": round(s_lbp, 4), "fusion": round(s_fusion, 4)}


def decide(scores: dict, matching_cfg: dict) -> tuple[bool, float, float, str]:
    method = matching_cfg.get("decision_method", "fusion")
    thr = float(matching_cfg["thresholds"][method])
    score = float(scores[method])
    return score >= thr, score, thr, method
