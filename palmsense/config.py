"""
config.py - Central settings for PalmSense.

All tunable numbers (thresholds, quality limits, sizes) live in settings.json
so the group can change them without touching code. The Dashboard page writes
new calibrated thresholds back to that file.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "palmsense.db"
SETTINGS_PATH = ROOT / "settings.json"
KB_PATH = ROOT / "knowledge_base" / "palmistry_kb.json"

# Secrets come ONLY from environment variables / .env (never hard-coded).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
ADMIN_PIN = os.getenv("ADMIN_PIN", "").strip()
TEMPLATE_KEY = os.getenv("PALMSENSE_TEMPLATE_KEY", "").strip()

DEFAULT_SETTINGS = {
    "image": {"max_side": 800, "min_side": 300, "roi_size": 256},
    "quality": {
        "min_brightness": 45,
        "max_brightness": 225,
        "max_clipped_fraction": 0.25,
        "min_hand_fraction": 0.06,
        "max_hand_fraction": 0.90,
        "min_palm_radius_px": 45,
        "min_sharpness": 6.0,
    },
    # Similarity convention: HIGHER score = more similar. Accept when score >= threshold.
    "matching": {
        "decision_method": "fusion",
        "thresholds": {"lines": 0.54, "lbp": 0.994, "fusion": 0.56},
        # min-max normalisation used only for fusion: (score - lo) / (hi - lo)
        "norm": {"lines": [0.05, 0.97], "lbp": [0.988, 0.998]},
        "fusion_weight_lines": 0.5,
        "calibrated_on": "synthetic test set (re-calibrate with your real dataset in Dashboard > Evaluation)",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return _merge(DEFAULT_SETTINGS, json.loads(SETTINGS_PATH.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
