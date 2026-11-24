"""
Command-line evaluation (same as Dashboard > Evaluation).

    python tools/evaluate.py dataset_synthetic
    python tools/evaluate.py dataset --hand right --save
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from palmsense.config import load_settings, save_settings  # noqa: E402
from palmsense.evaluation import run_evaluation  # noqa: E402


def pct(x):
    return "not measurable" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--save", action="store_true", help="write calibrated thresholds to settings.json")
    a = ap.parse_args()
    s = load_settings()
    r = run_evaluation(a.dataset, s, a.hand)
    print(f"Calibration people: {r['calibration_people']}\nTest people: {r['test_people']}")
    print(f"{'Method':8} {'Thr':>8} {'FAR':>8} {'FRR':>8} {'Acc':>8}  genuine/impostor  TP FN FP TN")
    for m, v in r["results"].items():
        print(f"{m:8} {v['threshold']:8.4f} {pct(v['FAR']):>8} {pct(v['FRR']):>8} {pct(v['accuracy']):>8}"
              f"  {v['genuine_trials']:>7}/{v['impostor_trials']:<8} {v['TP']:>3}{v['FN']:>3}{v['FP']:>3}{v['TN']:>3}")
    print(f"Mean latency: {r['mean_latency_ms']:.1f} ms | invalid test images: {len(r['failure_to_acquire'])}")
    for n in r["notes"]:
        print("note:", n)
    Path("evaluation_results.json").write_text(json.dumps(
        {k: r[k] for k in ("results", "calibrated_matching", "calibration_people", "test_people",
                           "mean_latency_ms", "failure_to_acquire", "notes")}, indent=2))
    print("Saved evaluation_results.json")
    if a.save:
        s["matching"]["thresholds"] = r["calibrated_matching"]["thresholds"]
        s["matching"]["norm"] = r["calibrated_matching"]["norm"]
        s["matching"]["calibrated_on"] = a.dataset
        save_settings(s)
        print("Calibrated thresholds saved to settings.json")


if __name__ == "__main__":
    main()
