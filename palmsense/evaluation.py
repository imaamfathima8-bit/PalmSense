"""
evaluation.py - STEP 6: Testing the biometric system (FAR, FRR, accuracy, latency).

Dataset layout (one folder per consenting participant, same hand in every photo):

    dataset/
        P01/ 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
        P02/ ...

Protocol (no data leakage):
  * Photos 1-3 of each person -> enrollment template.
  * Photos 4-5 of each person -> genuine test attempts (never used for enrollment).
  * A person's test photos against OTHER people's templates -> impostor attempts.
  * People are split into two groups:
        CALIBRATION group -> used ONLY to choose the threshold (at the EER point)
        TEST group        -> used ONLY to report the final FAR / FRR / accuracy
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .matching import build_template, compare
from .pipeline import analyze_palm

IMG_EXT = {".jpg", ".jpeg", ".png"}
METHODS = ("lines", "lbp", "fusion")


def _person_folders(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _images(folder: Path):
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT],
                  key=lambda f: (len(f.stem), f.stem))


def _process_group(people, hand, settings, progress=None, done=0, total=1):
    """Enroll each person (first 3 photos) and analyse their test photos."""
    templates, probes, notes = {}, [], []
    for person in people:
        imgs = _images(person)
        if len(imgs) < 4:
            notes.append(f"{person.name}: needs at least 4 photos (has {len(imgs)}) - skipped")
            continue
        enroll_vecs = []
        for f in imgs[:3]:
            r = analyze_palm(f.read_bytes(), hand, settings)
            done += 1
            if progress:
                progress(done / total)
            if r.ok:
                enroll_vecs.append(r.vectors)
            else:
                notes.append(f"{person.name}/{f.name} (enrollment) invalid: {r.reason}")
        if len(enroll_vecs) < 3:
            notes.append(f"{person.name}: fewer than 3 valid enrollment photos - skipped")
            done += len(imgs[3:])
            continue
        templates[person.name] = build_template(enroll_vecs)
        for f in imgs[3:]:
            t0 = time.perf_counter()
            r = analyze_palm(f.read_bytes(), hand, settings)
            done += 1
            if progress:
                progress(done / total)
            probes.append({"person": person.name, "file": f.name, "ok": r.ok, "reason": r.reason,
                           "vectors": r.vectors, "t_process": time.perf_counter() - t0})
    return templates, probes, notes, done


def _score_trials(templates, probes, matching_cfg):
    trials = []
    for p in probes:
        if not p["ok"] or p["person"] not in templates:
            continue
        for claimed, tpl in templates.items():
            t0 = time.perf_counter()
            s = compare(p["vectors"], tpl, matching_cfg)
            dt = p["t_process"] + (time.perf_counter() - t0)
            trials.append({"probe": f'{p["person"]}/{p["file"]}', "claimed": claimed,
                           "genuine": claimed == p["person"], "latency_ms": dt * 1000, **s})
    return trials


def rates(trials, method, thr):
    gen = [t[method] for t in trials if t["genuine"]]
    imp = [t[method] for t in trials if not t["genuine"]]
    tp = sum(s >= thr for s in gen)
    fn = len(gen) - tp
    fp = sum(s >= thr for s in imp)
    tn = len(imp) - fp
    far = fp / len(imp) if imp else None      # None = not measurable
    frr = fn / len(gen) if gen else None
    total = len(gen) + len(imp)
    acc = (tp + tn) / total if total else None
    return {"FAR": far, "FRR": frr, "accuracy": acc, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "genuine_trials": len(gen), "impostor_trials": len(imp)}


def eer_threshold(trials, method):
    """Pick the threshold where FAR and FRR are (closest to) equal.
    Candidate thresholds are the mid-points between neighbouring scores, so a
    perfectly separated set gets a threshold in the middle of the gap."""
    scores = sorted({t[method] for t in trials})
    if not scores:
        return 0.5, None
    cands = [scores[0] - 1e-3] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-3]
    rows = []
    for thr in cands:
        r = rates(trials, method, thr)
        if r["FAR"] is None or r["FRR"] is None:
            continue
        rows.append((abs(r["FAR"] - r["FRR"]), r["FAR"] + r["FRR"], thr, (r["FAR"] + r["FRR"]) / 2))
    if not rows:
        return 0.5, None
    best = min(rows, key=lambda x: (x[0], x[1]))
    ties = [x for x in rows if abs(x[0] - best[0]) < 1e-12 and abs(x[1] - best[1]) < 1e-12]
    chosen = ties[len(ties) // 2]
    return float(chosen[2]), chosen[3]


def roc_points(trials, method):
    scores = sorted({t[method] for t in trials}) or [0.0]
    pts = []
    for thr in [scores[0] - 1e-3] + scores + [scores[-1] + 1e-3]:
        r = rates(trials, method, thr)
        if r["FAR"] is not None and r["FRR"] is not None:
            pts.append({"threshold": thr, "FAR": r["FAR"], "TAR": 1 - r["FRR"]})
    return pts


def run_evaluation(dataset_dir: str, settings: dict, hand: str = "right", progress=None) -> dict:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    people = _person_folders(root)
    if len(people) < 2:
        raise ValueError("Need at least 2 participant folders (impostor trials need other people).")

    independent = len(people) >= 4
    calib_people = people[0::2] if independent else people
    test_people = people[1::2] if independent else people
    total = sum(len(_images(p)) for p in people) * (1 if independent else 1)

    cal_tpl, cal_probes, notes1, done = _process_group(calib_people, hand, settings, progress, 0, total)
    if independent:
        test_tpl, test_probes, notes2, _ = _process_group(test_people, hand, settings, progress, done, total)
    else:
        test_tpl, test_probes, notes2 = cal_tpl, cal_probes, []

    # --- 1) Calibrate fusion normalisation and thresholds on the CALIBRATION group ---
    m = dict(settings["matching"])
    m = {**m, "norm": dict(m["norm"]), "thresholds": dict(m["thresholds"])}
    cal_trials = _score_trials(cal_tpl, cal_probes, m)
    for meth in ("lines", "lbp"):
        g = [t[meth] for t in cal_trials if t["genuine"]]
        i = [t[meth] for t in cal_trials if not t["genuine"]]
        if g and i:
            lo, hi = float(np.percentile(i, 5)), float(np.percentile(g, 95))
            if hi > lo:
                m["norm"][meth] = [round(lo, 4), round(hi, 4)]
    cal_trials = _score_trials(cal_tpl, cal_probes, m)          # re-score fusion with new norm
    cal_eer = {}
    for meth in METHODS:
        thr, eer = eer_threshold(cal_trials, meth)
        m["thresholds"][meth] = round(thr, 4)
        cal_eer[meth] = eer

    # --- 2) Report on the independent TEST group with the frozen thresholds ---
    test_trials = _score_trials(test_tpl, test_probes, m)
    results = {}
    for meth in METHODS:
        r = rates(test_trials, meth, m["thresholds"][meth])
        _, test_eer = eer_threshold(test_trials, meth)
        r.update({"threshold": m["thresholds"][meth], "calibration_EER": cal_eer[meth], "test_EER": test_eer})
        results[meth] = r

    invalid = [p for p in test_probes if not p["ok"]]
    lat = [t["latency_ms"] for t in test_trials if t["claimed"] == t["probe"].split("/")[0]]
    return {
        "independent_split": independent,
        "calibration_people": [p.name for p in calib_people],
        "test_people": [p.name for p in test_people],
        "calibrated_matching": m,
        "results": results,
        "test_trials": test_trials,
        "calibration_trials": cal_trials,
        "roc": {meth: roc_points(test_trials, meth) for meth in METHODS},
        "failure_to_acquire": [{"probe": f'{p["person"]}/{p["file"]}', "reason": p["reason"]} for p in invalid],
        "mean_latency_ms": float(np.mean(lat)) if lat else None,
        "notes": notes1 + notes2,
    }
