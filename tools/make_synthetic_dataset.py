"""
make_synthetic_dataset.py - Creates FAKE (drawn) hand images for testing the pipeline.

These are NOT real biometrics. They exist only so the app and the evaluation
can be tested before the group collects real, consented palm photos.

    python tools/make_synthetic_dataset.py --people 12 --images 5 --out dataset_synthetic
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def _curve(p0, p1, p2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
    return pts.astype(np.int32)


def make_person(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "skin": np.array([65, 95, 140]) + rng.uniform(0, 1) * np.array([75, 80, 85]),
        "palm_w": rng.uniform(340, 400), "palm_h": rng.uniform(360, 430),
        "finger_len": rng.uniform(0.85, 1.15),
        "heart": (rng.uniform(-30, 30), rng.uniform(-60, 10), rng.uniform(-20, 20)),
        "head": (rng.uniform(-25, 25), rng.uniform(-20, 50), rng.uniform(-30, 30)),
        "life": (rng.uniform(-30, 30), rng.uniform(-40, 40)),
        "fate": rng.random() < 0.6,
        "fate_x": rng.uniform(-40, 40),
        "texture_seed": int(rng.integers(1e9)),
    }


def draw_hand(p: dict, sample_seed: int, left=False, size=(900, 1200)):
    rng = np.random.default_rng(sample_seed)
    W, H = size
    bg_col = rng.choice([[40, 40, 40], [235, 235, 235], [90, 60, 40], [20, 20, 25], [180, 180, 175]])
    img = np.full((H, W, 3), bg_col, np.float32) + rng.normal(0, 4, (H, W, 3))
    mask = np.zeros((H, W), np.uint8)

    cx, cy = W // 2, 760
    pw, ph = int(p["palm_w"]), int(p["palm_h"])
    top, bottom = cy - ph // 2, cy + ph // 2
    cv2.rectangle(mask, (cx - pw // 2 + 40, top), (cx + pw // 2 - 40, bottom), 255, -1)
    cv2.rectangle(mask, (cx - pw // 2, top + 40), (cx + pw // 2, bottom - 40), 255, -1)
    for dx in (-1, 1):
        for y in (top + 40, bottom - 40):
            cv2.circle(mask, (cx + dx * (pw // 2 - 40), y), 40, 255, -1)
    # forearm
    cv2.rectangle(mask, (cx - pw // 2 + 50, bottom - 60), (cx + pw // 2 - 70, H), 255, -1)
    # fingers: little, ring, middle, index (index next to thumb on the right)
    fw = pw / 4.6
    lens = np.array([0.55, 0.72, 0.80, 0.70]) * ph * p["finger_len"]
    for i, L in enumerate(lens):
        fx = int(cx - pw / 2 + fw * 0.62 + i * (pw - fw * 1.24) / 3)
        cv2.line(mask, (fx, top + 30), (fx, int(top - L)), 255, int(fw * 0.82))
        cv2.circle(mask, (fx, int(top - L)), int(fw * 0.41), 255, -1)
    # thumb
    cv2.line(mask, (cx + pw // 2 - 70, cy + 80), (cx + pw // 2 + 110, cy - 110), 255, int(fw * 1.05))
    cv2.circle(mask, (cx + pw // 2 + 110, cy - 110), int(fw * 0.52), 255, -1)

    # skin + person-specific fine texture
    skin = np.zeros((H, W, 3), np.float32) + p["skin"] * rng.uniform(0.95, 1.05)
    tex_rng = np.random.default_rng(p["texture_seed"])
    shade = np.zeros((H, W), np.float32)
    for _ in range(260):
        x, y = tex_rng.uniform(cx - pw / 2, cx + pw / 2), tex_rng.uniform(top, bottom)
        a, L = tex_rng.uniform(0, np.pi), tex_rng.uniform(15, 55)
        q0 = (x, y)
        q2 = (x + L * np.cos(a), y + L * np.sin(a))
        q1 = ((q0[0] + q2[0]) / 2 + tex_rng.uniform(-8, 8), (q0[1] + q2[1]) / 2 + tex_rng.uniform(-8, 8))
        cv2.polylines(shade, [_curve(q0, q1, q2, 20)], False, float(tex_rng.uniform(14, 30)), 2)
    # principal lines
    h1, h2, h3 = p["heart"]
    heart = _curve((cx - pw / 2 + 10, top + 95 + h1), (cx, top + 75 + h3), (cx + pw / 4, top + 25 + h2))
    g1, g2, g3 = p["head"]
    head = _curve((cx + pw / 2 - 25, top + 150 + g1), (cx, top + 170 + g3), (cx - pw / 2 + 50, top + 200 + g2))
    l1, l2 = p["life"]
    life = _curve((cx + pw / 2 - 30, top + 160), (cx + pw / 6 + l1, cy + 40 + l2), (cx + pw / 8, bottom - 5))
    for line, w in ((heart, 5), (head, 5), (life, 5)):
        cv2.polylines(shade, [line], False, 60.0, w)
    if p["fate"]:
        fate = _curve((cx + p["fate_x"], bottom - 10), (cx + p["fate_x"] + 5, cy), (cx + p["fate_x"], top + 170))
        cv2.polylines(shade, [fate], False, 45.0, 4)
    shade = cv2.GaussianBlur(shade, (5, 5), 0)
    skin -= shade[:, :, None] * np.array([0.8, 0.9, 1.0])
    skin += rng.normal(0, 3, (H, W, 3))
    m3 = (cv2.GaussianBlur(mask, (5, 5), 0) / 255.0)[:, :, None]
    img = img * (1 - m3) + skin * m3

    # capture variation: rotation, scale, shift, brightness, blur
    ang, sc = rng.uniform(-18, 18), rng.uniform(0.85, 1.05)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    M[:, 2] += rng.uniform(-40, 40, 2)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    img = img * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12)
    img = np.clip(img, 0, 255).astype(np.uint8)
    k = int(rng.choice([0, 3]))
    if k:
        img = cv2.GaussianBlur(img, (k, k), 0)
    if left:
        img = cv2.flip(img, 1)
    return img


def save(path, img):
    """Store at 600x800, JPEG quality 85 to keep the repository small."""
    img = cv2.resize(img, (img.shape[1] * 2 // 3, img.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=12)
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--out", default="dataset_synthetic")
    a = ap.parse_args()
    out = Path(a.out)
    for i in range(a.people):
        person = make_person(1000 + i)
        d = out / f"P{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(a.images):
            save(d / f"{j + 1}.jpg", draw_hand(person, 50_000 + i * 100 + j))
    # demo extras: a left/right pair and some invalid images
    demo = out.parent / "demo_images"
    demo.mkdir(exist_ok=True)
    p = make_person(777)
    save(demo / "right_palm.jpg", draw_hand(p, 1))
    save(demo / "left_palm.jpg", draw_hand(make_person(778), 2, left=True))
    blank = np.full((1200, 900, 3), 200, np.uint8)
    save(demo / "invalid_no_hand.jpg", blank)
    save(demo / "invalid_blurry.jpg", cv2.GaussianBlur(draw_hand(p, 3), (41, 41), 0))
    save(demo / "invalid_dark.jpg", (draw_hand(p, 4) * 0.12).astype(np.uint8))
    print(f"Synthetic dataset written to {out} and demo images to {demo}")


if __name__ == "__main__":
    main()
