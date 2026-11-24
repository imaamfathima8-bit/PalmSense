# 🎤 PalmSense: 5-Member Presentation Guide

**Total time: about 15 minutes (3 minutes each) + Q&A.** Each member explains one part, shows it live in the app, and owns the matching code file.

**One-line story for the opening slide:**
> "We built one app where your palm is your **password** (biometric login) and your **story** (AI palm reading). Both use the same image-processing pipeline."

---

## 👤 Member 1: Problem, Image Acquisition & Quality Check
**Code:** `palmsense/hand.py` → `load_image`, `check_exposure` · **App:** Home page, then upload `demo_images/invalid_*.jpg` on Verify

- **Problem:** passwords and cards can be lost or shared, but your palm is always with you.
- **Acquisition:** upload or webcam → fix phone rotation (EXIF) → **resize** to 800 px so every photo has the same scale.
- **Quality control** rejects bad photos *before* processing:
  - corrupted or wrong format → error
  - too small (< 300 px)
  - too dark or too bright (measured only on the palm)
  - **blurry** → *Laplacian variance*. The Laplacian finds edges, and a blurry photo has weak edges, so the variance is low.
- 🧠 **Analogy:** "Like a bouncer at the door: bad photos never get in."
- **Demo:** upload the blurry photo → **⚠️ Invalid Image**.

## 👤 Member 2: Hand Segmentation, Alignment & ROI
**Code:** `hand.py` → `segment_hand`, `finger_valleys`, `palm_circle`, `locate_palm` · **App:** stages 2–4

- **Segmentation:** convert to **YCrCb**. The **Cr** (red-difference) channel is high for *all* skin tones. **Otsu** picks the best threshold automatically, then **morphology** (opening removes dots, closing fills holes).
- **Finger valleys:** **convexity defects**, the deep gaps between the hand outline and its convex hull.
- **Alignment:** rotate the hand until the finger valleys form a **level line**.
- **ROI:** the **distance transform** finds the largest circle that fits inside the palm. Its centre is the palm centre, and we crop the square around it to 256×256.
- 🧠 **Analogy:** "Before comparing two faces you line up the eyes. We line up the finger gaps."
- **Demo:** stage *3. Detection & alignment* (red dots = valleys, circle = palm, pink square = ROI).

## 👤 Member 3: Preprocessing & Two Feature Methods
**Code:** `palmsense/features.py` · **App:** stages 5–12

- **Preprocessing:** grayscale → **Gaussian blur** (removes noise) → **CLAHE** (boosts contrast in small tiles, so lines stand out) → normalisation.
- **Method A: palm lines.** **Black-hat** = closing(image) − image, which turns dark thin lines bright. Threshold + morphology → line map → 32×32 grid = **1024 numbers**.
- **Method B: LBP texture.** Compare each pixel with its 8 neighbours (brighter = 1, darker = 0) to get an 8-bit pattern code. Histograms in a 4×4 grid = **160 numbers**.
- **Why two?** Lines capture the big structure; LBP captures fine skin texture. Each covers the other's weakness.
- 🧠 **Analogy:** "Method A is the map of roads, Method B is the texture of the ground."

## 👤 Member 4: Templates, Matching, Decision & Evaluation
**Code:** `matching.py`, `storage.py`, `evaluation.py` · **App:** Enroll → Verify (genuine + impostor) → Dashboard

- **Enrollment:** 3 photos → average the features → **one template**, **encrypted** (Fernet) in SQLite. No photos are stored.
- **Verification (1:1):** new photo → same pipeline → compare with the claimed user's template only.
  - A: **cosine similarity** · B: **1 − chi-square distance** · **Fusion** = average of both (normalised)
  - **Higher score = more similar. Accept if score ≥ threshold.**
- **Evaluation:** photos 1–3 enroll, 4–5 test. Other people's photos act as **impostors**.
  - **FAR** = impostors accepted ÷ impostor attempts (lower = safer)
  - **FRR** = genuine users rejected ÷ genuine attempts (lower = more convenient)
  - Threshold chosen where FAR = FRR (**EER**) on a *calibration* group, then tested on *different* people (no leakage).
- 🧠 **Analogy:** "The threshold is like the height bar at a theme park: too high and real riders are turned away (FRR), too low and anyone gets on (FAR)."
- **Demo:** genuine → ✅ (score ≈ 0.97), impostor → ⛔ (≈ 0.30), unknown ID → ⛔, Dashboard confusion matrix + ROC.

## 👤 Member 5: Palm Reading, RAG + LLM, Chatbot & Privacy
**Code:** `palm_lines.py`, `rag.py`, `llm.py`, `views/privacy.py` · **App:** Palm Reading → Ask PalmSense → Privacy

- **Line detection:** thin the line map to 1 pixel (**skeletonisation**). Search each line only in its usual zone (heart = top, head = middle, life = around the thumb, fate = centre). **RANSAC** fits the longest smooth curve and ignores random creases. Confidence → *Detected / Uncertain / Not visible*.
- **RAG:** we did not train an LLM. We wrote a **knowledge base of 51 palmistry entries**; **TF-IDF** finds the entries matching *our detected features*; only those + the features go to **Gemini**.
- **Safety:** no invented features, no health/lifespan predictions, always a disclaimer. **👁 Observed** (computer vision) is shown separately from **✨ Interpreted** (AI).
- **Chatbot:** remembers the conversation plus your features and report.
- **Privacy:** consent, pseudonymous IDs, encrypted templates, delete on request, secrets in `.env`. **Limitation:** a printed photo can spoof it (no liveness detection).
- 🧠 **Analogy:** "RAG is an open-book exam: the AI may only answer from the pages we hand it."

---

## ❓ Likely viva questions (everyone should know these)

| Question | Short answer |
|---|---|
| Why the Cr channel? | Skin is reddish; Cr separates skin from background across skin tones and is less affected by brightness than RGB. |
| Why CLAHE and not normal histogram equalisation? | CLAHE works on small tiles and limits contrast, so it doesn't over-amplify noise. |
| Why black-hat? | Palm lines are dark and thin; black-hat extracts exactly "dark thin things" and ignores lighting gradients. |
| Why mirror left hands? | So every palm has the thumb on the same side, letting the same zones and comparisons work. |
| Similarity or distance? | Similarity, so higher = more similar, and we accept if score ≥ threshold. |
| How was the threshold chosen? | At the Equal Error Rate on a calibration group, then tested on separate people. |
| Why can't you reach 0% FAR and 0% FRR on real data? | Lighting, pose and small datasets overlap the score distributions, so lowering one raises the other. |
| Is the palm reading true? | No. Palmistry is entertainment; we only measure lines and quote tradition. |
| What if there's no internet / API key? | The offline engine writes the report from the same knowledge base. |
| What's stored in the database? | User ID, hand side, encrypted feature vectors and a numeric audit log. Never photos. |

## ✅ Demo checklist (before presenting)
1. `streamlit run app.py` works and the sidebar says online/offline as expected.
2. Enroll **S01** with `dataset_synthetic/P01/1-3.jpg` (or real photos), and **S02** with `P02/1-3.jpg`.
3. Verify S01 with `P01/4.jpg` ✅, then S01 with `P02/4.jpg` ⛔, then `demo_images/invalid_blurry.jpg` ⚠️.
4. Palm Reading: `demo_images/left_palm.jpg` + `right_palm.jpg` → Generate report → Chat.
5. Dashboard → Run evaluation.
6. Swap in real palm photos taken with consent before the final demo.
