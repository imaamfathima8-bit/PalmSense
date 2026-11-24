# 🖐️ PalmSense: Palm Biometric Authentication + AI Palm Reading

**NIBM KIC · Digital Image Processing · Combined In-Class Assessment 01 + 02**

> *Your palm is your password, and your story.*

PalmSense is one Streamlit application with **one shared image-processing pipeline** used for two jobs:

| Part | What it does | Brief |
|---|---|---|
| 🔐 **Authentication** | Enroll 3+ palm photos → verify a claimed ID with a fresh photo → *Authentication Successful / Failed / Invalid Image* | ICA 02 |
| 🔮 **Palm Reading** | Upload left + right palm → detect heart/head/life/fate lines → palmistry report (RAG + LLM) → chatbot | ICA 01 |

⚠️ Educational prototype. It is **not** production security (no liveness detection, so a printed palm photo may fool it). Palmistry is for entertainment and is not scientifically validated.

---

## 1. Setup (5 minutes)

```bash
# 1. Python 3.10 – 3.13
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configuration
copy .env.example .env        # (macOS/Linux: cp .env.example .env)
#    -> add GEMINI_API_KEY (optional) and set ADMIN_PIN

# 3. Run
streamlit run app.py
```
The browser opens at **http://localhost:8501**.

**No API key?** The app still works: the report and chatbot switch to an **offline template engine** that uses the same knowledge base (RAG). The sidebar shows 🟢 online or 🟡 offline.

---

## 2. Project structure

```
PalmSense/
├── app.py                     # Streamlit entry point + navigation
├── settings.json              # thresholds & quality limits (sample configuration)
├── schema.sql                 # database schema
├── .env.example               # secrets template (copy to .env)
├── palmsense/                 # ---- the engine ----
│   ├── hand.py                # acquisition, quality checks, segmentation, alignment, ROI
│   ├── features.py            # preprocessing + feature A (palm lines) + feature B (LBP)
│   ├── pipeline.py            # analyze_palm(): runs every step, used by ALL pages
│   ├── matching.py            # templates, cosine / chi-square similarity, fusion, decision
│   ├── storage.py             # SQLite, Fernet-encrypted templates, audit log
│   ├── evaluation.py          # FAR, FRR, accuracy, EER, ROC, calibration/test split
│   ├── palm_lines.py          # heart/head/life/fate line detection (RANSAC), palm shape
│   ├── rag.py                 # TF-IDF retrieval over the palmistry knowledge base
│   ├── llm.py                 # Gemini report + chatbot, offline fallback, safety prompt
│   ├── config.py, ui.py       # settings + shared UI style
├── views/                     # one file per page (home, enroll, verify, reading, chat, dashboard, privacy)
├── knowledge_base/palmistry_kb.json   # 51 curated palmistry entries (our own wording)
├── tools/
│   ├── make_synthetic_dataset.py      # creates DRAWN test hands (not real biometrics)
│   └── evaluate.py                    # command-line evaluation
├── dataset_synthetic/         # 12 fake people × 5 images, for testing
└── demo_images/               # left/right demo palms + 3 invalid examples
```

---

## 3. The pipeline (same for enrollment, verification and reading)

| # | Step | Technique | Why |
|---|---|---|---|
| 1 | Acquisition | Upload JPG/PNG or webcam; EXIF rotation fix; **resize** to 800 px | same scale for every photo |
| 2 | Quality control | format/corruption, resolution ≥ 300 px, palm brightness, **Laplacian variance** (blur) | reject unusable images → *Invalid Image* |
| 3 | Segmentation | **YCrCb colour space, Cr channel + Otsu threshold**, morphology open/close, largest contour | skin has high Cr for all skin tones |
| 4 | Landmarks | **Convexity defects** = valleys between fingers; thumb valley → left/right hand | reference points |
| 5 | Alignment | image moments (long axis vertical) + rotate so finger valleys are level | same palm region even if hand is rotated |
| 6 | ROI | **Distance transform** → largest circle in the palm → square crop, 256×256, left hands mirrored | consistent palm region |
| 7 | Preprocess | **grayscale → Gaussian blur → CLAHE → normalisation** | remove noise, boost line contrast |
| 8 | Feature A | **Black-hat morphology** → line map → 32×32 grid (1024 values) | *where are the palm lines?* |
| 9 | Feature B | **Uniform LBP** (P=8, R=2), 4×4 grid histograms (160 values) | *what does the skin texture look like?* |
| 10 | Matching | A: **cosine similarity**; B: **1 − chi-square distance**; fusion = average of min-max normalised scores | |
| 11 | Decision | **similarity ≥ threshold → accept** (threshold calibrated at EER) | |
| 12 | Reading | skeletonise line map → **RANSAC curve fitting** per palm zone → confidence | detect heart/head/life/fate lines |

Every intermediate image (12 stages) is shown in the app.

---

## 4. Biometric evaluation protocol (ICA 02 §8 & §10)

1. Collect **≥10 consenting participants × 5 photos of the same hand** → `dataset/P01/1.jpg … 5.jpg`.
2. Photos **1–3 → enrollment template**, photos **4–5 → genuine tests** (never used for enrollment).
3. Each test photo vs every *other* person's template → **impostor tests**.
4. People are split: **even = calibration group** (chooses the threshold at the EER point), **odd = test group** (reports final metrics). No leakage.
5. Run **Dashboard → Evaluation** (or `python tools/evaluate.py dataset --save`).
6. Report FAR, FRR, accuracy, mean latency, trial counts, confusion matrix, ROC. A zero denominator shows *not measurable*.

Example with the synthetic test set (12 people, 6 for calibration and 6 for testing → 12 genuine + 60 impostor test trials):

| Method | Threshold | FAR | FRR | Accuracy |
|---|---|---|---|---|
| A · Palm lines | 0.535 | 0.0% | 0.0% | 100% |
| B · LBP texture | 0.994 | 0.0% | 0.0% | 100% |
| A+B · Fusion | 0.560 | 0.0% | 0.0% | 100% |

These synthetic results only prove the code works. **Real palms will score lower.** Re-calibrate with your real dataset and report those numbers.

---

## 5. Palm reading + LLM (ICA 01 §04)

* **Model:** Google Gemini (`GEMINI_MODEL`, default `gemini-flash-latest`) through the `google-genai` SDK.
* **Specialisation:** **RAG.** A curated knowledge base of 51 palmistry entries; TF-IDF + cosine similarity retrieves the entries that match the *detected* features.
* **Input:** structured JSON features of both palms (status, confidence, length, shape, clarity, palm shape) + measured left/right differences + retrieved entries.
* **Output:** markdown report with 8 fixed sections (lines, personality, career, finance, relationships, future (non-scientific), left vs right, uncertainty).
* **Safeguards:** a system prompt forbids inventing features, health/lifespan predictions and financial advice; uncertain lines are labelled tentative; a disclaimer is always included; the UI separates **👁 Observed by computer vision** from **✨ Interpreted by AI**.
* **Chatbot:** multi-turn; the context holds both palms' features, the report and newly retrieved entries for each question.

---

## 6. Privacy & security

Consent checkbox · pseudonymous IDs · **no raw images stored** · templates **Fernet-encrypted** · minimal audit log (no images) · deletion with admin PIN · secrets only in `.env` (git-ignored) · `data/` and `dataset/` git-ignored · known limitation: spoofing.

## 7. Tuning

All numbers live in `settings.json` (quality limits such as `min_sharpness`, thresholds). If real photos are rejected as blurry, lower `min_sharpness`.

## 8. Acknowledgements

OpenCV, NumPy, scikit-image, scikit-learn, Streamlit, Plotly, cryptography, Google Gemini API. Palmistry knowledge is paraphrased from traditional sources (e.g. Cheiro, *Language of the Hand*, 1897, public domain). The code was developed with AI assistance (Claude), as permitted by the lecturer; every team member must be able to explain it.
