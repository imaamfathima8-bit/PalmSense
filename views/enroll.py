import re

import numpy as np
import streamlit as st

from palmsense import storage
from palmsense.config import load_settings
from palmsense.matching import build_template, cosine_similarity
from palmsense.pipeline import analyze_palm
from palmsense.ui import PHOTO_TIPS, hero, pill, result_banner, section, stage_gallery

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


def render():
    settings = load_settings()
    hero("Step 1 · Enrollment", "Register your palm",
         "Choose a pseudonymous user ID, give consent and upload at least 3 photos of the SAME palm. "
         "We store only an encrypted numerical template - never your photos.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        section("1 · Your details")
        user_id = st.text_input("User ID (pseudonymous, e.g. S01 or student_07)", max_chars=20).strip()
        hand = st.radio("Which hand will you enroll?", ["right", "left"], horizontal=True,
                        format_func=lambda h: f"{'🫱' if h == 'right' else '🫲'} {h.title()} hand",
                        help="Use the same hand every time you verify.")
        consent = st.checkbox("I consent to storing my encrypted palm template",
                              help="For this class project only. You can withdraw and delete it any time "
                                   "on the Privacy & Admin page.")

    with right:
        section("2 · Palm photos (minimum 3)")
        st.caption(PHOTO_TIPS)
        tab_up, tab_cam = st.tabs(["📁 Upload photos", "📷 Use camera"])
        with tab_up:
            files = st.file_uploader("Upload 3-5 photos", type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True, label_visibility="collapsed")
        with tab_cam:
            shot = st.camera_input("Capture", label_visibility="collapsed", key="enroll_cam")
            c1, c2 = st.columns(2)
            if c1.button("➕ Add this capture", disabled=shot is None, width="stretch"):
                st.session_state.setdefault("enroll_caps", []).append(shot.getvalue())
            if c2.button("🗑️ Clear captures", width="stretch"):
                st.session_state["enroll_caps"] = []
            st.caption(f"Captured photos: {len(st.session_state.get('enroll_caps', []))}")

    images = [(f.name, f.getvalue()) for f in (files or [])]
    images += [(f"camera_{i + 1}.png", b) for i, b in enumerate(st.session_state.get("enroll_caps", []))]

    st.write("")
    go = st.button(f"🧬 Process {len(images)} photo(s) & enroll", type="primary", width="stretch")
    if not go:
        return

    # ---------- validation of the form (FR01 / FR10) ----------
    if not ID_PATTERN.match(user_id or ""):
        result_banner("warn", "Invalid user ID", "Use 3-20 letters, numbers, '_' or '-'. Do not use your real name.")
        return
    if storage.user_exists(user_id):
        result_banner("warn", "User ID already exists", "Choose a different ID, or delete the old enrollment first.")
        return
    if not consent:
        result_banner("warn", "Consent required", "Enrollment needs your informed consent.")
        return
    if len(images) < 3:
        result_banner("warn", "Not enough photos", f"You provided {len(images)}. At least 3 photos of the same palm are required.")
        return

    # ---------- run the pipeline on every photo ----------
    results = []
    with st.status("Running the image-processing pipeline…", expanded=False) as status:
        for name, data in images:
            st.write(f"Processing {name}")
            results.append((name, analyze_palm(data, hand, settings)))
        status.update(label="Pipeline finished", state="complete")

    section("Quality check results")
    cols = st.columns(min(len(results), 5))
    for i, (name, r) in enumerate(results):
        with cols[i % len(cols)]:
            if r.ok:
                st.image(r.stages["4. Palm ROI"], width="stretch")
                st.markdown(f"{pill('Valid', 'ok')} `{name}`<br><small>sharpness {r.info['sharpness']} · "
                            f"rotated {r.info['rotation_deg']}° · {r.duration_ms:.0f} ms</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"{pill('Rejected', 'no')} `{name}`", unsafe_allow_html=True)
                st.caption(r.reason)

    valid = [r for _, r in results if r.ok]
    if len(valid) < 3:
        result_banner("fail", "Enrollment not completed",
                      f"Only {len(valid)} valid photo(s). Replace the rejected photos and try again (need 3).")
        return

    # Same physical hand? Thumb side must agree across the samples (when it was detected).
    sides = {r.info["thumb_side"] for r in valid if r.info["thumb_source"] == "detected"}
    if len(sides) > 1:
        result_banner("fail", "Mixed hands detected",
                      "Some photos look like a left hand and some like a right hand. Use the same palm in every photo.")
        return
    thumb_side = sides.pop() if sides else ("right" if hand == "right" else "left")

    # Consistency between enrollment samples (palm-line similarity)
    n = len(valid)
    sim = np.array([[cosine_similarity(valid[i].vectors["lines"], valid[j].vectors["lines"]) for j in range(n)]
                    for i in range(n)])
    template = build_template([r.vectors for r in valid])
    storage.save_enrollment(user_id, hand, thumb_side, n, template)

    result_banner("ok", f"Enrollment successful - {user_id}",
                  f"{n} valid samples of the {hand} hand averaged into one encrypted template "
                  f"({len(template['lines'])} line values + {len(template['lbp'])} LBP values).")

    a, b = st.columns([1, 1.4], gap="large")
    with a:
        section("Sample consistency", "Palm-line cosine similarity between your enrollment photos")
        import plotly.express as px
        fig = px.imshow(sim, text_auto=".2f", color_continuous_scale="Purples", zmin=0, zmax=1,
                        x=[f"Photo {i + 1}" for i in range(n)], y=[f"Photo {i + 1}" for i in range(n)])
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        section("What was stored?")
        st.markdown(f"""
| Field | Value |
|---|---|
| User ID | `{user_id}` (pseudonymous) |
| Hand | {hand} (thumb on image {thumb_side}) |
| Samples used | {n} |
| Template | encrypted with Fernet (AES-128) |
| Raw photos | **not stored** |
""")
    with st.expander("🔬 See every image-processing stage (first sample)"):
        stage_gallery(valid[0].stages, cols=4)
    st.session_state["enroll_caps"] = []
