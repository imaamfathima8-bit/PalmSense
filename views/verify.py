import time

import plotly.graph_objects as go
import streamlit as st

from palmsense import storage
from palmsense.config import load_settings
from palmsense.matching import compare, decide
from palmsense.pipeline import analyze_palm
from palmsense.ui import PHOTO_TIPS, hero, image_input, result_banner, section, stage_gallery, stat

OK, FAIL, INVALID = "Authentication Successful", "Authentication Failed", "Invalid Image"


def _gauge(score, thr, method):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"valueformat": ".3f", "font": {"size": 42}},
        title={"text": f"{method} similarity score", "font": {"size": 14}},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#A78BFA", "thickness": 0.3},
               "steps": [{"range": [0, thr], "color": "rgba(244,63,94,.25)"},
                         {"range": [thr, 1], "color": "rgba(16,185,129,.25)"}],
               "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.9, "value": thr}}))
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#E7EAF3"})
    return fig


def render():
    settings = load_settings()
    m = settings["matching"]
    hero("Step 2 · Verification (1:1)", "Prove it's you",
         "Claim your user ID and show a fresh photo of the same palm. "
         "The new features are compared only with YOUR stored template.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        section("Claim an identity")
        users = [u["user_id"] for u in storage.list_users()]
        user_id = st.text_input("User ID", placeholder="e.g. S01").strip()
        if users:
            st.caption("Enrolled IDs: " + ", ".join(users[:15]))
        st.info(f"Decision rule: **{m['decision_method']}** score ≥ **{m['thresholds'][m['decision_method']]}** "
                f"→ accept. Higher score = more similar.", icon="⚖️")
    with right:
        section("Fresh palm photo")
        st.caption(PHOTO_TIPS)
        data = image_input("Image source", "verify")

    if not st.button("🔐 Verify identity", type="primary", width="stretch", disabled=data is None):
        return

    t0 = time.perf_counter()
    # ---------- FR10: unknown user / missing template ----------
    user = storage.get_user(user_id) if user_id else None
    if user is None:
        aid = storage.log_attempt(user_id or "(empty)", FAIL, "Unknown user ID")
        result_banner("fail", FAIL, f"Unknown user ID '{user_id}'. Please enroll first. (Attempt {aid})")
        return
    template = storage.load_template(user_id)
    if template is None:
        aid = storage.log_attempt(user_id, FAIL, "No template stored")
        result_banner("fail", FAIL, f"No biometric template found for '{user_id}'. Please re-enroll. (Attempt {aid})")
        return

    # ---------- same pipeline as enrollment ----------
    with st.spinner("Processing palm image…"):
        r = analyze_palm(data, user["hand_side"], settings)
    if not r.ok:
        ms = (time.perf_counter() - t0) * 1000
        aid = storage.log_attempt(user_id, INVALID, r.reason, duration_ms=round(ms, 1))
        result_banner("warn", INVALID, f"{r.reason} (Attempt {aid})")
        if r.stages:
            with st.expander("What the system saw"):
                stage_gallery(r.stages, cols=4)
        return

    # Wrong hand? (detected thumb side differs from the enrolled one)
    if r.info["thumb_source"] == "detected" and user["thumb_side"] and r.info["thumb_side"] != user["thumb_side"]:
        ms = (time.perf_counter() - t0) * 1000
        reason = f"Wrong hand - this user enrolled the {user['hand_side']} hand."
        aid = storage.log_attempt(user_id, INVALID, reason, duration_ms=round(ms, 1))
        result_banner("warn", INVALID, f"{reason} (Attempt {aid})")
        return

    scores = compare(r.vectors, template, m)
    accepted, score, thr, method = decide(scores, m)
    ms = (time.perf_counter() - t0) * 1000
    outcome = OK if accepted else FAIL
    aid = storage.log_attempt(user_id, outcome, "", method, score, thr, scores["lines"], scores["lbp"], round(ms, 1))

    if accepted:
        st.session_state["verified_user"] = user_id
        result_banner("ok", OK, f"Welcome back, {user_id}! Score {score:.3f} ≥ threshold {thr:.3f} · {ms:.0f} ms · {aid}")
        st.balloons()
    else:
        result_banner("fail", FAIL, f"Score {score:.3f} < threshold {thr:.3f}. This palm does not match {user_id}. · {aid}")

    g, t = st.columns([1, 1.1], gap="large")
    with g:
        st.plotly_chart(_gauge(score, thr, method.title()), width="stretch")
    with t:
        section("Score breakdown", "Each method on its own, plus the fused decision score")
        rows = []
        for meth, label, metric in (("lines", "A · Palm lines", "cosine similarity"),
                                    ("lbp", "B · LBP texture", "1 − chi-square distance"),
                                    ("fusion", "A + B · Fusion", "normalised average")):
            th = m["thresholds"][meth]
            rows.append({"Method": label, "Metric": metric, "Score": scores[meth], "Threshold": th,
                         "Result": "✅ match" if scores[meth] >= th else "❌ no match",
                         "Used for decision": "⭐" if meth == method else ""})
        st.dataframe(rows, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            stat("Processing time", f"{ms:.0f} ms", "end-to-end")
        with c2:
            stat("Attempt ID", aid, "saved to audit log")

    a, b = st.columns(2)
    with a:
        st.image(r.stages["3. Detection & alignment"], caption="Hand detection, finger valleys, palm circle & ROI",
                 width="stretch")
    with b:
        st.image(r.stages["10. Line map (threshold + morphology)"], caption="Extracted palm-line map (probe)",
                 width="stretch")
    with st.expander("🔬 All 12 image-processing stages"):
        stage_gallery(r.stages, cols=4)
