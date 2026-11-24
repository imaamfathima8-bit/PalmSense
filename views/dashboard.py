import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from palmsense import storage
from palmsense.config import ROOT, load_settings, save_settings
from palmsense.evaluation import run_evaluation
from palmsense.ui import hero, result_banner, section, stat

LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.02)",
              font={"color": "#E7EAF3"}, margin=dict(l=10, r=10, t=40, b=10))
LABELS = {"lines": "A · Palm lines", "lbp": "B · LBP texture", "fusion": "A+B · Fusion"}


def _pct(x):
    return "not measurable" if x is None else f"{x * 100:.1f}%"


def render():
    settings = load_settings()
    hero("Insights", "Dashboard & Evaluation",
         "Authentication history, performance testing (FAR, FRR, accuracy, latency) and threshold calibration.")
    t_log, t_eval, t_set = st.tabs(["📜 Authentication log", "🧪 Evaluation (FAR / FRR)", "⚙️ Decision settings"])

    # ------------------------------------------------------------------ log
    with t_log:
        log = storage.read_log()
        if not log:
            st.info("No authentication attempts yet. Try the **Verify** page.")
        else:
            df = pd.DataFrame(log)
            c = st.columns(4)
            with c[0]:
                stat("Attempts", str(len(df)))
            with c[1]:
                stat("Successful", str((df.outcome == "Authentication Successful").sum()))
            with c[2]:
                stat("Failed", str((df.outcome == "Authentication Failed").sum()))
            with c[3]:
                stat("Invalid image", str((df.outcome == "Invalid Image").sum()),
                     f"avg {df.duration_ms.dropna().mean():.0f} ms" if df.duration_ms.notna().any() else "")
            scored = df.dropna(subset=["score"]).sort_values("timestamp")
            if not scored.empty:
                fig = px.scatter(scored, x="timestamp", y="score", color="outcome", symbol="user_id",
                                 color_discrete_map={"Authentication Successful": "#10B981",
                                                     "Authentication Failed": "#F43F5E"})
                fig.add_hline(y=settings["matching"]["thresholds"][settings["matching"]["decision_method"]],
                              line_dash="dash", line_color="white", annotation_text="threshold")
                fig.update_traces(marker_size=11)
                fig.update_layout(height=320, title="Scores of recent attempts", **LAYOUT)
                st.plotly_chart(fig, width="stretch")
            st.dataframe(df[["attempt_id", "user_id", "timestamp", "method", "score", "threshold", "outcome",
                             "reason", "duration_ms"]], hide_index=True, width="stretch")
            st.caption("Privacy: the log stores numbers only - no palm images.")

    # ------------------------------------------------------------------ evaluation
    with t_eval:
        st.markdown("Put each participant's photos in their own folder: `dataset/P01/1.jpg … 5.jpg`. "
                    "Photos **1-3 = enrollment**, **4-5 = genuine tests**, other people's photos = **impostor tests**. "
                    "Half of the people **calibrate** the threshold, the other half **test** it (no leakage).")
        c1, c2 = st.columns([3, 1])
        default = "dataset" if (ROOT / "dataset").exists() else "dataset_synthetic"
        path = c1.text_input("Dataset folder", value=default)
        hand = c2.selectbox("Hand in dataset", ["right", "left"])
        if st.button("▶️ Run evaluation", type="primary"):
            folder = path if path.startswith("/") else str(ROOT / path)
            bar = st.progress(0.0, text="Processing images…")
            try:
                res = run_evaluation(folder, settings, hand, progress=lambda p: bar.progress(min(p, 1.0)))
                st.session_state["eval"] = res
            except (FileNotFoundError, ValueError) as e:
                result_banner("warn", "Cannot run evaluation", str(e))
            bar.empty()

        res = st.session_state.get("eval")
        if res:
            if not res["independent_split"]:
                st.warning("Fewer than 4 participants: calibration and test use the SAME people, "
                           "so results are optimistic (not independent).")
            st.caption(f"Calibration people: {', '.join(res['calibration_people'])} · "
                       f"Test people: {', '.join(res['test_people'])}")
            section("Results on the independent test group")
            cols = st.columns(3)
            for col, meth in zip(cols, ("lines", "lbp", "fusion")):
                r = res["results"][meth]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{LABELS[meth]}** · threshold `{r['threshold']:.4f}`")
                        st.markdown(
                            "<div style='display:flex;gap:18px;margin:6px 0'>" + "".join(
                                f"<div><div style='color:#8F98B0;font-size:.75rem;font-weight:700'>{k}</div>"
                                f"<div style='font-size:1.45rem;font-weight:800'>{_pct(v)}</div></div>"
                                for k, v in (("FAR", r["FAR"]), ("FRR", r["FRR"]), ("ACCURACY", r["accuracy"])))
                            + "</div>", unsafe_allow_html=True)
                        st.caption(f"{r['genuine_trials']} genuine · {r['impostor_trials']} impostor trials · "
                                   f"calibration EER {_pct(r['calibration_EER'])}")

            m1, m2, m3 = st.columns(3)
            with m1:
                stat("Mean latency", f"{res['mean_latency_ms']:.0f} ms" if res["mean_latency_ms"] else "-",
                     "per verification attempt")
            with m2:
                stat("Failure to acquire", str(len(res["failure_to_acquire"])), "invalid test images")
            with m3:
                f = res["results"]["fusion"]
                stat("Trials (fusion)", f"{f['genuine_trials'] + f['impostor_trials']}",
                     f"{f['genuine_trials']} genuine / {f['impostor_trials']} impostor")

            section("Ablation: which feature method works best?")
            rows = [{"Method": LABELS[k], "Threshold": v["threshold"], "FAR": _pct(v["FAR"]), "FRR": _pct(v["FRR"]),
                     "Accuracy": _pct(v["accuracy"]), "TP": v["TP"], "FN": v["FN"], "FP": v["FP"], "TN": v["TN"],
                     "Test EER": _pct(v["test_EER"])} for k, v in res["results"].items()]
            st.dataframe(rows, hide_index=True, width="stretch")

            meth = st.radio("Show charts for", list(LABELS), format_func=LABELS.get, horizontal=True, index=2)
            r = res["results"][meth]
            a, b = st.columns(2, gap="large")
            with a:
                cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                                x=["Accepted", "Rejected"], y=["Genuine", "Impostor"])
                fig.update_layout(height=330, title="Confusion matrix", coloraxis_showscale=False, **LAYOUT)
                st.plotly_chart(fig, width="stretch")
            with b:
                trials = pd.DataFrame(res["test_trials"])
                trials["type"] = np.where(trials.genuine, "Genuine", "Impostor")
                fig = px.histogram(trials, x=meth, color="type", nbins=30, barmode="overlay", opacity=0.75,
                                   color_discrete_map={"Genuine": "#10B981", "Impostor": "#F43F5E"})
                fig.add_vline(x=r["threshold"], line_dash="dash", line_color="white", annotation_text="threshold")
                fig.update_layout(height=330, title="Score distribution", xaxis_title="similarity score", **LAYOUT)
                st.plotly_chart(fig, width="stretch")
            roc = go.Figure()
            for k, color in (("lines", "#22D3EE"), ("lbp", "#F59E0B"), ("fusion", "#A78BFA")):
                pts = res["roc"][k]
                roc.add_trace(go.Scatter(x=[p["FAR"] for p in pts], y=[p["TAR"] for p in pts], mode="lines",
                                         name=LABELS[k], line=dict(color=color, width=3)))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dot", color="gray")))
            roc.update_layout(height=360, title="ROC curve (test group)", xaxis_title="False Accept Rate",
                              yaxis_title="True Accept Rate (1 − FRR)", **LAYOUT)
            st.plotly_chart(roc, width="stretch")

            if res["failure_to_acquire"] or res["notes"]:
                with st.expander("Invalid images & notes"):
                    for f in res["failure_to_acquire"]:
                        st.markdown(f"- `{f['probe']}`: {f['reason']}")
                    for n in res["notes"]:
                        st.markdown(f"- {n}")

            section("Apply calibration")
            new = res["calibrated_matching"]
            st.code(json.dumps({"thresholds": new["thresholds"], "norm": new["norm"]}, indent=2), language="json")
            if st.button("💾 Save these calibrated thresholds to settings.json"):
                settings["matching"]["thresholds"] = new["thresholds"]
                settings["matching"]["norm"] = new["norm"]
                settings["matching"]["calibrated_on"] = f"{path} (people: {', '.join(res['calibration_people'])})"
                save_settings(settings)
                st.success("Saved. Verification now uses the calibrated thresholds.")

    # ------------------------------------------------------------------ settings
    with t_set:
        m = settings["matching"]
        st.markdown(f"**Calibrated on:** {m['calibrated_on']}")
        st.markdown("Convention: **similarity score - higher = more similar - accept when score ≥ threshold.**")
        st.dataframe([{"Method": LABELS[k], "Threshold": v} for k, v in m["thresholds"].items()],
                     hide_index=True, width="stretch")
        choice = st.selectbox("Method used for the accept / reject decision", list(LABELS),
                              index=list(LABELS).index(m["decision_method"]), format_func=LABELS.get)
        if choice != m["decision_method"] and st.button("Save decision method"):
            settings["matching"]["decision_method"] = choice
            save_settings(settings)
            st.success(f"Decision method set to {LABELS[choice]}.")
            st.rerun()
