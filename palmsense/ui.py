"""ui.py - Shared look & feel (CSS) and small reusable UI components."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(139,92,246,.20), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,.13), transparent 55%),
    #0A0E1A;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F1424 0%, #0B0F1C 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
.block-container { padding-top: 2.2rem; max-width: 1250px; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* ---------- hero ---------- */
.ps-hero { padding: 34px 36px; border-radius: 26px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(34,211,238,.10) 60%, rgba(10,14,26,.2));
  border: 1px solid rgba(255,255,255,.09); margin-bottom: 22px; }
.ps-hero:after { content:""; position:absolute; right:-80px; top:-80px; width:280px; height:280px;
  background: radial-gradient(circle, rgba(236,72,153,.28), transparent 65%); }
.ps-badge { display:inline-block; padding:5px 12px; border-radius:999px; font-size:.75rem; font-weight:700;
  letter-spacing:.06em; text-transform:uppercase; color:#C4B5FD; background:rgba(139,92,246,.16);
  border:1px solid rgba(139,92,246,.35); }
.ps-hero h1 { font-size: 2.5rem; font-weight: 800; margin: 14px 0 6px 0; line-height:1.12;
  background: linear-gradient(90deg,#FFFFFF, #C4B5FD 55%, #67E8F9); -webkit-background-clip:text; color:transparent; }
.ps-hero p { color:#AAB2C8; font-size:1.05rem; max-width: 760px; margin:0; }

/* ---------- cards ---------- */
.ps-card { background: rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08);
  border-radius: 20px; padding: 20px 22px; height: 100%; }
.ps-card h4 { margin: 8px 0 6px 0; font-size: 1.08rem; }
.ps-card p { color:#9AA3BA; font-size:.93rem; margin:0; }
.ps-icon { width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:1.35rem; background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(34,211,238,.25)); }

/* ---------- section title ---------- */
.ps-section { margin: 26px 0 12px 0; }
.ps-section h3 { margin:0; font-size:1.35rem; font-weight:700; }
.ps-section p { margin:2px 0 0 0; color:#8F98B0; font-size:.93rem; }

/* ---------- pipeline chips ---------- */
.ps-flow { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.ps-chip { padding:8px 14px; border-radius:12px; background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.09); font-size:.85rem; font-weight:600; color:#DCE1EE; }
.ps-chip b { color:#A78BFA; margin-right:6px; }
.ps-arrow { color:#5B6480; font-weight:700; }

/* ---------- result banner ---------- */
.ps-result { border-radius:22px; padding:22px 26px; display:flex; gap:18px; align-items:center; margin: 8px 0 14px 0; }
.ps-result .big { font-size:2.3rem; }
.ps-result h2 { margin:0; font-size:1.55rem; font-weight:800; }
.ps-result p { margin:3px 0 0 0; color:#C9CFDF; }
.ps-ok   { background:linear-gradient(135deg, rgba(16,185,129,.25), rgba(16,185,129,.07)); border:1px solid rgba(16,185,129,.45); }
.ps-fail { background:linear-gradient(135deg, rgba(244,63,94,.25), rgba(244,63,94,.07));  border:1px solid rgba(244,63,94,.45); }
.ps-warn { background:linear-gradient(135deg, rgba(245,158,11,.25), rgba(245,158,11,.07)); border:1px solid rgba(245,158,11,.45); }

/* ---------- stat tiles ---------- */
.ps-stat { background: rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:16px 18px; }
.ps-stat .lbl { color:#8F98B0; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.ps-stat .val { font-size:1.75rem; font-weight:800; margin-top:4px; }
.ps-stat .sub { color:#7D86A0; font-size:.8rem; }

/* ---------- pills ---------- */
.ps-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:700; }
.ps-pill.ok { background:rgba(16,185,129,.18); color:#6EE7B7; }
.ps-pill.mid { background:rgba(245,158,11,.18); color:#FCD34D; }
.ps-pill.no { background:rgba(148,163,184,.16); color:#CBD5E1; }
.ps-pill.cv { background:rgba(34,211,238,.15); color:#67E8F9; }
.ps-pill.ai { background:rgba(236,72,153,.15); color:#F9A8D4; }

.ps-disclaimer { border-left: 3px solid #F59E0B; background: rgba(245,158,11,.08); padding: 10px 14px;
  border-radius: 10px; color:#E8D9B5; font-size:.88rem; }
.ps-mono { font-family: 'JetBrains Mono', monospace; }

/* ---------- streamlit widgets ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 700; padding: .55rem 1.1rem; border: 1px solid rgba(255,255,255,.12); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(90deg, #8B5CF6, #6366F1 60%, #06B6D4); border: none; color: white; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); }
[data-testid="stFileUploaderDropzone"] { border-radius: 16px; background: rgba(255,255,255,.03);
  border: 1.5px dashed rgba(139,92,246,.45); }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stImageCaption"], [data-testid="caption"] { color:#9AA3BA !important; }
[data-testid="stExpander"] { border-radius: 16px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.02); }
[data-testid="stMetric"] { background: rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 12px 16px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 14px; background: rgba(255,255,255,.03); }
.stTabs [aria-selected="true"] { background: rgba(139,92,246,.18) !important; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:18px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(badge: str, title: str, subtitle: str):
    st.markdown(f'<div class="ps-hero"><span class="ps-badge">{badge}</span>'
                f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="ps-section"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, text: str):
    st.markdown(f'<div class="ps-card"><div class="ps-icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True)


def flow(steps: list[str]):
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="ps-chip"><b>{i + 1}</b>{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="ps-arrow">→</span>')
    st.markdown(f'<div class="ps-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def result_banner(kind: str, title: str, detail: str):
    icon = {"ok": "✅", "fail": "⛔", "warn": "⚠️"}[kind]
    st.markdown(f'<div class="ps-result ps-{kind}"><div class="big">{icon}</div><div>'
                f'<h2>{html.escape(title)}</h2><p>{html.escape(detail)}</p></div></div>', unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = ""):
    st.markdown(f'<div class="ps-stat"><div class="lbl">{label}</div><div class="val">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "no") -> str:
    return f'<span class="ps-pill {kind}">{html.escape(text)}</span>'


def disclaimer(text: str):
    st.markdown(f'<div class="ps-disclaimer">⚠️ {text}</div>', unsafe_allow_html=True)


def stage_gallery(stages: dict, cols: int = 4):
    items = list(stages.items())
    for i in range(0, len(items), cols):
        row = st.columns(cols)
        for col, (name, img) in zip(row, items[i:i + cols]):
            with col:
                st.image(img, caption=name, width="stretch")


PHOTO_TIPS = ("📸 **Photo tips:** open palm facing the camera · fingers pointing **up** and slightly spread · "
              "plain, dark background · even light, no flash · whole hand inside the frame · hold steady.")


def image_input(label: str, key: str, help_text: str = ""):
    """Upload OR webcam capture. Returns raw bytes or None."""
    mode = st.radio(label, ["📁 Upload", "📷 Camera"], horizontal=True, key=f"{key}_mode", help=help_text)
    if mode == "📁 Upload":
        f = st.file_uploader("JPG / JPEG / PNG", type=["jpg", "jpeg", "png"], key=f"{key}_file",
                             label_visibility="collapsed")
    else:
        f = st.camera_input("Capture palm", key=f"{key}_cam", label_visibility="collapsed")
    return f.getvalue() if f is not None else None
