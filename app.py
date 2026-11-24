"""
PalmSense - Palm Biometric Authentication + AI Palm Reading
NIBM KIC | Digital Image Processing | Combined ICA 01 + ICA 02

Run:   streamlit run app.py
"""
import streamlit as st

from palmsense import llm
from palmsense.ui import inject_css
from views import chat, dashboard, enroll, home, privacy, reading, verify

st.set_page_config(page_title="PalmSense", page_icon="🖐️", layout="wide")
inject_css()

pages = {
    "PalmSense": [st.Page(home.render, title="Home", icon="🏠", default=True)],
    "Biometric Authentication": [
        st.Page(enroll.render, title="Enroll", icon="📝", url_path="enroll"),
        st.Page(verify.render, title="Verify", icon="🔐", url_path="verify"),
    ],
    "AI Palm Reading": [
        st.Page(reading.render, title="Palm Reading", icon="🔮", url_path="reading"),
        st.Page(chat.render, title="Ask PalmSense", icon="💬", url_path="chat"),
    ],
    "Insights": [
        st.Page(dashboard.render, title="Dashboard", icon="📊", url_path="dashboard"),
        st.Page(privacy.render, title="Privacy & Admin", icon="🛡️", url_path="privacy"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🖐️ PalmSense")
    st.caption("Your palm is your password - and your story.")
    if st.session_state.get("verified_user"):
        st.success(f"Verified as **{st.session_state['verified_user']}**")
    st.caption(("🟢 AI model: Gemini online" if llm.is_online() else "🟡 AI model: offline mode (no API key)"))
    st.caption("Educational prototype · NIBM KIC · DIP")

nav.run()
