import hmac

import pandas as pd
import streamlit as st

from palmsense import config, storage
from palmsense.ui import hero, result_banner, section


def _pin_ok(pin: str) -> bool:
    return bool(config.ADMIN_PIN) and hmac.compare_digest(pin, config.ADMIN_PIN)


def render():
    hero("Responsible use", "Privacy & Admin",
         "How PalmSense protects biometric data, and where participants can withdraw consent.")

    a, b = st.columns(2, gap="large")
    with a:
        section("🛡️ Our data policy")
        st.markdown("""
- **Consent first** - nobody is enrolled without ticking the consent box.
- **Pseudonymous IDs** - we store an ID like `S01`, never names or contact details.
- **No raw images** - photos are processed in memory and discarded; only numbers are kept.
- **Encrypted templates** - feature vectors are encrypted with Fernet (AES) using a key from `.env`.
- **Minimal audit log** - attempt ID, user ID, time, score, outcome, duration. No images.
- **Right to delete** - an enrollment can be deleted at any time (below).
- **Retention** - all templates are deleted at the end of the assessment.
- **Secrets** - API keys, the admin PIN and the encryption key live only in `.env` (git-ignored).
""")
    with b:
        section("⚠️ Known limitations")
        st.markdown("""
- **Spoofing:** a printed or on-screen palm photo may be accepted - there is **no liveness detection**.
- **Small dataset:** ~10 participants only supports proof-of-concept conclusions.
- **Lighting & pose:** strong shadows, closed fingers or unusual backgrounds can cause rejections.
- **Not for real security:** do not use this prototype for high-stakes access control.
- **Palmistry** is entertainment only - readings are not scientific predictions.
""")

    section("👥 Enrolled users")
    users = storage.list_users()
    if users:
        st.dataframe(pd.DataFrame(users), hide_index=True, width="stretch")
    else:
        st.caption("No users enrolled yet.")

    section("🗑️ Withdraw consent / delete enrollment")
    if not config.ADMIN_PIN:
        st.warning("Admin actions are disabled: set `ADMIN_PIN` in your `.env` file.", icon="🔒")
        return
    c1, c2 = st.columns(2)
    uid = c1.selectbox("User ID to delete", [u["user_id"] for u in users] or ["-"])
    pin = c2.text_input("Admin PIN", type="password")
    d1, d2 = st.columns(2)
    if d1.button("Delete user & template", type="primary", width="stretch", disabled=not users):
        if not _pin_ok(pin):
            result_banner("fail", "Wrong admin PIN", "Deletion refused.")
        elif storage.delete_user(uid):
            result_banner("ok", "Deleted", f"User {uid} and their encrypted template were permanently removed.")
        else:
            result_banner("warn", "Not found", f"User {uid} does not exist.")
    if d2.button("Clear authentication log", width="stretch"):
        if _pin_ok(pin):
            storage.clear_log()
            result_banner("ok", "Log cleared", "All audit log entries were removed.")
        else:
            result_banner("fail", "Wrong admin PIN", "Action refused.")
