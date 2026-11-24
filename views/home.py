import streamlit as st

from palmsense import storage
from palmsense.ui import card, disclaimer, flow, hero, section, stat


def render():
    hero("NIBM KIC · Digital Image Processing",
         "Your palm is your password<br>and your story.",
         "PalmSense uses one digital image-processing pipeline for two jobs: verifying who you are "
         "from your palm print (biometric authentication), and reading your palm lines for a fun, "
         "AI-powered palmistry report with a chatbot.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("🔐", "Palm Authentication", "Enroll 3 palm photos, then log in with a new photo. "
             "Two feature methods (palm lines + LBP texture) give a similarity score and an accept / reject decision.")
    with c2:
        card("🔮", "AI Palm Reading", "Upload both palms. We detect the heart, head, life and fate lines, "
             "then a palmistry-specialised AI (RAG) writes a structured reading.")
    with c3:
        card("💬", "Ask PalmSense", "Chat about your reading. The chatbot only talks about features that "
             "were actually detected in your photos.")

    section("One shared pipeline", "Enrollment, verification and palm reading all use exactly the same steps.")
    flow(["Upload / Camera", "Quality check", "Hand segmentation", "Alignment", "Palm ROI",
          "Grayscale + Denoise", "CLAHE", "Line & texture features"])
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("**🔐 Authentication branch**")
        flow(["Template", "Similarity score", "Threshold", "Accept / Reject / Invalid"])
    with b:
        st.markdown("**🔮 Palm reading branch**")
        flow(["Line detection", "Structured features", "RAG + LLM", "Report & Chat"])

    section("System at a glance")
    users = storage.list_users()
    log = storage.read_log()
    ok = sum(r["outcome"] == "Authentication Successful" for r in log)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        stat("Enrolled users", str(len(users)), "pseudonymous IDs only")
    with s2:
        stat("Auth attempts", str(len(log)), "stored in audit log")
    with s3:
        stat("Successful", str(ok), f"{(ok / len(log) * 100):.0f}% of attempts" if log else "no attempts yet")
    with s4:
        stat("Raw images stored", "0", "templates only, encrypted")

    section("How to take a good palm photo")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        card("✋", "Open palm", "Palm facing the camera, fingers pointing up and slightly apart.")
    with t2:
        card("🖤", "Plain background", "A dark, plain surface makes the hand easy to separate.")
    with t3:
        card("💡", "Even lighting", "Daylight or a bright room. No flash, no strong shadows.")
    with t4:
        card("🎯", "Whole hand in frame", "Keep the full hand visible and the camera steady.")

    st.write("")
    disclaimer("Educational prototype only - not production-grade security. A printed photo of a palm may fool it "
               "(no liveness detection). Palmistry is for entertainment and is not scientifically validated.")
