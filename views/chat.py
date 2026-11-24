import streamlit as st

from palmsense import llm
from palmsense.ui import disclaimer, hero, pill

SUGGESTIONS = [
    "What does my heart line traditionally represent?",
    "What differences are visible between my left and right palms?",
    "What does palmistry say about my career?",
    "Can you explain my life line in more detail?",
    "What are the traditional interpretations of my financial prospects?",
]


def render():
    hero("Context-aware chatbot", "Ask PalmSense",
         "Ask follow-up questions about your reading. The assistant only uses the features detected in "
         "your photos, your report and the curated palmistry knowledge base.")
    rd = st.session_state.get("reading")
    if not rd:
        st.info("First analyse both palms on the **Palm Reading** page - the chatbot needs that context.", icon="🔮")
        return
    if not rd.get("report"):
        with st.spinner("Preparing your reading as chat context…"):
            rd["report"] = llm.generate_report(rd["left"]["features"], rd["right"]["features"], rd["differences"])

    context = {"left": rd["left"]["features"], "right": rd["right"]["features"],
               "differences": rd["differences"], "report": rd["report"]["markdown"]}
    history = st.session_state.setdefault("chat_history", [])

    disclaimer(llm.DISCLAIMER)
    st.write("")
    cols = st.columns(len(SUGGESTIONS))
    clicked = None
    for c, q in zip(cols, SUGGESTIONS):
        if c.button(q, width="stretch"):
            clicked = q

    for msg in history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔮"):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.markdown(pill(msg["source"], "ai"), unsafe_allow_html=True)

    question = st.chat_input("Ask about your palm lines, career, relationships…") or clicked
    if question:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🔮"):
            with st.spinner("Thinking…"):
                res = llm.chat(question, history, context)
            st.markdown(res["answer"])
            st.markdown(pill(res["source"], "ai"), unsafe_allow_html=True)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": res["answer"], "source": res["source"]})

    if history and st.button("🧹 Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
