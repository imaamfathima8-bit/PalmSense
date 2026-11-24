"""
llm.py - Palmistry report generation and the context-aware chatbot.

Model   : Google Gemini (model name set in .env, default 'gemini-flash-latest')
Special.: RAG - curated palmistry knowledge base (see rag.py) + strict system prompt
Input   : structured features from BOTH palms (JSON) + retrieved knowledge entries
Output  : structured markdown report with fixed sections
Fallback: if there is no API key / no internet, an OFFLINE template engine writes the
          report from the same knowledge base, so the demo always works.
"""
from __future__ import annotations

import json

from . import config
from .rag import retrieve, retrieve_for_reading

DISCLAIMER = ("Palmistry is a cultural and entertainment practice. It is not scientifically validated "
              "and must not be used for medical, financial, career or relationship decisions.")

SYSTEM_PROMPT = f"""You are PalmSense Reader, a friendly palmistry specialist inside a university
Digital Image Processing project.

STRICT RULES
1. Use ONLY the palm features given in OBSERVED FEATURES (they were measured by computer vision).
   Never invent lines, marks or shapes that are not listed.
2. If a line is "Uncertain", say the reading is tentative. If it is "Not clearly visible", say it
   could not be seen and do NOT interpret it (except the fate line, whose absence is common).
3. Base interpretations on the PALMISTRY KNOWLEDGE entries provided. Say "traditionally" or
   "in palmistry" - never present interpretations as facts or predictions.
4. Never predict health, illness, lifespan or death. Never give financial or medical advice.
5. Keep the language simple, warm and positive. Short paragraphs and bullet points.
6. Always keep this disclaimer in mind and repeat it when relevant: {DISCLAIMER}
"""

REPORT_SECTIONS = [
    "Major Lines (Traditional Interpretation)",
    "Personality Themes",
    "Career & Professional Life",
    "Finance & Prosperity",
    "Relationships & Emotions",
    "Future Narrative (non-scientific)",
    "Left vs Right Palm",
    "Uncertainty & Limitations",
]


def is_online() -> bool:
    return bool(config.GEMINI_API_KEY)


def _gemini(contents, system: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    last_err = None
    for model in dict.fromkeys([config.GEMINI_MODEL, "gemini-flash-latest", "gemini-2.5-flash"]):
        try:
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.6),
            )
            if resp.text:
                return resp.text, f"Gemini ({model}) + RAG"
        except Exception as e:  # try the next model name
            last_err = e
    raise RuntimeError(f"Gemini request failed: {last_err}")


def _kb_block(entries) -> str:
    return "\n".join(f'[{e["id"]}] ({e["line"]} / {e["section"]}) {e["text"]}' for e in entries)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def generate_report(left: dict, right: dict, differences: list[str]) -> dict:
    kb = retrieve_for_reading(left, right)
    prompt = (
        "OBSERVED FEATURES (computer vision, JSON):\n"
        f"LEFT PALM: {json.dumps(left)}\nRIGHT PALM: {json.dumps(right)}\n\n"
        "OBSERVED LEFT/RIGHT DIFFERENCES:\n- " + "\n- ".join(differences) + "\n\n"
        "PALMISTRY KNOWLEDGE (retrieved):\n" + _kb_block(kb) + "\n\n"
        "Write the palm reading in markdown using EXACTLY these level-2 headings, in this order:\n"
        + "\n".join(f"## {s}" for s in REPORT_SECTIONS)
        + "\nUnder each heading write 2-5 short bullet points. Cite knowledge ids like [KB012]. "
          "Mention which hand each point comes from. End with the disclaimer."
    )
    if is_online():
        try:
            text, source = _gemini([{"role": "user", "parts": [{"text": prompt}]}], SYSTEM_PROMPT)
            return {"markdown": text, "source": source, "knowledge": kb, "error": None}
        except Exception as e:
            return {"markdown": offline_report(left, right, differences), "source": "Offline template engine + RAG",
                    "knowledge": kb, "error": str(e)}
    return {"markdown": offline_report(left, right, differences), "source": "Offline template engine + RAG",
            "knowledge": kb, "error": None}


def _attr_text(info) -> str:
    a = info.get("attributes", {})
    parts = [a.get(k) for k in ("length", "shape", "clarity") if a.get(k)]
    parts += [a.get(k) for k in ("ending", "slope", "arc") if a.get(k)]
    return ", ".join(parts)


def _best(query, line=None, section=None):
    for e in retrieve(query, k=6, line=line):
        if (line is None or e["line"] == line) and (section is None or e["section"] == section):
            return e
    return None


def offline_report(left: dict, right: dict, differences: list[str]) -> str:
    names = {"heart": "Heart line", "head": "Head line", "life": "Life line", "fate": "Fate line"}
    md = []
    # Major lines
    md.append(f"## {REPORT_SECTIONS[0]}")
    for key in ("heart", "head", "life", "fate"):
        for hand, feats in (("Left", left), ("Right", right)):
            info = feats["lines"][key]
            if info["status"] == "Not clearly visible" and key != "fate":
                md.append(f"- **{names[key]} ({hand})**: not clearly visible in this photo - no reading given.")
                continue
            q = f'{key} {_attr_text(info)}' if info["status"] != "Not clearly visible" else f"{key} not clearly visible absent"
            e = _best(q, line=key)
            tag = " *(tentative - line is faint/uncertain)*" if info["status"] == "Uncertain" else ""
            obs = _attr_text(info) or "not visible"
            md.append(f"- **{names[key]} ({hand}, observed: {obs})**: {e['text'] if e else ''} [{e['id'] if e else '-'}]{tag}")
    # Thematic sections
    themes = [("Personality Themes", "personality"), ("Career & Professional Life", "career"),
              ("Finance & Prosperity", "finance"), ("Relationships & Emotions", "relationships"),
              ("Future Narrative (non-scientific)", "future")]
    for title, section in themes:
        md.append(f"\n## {title}")
        used = set()
        for hand, feats in (("Right", right), ("Left", left)):
            shape = feats["palm_shape"]
            queries = [(f'shape {shape["element_type"]}', "shape")]
            queries += [(f'{k} {_attr_text(v)}', k) for k, v in feats["lines"].items() if v["status"] != "Not clearly visible"]
            if section == "finance":
                queries = [(f'finance {shape["element_type"]} shape', "finance"), ("finance head long deep", "finance")]
            for q, line in queries:
                e = _best(q, line=line, section=section)
                if e and e["id"] not in used:
                    used.add(e["id"])
                    md.append(f"- ({hand} palm) {e['text']} [{e['id']}]")
                if len(used) >= 4:
                    break
        if section == "finance":
            md.append(f"- {_best('finance general', line='finance')['text']}")
        if section == "future":
            md.append("- This is a traditional story for fun only - palm lines cannot predict future events.")
        if not used:
            md.append("- No clear traditional theme could be linked to the detected features.")
    # Comparison
    md.append(f"\n## {REPORT_SECTIONS[6]}")
    md += [f"- {d}" for d in differences]
    md.append(f"- {_best('comparison left right difference', line='comparison')['text']}")
    # Uncertainty
    md.append(f"\n## {REPORT_SECTIONS[7]}")
    for hand, feats in (("Left", left), ("Right", right)):
        unsure = [names[k] for k, v in feats["lines"].items() if v["status"] != "Detected"]
        if unsure:
            md.append(f"- {hand} palm: {', '.join(unsure)} could not be detected with high confidence.")
    md.append("- Line detection depends on lighting, focus and hand position; faint lines may be missed.")
    md.append("- Palm shape (element type) is an approximate measurement from the hand outline.")
    md.append(f"\n> **Disclaimer:** {DISCLAIMER}")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Chatbot
# --------------------------------------------------------------------------- #
def chat(question: str, history: list[dict], context: dict) -> dict:
    """context = {'left': feats, 'right': feats, 'differences': [...], 'report': markdown}"""
    kb = retrieve(question, k=5)
    system = (SYSTEM_PROMPT + "\nYou are now chatting with the user about THEIR palm reading.\n"
              f"OBSERVED FEATURES LEFT: {json.dumps(context['left'])}\n"
              f"OBSERVED FEATURES RIGHT: {json.dumps(context['right'])}\n"
              f"OBSERVED DIFFERENCES: {context['differences']}\n"
              f"THEIR REPORT:\n{context['report']}\n"
              f"RELEVANT PALMISTRY KNOWLEDGE:\n{_kb_block(kb)}\n"
              "Answer in under 180 words. If asked about something not observed, say it was not detected.")
    if is_online():
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in history[-10:]]
        contents.append({"role": "user", "parts": [{"text": question}]})
        try:
            text, source = _gemini(contents, system)
            return {"answer": text, "source": source, "knowledge": kb}
        except Exception as e:
            ans = offline_chat(question, context)
            return {"answer": ans + f"\n\n_(Online model unavailable: {e})_", "source": "Offline", "knowledge": kb}
    return {"answer": offline_chat(question, context), "source": "Offline template engine + RAG", "knowledge": kb}


def offline_chat(question: str, context: dict) -> str:
    q = question.lower()
    names = {"heart": "heart line", "head": "head line", "life": "life line", "fate": "fate line"}
    topic_line = next((k for k in names if k in q), None)
    if any(w in q for w in ("health", "die", "death", "sick", "disease", "how long will i live", "lifespan")):
        return ("Palm lines cannot tell anything about health or lifespan - that is a common myth. "
                "For any health question please talk to a medical professional.\n\n" f"_{DISCLAIMER}_")
    if any(w in q for w in ("difference", "compare", "left and right", "both hands", "left vs")):
        return "**Observed left vs right differences:**\n- " + "\n- ".join(context["differences"]) + f"\n\n_{DISCLAIMER}_"
    if topic_line is None:
        for word, line in (("love", "heart"), ("relationship", "heart"), ("emotion", "heart"), ("career", "head"),
                           ("job", "fate"), ("work", "fate"), ("money", "head"), ("financ", "head"),
                           ("wealth", "head"), ("think", "head"), ("energy", "life"), ("personality", "head")):
            if word in q:
                topic_line = line
                break
    if topic_line is None:
        return ("I can explain your heart, head, life or fate line, your hand shape, your career, "
                "relationship or finance themes, or the differences between your left and right palm. "
                "What would you like to know?\n\n" f"_{DISCLAIMER}_")
    out = [f"**Your {names[topic_line]}** (observed by the image-processing pipeline):"]
    for hand in ("left", "right"):
        info = context[hand]["lines"][topic_line]
        if info["status"] == "Not clearly visible":
            out.append(f"- {hand.title()} palm: not clearly visible, so I cannot interpret it.")
        else:
            a = _attr_text(info)
            e = _best(f"{topic_line} {a} {q}", line=topic_line)
            note = " (tentative)" if info["status"] == "Uncertain" else ""
            out.append(f"- {hand.title()} palm - {a}{note}: {e['text'] if e else ''}")
    sec = next((s for w, s in (("career", "career"), ("job", "career"), ("money", "finance"),
                               ("financ", "finance"), ("love", "relationships")) if w in q), None)
    if sec:
        e = _best(f"{sec} {context['right']['palm_shape']['element_type']}", section=sec)
        if e:
            out.append(f"- Hand shape ({context['right']['palm_shape']['element_type']}): {e['text']}")
    out.append(f"\n_{DISCLAIMER}_")
    return "\n".join(out)
