"""
rag.py - Retrieval-Augmented Generation (RAG): the "palmistry specialisation".

We do NOT train an LLM. Instead we keep a small, curated palmistry knowledge
base (knowledge_base/palmistry_kb.json). For every reading or chat question we:
   1. turn the detected palm features into search queries,
   2. find the most relevant knowledge entries with TF-IDF + cosine similarity,
   3. give ONLY those entries + the detected features to the LLM.
So the LLM answers like a palmistry specialist and cannot invent unseen features.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import KB_PATH


@lru_cache(maxsize=1)
def _index():
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    entries = kb["entries"]
    docs = [f'{e["line"]} {e["line"]} {e["condition"]} {e["section"]} {e["text"]}' for e in entries]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(docs)
    return kb, entries, vec, matrix


def knowledge_base() -> dict:
    return _index()[0]


def retrieve(query: str, k: int = 5, line: str | None = None) -> list[dict]:
    _, entries, vec, matrix = _index()
    sims = cosine_similarity(vec.transform([query]), matrix).ravel()
    if line:
        sims = sims + 0.25 * np.array([e["line"] == line for e in entries])  # boost the requested line
    order = sims.argsort()[::-1][:k]
    return [{**entries[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0]


def feature_queries(feats: dict) -> list[tuple[str, str]]:
    """Build one query per detected feature, e.g. 'heart long gently curved deep'."""
    qs = []
    for key, info in feats["lines"].items():
        if info["status"] == "Not clearly visible":
            qs.append((key, f"{key} not clearly visible absent"))
        else:
            a = info.get("attributes", {})
            words = [a.get(x, "") for x in ("length", "shape", "clarity", "ending", "slope", "arc")]
            if a.get("joined_with_life_line") is not None:
                words.append("joined with life line" if a["joined_with_life_line"] else "separate from life line")
            qs.append((key, f"{key} {' '.join(words)} present"))
    s = feats["palm_shape"]
    qs.append(("shape", f'shape {s["element_type"]} {s["palm"]} {s["fingers"]}'))
    qs.append(("finance", f'finance {s["element_type"]} head fate general'))
    qs.append(("general", "minor creases many" if feats.get("minor_creases", 0) > 10 else "minor creases few"))
    return qs


def retrieve_for_reading(left: dict, right: dict) -> list[dict]:
    seen, out = set(), []
    for feats in (left, right):
        for line, q in feature_queries(feats):
            for e in retrieve(q, k=3, line=line if line != "general" else None):
                if e["id"] not in seen:
                    seen.add(e["id"])
                    out.append(e)
    for q in ("comparison left right difference", "general disclaimer", "general health"):
        for e in retrieve(q, k=1):
            if e["id"] not in seen:
                seen.add(e["id"])
                out.append(e)
    return out
