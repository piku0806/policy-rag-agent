"""BM25 retriever (no external services needed).

Keyword retrieval is a strong, explainable baseline for policy text full of exact terms
("per diem", "legal hold"). Swap in a vector index (Databricks Vector Search, Azure AI Search)
by implementing the same `search` method.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import mlflow
from mlflow.entities import SpanType

from policyrag.ingest import Chunk

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its may must my
of on or our per so than that the their there this to us was we what when where which who will with you your
should would could get any all much many long days day""".split())


def stem(t: str) -> str:
    """Tiny suffix stripper: submitted/submitting/submits -> submit, receipts -> receipt."""
    for suf in ("ing", "ed", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf) and not t.endswith("ss"):
            t = t[: -len(suf)]
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiouls":
                t = t[:-1]
            break
    return t


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9$%]+", text.lower())
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        out.append(stem(t))
    return out


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def _score(self, q: list[str], i: int) -> float:
        dl, tf, s = len(self.docs[i]), self.tf[i], 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def search(self, query: str, k: int = 4) -> list[dict]:
        q = tokenize(query)
        scored = sorted(((self._score(q, i), i) for i in range(len(self.chunks))), reverse=True)[:k]
        return [{**self.chunks[i].to_dict(), "score": round(s, 3)} for s, i in scored if s > 0]
