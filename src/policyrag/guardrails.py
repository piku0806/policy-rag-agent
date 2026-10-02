"""Guardrails for retrieved content."""
from __future__ import annotations

import re

INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|note to (ai|llm|assistant)s?|"
    r"disregard (your|all|previous) (rules|instructions)|you are now|system prompt)", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def sanitize(text: str) -> tuple[str, list[str]]:
    """Drop any sentence that carries instructions aimed at an AI system."""
    kept, removed = [], []
    for s in split_sentences(text):
        (removed if INJECTION_RE.search(s) else kept).append(s)
    return " ".join(kept), removed
