"""Load markdown policy documents and split them into section-level chunks.

Section-level chunks (one per `##` heading) keep each rule together with its heading,
which makes citations meaningful ("travel_expense_policy#meals-and-per-diem").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str
    external: bool  # supplied by a third party, not reviewed internally

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text()
        title = re.search(r"^# (.+)$", raw, re.MULTILINE).group(1).strip()
        external = "supplied by vendor" in title.lower() or "provided by an external vendor" in raw.lower()
        parts = re.split(r"^## (.+)$", raw, flags=re.MULTILINE)
        preamble = parts[0].split("\n", 1)[1].strip() if "\n" in parts[0] else ""
        if preamble:
            chunks.append(Chunk(f"{path.stem}#overview", path.stem, title, "Overview", preamble, external))
        for heading, body in zip(parts[1::2], parts[2::2]):
            chunks.append(Chunk(f"{path.stem}#{_slug(heading)}", path.stem, title, heading.strip(), body.strip(), external))
    return chunks
