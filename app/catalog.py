"""
Loads and normalizes the SHL product catalog.

The raw scrape has a `keys` field with full category names
(e.g. "Personality & Behavior"). SHL's public catalog uses single-letter
codes for these (visible as column icons on shl.com), which is also the
convention used in the reference conversation traces (P, K, A, B, S, C, D, E).
We map to those codes here so the API response matches what evaluators
and traces expect.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import CATALOG_PATH


CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    description: str
    keys: list  # full category names
    test_type: str  # comma-joined codes, e.g. "P,C"
    job_levels: list
    languages: list
    duration: str
    search_text: str = field(default="", repr=False)

    def to_recommendation(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }


def _codes_for(keys: list) -> str:
    codes = [CATEGORY_TO_CODE.get(k, "") for k in keys]
    codes = [c for c in codes if c]
    return ",".join(codes) if codes else "K"


def load_catalog(path: str = CATALOG_PATH) -> list:
    """Load catalog.json (tolerant of stray control chars from the scrape)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw, strict=False)

    items = []
    for d in data:
        name = (d.get("name") or "").strip()
        url = (d.get("link") or "").strip()
        if not name or not url:
            continue  # never surface an item we can't cite a real URL for
        description = (d.get("description") or "").strip()
        keys = d.get("keys") or []
        job_levels = d.get("job_levels") or []
        languages = d.get("languages") or []
        duration = (d.get("duration") or "").strip()

        # Build a rich search text from all available fields so lexical
        # retrieval can match on any dimension the user might mention.
        raw_langs = (d.get("languages_raw") or "").strip()
        remote_tag = "remote online" if (d.get("remote") or "").lower() == "yes" else ""
        adaptive_tag = "adaptive" if (d.get("adaptive") or "").lower() == "yes" else ""

        search_text = " ".join(
            [
                name,
                description,
                " ".join(keys),
                " ".join(job_levels),
                raw_langs,
                remote_tag,
                adaptive_tag,
                # strip "(New)" / punctuation noise but keep it too, cheap
                re.sub(r"[()]", " ", name),
            ]
        ).lower()

        items.append(
            CatalogItem(
                entity_id=str(d.get("entity_id", "")),
                name=name,
                url=url,
                description=description,
                keys=keys,
                test_type=_codes_for(keys),
                job_levels=job_levels,
                languages=languages,
                duration=duration,
                search_text=search_text,
            )
        )
    return items


def find_by_name(items: list, name: str) -> Optional[CatalogItem]:
    """Exact (case-insensitive) name match, used to re-ground LLM output."""
    name_l = name.strip().lower()
    for it in items:
        if it.name.lower() == name_l:
            return it
    return None
