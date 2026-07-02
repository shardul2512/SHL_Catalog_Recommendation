"""
Hybrid lexical retrieval over the catalog.

Design choice: with only 377 items and mostly technical/skill vocabulary
that overlaps heavily with real job-description language ("Java", "AWS",
"Excel", "leadership", "safety"), a lexical hybrid (BM25 + TF-IDF cosine)
gets us most of the value of embeddings without:
  - a model download at cold start (Render free tier + 2 min health-check
    budget makes this risky),
  - extra API cost/latency per request,
  - another external dependency to keep warm.

BM25 rewards exact/rare-term overlap (great for "OPQ32r", "Docker", "SVAR").
TF-IDF cosine smooths over phrasing differences. We fuse both via min-max
normalized weighted sum. This is swappable later for a real embedding
index (see NOTES in README) without touching the agent logic, since the
public interface is just `search(query, top_k) -> list[CatalogItem]`.
"""
import difflib
import re
from typing import Optional

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import CatalogItem, load_catalog

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

_SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness",
             "ing", "edly", "ed", "ies", "ly", "s")


def _stem(token: str) -> str:
    """Crude suffix stripper (not a real Porter stemmer, but enough to
    equate e.g. 'skills'/'skill', 're-skilling'/'skill', 'audits'/'audit'
    without pulling in an NLP dependency). Requires the stem to keep at
    least 3 chars so short words like 'sales' -> 'sale' don't get mangled
    into something unrecognizable."""
    if len(token) <= 4 or token.isdigit():
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


def _tokenize(text: str) -> list:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


class CatalogIndex:
    def __init__(self, items: Optional[list] = None):
        self.items: list = items if items is not None else load_catalog()
        self._corpus = [it.search_text for it in self.items]
        self._tokenized = [_tokenize(t) for t in self._corpus]

        self.bm25 = BM25Okapi(self._tokenized)

        self.tfidf = TfidfVectorizer(
            tokenizer=_tokenize, lowercase=False, min_df=1
        )
        self._tfidf_matrix = self.tfidf.fit_transform(self._corpus)

        self.names = [it.name for it in self.items]
        self._name_lower_to_item = {it.name.lower(): it for it in self.items}

    def search(self, query: str, top_k: int = 18) -> list:
        if not query or not query.strip():
            return []

        q_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)

        q_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        def _norm(scores):
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-9:
                return [0.0] * len(scores)
            return [(s - lo) / (hi - lo) for s in scores]

        bm25_n = _norm(bm25_scores)
        tfidf_n = _norm(tfidf_scores)

        fused = [0.6 * b + 0.4 * t for b, t in zip(bm25_n, tfidf_n)]

        ranked = sorted(
            range(len(self.items)), key=lambda i: fused[i], reverse=True
        )
        return [self.items[i] for i in ranked[:top_k] if fused[i] > 0]

    def fuzzy_find(self, name: str, cutoff: float = 0.6) -> Optional[CatalogItem]:
        """Best-effort name grounding, used to resolve LLM-mentioned item
        names (e.g. for compare/refine) back to a real catalog entry.

        Uses token overlap, word inclusion checks, and edit distance to robustly
        match variations like 'Verify G+' to 'Verify - G+', 'OPQ32r' to its full name,
        and SVAR country codes accurately.
        """
        name_l = name.strip().lower()
        if name_l in self._name_lower_to_item:
            return self._name_lower_to_item[name_l]

        def tokenize(s):
            # Clean common noise like (new) and convert to lowercase alphanumeric tokens
            s = re.sub(r"\s*\(new\)\s*$", "", s.lower()).strip()
            return re.findall(r"[a-z0-9+#]+", s)

        q_tokens = tokenize(name_l)
        if not q_tokens:
            return None

        best_score = -1.0
        best_item = None

        for cat_name, item in self._name_lower_to_item.items():
            cat_tokens = tokenize(cat_name)
            if not cat_tokens:
                continue

            intersection = set(q_tokens) & set(cat_tokens)
            if not intersection:
                continue

            # Check if any crucial query word is missing from the catalog name.
            # Country codes ('us', 'uk', 'aus') and specific tech names must not be mis-matched.
            missing_query_words = set(q_tokens) - set(cat_tokens)
            critical_miss = False
            for w in missing_query_words:
                if len(w) <= 3 or w in {"java", "excel", "word", "svar", "opq", "g+"}:
                    critical_miss = True
                    break
            if critical_miss:
                continue

            # Token overlap ratio (fraction of query tokens matched)
            overlap_score = len(intersection) / len(q_tokens)

            # Character level sequence match ratio
            char_ratio = difflib.SequenceMatcher(None, name_l, cat_name).ratio()

            # Combine token overlap (primary) and char ratio (secondary)
            score = (overlap_score * 10.0) + char_ratio

            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    def find_all_mentions(self, text: str, min_ratio: float = 0.72) -> list:
        """Scan free text for any catalog item names mentioned (used to pull
        in comparison targets that might not surface in top-K search)."""
        text_l = text.lower()
        found = []
        for it in self.items:
            n = it.name.lower()
            # Clean parentheticals (e.g. "(general)", "(new)") and extra whitespace
            core = re.sub(r"\s*\(.*?\)\s*", " ", n).strip()
            core = re.sub(r"\s+", " ", core)
            if core and core in text_l:
                found.append(it)
        return found


_index_singleton: Optional[CatalogIndex] = None


def get_index() -> CatalogIndex:
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = CatalogIndex()
    return _index_singleton
