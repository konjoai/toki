"""Prompt similarity dedup via TF-IDF cosine.

Pure-stdlib TF-IDF index for detecting near-duplicate adversarial prompts.
The index recomputes IDF lazily on each cosine query so newly-added docs
contribute to document frequency.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({"the", "and", "of", "to", "for", "a", "is", "in", "on", "with"})
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, drop tokens <2 chars and stopwords."""
    lowered = text.lower()
    raw = _TOKEN_RE.findall(lowered)
    return [t for t in raw if len(t) >= 2 and t not in _STOPWORDS]


def tf(tokens: list[str]) -> dict[str, float]:
    """Raw term frequency dict (term -> count)."""
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _idf(docs: list[dict[str, float]]) -> dict[str, float]:
    """Smoothed IDF: log((N+1)/(df+1)) + 1 for each term across docs."""
    n_docs = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for term in doc:
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n_docs + 1) / (count + 1)) + 1 for term, count in df.items()}


def _tfidf_vector(tf_map: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    """Multiply tf by idf weights; unseen terms get idf=log(N+1)+1 effective 0 contribution."""
    return {term: count * idf.get(term, 0.0) for term, count in tf_map.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors. Returns 0.0 on either-empty."""
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, weight in a.items():
        other = b.get(term)
        if other is not None:
            dot += weight * other
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mint_id(text: str) -> str:
    """12-char sha1 prefix of text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class TfidfIndex:
    """Tiny TF-IDF index — IDF recomputed lazily over inserted docs."""

    _docs: list[dict[str, float]] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)

    def add(self, text: str, *, attack_id: Optional[str] = None) -> str:
        """Add a doc; returns assigned id (sha1[:12] if not provided)."""
        assigned = attack_id if attack_id is not None else _mint_id(text)
        self._docs.append(tf(tokenize(text)))
        self._ids.append(assigned)
        return assigned

    def cosine(self, attack_id: str, text: str) -> float:
        """Cosine similarity between stored doc with attack_id and text."""
        if attack_id not in self._ids:
            return 0.0
        idx = self._ids.index(attack_id)
        query_tf = tf(tokenize(text))
        idf = _idf(self._docs + [query_tf])
        stored_vec = _tfidf_vector(self._docs[idx], idf)
        query_vec = _tfidf_vector(query_tf, idf)
        return _cosine(stored_vec, query_vec)

    def nearest(
        self, text: str, *, threshold: float = 0.95
    ) -> Optional[tuple[str, float]]:
        """Closest inserted doc above threshold; None if no match or empty index."""
        if not self._docs:
            return None
        query_tf = tf(tokenize(text))
        if not query_tf:
            return None
        idf = _idf(self._docs + [query_tf])
        query_vec = _tfidf_vector(query_tf, idf)
        best_id: Optional[str] = None
        best_sim = 0.0
        for stored_id, stored_tf in zip(self._ids, self._docs):
            stored_vec = _tfidf_vector(stored_tf, idf)
            sim = _cosine(stored_vec, query_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = stored_id
        if best_id is not None and best_sim >= threshold:
            return (best_id, best_sim)
        return None

    def __len__(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()
        self._ids.clear()


@dataclass(frozen=True)
class DedupVerdict:
    """Result of a dedup check."""

    is_duplicate: bool
    similar_attack_id: Optional[str]
    similarity: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "is_duplicate": self.is_duplicate,
            "similar_attack_id": self.similar_attack_id,
            "similarity": self.similarity,
            "threshold": self.threshold,
        }


class DedupChecker:
    """Stateful near-duplicate checker backed by a TfidfIndex."""

    def __init__(self, *, threshold: float = 0.95, tracker=None) -> None:
        self._index = TfidfIndex()
        self._threshold = threshold
        self._tracker = tracker

    def check(self, prompt: str) -> DedupVerdict:
        """Return a verdict without mutating the index."""
        match = self._index.nearest(prompt, threshold=self._threshold)
        if match is None:
            return DedupVerdict(
                is_duplicate=False,
                similar_attack_id=None,
                similarity=0.0,
                threshold=self._threshold,
            )
        attack_id, sim = match
        return DedupVerdict(
            is_duplicate=True,
            similar_attack_id=attack_id,
            similarity=sim,
            threshold=self._threshold,
        )

    def check_and_record(
        self, prompt: str, attack_type: str, result: str, **kwargs
    ) -> DedupVerdict:
        """Check duplicate; if novel, add to index and record to tracker."""
        verdict = self.check(prompt)
        if verdict.is_duplicate:
            return verdict
        self._index.add(prompt)
        if self._tracker is not None:
            try:
                self._tracker.record(attack_type, prompt, result, **kwargs)
            except Exception as exc:
                logger.warning("DedupChecker tracker.record failed: %s", exc)
                raise
        return verdict

    def __len__(self) -> int:
        return len(self._index)
