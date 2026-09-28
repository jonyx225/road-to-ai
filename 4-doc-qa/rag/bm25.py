"""Keyword search with BM25, written from scratch so you can see how it works.

Vector search is good at meaning but can miss exact strings such as error codes,
product names or numbers. BM25 is the opposite: it scores chunks by how many of the
query's words they contain, weighting rare words more heavily. Using both
("hybrid" search) covers each one's blind spots.

score(query, chunk) = sum over query words w of
    idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk_length / avg_chunk_length))
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from had has have how i if in is it its of on or "
    "our that the their them then there these they this to was we were what when where which who "
    "will with you your".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase words. Hyphenated identifiers like "e-4102" are kept whole AND split
    into their parts ("e-4102", "4102") so both spellings of a query can match."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        if "-" in token:
            tokens += [part for part in token.split("-") if len(part) > 1 or part.isdigit()]
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(documents)
        counts = [Counter(tokenize(doc)) for doc in documents]
        self.doc_len = np.array([sum(c.values()) for c in counts], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if self.n else 0.0

        document_frequency: Counter = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(counts):
            document_frequency.update(c.keys())
            for term, tf in c.items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            for i, tf in self.postings[term]:
                length_norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return scores

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (index, score) pairs with score > 0, best first. Ties keep document order."""
        scores = self.scores(query)
        order = np.lexsort((np.arange(self.n), -scores))[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
