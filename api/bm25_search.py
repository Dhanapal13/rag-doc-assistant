"""
bm25_search.py
---------------
Lexical (keyword) retrieval using BM25, meant to be combined with the
existing dense vector retrieval (SentenceTransformer + Chroma) into a
hybrid search pipeline.

Why combine BM25 with vector search:
Dense embeddings are great at matching semantically similar text even when
the wording differs, but they can miss exact keyword matches - product
codes, acronyms, rare technical terms, numbers, named entities - that a
sparse lexical method like BM25 catches easily. Hybrid search (BM25 +
dense, fused) generally outperforms either method alone, especially on
domain-specific corpora such as regulatory / technical PDFs.

This module provides:
    - BM25Index: an in-memory BM25 index over (chunk_id, text) pairs,
      built from the *same* chunks that get embedded and stored in Chroma,
      with optional disk persistence so it survives process restarts.
    - reciprocal_rank_fusion: merges a BM25-ranked id list and a
      vector-search-ranked id list into one fused ranking. RRF only needs
      each id's *rank* in each list, not its raw score, which sidesteps the
      problem that BM25 scores and cosine-similarity scores live on
      different, incomparable scales.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import re
import pickle
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over (chunk_id, text) pairs, with optional disk persistence."""

    def __init__(self):
        self.logger = structlog.get_logger()
        self.chunk_ids: List[str] = []
        self.corpus: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        """(Re)build the index from scratch over the given chunks."""
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids = list(chunk_ids)
        self.corpus = list(texts)
        self._rebuild()
        self.logger.info("bm25_index_built", num_chunks=len(self.corpus))

    def add(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Add more chunks. BM25Okapi has no incremental update, so this rebuilds."""
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids.extend(chunk_ids)
        self.corpus.extend(texts)
        self._rebuild()
        self.logger.info("bm25_index_updated", num_chunks=len(self.corpus))

    def _rebuild(self) -> None:
        tokenized = [_tokenize(t) for t in self.corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def remove(self, chunk_ids: Sequence[str]) -> None:
        """
        Remove chunks by id (e.g. the old chunks of a document that's being
        re-ingested) and rebuild. BM25Okapi has no incremental removal
        either, so - like add() - this is a full rebuild over what's left.
        """
        remove_set = set(chunk_ids)
        if not remove_set:
            return
        keep_ids: List[str] = []
        keep_corpus: List[str] = []
        for cid, text in zip(self.chunk_ids, self.corpus):
            if cid not in remove_set:
                keep_ids.append(cid)
                keep_corpus.append(text)
        removed = len(self.chunk_ids) - len(keep_ids)
        self.chunk_ids = keep_ids
        self.corpus = keep_corpus
        self._rebuild()
        self.logger.info("bm25_index_chunks_removed", removed=removed, remaining=len(self.corpus))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Return up to top_k (chunk_id, text, bm25_score) tuples, best first."""
        if self._bm25 is None or not self.corpus:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], self.corpus[i], float(scores[i])) for i in ranked_idx]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "corpus": self.corpus}, f)
        self.logger.info("bm25_index_saved", path=path)

    def load(self, path: str) -> bool:
        """Load a previously saved index. Returns False (no-op) if the file doesn't exist."""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.build(data["chunk_ids"], data["corpus"])
        self.logger.info("bm25_index_loaded", path=path)
        return True


def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[str]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Fuse multiple ranked lists of ids into one ranking using Reciprocal Rank
    Fusion (RRF):

        score(id) = sum_over_lists( weight / (k + rank_in_list + 1) )

    RRF needs no score normalization between BM25 and vector-similarity
    scores (which live on different scales) - it only uses each id's
    position/rank within each list, which is why it's the standard way to
    combine lexical and dense retrieval results. `k` (default 60, the value
    used in the original RRF paper) dampens the influence of any single
    list's top rank.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match number of ranked_lists")

    fused: Dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_list):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)

    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)