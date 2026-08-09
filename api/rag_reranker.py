"""
reranker.py
-----------
Cross-encoder based re-ranking for retrieved chunks.

Why a re-ranker is needed:
The embedding models used for retrieval (HuggingFaceEmbedding / SentenceTransformer)
are *bi-encoders*: the query and each document are embedded independently into
vectors, and similarity is a fast dot-product/cosine comparison. This is what makes
vector search over millions of chunks feasible, but it's an approximation - the model
never actually looks at the query and a candidate document *together*.

A *cross-encoder* re-ranker does look at (query, document) pairs jointly and outputs
a single relevance score. It is far more accurate but much slower, so it is never used
to search the whole corpus - only to re-score a small shortlist that a bi-encoder
already narrowed down.

Standard pattern used here:
    1. Bi-encoder retrieval: get top_k (e.g. 15-20) candidate chunks - fast, coarse.
    2. Cross-encoder re-rank: score all top_k pairs - slow but only 15-20 calls.
    3. Keep top_n (e.g. 4) highest scoring chunks to pass to the LLM as context.
"""

from typing import List, Tuple
import structlog

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required for reranking. "
        "pip install sentence-transformers"
    ) from e


class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.logger = structlog.get_logger()
        self.model_name = model_name
        self.logger.info("reranker_loading", model=model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 4,
    ) -> List[Tuple[str, float]]:
        """
        Score every (query, document) pair and return the top_n documents,
        each paired with its relevance score, sorted best-first.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)  # higher = more relevant

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        top = ranked[:top_n]

        self.logger.info(
            "rerank_completed",
            candidates=len(documents),
            kept=len(top),
            top_score=float(top[0][1]) if top else None,
            bottom_score=float(top[-1][1]) if top else None,
        )
        return top

    def rerank_texts_only(self, query: str, documents: List[str], top_n: int = 4) -> List[str]:
        """Convenience wrapper that drops the scores, returning just the text."""
        return [doc for doc, _score in self.rerank(query, documents, top_n=top_n)]