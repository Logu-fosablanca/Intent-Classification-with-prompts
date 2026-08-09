"""
HybridRetriever: FAISS (dense ANN) + BM25 (lexical) retrieval fused via
Reciprocal Rank Fusion (RRF).

Sits underneath ExampleStore as an optional ranking layer. Dense embeddings
blur exact tokens — account numbers, "ending in 4521", merchant names — that
BM25 catches directly; RRF combines the two ranked lists without needing to
calibrate their very different score scales (cosine similarity ∈ [-1, 1] vs.
unbounded BM25 scores).

Degrades gracefully: if faiss-cpu / rank-bm25 are not installed, `available`
is False and callers are expected to fall back to their existing dense-only
ranking (see ExampleStore.retrieve_from_embedding).
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from query_classifier.config import HYBRID_RRF_K, FAISS_INDEX_TYPE

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """
    Combines dense (FAISS) and lexical (BM25) retrieval over the same corpus.

    Built from parallel arrays (texts + embeddings) that the caller already
    owns — rebuilding the indices is cheap relative to encoding, since no
    embedding computation happens here.

    Usage:
        retriever = HybridRetriever()
        retriever.build(texts, embeddings)          # after any corpus change
        fused = retriever.search(query_text, query_emb, k=10)
        # fused: [{"index": int, "score": float}, ...] sorted by RRF score desc.
        # `index` refers back into the `texts`/`embeddings` passed to build().
    """

    def __init__(self, rrf_k: int = HYBRID_RRF_K, index_type: str = FAISS_INDEX_TYPE):
        self.rrf_k = rrf_k
        self.index_type = index_type
        self._faiss_index = None
        self._bm25 = None
        self._size = 0
        self.available = self._check_deps()

    @staticmethod
    def _check_deps() -> bool:
        try:
            import faiss  # noqa: F401
            from rank_bm25 import BM25Okapi  # noqa: F401
            return True
        except ImportError as e:
            logger.warning(
                f"HybridRetriever: faiss-cpu / rank-bm25 not installed ({e}). "
                "Install with `pip install query-classifier[rerank]` to enable "
                "hybrid FAISS+BM25 retrieval. Falling back to dense-only ranking."
            )
            return False

    def build(self, texts: List[str], embeddings: Optional[np.ndarray]):
        """
        (Re)build both indices from the current corpus.

        O(N) — call once after a batch of adds/loads, not per query. No-op
        (indices cleared) if the corpus is empty or optional deps are missing.
        """
        self._size = len(texts)
        self._faiss_index = None
        self._bm25 = None

        if self._size == 0 or not self.available or embeddings is None:
            return

        import faiss
        from rank_bm25 import BM25Okapi

        # --- Dense: FAISS over L2-normalized vectors (inner product == cosine) ---
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        vecs = vecs / norms

        dim = vecs.shape[1]
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self._faiss_index = index

        # --- Lexical: BM25 over tokenized texts ---
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])

        logger.info(
            f"HybridRetriever: built FAISS({self.index_type}) + BM25 over {self._size} items."
        )

    def search(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int = 10,
        pool: int = 50,
    ) -> List[Dict]:
        """
        Return the RRF-fused ranking as [{"index": int, "score": float}, ...],
        sorted descending by fused score, top-k.

        `pool` controls how many results each individual retriever contributes
        before fusion — wider pool improves recall at the cost of more fusion
        work. `score` here is the raw RRF score (tiny, e.g. ~0.03), NOT a
        similarity — callers that need an interpretable score should recompute
        cosine similarity for the returned indices themselves.
        """
        if self._size == 0 or not self.available or self._faiss_index is None or self._bm25 is None:
            return []

        pool = min(pool, self._size)

        # Dense ranking
        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        _, dense_idx = self._faiss_index.search(q, pool)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_idx[0]) if idx != -1}

        # Lexical ranking
        bm25_scores = self._bm25.get_scores(_tokenize(query_text))
        bm25_order = np.argsort(bm25_scores)[::-1][:pool]
        lexical_rank = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        # Reciprocal Rank Fusion
        fused = []
        for idx in set(dense_rank) | set(lexical_rank):
            score = 0.0
            if idx in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[idx] + 1)
            if idx in lexical_rank:
                score += 1.0 / (self.rrf_k + lexical_rank[idx] + 1)
            fused.append({"index": idx, "score": score})

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:k]
