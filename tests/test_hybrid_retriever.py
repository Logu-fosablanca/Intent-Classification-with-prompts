"""
Tests for HybridRetriever — FAISS dense + BM25 lexical fusion via RRF.

Assertions are deliberately robust to exact RRF arithmetic (which depends on
FAISS/BM25 internal tie-breaking) rather than hardcoding predicted indices:
where a specific outcome is asserted, it's re-derived inside the test from
the same cosine-similarity formula the retriever itself uses, not guessed.
"""

import numpy as np
import pytest

from query_classifier.hybrid_retriever import HybridRetriever, _tokenize


def _cosine_order(query_emb: np.ndarray, embeddings: np.ndarray):
    """Reference pure-dense ranking, independent of HybridRetriever internals."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(embeddings, axis=1) + 1e-8
    scores = embeddings.dot(q) / norms
    return list(np.argsort(scores)[::-1])


# ---------------------------------------------------------------------------
# A larger, asymmetric corpus to avoid accidental embedding-symmetry ties.
# ---------------------------------------------------------------------------
_TEXTS = [
    "check my account balance",           # 0
    "what is my current balance",         # 1
    "how much money do i have",           # 2
    "block the card ending 4521",         # 3  <- unique lexical token "4521"
    "transfer money to savings account",  # 4
    "send funds to john",                 # 5
    "apply for a personal loan",          # 6
    "what are your mortgage rates",       # 7
]

_EMBEDDINGS = np.array(
    [
        [0.90, 0.05, 0.02, 0.01],
        [0.85, 0.08, 0.03, 0.02],
        [0.80, 0.10, 0.05, 0.03],
        [0.15, 0.80, 0.03, 0.02],
        [0.05, 0.10, 0.85, 0.03],
        [0.03, 0.08, 0.80, 0.05],
        [0.02, 0.03, 0.05, 0.90],
        [0.01, 0.02, 0.04, 0.85],
    ],
    dtype=np.float32,
)


@pytest.fixture
def retriever():
    r = HybridRetriever()
    r.build(_TEXTS, _EMBEDDINGS)
    return r


class TestAvailability:
    def test_available_true_when_deps_installed(self):
        # faiss-cpu + rank-bm25 are installed in this test environment.
        assert HybridRetriever()._check_deps() is True

    def test_build_empty_corpus_is_noop(self):
        r = HybridRetriever()
        r.build([], None)
        assert r._faiss_index is None
        assert r._bm25 is None
        assert r.search("anything", np.zeros(4, dtype=np.float32)) == []

    def test_search_before_build_returns_empty(self):
        r = HybridRetriever()
        assert r.search("query", np.zeros(4, dtype=np.float32)) == []


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert _tokenize("Block the Card") == ["block", "the", "card"]

    def test_keeps_alphanumeric_tokens(self):
        assert _tokenize("ending 4521 now") == ["ending", "4521", "now"]

    def test_strips_punctuation(self):
        assert _tokenize("balance? now!") == ["balance", "now"]


class TestBuildAndSearch:
    def test_build_sets_size(self, retriever):
        assert retriever._size == len(_TEXTS)
        assert retriever._faiss_index is not None
        assert retriever._bm25 is not None

    def test_pure_dense_query_matches_reference_top1(self, retriever):
        # No lexical overlap at all with the corpus -> BM25 scores are all
        # tied at 0, so fusion should not disturb the dense front-runner.
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        expected_top1 = _cosine_order(query_emb, _EMBEDDINGS)[0]
        results = retriever.search("zzz qqq wwq", query_emb, k=8)
        assert results[0]["index"] == expected_top1

    def test_results_sorted_by_fused_score_desc(self, retriever):
        query_emb = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        results = retriever.search("balance card", query_emb, k=8)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_result_count(self, retriever):
        query_emb = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        results = retriever.search("balance card transfer", query_emb, k=3)
        assert len(results) <= 3

    def test_pool_capped_by_corpus_size(self, retriever):
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = retriever.search("balance", query_emb, k=10, pool=1000)
        assert len(results) <= len(_TEXTS)

    def test_all_result_indices_are_valid(self, retriever):
        query_emb = np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float32)
        results = retriever.search("loan mortgage transfer", query_emb, k=8)
        for r in results:
            assert 0 <= r["index"] < len(_TEXTS)

    def test_exclusive_lexical_token_promotes_its_document(self, retriever):
        """
        "4521" appears in exactly one document (idx 3). Even with a dense
        query that favours entirely different documents (the balance cluster,
        idx 0-2), fusing in the exclusive lexical match should rank idx 3
        higher than pure-dense ranking alone would.
        """
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # balance-leaning
        dense_only_rank = _cosine_order(query_emb, _EMBEDDINGS).index(3)

        results = retriever.search("the card ending 4521", query_emb, k=8)
        fused_rank = [r["index"] for r in results].index(3)

        assert fused_rank < dense_only_rank


class TestRebuild:
    def test_rebuild_reflects_new_corpus(self):
        r = HybridRetriever()
        r.build(_TEXTS[:2], _EMBEDDINGS[:2])
        assert r._size == 2

        r.build(_TEXTS, _EMBEDDINGS)
        assert r._size == len(_TEXTS)
        query_emb = np.array([0.15, 0.8, 0.03, 0.02], dtype=np.float32)
        results = r.search("card 4521", query_emb, k=8)
        assert results[0]["index"] == 3


class TestGracefulFallback:
    def test_unavailable_retriever_search_returns_empty(self):
        r = HybridRetriever()
        r.build(_TEXTS, _EMBEDDINGS)
        # Simulate missing optional deps discovered after build.
        r.available = False
        assert r.search("balance", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)) == []
