"""
Tests for SemanticRouter using MockEncoder (offline, no model downloads).
"""

import numpy as np
import pytest

from tests.conftest import MockEncoder, SAMPLE_INTENTS
from query_classifier.semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Fixture: router backed by MockEncoder (bypasses SentenceTransformer loading)
# ---------------------------------------------------------------------------

@pytest.fixture
def router(encoder):
    """SemanticRouter with MockEncoder injected directly."""
    r = SemanticRouter.__new__(SemanticRouter)
    r.intents = SAMPLE_INTENTS
    r.descriptions = [i["description"] for i in SAMPLE_INTENTS]
    r.model = encoder
    r.vectorizer = None
    r.tfidf_matrix = None
    r._intent_embeddings = np.array(
        encoder.encode(r.descriptions, show_progress_bar=False),
        dtype=np.float32,
    )
    return r


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

class TestEncodeQuery:
    def test_returns_ndarray(self, router):
        emb = router.encode_query("check my balance")
        assert isinstance(emb, np.ndarray)

    def test_returns_float32(self, router):
        emb = router.encode_query("what is my account balance")
        assert emb.dtype == np.float32

    def test_returns_none_without_model(self, encoder):
        r = SemanticRouter.__new__(SemanticRouter)
        r.model = None
        r.vectorizer = None
        result = r.encode_query("anything")
        assert result is None

    def test_different_queries_different_embeddings(self, router):
        emb1 = router.encode_query("check balance")
        emb2 = router.encode_query("block my card")
        assert not np.allclose(emb1, emb2)


# ---------------------------------------------------------------------------
# find_top_k_from_embedding
# ---------------------------------------------------------------------------

class TestFindTopKFromEmbedding:
    def test_returns_correct_count(self, router, encoder):
        emb = encoder.encode("check my balance")
        results = router.find_top_k_from_embedding(emb, k=3)
        assert len(results) == 3

    def test_returns_all_if_k_larger_than_intents(self, router, encoder):
        emb = encoder.encode("balance")
        results = router.find_top_k_from_embedding(emb, k=100)
        assert len(results) == len(SAMPLE_INTENTS)

    def test_result_structure(self, router, encoder):
        emb = encoder.encode("check balance")
        results = router.find_top_k_from_embedding(emb, k=1)
        assert "intent" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_sorted_descending(self, router, encoder):
        emb = encoder.encode("check balance")
        results = router.find_top_k_from_embedding(emb, k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_balance_query_returns_check_balance(self, router, encoder):
        emb = encoder.encode("account balance")
        results = router.find_top_k_from_embedding(emb, k=3)
        top_names = [r["intent"]["name"] for r in results]
        assert "check_balance" in top_names

    def test_no_embeddings_returns_fallback(self, encoder):
        r = SemanticRouter.__new__(SemanticRouter)
        r.intents = SAMPLE_INTENTS
        r.model = encoder
        r._intent_embeddings = None
        emb = encoder.encode("anything")
        results = r.find_top_k_from_embedding(emb, k=3)
        assert len(results) == 3
        assert all(r["score"] == 0.0 for r in results)

    def test_scores_in_valid_cosine_range(self, router, encoder):
        emb = encoder.encode("transfer money to savings")
        results = router.find_top_k_from_embedding(emb, k=len(SAMPLE_INTENTS))
        for r in results:
            assert -1.01 <= r["score"] <= 1.01


# ---------------------------------------------------------------------------
# find_top_k (high-level)
# ---------------------------------------------------------------------------

class TestFindTopK:
    def test_returns_correct_count(self, router):
        results = router.find_top_k("block my card", k=3)
        assert len(results) == 3

    def test_sorted_descending(self, router):
        results = router.find_top_k("loan application", k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_card_query_finds_block_card(self, router):
        results = router.find_top_k("block my card", k=3)
        top_names = [r["intent"]["name"] for r in results]
        assert "block_card" in top_names

    def test_loan_query_finds_apply_loan(self, router):
        results = router.find_top_k("apply for a personal loan", k=3)
        top_names = [r["intent"]["name"] for r in results]
        assert "apply_loan" in top_names

    def test_no_model_no_vectorizer_returns_fallback(self):
        r = SemanticRouter.__new__(SemanticRouter)
        r.intents = SAMPLE_INTENTS
        r.model = None
        r.vectorizer = None
        r._intent_embeddings = None
        results = r.find_top_k("anything", k=3)
        assert len(results) == 3
        assert all(res["score"] == 0.0 for res in results)
