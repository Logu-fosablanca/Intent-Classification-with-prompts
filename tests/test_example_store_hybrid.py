"""
Tests for ExampleStore's hybrid FAISS+BM25 retrieval integration.

Verifies backward compatibility (hybrid off by default, dense-only fallback
without query_text) and that the reported "score" stays a cosine similarity
regardless of which ranking (dense-only or hybrid-fused) picked the order.
"""

import numpy as np
import pytest

from query_classifier.example_store import ExampleStore
from tests.conftest import MockEncoder, SAMPLE_EXAMPLES


@pytest.fixture
def encoder():
    return MockEncoder()


class TestHybridDefaults:
    def test_hybrid_off_by_default(self, encoder):
        store = ExampleStore(encoder=encoder)
        assert store.use_hybrid is False
        assert store._hybrid_retriever is None

    def test_hybrid_can_be_explicitly_enabled(self, encoder):
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        assert store.use_hybrid is True
        assert store._hybrid_retriever is not None

    def test_hybrid_can_be_explicitly_disabled(self, encoder):
        store = ExampleStore(encoder=encoder, use_hybrid=False)
        assert store.use_hybrid is False
        assert store._hybrid_retriever is None


class TestHybridRetrievalBehaviour:
    def test_without_query_text_falls_back_to_dense_only(self, encoder):
        """Hybrid enabled but no query_text passed -> identical to dense-only."""
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        store.add_examples_bulk(SAMPLE_EXAMPLES)

        query_emb = encoder.encode("account balance")
        with_text = store.retrieve_from_embedding(query_emb, k=5, query_text="account balance")
        without_text = store.retrieve_from_embedding(query_emb, k=5)

        # Both are valid results; without query_text must match a plain dense
        # store's ordering exactly (no BM25 involvement possible).
        plain_store = ExampleStore(encoder=encoder, use_hybrid=False)
        plain_store.add_examples_bulk(SAMPLE_EXAMPLES)
        plain_results = plain_store.retrieve_from_embedding(query_emb, k=5)

        assert [r["intent"] for r in without_text] == [r["intent"] for r in plain_results]

    def test_reported_score_is_cosine_not_rrf(self, encoder):
        """
        Regardless of hybrid fusion picking the ranking order, the 'score'
        field must stay a cosine similarity (roughly [-1, 1]) so downstream
        confidence blending/gating in nlp_engine.py keeps working unchanged.
        RRF scores are tiny (~0.01-0.03) — if this leaked through, every
        result would trip the low-similarity confidence gate.
        """
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        store.add_examples_bulk(SAMPLE_EXAMPLES)

        query_emb = encoder.encode("check my account balance")
        results = store.retrieve_from_embedding(
            query_emb, k=5, query_text="check my account balance"
        )
        for r in results:
            assert -1.0001 <= r["score"] <= 1.0001

    def test_hybrid_index_rebuilds_after_bulk_add(self, encoder):
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        store.add_examples_bulk(SAMPLE_EXAMPLES[:3])
        assert store._hybrid_dirty is True

        query_emb = encoder.encode("balance")
        store.retrieve_from_embedding(query_emb, k=2, query_text="balance")
        assert store._hybrid_dirty is False

        store.add_examples_bulk(SAMPLE_EXAMPLES[3:6])
        assert store._hybrid_dirty is True  # new examples invalidate the index

        # Retrieval after the second add should see the newly added examples.
        store.retrieve_from_embedding(query_emb, k=10, query_text="balance")
        assert store._hybrid_dirty is False
        assert store._hybrid_retriever._size == 6

    def test_hybrid_retrieval_still_respects_per_intent_limit(self, encoder):
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        store.add_examples_bulk([
            {"text": "what is my balance", "intent": "check_balance"},
            {"text": "check my account balance", "intent": "check_balance"},
            {"text": "how much money do I have", "intent": "check_balance"},
            {"text": "show me my balance", "intent": "check_balance"},
            {"text": "transfer money to savings", "intent": "transfer_money"},
        ])
        query_emb = encoder.encode("account balance")
        results = store.retrieve_from_embedding(
            query_emb, k=6, per_intent_limit=2, query_text="account balance"
        )
        counts = {}
        for r in results:
            counts[r["intent"]] = counts.get(r["intent"], 0) + 1
        for intent, count in counts.items():
            assert count <= 2

    def test_retrieve_method_passes_query_text_through(self, encoder):
        """The standalone retrieve() helper should activate hybrid ranking too."""
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        store.add_examples_bulk(SAMPLE_EXAMPLES)
        results = store.retrieve("check my balance", k=3)
        assert len(results) == 3
        assert store._hybrid_dirty is False

    def test_empty_store_with_hybrid_enabled_returns_empty(self, encoder):
        store = ExampleStore(encoder=encoder, use_hybrid=True)
        query_emb = encoder.encode("balance")
        results = store.retrieve_from_embedding(query_emb, k=5, query_text="balance")
        assert results == []
