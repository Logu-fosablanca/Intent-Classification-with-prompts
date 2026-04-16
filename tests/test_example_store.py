"""
Tests for ExampleStore — single-turn, multi-turn, retrieval, persistence.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from tests.conftest import MockEncoder, SAMPLE_EXAMPLES, MULTI_TURN_EXAMPLES
from query_classifier.example_store import ExampleStore


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_add_single_example(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_example("what is my balance", "check_balance")
        assert len(store) == 1
        assert store._texts[0] == "what is my balance"
        assert store._intents[0] == "check_balance"
        assert store._history_contexts[0] is None

    def test_add_multi_turn_example(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_example(
            text="for last 6 months",
            intent_name="bank_statement",
            history_context="I need my bank statement",
        )
        assert len(store) == 1
        assert store._history_contexts[0] == "I need my bank statement"

    def test_add_examples_bulk_single_turn(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(SAMPLE_EXAMPLES)
        assert len(store) == len(SAMPLE_EXAMPLES)

    def test_add_examples_bulk_multi_turn(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(MULTI_TURN_EXAMPLES)
        assert len(store) == len(MULTI_TURN_EXAMPLES)
        # All have history_context
        assert all(c is not None for c in store._history_contexts)

    def test_add_examples_bulk_mixed(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(SAMPLE_EXAMPLES)
        store.add_examples_bulk(MULTI_TURN_EXAMPLES)
        assert len(store) == len(SAMPLE_EXAMPLES) + len(MULTI_TURN_EXAMPLES)

    def test_embeddings_shape(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(SAMPLE_EXAMPLES)
        assert store._embeddings is not None
        assert store._embeddings.shape[0] == len(SAMPLE_EXAMPLES)

    def test_empty_store(self, encoder):
        store = ExampleStore(encoder=encoder)
        assert len(store) == 0
        assert store._embeddings is None

    def test_add_example_no_encoder(self):
        store = ExampleStore.__new__(ExampleStore)
        store._texts = []
        store._intents = []
        store._history_contexts = []
        store._embeddings = None
        store.encoder = None
        # Should warn and not crash
        store.add_example("test", "intent")
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_retrieve_returns_correct_number(self, populated_store, encoder):
        query_emb = encoder.encode("check balance")
        results = populated_store.retrieve_from_embedding(query_emb, k=3)
        assert len(results) == 3

    def test_retrieve_returns_less_than_k_if_store_smaller(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_example("check balance", "check_balance")
        query_emb = encoder.encode("balance")
        results = store.retrieve_from_embedding(query_emb, k=10)
        assert len(results) == 1

    def test_retrieve_empty_store(self, encoder):
        store = ExampleStore(encoder=encoder)
        query_emb = encoder.encode("balance")
        results = store.retrieve_from_embedding(query_emb, k=5)
        assert results == []

    def test_retrieve_result_structure(self, populated_store, encoder):
        query_emb = encoder.encode("check balance")
        results = populated_store.retrieve_from_embedding(query_emb, k=1)
        assert "text" in results[0]
        assert "intent" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_retrieve_sorted_by_score(self, populated_store, encoder):
        query_emb = encoder.encode("account balance check")
        results = populated_store.retrieve_from_embedding(query_emb, k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_correct_intent_for_balance_query(self, populated_store, encoder):
        # "account balance" should retrieve check_balance examples
        query_emb = encoder.encode("account balance")
        results = populated_store.retrieve_from_embedding(query_emb, k=3)
        top_intents = [r["intent"] for r in results]
        assert "check_balance" in top_intents

    def test_multi_turn_context_enriches_retrieval(self, encoder):
        """
        A bare follow-up "for last 6 months" alone retrieves poorly.
        The same phrase prepended with "I need my bank statement" retrieves
        the multi-turn example correctly.
        """
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk([
            {"text": "I need my bank statement", "intent": "bank_statement"},
            {
                "text": "for last 6 months",
                "intent": "bank_statement",
                "history_context": "I need my bank statement",
            },
        ])

        # Bare follow-up: low-signal query
        bare_emb = encoder.encode("for last 6 months")
        bare_results = store.retrieve_from_embedding(bare_emb, k=2)

        # Context-enriched query
        enriched_emb = encoder.encode("I need my bank statement for last 6 months")
        enriched_results = store.retrieve_from_embedding(enriched_emb, k=2)

        # Enriched query must score higher for the multi-turn example
        enriched_top_score = enriched_results[0]["score"]
        bare_top_score = bare_results[0]["score"]
        assert enriched_top_score >= bare_top_score

    def test_retrieve_via_retrieve_method(self, populated_store):
        results = populated_store.retrieve("check my balance", k=3)
        assert len(results) == 3
        assert all("intent" in r for r in results)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, populated_store, encoder, tmp_path):
        path = str(tmp_path / "store")
        populated_store.save(path)

        assert os.path.exists(f"{path}.json")
        assert os.path.exists(f"{path}.npy")

        loaded = ExampleStore(encoder=encoder)
        loaded.load(path)

        assert len(loaded) == len(populated_store)
        assert loaded._texts == populated_store._texts
        assert loaded._intents == populated_store._intents
        assert loaded._history_contexts == populated_store._history_contexts

    def test_save_json_structure(self, populated_store, tmp_path):
        path = str(tmp_path / "store")
        populated_store.save(path)
        with open(f"{path}.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert "text" in data[0]
        assert "intent" in data[0]
        assert "history_context" in data[0]

    def test_load_recomputes_embeddings_without_npy(self, populated_store, encoder, tmp_path):
        path = str(tmp_path / "store")
        populated_store.save(path)
        os.remove(f"{path}.npy")  # force recompute

        loaded = ExampleStore(encoder=encoder)
        loaded.load(path)
        assert loaded._embeddings is not None
        assert loaded._embeddings.shape[0] == len(populated_store)

    def test_embeddings_preserved_after_load(self, populated_store, encoder, tmp_path):
        path = str(tmp_path / "store")
        populated_store.save(path)

        loaded = ExampleStore(encoder=encoder)
        loaded.load(path)

        # Retrieval should work the same on loaded store
        query_emb = encoder.encode("account balance")
        original_results = populated_store.retrieve_from_embedding(query_emb, k=3)
        loaded_results = loaded.retrieve_from_embedding(query_emb, k=3)

        assert [r["intent"] for r in original_results] == [r["intent"] for r in loaded_results]


# ---------------------------------------------------------------------------
# Coverage utilities
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_intent_coverage(self, populated_store):
        coverage = populated_store.intent_coverage()
        assert isinstance(coverage, dict)
        assert "check_balance" in coverage
        assert coverage["check_balance"] == 2  # from SAMPLE_EXAMPLES

    def test_multi_turn_coverage(self, populated_store):
        coverage = populated_store.multi_turn_coverage()
        assert isinstance(coverage, dict)
        assert "bank_statement" in coverage
        assert "block_card" in coverage

    def test_multi_turn_coverage_empty_store(self, encoder):
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(SAMPLE_EXAMPLES)  # no multi-turn
        assert store.multi_turn_coverage() == {}
