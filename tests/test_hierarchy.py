"""
Tests for HierarchicalRouter: coarse-to-fine two-level routing.
"""

import numpy as np
import pytest

from tests.conftest import MockEncoder, SAMPLE_INTENTS, SAMPLE_HIERARCHY
from query_classifier.hierarchy import HierarchicalRouter


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def hier_router(encoder):
    return HierarchicalRouter(
        intents=SAMPLE_INTENTS,
        hierarchy=SAMPLE_HIERARCHY,
        encoder=encoder,
        top_n_categories=2,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_category_names_populated(self, hier_router):
        assert set(hier_router.category_names) == set(SAMPLE_HIERARCHY.keys())

    def test_category_embeddings_shape(self, hier_router):
        n_cats = len(SAMPLE_HIERARCHY)
        assert hier_router.category_embeddings.shape[0] == n_cats

    def test_per_category_intent_embeddings_built(self, hier_router):
        for cat_name in SAMPLE_HIERARCHY:
            assert cat_name in hier_router.category_intent_embeddings
            assert cat_name in hier_router.category_intent_lists

    def test_intents_by_name_populated(self, hier_router):
        for intent in SAMPLE_INTENTS:
            assert intent["name"] in hier_router.intents_by_name


# ---------------------------------------------------------------------------
# route() — basic
# ---------------------------------------------------------------------------

class TestRoute:
    def test_returns_tuple(self, hier_router, encoder):
        emb = encoder.encode("check my balance")
        result = hier_router.route(emb)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matched_categories_non_empty(self, hier_router, encoder):
        emb = encoder.encode("check my balance")
        cats, _ = hier_router.route(emb)
        assert len(cats) > 0

    def test_top_intents_non_empty(self, hier_router, encoder):
        emb = encoder.encode("check balance")
        _, intents = hier_router.route(emb)
        assert len(intents) > 0

    def test_intent_result_structure(self, hier_router, encoder):
        emb = encoder.encode("check balance")
        _, intents = hier_router.route(emb)
        item = intents[0]
        assert "intent" in item
        assert "score" in item
        assert "category" in item
        assert isinstance(item["score"], float)

    def test_top_intents_sorted_descending(self, hier_router, encoder):
        emb = encoder.encode("account balance")
        _, intents = hier_router.route(emb)
        scores = [i["score"] for i in intents]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_intents_limit(self, hier_router, encoder):
        emb = encoder.encode("card block")
        _, intents = hier_router.route(emb, top_k_intents=2)
        assert len(intents) <= 2

    def test_balance_query_finds_check_balance(self, hier_router, encoder):
        emb = encoder.encode("what is my account balance")
        _, intents = hier_router.route(emb)
        names = [i["intent"]["name"] for i in intents]
        assert "check_balance" in names

    def test_card_query_finds_block_card(self, hier_router, encoder):
        emb = encoder.encode("block my credit card")
        _, intents = hier_router.route(emb)
        names = [i["intent"]["name"] for i in intents]
        assert "block_card" in names

    def test_intents_come_from_matched_categories_only(self, hier_router, encoder):
        emb = encoder.encode("account balance statement")
        cats, intents = hier_router.route(emb, top_k_intents=5)
        for item in intents:
            assert item["category"] in cats


# ---------------------------------------------------------------------------
# Prior category pinning (multi-turn)
# ---------------------------------------------------------------------------

class TestPriorCategoryPinning:
    def test_prior_category_appears_in_matched(self, hier_router, encoder):
        # "for last 6 months" alone has no vocab signal — would miss "accounts"
        emb = encoder.encode("for last 6 months")
        cats_no_prior, _ = hier_router.route(emb)

        cats_with_prior, _ = hier_router.route(emb, prior_categories=["accounts"])
        assert "accounts" in cats_with_prior

    def test_prior_category_intents_included(self, hier_router, encoder):
        # After pinning "accounts", bank_statement should appear in candidates
        emb = encoder.encode("for last 6 months")
        _, intents = hier_router.route(emb, prior_categories=["accounts"])
        names = [i["intent"]["name"] for i in intents]
        assert "bank_statement" in names or "check_balance" in names

    def test_invalid_prior_category_ignored(self, hier_router, encoder):
        emb = encoder.encode("check balance")
        cats, _ = hier_router.route(emb, prior_categories=["nonexistent_category"])
        assert "nonexistent_category" not in cats

    def test_prior_category_already_in_top_not_duplicated(self, hier_router, encoder):
        emb = encoder.encode("what is my account balance")
        cats, _ = hier_router.route(emb, prior_categories=["accounts"])
        assert cats.count("accounts") == 1


# ---------------------------------------------------------------------------
# get_category_for_intent
# ---------------------------------------------------------------------------

class TestGetCategoryForIntent:
    def test_known_intent_returns_category(self, hier_router):
        cat = hier_router.get_category_for_intent("check_balance")
        assert cat == "accounts"

    def test_block_card_is_in_cards(self, hier_router):
        cat = hier_router.get_category_for_intent("block_card")
        assert cat == "cards"

    def test_unknown_intent_returns_none(self, hier_router):
        cat = hier_router.get_category_for_intent("does_not_exist")
        assert cat is None

    def test_all_sample_intents_have_categories(self, hier_router):
        for intent in SAMPLE_INTENTS:
            cat = hier_router.get_category_for_intent(intent["name"])
            assert cat is not None, f"Intent '{intent['name']}' has no category"
