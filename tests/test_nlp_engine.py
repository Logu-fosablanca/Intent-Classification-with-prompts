"""
Tests for IntentClassifier — covers:
  - TurnMode and ClassificationMode enums
  - _validate_mode_config fail-fast
  - _extract_prior_intent (both formats)
  - _build_rag_query (sliding window context)
  - _extract_json (plain, fenced, brace-scan)
  - _build_prompt (mode-aware sections)
  - classify() with mocked LLM (all three modes)
  - Confidence gate (low retrieval score caps confidence)
  - Verification pass
  - Language detection disabled path
"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from query_classifier.nlp_engine import IntentClassifier, ClassificationMode, TurnMode
from tests.conftest import (
    MockEncoder, SAMPLE_INTENTS, SAMPLE_HIERARCHY,
    SAMPLE_EXAMPLES, MULTI_TURN_EXAMPLES,
)
from query_classifier.example_store import ExampleStore


# ---------------------------------------------------------------------------
# Helper: build a minimal IntentClassifier without loading any real models
# ---------------------------------------------------------------------------

def _make_classifier(
    mode=ClassificationMode.FLAT,
    turn_mode=TurnMode.SINGLE,
    example_store=None,
    intent_hierarchy=None,
    encoder=None,
):
    """
    Constructs IntentClassifier bypassing __init__ entirely so that
    no SentenceTransformer, LLM, or language-detection model is loaded.
    """
    clf = IntentClassifier.__new__(IntentClassifier)
    clf.mode = ClassificationMode(mode)
    clf.turn_mode = TurnMode(turn_mode)
    clf.llm_provider = "ollama"
    clf.llm_model_name = "test-model"
    clf.llm_base_url = "http://localhost:11434"
    clf.llm_api_key = ""
    clf.enable_lang_detect = False
    clf.tokenizer = None
    clf.lang_model = None
    clf._intents_by_name = {i["name"]: i for i in SAMPLE_INTENTS}
    clf._hierarchy = intent_hierarchy or {}

    # Inject MockEncoder-backed router
    from query_classifier.semantic_router import SemanticRouter
    import numpy as np

    router = SemanticRouter.__new__(SemanticRouter)
    router.intents = SAMPLE_INTENTS
    router.descriptions = [i["description"] for i in SAMPLE_INTENTS]
    router.model = encoder or MockEncoder()
    router.vectorizer = None
    router.tfidf_matrix = None
    router._intent_embeddings = np.array(
        router.model.encode(router.descriptions, show_progress_bar=False),
        dtype=np.float32,
    )
    clf.router = router

    # Hierarchical router
    clf.hier_router = None
    if clf.mode == ClassificationMode.HIERARCHICAL_RAG and intent_hierarchy:
        from query_classifier.hierarchy import HierarchicalRouter
        clf.hier_router = HierarchicalRouter(
            intents=SAMPLE_INTENTS,
            hierarchy=intent_hierarchy,
            encoder=router.model,
            top_n_categories=2,
        )

    clf.example_store = example_store
    return clf


def _make_store(encoder):
    store = ExampleStore(encoder=encoder)
    store.add_examples_bulk(SAMPLE_EXAMPLES)
    store.add_examples_bulk(MULTI_TURN_EXAMPLES)
    return store


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------

class TestEnums:
    def test_turn_mode_values(self):
        assert TurnMode.SINGLE == "single"
        assert TurnMode.MULTI == "multi"

    def test_classification_mode_values(self):
        assert ClassificationMode.FLAT == "flat"
        assert ClassificationMode.FLAT_RAG == "flat_rag"
        assert ClassificationMode.HIERARCHICAL_RAG == "hierarchical_rag"

    def test_turn_mode_from_string(self):
        assert TurnMode("single") == TurnMode.SINGLE
        assert TurnMode("multi") == TurnMode.MULTI

    def test_classification_mode_from_string(self):
        assert ClassificationMode("hierarchical_rag") == ClassificationMode.HIERARCHICAL_RAG


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_hierarchical_rag_without_hierarchy_raises(self):
        with pytest.raises(ValueError, match="intent_hierarchy"):
            IntentClassifier._validate_mode_config(
                mode=ClassificationMode.HIERARCHICAL_RAG,
                example_store=None,
                intent_hierarchy=None,
            )

    def test_hierarchical_rag_with_hierarchy_ok(self):
        # Should not raise
        IntentClassifier._validate_mode_config(
            mode=ClassificationMode.HIERARCHICAL_RAG,
            example_store=None,
            intent_hierarchy=SAMPLE_HIERARCHY,
        )

    def test_flat_with_example_store_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            IntentClassifier._validate_mode_config(
                mode=ClassificationMode.FLAT,
                example_store=object(),  # not None
                intent_hierarchy=None,
            )
        assert "flat" in caplog.text.lower() or "example_store" in caplog.text.lower()

    def test_flat_rag_without_hierarchy_ok(self):
        IntentClassifier._validate_mode_config(
            mode=ClassificationMode.FLAT_RAG,
            example_store=None,
            intent_hierarchy=None,
        )


# ---------------------------------------------------------------------------
# _extract_prior_intent
# ---------------------------------------------------------------------------

class TestExtractPriorIntent:
    def setup_method(self):
        self.clf = _make_classifier()

    def test_explicit_field(self):
        history = [
            {"role": "user", "content": "I need my bank statement"},
            {"role": "assistant", "intent_classified": "bank_statement", "content": "..."},
        ]
        assert self.clf._extract_prior_intent(history) == "bank_statement"

    def test_bracket_format(self):
        history = [
            {"role": "user", "content": "block my card"},
            {"role": "assistant", "content": "Got it. [intent: block_card]"},
        ]
        assert self.clf._extract_prior_intent(history) == "block_card"

    def test_no_assistant_turn_returns_none(self):
        history = [{"role": "user", "content": "hello"}]
        assert self.clf._extract_prior_intent(history) is None

    def test_empty_history_returns_none(self):
        assert self.clf._extract_prior_intent([]) is None

    def test_explicit_field_takes_precedence_over_bracket(self):
        history = [
            {"role": "assistant",
             "intent_classified": "check_balance",
             "content": "[intent: transfer_money]"},
        ]
        assert self.clf._extract_prior_intent(history) == "check_balance"

    def test_picks_most_recent_assistant(self):
        history = [
            {"role": "assistant", "intent_classified": "check_balance"},
            {"role": "user", "content": "now something else"},
            {"role": "assistant", "intent_classified": "transfer_money"},
        ]
        assert self.clf._extract_prior_intent(history) == "transfer_money"


# ---------------------------------------------------------------------------
# _build_rag_query
# ---------------------------------------------------------------------------

class TestBuildRagQuery:
    def setup_method(self):
        self.clf = _make_classifier()

    def test_no_history_returns_text(self):
        result = self.clf._build_rag_query("hello", None)
        assert result == "hello"

    def test_empty_history_returns_text(self):
        result = self.clf._build_rag_query("hello", [])
        assert result == "hello"

    def test_prepends_prior_user_turn(self):
        history = [
            {"role": "user", "content": "I need my bank statement"},
            {"role": "assistant", "content": "Sure, which period?"},
        ]
        result = self.clf._build_rag_query("for last 6 months", history)
        assert "I need my bank statement" in result
        assert "for last 6 months" in result

    def test_window_limits_context(self):
        history = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "turn 3"},
            {"role": "assistant", "content": "..."},
        ]
        result = self.clf._build_rag_query("now", history, window=2)
        # Should include turn 2 and turn 3 but not turn 1
        assert "turn 3" in result
        assert "turn 2" in result
        assert "turn 1" not in result

    def test_current_text_not_duplicated(self):
        history = [{"role": "user", "content": "for last 6 months"}]
        result = self.clf._build_rag_query("for last 6 months", history)
        # Same as text — history item == current text, should not duplicate
        assert result == "for last 6 months"

    def test_only_user_turns_prepended(self):
        history = [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "check balance"},
        ]
        result = self.clf._build_rag_query("how much", history)
        assert "How can I help?" not in result
        assert "check balance" in result


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def setup_method(self):
        self.clf = _make_classifier()

    def test_plain_json(self):
        content = '{"name": "check_balance", "confidence": 0.9, "reasoning": "test"}'
        result = self.clf._extract_json(content)
        assert result["name"] == "check_balance"
        assert result["confidence"] == 0.9

    def test_json_fenced_with_json_tag(self):
        content = '```json\n{"name": "transfer_money", "confidence": 0.85}\n```'
        result = self.clf._extract_json(content)
        assert result["name"] == "transfer_money"

    def test_json_fenced_without_tag(self):
        content = '```\n{"name": "block_card", "confidence": 0.7}\n```'
        result = self.clf._extract_json(content)
        assert result["name"] == "block_card"

    def test_json_embedded_in_text(self):
        content = 'Here is the result: {"name": "apply_loan", "confidence": 0.8} done.'
        result = self.clf._extract_json(content)
        assert result["name"] == "apply_loan"

    def test_verify_format(self):
        content = '{"is_correct": true, "better_intent": null, "confidence_score": 0.9}'
        result = self.clf._extract_json(content)
        assert result["is_correct"] is True

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            self.clf._extract_json("no json here at all")


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def setup_method(self):
        self.top_matches = [
            {"intent": {"name": "check_balance", "description": "Check account balance."}, "score": 0.9},
            {"intent": {"name": "bank_statement", "description": "Get bank statement."}, "score": 0.7},
        ]
        self.rag_examples = [
            {"text": "what is my balance", "intent": "check_balance", "score": 0.92},
        ]

    def test_flat_prompt_has_candidates(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT)
        prompt = clf._build_prompt("check balance", self.top_matches, [], [], None)
        assert "check_balance" in prompt
        assert "CANDIDATE INTENTS" in prompt

    def test_flat_rag_includes_retrieved_examples(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT_RAG)
        prompt = clf._build_prompt("check balance", self.top_matches, self.rag_examples, [], None)
        assert "RETRIEVED EXAMPLES" in prompt
        assert "what is my balance" in prompt

    def test_hierarchical_includes_domain_context(self):
        clf = _make_classifier(
            mode=ClassificationMode.HIERARCHICAL_RAG,
            intent_hierarchy=SAMPLE_HIERARCHY,
        )
        prompt = clf._build_prompt(
            "check balance", self.top_matches, self.rag_examples, ["accounts"], None
        )
        assert "DOMAIN CONTEXT" in prompt
        assert "Accounts" in prompt

    def test_multi_turn_includes_conversation_history(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.MULTI)
        history = [{"role": "user", "content": "I need my statement"}]
        prompt = clf._build_prompt("for last 6 months", self.top_matches, [], [], history)
        assert "CONVERSATION HISTORY" in prompt
        assert "I need my statement" in prompt

    def test_single_turn_omits_conversation_history(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE)
        history = [{"role": "user", "content": "I need my statement"}]
        prompt = clf._build_prompt("for last 6 months", self.top_matches, [], [], history)
        assert "CONVERSATION HISTORY" not in prompt

    def test_no_rag_examples_shows_fallback_text(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT_RAG)
        prompt = clf._build_prompt("check balance", self.top_matches, [], [], None)
        assert "No labeled examples" in prompt


# ---------------------------------------------------------------------------
# classify() — full async pipeline with mocked LLM
# ---------------------------------------------------------------------------

def _make_mock_llm_response(intent_name="check_balance", confidence=0.9):
    mock_response = {
        "message": {
            "content": json.dumps({
                "name": intent_name,
                "confidence": confidence,
                "reasoning": "mock reasoning",
            })
        }
    }
    return mock_response


async def _mock_chat(**kwargs):
    return _make_mock_llm_response()


class TestClassify:
    @pytest.fixture
    def encoder(self):
        return MockEncoder()

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _patch_llm(self, clf, response=None):
        """Replace _get_client with one that returns a mock AsyncClient."""
        mock_client = MagicMock()
        resp = response or _make_mock_llm_response()
        mock_client.chat = AsyncMock(return_value=resp)
        clf._get_client = MagicMock(return_value=mock_client)
        return mock_client

    def test_flat_single_turn_returns_tuple(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)
        self._patch_llm(clf)
        result = self._run(clf.classify("what is my balance"))
        assert isinstance(result, tuple)
        assert len(result) == 3
        intent, conf, lang = result
        assert isinstance(intent, str)
        assert isinstance(conf, float)
        assert isinstance(lang, str)

    def test_flat_single_turn_returns_llm_intent(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)
        self._patch_llm(clf, _make_mock_llm_response("check_balance", 0.95))
        intent, conf, _ = self._run(clf.classify("what is my balance"))
        assert intent == "check_balance"
        assert conf == 0.95

    def test_flat_rag_uses_example_store(self, encoder):
        store = _make_store(encoder)
        clf = _make_classifier(
            mode=ClassificationMode.FLAT_RAG,
            turn_mode=TurnMode.SINGLE,
            example_store=store,
            encoder=encoder,
        )
        self._patch_llm(clf)
        # Should not raise; RAG retrieval runs
        result = self._run(clf.classify("check my balance"))
        assert result is not None

    def test_hierarchical_rag_mode(self, encoder):
        store = _make_store(encoder)
        clf = _make_classifier(
            mode=ClassificationMode.HIERARCHICAL_RAG,
            turn_mode=TurnMode.MULTI,
            example_store=store,
            intent_hierarchy=SAMPLE_HIERARCHY,
            encoder=encoder,
        )
        self._patch_llm(clf, _make_mock_llm_response("check_balance", 0.9))
        intent, conf, _ = self._run(clf.classify("what is my account balance"))
        assert intent == "check_balance"

    def test_multi_turn_with_history(self, encoder):
        store = _make_store(encoder)
        clf = _make_classifier(
            mode=ClassificationMode.FLAT_RAG,
            turn_mode=TurnMode.MULTI,
            example_store=store,
            encoder=encoder,
        )
        self._patch_llm(clf, _make_mock_llm_response("bank_statement", 0.88))
        history = [
            {"role": "user", "content": "I need my bank statement"},
            {"role": "assistant", "intent_classified": "bank_statement", "content": "Sure"},
        ]
        intent, conf, _ = self._run(clf.classify("for last 6 months", conversation_history=history))
        assert intent == "bank_statement"

    def test_llm_failure_falls_back_to_top_semantic(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=Exception("LLM down"))
        clf._get_client = MagicMock(return_value=mock_client)

        # Should not raise; falls back to semantic top match
        intent, conf, _ = self._run(clf.classify("check balance"))
        assert isinstance(intent, str)

    def test_confidence_gate_low_retrieval_caps_confidence(self, encoder):
        store = _make_store(encoder)
        clf = _make_classifier(
            mode=ClassificationMode.FLAT_RAG,
            turn_mode=TurnMode.SINGLE,
            example_store=store,
            encoder=encoder,
        )
        # LLM returns 0.99 confidence
        self._patch_llm(clf, _make_mock_llm_response("check_balance", 0.99))

        # Use a query that will have low similarity to all stored examples
        # so the confidence gate triggers
        # Patch _get_rag_examples to return low-score examples
        clf._get_rag_examples = MagicMock(
            return_value=[{"text": "xyz", "intent": "check_balance", "score": 0.10}]
        )
        _, conf, _ = self._run(clf.classify("xyzzy obscure query"))
        assert conf <= 0.55

    def test_single_turn_ignores_history(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)
        self._patch_llm(clf, _make_mock_llm_response("check_balance", 0.9))

        history = [{"role": "user", "content": "some prior turn"}]
        result = self._run(clf.classify("what is my balance", conversation_history=history))
        assert result is not None  # Should not error regardless of history

    def test_verify_pass_accepted(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)

        mock_client = MagicMock()
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: classification
                return _make_mock_llm_response("check_balance", 0.9)
            else:
                # Second call: verification
                return {
                    "message": {
                        "content": json.dumps({
                            "is_correct": True,
                            "better_intent": None,
                            "confidence_score": 0.93,
                        })
                    }
                }

        mock_client.chat = AsyncMock(side_effect=side_effect)
        clf._get_client = MagicMock(return_value=mock_client)

        intent, conf, _ = self._run(clf.classify("check balance", verify=True))
        assert intent == "check_balance"
        assert call_count == 2

    def test_verify_pass_corrects_intent(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=encoder)

        mock_client = MagicMock()
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_llm_response("wrong_intent", 0.6)
            else:
                return {
                    "message": {
                        "content": json.dumps({
                            "is_correct": False,
                            "better_intent": "check_balance",
                            "confidence_score": 0.85,
                        })
                    }
                }

        mock_client.chat = AsyncMock(side_effect=side_effect)
        clf._get_client = MagicMock(return_value=mock_client)

        intent, conf, _ = self._run(clf.classify("check balance", verify=True))
        assert intent == "check_balance"
        assert conf == 0.85

    def test_lang_detect_disabled_returns_unknown(self, encoder):
        clf = _make_classifier(mode=ClassificationMode.FLAT, encoder=encoder)
        clf.enable_lang_detect = False
        result = clf.detect_language("hello")
        assert result == "unknown"
