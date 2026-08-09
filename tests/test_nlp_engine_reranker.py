"""
Tests for IntentClassifier.classify() reranker wiring — the branch added to
step 7-8 that replaces the generative LLM call with QwenReranker.rerank()
when self.reranker is configured.

Uses `async def test_...` directly (pytest-asyncio, asyncio_mode="auto" in
pyproject.toml) rather than test_nlp_engine.py's `_run()` helper, which
depends on `asyncio.get_event_loop()` — deprecated/removed as an implicit
loop-creator on this environment's Python version, independent of this change.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from query_classifier.nlp_engine import ClassificationMode, TurnMode
from tests.test_nlp_engine import _make_classifier, _make_mock_llm_response
from tests.conftest import MockEncoder


def _mock_reranker(return_value):
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=return_value)
    return reranker


class TestRerankerWiring:
    @pytest.mark.asyncio
    async def test_reranker_result_used_when_configured(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=MockEncoder())
        clf.rerank_mode = "listwise"
        clf.reranker = _mock_reranker(("check_balance", 0.87, {"ranking": []}))

        # The generative client must never be called on this path.
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=AssertionError("generative LLM should not be called"))
        clf._get_client = MagicMock(return_value=mock_client)

        intent, conf, _ = await clf.classify("what is my balance")
        assert intent == "check_balance"
        assert conf == 0.87
        clf.reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reranker_receives_configured_mode(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=MockEncoder())
        clf.rerank_mode = "pointwise"
        clf.reranker = _mock_reranker(("check_balance", 0.6, {"scores": {}}))
        clf._get_client = MagicMock(return_value=MagicMock(chat=AsyncMock()))

        await clf.classify("what is my balance")
        call_args = clf.reranker.rerank.call_args
        assert call_args.args[0] == "pointwise"

    @pytest.mark.asyncio
    async def test_reranker_failure_falls_back_to_top_semantic(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=MockEncoder())
        clf.rerank_mode = "listwise"
        reranker = MagicMock()
        reranker.rerank = AsyncMock(side_effect=RuntimeError("reranker down"))
        clf.reranker = reranker
        clf._get_client = MagicMock(return_value=MagicMock(chat=AsyncMock()))

        intent, conf, _ = await clf.classify("what is my balance")
        assert isinstance(intent, str)  # falls back to top_matches[0], no crash

    @pytest.mark.asyncio
    async def test_confidence_blend_and_gate_apply_to_reranker_path(self):
        from query_classifier.example_store import ExampleStore
        from tests.conftest import SAMPLE_EXAMPLES, MULTI_TURN_EXAMPLES

        encoder = MockEncoder()
        store = ExampleStore(encoder=encoder)
        store.add_examples_bulk(SAMPLE_EXAMPLES)
        store.add_examples_bulk(MULTI_TURN_EXAMPLES)

        clf = _make_classifier(
            mode=ClassificationMode.FLAT_RAG,
            turn_mode=TurnMode.SINGLE,
            example_store=store,
            encoder=encoder,
        )
        clf.rerank_mode = "listwise"
        clf.reranker = _mock_reranker(("check_balance", 0.99, {}))
        clf._get_client = MagicMock(return_value=MagicMock(chat=AsyncMock()))

        # Force a known low retrieval score so the confidence gate triggers,
        # exactly like the existing generative-path test does.
        clf._get_rag_examples = MagicMock(
            return_value=[{"text": "xyz", "intent": "check_balance", "score": 0.10}]
        )

        _, conf, _ = await clf.classify("obscure query")
        assert conf <= 0.55

    @pytest.mark.asyncio
    async def test_verify_pass_still_uses_generative_client_in_reranker_mode(self):
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=MockEncoder())
        clf.rerank_mode = "listwise"
        clf.reranker = _mock_reranker(("check_balance", 0.8, {}))

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"content": '{"is_correct": true, "better_intent": null, "confidence_score": 0.9}'}
        })
        clf._get_client = MagicMock(return_value=mock_client)

        intent, conf, _ = await clf.classify("what is my balance", verify=True)
        assert intent == "check_balance"
        assert conf == 0.9
        mock_client.chat.assert_awaited_once()  # only the verify call, not classification

    @pytest.mark.asyncio
    async def test_generative_path_unaffected_when_reranker_none(self):
        """Regression guard: default (reranker=None) still uses the original path."""
        clf = _make_classifier(mode=ClassificationMode.FLAT, turn_mode=TurnMode.SINGLE, encoder=MockEncoder())
        assert clf.reranker is None

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_mock_llm_response("check_balance", 0.95))
        clf._get_client = MagicMock(return_value=mock_client)

        intent, conf, _ = await clf.classify("what is my balance")
        assert intent == "check_balance"
        assert conf == 0.95
