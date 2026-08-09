"""
Tests for QwenReranker — listwise and pointwise LLM reranking.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from query_classifier.reranker import QwenReranker, _extract_json
from tests.conftest import SAMPLE_INTENTS


def _candidates(names):
    by_name = {i["name"]: i for i in SAMPLE_INTENTS}
    return [{"intent": by_name[n], "score": 0.5} for n in names]


def _rag_examples():
    return [
        {"text": "what is my balance", "intent": "check_balance", "score": 0.9},
        {"text": "I need my bank statement", "intent": "bank_statement", "score": 0.8},
    ]


def _patch_chat(reranker, content=None, side_effect=None):
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.chat = AsyncMock(side_effect=side_effect)
    else:
        mock_client.chat = AsyncMock(return_value={"message": {"content": content}})
    reranker._get_client = MagicMock(return_value=mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"relevance": 0.8}')
        assert result["relevance"] == 0.8

    def test_fenced_json(self):
        result = _extract_json('```json\n{"relevance": 0.5}\n```')
        assert result["relevance"] == 0.5

    def test_strips_think_block(self):
        content = "<think>reasoning about the query...</think>\n{\"relevance\": 0.7}"
        result = _extract_json(content)
        assert result["relevance"] == 0.7

    def test_embedded_json_in_text(self):
        content = 'The answer is: {"relevance": 0.6} thanks.'
        result = _extract_json(content)
        assert result["relevance"] == 0.6

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            _extract_json("no json here")


# ---------------------------------------------------------------------------
# Listwise
# ---------------------------------------------------------------------------

class TestRerankListwise:
    @pytest.mark.asyncio
    async def test_picks_highest_relevance(self):
        reranker = QwenReranker(model_name="qwen3:4b")
        content = json.dumps({
            "ranking": [
                {"name": "check_balance", "relevance": 0.4},
                {"name": "bank_statement", "relevance": 0.9},
            ]
        })
        _patch_chat(reranker, content=content)

        candidates = _candidates(["check_balance", "bank_statement"])
        name, conf, raw = await reranker.rerank_listwise(
            "I need my statement", candidates, _rag_examples()
        )
        assert name == "bank_statement"
        assert conf == 0.9
        assert "ranking" in raw

    @pytest.mark.asyncio
    async def test_raises_on_no_candidates(self):
        reranker = QwenReranker()
        with pytest.raises(ValueError, match="no candidates"):
            await reranker.rerank_listwise("query", [], [])

    @pytest.mark.asyncio
    async def test_raises_on_empty_ranking(self):
        reranker = QwenReranker()
        _patch_chat(reranker, content=json.dumps({"ranking": []}))
        candidates = _candidates(["check_balance"])
        with pytest.raises(ValueError, match="empty ranking"):
            await reranker.rerank_listwise("query", candidates, [])

    @pytest.mark.asyncio
    async def test_filters_hallucinated_names(self):
        """If the model invents a name outside the candidate set, prefer
        entries that are actually valid candidates."""
        reranker = QwenReranker()
        content = json.dumps({
            "ranking": [
                {"name": "not_a_real_intent", "relevance": 0.99},
                {"name": "check_balance", "relevance": 0.6},
            ]
        })
        _patch_chat(reranker, content=content)
        candidates = _candidates(["check_balance"])
        name, conf, _ = await reranker.rerank_listwise("query", candidates, [])
        assert name == "check_balance"

    @pytest.mark.asyncio
    async def test_strips_think_block_from_response(self):
        reranker = QwenReranker()
        content = (
            "<think>let me consider the options</think>\n"
            + json.dumps({"ranking": [{"name": "check_balance", "relevance": 0.7}]})
        )
        _patch_chat(reranker, content=content)
        candidates = _candidates(["check_balance"])
        name, conf, _ = await reranker.rerank_listwise("query", candidates, [])
        assert name == "check_balance"
        assert conf == 0.7


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------

class TestRerankPointwise:
    @pytest.mark.asyncio
    async def test_picks_highest_scored_candidate(self):
        reranker = QwenReranker()
        candidates = _candidates(["check_balance", "bank_statement"])

        async def fake_chat(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            if "check_balance" in prompt:
                return {"message": {"content": '{"relevance": 0.3}'}}
            return {"message": {"content": '{"relevance": 0.85}'}}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        reranker._get_client = MagicMock(return_value=mock_client)

        name, conf, raw = await reranker.rerank_pointwise(
            "I need my statement", candidates, _rag_examples()
        )
        assert name == "bank_statement"
        assert conf == 0.85
        assert raw["scores"]["check_balance"] == 0.3
        assert raw["scores"]["bank_statement"] == 0.85

    @pytest.mark.asyncio
    async def test_raises_on_no_candidates(self):
        reranker = QwenReranker()
        with pytest.raises(ValueError, match="no candidates"):
            await reranker.rerank_pointwise("query", [], [])

    @pytest.mark.asyncio
    async def test_failed_call_scores_zero_not_raises(self):
        reranker = QwenReranker()
        candidates = _candidates(["check_balance", "bank_statement"])

        async def fake_chat(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            if "check_balance" in prompt:
                raise RuntimeError("LLM unreachable")
            return {"message": {"content": '{"relevance": 0.4}'}}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        reranker._get_client = MagicMock(return_value=mock_client)

        name, conf, raw = await reranker.rerank_pointwise("query", candidates, [])
        assert raw["scores"]["check_balance"] == 0.0
        assert name == "bank_statement"

    @pytest.mark.asyncio
    async def test_score_clamped_to_0_1_range(self):
        reranker = QwenReranker()
        _patch_chat(reranker, content='{"relevance": 1.5}')
        candidates = _candidates(["check_balance"])
        name, conf, _ = await reranker.rerank_pointwise("query", candidates, [])
        assert conf == 1.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestRerankDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_to_listwise(self):
        reranker = QwenReranker()
        _patch_chat(reranker, content=json.dumps(
            {"ranking": [{"name": "check_balance", "relevance": 0.9}]}
        ))
        candidates = _candidates(["check_balance"])
        name, conf, _ = await reranker.rerank("listwise", "query", candidates, [])
        assert name == "check_balance"

    @pytest.mark.asyncio
    async def test_dispatches_to_pointwise(self):
        reranker = QwenReranker()
        _patch_chat(reranker, content='{"relevance": 0.6}')
        candidates = _candidates(["check_balance"])
        name, conf, _ = await reranker.rerank("pointwise", "query", candidates, [])
        assert name == "check_balance"
        assert conf == 0.6

    @pytest.mark.asyncio
    async def test_unknown_mode_raises(self):
        reranker = QwenReranker()
        with pytest.raises(ValueError, match="Unknown rerank mode"):
            await reranker.rerank("bogus", "query", _candidates(["check_balance"]), [])
