"""
Tests for ExampleStoreBootstrapper.

All LLM calls are mocked — no running Ollama required.
"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from query_classifier.bootstrapper import ExampleStoreBootstrapper
from query_classifier.example_store import ExampleStore
from tests.conftest import MockEncoder, SAMPLE_INTENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bootstrapper():
    b = ExampleStoreBootstrapper(
        llm_model_name="test-model",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
        max_retries=1,
        concurrency=2,
    )
    return b


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_single_turn_response(intent_name, n=3):
    utterances = [f"example utterance {i} for {intent_name}" for i in range(n)]
    return {"message": {"content": json.dumps(utterances)}}


def _mock_multi_turn_response(intent_name, n=2):
    items = [
        {"history_context": f"I want {intent_name}", "user_followup": f"follow up {i}"}
        for i in range(n)
    ]
    return {"message": {"content": json.dumps(items)}}


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def setup_method(self):
        self.b = _make_bootstrapper()

    def test_plain_array(self):
        content = '["a", "b", "c"]'
        result = self.b._parse_json(content)
        assert result == ["a", "b", "c"]

    def test_fenced_json(self):
        content = '```json\n["x", "y"]\n```'
        result = self.b._parse_json(content)
        assert result == ["x", "y"]

    def test_fenced_no_tag(self):
        content = '```\n["p", "q"]\n```'
        result = self.b._parse_json(content)
        assert result == ["p", "q"]

    def test_array_embedded_in_text(self):
        content = 'Here are the results: ["a", "b"] end.'
        result = self.b._parse_json(content)
        assert result == ["a", "b"]

    def test_object_array(self):
        content = '[{"history_context": "hi", "user_followup": "ok"}]'
        result = self.b._parse_json(content)
        assert result[0]["history_context"] == "hi"

    def test_raises_on_invalid(self):
        with pytest.raises(Exception):
            self.b._parse_json("not json at all {}")


# ---------------------------------------------------------------------------
# _generate_single_turn
# ---------------------------------------------------------------------------

class TestGenerateSingleTurn:
    def setup_method(self):
        self.b = _make_bootstrapper()

    def _patch_llm(self, response_content):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": response_content}})
        self.b._get_client = MagicMock(return_value=mock_client)

    def test_returns_list_of_dicts(self):
        self._patch_llm(json.dumps(["what is my balance", "check my account"]))
        intent = SAMPLE_INTENTS[0]  # check_balance
        results = _run(self.b._generate_single_turn(intent, n=2))
        assert isinstance(results, list)
        assert len(results) == 2

    def test_result_structure(self):
        self._patch_llm(json.dumps(["check my balance"]))
        intent = SAMPLE_INTENTS[0]
        results = _run(self.b._generate_single_turn(intent, n=1))
        assert results[0]["text"] == "check my balance"
        assert results[0]["intent"] == intent["name"]
        assert "history_context" not in results[0]

    def test_empty_strings_filtered(self):
        self._patch_llm(json.dumps(["valid utterance", "", "  "]))
        intent = SAMPLE_INTENTS[0]
        results = _run(self.b._generate_single_turn(intent, n=3))
        assert len(results) == 1

    def test_llm_failure_returns_empty_after_retries(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=Exception("LLM error"))
        self.b._get_client = MagicMock(return_value=mock_client)
        intent = SAMPLE_INTENTS[0]
        results = _run(self.b._generate_single_turn(intent, n=3))
        assert results == []

    def test_fenced_json_parsed_correctly(self):
        self._patch_llm('```json\n["block my card", "freeze card"]\n```')
        intent = SAMPLE_INTENTS[3]  # block_card
        results = _run(self.b._generate_single_turn(intent, n=2))
        assert len(results) == 2
        assert all(r["intent"] == "block_card" for r in results)


# ---------------------------------------------------------------------------
# _generate_multi_turn
# ---------------------------------------------------------------------------

class TestGenerateMultiTurn:
    def setup_method(self):
        self.b = _make_bootstrapper()
        self.encoder = MockEncoder()

    def _make_store_with_examples(self):
        store = ExampleStore(encoder=self.encoder)
        store.add_examples_bulk([
            {"text": "what is my balance", "intent": "check_balance"},
            {"text": "check my account", "intent": "check_balance"},
        ])
        return store

    def _patch_llm(self, response_content):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": response_content}})
        self.b._get_client = MagicMock(return_value=mock_client)

    def test_returns_list_with_history_context(self):
        items = [{"history_context": "I need my balance", "user_followup": "the savings one"}]
        self._patch_llm(json.dumps(items))
        intent = SAMPLE_INTENTS[0]
        store = self._make_store_with_examples()
        results = _run(self.b._generate_multi_turn(intent, store, n=1))
        assert len(results) == 1
        assert results[0]["history_context"] == "I need my balance"
        assert results[0]["text"] == "the savings one"
        assert results[0]["intent"] == intent["name"]

    def test_missing_fields_filtered(self):
        items = [
            {"history_context": "valid", "user_followup": "valid followup"},
            {"history_context": "", "user_followup": "no context"},
            {"history_context": "no followup", "user_followup": ""},
        ]
        self._patch_llm(json.dumps(items))
        intent = SAMPLE_INTENTS[0]
        store = self._make_store_with_examples()
        results = _run(self.b._generate_multi_turn(intent, store, n=3))
        assert len(results) == 1

    def test_works_with_empty_store(self):
        items = [{"history_context": "I need a statement", "user_followup": "last 3 months"}]
        self._patch_llm(json.dumps(items))
        intent = SAMPLE_INTENTS[1]  # bank_statement
        store = ExampleStore(encoder=self.encoder)
        results = _run(self.b._generate_multi_turn(intent, store, n=1))
        assert len(results) == 1

    def test_llm_failure_returns_empty(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=Exception("error"))
        self.b._get_client = MagicMock(return_value=mock_client)
        intent = SAMPLE_INTENTS[0]
        store = self._make_store_with_examples()
        results = _run(self.b._generate_multi_turn(intent, store, n=2))
        assert results == []


# ---------------------------------------------------------------------------
# bootstrap_single_turn
# ---------------------------------------------------------------------------

class TestBootstrapSingleTurn:
    def setup_method(self):
        self.b = _make_bootstrapper()
        self.encoder = MockEncoder()

    def _patch_llm_for_all(self, n_per_intent=2):
        """LLM always returns n utterances per call."""
        async def side_effect(**kwargs):
            # Extract intent name from prompt to make unique utterances
            prompt = kwargs["messages"][0]["content"]
            for intent in SAMPLE_INTENTS:
                if intent["name"] in prompt:
                    return {"message": {"content": json.dumps(
                        [f"{intent['name']} example {i}" for i in range(n_per_intent)]
                    )}}
            return {"message": {"content": "[]"}}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=side_effect)
        self.b._get_client = MagicMock(return_value=mock_client)

    def test_adds_examples_to_store(self):
        self._patch_llm_for_all(n_per_intent=2)
        store = ExampleStore(encoder=self.encoder)
        _run(self.b.bootstrap_single_turn(store, SAMPLE_INTENTS, n_per_intent=2))
        assert len(store) > 0

    def test_returns_count(self):
        self._patch_llm_for_all(n_per_intent=2)
        store = ExampleStore(encoder=self.encoder)
        n = _run(self.b.bootstrap_single_turn(store, SAMPLE_INTENTS, n_per_intent=2))
        assert n == len(store)

    def test_all_intents_get_examples(self):
        self._patch_llm_for_all(n_per_intent=2)
        store = ExampleStore(encoder=self.encoder)
        _run(self.b.bootstrap_single_turn(store, SAMPLE_INTENTS, n_per_intent=2))
        coverage = store.intent_coverage()
        for intent in SAMPLE_INTENTS:
            assert intent["name"] in coverage


# ---------------------------------------------------------------------------
# run_full_bootstrap
# ---------------------------------------------------------------------------

class TestRunFullBootstrap:
    def setup_method(self):
        self.b = _make_bootstrapper()
        self.encoder = MockEncoder()

    def test_summary_keys(self):
        async def mock_single(store, intents, n_per_intent):
            examples = [{"text": f"ex {i}", "intent": intents[0]["name"]} for i in range(3)]
            store.add_examples_bulk(examples)
            return 3

        async def mock_multi(store, intents, n_per_intent):
            examples = [{
                "text": "follow up",
                "intent": intents[0]["name"],
                "history_context": "initial turn",
            }]
            store.add_examples_bulk(examples)
            return 1

        self.b.bootstrap_single_turn = mock_single
        self.b.bootstrap_multi_turn = mock_multi

        store = ExampleStore(encoder=self.encoder)
        summary = _run(self.b.run_full_bootstrap(
            store=store,
            intents=SAMPLE_INTENTS[:1],
            save_path=None,
            n_single=3,
            n_multi=1,
        ))

        assert "total" in summary
        assert "single_turn_generated" in summary
        assert "multi_turn_generated" in summary
        assert "intents_covered" in summary
        assert "multi_turn_coverage" in summary
        assert summary["single_turn_generated"] == 3
        assert summary["multi_turn_generated"] == 1

    def test_save_called_when_path_provided(self, tmp_path):
        async def mock_single(store, intents, n_per_intent):
            return 0

        async def mock_multi(store, intents, n_per_intent):
            return 0

        self.b.bootstrap_single_turn = mock_single
        self.b.bootstrap_multi_turn = mock_multi

        store = ExampleStore(encoder=self.encoder)
        # Add at least one example so save works
        store.add_example("test", "check_balance")

        save_path = str(tmp_path / "bootstrap_store")
        _run(self.b.run_full_bootstrap(
            store=store,
            intents=SAMPLE_INTENTS[:1],
            save_path=save_path,
            n_single=0,
            n_multi=0,
        ))

        import os
        assert os.path.exists(f"{save_path}.json")
