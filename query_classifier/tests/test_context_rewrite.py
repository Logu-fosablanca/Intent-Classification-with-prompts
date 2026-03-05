import asyncio
from query_classifier.nlp_engine import IntentClassifier


def _build_classifier_without_init():
    classifier = IntentClassifier.__new__(IntentClassifier)
    classifier.llm_model_name = "dummy"
    return classifier


def test_rewrite_for_vague_followup_uses_history_and_intent():
    classifier = _build_classifier_without_init()
    history = [
        {"role": "user", "content": "I need my account statement"},
        {"role": "assistant", "content": "Sure, for which period?", "intent_classified": "account_statement"},
    ]

    rewritten = classifier._rewrite_query_for_semantic_search("for last 6 months", history)

    assert "Previous user request: I need my account statement" in rewritten
    assert "Current active intent: account_statement" in rewritten
    assert rewritten.endswith("Follow-up: for last 6 months")


def test_rewrite_keeps_standalone_query_unchanged():
    classifier = _build_classifier_without_init()
    history = [
        {"role": "user", "content": "I need my account statement"},
        {"role": "assistant", "content": "Sure, for which period?", "intent_classified": "account_statement"},
    ]

    original = "I want to apply for a new credit card"
    rewritten = classifier._rewrite_query_for_semantic_search(original, history)

    assert rewritten == original


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    async def chat(self, model, messages):
        return {"message": {"content": self.payload}}


def test_llm_rewrite_query_uses_rewritten_text_when_requested():
    classifier = _build_classifier_without_init()
    history = [{"role": "user", "content": "I want statement"}]
    client = _FakeClient('{"needs_rewrite": true, "rewritten_query": "Get account statement for last 6 months"}')

    rewritten = asyncio.run(classifier._rewrite_query_with_llm(client, "for last 6 months", history))

    assert rewritten == "Get account statement for last 6 months"


def test_llm_rewrite_query_falls_back_on_invalid_response():
    classifier = _build_classifier_without_init()
    history = [
        {"role": "user", "content": "I need my account statement"},
        {"role": "assistant", "intent_classified": "account_statement"},
    ]
    client = _FakeClient('not json')

    rewritten = asyncio.run(classifier._rewrite_query_with_llm(client, "for last 6 months", history))

    assert "account_statement" in rewritten
    assert rewritten.endswith("Follow-up: for last 6 months")
