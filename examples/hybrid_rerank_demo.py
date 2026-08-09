"""
Hybrid Retrieval + LLM Reranker Demo.

Compares the original generative classification call against the two
reranker strategies (listwise, pointwise) over the same hybrid-retrieved
(FAISS dense + BM25 lexical) candidates.

Requires:
  pip install query-classifier[rerank]   # faiss-cpu, rank-bm25, ollama
  ollama pull qwen3:4b                   # or set RERANKER_MODEL_NAME

Hybrid retrieval and reranking are both OFF by default (USE_HYBRID_RETRIEVAL=
false, RERANK_MODE=off) — this demo turns them on explicitly so existing
IntentClassifier usage elsewhere is unaffected unless opted in.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_classifier import IntentClassifier, ClassificationMode, TurnMode, ExampleStore
from examples.banking_intents import INTENTS

SEED_EXAMPLES = [
    {"text": "What is my current balance?", "intent": "account_check_balance"},
    {"text": "How much money do I have in my account?", "intent": "account_check_balance"},
    {"text": "Check my savings balance", "intent": "account_check_balance"},
    {"text": "I need my bank statement for last month", "intent": "account_statement_request"},
    {"text": "Can you send me a PDF statement?", "intent": "account_statement_request"},
    {"text": "Block my credit card immediately", "intent": "card_block"},
    {"text": "I lost my card, freeze it now", "intent": "card_block"},
    {"text": "My card was stolen ending in 4521, please deactivate it", "intent": "card_block"},
    {"text": "I'd like to apply for a personal loan", "intent": "loan_apply"},
    {"text": "Can I get a loan for $10,000?", "intent": "loan_apply"},
]


async def run_demo():
    print("=" * 70)
    print("  Hybrid Retrieval (FAISS + BM25) + LLM Reranker Demo")
    print("=" * 70)

    # ExampleStore with hybrid FAISS+BM25 retrieval enabled — the exact card
    # digits in the query below ("4521") are the kind of token dense
    # embeddings alone tend to blur but BM25 catches directly.
    store = ExampleStore(use_hybrid=True)
    store.add_examples_bulk(SEED_EXAMPLES)
    print(f"\n[Setup] ExampleStore ready: {len(store)} examples, hybrid={store.use_hybrid}")

    configs = [
        ("Generative (baseline)", dict(rerank_mode="off")),
        ("Listwise reranker",     dict(rerank_mode="listwise", reranker_model_name="qwen3:4b")),
        ("Pointwise reranker",    dict(rerank_mode="pointwise", reranker_model_name="qwen3:4b")),
    ]

    classifiers = {
        label: IntentClassifier(
            intents=INTENTS,
            mode=ClassificationMode.FLAT_RAG,
            turn_mode=TurnMode.SINGLE,
            llm_model_name="llama3",
            example_store=store,
            enable_lang_detect=False,
            **kwargs,
        )
        for label, kwargs in configs
    }
    print("  Classifiers ready:", ", ".join(classifiers))

    test_queries = [
        "How much money is left in my account?",
        "My card ending in 4521 was stolen, freeze it",
        "I want to buy a house, what loans do you have?",
    ]

    print("\n" + "-" * 70)
    for query in test_queries:
        print(f"\nQuery: {query!r}")
        for label, clf in classifiers.items():
            intent, conf, _ = await clf.classify(query)
            print(f"  {label:<24} -> {intent} (conf={conf:.2f})")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
