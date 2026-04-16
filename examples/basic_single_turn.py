"""
Basic single-turn intent classification.

Demonstrates the simplest possible setup:
  - FLAT mode: SemanticRouter → LLM (no RAG, no hierarchy)
  - FLAT_RAG mode: SemanticRouter → ExampleStore → LLM

Each call to classify() is fully independent — no conversation history.

Run:
    python examples/basic_single_turn.py
"""

import asyncio

from query_classifier import IntentClassifier, ClassificationMode, TurnMode, ExampleStore
from examples.banking_intents import INTENTS


# ---------------------------------------------------------------------------
# Seed examples (minimal — a few per intent for FLAT_RAG)
# ---------------------------------------------------------------------------

SEED_EXAMPLES = [
    {"text": "what is my account balance",        "intent": "check_balance"},
    {"text": "how much money do I have",           "intent": "check_balance"},
    {"text": "I need my bank statement",           "intent": "bank_statement"},
    {"text": "download my statement last month",   "intent": "bank_statement"},
    {"text": "transfer money to my savings",       "intent": "transfer_money"},
    {"text": "send 500 to john",                   "intent": "transfer_money"},
    {"text": "block my lost card",                 "intent": "block_card"},
    {"text": "freeze my credit card",              "intent": "block_card"},
    {"text": "apply for a personal loan",          "intent": "apply_loan"},
    {"text": "I want to take out a home loan",     "intent": "apply_loan"},
    {"text": "speak to a human agent",             "intent": "contact_support"},
    {"text": "connect me to customer support",     "intent": "contact_support"},
]

TEST_QUERIES = [
    "how much is in my account",
    "I want a copy of my statement",
    "can I send money to another bank",
    "my card got stolen, block it",
    "can I get a loan for my business",
    "I need to talk to someone",
]


async def run_flat():
    """FLAT mode — pure semantic routing, no examples needed."""
    print("\n" + "=" * 60)
    print("FLAT mode (SemanticRouter → LLM)")
    print("=" * 60)

    nlp = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.FLAT,
        turn_mode=TurnMode.SINGLE,
        enable_lang_detect=False,
    )

    for query in TEST_QUERIES:
        intent, conf, lang = await nlp.classify(query)
        print(f"  Query : {query!r}")
        print(f"  Intent: {intent}  (confidence: {conf:.2f})")
        print()


async def run_flat_rag():
    """FLAT_RAG mode — adds few-shot evidence from ExampleStore."""
    print("\n" + "=" * 60)
    print("FLAT_RAG mode (SemanticRouter → ExampleStore → LLM)")
    print("=" * 60)

    store = ExampleStore()
    store.add_examples_bulk(SEED_EXAMPLES)
    print(f"  ExampleStore: {len(store)} examples loaded.\n")

    nlp = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.FLAT_RAG,
        turn_mode=TurnMode.SINGLE,
        example_store=store,
        enable_lang_detect=False,
    )

    for query in TEST_QUERIES:
        intent, conf, lang = await nlp.classify(query)
        print(f"  Query : {query!r}")
        print(f"  Intent: {intent}  (confidence: {conf:.2f})")
        print()


async def main():
    await run_flat()
    await run_flat_rag()


if __name__ == "__main__":
    asyncio.run(main())
