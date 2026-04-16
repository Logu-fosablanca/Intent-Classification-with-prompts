"""
Custom intents — shows how to plug in your own intent set and hierarchy.

This example uses a small e-commerce intent set (not banking) to show
that the library is domain-agnostic.

Demonstrates:
  - Defining intents and hierarchy for a new domain
  - All three ClassificationModes side-by-side
  - Fail-fast validation (HIERARCHICAL_RAG without hierarchy)
  - Single-turn vs multi-turn mode

Run:
    python examples/custom_intents.py
"""

import asyncio

from query_classifier import (
    IntentClassifier,
    ClassificationMode,
    TurnMode,
    ExampleStore,
)


# ---------------------------------------------------------------------------
# Custom domain: e-commerce support bot
# ---------------------------------------------------------------------------

ECOMMERCE_INTENTS = [
    {"name": "track_order",    "description": "Customer wants to track the status of their order."},
    {"name": "return_item",    "description": "Customer wants to return or exchange a purchased item."},
    {"name": "cancel_order",   "description": "Customer wants to cancel an order they placed."},
    {"name": "payment_issue",  "description": "Customer has a problem with a payment or charge."},
    {"name": "product_info",   "description": "Customer is asking for information about a product."},
    {"name": "contact_agent",  "description": "Customer wants to speak to a human support agent."},
]

ECOMMERCE_HIERARCHY = {
    "orders": {
        "description": "Managing orders: tracking, cancelling, modifying.",
        "intents": ["track_order", "cancel_order"],
    },
    "returns": {
        "description": "Product returns and exchanges.",
        "intents": ["return_item"],
    },
    "payments": {
        "description": "Payment problems, charges, refunds.",
        "intents": ["payment_issue"],
    },
    "products": {
        "description": "Product questions, availability, specifications.",
        "intents": ["product_info"],
    },
    "support": {
        "description": "Human agent escalation and general support.",
        "intents": ["contact_agent"],
    },
}

ECOMMERCE_EXAMPLES = [
    {"text": "where is my package",             "intent": "track_order"},
    {"text": "I want to return this shirt",     "intent": "return_item"},
    {"text": "please cancel my order",          "intent": "cancel_order"},
    {"text": "I was charged twice",             "intent": "payment_issue"},
    {"text": "does this come in blue",          "intent": "product_info"},
    {"text": "connect me to an agent",          "intent": "contact_agent"},
    # Multi-turn
    {
        "text": "order number 12345",
        "intent": "track_order",
        "history_context": "I want to track my order",
    },
    {
        "text": "the red jacket I bought last week",
        "intent": "return_item",
        "history_context": "I want to return an item",
    },
]

TEST_QUERIES = [
    "has my order shipped yet",
    "I'd like to exchange this for a different size",
    "please cancel what I just ordered",
    "there's an extra charge on my card",
    "what are the dimensions of this product",
]


async def compare_modes():
    """Run all three modes on the same queries and compare outputs."""
    store = ExampleStore()
    store.add_examples_bulk(ECOMMERCE_EXAMPLES)

    flat = IntentClassifier(
        intents=ECOMMERCE_INTENTS,
        mode=ClassificationMode.FLAT,
        turn_mode=TurnMode.SINGLE,
        enable_lang_detect=False,
    )
    flat_rag = IntentClassifier(
        intents=ECOMMERCE_INTENTS,
        mode=ClassificationMode.FLAT_RAG,
        turn_mode=TurnMode.SINGLE,
        example_store=store,
        enable_lang_detect=False,
    )
    hier_rag = IntentClassifier(
        intents=ECOMMERCE_INTENTS,
        mode=ClassificationMode.HIERARCHICAL_RAG,
        turn_mode=TurnMode.MULTI,
        example_store=store,
        intent_hierarchy=ECOMMERCE_HIERARCHY,
        enable_lang_detect=False,
    )

    print("\n" + "=" * 70)
    print("Mode comparison — E-commerce domain")
    print("=" * 70)
    print(f"{'Query':<45} {'FLAT':<20} {'FLAT_RAG':<20} {'HIER_RAG'}")
    print("-" * 110)

    for query in TEST_QUERIES:
        r1 = await flat.classify(query)
        r2 = await flat_rag.classify(query)
        r3 = await hier_rag.classify(query)
        print(f"{query:<45} {r1[0]:<20} {r2[0]:<20} {r3[0]}")


async def demo_fail_fast():
    """Show that HIERARCHICAL_RAG without a hierarchy raises immediately."""
    print("\n" + "=" * 70)
    print("Fail-fast validation demo")
    print("=" * 70)
    try:
        IntentClassifier(
            intents=ECOMMERCE_INTENTS,
            mode=ClassificationMode.HIERARCHICAL_RAG,
            # intent_hierarchy intentionally omitted
            enable_lang_detect=False,
        )
        print("  ERROR: should have raised!")
    except ValueError as e:
        print(f"  ValueError raised correctly:\n  {str(e)[:120]}...")


async def demo_multi_turn():
    """Show multi-turn mode in action for e-commerce."""
    print("\n" + "=" * 70)
    print("Multi-turn conversation (e-commerce)")
    print("=" * 70)

    store = ExampleStore()
    store.add_examples_bulk(ECOMMERCE_EXAMPLES)

    nlp = IntentClassifier(
        intents=ECOMMERCE_INTENTS,
        mode=ClassificationMode.HIERARCHICAL_RAG,
        turn_mode=TurnMode.MULTI,
        example_store=store,
        intent_hierarchy=ECOMMERCE_HIERARCHY,
        enable_lang_detect=False,
    )

    history = []
    turns = [
        "I want to track my order",
        "order number 12345",           # follow-up should stay track_order
    ]

    for user_text in turns:
        print(f"\n  USER: {user_text!r}")
        intent, conf, _ = await nlp.classify(user_text, conversation_history=history)
        print(f"  → {intent}  ({conf:.2f})")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": "...", "intent_classified": intent})


async def main():
    await compare_modes()
    await demo_fail_fast()
    await demo_multi_turn()


if __name__ == "__main__":
    asyncio.run(main())
