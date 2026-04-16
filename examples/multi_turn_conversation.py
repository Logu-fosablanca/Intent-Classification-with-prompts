"""
Multi-turn conversation demo.

Shows how TurnMode.MULTI makes the classifier context-aware:
  - Prior categories are pinned so follow-ups stay in the right domain.
  - RAG query is enriched with recent user utterances so bare follow-ups
    like "for last 6 months" retrieve meaningful examples.

Three full conversations are simulated end-to-end.

Run:
    python examples/multi_turn_conversation.py
"""

import asyncio

from query_classifier import (
    IntentClassifier,
    ClassificationMode,
    TurnMode,
    ExampleStore,
)
from examples.banking_intents import INTENTS, INTENT_HIERARCHY


# ---------------------------------------------------------------------------
# Seed examples — includes multi-turn follow-ups
# ---------------------------------------------------------------------------

SEED_EXAMPLES = [
    # Single-turn
    {"text": "what is my account balance",        "intent": "check_balance"},
    {"text": "how much money do I have",           "intent": "check_balance"},
    {"text": "I need my bank statement",           "intent": "bank_statement"},
    {"text": "download my statement for last month", "intent": "bank_statement"},
    {"text": "block my credit card",               "intent": "block_card"},
    {"text": "freeze my card",                     "intent": "block_card"},
    {"text": "transfer money to savings",          "intent": "transfer_money"},
    {"text": "send 500 dollars to john",           "intent": "transfer_money"},
    # Multi-turn — history_context is the user's opening message
    {
        "text": "for last 6 months",
        "intent": "bank_statement",
        "history_context": "I need my bank statement",
    },
    {
        "text": "yes block it now",
        "intent": "block_card",
        "history_context": "my card was stolen",
    },
    {
        "text": "to my savings account",
        "intent": "transfer_money",
        "history_context": "I want to transfer some money",
    },
]


def _format_history(history):
    lines = []
    for turn in history:
        role = turn["role"].upper()
        content = turn.get("content", "")
        intent = turn.get("intent_classified", "")
        if intent:
            content += f"  [intent: {intent}]"
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


async def simulate_conversation(nlp, turns, title):
    """
    Simulate a multi-turn conversation.

    `turns` is a list of user utterances. History is maintained automatically.
    """
    print("\n" + "=" * 60)
    print(f"Conversation: {title}")
    print("=" * 60)

    history = []

    for user_text in turns:
        print(f"\n  USER: {user_text!r}")

        intent, conf, lang = await nlp.classify(
            user_text,
            conversation_history=history,
        )

        print(f"  CLASSIFIED → intent='{intent}'  confidence={conf:.2f}")

        # Append user turn and assistant turn to history
        history.append({"role": "user", "content": user_text})
        history.append({
            "role": "assistant",
            "content": f"Understood — routing to {intent}.",
            "intent_classified": intent,
        })

    print()
    print("  Full history:")
    print(_format_history(history))


async def main():
    store = ExampleStore()
    store.add_examples_bulk(SEED_EXAMPLES)
    print(f"ExampleStore: {len(store)} examples ({store.multi_turn_coverage()})")

    nlp = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.HIERARCHICAL_RAG,
        turn_mode=TurnMode.MULTI,
        example_store=store,
        intent_hierarchy=INTENT_HIERARCHY,
        enable_lang_detect=False,
    )

    # Conversation 1: bank statement follow-up
    await simulate_conversation(
        nlp,
        turns=[
            "I need my bank statement",
            "for last 6 months",         # ambiguous bare follow-up — should stay bank_statement
        ],
        title="Bank statement with follow-up",
    )

    # Conversation 2: card blocking with confirmation
    await simulate_conversation(
        nlp,
        turns=[
            "my card was stolen",
            "yes block it now",           # short confirmation — should stay block_card
        ],
        title="Card blocking with confirmation",
    )

    # Conversation 3: topic change mid-conversation
    await simulate_conversation(
        nlp,
        turns=[
            "what is my balance",
            "I also need to transfer money",   # topic change — should switch intent
            "500 dollars to savings",
        ],
        title="Balance check then transfer",
    )


if __name__ == "__main__":
    asyncio.run(main())
