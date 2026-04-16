"""
Automated example store bootstrapping.

Demonstrates how ExampleStoreBootstrapper eliminates the need for manual
labeling by using the LLM itself to generate training examples.

Two paths:
  PATH A — fresh bootstrap (runs when no saved store exists)
  PATH B — load from disk (fast startup on subsequent runs)

After bootstrapping, the store is saved to disk and reused on subsequent
runs — bootstrap only needs to run once (or when new intents are added).

Requirements:
  - A running Ollama instance at http://localhost:11434
    (or set OLLAMA_BASE_URL env var to your endpoint)
  - The model configured in config.py must be pulled (e.g. llama3)

Run:
    python examples/bootstrap_store.py
"""

import asyncio
import os

from query_classifier import (
    IntentClassifier,
    ClassificationMode,
    TurnMode,
    ExampleStore,
    ExampleStoreBootstrapper,
)
from examples.banking_intents import INTENTS, INTENT_HIERARCHY


STORE_PATH = "bootstrap_example_store"   # saves bootstrap_example_store.json + .npy

TEST_QUERIES = [
    "what is my account balance",
    "I need a bank statement for last 3 months",
    "transfer 1000 dollars to my savings account",
    "block my stolen credit card",
    "I want to apply for a home loan",
    "connect me to customer support",
]


async def bootstrap_and_save():
    """Generate examples via LLM and save to disk."""
    print("No saved store found. Running bootstrap...")

    store = ExampleStore()

    bootstrapper = ExampleStoreBootstrapper(
        llm_model_name=os.getenv("BOOTSTRAP_MODEL", "llama3"),
        llm_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        max_retries=2,
        concurrency=3,
    )

    summary = await bootstrapper.run_full_bootstrap(
        store=store,
        intents=INTENTS,
        save_path=STORE_PATH,
        n_single=int(os.getenv("BOOTSTRAP_N_SINGLE", "6")),
        n_multi=int(os.getenv("BOOTSTRAP_N_MULTI", "4")),
    )

    print(f"\nBootstrap complete:")
    print(f"  Total examples  : {summary['total']}")
    print(f"  Single-turn     : {summary['single_turn_generated']}")
    print(f"  Multi-turn      : {summary['multi_turn_generated']}")
    print(f"  Intents covered : {summary['intents_covered']}")
    return store


def load_store():
    """Load a previously bootstrapped store from disk."""
    store = ExampleStore()
    store.load(STORE_PATH)
    print(f"Loaded store: {len(store)} examples from '{STORE_PATH}'")
    return store


async def run_classification(store):
    """Classify test queries using the bootstrapped store."""
    nlp = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.HIERARCHICAL_RAG,
        turn_mode=TurnMode.SINGLE,
        example_store=store,
        intent_hierarchy=INTENT_HIERARCHY,
        enable_lang_detect=False,
    )

    print("\nClassification results:")
    print("-" * 60)
    for query in TEST_QUERIES:
        intent, conf, _ = await nlp.classify(query)
        print(f"  {query!r}")
        print(f"    → {intent}  ({conf:.2f})")
    print()


async def show_coverage(store):
    """Print coverage statistics for the bootstrapped store."""
    print("\nExampleStore coverage:")
    coverage = store.intent_coverage()
    for intent_name, count in sorted(coverage.items()):
        print(f"  {intent_name:<35} {count:>3} examples")

    multi = store.multi_turn_coverage()
    print(f"\nMulti-turn coverage ({len(multi)} intents):")
    for intent_name, count in sorted(multi.items()):
        print(f"  {intent_name:<35} {count:>3} follow-up examples")


async def main():
    json_path = f"{STORE_PATH}.json"

    if os.path.exists(json_path):
        store = load_store()
    else:
        store = await bootstrap_and_save()

    await show_coverage(store)
    await run_classification(store)


if __name__ == "__main__":
    asyncio.run(main())
