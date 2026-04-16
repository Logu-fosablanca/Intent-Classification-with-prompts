# Intent Classifier — REIC Implementation

A production-grade, configurable intent classification library implementing learnings from the **REIC: RAG-Enhanced Intent Classification at Scale** paper (Amazon, 2024).

Combines **Semantic Routing** (SentenceTransformers), **Retrieval-Augmented Generation (RAG)** for few-shot evidence, and **Hierarchical Routing** (coarse category → fine intent) with full **multi-turn conversation** support.

---

## Architecture

```
User Query
    │
    ▼
SemanticRouter ──── encode_query() ────► shared embedding
    │                                           │
    ▼                                           ▼
[FLAT]  Top-K intents              ExampleStore.retrieve()
    │   (flat cosine sim)          (context-enriched RAG query)
    │                                           │
[HIERARCHICAL_RAG]                             │
    │                                           │
    ▼                                           ▼
HierarchicalRouter                     Few-shot examples
  Coarse: category embeddings          (filtered by category)
  Fine:   intent embeddings                     │
  Prior category pinning  ◄────────────────────┘
    │
    ▼
LLM (Ollama / OpenAI-compatible)
  + conversation history (MULTI mode)
  + retrieved examples as evidence
  + optional verification pass
    │
    ▼
(intent_name, confidence, language)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Three classification modes** | `FLAT`, `FLAT_RAG`, `HIERARCHICAL_RAG` — configurable at init |
| **Two turn modes** | `SINGLE` (stateless) and `MULTI` (conversation-aware) |
| **RAG retrieval** | ExampleStore with cosine similarity; context-enriched query for follow-ups |
| **Hierarchical routing** | Two-level coarse→fine routing reduces cross-category noise at scale |
| **Prior category pinning** | Follow-ups stay in the correct domain even with zero semantic signal |
| **Automated bootstrapping** | LLM generates single-turn and multi-turn examples — no manual labeling needed |
| **Shared query embedding** | Encoded once, reused for both routing and RAG (no double encoding) |
| **Confidence gate** | Low retrieval similarity caps LLM confidence to signal uncharted territory |
| **Lazy heavy imports** | `torch`, `transformers`, `ollama` only imported when actually used |
| **123 offline unit tests** | Full test suite runs without GPU, LLM, or model downloads |

---

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install .
```

---

## Quick Start

### Simplest setup — FLAT mode (no RAG, no hierarchy)

```python
import asyncio
from query_classifier import IntentClassifier, ClassificationMode, TurnMode

INTENTS = [
    {"name": "check_balance",  "description": "User wants to check their account balance."},
    {"name": "transfer_money", "description": "User wants to transfer money to another account."},
    {"name": "block_card",     "description": "User wants to block a lost or stolen card."},
]

async def main():
    nlp = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.FLAT,
        turn_mode=TurnMode.SINGLE,
        llm_model_name="llama3",
        llm_base_url="http://localhost:11434",
    )
    intent, confidence, language = await nlp.classify("how much money do I have?")
    print(f"Intent: {intent}  Confidence: {confidence:.2f}")

asyncio.run(main())
```

### FLAT_RAG — add few-shot evidence from labeled examples

```python
from query_classifier import IntentClassifier, ClassificationMode, ExampleStore

store = ExampleStore()
store.add_examples_bulk([
    {"text": "what is my balance",   "intent": "check_balance"},
    {"text": "check my account",     "intent": "check_balance"},
    {"text": "transfer to savings",  "intent": "transfer_money"},
])

nlp = IntentClassifier(
    intents=INTENTS,
    mode=ClassificationMode.FLAT_RAG,
    example_store=store,
)
intent, conf, lang = await nlp.classify("how much is in my account")
```

### HIERARCHICAL_RAG — full REIC pipeline with hierarchy

```python
from query_classifier import IntentClassifier, ClassificationMode, TurnMode

HIERARCHY = {
    "accounts": {
        "description": "Managing bank accounts: balance, statements.",
        "intents": ["check_balance", "bank_statement"],
    },
    "cards": {
        "description": "Card management: block, unblock.",
        "intents": ["block_card"],
    },
}

nlp = IntentClassifier(
    intents=INTENTS,
    mode=ClassificationMode.HIERARCHICAL_RAG,
    turn_mode=TurnMode.MULTI,
    example_store=store,
    intent_hierarchy=HIERARCHY,
)
```

### Multi-turn conversation

```python
history = []

# Turn 1
intent, conf, _ = await nlp.classify("I need my bank statement", conversation_history=history)
history += [
    {"role": "user",      "content": "I need my bank statement"},
    {"role": "assistant", "content": "Sure.", "intent_classified": intent},
]

# Turn 2 — bare follow-up: RAG query is enriched with history automatically
intent, conf, _ = await nlp.classify("for last 6 months", conversation_history=history)
# → bank_statement  (not lost despite zero semantic signal in the bare phrase)
```

---

## Classification Modes

| Mode | Pipeline | When to use |
|------|----------|-------------|
| `FLAT` | SemanticRouter → LLM | No labeled examples; fastest setup |
| `FLAT_RAG` | SemanticRouter → ExampleStore → LLM | Have examples; small/flat intent set |
| `HIERARCHICAL_RAG` | HierarchicalRouter → ExampleStore (category-filtered) → LLM | Have examples AND a hierarchy; best accuracy at scale |

```python
# All three modes use the same classify() signature
intent, confidence, language = await nlp.classify(
    text,
    conversation_history=history,  # ignored in SINGLE mode
    verify=False,                   # optional second LLM verification pass
)
```

---

## Turn Modes

| Mode | Behaviour |
|------|-----------|
| `SINGLE` | Each call is independent. History is ignored. RAG uses raw query only. |
| `MULTI` | History enriches routing (prior category pinning) and RAG retrieval (sliding window context). |

```python
# Library-level configuration
nlp = IntentClassifier(intents=..., turn_mode=TurnMode.MULTI)

# Or via environment variable
# TURN_MODE=multi
```

---

## ExampleStore

Stores labeled utterances with optional `history_context` for multi-turn examples.

```python
from query_classifier import ExampleStore

store = ExampleStore()

# Single-turn
store.add_example("what is my balance", "check_balance")

# Multi-turn: embedding = encode("I need my bank statement for last 6 months")
store.add_example(
    text="for last 6 months",
    intent_name="bank_statement",
    history_context="I need my bank statement",
)

# Bulk load from list of dicts
store.add_examples_bulk(examples)

# Retrieve top-k by similarity
results = store.retrieve("account balance", k=5)
# [{"text": ..., "intent": ..., "score": ...}, ...]

# Persistence
store.save("my_store")    # writes my_store.json + my_store.npy
store.load("my_store")    # loads both; recomputes embeddings if .npy missing

# Coverage stats
store.intent_coverage()       # {"check_balance": 4, "block_card": 3, ...}
store.multi_turn_coverage()   # intents that have follow-up examples
```

---

## Automated Bootstrapping

Generate a fully labeled ExampleStore using the LLM itself — no manual labeling needed.

```python
import asyncio
from query_classifier import ExampleStore, ExampleStoreBootstrapper

store = ExampleStore()

bootstrapper = ExampleStoreBootstrapper(
    llm_model_name="llama3",
    llm_base_url="http://localhost:11434",
    concurrency=3,   # parallel LLM calls
    max_retries=2,
)

summary = await bootstrapper.run_full_bootstrap(
    store=store,
    intents=INTENTS,
    save_path="my_store",   # saved to my_store.json + my_store.npy
    n_single=8,             # utterances per intent (single-turn)
    n_multi=6,              # follow-up examples per intent (multi-turn)
)
# {"total": 140, "single_turn_generated": 80, "multi_turn_generated": 60, ...}
```

Bootstrap runs once. On subsequent startups, load from disk:

```python
store = ExampleStore()
store.load("my_store")
```

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend |
| `LLM_API_BASE` | `http://localhost:11434` | LLM API base URL |
| `LLM_MODEL_NAME` | `llama3` | Model name |
| `LLM_API_KEY` | _(empty)_ | Bearer token for authenticated endpoints |
| `ROUTER_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `RAG_TOP_K_EXAMPLES` | `6` | Examples retrieved per query |
| `EXAMPLE_STORE_PATH` | _(empty)_ | Auto-load store from this path on init |
| `TURN_MODE` | `single` | Default turn mode (`single` / `multi`) |
| `BOOTSTRAP_N_SINGLE` | `8` | Single-turn examples per intent during bootstrap |
| `BOOTSTRAP_N_MULTI` | `6` | Multi-turn examples per intent during bootstrap |
| `BOOTSTRAP_CONCURRENCY` | `3` | Max parallel LLM calls during bootstrap |

---

## Project Structure

```
query_classifier/
├── nlp_engine.py        # IntentClassifier — main entry point
├── example_store.py     # ExampleStore — labeled utterance store with RAG retrieval
├── semantic_router.py   # SemanticRouter — embedding-based intent routing
├── hierarchy.py         # HierarchicalRouter — two-level coarse→fine routing
├── bootstrapper.py      # ExampleStoreBootstrapper — automated example generation
├── config.py            # All configurable settings (env var backed)
└── __init__.py          # Public API exports

examples/
├── basic_single_turn.py       # FLAT vs FLAT_RAG side-by-side
├── multi_turn_conversation.py # Three full multi-turn conversations
├── custom_intents.py          # Plug-in your own domain (e-commerce)
├── bootstrap_store.py         # Full bootstrap → save → load → classify
├── reic_demo.py               # Complete REIC demo with all three modes
└── banking_intents.py         # Banking intent + hierarchy definitions

tests/
├── conftest.py                # MockEncoder (offline, deterministic), fixtures
├── test_example_store.py      # Population, retrieval, persistence, coverage
├── test_semantic_router.py    # encode_query, find_top_k, fallback paths
├── test_hierarchy.py          # Init, routing, prior category pinning
├── test_nlp_engine.py         # Enums, validation, classify() all modes, mocked LLM
└── test_bootstrapper.py       # parse_json, generate, bootstrap, full run
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 123 tests run fully offline — no Ollama, no GPU, no model downloads required.

---

## Running Examples

```bash
# Requires a running Ollama instance (ollama serve)

python examples/basic_single_turn.py
python examples/multi_turn_conversation.py
python examples/custom_intents.py
python examples/bootstrap_store.py
python examples/reic_demo.py
```

---

## License

MIT
