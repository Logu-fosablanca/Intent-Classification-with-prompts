
# Query Classifier

A generic, scalable intent classification library that combines **Semantic Routing** (SentenceTransformers/TF-IDF) for speed and pre-filtering with **LLMs** (Ollama, etc.) for intelligent reasoning and verification.

## Features

- **Hybrid Approach**: Fast semantic search + LLM reasoning.
- **Configurable**: Use any LLM backend (Ollama, OpenAI compatible, etc.).
- **Parallel Execution**: Language detection and semantic retrieval run in parallel.
- **Verification Loop**: Optional self-verification step by the LLM to reduce hallucinations.
- **Generic**: Define your own intents and prompts.

## Installation

```bash
pip install .
```

## Quick Start

1. Define your intents (list of dicts with `name` and `description`).
2. Initialize the `IntentClassifier`.
3. Call `classify()`.

### Example

```python
import asyncio
from query_classifier.nlp_engine import IntentClassifier

# 1. Define Intents
intents = [
    {"name": "book_flight", "description": "User wants to book a flight ticket."},
    {"name": "check_weather", "description": "User asks about weather conditions."},
    {"name": "contact_support", "description": "User wants to talk to customer service."}
]

async def main():
    # 2. Initialize
    # You can pass explicit config or fallback to env vars
    nlp = IntentClassifier(
        intents=intents,
        llm_model_name="llama3",  # Optional: Link specific model
        llm_base_url="http://localhost:11434", # Optional: Link specific provider
        # llm_api_key="sk-..." # Optional: For authenticated endpoints
    )

    # 3. Classify
    text = "I need to fly to New York tomorrow"
    label, score, lang = await nlp.classify(text)
    
    print(f"Intent: {label}, Confidence: {score}, Language: {lang}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

Set environment variables to configure the LLM provider:

- `LLM_PROVIDER`: `ollama` (default)
- `LLM_API_BASE`: `http://localhost:11434` (default)
- `LLM_MODEL_NAME`: `llama3` (default)
- `LLM_API_KEY`: (Optional) Bearer token for authentication.


## Running Examples

You can run the provided examples directly from the root directory:

```bash
# Basic usage demo
python examples/basic_usage.py

# Full multi-agent interaction demo
python examples/agent_demo.py
```

## License

MIT
