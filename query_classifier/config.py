
import os

# Configuration for AI Models

# LLM Provider
# Options: "ollama", "openai", "azure", "custom"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Generic LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434")  # Default for Ollama
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_CLIENT_ID = os.getenv("LLM_CLIENT_ID", "")
LLM_CLIENT_SECRET = os.getenv("LLM_CLIENT_SECRET", "")

# Semantic Router Embedding Model (SentenceTransformers)
# Default: "all-MiniLM-L6-v2"
ROUTER_EMBEDDING_MODEL = os.getenv("ROUTER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Language Detection Model (Hugging Face Transformers)
# Default: "papluca/xlm-roberta-base-language-detection"
LANG_DETECT_MODEL = os.getenv("LANG_DETECT_MODEL", "papluca/xlm-roberta-base-language-detection")

# RAG / ExampleStore Configuration (REIC)
# Number of labeled examples retrieved per query for few-shot prompting
RAG_TOP_K_EXAMPLES = int(os.getenv("RAG_TOP_K_EXAMPLES", "6"))
# Max examples from any single intent (diversity control)
RAG_PER_INTENT_LIMIT = int(os.getenv("RAG_PER_INTENT_LIMIT", "2"))
# Weight for blending retrieval similarity score into final confidence (0 = LLM only, 1 = retrieval only)
RAG_CONFIDENCE_BLEND = float(os.getenv("RAG_CONFIDENCE_BLEND", "0.25"))
# Optional path to a pre-built example store JSON file to load at startup
EXAMPLE_STORE_PATH = os.getenv("EXAMPLE_STORE_PATH", "")

# Turn Mode — "single" or "multi" (see TurnMode enum)
TURN_MODE = os.getenv("TURN_MODE", "single")

# Bootstrap Configuration (ExampleStoreBootstrapper)
# Number of single-turn examples to generate per intent via LLM
BOOTSTRAP_N_SINGLE = int(os.getenv("BOOTSTRAP_N_SINGLE", "8"))
# Number of multi-turn follow-up examples to generate per intent via LLM
BOOTSTRAP_N_MULTI = int(os.getenv("BOOTSTRAP_N_MULTI", "6"))
# Max concurrent LLM calls during bootstrap
BOOTSTRAP_CONCURRENCY = int(os.getenv("BOOTSTRAP_CONCURRENCY", "3"))
# Max retries per intent if LLM call fails during bootstrap
BOOTSTRAP_MAX_RETRIES = int(os.getenv("BOOTSTRAP_MAX_RETRIES", "2"))
