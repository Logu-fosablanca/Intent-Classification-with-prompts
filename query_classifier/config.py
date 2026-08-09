
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

# Embedding Provider
# Options: "local" | "openai" | "cohere" | "voyageai" | "jina" | "ollama"
# Use any API-based provider on memory-constrained hosts (e.g. Render free 512 MB)
# See query_classifier/encoder.py for full documentation.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")   # API key for the chosen provider
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "")     # overrides provider default model
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")  # for Ollama or OpenAI-compatible endpoints

# Semantic Router Embedding Model (used when EMBEDDING_PROVIDER=local)
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

# LLM Reranker (e.g. Qwen3-4B via Ollama) — replaces the generative
# classification call with a reranking pass over retrieved candidates.
# Options: "off" | "listwise" | "pointwise"
#   off       : original behaviour — one LLM call generates {name, confidence}.
#   listwise  : one LLM call ranks/scores all candidates at once (same call
#               cost as "off", just a reranking-framed prompt).
#   pointwise : one LLM call PER candidate intent, each scored independently.
#               More calls, but confidence is a genuine per-candidate score
#               instead of a single self-reported number for the winner.
RERANK_MODE = os.getenv("RERANK_MODE", "off")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "qwen3:4b")
# Max concurrent LLM calls during pointwise reranking (one call per candidate)
RERANK_POINTWISE_CONCURRENCY = int(os.getenv("RERANK_POINTWISE_CONCURRENCY", "4"))

# Hybrid Retrieval (FAISS dense + BM25 lexical, fused via Reciprocal Rank
# Fusion). Only affects ExampleStore retrieval — intent routing stays pure
# dense (few, short descriptions; lexical fusion adds little value there).
# Requires optional deps: pip install query-classifier[rerank]
USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "false").lower() == "true"
# RRF constant — standard default, higher = flatter weighting of rank position
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
# FAISS index type: "flat" (exact, recommended below ~100k examples) | "hnsw" (ANN, larger scale)
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat")

# Bootstrap Configuration (ExampleStoreBootstrapper)
# Number of single-turn examples to generate per intent via LLM
BOOTSTRAP_N_SINGLE = int(os.getenv("BOOTSTRAP_N_SINGLE", "8"))
# Number of multi-turn follow-up examples to generate per intent via LLM
BOOTSTRAP_N_MULTI = int(os.getenv("BOOTSTRAP_N_MULTI", "6"))
# Max concurrent LLM calls during bootstrap
BOOTSTRAP_CONCURRENCY = int(os.getenv("BOOTSTRAP_CONCURRENCY", "3"))
# Max retries per intent if LLM call fails during bootstrap
BOOTSTRAP_MAX_RETRIES = int(os.getenv("BOOTSTRAP_MAX_RETRIES", "2"))
