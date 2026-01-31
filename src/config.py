
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
