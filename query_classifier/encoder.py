"""
Pluggable embedding encoder backends.

Configuration (env vars):
    EMBEDDING_PROVIDER  — which backend to use (default: "local")
    EMBEDDING_API_KEY   — API key for the chosen provider
    EMBEDDING_MODEL     — model name override (each provider has a sensible default)
    EMBEDDING_BASE_URL  — base URL override (useful for Ollama or self-hosted endpoints)

Supported providers:
    local     SentenceTransformers (runs on-device, needs ~500 MB RAM)
    openai    OpenAI embeddings API  (text-embedding-3-small default)
    cohere    Cohere embeddings API  (embed-english-v3.0 default)
    voyageai  Voyage AI API          (voyage-3-lite default)
    jina      Jina AI API            (jina-embeddings-v3 default, 1M free/mo)
    ollama    Ollama local server    (nomic-embed-text default)

All backends implement the same interface:
    encoder.encode(text_or_list, show_progress_bar=False) -> np.ndarray
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEncoder(ABC):
    """Common interface shared by all encoder backends."""

    @abstractmethod
    def encode(
        self,
        sentences: Union[str, list],
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode one string or a list of strings into float32 embeddings.

        Returns:
            1-D array  (shape: [dim])          when sentences is a str
            2-D array  (shape: [n, dim])        when sentences is a list
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _post(url: str, payload: dict, headers: dict) -> dict:
        import requests
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _normalise(sentences: Union[str, list]) -> tuple[list, bool]:
        """Return (list_of_strings, was_single_string)."""
        if isinstance(sentences, str):
            return [sentences], True
        return list(sentences), False

    @staticmethod
    def _to_array(vectors: list, single: bool) -> np.ndarray:
        arr = np.array(vectors, dtype=np.float32)
        return arr[0] if single else arr


# ---------------------------------------------------------------------------
# Local — SentenceTransformers
# ---------------------------------------------------------------------------

class LocalEncoder(BaseEncoder):
    """
    On-device encoder using the sentence-transformers library.
    Requires ~500 MB RAM; great for local dev and self-hosted deployments.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import
        logger.info(f"[encoder] Loading local SentenceTransformer: {model_name}")
        self._model = SentenceTransformer(model_name)

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        vecs = self._model.encode(texts, show_progress_bar=show_progress_bar)
        return self._to_array(vecs.tolist(), single)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIEncoder(BaseEncoder):
    """
    OpenAI embeddings API.
    Default model: text-embedding-3-small (~$0.02 / 1M tokens).
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    ENDPOINT = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = ""):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'openai'")
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/embeddings" if base_url else self.ENDPOINT

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        data = self._post(
            self._endpoint,
            {"model": self._model, "input": texts},
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        vectors = [item["embedding"] for item in data["data"]]
        return self._to_array(vectors, single)


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------

class CohereEncoder(BaseEncoder):
    """
    Cohere embeddings API.
    Default model: embed-english-v3.0.
    Free trial: 1 000 API calls/month.
    """

    DEFAULT_MODEL = "embed-english-v3.0"
    ENDPOINT = "https://api.cohere.ai/v1/embed"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **_):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'cohere'")
        self._api_key = api_key
        self._model = model

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        data = self._post(
            self.ENDPOINT,
            {
                "model": self._model,
                "texts": texts,
                "input_type": "search_query",
                "embedding_types": ["float"],
            },
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        vectors = data["embeddings"]["float"]
        return self._to_array(vectors, single)


# ---------------------------------------------------------------------------
# Voyage AI
# ---------------------------------------------------------------------------

class VoyageEncoder(BaseEncoder):
    """
    Voyage AI embeddings API (used by Anthropic internally).
    Default model: voyage-3-lite.
    Free tier: 50 M tokens/month.
    """

    DEFAULT_MODEL = "voyage-3-lite"
    ENDPOINT = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **_):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'voyageai'")
        self._api_key = api_key
        self._model = model

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        data = self._post(
            self.ENDPOINT,
            {"model": self._model, "input": texts},
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        vectors = [item["embedding"] for item in data["data"]]
        return self._to_array(vectors, single)


# ---------------------------------------------------------------------------
# Jina AI
# ---------------------------------------------------------------------------

class JinaEncoder(BaseEncoder):
    """
    Jina AI embeddings API.
    Default model: jina-embeddings-v3.
    Free tier: 1 M tokens/month, no credit card required.
    Get a key at https://jina.ai
    """

    DEFAULT_MODEL = "jina-embeddings-v3"
    ENDPOINT = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **_):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for provider 'jina'")
        self._api_key = api_key
        self._model = model

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        data = self._post(
            self.ENDPOINT,
            {"model": self._model, "input": texts},
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        vectors = [item["embedding"] for item in data["data"]]
        return self._to_array(vectors, single)


# ---------------------------------------------------------------------------
# Ollama (local server)
# ---------------------------------------------------------------------------

class OllamaEncoder(BaseEncoder):
    """
    Ollama local embedding server.
    Default model: nomic-embed-text (pull with: ollama pull nomic-embed-text).
    No API key needed; runs fully on-device.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL, base_url: str = ""):
        self._model = model
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

    def encode(self, sentences, show_progress_bar=False, **kwargs) -> np.ndarray:
        texts, single = self._normalise(sentences)
        import requests
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return self._to_array(vectors, single)


# ---------------------------------------------------------------------------
# Provider registry + factory
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "local":    LocalEncoder,
    "openai":   OpenAIEncoder,
    "cohere":   CohereEncoder,
    "voyageai": VoyageEncoder,
    "jina":     JinaEncoder,
    "ollama":   OllamaEncoder,
}


def get_encoder(local_model_name: str = LocalEncoder.DEFAULT_MODEL) -> BaseEncoder:
    """
    Return the encoder configured by environment variables.

    Env vars read:
        EMBEDDING_PROVIDER   provider key (default: "local")
        EMBEDDING_API_KEY    API key for the provider
        EMBEDDING_MODEL      model name override
        EMBEDDING_BASE_URL   base URL override (OpenAI-compatible or Ollama)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()
    api_key  = os.getenv("EMBEDDING_API_KEY", "")
    model    = os.getenv("EMBEDDING_MODEL", "")
    base_url = os.getenv("EMBEDDING_BASE_URL", "")

    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Supported: {supported}"
        )

    cls = PROVIDER_REGISTRY[provider]

    # For local provider the model comes from the router config, not EMBEDDING_MODEL
    if provider == "local":
        model = model or local_model_name
        logger.info(f"[encoder] Provider=local model={model}")
        return cls(model_name=model)

    model = model or cls.DEFAULT_MODEL  # type: ignore[attr-defined]
    logger.info(f"[encoder] Provider={provider} model={model}")
    return cls(api_key=api_key, model=model, base_url=base_url)
