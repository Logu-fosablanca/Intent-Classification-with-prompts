"""
Pluggable encoder backends.

Set EMBEDDING_PROVIDER=jina  (+ JINA_API_KEY) to use the Jina AI API.
Leave unset or set EMBEDDING_PROVIDER=local to use SentenceTransformers.

Both backends expose the same interface:
    encoder.encode(text_or_list, show_progress_bar=False) -> np.ndarray
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class JinaEncoder:
    """
    Drop-in replacement for SentenceTransformer that calls the Jina AI
    embeddings API.  Free tier: 1 million tokens/month, no credit card.
    Sign up at https://jina.ai to get a key.
    """

    ENDPOINT = "https://api.jina.ai/v1/embeddings"
    DEFAULT_MODEL = "jina-embeddings-v3"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("JINA_API_KEY is required when EMBEDDING_PROVIDER=jina")
        self._api_key = api_key
        self._model = model

    def encode(
        self,
        sentences: Union[str, list],
        show_progress_bar: bool = False,  # ignored, kept for interface parity
        **kwargs,
    ) -> np.ndarray:
        import requests  # stdlib-available on any Python env

        if isinstance(sentences, str):
            sentences = [sentences]
            single = True
        else:
            single = False

        payload = {"model": self._model, "input": sentences}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.ENDPOINT, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        embeddings = np.array(
            [item["embedding"] for item in data["data"]], dtype=np.float32
        )
        return embeddings[0] if single else embeddings


def get_encoder(model_name: str):
    """
    Factory that returns the right encoder based on EMBEDDING_PROVIDER env var.

    EMBEDDING_PROVIDER=local  (default) -> SentenceTransformer(model_name)
    EMBEDDING_PROVIDER=jina             -> JinaEncoder(JINA_API_KEY)
    """
    import os

    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    if provider == "jina":
        api_key = os.getenv("JINA_API_KEY", "")
        jina_model = os.getenv("JINA_MODEL", JinaEncoder.DEFAULT_MODEL)
        logger.info(f"Using Jina AI encoder (model={jina_model})")
        return JinaEncoder(api_key=api_key, model=jina_model)

    # Default: local SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Using local SentenceTransformer encoder (model={model_name})")
        return SentenceTransformer(model_name)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Either install it or set EMBEDDING_PROVIDER=jina with a JINA_API_KEY."
        ) from e
