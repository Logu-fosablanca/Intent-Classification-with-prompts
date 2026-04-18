
import numpy as np
import logging
from typing import List, Optional

from query_classifier.config import ROUTER_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class SemanticRouter:
    def __init__(self, intents: list, model_name: Optional[str] = None):
        self.intents = intents
        self.descriptions = [i["description"] for i in self.intents]
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self._intent_embeddings: Optional[np.ndarray] = None

        if not model_name:
            model_name = ROUTER_EMBEDDING_MODEL

        logger.info(f"Attempting to load embedding model: {model_name}...")
        try:
            from query_classifier.encoder import get_encoder
            self.model = get_encoder(model_name)
            self._intent_embeddings = np.array(
                self.model.encode(self.descriptions, show_progress_bar=False),
                dtype=np.float32,
            )
            logger.info("Semantic Router ready.")
        except ImportError as e:
            logger.warning(f"Encoder not available ({e}). Using TF-IDF fallback.")
            self._setup_tfidf()
        except Exception as e:
            logger.warning(f"Failed to load encoder ({e}). Using TF-IDF fallback.")
            self._setup_tfidf()

    def _setup_tfidf(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.descriptions)
            self.cosine_similarity = cosine_similarity
            logger.info("Semantic Router ready with TF-IDF.")
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")

    # ------------------------------------------------------------------
    # Encoding API (used by IntentClassifier to share the query embedding)
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encode a query string into a float32 embedding vector.

        Returns None if the model is unavailable (TF-IDF fallback mode).
        IntentClassifier calls this once and passes the result to both
        find_top_k_from_embedding() and ExampleStore.retrieve_from_embedding()
        to avoid encoding the same query twice.
        """
        if self.model is not None:
            return np.array(self.model.encode(query, show_progress_bar=False), dtype=np.float32)
        return None

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def find_top_k_from_embedding(self, query_emb: np.ndarray, k: int = 5) -> List[dict]:
        """
        Find top-k intents using a pre-computed query embedding.

        Preferred over find_top_k() when the embedding was already computed
        for other purposes (e.g. ExampleStore retrieval).
        """
        if self._intent_embeddings is None:
            logger.warning("No intent embeddings available. Falling back to random.")
            return [{"intent": self.intents[i], "score": 0.0} for i in range(min(k, len(self.intents)))]

        query_emb = np.array(query_emb, dtype=np.float32)
        norm_q = np.linalg.norm(query_emb) + 1e-8
        norms = np.linalg.norm(self._intent_embeddings, axis=1) + 1e-8
        scores = self._intent_embeddings.dot(query_emb) / (norms * norm_q)

        top_k = min(k, len(self.intents))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"intent": self.intents[idx], "score": float(scores[idx])}
            for idx in top_indices
        ]

    def find_top_k(self, query: str, k: int = 5) -> List[dict]:
        """
        Find top-k intents for a query string.

        For production use inside IntentClassifier, prefer encode_query() +
        find_top_k_from_embedding() to share the embedding with ExampleStore.
        This method is kept for backward compatibility and standalone use.
        """
        try:
            if self.model is not None:
                query_emb = self.encode_query(query)
                return self.find_top_k_from_embedding(query_emb, k=k)

            elif self.vectorizer is not None:
                query_vec = self.vectorizer.transform([query])
                scores = self.cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                top_k = min(k, len(self.intents))
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [
                    {"intent": self.intents[idx], "score": float(scores[idx])}
                    for idx in top_indices
                ]

            else:
                logger.warning("No router model available. Returning first k intents.")
                return [
                    {"intent": self.intents[i], "score": 0.0}
                    for i in range(min(k, len(self.intents)))
                ]

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return [
                {"intent": self.intents[i], "score": 0.0}
                for i in range(min(k, len(self.intents)))
            ]
