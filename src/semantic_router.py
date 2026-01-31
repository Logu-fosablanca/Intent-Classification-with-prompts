
import numpy as np
import logging
from src.intents_db import INTENTS
from src.config import ROUTER_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class SemanticRouter:
    def __init__(self, model_name=ROUTER_EMBEDDING_MODEL):
        self.intents = INTENTS
        self.descriptions = [i['description'] for i in self.intents]
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
        logger.info(f"Attempting to load embedding model: {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embeddings = self.model.encode(self.descriptions)
            logger.info(f"Semantic Router ready with SentenceTransformer.")
        except ImportError as e:
            logger.warning(f"SentenceTransformer not available ({e}). Using TF-IDF fallback.")
            self._setup_tfidf()
        except Exception as e:
            logger.warning(f"Failed to load/use SentenceTransformer ({e}). Using TF-IDF fallback.")
            self._setup_tfidf()

    def _setup_tfidf(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.descriptions)
            self.cosine_similarity = cosine_similarity
            logger.info("Semantic Router ready with TF-IDF.")
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")

    def find_top_k(self, query: str, k: int = 5):
        try:
            # 1. Try SBERT
            if self.model:
                query_embedding = self.model.encode([query])[0]
                norm_query = np.linalg.norm(query_embedding)
                norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
                scores = np.dot(self.embeddings, query_embedding) / (norm_embeddings * norm_query)
            
            # 2. Try TF-IDF
            elif self.vectorizer:
                query_vec = self.vectorizer.transform([query])
                scores = self.cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # 3. Fail safe
            else:
                logger.warning("No router model available. Returning random.")
                return [{"intent": self.intents[i], "score": 0.0} for i in range(k)]

            # Common Sort Logic
            top_k_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_k_indices:
                results.append({
                    "intent": self.intents[idx],
                    "score": float(scores[idx])
                })
            return results

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return [{"intent": self.intents[i], "score": 0.0} for i in range(k)]

