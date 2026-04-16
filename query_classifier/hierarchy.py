"""
HierarchicalRouter: two-level coarse-to-fine intent routing.

Implements the hierarchical classification component from REIC
(RAG-Enhanced Intent Classification at Scale).

Flow:
  Query
    │
    ▼
  Coarse router  →  top-N categories   (e.g. "cards", "transactions")
    │
    ▼
  Fine router    →  top-K intents       (search restricted to matched categories)
    │
    ▼
  ExampleStore   →  top-M examples      (filtered to matched categories)

Why this matters:
  - Flat search over 25 intents introduces cross-category noise
    (e.g. "block my card" might weakly match loan descriptions).
  - Hierarchical search first confirms the domain, then focuses precisely.
  - At scale (hundreds of intents) this is not optional — it's the only way
    to keep retrieval quality high without exploding candidate set size.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HierarchicalRouter:
    """
    Two-level semantic router: category → intent.

    Args:
        intents: Full list of {"name": str, "description": str} dicts.
        hierarchy: Dict mapping category_name → {"description": str, "intents": [intent_name, ...]}.
        encoder: A pre-loaded SentenceTransformer (shared with SemanticRouter to avoid reload).
        top_n_categories: How many top categories to keep at the coarse level.
                          Default 2 — covers ambiguous queries that span categories.
    """

    def __init__(
        self,
        intents: List[dict],
        hierarchy: Dict[str, dict],
        encoder,
        top_n_categories: int = 2,
    ):
        self.intents_by_name: Dict[str, dict] = {i["name"]: i for i in intents}
        self.hierarchy = hierarchy
        self.encoder = encoder
        self.top_n_categories = top_n_categories

        # Build category-level embeddings
        self.category_names: List[str] = list(hierarchy.keys())
        category_descriptions = [hierarchy[c]["description"] for c in self.category_names]

        logger.info(f"HierarchicalRouter: encoding {len(self.category_names)} category descriptions...")
        cat_embs = self.encoder.encode(category_descriptions, show_progress_bar=False)
        self.category_embeddings = np.array(cat_embs, dtype=np.float32)

        # Build per-category intent embeddings (fine-grained)
        self.category_intent_embeddings: Dict[str, np.ndarray] = {}
        self.category_intent_lists: Dict[str, List[dict]] = {}

        for cat_name, cat_data in hierarchy.items():
            cat_intents = [
                self.intents_by_name[name]
                for name in cat_data["intents"]
                if name in self.intents_by_name
            ]
            if not cat_intents:
                continue
            descriptions = [i["description"] for i in cat_intents]
            embs = self.encoder.encode(descriptions, show_progress_bar=False)
            self.category_intent_embeddings[cat_name] = np.array(embs, dtype=np.float32)
            self.category_intent_lists[cat_name] = cat_intents

        logger.info("HierarchicalRouter ready.")

    def route(
        self,
        query_emb: np.ndarray,
        top_k_intents: int = 5,
        prior_categories: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[dict]]:
        """
        Two-level routing from a pre-computed query embedding.

        Args:
            query_emb: Pre-computed query embedding (same model as encoder).
            top_k_intents: Number of fine-grained intent candidates to return.
            prior_categories: Categories from the previous conversation turn.
                              These are merged into matched_categories so that
                              follow-up queries ("for last 6 months") stay in
                              the correct domain even if the bare phrase scores
                              poorly against category descriptions.

        Returns:
            Tuple of:
              - matched_categories: list of category names selected at coarse level
              - top_intents: list of {"intent": dict, "score": float, "category": str}
                             sorted by fine-grained similarity score descending
        """
        query_emb = np.array(query_emb, dtype=np.float32)
        norm_q = np.linalg.norm(query_emb) + 1e-8

        # --- Coarse level: find top-N categories ---
        cat_norms = np.linalg.norm(self.category_embeddings, axis=1) + 1e-8
        cat_scores = self.category_embeddings.dot(query_emb) / (cat_norms * norm_q)

        top_n = min(self.top_n_categories, len(self.category_names))
        top_cat_indices = np.argsort(cat_scores)[::-1][:top_n]
        matched_categories = [self.category_names[i] for i in top_cat_indices]

        # Merge prior categories so continuation turns stay in the right domain.
        # E.g. "for last 6 months" after "account statement" should keep "accounts"
        # even if the bare phrase scores poorly against category descriptions.
        if prior_categories:
            for cat in prior_categories:
                if cat in self.category_names and cat not in matched_categories:
                    matched_categories.append(cat)
                    logger.info(f"Coarse routing: pinned prior category '{cat}' from history.")

        logger.info(
            f"Coarse routing: top categories = "
            + str([(self.category_names[i], round(float(cat_scores[i]), 3)) for i in top_cat_indices])
        )

        # --- Fine level: search within matched categories only ---
        all_candidates = []
        for cat_name in matched_categories:
            if cat_name not in self.category_intent_embeddings:
                continue
            intent_embs = self.category_intent_embeddings[cat_name]
            intent_list = self.category_intent_lists[cat_name]

            int_norms = np.linalg.norm(intent_embs, axis=1) + 1e-8
            int_scores = intent_embs.dot(query_emb) / (int_norms * norm_q)

            for idx, score in enumerate(int_scores):
                all_candidates.append({
                    "intent": intent_list[idx],
                    "score": float(score),
                    "category": cat_name,
                })

        # Sort all candidates across matched categories by score
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_intents = all_candidates[:top_k_intents]

        logger.info(
            f"Fine routing: top intents = "
            + str([(c["intent"]["name"], round(c["score"], 3)) for c in top_intents])
        )

        return matched_categories, top_intents

    def get_category_for_intent(self, intent_name: str) -> Optional[str]:
        """Return the category an intent belongs to."""
        for cat_name, cat_data in self.hierarchy.items():
            if intent_name in cat_data["intents"]:
                return cat_name
        return None
