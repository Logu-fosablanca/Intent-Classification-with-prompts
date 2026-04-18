"""
ExampleStore: RAG-based labeled utterance store for intent classification.

Implements the retrieval component from REIC (RAG-Enhanced Intent Classification at Scale).

Key design:
    Single-turn examples:  stored with embedding of text alone.
    Multi-turn examples:   stored with embedding of "history_context text" so that
                           retrieval using a context-enriched query finds them correctly.

Both types coexist in the same store. Retrieval naturally surfaces whichever matches
the incoming query (raw or context-enriched) best.

The displayed text in the LLM prompt is always the raw utterance — only the embedding
uses the context. This keeps the few-shot examples readable.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional

from query_classifier.config import ROUTER_EMBEDDING_MODEL, RAG_PER_INTENT_LIMIT

logger = logging.getLogger(__name__)


class ExampleStore:
    """
    A retrieval-augmented example store for intent classification.

    Supports both single-turn and multi-turn labeled examples.

    Single-turn usage:
        store.add_example("what is my balance", "account_check_balance")

    Multi-turn usage (follow-up utterances):
        store.add_example(
            text="for last 6 months",
            intent_name="account_statement_request",
            history_context="I need my bank statement",
        )
        # Stored embedding = encode("I need my bank statement for last 6 months")
        # At retrieval, rag_query = "I need my bank statement for last 6 months"
        # → high cosine similarity → correctly retrieved ✅

    At inference time, IntentClassifier builds a context-enriched RAG query
    (last user utterances + current text) and calls retrieve_from_embedding().
    Both single-turn and multi-turn stored examples participate in retrieval —
    whichever matches the enriched query best surfaces to the top.
    """

    def __init__(self, encoder=None, model_name: Optional[str] = None):
        """
        Args:
            encoder: A pre-loaded SentenceTransformer instance.
                     Pass SemanticRouter.model to avoid loading the model twice.
            model_name: If encoder is None, load this SentenceTransformer model.
                        Defaults to ROUTER_EMBEDDING_MODEL from config.
        """
        self._texts: List[str] = []
        self._intents: List[str] = []
        self._history_contexts: List[Optional[str]] = []   # None for single-turn examples
        self._embeddings: Optional[np.ndarray] = None      # shape (N, D)

        if encoder is not None:
            self.encoder = encoder
            logger.info("ExampleStore: using shared encoder from SemanticRouter.")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                model = model_name or ROUTER_EMBEDDING_MODEL
                self.encoder = SentenceTransformer(model)
                logger.info(f"ExampleStore: loaded encoder '{model}'.")
            except Exception as e:
                logger.error(f"ExampleStore: failed to load encoder: {e}")
                self.encoder = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_string(text: str, history_context: Optional[str]) -> str:
        """
        Build the string to embed for a given example.

        Single-turn : embed text alone.
        Multi-turn  : embed "history_context text" so the stored vector is
                      aligned with the context-enriched retrieval query.
        """
        if history_context:
            return f"{history_context.strip()} {text.strip()}"
        return text

    def _append_one(self, text: str, intent: str, history_context: Optional[str], emb: np.ndarray):
        self._texts.append(text)
        self._intents.append(intent)
        self._history_contexts.append(history_context)
        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        self._embeddings = emb if self._embeddings is None else np.vstack([self._embeddings, emb])

    # ------------------------------------------------------------------
    # Population API
    # ------------------------------------------------------------------

    def add_example(
        self,
        text: str,
        intent_name: str,
        history_context: Optional[str] = None,
    ):
        """
        Add a single labeled utterance.

        Args:
            text: The user utterance (displayed as-is in the LLM prompt).
            intent_name: The correct intent label.
            history_context: Optional prior user turns as a single string.
                             Provide this for follow-up utterances so the stored
                             embedding captures the full conversational meaning.

                             Example:
                               text            = "for last 6 months"
                               history_context = "I need my bank statement"
                               stored embed    = encode("I need my bank statement for last 6 months")
        """
        if self.encoder is None:
            logger.warning("ExampleStore: no encoder; example not added.")
            return
        embed_str = self._embed_string(text, history_context)
        emb = self.encoder.encode(embed_str, show_progress_bar=False)
        self._append_one(text, intent_name, history_context, emb)

    def add_examples_bulk(self, examples: List[Dict]):
        """
        Add labeled utterances efficiently via batch encoding.

        Args:
            examples: List of dicts with keys:
                "text"            : str  — the user utterance (required)
                "intent"          : str  — intent label (required)
                "history_context" : str  — prior user turns (optional, for multi-turn examples)

        Example with multi-turn entries:
            [
                {"text": "what is my balance",     "intent": "account_check_balance"},
                {"text": "for last 6 months",      "intent": "account_statement_request",
                 "history_context": "I need my bank statement"},
            ]
        """
        if not examples or self.encoder is None:
            return

        embed_strings = [
            self._embed_string(e["text"], e.get("history_context"))
            for e in examples
        ]

        embeddings = self.encoder.encode(embed_strings, batch_size=64, show_progress_bar=False)
        new_embs = np.array(embeddings, dtype=np.float32)

        for e in examples:
            self._texts.append(e["text"])
            self._intents.append(e["intent"])
            self._history_contexts.append(e.get("history_context"))

        self._embeddings = (
            new_embs if self._embeddings is None
            else np.vstack([self._embeddings, new_embs])
        )

        n_multi = sum(1 for e in examples if e.get("history_context"))
        logger.info(
            f"ExampleStore: added {len(examples)} examples "
            f"({n_multi} multi-turn, {len(examples) - n_multi} single-turn) "
            f"| total: {len(self._texts)}"
        )

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def retrieve_from_embedding(
        self,
        query_emb: np.ndarray,
        k: int = 6,
        per_intent_limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar examples using a pre-computed query embedding.

        Implements the diversity control from REIC: at most `per_intent_limit`
        examples per intent are returned, preventing a single dominant intent from
        monopolising the few-shot evidence shown to the LLM.

        The caller passes the context-enriched RAG embedding (built from last user
        turns + current query). Both single-turn and multi-turn stored examples
        participate — whichever embedding is most similar surfaces to the top.

        Args:
            query_emb: Pre-computed embedding (same model as encoder).
                       Should be encode(last_user_turns + " " + current_text)
                       when conversation history exists.
            k: Maximum number of examples to return.
            per_intent_limit: Maximum examples from any single intent.
                              Defaults to RAG_PER_INTENT_LIMIT from config (2).
                              Set to None or a large number to disable.

        Returns:
            List of {"text": str, "intent": str, "score": float} sorted by
            similarity descending. text is always the raw utterance (readable).
        """
        if self._embeddings is None or len(self._texts) == 0:
            return []

        if per_intent_limit is None:
            per_intent_limit = RAG_PER_INTENT_LIMIT

        query_emb = np.array(query_emb, dtype=np.float32)
        norm_q = np.linalg.norm(query_emb) + 1e-8

        norms = np.linalg.norm(self._embeddings, axis=1) + 1e-8
        scores = self._embeddings.dot(query_emb) / (norms * norm_q)

        results: List[Dict] = []
        intent_counts: Dict[str, int] = {}

        for idx in np.argsort(scores)[::-1]:
            intent = self._intents[idx]
            if intent_counts.get(intent, 0) >= per_intent_limit:
                continue
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            results.append({
                "text":   self._texts[idx],
                "intent": intent,
                "score":  float(scores[idx]),
            })
            if len(results) >= k:
                break

        return results

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        """
        Retrieve top-k examples by encoding the query on the fly.
        For standalone / offline use only.
        """
        if self.encoder is None:
            return []
        query_emb = self.encoder.encode(query, show_progress_bar=False)
        return self.retrieve_from_embedding(query_emb, k=k)

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Persist examples to disk.

        Saves two files:
          <path>.json  — text, intent, and history_context labels
          <path>.npy   — embeddings matrix (avoids recomputing on reload)
        """
        data = [
            {"text": t, "intent": i, "history_context": h}
            for t, i, h in zip(self._texts, self._intents, self._history_contexts)
        ]
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self._embeddings is not None:
            np.save(f"{path}.npy", self._embeddings)

        n_multi = sum(1 for h in self._history_contexts if h)
        logger.info(
            f"ExampleStore: saved {len(data)} examples "
            f"({n_multi} multi-turn) to {path}.json + {path}.npy"
        )

    def load(self, path: str):
        """
        Load examples from disk.
        Loads embeddings from .npy if available; otherwise recomputes them
        using history_context if present (requires encoder).
        """
        with open(f"{path}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self._texts = [e["text"] for e in data]
        self._intents = [e["intent"] for e in data]
        self._history_contexts = [e.get("history_context") for e in data]

        try:
            self._embeddings = np.load(f"{path}.npy")
            logger.info(
                f"ExampleStore: loaded {len(data)} examples + precomputed embeddings from {path}.*"
            )
        except FileNotFoundError:
            logger.warning(f"ExampleStore: no .npy at {path}.npy — recomputing embeddings.")
            if self.encoder is None:
                logger.error("ExampleStore: cannot recompute — no encoder.")
                return
            embed_strings = [
                self._embed_string(t, h)
                for t, h in zip(self._texts, self._history_contexts)
            ]
            embeddings = self.encoder.encode(embed_strings, batch_size=64, show_progress_bar=False)
            self._embeddings = np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._texts)

    def intent_coverage(self) -> Dict[str, int]:
        """Count of examples per intent."""
        counts: Dict[str, int] = {}
        for intent in self._intents:
            counts[intent] = counts.get(intent, 0) + 1
        return counts

    def multi_turn_coverage(self) -> Dict[str, int]:
        """Count of multi-turn examples per intent — useful for auditing gaps."""
        counts: Dict[str, int] = {}
        for intent, ctx in zip(self._intents, self._history_contexts):
            if ctx:
                counts[intent] = counts.get(intent, 0) + 1
        return counts
