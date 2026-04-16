
import logging
import asyncio
import json
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from query_classifier.semantic_router import SemanticRouter
from query_classifier.config import (
    LLM_MODEL_NAME, LANG_DETECT_MODEL, LLM_PROVIDER,
    LLM_API_BASE, LLM_API_KEY, RAG_TOP_K_EXAMPLES, EXAMPLE_STORE_PATH,
    TURN_MODE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn Mode
# ---------------------------------------------------------------------------

class TurnMode(str, Enum):
    """
    Controls whether the classifier operates in single-turn or multi-turn mode.

    SINGLE
        Each call to classify() is treated as a standalone query.
        Conversation history is ignored even if passed.
        RAG retrieval uses the raw query only.
        Use when: every user query is independent (e.g. search, form inputs).

    MULTI
        classify() uses conversation_history to enrich both routing and RAG.
        - Prior categories are pinned so continuation turns stay in the right domain.
        - RAG query is built from recent user utterances + current text so bare
          follow-ups like "for last 6 months" retrieve relevant examples.
        Use when: building a conversational assistant or chatbot.
    """
    SINGLE = "single"
    MULTI  = "multi"


# ---------------------------------------------------------------------------
# Classification Mode
# ---------------------------------------------------------------------------

class ClassificationMode(str, Enum):
    """
    Controls which classification pipeline is used.

    FLAT
        SemanticRouter → LLM.
        The original behaviour. No RAG, no hierarchy.
        Use when: you have no labeled examples, or want the fastest setup.

    FLAT_RAG
        SemanticRouter → ExampleStore (all intents) → LLM.
        Adds REIC-style few-shot evidence without hierarchical routing.
        Use when: you have labeled examples but your intent set is small/flat.

    HIERARCHICAL_RAG
        HierarchicalRouter (coarse category → fine intent)
        → ExampleStore (category-filtered) → LLM.
        Full REIC pipeline. Best accuracy at scale.
        Use when: you have labeled examples AND a defined intent hierarchy.
    """
    FLAT = "flat"
    FLAT_RAG = "flat_rag"
    HIERARCHICAL_RAG = "hierarchical_rag"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Configurable intent classifier supporting three classification modes
    and two turn modes.

    Examples
    --------
    # Single-turn, flat (no RAG, no hierarchy) — simplest setup
    nlp = IntentClassifier(intents=INTENTS, mode="flat", turn_mode="single")

    # Multi-turn, flat + RAG — conversational with few-shot evidence
    nlp = IntentClassifier(
        intents=INTENTS,
        mode="flat_rag",
        turn_mode="multi",
        example_store=store,
    )

    # Multi-turn, hierarchical + RAG — full REIC pipeline
    nlp = IntentClassifier(
        intents=INTENTS,
        mode="hierarchical_rag",
        turn_mode="multi",
        example_store=store,
        intent_hierarchy=INTENT_HIERARCHY,
    )

    # classify() signature is identical regardless of mode or turn_mode
    # Single-turn: history is ignored
    intent, conf, lang = await nlp.classify("how much money do I have?")

    # Multi-turn: pass history and the classifier handles context automatically
    intent, conf, lang = await nlp.classify("for last 6 months", conversation_history=history)
    """

    def __init__(
        self,
        intents: List[dict],
        mode: Union[ClassificationMode, str] = ClassificationMode.FLAT,
        turn_mode: Union[TurnMode, str] = TURN_MODE,
        # LLM
        llm_provider: str = LLM_PROVIDER,
        llm_model_name: str = LLM_MODEL_NAME,
        llm_base_url: str = LLM_API_BASE,
        llm_api_key: str = LLM_API_KEY,
        # Embeddings
        embedding_model: Optional[str] = None,
        # Language detection
        lang_detect_model: str = LANG_DETECT_MODEL,
        enable_lang_detect: bool = True,
        # RAG (required for FLAT_RAG and HIERARCHICAL_RAG)
        example_store=None,
        # Hierarchy (required for HIERARCHICAL_RAG)
        intent_hierarchy: Optional[Dict] = None,
        top_n_categories: int = 2,
    ):
        """
        Parameters
        ----------
        intents        : list of {"name": str, "description": str}
        mode           : ClassificationMode or str — "flat", "flat_rag", "hierarchical_rag"
        turn_mode      : TurnMode or str — "single" or "multi"
                         "single" ignores conversation history entirely.
                         "multi"  uses history to enrich routing and RAG retrieval.
        llm_provider   : LLM backend (currently "ollama" or compatible)
        llm_model_name : model name passed to the LLM API
        llm_base_url   : LLM API base URL
        llm_api_key    : optional Bearer token
        embedding_model: SentenceTransformer model name override
        lang_detect_model : HuggingFace model for language detection
        enable_lang_detect : set False to skip language detection entirely
        example_store  : ExampleStore instance (required for *_rag modes)
        intent_hierarchy : dict mapping category → {"description", "intents"}
                           (required for hierarchical_rag mode)
        top_n_categories : how many top categories to consider at coarse level
        """
        self.mode = ClassificationMode(mode)
        self.turn_mode = TurnMode(turn_mode)
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.enable_lang_detect = enable_lang_detect

        logger.info(f"Initializing IntentClassifier | mode={self.mode.value} | turn_mode={self.turn_mode.value}")

        # Intent lookup used for contextual query building (routing context needs description)
        self._intents_by_name: Dict[str, dict] = {i["name"]: i for i in intents}

        # --- Validate mode requirements up-front (fail fast) ---
        self._validate_mode_config(mode=self.mode, example_store=example_store,
                                   intent_hierarchy=intent_hierarchy)

        # --- Semantic Router (shared encoder, used by all modes) ---
        router_kwargs: dict = {"intents": intents}
        if embedding_model:
            router_kwargs["model_name"] = embedding_model
        self.router = SemanticRouter(**router_kwargs)

        # --- Hierarchical Router (HIERARCHICAL_RAG only) ---
        self.hier_router = None
        if self.mode == ClassificationMode.HIERARCHICAL_RAG and intent_hierarchy:
            from query_classifier.hierarchy import HierarchicalRouter
            self.hier_router = HierarchicalRouter(
                intents=intents,
                hierarchy=intent_hierarchy,
                encoder=self.router.model,
                top_n_categories=top_n_categories,
            )
            self._hierarchy = intent_hierarchy
        else:
            self._hierarchy = {}

        # --- Example Store (RAG modes) ---
        self.example_store = example_store
        if self.example_store is None and EXAMPLE_STORE_PATH:
            self.example_store = self._try_autoload_store(EXAMPLE_STORE_PATH)

        if self.example_store is not None:
            logger.info(f"ExampleStore: {len(self.example_store)} labeled examples loaded.")
        elif self.mode in (ClassificationMode.FLAT_RAG, ClassificationMode.HIERARCHICAL_RAG):
            logger.warning(
                f"Mode '{self.mode.value}' works best with an ExampleStore. "
                "Classification will proceed with description-only prompting."
            )

        # --- Language Detection ---
        self.tokenizer = None
        self.lang_model = None
        if enable_lang_detect:
            logger.info(f"Loading language detection model ({lang_detect_model})...")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(lang_detect_model)
                self.lang_model = AutoModelForSequenceClassification.from_pretrained(
                    lang_detect_model
                )
                logger.info("Language detection model loaded.")
            except Exception as e:
                logger.warning(f"Language detection failed to load: {e}. Disabling.")
                self.enable_lang_detect = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mode_config(
        mode: ClassificationMode,
        example_store,
        intent_hierarchy: Optional[Dict],
    ):
        """Raise early with a clear message if mode requirements aren't met."""
        if mode == ClassificationMode.HIERARCHICAL_RAG and not intent_hierarchy:
            raise ValueError(
                "ClassificationMode.HIERARCHICAL_RAG requires 'intent_hierarchy' to be provided.\n"
                "Define a hierarchy dict mapping category names to their descriptions and intent lists,\n"
                "then pass it as intent_hierarchy=YOUR_HIERARCHY.\n"
                "See examples/banking_intents.py for a reference structure."
            )
        if mode == ClassificationMode.FLAT and example_store is not None:
            logger.warning(
                "An example_store was provided but mode='flat' does not use it. "
                "Switch to mode='flat_rag' or 'hierarchical_rag' to enable RAG."
            )

    def _try_autoload_store(self, path: str):
        try:
            from query_classifier.example_store import ExampleStore
            store = ExampleStore(encoder=self.router.model)
            store.load(path)
            logger.info(f"Auto-loaded ExampleStore from {path}")
            return store
        except Exception as e:
            logger.warning(f"Could not auto-load ExampleStore from '{path}': {e}")
            return None

    # ------------------------------------------------------------------
    # Conversation History Utilities
    # ------------------------------------------------------------------

    def _extract_prior_intent(self, conversation_history: List[dict]) -> Optional[str]:
        """
        Extract the most recently classified intent from conversation history.

        Supports two history formats:
          1. {"role": "assistant", "intent_classified": "account_statement_request"}
             — explicit field, recommended for clean integration.
          2. {"role": "assistant", "content": "[intent: account_statement_request]"}
             — parsed from the demo's bracket notation, for backward compat.

        Returns None if no prior intent can be found.
        """
        for turn in reversed(conversation_history):
            if turn.get("role") != "assistant":
                continue
            # Format 1: explicit field
            if turn.get("intent_classified"):
                return turn["intent_classified"]
            # Format 2: "[intent: X]" in content
            content = turn.get("content", "")
            if "intent:" in content:
                try:
                    return content.split("intent:")[-1].strip().rstrip("]").strip()
                except Exception:
                    pass
        return None

    def _build_rag_query(
        self,
        text: str,
        conversation_history: Optional[List[dict]],
        window: int = 2,
    ) -> str:
        """
        Build the RAG retrieval query by prepending recent user utterances.

        Problem:
            A bare follow-up like "for last 6 months" has no semantic signal on its
            own. Its embedding lands near generic time/date concepts, not near
            bank statement examples in the store. RAG retrieves nothing useful.

        Fix:
            Prepend the last `window` user turns so the retrieval query becomes
            "I need my bank statement for last 6 months" — which correctly matches
            examples labeled account_statement_request.

        Why user utterances and not intent names/descriptions:
            - Uses natural language the embedding model was trained on.
            - No dependency on prior classification being correct.
            - A misclassified prior turn would corrupt intent-based context;
              the user's own words never can.

        Args:
            text: Current raw user query.
            conversation_history: Full conversation history list.
            window: How many prior user turns to prepend. Default 2 keeps the
                    query focused without growing too long.

        Returns:
            Context-enriched string for embedding, or raw text if no history.
        """
        if not conversation_history:
            return text

        prior_turns = []
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                content = turn.get("content", "").strip()
                if content and content != text:
                    prior_turns.append(content)
                if len(prior_turns) >= window:
                    break

        if not prior_turns:
            return text

        # Join from oldest to newest, then append current query
        context = " ".join(reversed(prior_turns))
        return f"{context} {text}"

    # ------------------------------------------------------------------
    # Routing (mode-dispatched, history-aware)
    # ------------------------------------------------------------------

    def _route(
        self,
        query_emb: np.ndarray,
        prior_categories: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[dict]]:
        """
        Route a query to candidate intents.

        Parameters
        ----------
        query_emb : embedding of the (possibly context-enriched) query
        prior_categories : category names from the previous turn, merged into
                           the coarse routing result so continuation queries
                           stay in the correct domain.

        Returns
        -------
        matched_categories : list of category names (empty for flat modes)
        top_matches : list of {"intent": dict, "score": float, ["category": str]}
        """
        if self.mode == ClassificationMode.HIERARCHICAL_RAG and self.hier_router:
            matched_categories, top_matches = self.hier_router.route(
                query_emb,
                top_k_intents=5,
                prior_categories=prior_categories,
            )
            return matched_categories, top_matches
        else:
            top_matches = self.router.find_top_k_from_embedding(query_emb, k=5)
            return [], top_matches

    # ------------------------------------------------------------------
    # RAG Retrieval (mode-aware category filtering)
    # ------------------------------------------------------------------

    def _get_rag_examples(
        self, query_emb: np.ndarray, matched_categories: List[str]
    ) -> List[dict]:
        """
        Retrieve few-shot examples from the ExampleStore.

        For HIERARCHICAL_RAG: restricts examples to intents within matched
        categories — keeps evidence focused. Falls back to global retrieval
        if category filtering yields fewer than 3 examples.

        For FLAT / FLAT_RAG: retrieves globally (no category filter).
        """
        if self.example_store is None or self.mode == ClassificationMode.FLAT:
            return []

        all_examples = self.example_store.retrieve_from_embedding(
            query_emb, k=RAG_TOP_K_EXAMPLES
        )

        if not matched_categories or self.mode != ClassificationMode.HIERARCHICAL_RAG:
            return all_examples

        # Build set of allowed intent names from matched categories
        allowed_intents = set()
        for cat in matched_categories:
            cat_data = self._hierarchy.get(cat, {})
            allowed_intents.update(cat_data.get("intents", []))

        filtered = [e for e in all_examples if e["intent"] in allowed_intents]

        # If category filter leaves too few examples, supplement globally
        if len(filtered) < 3:
            logger.info(
                f"Category filter yielded only {len(filtered)} examples — "
                "supplementing with global retrieval."
            )
            seen_texts = {e["text"] for e in filtered}
            for ex in all_examples:
                if ex["text"] not in seen_texts:
                    filtered.append(ex)
                if len(filtered) >= RAG_TOP_K_EXAMPLES:
                    break

        return filtered

    # ------------------------------------------------------------------
    # Prompt Building (mode-aware)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        text: str,
        top_matches: List[dict],
        rag_examples: List[dict],
        matched_categories: List[str],
        conversation_history: Optional[List[dict]],
    ) -> str:
        sections = []

        # --- Conversation History (MULTI turn mode only) ---
        if self.turn_mode == TurnMode.MULTI and conversation_history:
            history_str = json.dumps(conversation_history, indent=2)
            sections.append(
                "CONVERSATION HISTORY:\n"
                + history_str
                + "\n\nCRITICAL: If the query is a follow-up, refinement, or continuation "
                "(e.g. providing a date, saying yes/no, 'what about...?'), classify it as the "
                "SAME intent as the previous turn. Only assign a NEW intent if the topic changes."
            )

        # --- Category Context (hierarchical mode only) ---
        if matched_categories and self.mode == ClassificationMode.HIERARCHICAL_RAG:
            cat_display = ", ".join(
                c.replace("_", " ").title() for c in matched_categories
            )
            sections.append(
                f"DOMAIN CONTEXT: This query was matched to the following domain(s): {cat_display}.\n"
                "All candidate intents below are within these domains."
            )

        # --- RAG Examples ---
        if rag_examples:
            example_lines = "\n".join(
                f'  {i+1}. "{ex["text"]}" → {ex["intent"]}  (similarity: {ex["score"]:.2f})'
                for i, ex in enumerate(rag_examples)
            )
            sections.append(
                "RETRIEVED EXAMPLES — similar past queries and their verified intents:\n"
                + example_lines
                + "\n\nUse these examples as your primary classification evidence."
            )
        elif self.mode != ClassificationMode.FLAT:
            sections.append(
                "No labeled examples available. Rely on intent descriptions and reasoning."
            )

        # --- Candidate Intents ---
        candidate_lines = "\n".join(
            f'  - {m["intent"]["name"]}: {m["intent"]["description"]}'
            for m in top_matches
        )
        sections.append("CANDIDATE INTENTS:\n" + candidate_lines)

        # --- Query ---
        sections.append(f'User Query: "{text}"')

        # --- Instructions (adapt by mode) ---
        if self.mode == ClassificationMode.FLAT:
            instructions = (
                "Select the single best matching intent from the candidates above.\n"
                "Use the intent descriptions to guide your decision."
            )
        elif self.mode == ClassificationMode.FLAT_RAG:
            instructions = (
                "Study the retrieved examples — they show how similar queries were classified before.\n"
                "If examples strongly agree on one intent, prefer that classification.\n"
                "If examples are mixed or absent, use the candidate descriptions to decide."
            )
        else:  # HIERARCHICAL_RAG
            instructions = (
                "You are working within a focused domain (see DOMAIN CONTEXT above).\n"
                "Study the retrieved examples as primary evidence.\n"
                "If examples strongly agree on one intent, prefer it.\n"
                "If mixed or absent, use the candidate descriptions within this domain to decide."
            )

        body = "\n\n".join(sections)

        return (
            f"You are an expert intent classifier.\n\n"
            f"{body}\n\n"
            f"{instructions}\n\n"
            "Return ONLY valid JSON with no extra text:\n"
            "{\n"
            '    "name": "intent_name",\n'
            '    "confidence": 0.0,\n'
            '    "reasoning": "one concise sentence"\n'
            "}"
        )

    def _build_verify_prompt(
        self,
        text: str,
        intent_name: str,
        rag_examples: List[dict],
        conversation_history: Optional[List[dict]],
    ) -> str:
        history_block = ""
        if conversation_history:
            history_str = json.dumps(conversation_history, indent=2)
            history_block = f"CONVERSATION HISTORY:\n{history_str}\n\n"

        example_context = ""
        if rag_examples:
            lines = "\n".join(
                f'  - "{ex["text"]}" → {ex["intent"]}' for ex in rag_examples[:3]
            )
            example_context = f"Reference examples:\n{lines}\n\n"

        return (
            f"Verify this intent classification:\n"
            f"{history_block}"
            f"{example_context}"
            f'User Query: "{text}"\n'
            f'Selected Intent: "{intent_name}"\n\n'
            "Does the query match this intent, either semantically or as a valid "
            "follow-up based on conversation history?\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '    "is_correct": true,\n'
            '    "better_intent": null,\n'
            '    "confidence_score": 0.0\n'
            "}"
        )

    # ------------------------------------------------------------------
    # Language Detection
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        if not self.enable_lang_detect or self.lang_model is None:
            return "unknown"
        try:
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.lang_model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            return self.lang_model.config.id2label[predicted_class_id]
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"

    # ------------------------------------------------------------------
    # JSON Extraction
    # ------------------------------------------------------------------

    def _extract_json(self, content: str) -> dict:
        clean = content
        if "```json" in content:
            clean = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            clean = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        for start_idx, char in enumerate(content):
            if char != "{":
                continue
            depth, end_idx = 0, -1
            for i, c in enumerate(content[start_idx:], start=start_idx):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                continue
            try:
                candidate = json.loads(content[start_idx: end_idx + 1])
                if isinstance(candidate, dict) and (
                    "name" in candidate or "is_correct" in candidate
                ):
                    return candidate
            except json.JSONDecodeError:
                continue

        raise ValueError(f"No valid JSON found in LLM response: {content[:200]}")

    # ------------------------------------------------------------------
    # LLM Client
    # ------------------------------------------------------------------

    def _get_client(self):
        import ollama
        headers = {}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        return ollama.AsyncClient(host=self.llm_base_url, headers=headers)

    # ------------------------------------------------------------------
    # Main Classification Entry Point
    # ------------------------------------------------------------------

    async def classify(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        conversation_history: Optional[List[dict]] = None,
        verify: bool = False,
    ) -> Tuple[str, float, str]:
        """
        Classify a user query. Behaviour adapts to the configured mode.

        Parameters
        ----------
        text : user input query
        candidate_labels : unused, kept for API compatibility
        conversation_history : list of {"role": str, "content": str} for multi-turn
        verify : run an optional second LLM verification pass (disabled by default)

        Returns
        -------
        (intent_name, confidence_score, language_code)
        """
        loop = asyncio.get_running_loop()

        # 1. Language detection in parallel
        future_lang = loop.run_in_executor(None, self.detect_language, text)

        is_multi = self.turn_mode == TurnMode.MULTI

        # 2. Extract prior intent + categories from history.
        #    Skipped entirely in SINGLE turn mode.
        prior_categories: List[str] = []
        if is_multi and conversation_history:
            prior_intent = self._extract_prior_intent(conversation_history)
            if prior_intent and self.hier_router:
                cat = self.hier_router.get_category_for_intent(prior_intent)
                if cat:
                    prior_categories = [cat]

        # 3. Encode the raw query for ROUTING.
        query_emb = await loop.run_in_executor(None, self.router.encode_query, text)

        # 4. Build and encode the RAG retrieval query.
        #    SINGLE turn mode: always uses raw query.
        #    MULTI turn mode: prepends recent user utterances so bare follow-ups
        #    like "for last 6 months" retrieve meaningful examples from the store.
        if is_multi and conversation_history and self.example_store is not None:
            rag_query = self._build_rag_query(text, conversation_history)
            if rag_query != text:
                logger.info(f"RAG query (context-enriched): '{rag_query}'")
                rag_emb = await loop.run_in_executor(None, self.router.encode_query, rag_query)
            else:
                rag_emb = query_emb
        else:
            rag_emb = query_emb

        # 5. Route using the raw query embedding.
        #    Prior categories pinned so continuation turns stay in the right domain.
        if query_emb is not None:
            matched_categories, top_matches = await loop.run_in_executor(
                None, self._route, query_emb, prior_categories
            )
        else:
            top_matches = await loop.run_in_executor(None, self.router.find_top_k, text, 5)
            matched_categories = []

        logger.info(
            f"[{self.mode.value}] Candidates: {[m['intent']['name'] for m in top_matches]}"
        )

        # 6. RAG retrieval using the context-enriched embedding.
        rag_examples: List[dict] = []
        if rag_emb is not None:
            rag_examples = await loop.run_in_executor(
                None, self._get_rag_examples, rag_emb, matched_categories
            )
            if rag_examples:
                logger.info(
                    f"RAG examples: "
                    + str([(e["intent"], round(e["score"], 2)) for e in rag_examples])
                )

        # 7. Build mode-aware prompt
        prompt = self._build_prompt(
            text, top_matches, rag_examples, matched_categories, conversation_history
        )

        # 8. LLM classification
        client = self._get_client()
        try:
            logger.info(f"LLM classify | query='{text}'")
            response = await client.chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"]
            logger.info(f"LLM response: {content}")
            result = self._extract_json(content)
        except Exception as e:
            logger.error(f"LLM inference failed: {e}. Falling back to top semantic match.")
            lang = await future_lang
            best = top_matches[0]
            return best["intent"]["name"], best["score"], lang

        intent_name = result.get("name", top_matches[0]["intent"]["name"])
        llm_confidence = float(result.get("confidence", 0.5))

        # 9. Confidence gate: low retrieval similarity signals uncharted territory
        if rag_examples and rag_examples[0]["score"] < 0.35:
            logger.warning(
                f"Low retrieval similarity ({rag_examples[0]['score']:.2f}) — "
                "capping confidence."
            )
            llm_confidence = min(llm_confidence, 0.55)

        # 10. Optional verification (opt-in, disabled by default)
        if verify:
            verify_prompt = self._build_verify_prompt(
                text, intent_name, rag_examples, conversation_history
            )
            try:
                logger.info("Running optional verification pass...")
                v_response = await client.chat(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": verify_prompt}],
                )
                v_result = self._extract_json(v_response["message"]["content"])
                logger.info(f"Verification: {v_result}")

                if not v_result.get("is_correct", True):
                    better = v_result.get("better_intent")
                    v_score = float(v_result.get("confidence_score", 0.0))
                    lang = await future_lang
                    return (better, v_score, lang) if better else ("general_irrelevant", 0.0, lang)
                else:
                    llm_confidence = float(v_result.get("confidence_score", llm_confidence))
            except Exception as e:
                logger.warning(f"Verification failed ({e}). Using original result.")

        lang = await future_lang
        logger.info(
            f"Result: intent='{intent_name}' | confidence={llm_confidence:.2f} | lang='{lang}'"
        )
        return intent_name, llm_confidence, lang
