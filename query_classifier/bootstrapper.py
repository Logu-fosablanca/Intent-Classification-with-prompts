"""
ExampleStoreBootstrapper: automated example generation for the ExampleStore.

Uses the configured LLM to synthesize labeled examples — both single-turn and
multi-turn conversation follow-ups — so the ExampleStore can be populated
without any manual labeling.

Two generation modes
--------------------
1. bootstrap_single_turn()
   Generates standalone utterances for each intent from its description.
   Solves the cold-start problem — no real data needed at all.

2. bootstrap_multi_turn()
   Generates follow-up utterances paired with conversation history context.
   Solves the retrieval gap for bare follow-ups like "for last 6 months"
   which have no semantic signal on their own.

Typical usage
-------------
    bootstrapper = ExampleStoreBootstrapper(
        llm_base_url="http://localhost:11434",
        llm_model_name="llama3",
    )
    store = ExampleStore(encoder=router.model)
    await bootstrapper.run_full_bootstrap(
        store=store,
        intents=INTENTS,
        save_path="my_store",
        n_single=8,
        n_multi=6,
    )

The generated examples are saved to disk and can be loaded on every subsequent
startup — bootstrap only needs to run once (or when new intents are added).
"""

import json
import logging
import asyncio
from typing import List, Dict, Optional

from query_classifier.config import (
    LLM_MODEL_NAME, LLM_API_BASE, LLM_API_KEY,
    BOOTSTRAP_N_SINGLE, BOOTSTRAP_N_MULTI,
    BOOTSTRAP_CONCURRENCY, BOOTSTRAP_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SINGLE_TURN_PROMPT = """You are building training data for a banking intent classifier.

Intent name: "{intent_name}"
Intent description: "{intent_description}"

Generate {n} realistic, DIVERSE user utterances that belong to this intent.
Rules:
- Each utterance must be something a real banking customer would actually say.
- Vary vocabulary, phrasing, formality, and length.
- Include at least one non-English utterance if plausible for a global bank.
- Do NOT reuse the description verbatim.
- Do NOT include the intent name in the utterances.

Return ONLY a valid JSON array of strings, no extra text:
["utterance 1", "utterance 2", ...]"""


_MULTI_TURN_PROMPT = """You are building conversational training data for a banking intent classifier.

Intent name: "{intent_name}"
Intent description: "{intent_description}"
Example single-turn queries for this intent:
{samples}

Generate {n} realistic multi-turn conversation examples for this intent.
Each example simulates:
  1. The user making an initial request related to this intent.
  2. The assistant asking ONE short clarifying question.
  3. The user giving a short follow-up answer that is AMBIGUOUS on its own
     but clearly belongs to this intent given the conversation context.

Rules:
- "history_context" = what the USER said initially (not the assistant question).
  It should be a realistic first utterance for this intent.
- "user_followup" = the user's response to clarification — keep it SHORT (2-8 words).
  It must be semantically ambiguous without the context (e.g. "for last 6 months",
  "yes block it", "the one ending in 4521").
- Vary the scenarios: different clarification types, phrasings, levels of detail.
- Do NOT make the followup so generic it could belong to any intent.

Return ONLY a valid JSON array, no extra text:
[
  {{"history_context": "...", "user_followup": "..."}},
  ...
]"""


# ---------------------------------------------------------------------------
# Bootstrapper
# ---------------------------------------------------------------------------

class ExampleStoreBootstrapper:
    """
    Automated LLM-based example generator for the ExampleStore.

    Parameters
    ----------
    llm_model_name : LLM model to use for generation (same as classifier).
    llm_base_url   : LLM API base URL.
    llm_api_key    : Optional Bearer token.
    max_retries    : How many times to retry a failed LLM call per intent.
    concurrency    : Max concurrent LLM calls (controls API load).
    """

    def __init__(
        self,
        llm_model_name: str = LLM_MODEL_NAME,
        llm_base_url: str = LLM_API_BASE,
        llm_api_key: str = LLM_API_KEY,
        max_retries: int = BOOTSTRAP_MAX_RETRIES,
        concurrency: int = BOOTSTRAP_CONCURRENCY,
    ):
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Internal LLM helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        import ollama
        headers = {}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        return ollama.AsyncClient(host=self.llm_base_url, headers=headers)

    async def _call_llm(self, prompt: str) -> str:
        """Single LLM call, returns raw content string."""
        async with self._semaphore:
            client = self._get_client()
            response = await client.chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]

    def _parse_json(self, content: str):
        """Robustly extract JSON from LLM response."""
        clean = content
        if "```json" in content:
            clean = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            clean = content.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            start = clean.find("[")
            end = clean.rfind("]")
            if start != -1 and end != -1:
                return json.loads(clean[start: end + 1])
            raise

    # ------------------------------------------------------------------
    # Single-turn generation
    # ------------------------------------------------------------------

    async def _generate_single_turn(
        self, intent: dict, n: int
    ) -> List[Dict]:
        """
        Generate n single-turn utterances for one intent.
        Returns list of {"text": str, "intent": str}.
        """
        prompt = _SINGLE_TURN_PROMPT.format(
            intent_name=intent["name"],
            intent_description=intent["description"],
            n=n,
        )

        for attempt in range(self.max_retries + 1):
            try:
                content = await self._call_llm(prompt)
                utterances = self._parse_json(content)

                if not isinstance(utterances, list):
                    raise ValueError("Expected a JSON array.")

                results = [
                    {"text": str(u).strip(), "intent": intent["name"]}
                    for u in utterances
                    if u and str(u).strip()
                ]
                logger.info(
                    f"[single-turn] {intent['name']}: generated {len(results)} examples."
                )
                return results

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"[single-turn] {intent['name']} attempt {attempt+1} failed: {e}. Retrying..."
                    )
                else:
                    logger.error(
                        f"[single-turn] {intent['name']}: all attempts failed. Skipping."
                    )
        return []

    async def bootstrap_single_turn(
        self,
        store,
        intents: List[dict],
        n_per_intent: int = BOOTSTRAP_N_SINGLE,
    ) -> int:
        """
        Generate single-turn examples for all intents and add to the store.

        Parameters
        ----------
        store          : ExampleStore to populate.
        intents        : Full intent list (same format as IntentClassifier).
        n_per_intent   : Number of utterances to generate per intent.

        Returns
        -------
        Total number of examples successfully added.
        """
        logger.info(
            f"Bootstrapping single-turn examples: {len(intents)} intents × {n_per_intent} each..."
        )

        tasks = [self._generate_single_turn(intent, n_per_intent) for intent in intents]
        results = await asyncio.gather(*tasks)

        all_examples = [ex for batch in results for ex in batch]
        store.add_examples_bulk(all_examples)

        logger.info(f"Single-turn bootstrap complete: {len(all_examples)} examples added.")
        return len(all_examples)

    # ------------------------------------------------------------------
    # Multi-turn generation
    # ------------------------------------------------------------------

    async def _generate_multi_turn(
        self, intent: dict, store, n: int
    ) -> List[Dict]:
        """
        Generate n multi-turn follow-up examples for one intent.
        Uses existing single-turn examples from the store as samples.
        Returns list of {"text", "intent", "history_context"}.
        """
        # Pull sample single-turn examples from store for this intent
        # so the LLM sees what realistic utterances look like
        samples = []
        if store is not None:
            for idx, (text, stored_intent, ctx) in enumerate(
                zip(store._texts, store._intents, store._history_contexts)
            ):
                if stored_intent == intent["name"] and ctx is None:
                    samples.append(f'"{text}"')
                if len(samples) >= 4:
                    break

        samples_str = "\n".join(samples) if samples else "(none yet — use the description)"

        prompt = _MULTI_TURN_PROMPT.format(
            intent_name=intent["name"],
            intent_description=intent["description"],
            samples=samples_str,
            n=n,
        )

        for attempt in range(self.max_retries + 1):
            try:
                content = await self._call_llm(prompt)
                parsed = self._parse_json(content)

                if not isinstance(parsed, list):
                    raise ValueError("Expected a JSON array.")

                results = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    ctx = str(item.get("history_context", "")).strip()
                    followup = str(item.get("user_followup", "")).strip()
                    if ctx and followup:
                        results.append({
                            "text": followup,
                            "intent": intent["name"],
                            "history_context": ctx,
                        })

                logger.info(
                    f"[multi-turn] {intent['name']}: generated {len(results)} examples."
                )
                return results

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"[multi-turn] {intent['name']} attempt {attempt+1} failed: {e}. Retrying..."
                    )
                else:
                    logger.error(
                        f"[multi-turn] {intent['name']}: all attempts failed. Skipping."
                    )
        return []

    async def bootstrap_multi_turn(
        self,
        store,
        intents: List[dict],
        n_per_intent: int = BOOTSTRAP_N_MULTI,
    ) -> int:
        """
        Generate multi-turn follow-up examples for all intents and add to store.

        Runs AFTER bootstrap_single_turn so the LLM can reference real examples
        when generating conversation patterns.

        Parameters
        ----------
        store          : ExampleStore (should already have single-turn examples).
        intents        : Full intent list.
        n_per_intent   : Number of follow-up examples to generate per intent.

        Returns
        -------
        Total number of multi-turn examples successfully added.
        """
        logger.info(
            f"Bootstrapping multi-turn examples: {len(intents)} intents × {n_per_intent} each..."
        )

        tasks = [self._generate_multi_turn(intent, store, n_per_intent) for intent in intents]
        results = await asyncio.gather(*tasks)

        all_examples = [ex for batch in results for ex in batch]
        store.add_examples_bulk(all_examples)

        logger.info(f"Multi-turn bootstrap complete: {len(all_examples)} examples added.")
        return len(all_examples)

    # ------------------------------------------------------------------
    # Full bootstrap
    # ------------------------------------------------------------------

    async def run_full_bootstrap(
        self,
        store,
        intents: List[dict],
        save_path: Optional[str] = None,
        n_single: int = BOOTSTRAP_N_SINGLE,
        n_multi: int = BOOTSTRAP_N_MULTI,
    ) -> dict:
        """
        Run single-turn + multi-turn bootstrap in sequence, then save to disk.

        Single-turn runs first so multi-turn generation can reference those
        examples when crafting realistic conversation patterns.

        Parameters
        ----------
        store      : ExampleStore to populate (may already have seed examples).
        intents    : Full intent list.
        save_path  : Path prefix to save the store (e.g. "my_store" saves
                     my_store.json + my_store.npy). Pass None to skip saving.
        n_single   : Single-turn examples per intent.
        n_multi    : Multi-turn examples per intent.

        Returns
        -------
        Summary dict with counts of examples generated.
        """
        n_s = await self.bootstrap_single_turn(store, intents, n_per_intent=n_single)
        n_m = await self.bootstrap_multi_turn(store, intents, n_per_intent=n_multi)

        if save_path:
            store.save(save_path)
            logger.info(f"Store saved to {save_path}.json + {save_path}.npy")

        summary = {
            "total": len(store),
            "single_turn_generated": n_s,
            "multi_turn_generated": n_m,
            "intents_covered": len(store.intent_coverage()),
            "multi_turn_coverage": store.multi_turn_coverage(),
        }
        logger.info(f"Bootstrap summary: {summary}")
        return summary
