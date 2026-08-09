"""
QwenReranker: LLM-based reranking of candidate intents.

Replaces IntentClassifier's generative "pick one and self-report a confidence"
classification call with a reranking pass over the (typically hybrid-retrieved)
candidates. Two strategies:

  listwise  — one prompt, all candidates + RAG evidence, model returns a
              relevance-scored ranking. Same call cost as the original
              classify() LLM call; only the prompt framing changes
              (ranking, not open-ended generation).

  pointwise — one prompt PER candidate intent ("does this query match this
              specific intent, 0-1"), argmax wins. More calls (N = number of
              candidates), but the confidence score is a genuine per-candidate
              judgement rather than a single self-reported number for the
              eventual winner only.

Both reuse the same ollama.AsyncClient pattern as IntentClassifier, and the
same <think>-stripping JSON extraction needed for reasoning models like Qwen3.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from query_classifier.config import (
    LLM_API_BASE, LLM_API_KEY, RERANKER_MODEL_NAME, RERANK_POINTWISE_CONCURRENCY,
)

logger = logging.getLogger(__name__)

_VALID_MODES = ("listwise", "pointwise")


def _extract_json(content: str) -> dict:
    """Same brace-scanning fallback as IntentClassifier._extract_json, kept
    standalone so this module has no dependency on nlp_engine."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

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
            return json.loads(content[start_idx: end_idx + 1])
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON found in reranker response: {content[:200]}")


class QwenReranker:
    """
    LLM reranker over candidate intents, served via Ollama (e.g. `qwen3:4b`).

    Parameters
    ----------
    model_name : reranker model name (default RERANKER_MODEL_NAME, "qwen3:4b").
                 Deliberately separate from IntentClassifier's llm_model_name —
                 the reranker is typically a smaller, cheaper model.
    base_url    : Ollama host. Defaults to LLM_API_BASE.
    api_key     : optional Bearer token. Defaults to LLM_API_KEY.
    pointwise_concurrency : max concurrent LLM calls during pointwise reranking.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        base_url: str = LLM_API_BASE,
        api_key: str = LLM_API_KEY,
        pointwise_concurrency: int = RERANK_POINTWISE_CONCURRENCY,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(pointwise_concurrency)

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    def _get_client(self):
        import ollama
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return ollama.AsyncClient(host=self.base_url, headers=headers)

    async def _chat(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    @staticmethod
    def _group_examples_by_intent(rag_examples: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for ex in rag_examples:
            grouped.setdefault(ex["intent"], []).append(ex)
        return grouped

    # ------------------------------------------------------------------
    # Listwise
    # ------------------------------------------------------------------

    def _build_listwise_prompt(
        self, text: str, candidates: List[Dict], examples_by_intent: Dict[str, List[Dict]]
    ) -> str:
        lines = []
        for i, c in enumerate(candidates):
            name = c["intent"]["name"]
            desc = c["intent"]["description"]
            examples = examples_by_intent.get(name, [])
            ex_str = "; ".join(f'"{e["text"]}"' for e in examples[:3]) or "(no examples)"
            lines.append(f'  [{i + 1}] {name}: {desc}\n      examples: {ex_str}')
        candidate_block = "\n".join(lines)

        return (
            "You are a reranking model for intent classification. Rank the candidate "
            "intents below by how well they match the user query, using the retrieved "
            "examples as evidence.\n\n"
            f'User Query: "{text}"\n\n'
            f"CANDIDATES:\n{candidate_block}\n\n"
            "Return ONLY valid JSON with no extra text:\n"
            "{\n"
            '  "ranking": [\n'
            '    {"name": "intent_name", "relevance": 0.0}\n'
            "  ]\n"
            "}\n"
            "List ALL candidates, most relevant first. relevance is 0-1."
        )

    async def rerank_listwise(
        self, text: str, candidates: List[Dict], rag_examples: List[Dict]
    ) -> Tuple[str, float, dict]:
        """
        Returns (winning_intent_name, confidence, raw_result). raw_result
        includes the full ranking, useful for logging/debugging.
        """
        if not candidates:
            raise ValueError("rerank_listwise called with no candidates.")

        examples_by_intent = self._group_examples_by_intent(rag_examples)
        prompt = self._build_listwise_prompt(text, candidates, examples_by_intent)
        content = await self._chat(prompt)
        result = _extract_json(content)

        ranking = result.get("ranking") or []
        if not ranking:
            raise ValueError("Reranker returned an empty ranking.")

        valid_names = {c["intent"]["name"] for c in candidates}
        filtered = [r for r in ranking if r.get("name") in valid_names]
        ranking = filtered or ranking

        top = max(ranking, key=lambda r: float(r.get("relevance", 0.0)))
        return top["name"], float(top.get("relevance", 0.5)), result

    # ------------------------------------------------------------------
    # Pointwise
    # ------------------------------------------------------------------

    def _build_pointwise_prompt(self, text: str, candidate: Dict, examples: List[Dict]) -> str:
        name = candidate["intent"]["name"]
        desc = candidate["intent"]["description"]
        ex_str = "\n".join(f'  - "{e["text"]}"' for e in examples[:5]) or "  (no examples)"

        return (
            "You are a reranking model. Judge whether the user query matches ONE "
            "specific candidate intent.\n\n"
            f'User Query: "{text}"\n\n'
            f'Candidate Intent: "{name}" — {desc}\n'
            f"Reference examples for this intent:\n{ex_str}\n\n"
            "Return ONLY valid JSON with no extra text:\n"
            '{"relevance": 0.0}\n'
            "relevance is 0-1: how well the query matches this SPECIFIC intent."
        )

    async def _score_one(
        self, text: str, candidate: Dict, examples_by_intent: Dict[str, List[Dict]]
    ) -> Tuple[str, float]:
        name = candidate["intent"]["name"]
        examples = examples_by_intent.get(name, [])
        prompt = self._build_pointwise_prompt(text, candidate, examples)
        async with self._semaphore:
            try:
                content = await self._chat(prompt)
                result = _extract_json(content)
                score = float(result.get("relevance", 0.0))
            except Exception as e:
                logger.warning(f"Pointwise rerank failed for '{name}': {e}. Scoring 0.0.")
                score = 0.0
        return name, max(0.0, min(1.0, score))

    async def rerank_pointwise(
        self, text: str, candidates: List[Dict], rag_examples: List[Dict]
    ) -> Tuple[str, float, dict]:
        """
        Returns (winning_intent_name, confidence, raw_result). raw_result =
        {"scores": {intent_name: score, ...}} for logging/debugging.
        """
        if not candidates:
            raise ValueError("rerank_pointwise called with no candidates.")

        examples_by_intent = self._group_examples_by_intent(rag_examples)
        pairs = await asyncio.gather(
            *(self._score_one(text, c, examples_by_intent) for c in candidates)
        )
        scores = dict(pairs)

        best_name = max(scores, key=scores.get)
        return best_name, scores[best_name], {"scores": scores}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def rerank(
        self, mode: str, text: str, candidates: List[Dict], rag_examples: List[Dict]
    ) -> Tuple[str, float, dict]:
        if mode == "pointwise":
            return await self.rerank_pointwise(text, candidates, rag_examples)
        elif mode == "listwise":
            return await self.rerank_listwise(text, candidates, rag_examples)
        raise ValueError(f"Unknown rerank mode: {mode!r}. Use one of {_VALID_MODES}.")
