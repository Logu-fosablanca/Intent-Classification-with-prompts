
import logging
import asyncio
import json
import ollama
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from query_classifier.semantic_router import SemanticRouter
from query_classifier.config import LLM_MODEL_NAME, LANG_DETECT_MODEL, LLM_PROVIDER, LLM_API_BASE, LLM_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, intents: list, 
                 llm_provider=LLM_PROVIDER,
                 llm_model_name=LLM_MODEL_NAME,
                 llm_base_url=LLM_API_BASE,
                 llm_api_key=LLM_API_KEY,
                 embedding_model=None,
                 lang_detect_model=LANG_DETECT_MODEL):
        
        logger.info("Initializing Intent Classifier...")
        
        # Configuration
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.embedding_model = embedding_model
        
        # Initialize Router
        # Pass embedding_model if provided, otherwise let Router use its default
        if embedding_model:
            self.router = SemanticRouter(intents=intents, model_name=embedding_model)
        else:
            self.router = SemanticRouter(intents=intents)
        
        # Load Language Detection Model
        logger.info(f"Loading Language Detection Model ({lang_detect_model})...")
        self.tokenizer = AutoTokenizer.from_pretrained(lang_detect_model)
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(lang_detect_model)
    
    def detect_language(self, text: str) -> str:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.lang_model(**inputs).logits
            
            predicted_class_id = logits.argmax().item()
            language = self.lang_model.config.id2label[predicted_class_id]
            return language
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"

    def _extract_json(self, content):
        """Helper to robustly extract a JSON object containing 'name' or 'is_correct' from text."""
        try:
            # 1. Try cleaning markdown first
            clean_content = content
            if "```json" in content:
                clean_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                clean_content = content.split("```")[1].split("```")[0].strip()
            
            try:
                return json.loads(clean_content)
            except:
                pass # Continue to brace scanning
            
            # 2. Search for all potential JSON objects
            start_indices = [i for i, char in enumerate(content) if char == '{']
            
            for start_idx in start_indices:
                try:
                    depth = 0
                    end_idx = -1
                    for i, char in enumerate(content[start_idx:], start=start_idx):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end_idx = i
                                break
                    
                    if end_idx != -1:
                        candidate_str = content[start_idx:end_idx+1]
                        candidate_json = json.loads(candidate_str)
                        if isinstance(candidate_json, dict):
                             # Accept if it looks like a classification result OR verification result
                             if 'name' in candidate_json or 'is_correct' in candidate_json:
                                return candidate_json
                except:
                    continue
            
            raise ValueError("No valid JSON object found")

        except Exception as e:
            # logger.error(f"JSON extraction error: {e}") 
            raise e

    async def classify(self, text: str, candidate_labels: list = None, conversation_history: list = None):
        loop = asyncio.get_running_loop()
        
        # 1. Start Language Detection (Independent Task)
        # Run in executor as it might be CPU bound with local model
        future_lang = loop.run_in_executor(None, self.detect_language, text)

        # 2. Semantic Retrieval (needed for LLM)
        # Also run in executor
        top_matches = await loop.run_in_executor(None, self.router.find_top_k, text, 5)
        
        candidates_str = json.dumps([m['intent'] for m in top_matches], indent=2)
        logger.info(f"Top 5 Semantic Matches: {[m['intent']['name'] for m in top_matches]}")

        # 3. LLM Classification (Reasoning)
        history_context = ""
        if conversation_history:
             history_str = json.dumps(conversation_history, indent=2)
             history_context = (
                 f"\nConversation History:\n{history_str}\n\n"
                 "CRITICAL INSTRUCTION: Analyze if the 'User Query' is a follow-up, refinement, or continuation of the previous intent in the history.\n"
                 "- IF it is a continuation (e.g., providing dates, saying 'yes/no', asking 'what about...?'), YOU MUST CLASSIFY IT AS THE SAME INTENT AS THE PREVIOUS TURN.\n"
                 "- Only choose a NEW intent if the user clearly changes the topic."
             )

        prompt = f"""
        You are an intelligent intent classifier.
        {history_context}
        User Query: "{text}"
        
        Below are the top 5 possible semantic matches for this specific query (ignoring context):
        {candidates_str}
        
        Task:
        1. Check the Conversation History (if exists).
        2. Decide if this is a CONTINUATION of the active intent or a NEW request.
        3. Select the BEST intent.
        
        Return ONLY a JSON object:
        {{
            "name": "intent_name",
            "confidence": 0.95,
            "reasoning": "Explain why it is a continuation or a new intent."
        }}
        """
        
        try:
            # Use AsyncClient for non-blocking IO
            # Use AsyncClient for non-blocking IO
            if self.llm_provider == "ollama":
                headers = {}
                if self.llm_api_key:
                    headers['Authorization'] = f"Bearer {self.llm_api_key}"
                client = ollama.AsyncClient(host=self.llm_base_url, headers=headers)
            else:
                logger.warning(f"Provider '{self.llm_provider}' not explicitly handled. Defaulting to Ollama logic with base url: {self.llm_base_url}")
                headers = {}
                if self.llm_api_key:
                    headers['Authorization'] = f"Bearer {self.llm_api_key}"
                client = ollama.AsyncClient(host=self.llm_base_url, headers=headers)
                
                client = ollama.AsyncClient(host=self.llm_base_url, headers=headers)
                
            logger.info(f"Sending LLM Reasoning Request for query: '{text}'")
            response = await client.chat(model=self.llm_model_name, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            logger.info(f"LLM Response: {content}")
            
            try:
                result = self._extract_json(content)
            except Exception as e:
                 logger.error(f"Failed to extract JSON from content: {content[:100]}... Error: {e}")
                 raise e
            
            # 4. Verification (Simple Logic)
            verify_prompt = f"""
            Verify this intent classification:
            {history_context}
            User Query: "{text}"
            Selected Intent: "{result['name']}"
            
            Task:
            Check if the query matches the intent, EITHER semantically OR as a valid follow-up/parameter refinement based on the history.
            
            Return JSON:
            {{
                "is_correct": true/false,
                "better_intent": "alternative_intent_name_if_false_or_null",
                "confidence_score": 0.0_to_1.0
            }}
            """
            
            try:
                logger.info("Sending LLM Verification Request...")
                verify_response = await client.chat(model=self.llm_model_name, messages=[{'role': 'user', 'content': verify_prompt}])
                verify_content = verify_response['message']['content']
                logger.info(f"Verification Response: {verify_content}")
                
                verify_result = self._extract_json(verify_content)
                logger.info(f"Verification Result: {verify_result}")
                
                if not verify_result['is_correct']:
                    logger.warning(f"Verification rejected intent {result['name']}. Suggestion: {verify_result.get('better_intent')}")
                    # Await language detection before returning
                    lang = await future_lang
                    logger.info(f"Detected language: {lang}")
                    
                    # If rejected, maybe fallback to General Irrelevant or the suggestion if valid
                    if verify_result.get('better_intent'):
                         return verify_result['better_intent'], verify_result['confidence_score'], lang
                    else:
                        return "general_irrelevant", 0.0, lang
                else:
                    # Confirmed
                    lang = await future_lang
                    logger.info(f"Detected language: {lang}")
                    return result['name'], verify_result['confidence_score'], lang

            except Exception as e:
                logger.warning(f"Verification step failed ({e}). Returning original result.")
                lang = await future_lang
                return result['name'], result['confidence'], lang

            lang = await future_lang
            return result['name'], result['confidence'], lang

        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            # Fallback to top semantic match
            best_match = top_matches[0]
            lang = await future_lang
            return best_match['intent']['name'], best_match['score'], lang
