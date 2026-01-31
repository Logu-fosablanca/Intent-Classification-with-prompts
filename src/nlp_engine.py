
import logging
import asyncio
import json
import ollama
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.semantic_router import SemanticRouter
from src.config import LLM_MODEL_NAME, LANG_DETECT_MODEL, OLLAMA_BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        logger.info("Initializing Intent Classifier (Ollama + Semantic Router)...")
        self.router = SemanticRouter()
        self.ollama_model = LLM_MODEL_NAME
        
        # Load Language Detection Model
        logger.info(f"Loading Language Detection Model ({LANG_DETECT_MODEL})...")
        self.tokenizer = AutoTokenizer.from_pretrained(LANG_DETECT_MODEL)
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(LANG_DETECT_MODEL)
    
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

    async def classify(self, text: str, candidate_labels: list = None):
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
        prompt = f"""
        You are an intelligent intent classifier.
        User Query: "{text}"
        
        Below are the top 5 possible intents matching this query:
        {candidates_str}
        
        Analyze the query and select the BEST matching intent from the list.
        If none match well, choose 'general_irrelevant'.
        
        Return ONLY a JSON object with this format:
        {{
            "name": "intent_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation"
        }}
        """
        
        try:
            # Use AsyncClient for non-blocking IO
            client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
            response = await client.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            
            # Simple cleanup for JSON parsing
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                

            result = json.loads(content)
            
            # 4. Verification (Simple Logic)
            verify_prompt = f"""
            Verify this intent classification:
            User Query: "{text}"
            Selected Intent: "{result['name']}"
            
            Does this query functionally match the intent described?
            Return JSON:
            {{
                "is_correct": true/false,
                "better_intent": "alternative_intent_name_if_false_or_null",
                "confidence_score": 0.0_to_1.0
            }}
            """
            
            try:
                verify_response = await client.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': verify_prompt}])
                verify_content = verify_response['message']['content']
                if "```json" in verify_content:
                    verify_content = verify_content.split("```json")[1].split("```")[0].strip()
                elif "```" in verify_content:
                    verify_content = verify_content.split("```")[1].split("```")[0].strip()
                
                verify_result = json.loads(verify_content)
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
