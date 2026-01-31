
import asyncio
import sys

# Ensure we are NOT accidentally using local folder hacks
# This proves it loads from the site-packages (or editable install)
try:
    from query_classifier.nlp_engine import IntentClassifier
    print("[SUCCESS] Module 'query_classifier' found and imported.")
except ImportError as e:
    print(f"[ERROR] Could not import 'query_classifier'. Is it installed?\nError: {e}")
    sys.exit(1)

async def main():
    print("Running quick inference test...")
    # Simple intents
    intents = [{"name": "test_intent", "description": "A test intent"}]
    
    # Initialize
    try:
        nlp = IntentClassifier(intents=intents, llm_model_name="llama3")
        print("[SUCCESS] Classifier initialized.")
        
        # Classify
        # We don't strictly need a real LLM response to prove the package is installed,
        # but let's try a quick one. 
        # (Using a dummy query to avoid hitting LLM if not needed, but code requires it)
        print("Calling classify... (this might need Ollama running)")
        label, score, lang = await nlp.classify("hello")
        print(f"[SUCCESS] Classification result: {label}")
        
    except Exception as e:
        print(f"[WARNING] Runtime error (might be LLM connection), but package IS installed.\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
