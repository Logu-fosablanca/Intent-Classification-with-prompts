
import asyncio
import logging
from src.nlp_engine import IntentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_async_classification():
    print("Initializing Async Classifier...")
    nlp = IntentClassifier()
    
    text = "I lost my credit card, please block it."
    print(f"\nTesting with query: '{text}'")
    
    # We expect this to run with Async Ollama client now
    label, score, lang = await nlp.classify(text)
    
    print(f"\n--- Result ---")
    print(f"Intent: {label}")
    print(f"Confidence: {score}")
    print(f"Language: {lang}")

if __name__ == "__main__":
    asyncio.run(test_async_classification())
