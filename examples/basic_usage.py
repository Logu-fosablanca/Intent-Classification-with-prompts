
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query_classifier.nlp_engine import IntentClassifier
from banking_intents import INTENTS

async def main():
    # 1. Initialize with your custom intents
    # You can also pass explicit config: llm_model_name="llama3", embedding_model="..."
    print("Initializing Classifier...")
    nlp = IntentClassifier(
        intents=INTENTS,
        llm_model_name="llama3",  # Explicitly link model
        llm_base_url="http://localhost:11434", # Explicitly link provider URL
        # llm_api_key="sk-..." # Optional: Start with authentication
    )
    
    # 2. Define queries to test
    queries = [
        "I lost my card, can you block it?",
        "What is the status of my loan application?",
        "I need to talk to a human agent"
    ]
    
    # 3. Classify
    print(f"\n{'Query':<40} | {'Intent':<25} | {'Score':<5} | {'Lang':<5}")
    print("-" * 85)
    
    for text in queries:
        label, score, lang = await nlp.classify(text)
        print(f"{text:<40} | {label:<25} | {score:.2f}  | {lang:<5}")

if __name__ == "__main__":
    asyncio.run(main())
