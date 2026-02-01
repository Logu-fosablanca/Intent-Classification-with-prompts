import asyncio
import os
from query_classifier import IntentClassifier

# Define simple banking intents
intents = [
    {"name": "account_statement", "description": "User wants to download or view their bank statement."},
    {"name": "check_balance", "description": "User wants to check their current account balance."},
    {"name": "credit_card_apply", "description": "User wants to apply for a new credit card."}
]

async def main():
    print("--- Context-Aware Classification Demo ---")
    classifier = IntentClassifier(intents=intents, llm_model_name="llama3")

    history = []
    
    # 1. First Query
    q1 = "I want account statement"
    print(f"\nUser: {q1}")
    intent1, conf1, _ = await classifier.classify(q1, conversation_history=history)
    print(f"Bot: Identified intent '{intent1}'")
    
    # Update history
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content":"please give date","intent_classified": intent1})
    
    # 2. Follow-up Query (Ambiguous)
    q2 = "for last 6 months" 
    print(f"\nUser: {q2}")
    print("(Sending with conversation history...)")
    
    intent2, conf2, _ = await classifier.classify(q2, conversation_history=history)
    print(f"Bot: Identified intent '{intent2}' (Confidence: {conf2})")
    
    if intent2 == "account_statement":
        print("\nSUCCESS: The bot correctly understood 'for last 6 months' relies on 'account statement'.")
    else:
        print(f"\nRESULT: The bot classified it as '{intent2}'.")

if __name__ == "__main__":
    asyncio.run(main())
