import asyncio
import logging
import sys
import os



from query_classifier.simple_agent import SimpleAgent
from query_classifier.nlp_engine import IntentClassifier
from banking_intents import INTENTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("System")

# Initialize NLP Engine
nlp = IntentClassifier(intents=INTENTS)
candidate_labels = ["sales", "support"]

# Define Agents
# Define Agents (Using new ports to avoid conflicts)
router = SimpleAgent("router", 8010)
sales_agent = SimpleAgent("sales", 8011)
support_agent = SimpleAgent("support", 8012)

# Router Logic
@router.on_message("UserRequest")
async def handle_user_request(sender, payload):
    text = payload.get("text")
    logger.info(f"Router processing: {text}")
    
    # Classify
    label, score, lang = await nlp.classify(text, candidate_labels)
    logger.info(f"Classification: {label} ({score:.2f})")
    
    # Dispatch
    msg_payload = {"original_text": text, "intent": label, "confidence": score}
    
    if label == "sales":
        await router.send("http://localhost:8011/submit", "TaskDispatch", msg_payload)
    elif label == "support":
        await router.send("http://localhost:8012/submit", "TaskDispatch", msg_payload)

# Worker Logic
@sales_agent.on_message("TaskDispatch")
async def sales_handler(sender, payload):
    logger.info(f"Sales Agent working on: {payload['original_text']}")
    # Simulate response back to router or user? 
    # For A2A, maybe report back to router.
    await sales_agent.send("http://localhost:8010/submit", "TaskResponse", {"status": "completed", "message": "Sales info sent"})

@support_agent.on_message("TaskDispatch")
async def support_handler(sender, payload):
    logger.info(f"Support Agent working on: {payload['original_text']}")
    await support_agent.send("http://localhost:8010/submit", "TaskResponse", {"status": "completed", "message": "Ticket created"})

@router.on_message("TaskResponse")
async def router_handle_response(sender, payload):
    logger.info(f"Router received completion from {sender}: {payload}")

async def main():
    await router.start()
    await sales_agent.start()
    await support_agent.start()
    
    logger.info("All agents running. Press Ctrl+C to stop.")
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
