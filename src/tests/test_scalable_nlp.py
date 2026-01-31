
import logging
from src.nlp_engine import IntentClassifier
from src.intents_db import INTENTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScalableNLP")

def test_nlp():
    logger.info("Initializing NLP Engine...")
    nlp = IntentClassifier()
    
    test_queries = [
        "I need to reset my password because I forgot it",
        "How much does the enterprise plan cost?",
        "My bill is wrong, I was overcharged",
        "The api isn't responding with 200 OK",
        "I want to cancel my subscription immediately"
    ]
    
    for query in test_queries:
        print("-" * 50)
        logger.info(f"Query: {query}")
        intent, conf, lang = nlp.classify(query)
        logger.info(f"Result: {intent} (Conf: {conf})")

if __name__ == "__main__":
    test_nlp()
