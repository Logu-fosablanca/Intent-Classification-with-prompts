import logging
import asyncio
from src.simple_agent import SimpleAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User Agent
user = SimpleAgent("user", 8005)

async def main():
    # We don't necessarily need to start the user server if we just want to send, 
    # but send() is an instance method using ClientSession.
    
    logger.info("Sending request to Router...")
    await user.send("http://localhost:8000/submit", "UserRequest", {"text": "My account is locked"})
    
    # Give it a moment to process
    await asyncio.sleep(2)
    
    logger.info("Sending second request...")
    await user.send("http://localhost:8000/submit", "UserRequest", {"text": "I want to buy enterprise plan"})

if __name__ == "__main__":
    asyncio.run(main())
