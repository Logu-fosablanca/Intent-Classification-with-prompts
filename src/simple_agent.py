import asyncio
import logging
from aiohttp import web, ClientSession

logger = logging.getLogger(__name__)

class SimpleAgent:
    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.handlers = {} # msg_type -> handler_func

        # Default endpoint for messages
        self.routes.post("/submit")(self.handle_request)
        self.app.add_routes(self.routes)
    
    def on_message(self, message_type_name):
        def decorator(func):
            self.handlers[message_type_name] = func
            return func
        return decorator

    async def handle_request(self, request):
        try:
            data = await request.json()
            msg_type = data.get("type")
            payload = data.get("payload")
            sender = data.get("sender", "unknown")
            
            if msg_type in self.handlers:
                # Call handler: handler(sender, payload)
                logger.info(f"[{self.name}] Received {msg_type} from {sender}")
                asyncio.create_task(self.handlers[msg_type](sender, payload))
                return web.json_response({"status": "received"})
            else:
                logger.warning(f"[{self.name}] No handler for {msg_type}")
                return web.json_response({"status": "ignored", "reason": "no handler"}, status=400)
        except Exception as e:
            logger.error(f"[{self.name}] Error handling request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def send(self, target_url, msg_type, payload):
        async with ClientSession() as session:
            msg = {
                "type": msg_type,
                "payload": payload,
                "sender": self.name
            }
            try:
                async with session.post(target_url, json=msg) as resp:
                    logger.info(f"[{self.name}] Sent {msg_type} to {target_url} - Status: {resp.status}")
                    return await resp.json()
            except Exception as e:
                logger.error(f"[{self.name}] Failed to send to {target_url}: {e}")

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        logger.info(f"Agent '{self.name}' listening on http://localhost:{self.port}")
        # Keep running
        # await asyncio.Event().wait() 
