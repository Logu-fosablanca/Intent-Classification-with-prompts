from uagents import Model

class UserRequest(Model):
    text: str

class TaskDispatch(Model):
    original_text: str
    intent: str
    confidence: float

class TaskResponse(Model):
    status: str
    message: str
