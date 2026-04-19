__version__ = "0.2.0"

from .nlp_engine import IntentClassifier, ClassificationMode, TurnMode
from .example_store import ExampleStore
from .bootstrapper import ExampleStoreBootstrapper
from .encoder import BaseEncoder, get_encoder, PROVIDER_REGISTRY
from .semantic_router import SemanticRouter

# SimpleAgent is an optional aiohttp-based utility unrelated to the REIC pipeline.
try:
    from .simple_agent import SimpleAgent
except ImportError:
    SimpleAgent = None  # type: ignore

__all__ = [
    # Core
    "IntentClassifier",
    "ClassificationMode",
    "TurnMode",
    "ExampleStore",
    "ExampleStoreBootstrapper",
    "SemanticRouter",
    # Encoder
    "BaseEncoder",
    "get_encoder",
    "PROVIDER_REGISTRY",
    # Version
    "__version__",
    # Optional
    "SimpleAgent",
]
