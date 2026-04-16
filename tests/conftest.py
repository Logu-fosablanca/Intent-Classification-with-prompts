"""
Shared fixtures and mock utilities for the test suite.

MockEncoder — a SentenceTransformer-compatible encoder that returns
deterministic, semantically meaningful embeddings based on keyword overlap.
No real model downloads required. Tests run fast and offline.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Vocabulary used by MockEncoder
# Each dimension corresponds to one keyword.
# ---------------------------------------------------------------------------
_VOCAB = [
    "balance", "statement", "transfer", "card", "loan", "support",
    "block", "pin", "account", "transaction", "month", "last",
    "money", "bank", "credit", "mortgage", "atm", "branch",
    "history", "payment", "apply", "repay", "score", "app", "security",
]


class MockEncoder:
    """
    Deterministic bag-of-words encoder for testing.

    Embeddings are built from keyword presence so cosine similarity
    reflects actual semantic overlap:
      - "check my balance"  ≈  "what is my balance"   (both hit 'balance')
      - "transfer $500"     ≈  "send money"            (both hit 'transfer'/'money')
      - "for last 6 months" alone → near-zero vector (no vocab hits)
      - "bank statement for last 6 months" → hits 'statement','bank','month','last'
    """

    def encode(self, texts_or_text, batch_size=64, show_progress_bar=False):
        if isinstance(texts_or_text, str):
            return self._encode_one(texts_or_text)
        return np.array(
            [self._encode_one(t) for t in texts_or_text], dtype=np.float32
        )

    def _encode_one(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        vec = np.zeros(len(_VOCAB), dtype=np.float32)
        for i, word in enumerate(_VOCAB):
            if word in text_lower:
                vec[i] = 1.0
        # Avoid pure zero vectors — use a hash-based fallback dimension
        if vec.sum() == 0:
            idx = abs(hash(text)) % len(_VOCAB)
            vec[idx] = 0.1
        return vec


# ---------------------------------------------------------------------------
# Shared intent definitions
# ---------------------------------------------------------------------------

SAMPLE_INTENTS = [
    {"name": "check_balance",    "description": "User wants to check their account balance."},
    {"name": "bank_statement",   "description": "User wants to download or view a bank statement."},
    {"name": "transfer_money",   "description": "User wants to transfer money to another account."},
    {"name": "block_card",       "description": "User wants to block a lost or stolen card."},
    {"name": "apply_loan",       "description": "User wants to apply for a loan."},
    {"name": "contact_support",  "description": "User wants to speak to a human support agent."},
]

SAMPLE_HIERARCHY = {
    "accounts": {
        "description": "Managing bank accounts: balance, statements.",
        "intents": ["check_balance", "bank_statement"],
    },
    "transactions": {
        "description": "Money movement: transfers, payments.",
        "intents": ["transfer_money"],
    },
    "cards": {
        "description": "Card management: block, unblock.",
        "intents": ["block_card"],
    },
    "loans": {
        "description": "Borrowing: loan applications, repayments.",
        "intents": ["apply_loan"],
    },
    "support": {
        "description": "Customer support and assistance.",
        "intents": ["contact_support"],
    },
}

SAMPLE_EXAMPLES = [
    {"text": "what is my balance",              "intent": "check_balance"},
    {"text": "check my account balance",        "intent": "check_balance"},
    {"text": "I need my bank statement",        "intent": "bank_statement"},
    {"text": "download my statement last month","intent": "bank_statement"},
    {"text": "transfer money to savings",       "intent": "transfer_money"},
    {"text": "send money to john",              "intent": "transfer_money"},
    {"text": "block my card",                   "intent": "block_card"},
    {"text": "freeze my credit card",           "intent": "block_card"},
    {"text": "apply for a personal loan",       "intent": "apply_loan"},
    {"text": "I want to contact support",       "intent": "contact_support"},
]

MULTI_TURN_EXAMPLES = [
    {
        "text": "for last 6 months",
        "intent": "bank_statement",
        "history_context": "I need my bank statement",
    },
    {
        "text": "yes freeze it now",
        "intent": "block_card",
        "history_context": "did you lose your card",
    },
]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    return MockEncoder()


@pytest.fixture
def sample_intents():
    return SAMPLE_INTENTS


@pytest.fixture
def sample_hierarchy():
    return SAMPLE_HIERARCHY


@pytest.fixture
def populated_store(encoder):
    """ExampleStore pre-loaded with single-turn and multi-turn examples."""
    from query_classifier.example_store import ExampleStore
    store = ExampleStore(encoder=encoder)
    store.add_examples_bulk(SAMPLE_EXAMPLES)
    store.add_examples_bulk(MULTI_TURN_EXAMPLES)
    return store
