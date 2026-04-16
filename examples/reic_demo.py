"""
REIC Demo — three classification modes side by side.

Mode 1 | FLAT              SemanticRouter + LLM only. No RAG, no hierarchy.
Mode 2 | FLAT_RAG          SemanticRouter + ExampleStore (global) + LLM.
Mode 3 | HIERARCHICAL_RAG  HierarchicalRouter (coarse→fine) + ExampleStore
                           (category-filtered) + LLM. Full REIC pipeline.

All three share the same classify() call signature — the mode is set at
construction time and is transparent to the caller.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_classifier import IntentClassifier, ClassificationMode, TurnMode, ExampleStore, ExampleStoreBootstrapper
from examples.banking_intents import INTENTS, INTENT_HIERARCHY

# ---------------------------------------------------------------------------
# Seed labeled examples
# ---------------------------------------------------------------------------
SEED_EXAMPLES = [
    # account_check_balance
    {"text": "What is my current balance?", "intent": "account_check_balance"},
    {"text": "How much money do I have in my account?", "intent": "account_check_balance"},
    {"text": "Check my savings balance", "intent": "account_check_balance"},
    {"text": "Mera balance kitna hai?", "intent": "account_check_balance"},
    {"text": "What's left in my checking account?", "intent": "account_check_balance"},
    # account_statement_request
    {"text": "I need my bank statement for last month", "intent": "account_statement_request"},
    {"text": "Can you send me a PDF statement?", "intent": "account_statement_request"},
    {"text": "Download my account statement", "intent": "account_statement_request"},
    # account_open_new
    {"text": "I want to open a new bank account", "intent": "account_open_new"},
    {"text": "How do I open a savings account?", "intent": "account_open_new"},
    {"text": "Can I create a new current account?", "intent": "account_open_new"},
    {"text": "I'd like to open a joint account with my spouse", "intent": "account_open_new"},
    {"text": "What documents do I need to open an account?", "intent": "account_open_new"},
    # account_close_request
    {"text": "I want to close my bank account", "intent": "account_close_request"},
    {"text": "Please close my savings account", "intent": "account_close_request"},
    {"text": "How do I shut down my account?", "intent": "account_close_request"},
    {"text": "I no longer need this account, close it", "intent": "account_close_request"},
    {"text": "Can you terminate my current account?", "intent": "account_close_request"},
    # account_update_details
    {"text": "I need to update my address", "intent": "account_update_details"},
    {"text": "Change my phone number on the account", "intent": "account_update_details"},
    {"text": "My email has changed, please update it", "intent": "account_update_details"},
    {"text": "Update my personal details on file", "intent": "account_update_details"},
    {"text": "I got married and need to change my last name", "intent": "account_update_details"},
    # transaction_transfer_funds
    {"text": "Transfer $500 to my savings account", "intent": "transaction_transfer_funds"},
    {"text": "Send money to John's account", "intent": "transaction_transfer_funds"},
    {"text": "I want to move funds between accounts", "intent": "transaction_transfer_funds"},
    {"text": "Wire $1000 to IBAN DE89370400440532013000", "intent": "transaction_transfer_funds"},
    # transaction_history
    {"text": "Show me my recent transactions", "intent": "transaction_history"},
    {"text": "What did I spend last week?", "intent": "transaction_history"},
    {"text": "List all transactions from March", "intent": "transaction_history"},
    # transaction_pay_bill
    {"text": "Pay my electricity bill", "intent": "transaction_pay_bill"},
    {"text": "I want to pay my credit card bill", "intent": "transaction_pay_bill"},
    {"text": "Schedule a bill payment for next week", "intent": "transaction_pay_bill"},
    # transaction_dispute
    {"text": "I don't recognize a charge on my account", "intent": "transaction_dispute"},
    {"text": "There's a fraudulent transaction I want to dispute", "intent": "transaction_dispute"},
    {"text": "Someone charged me twice, I want a refund", "intent": "transaction_dispute"},
    # transaction_recurring_setup
    {"text": "Set up a monthly standing order", "intent": "transaction_recurring_setup"},
    {"text": "I want to automate my rent payment every month", "intent": "transaction_recurring_setup"},
    {"text": "Create a recurring transfer of $200 weekly", "intent": "transaction_recurring_setup"},
    {"text": "Set up automatic bill payments", "intent": "transaction_recurring_setup"},
    {"text": "Schedule a repeating payment to my landlord", "intent": "transaction_recurring_setup"},
    # card_block
    {"text": "Block my credit card immediately", "intent": "card_block"},
    {"text": "I lost my card, freeze it now", "intent": "card_block"},
    {"text": "My card was stolen, please deactivate it", "intent": "card_block"},
    # card_unblock
    {"text": "Unblock my debit card", "intent": "card_unblock"},
    {"text": "I found my card, please unfreeze it", "intent": "card_unblock"},
    {"text": "Reactivate my blocked card", "intent": "card_unblock"},
    {"text": "My card was blocked by mistake, unlock it", "intent": "card_unblock"},
    {"text": "Can you enable my card again?", "intent": "card_unblock"},
    # card_new_request
    {"text": "I want a new debit card", "intent": "card_new_request"},
    {"text": "Apply for a credit card", "intent": "card_new_request"},
    {"text": "Order a replacement card", "intent": "card_new_request"},
    {"text": "My card expired, I need a new one", "intent": "card_new_request"},
    {"text": "Request a contactless card", "intent": "card_new_request"},
    # card_pin_change
    {"text": "I want to change my PIN", "intent": "card_pin_change"},
    {"text": "Reset my ATM PIN", "intent": "card_pin_change"},
    {"text": "How do I update my card PIN?", "intent": "card_pin_change"},
    # card_limit_change
    {"text": "Increase my card spending limit", "intent": "card_limit_change"},
    {"text": "Lower my daily ATM withdrawal limit", "intent": "card_limit_change"},
    {"text": "Change my credit card limit to $5000", "intent": "card_limit_change"},
    {"text": "I want to raise my contactless payment limit", "intent": "card_limit_change"},
    {"text": "Can you adjust my card transaction limit?", "intent": "card_limit_change"},
    # loan_apply
    {"text": "I'd like to apply for a personal loan", "intent": "loan_apply"},
    {"text": "Can I get a loan for $10,000?", "intent": "loan_apply"},
    {"text": "What loans are available?", "intent": "loan_apply"},
    # loan_status_check
    {"text": "What is the status of my loan application?", "intent": "loan_status_check"},
    {"text": "Has my loan been approved yet?", "intent": "loan_status_check"},
    {"text": "Check my loan application progress", "intent": "loan_status_check"},
    {"text": "When will my loan be processed?", "intent": "loan_status_check"},
    {"text": "I applied for a loan last week, what's the update?", "intent": "loan_status_check"},
    # loan_repayment
    {"text": "I want to make a loan repayment", "intent": "loan_repayment"},
    {"text": "Pay off my outstanding loan balance", "intent": "loan_repayment"},
    {"text": "How do I repay my personal loan?", "intent": "loan_repayment"},
    {"text": "Make an early loan settlement", "intent": "loan_repayment"},
    {"text": "I want to pay an extra installment on my loan", "intent": "loan_repayment"},
    # credit_score_check
    {"text": "What is my credit score?", "intent": "credit_score_check"},
    {"text": "Check my CIBIL score", "intent": "credit_score_check"},
    {"text": "How good is my credit rating?", "intent": "credit_score_check"},
    # mortgage_inquiry
    {"text": "What mortgage options do you offer?", "intent": "mortgage_inquiry"},
    {"text": "I'm looking to buy a house, tell me about mortgages", "intent": "mortgage_inquiry"},
    {"text": "What are your current mortgage rates?", "intent": "mortgage_inquiry"},
    {"text": "How much can I borrow for a home loan?", "intent": "mortgage_inquiry"},
    {"text": "Explain your fixed vs variable mortgage plans", "intent": "mortgage_inquiry"},
    # support_contact_human
    {"text": "Let me speak to a real person", "intent": "support_contact_human"},
    {"text": "Connect me to a human agent", "intent": "support_contact_human"},
    {"text": "I want to talk to customer service", "intent": "support_contact_human"},
    # support_branch_locator
    {"text": "Where is the nearest bank branch?", "intent": "support_branch_locator"},
    {"text": "Find a branch close to me", "intent": "support_branch_locator"},
    {"text": "What are the branch opening hours?", "intent": "support_branch_locator"},
    {"text": "Is there a branch in downtown?", "intent": "support_branch_locator"},
    {"text": "I need to visit a branch, where is the closest one?", "intent": "support_branch_locator"},
    # support_atm_locator
    {"text": "Where is the nearest ATM?", "intent": "support_atm_locator"},
    {"text": "Find an ATM near me", "intent": "support_atm_locator"},
    {"text": "Is there a cash machine close by?", "intent": "support_atm_locator"},
    # support_mobile_app_help
    {"text": "The mobile app is not working", "intent": "support_mobile_app_help"},
    {"text": "I can't log into the banking app", "intent": "support_mobile_app_help"},
    {"text": "How do I use mobile banking?", "intent": "support_mobile_app_help"},
    {"text": "The app keeps crashing", "intent": "support_mobile_app_help"},
    {"text": "I'm having trouble with the online banking portal", "intent": "support_mobile_app_help"},
    # support_security_alert
    {"text": "I think my account has been hacked", "intent": "support_security_alert"},
    {"text": "Report suspicious login activity", "intent": "support_security_alert"},
    {"text": "Someone is trying to access my account", "intent": "support_security_alert"},

    # ------------------------------------------------------------------
    # Multi-turn examples — follow-up utterances with history_context.
    # The embedding stored = encode(history_context + " " + text) so that
    # a context-enriched RAG query finds these correctly at inference time.
    # ------------------------------------------------------------------

    # account_statement_request continuations
    {"text": "for last 6 months",        "intent": "account_statement_request",
     "history_context": "I need my bank statement"},
    {"text": "make it the last year",    "intent": "account_statement_request",
     "history_context": "I need my bank statement for last 6 months"},
    {"text": "yes please send it",       "intent": "account_statement_request",
     "history_context": "do you want a PDF statement"},
    {"text": "in PDF format",            "intent": "account_statement_request",
     "history_context": "what format do you want the statement"},

    # account_check_balance continuations
    {"text": "what about savings",       "intent": "account_check_balance",
     "history_context": "your current account balance is"},
    {"text": "and my joint account",     "intent": "account_check_balance",
     "history_context": "your savings balance is"},

    # transaction_transfer_funds continuations
    {"text": "make it 500",              "intent": "transaction_transfer_funds",
     "history_context": "how much would you like to transfer"},
    {"text": "to my savings account",   "intent": "transaction_transfer_funds",
     "history_context": "where would you like to send the money"},

    # transaction_dispute continuations
    {"text": "yes that one on tuesday",  "intent": "transaction_dispute",
     "history_context": "which transaction do you want to dispute"},
    {"text": "I never made that payment","intent": "transaction_dispute",
     "history_context": "we see a charge from amazon"},

    # card_block continuations
    {"text": "yes block it immediately", "intent": "card_block",
     "history_context": "did you lose your debit card"},
    {"text": "the one ending in 4521",   "intent": "card_block",
     "history_context": "which card would you like to block"},

    # loan_apply continuations
    {"text": "personal loan",            "intent": "loan_apply",
     "history_context": "what type of loan are you interested in"},
    {"text": "about 10000",              "intent": "loan_apply",
     "history_context": "how much would you like to borrow"},

    # support_contact_human continuations
    {"text": "yes connect me now",       "intent": "support_contact_human",
     "history_context": "would you like to speak to an agent"},
]


async def run_demo():
    print("=" * 70)
    print("  REIC Demo — Three Classification Modes")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Build ExampleStore.
    #
    # Two paths depending on whether a pre-built store exists:
    #
    # PATH A — First run / no saved store:
    #   1. Add manual seed examples (covers all 25 intents, single + multi-turn)
    #   2. Run ExampleStoreBootstrapper to auto-generate more examples via LLM
    #   3. Save to disk for future runs
    #
    # PATH B — Subsequent runs:
    #   Load the pre-built store from disk (instant, embeddings cached in .npy)
    # -----------------------------------------------------------------------
    import os
    store_path = "banking_example_store"

    if os.path.exists(f"{store_path}.json"):
        print(f"\n[Setup] Loading existing ExampleStore from {store_path}.*")
        store = ExampleStore()
        store.load(store_path)
        print(f"  {len(store)} examples | {len(store.intent_coverage())} intents covered")
        print(f"  Multi-turn coverage: {store.multi_turn_coverage()}")
    else:
        print("\n[Setup] Building ExampleStore from scratch...")

        # Step 1: Add manual seed examples
        store = ExampleStore()
        store.add_examples_bulk(SEED_EXAMPLES)
        print(f"  Seed examples: {len(store)} total")

        # Step 2: Auto-generate more examples using the LLM
        # This fills multi-turn gaps for ALL 25 intents automatically.
        print("\n[Setup] Running ExampleStoreBootstrapper (LLM-generated examples)...")
        # n_single, n_multi, concurrency, max_retries all read from config.py
        # or environment variables (BOOTSTRAP_N_SINGLE, BOOTSTRAP_N_MULTI, etc.)
        bootstrapper = ExampleStoreBootstrapper(
            llm_model_name="llama3",  # replace with your cloud model
        )
        summary = await bootstrapper.run_full_bootstrap(
            store=store,
            intents=INTENTS,
            save_path=store_path,
            # n_single and n_multi use BOOTSTRAP_N_SINGLE / BOOTSTRAP_N_MULTI from config
        )
        print(f"\n  Bootstrap complete:")
        print(f"    Total examples      : {summary['total']}")
        print(f"    Single-turn added   : {summary['single_turn_generated']}")
        print(f"    Multi-turn added    : {summary['multi_turn_generated']}")
        print(f"    Intents covered     : {summary['intents_covered']}/25")
        print(f"    Multi-turn coverage : {summary['multi_turn_coverage']}")

    # -----------------------------------------------------------------------
    # Instantiate classifiers — mode × turn_mode combinations
    # -----------------------------------------------------------------------
    print("\n[Setup] Initialising classifiers...")

    # Single-turn, Flat — each query independent, no RAG
    nlp_flat_single = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.FLAT,
        turn_mode=TurnMode.SINGLE,
        llm_model_name="llama3",
        enable_lang_detect=False,
    )

    # Single-turn, Flat + RAG — independent queries with few-shot evidence
    nlp_flat_rag_single = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.FLAT_RAG,
        turn_mode=TurnMode.SINGLE,
        llm_model_name="llama3",
        example_store=store,
        enable_lang_detect=False,
    )

    # Multi-turn, Hierarchical + RAG — full REIC, conversational
    nlp_hier_rag_multi = IntentClassifier(
        intents=INTENTS,
        mode=ClassificationMode.HIERARCHICAL_RAG,
        turn_mode=TurnMode.MULTI,
        llm_model_name="llama3",
        example_store=store,
        intent_hierarchy=INTENT_HIERARCHY,
        top_n_categories=2,
        enable_lang_detect=True,
    )

    print("  All classifiers ready.\n")

    # -----------------------------------------------------------------------
    # Side-by-side comparison
    # -----------------------------------------------------------------------
    test_queries = [
        "How much money is left in my account?",
        "I think there's a fraudulent charge on my card",
        "Mera balance check karo",
        "Where can I find a cash machine near here?",
        "Set up auto-payment for my rent every month",
        "My card got stolen, freeze it",
        "I want to buy a house, what loans do you have?",
        "The banking app keeps crashing on my phone",
    ]

    header = f"  {'Query':<42} {'FLAT/single':<28} {'FLAT_RAG/single':<28} {'HIER_RAG/multi':<28}"
    print("=" * 70)
    print("  Mode × TurnMode comparison (single-turn queries)")
    print("=" * 70)
    print(header)
    print("  " + "-" * 128)

    def fmt(intent, conf):
        return f"{intent} ({conf:.2f})"

    for query in test_queries:
        flat_intent, flat_conf, _ = await nlp_flat_single.classify(query)
        flat_rag_intent, flat_rag_conf, _ = await nlp_flat_rag_single.classify(query)
        hier_intent, hier_conf, _ = await nlp_hier_rag_multi.classify(query)

        print(
            f"  {query[:40]:<42} "
            f"{fmt(flat_intent, flat_conf):<28} "
            f"{fmt(flat_rag_intent, flat_rag_conf):<28} "
            f"{fmt(hier_intent, hier_conf):<28}"
        )

    # -----------------------------------------------------------------------
    # Multi-turn with HIERARCHICAL_RAG
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Multi-turn context (HIERARCHICAL_RAG)")
    print("=" * 70)

    history = []
    turns = [
        "I need my bank statement",
        "For the last 6 months please",
        "Actually, make it the last year",
        "Also, what's my balance?",
    ]
    for turn in turns:
        intent, conf, lang = await nlp_hier_rag_multi.classify(
            turn, conversation_history=history
        )
        print(f"  User : {turn}")
        print(f"  Agent: [{intent}] conf={conf:.2f} lang={lang}\n")
        history.append({"role": "user", "content": turn})
        # intent_classified is the recommended field — _extract_prior_intent()
        # reads this directly without any string parsing.
        history.append({
            "role": "assistant",
            "content": f"Classified as: {intent}",
            "intent_classified": intent,
        })

    # -----------------------------------------------------------------------
    # Demonstrate fail-fast validation
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  Fail-fast validation demo")
    print("=" * 70)
    try:
        bad = IntentClassifier(
            intents=INTENTS,
            mode=ClassificationMode.HIERARCHICAL_RAG,
            turn_mode=TurnMode.MULTI,
            # intentionally omitting intent_hierarchy
        )
    except ValueError as e:
        print(f"  Caught expected error:\n  {e}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
