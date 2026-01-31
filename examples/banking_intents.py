
INTENTS = [
    # --- ACCOUNTS ---
    {"name": "account_check_balance", "description": "User wants to check their account balance."},
    {"name": "account_statement_request", "description": "User wants to download or view a bank statement."},
    {"name": "account_open_new", "description": "User wants to open a new bank account."},
    {"name": "account_close_request", "description": "User wants to close their bank account."},
    {"name": "account_update_details", "description": "User wants to update personal account details (address, phone, etc.)."},

    # --- TRANSACTIONS ---
    {"name": "transaction_history", "description": "User wants to view recent transactions."},
    {"name": "transaction_transfer_funds", "description": "User wants to transfer money to another account."},
    {"name": "transaction_pay_bill", "description": "User wants to pay a bill."},
    {"name": "transaction_dispute", "description": "User wants to dispute a specific transaction."},
    {"name": "transaction_recurring_setup", "description": "User wants to set up a recurring payment."},

    # --- CARDS ---
    {"name": "card_block", "description": "User wants to block a lost or stolen card."},
    {"name": "card_unblock", "description": "User wants to unblock a card."},
    {"name": "card_new_request", "description": "User wants to request a new debit or credit card."},
    {"name": "card_pin_change", "description": "User wants to change their card PIN."},
    {"name": "card_limit_change", "description": "User wants to change their card spending limit."},

    # --- LOANS & CREDIT ---
    {"name": "loan_apply", "description": "User wants to apply for a loan."},
    {"name": "loan_status_check", "description": "User wants to check the status of a loan application."},
    {"name": "loan_repayment", "description": "User wants to make a loan repayment."},
    {"name": "credit_score_check", "description": "User wants to check their credit score."},
    {"name": "mortgage_inquiry", "description": "User asks about mortgage options."},

    # --- GENERAL SUPPORT ---
    {"name": "support_contact_human", "description": "User wants to speak to a human agent."},
    {"name": "support_branch_locator", "description": "User wants to find the nearest bank branch."},
    {"name": "support_atm_locator", "description": "User wants to find the nearest ATM."},
    {"name": "support_mobile_app_help", "description": "User needs help using the mobile banking app."},
    {"name": "support_security_alert", "description": "User wants to report suspicious activity."}
]
