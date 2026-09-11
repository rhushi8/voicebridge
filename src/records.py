"""Customer record access.

In a real deployment this would call the bank's API for a live record. For now
it reads the mock database in data/customers.json. Same interface either way,
so swapping in a real API later touches only this file.
"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "customers.json"


def load_customers():
    """Return the full list of customer records (list of dicts)."""
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def format_record(customer):
    """Render one customer record as readable text for the LLM's briefing."""
    lines = [
        f"Name: {customer['name']}",
        f"Phone: {customer['phone']}",
        f"Outstanding balance: {customer['balance']} rupees",
        f"Due date: {customer['due_date']}",
        f"Last payment: {customer['last_payment']['amount']} rupees on {customer['last_payment']['date']}",
        f"Account flags: {', '.join(customer.get('flags', [])) or 'none'}",
        "Payment history:",
    ]
    for payment in customer["payment_history"]:
        lines.append(f"  - {payment['amount']} rupees on {payment['date']}")
    return "\n".join(lines)
