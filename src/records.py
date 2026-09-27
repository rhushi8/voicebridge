"""Mock DB for now. Swap in the bank API here and nowhere else."""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "customers.json"


def load_customers():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def format_record(customer):
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
