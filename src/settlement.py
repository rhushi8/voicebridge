"""Settlement bounds live in code so the LLM can't invent a discount."""

from datetime import date

# (min days overdue, max discount), checked top-down.
TIERS = (
    (180, 0.30),
    (90, 0.20),
    (30, 0.10),
)


def days_overdue(due_date_str, today=None):
    today = today or date.today()
    return (today - date.fromisoformat(due_date_str)).days


def max_discount(days):
    for min_days, discount in TIERS:
        if days >= min_days:
            return discount
    return 0.0


def settlement_floor(balance, days):
    """Lowest lump sum allowed, or None if no authority."""
    discount = max_discount(days)
    if discount == 0.0 or balance <= 0:
        return None
    return round(balance * (1 - discount))


def format_settlement_authority(customer, today=None):
    days = days_overdue(customer["due_date"], today)
    floor = settlement_floor(customer["balance"], days)
    if floor is None:
        return ("Settlement authority: NONE for this account. Do not offer or "
                "accept any discount or settlement; if asked, say the account "
                "does not qualify and offer a payment plan instead.")
    discount_pct = int(max_discount(days) * 100)
    return (
        f"Settlement authority (one-time lump sum only): this account is {days} "
        f"days overdue, so you may accept as low as {floor} rupees "
        f"(up to {discount_pct}% off the balance). Negotiate: start near the "
        f"full balance, concede gradually, never go below {floor} rupees, and "
        f"never volunteer the floor figure yourself."
    )
