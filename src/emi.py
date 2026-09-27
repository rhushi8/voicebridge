"""Reducing-balance EMI in code, never the LLM. Keep in sync with payment-plans.md."""

ANNUAL_INTEREST_RATE = 0.12  # 12% p.a. on the reducing balance
TENURES_MONTHS = (6, 9, 12)
MIN_BALANCE = 5000  # plans only at or above this


def monthly_emi(principal, months, annual_rate=ANNUAL_INTEREST_RATE):
    r = annual_rate / 12
    if r == 0:
        return round(principal / months)
    growth = (1 + r) ** months
    return round(principal * r * growth / (growth - 1))


def plan_options(principal, tenures=TENURES_MONTHS, annual_rate=ANNUAL_INTEREST_RATE):
    return [(m, monthly_emi(principal, m, annual_rate)) for m in tenures]


def format_plan_options(principal):
    if principal < MIN_BALANCE:
        return (f"This balance ({principal} rupees) is below {MIN_BALANCE} rupees, "
                "so it is not eligible for a payment plan.")
    rate_pct = ANNUAL_INTEREST_RATE * 100
    lines = [
        f"Pre-computed payment plan options for the outstanding balance of "
        f"{principal} rupees, at {rate_pct:g}% annual interest on the reducing "
        f"balance. Quote these exact figures only:"
    ]
    for months, emi in plan_options(principal):
        lines.append(f"  - {months} months: {emi} rupees per month")
    return "\n".join(lines)
