"""Deterministic EMI (equated monthly installment) calculation.

Real banks do NOT split a balance evenly -- they charge interest on the
reducing balance, using the standard EMI formula:

    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

where  P = principal (outstanding balance)
       r = monthly interest rate (annual rate / 12)
       n = number of monthly installments

This lives in Python, never in the LLM. Money math must be exact and identical
on every call -- a language model estimating an installment is exactly the
failure mode this module exists to remove.

The terms below are kept in sync with data/policies/payment-plans.md.
"""

ANNUAL_INTEREST_RATE = 0.12   # 12% per annum on the reducing balance
TENURES_MONTHS = (6, 9, 12)
MIN_BALANCE = 5000            # plans are only offered at or above this balance


def monthly_emi(principal, months, annual_rate=ANNUAL_INTEREST_RATE):
    """Return the exact EMI for `principal` over `months`, rounded to rupees."""
    r = annual_rate / 12
    if r == 0:
        return round(principal / months)
    growth = (1 + r) ** months
    return round(principal * r * growth / (growth - 1))


def plan_options(principal, tenures=TENURES_MONTHS, annual_rate=ANNUAL_INTEREST_RATE):
    """Return [(months, emi), ...] for the given balance."""
    return [(m, monthly_emi(principal, m, annual_rate)) for m in tenures]


def format_plan_options(principal):
    """Render the plan options as text for the LLM briefing.

    Returns an eligibility note instead if the balance is too low.
    """
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
