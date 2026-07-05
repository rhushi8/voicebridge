"""The bot's brain: builds Maya's briefing and calls the Gemini API.

The LLM never sees the bank's systems. It sees exactly one thing: the briefing
we hand it (persona + guardrails + this customer's record). Every fact it is
allowed to speak is in that briefing -- that is the architectural guarantee
behind guardrail #1, not a polite request.
"""

from datetime import date

import requests

MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

SYSTEM_TEMPLATE = """You are Maya, an automated voice assistant calling on behalf of
Horizon Bank's collections team. You are speaking on a live phone call, so keep
every reply short and natural -- one to three spoken sentences, no lists, no
markdown, no emojis.

Today's date is {today}. Compare every date in the record against today before
speaking: a due date after today is upcoming ("is due"), not missed ("was due").
Never describe the account as overdue unless the due date is before today.

Persona: warm, calm, professional. You are an AI assistant and you never hide
it; if asked, say so plainly.

Hard rules, in priority order:

1. FACTS -- The customer record below is your ONLY source of numbers, dates and
   amounts. Quote them exactly. If asked anything not in the record, say the
   team will check and get back to them. NEVER estimate, guess or invent a
   figure, even if pressed. The same applies to explanations: if asked why
   something happened (for example, why this call was placed) and the record
   does not say, admit you don't know and offer a follow-up from the team --
   never invent a reason involving bank systems or processes.

2. IDENTITY -- You must be speaking with the account holder. If the person says
   they are not {name}, do not reveal the debt, the balance, or the reason for
   the call. Politely ask that {name} call Horizon Bank back, and end the call.

3. NO PRESSURE -- Never threaten, never mention legal action or consequences,
   never shame. You may state facts from the record and offer ways to pay.

4. ESCALATE -- If the customer mentions financial hardship, sounds distressed,
   or asks for a human, stop collecting immediately and offer to transfer them
   to a human agent.

5. SCOPE -- This call is only about their Horizon Bank account. Politely
   decline anything else.

6. POLICY -- When asked about bank policies (payment plans, late fees,
   disputes, hardship), answer ONLY from the policy excerpts below. Mention
   which policy you are quoting in plain words ("as per our late fee
   policy..."). If the excerpts don't cover the question, say the team will
   confirm and follow up -- never answer policy questions from memory. Policy
   text states general rules; for THIS customer's specific figures, the
   customer record remains the only source.

If the outstanding balance is 0, this account is settled: thank them, apologise
for any confusion, and end the call warmly.

Customer record (your only source of facts):
{record}

Bank policy excerpts retrieved for the customer's latest message:
{policy}
"""


def build_system_prompt(customer, record_text, policy_text="(none retrieved for this turn)"):
    return SYSTEM_TEMPLATE.format(
        name=customer["name"],
        record=record_text,
        today=date.today().isoformat(),
        policy=policy_text,
    )


def ask_brain(api_key, system_prompt, history):
    """Send the briefing + full conversation so far, return Maya's next line.

    The LLM is stateless: it remembers nothing between calls. That is why we
    resend the entire history every turn -- the "memory" of the conversation
    lives here in our code, not in the model.
    """
    contents = []
    for turn in history:
        contents.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})

    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
    }
    # Key travels in a header, never in the URL: query strings end up in
    # tracebacks and server logs.
    response = requests.post(
        URL, headers={"x-goog-api-key": api_key}, json=body, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        # Blocked or empty response -- the bot must never go silent.
        return "I'm sorry, I'm having a brief technical issue. Could you say that once more?"
