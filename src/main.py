import os
import re
import sys
import time
import uuid
from datetime import date

import requests
from dotenv import load_dotenv

import calllog
import rag
import settlement
from brain import MODEL, ask_brain, build_system_prompt
from records import format_record, load_customers

RETRY_LINE = "I'm sorry, I'm having a brief technical issue. Could you say that once more?"

OPENING = (
    "Hi, this is Maya, an automated assistant calling from Horizon Bank "
    "regarding your account. Am I speaking with {name}?"
)

# Legally required disclosure. Only after identity is confirmed.
MIRANDA = (
    "Please note, this is an attempt to collect a debt, and any information "
    "obtained will be used for that purpose."
)

HANGUP_WORDS = {"quit", "exit", "bye", "goodbye"}

# Outcome tags count only if the customer's last words sound like assent.
# ponytail: word-set heuristic, swap for an LLM confirmation pass if it misses.
ASSENT_WORDS = {"yes", "yeah", "yep", "okay", "ok", "sure", "fine", "agreed",
                "agree", "deal", "done", "works", "confirm", "confirmed",
                "haan", "theek", "accha"}

TAG_PATTERN = re.compile(r"<<([A-Z_]+)(?:\|([^>]*))?>>")


def parse_tags(reply):
    tags = TAG_PATTERN.findall(reply)
    clean = TAG_PATTERN.sub("", reply).strip()
    return clean, tags


def send_sms(phone, text):
    """Simulated. Production would call Twilio or similar."""
    print(f"  [sms -> {phone}] {text}")
    return text


def payment_link():
    return f"https://pay.horizonbank.example/{uuid.uuid4().hex[:8]}"


def pick_customer(customers):
    print("Which customer is Maya dialling today?\n")
    for i, customer in enumerate(customers, start=1):
        print(f"  {i}. {customer['name']} (balance: {customer['balance']} rupees)")
    valid = {str(n) for n in range(1, len(customers) + 1)}
    while True:
        choice = input(f"\nPick 1-{len(customers)}: ").strip()
        if choice in valid:
            return customers[int(choice) - 1]
        print("Just the digit, e.g. 1")


def main():
    voice_mode = "--voice" in sys.argv

    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not gemini_key or gemini_key.startswith("paste"):
        print("No GEMINI_API_KEY in .env. It runs the RAG embeddings.")
        print("Get one free at aistudio.google.com/apikey")
        return
    if not groq_key:
        print("No GROQ_API_KEY in .env. It runs Maya's brain.")
        print("Get one free at console.groq.com")
        return

    print("Indexing bank policies...")
    policy_index = rag.build_index(gemini_key)

    cfg = asr_model = voice = None
    if voice_mode:
        import voice

        cfg = voice.load_config()
        print("Loading the speech recognition model...")
        asr_model = voice.load_asr_model(cfg)

    customers = load_customers()
    customer = pick_customer(customers)

    # Pre-call scrub: bankruptcy-flagged accounts are never called.
    if "bankruptcy" in customer.get("flags", []):
        print(f"\n[scrubbed] {customer['name']} has a bankruptcy flag, so "
              "collection calls are not permitted. Account routed to legal.")
        log_path = calllog.start_call(customer["name"], "voice" if voice_mode else "text")
        calllog.end_call(log_path, "SCRUBBED_BANKRUPTCY")
        return

    record_text = format_record(customer)
    log_path = calllog.start_call(customer["name"], "voice" if voice_mode else "text")

    print("\n--- ringing... call connected ---\n")

    # Scripted, not generated: compliance wording must be deterministic.
    opening = OPENING.format(name=customer["name"])
    print(f"Maya: {opening}\n")
    if voice_mode:
        voice.speak(opening, cfg)
    history = [{"role": "model", "text": opening}]

    miranda_played = False
    disposition = "NO_AGREEMENT"
    sms_sent_for = set()

    # One write point for the outcome, however the call ends.
    try:
        while True:
            timings = {}

            if voice_mode:
                print("(listening...)")
                t0 = time.perf_counter()
                user_text = voice.listen(cfg, asr_model)
                timings["asr_ms"] = round((time.perf_counter() - t0) * 1000)
                if not user_text:
                    print("(heard nothing)")
                    voice.speak("Sorry, I didn't catch that. Could you say that again?", cfg)
                    continue
                print(f"You: {user_text}")
            else:
                user_text = input("You: ").strip()
                if not user_text:
                    continue

            spoken_words = {w.strip(".,!?;:") for w in user_text.lower().split()}
            if spoken_words & HANGUP_WORDS:
                closing = "Thank you for your time. Goodbye!"
                print(f"\nMaya: {closing}")
                if voice_mode:
                    voice.speak(closing, cfg)
                print("\n--- call ended ---")
                break

            history.append({"role": "user", "text": user_text})

            # Retrieve on a window: a bare "yes" means nothing alone.
            retrieval_query = user_text
            if len(user_text.split()) <= 3 and len(history) >= 2:
                retrieval_query = history[-2]["text"] + " " + user_text

            t0 = time.perf_counter()
            try:
                policy_chunks = rag.retrieve(gemini_key, policy_index, retrieval_query)
            except requests.RequestException:
                policy_chunks = []
            timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
            policy_text = rag.format_policy_context(policy_chunks)
            system_prompt = build_system_prompt(customer, record_text, policy_text)

            # Never go silent: on LLM failure, ask the caller to repeat.
            t0 = time.perf_counter()
            try:
                reply = ask_brain(groq_key, system_prompt, history)
            except requests.RequestException as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    print(f"  [{MODEL} rate limited, retries exhausted]")
                else:
                    print(f"  [model call failed after 1 retry: {e}]")
                reply = RETRY_LINE
            timings["llm_ms"] = round((time.perf_counter() - t0) * 1000)

            # The model emits tags, the code takes the action.
            reply, tags = parse_tags(reply)
            end_call_after = None
            turn_events = []

            assented = bool(spoken_words & ASSENT_WORDS)

            for tag, arg in tags:
                if tag in ("PTP", "PLAN", "SETTLE") and not assented:
                    turn_events.append(f"ignored unconfirmed {tag.lower()} {arg or ''}".strip())
                    continue

                if tag == "VERIFIED" and not miranda_played:
                    miranda_played = True
                    print(f"Maya: {MIRANDA}")
                    if voice_mode:
                        voice.speak(MIRANDA, cfg)
                    turn_events.append("verified")

                elif tag == "WRONG_PARTY":
                    disposition = "WRONG_PARTY"
                    end_call_after = "--- call ended (wrong party, nothing disclosed) ---"

                elif tag == "TRANSFER":
                    disposition = "TRANSFERRED"
                    end_call_after = "--- call handed off to a human agent ---"

                elif tag == "PTP" and arg and f"PTP|{arg}" not in sms_sent_for:
                    sms_sent_for.add(f"PTP|{arg}")
                    parts = arg.split("|")
                    try:
                        ptp_date = date.fromisoformat(parts[0].strip())
                        ptp_amount = int(float(parts[1].strip())) if len(parts) > 1 else customer["balance"]
                        disposition = "PTP"
                        turn_events.append(f"ptp {ptp_amount} by {ptp_date}")
                        send_sms(customer["phone"],
                                 f"Horizon Bank: as agreed, pay {ptp_amount} rupees by "
                                 f"{ptp_date} here: {payment_link()}")
                    except (ValueError, IndexError):
                        print(f"  [invalid PTP tag ignored: {arg!r}]")

                elif tag == "PLAN" and arg and f"PLAN|{arg}" not in sms_sent_for:
                    sms_sent_for.add(f"PLAN|{arg}")
                    disposition = "PLAN_AGREED"
                    turn_events.append(f"plan {arg.strip()} months")
                    send_sms(customer["phone"],
                             f"Horizon Bank: confirm your {arg.strip()}-month payment "
                             f"plan here: {payment_link()}")

                elif tag == "SETTLE" and arg and f"SETTLE|{arg}" not in sms_sent_for:
                    sms_sent_for.add(f"SETTLE|{arg}")
                    try:
                        amount = int(float(arg.strip()))
                        days = settlement.days_overdue(customer["due_date"])
                        floor = settlement.settlement_floor(customer["balance"], days)
                        if floor is None or amount < floor:
                            # Below-authority yes is flagged and the reply replaced.
                            turn_events.append(f"SETTLEMENT_VIOLATION {amount} < floor {floor}")
                            print(f"  [audit] settlement {amount} below authority ({floor}), blocked")
                            reply = ("I'm sorry, I spoke too soon -- I'm actually not able "
                                     "to accept that amount. I can connect you with a human "
                                     "agent to discuss further options, or we can look at a "
                                     "payment plan instead.")
                        else:
                            disposition = "SETTLED"
                            turn_events.append(f"settled {amount}")
                            send_sms(customer["phone"],
                                     f"Horizon Bank: complete your settlement of {amount} "
                                     f"rupees within 7 days: {payment_link()}")
                    except ValueError:
                        print(f"  [invalid SETTLE tag ignored: {arg!r}]")

                elif tag == "DISPUTE":
                    disposition = "DISPUTE"
                    turn_events.append("dispute raised")

            history.append({"role": "model", "text": reply})
            print(f"\nMaya: {reply}\n")

            if voice_mode:
                t0 = time.perf_counter()
                voice.speak(reply, cfg)
                timings["tts_ms"] = round((time.perf_counter() - t0) * 1000)

            calllog.log_turn(
                log_path,
                user_text,
                reply,
                [f"{c['doc']} / {c['heading']}" for c in policy_chunks],
                timings,
                events=turn_events,
            )

            if end_call_after:
                print(end_call_after)
                break

    finally:
        calllog.end_call(log_path, disposition)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- call ended (interrupted) ---")
