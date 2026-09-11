# VoiceBridge

A debt-collections voice agent for a fictional bank called Horizon Bank. You
speak, it answers, and every number it says traces back to a record rather than
to the model.

> All customer data and policies here are made up.

## The pipeline

```
caller audio → VAD → ASR → endpointing → LLM + RAG → TTS → caller
             (webrtcvad) (faster-whisper)  (Groq)   (edge-tts)
```

Every turn runs that loop. The model never invents a fact and never does
arithmetic. Customer figures come from the account record, policy answers come
from retrieved policy text, installment amounts come from a deterministic EMI
calculator, and the legally required opening disclosure is scripted in code
instead of generated.

Two providers, deliberately. Groq runs the conversation, Gemini computes the
embeddings behind retrieval. When Groq retired the model the brain was pinned
to, retrieval carried on working, which is the argument for the split in one
sentence.

## What it does

Customer figures come from mock account records, and every number the bot says
traces back to one of them.

Policy questions go through retrieval. Documents are chunked by section, the
index is cached, and chunks are injected per turn above a similarity threshold
of 0.6 that was calibrated against real queries rather than picked. Answers
name the policy they came from.

Voice is neural TTS out with `edge-tts`, and local ASR in with `faster-whisper`.

VAD aggressiveness, endpointing silence windows, pre-speech buffering and TTS
prosody all live in `config.yaml`, and each knob is documented with the symptom
you get when it's set wrong.

Payment plans over 6, 9 and 12 months are computed in Python with the
reducing-balance formula. The model quotes those figures and never derives them.

The collections workflow is modelled on how production collections AI actually
runs. The mini-Miranda disclosure plays only after identity is confirmed,
promises to pay are captured, settlement negotiation is bounded by a coded
authority engine, payment links are simulated over SMS, bankruptcy-flagged
accounts are scrubbed before dialling, and every call ends with a disposition.

The brain reports events as hidden control tags: `<<VERIFIED>>`,
`<<PTP|date|amt>>`, `<<SETTLE|amt>>`, `<<TRANSFER>>`, `<<WRONG_PARTY>>` and
`<<DISPUTE>>`. The code strips them, validates them, and takes the action. A
settlement below the floor is flagged as an audit violation rather than quietly
accepted.

Per-turn stage latencies and call outcomes are logged to JSONL transcripts, and
`src/report.py` aggregates latency and the disposition mix.

## Guardrails

These live in the system briefing and are checked by scripted adversarial calls.

1. Facts come from the record. No invented figures, no invented explanations.
2. Identity first. Wrong person means nothing is revealed.
3. No threats and no pressure.
4. Hardship or distress stops collection and hands off to a human.
5. Bank-account scope only.
6. Policy answers only from retrieved excerpts, with the policy named.
7. Installment amounts only from the pre-computed EMI options.
8. Settlements only within the coded authority tiers, where the discount grows
   with days overdue. The floor is never revealed, and the code audits every
   agreed amount independently of the model.

Rule 8 is the one worth testing. On an earlier model the bot crossed its own
floor twice under pressure during a live negotiation, and the code-side auditor
caught both. Prompt-only enforcement wasn't enough, so the floor check moved
into code, and a below-floor acceptance now replaces the spoken reply.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then add two free API keys to `.env`. `GROQ_API_KEY` from console.groq.com runs
the brain, and `GEMINI_API_KEY` from aistudio.google.com/apikey runs the
embeddings.

## Run

```
python src\main.py            text call, type as the customer
python src\main.py --voice    voice call, needs a mic and speakers
python src\report.py          latency and outcome report across logged calls
```

The first run indexes the policy documents, and the first voice run downloads
the ASR model. Both are cached afterwards.

## Tuning the voice pipeline

| Knob | Too low | Too high |
|------|---------|----------|
| `vad.aggressiveness` (0-3) | background noise counts as speech | soft speakers get dropped |
| `endpointing.silence_after_speech_ms` | bot interrupts mid-sentence pauses | dead air before every reply |
| `endpointing.min_speech_ms` | coughs become speech | short answers like "yes" get ignored |
| `endpointing.pre_speech_buffer_ms` | first syllable clipped | slight lag entering speech |
| `tts.rate` | robotic drawl | rushed, pushy collector |
| `asr.model_size` | mishears accents and numbers | slower replies |

## Measured performance

From real logged calls, via `src/report.py`.

| Stage | Typical |
|-------|---------|
| LLM (Groq, `openai/gpt-oss-120b`) | 730-1090 ms |
| RAG retrieval | ~600 ms |
| ASR transcription (`small`) | ~1.5 s |
| TTS synthesis | ~2 s |

The brain started on Gemini, where a reply took 8 to 14 seconds and the free
tier allowed 20 requests a day. Moving it to Groq cut that to under a second,
which is what turned the demo from hold music into a conversation.

Groq has since retired `llama-3.3-70b-versatile`, which is where the earlier
430 ms figure came from. `openai/gpt-oss-120b` replaced it after benchmarking
both available sizes on a real turn, and the adversarial settlement call was
re-run on the new model to confirm the guardrails still hold.

## Layout

```
config.yaml          voice pipeline tuning knobs
data/customers.json  mock account records, a bank API in production
data/policies/       policy documents, the RAG corpus
src/main.py          call loop for text or voice, handoff detection
src/brain.py         system briefing, guardrails, Groq call
src/rag.py           chunking, embeddings, retrieval
src/emi.py           reducing-balance EMI calculator
src/settlement.py    settlement authority tiers, negotiation bounds in code
src/voice.py         TTS, VAD and endpointing state machine, ASR
src/records.py       customer record access, swappable for a real API
src/calllog.py       per-turn transcript and latency logging
src/report.py        performance aggregation
logs/                call transcripts, gitignored
```

## Known limits

- Conversation history is unbounded. A very long call keeps growing the prompt,
  since the whole history is resent every turn.
- TTS needs internet. `edge-tts` is a cloud voice, while ASR is local, so only
  the speaking half breaks offline.
- Free-tier rate limits are real. A burst of turns can trip a 429. The brain
  honours the `Retry-After` header and waits, up to ten seconds, before falling
  back to a spoken retry line.
- Endpointing is tuned for one voice in one room. The defaults in `config.yaml`
  suit the machine it was built on, and a noisier room wants a different
  `vad.aggressiveness`.
