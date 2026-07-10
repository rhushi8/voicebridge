# Collections Voice Agent

A GenAI-powered debt-collections assistant for a fictional bank ("Horizon
Bank") — built end to end the way a Forward Deployed AI Engineer would:
an LLM brain with hard guardrails, RAG over bank policy documents, a real
ASR/TTS voice layer, tunable VAD/endpointing configuration, deterministic
money math, and per-turn performance logging.

> All customer data and policies in this repo are fictional.

## The pipeline

```
caller audio → VAD → ASR → endpointing → LLM + RAG → TTS → caller
             (webrtcvad) (faster-whisper)   (Groq)   (edge-tts)
```

Every conversation turn runs this loop. The model never invents a fact and
never does arithmetic: customer figures come only from the account record,
policy answers only from retrieved policy text, installment amounts only from
a deterministic EMI calculator, and compliance-critical wording (the opening
AI disclosure) is scripted in code rather than generated.

Two providers, two jobs, on purpose: **Groq** (`llama-3.3-70b-versatile`) runs
the conversational brain — measured at ~300ms per reply — while **Gemini**
(`gemini-embedding-001`) computes the embeddings that power RAG retrieval.

## What's implemented

- [x] **Brain on rails** — mock customer records; every number the bot speaks
      traces to the record, never to the model
- [x] **RAG** — embeddings retrieval over bank policy docs, chunked by
      section, cached index, per-turn injection with a similarity threshold
      calibrated against real queries (0.6), citations by policy name
- [x] **Voice** — neural TTS out (`edge-tts`), local ASR in (`faster-whisper`)
- [x] **Deep configuration** — VAD aggressiveness, endpointing silence windows,
      pre-speech buffering, TTS prosody: all tunable in `config.yaml`, each
      knob documented with its failure symptom
- [x] **Deterministic EMI** — 6/9/12-month payment plans computed in Python
      with the reducing-balance formula; the LLM quotes the figures, never
      derives them
- [x] **Human handoff** — the brain emits a hidden control tag when it escalates;
      the code strips it, speaks the handoff line, and ends the call
- [x] **Bot performance** — per-turn stage latencies (ASR / retrieval / LLM /
      TTS) logged to JSONL transcripts, aggregated by `src/report.py`

## Guardrails (in the system briefing; verified by scripted adversarial calls)

1. Facts only from the record — no invented figures, no invented explanations
2. Identity check first; wrong person → reveal nothing
3. No threats or pressure, ever
4. Hardship or distress → stop collecting, hand off to a human agent
5. Bank-account scope only
6. Policy answers only from retrieved excerpts, with the policy named
7. Installment amounts quoted only from the pre-computed EMI options

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then add two free API keys to `.env`:
`GROQ_API_KEY` (console.groq.com — the brain) and
`GEMINI_API_KEY` (aistudio.google.com/apikey — RAG embeddings).

## Run

```
python src\main.py            text call (type as the customer)
python src\main.py --voice    voice call (speak; requires mic + speakers)
python src\report.py          bot performance report across logged calls
```

First run indexes the policy documents and (voice mode) downloads the ASR
model; both are cached afterwards.

## Tuning the voice pipeline (config.yaml)

| Knob | Too low | Too high |
|------|---------|----------|
| `vad.aggressiveness` (0–3) | background noise counts as speech | soft speakers get dropped |
| `endpointing.silence_after_speech_ms` | bot interrupts mid-sentence pauses | awkward dead air before every reply |
| `endpointing.min_speech_ms` | coughs become "speech" | short answers ("yes") get ignored |
| `endpointing.pre_speech_buffer_ms` | first syllable clipped | slight lag entering speech |
| `tts.rate` | robotic drawl | rushed, pushy collector |
| `asr.model_size` | mishears accents/numbers | slower replies |

## Measured performance

Per-turn latency from real logged calls (`src/report.py`):

| Stage | Typical |
|-------|---------|
| LLM (Groq) | 270–690ms |
| RAG retrieval | ~600ms |
| ASR transcription (`small`) | ~1.5s |
| TTS synthesis | ~2s |

Swapping the brain from Gemini to Groq cut LLM latency from 8–14s to under
700ms, which is what turned the demo from "waiting on hold" into a
conversation.

## Repo layout

```
config.yaml          voice pipeline tuning knobs
data/customers.json  mock account records (a bank API in production)
data/policies/       bank policy documents (the RAG corpus)
src/main.py          call loop (text or voice), handoff detection
src/brain.py         system briefing, guardrails, Groq call
src/rag.py           chunking, embeddings, retrieval
src/emi.py           deterministic reducing-balance EMI calculator
src/voice.py         TTS, VAD + endpointing state machine, ASR
src/records.py       customer record access (swappable for a real API)
src/calllog.py       per-turn transcript + latency logging
src/report.py        performance aggregation
logs/                call transcripts (gitignored)
```
