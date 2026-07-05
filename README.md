# Collections Voice Agent

A GenAI-powered debt-collections assistant for a fictional bank ("Horizon
Bank") — built end to end the way a Forward Deployed AI Engineer would:
LLM brain with hard guardrails, RAG over bank policy documents, a real
ASR/TTS voice layer, tunable VAD/endpointing configuration, and per-turn
performance logging.

> All customer data and policies in this repo are fictional.

## The pipeline

```
caller audio → VAD → ASR → endpointing → LLM + RAG → TTS → caller
              (webrtcvad) (faster-whisper)  (Gemini)   (edge-tts)
```

Every conversation turn runs this loop. The LLM never invents facts: customer
figures come only from the account record, policy answers come only from
retrieved policy text, and compliance-critical wording (the opening
disclosure) is scripted in code, not generated.

## What's implemented

- [x] Stage 1 — the brain on rails: text chat, mock customer records, LLM
      replies where every number comes from the record, never the model
- [x] Stage 2 — RAG: embeddings-based retrieval (`gemini-embedding-001`) over
      bank policy docs, chunked by section, cached index, per-turn injection
      with a calibrated similarity threshold, citations by policy name
- [x] Stage 3 — voice: neural TTS out (`edge-tts`), local ASR in
      (`faster-whisper`), push-free turn-taking
- [x] Stage 4 — deep configuration: webrtcvad VAD aggressiveness, endpointing
      silence windows, pre-speech buffering, and TTS prosody — all tunable in
      `config.yaml`, with documented failure symptoms per knob
- [x] Stage 5 — bot performance: per-turn stage latencies (ASR / retrieval /
      LLM / TTS) logged to JSONL transcripts, aggregated by `src/report.py`

## Guardrails (enforced in the system briefing, verified by scripted attacks)

1. Facts only from the record — no invented figures, no invented explanations
2. Identity check first; wrong person → reveal nothing
3. No threats or pressure, ever
4. Hardship or distress → stop collecting, offer a human agent
5. Bank-account scope only
6. Policy questions answered only from retrieved policy excerpts, with the
   policy named

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      (paste your Gemini API key into .env)
```

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

## Repo layout

```
config.yaml          all voice pipeline tuning knobs
data/customers.json  mock account records (would be the bank's API in production)
data/policies/       bank policy documents (the RAG corpus)
src/main.py          call loop (text or voice)
src/brain.py         system briefing + Gemini call
src/rag.py           chunking, embeddings, retrieval
src/voice.py         TTS, VAD + endpointing state machine, ASR
src/calllog.py       per-turn transcript + latency logging
src/report.py        performance aggregation
logs/                call transcripts (gitignored)
```
