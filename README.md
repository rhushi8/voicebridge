# Collections Voice Agent

A GenAI-powered debt-collections assistant for a fictional bank — built to mirror a
real Forward Deployed AI Engineer workflow end to end: LLM brain, RAG over bank
policy documents, ASR/TTS voice layer, and tunable VAD/endpointing configuration.

> All customer data in this repo is fictional.

## Why this exists

Conversational collections is one of the hardest voice-AI problems: real-time
latency budgets, stressed callers, and zero tolerance for invented numbers.
This project builds that pipeline station by station:

```
caller audio → VAD → ASR → endpointing → LLM + RAG → TTS → caller
```

## Roadmap

- [x] Stage 0 — project skeleton, Python 3.12, git
- [x] Stage 1 — the brain on rails: text chat, mock customer records, LLM replies
      (every number comes from the record, never from the model)
- [x] Stage 2 — RAG: embeddings-based retrieval over bank policy docs;
      the closest policy sections are injected into the briefing every turn
- [ ] Stage 3 — voice: TTS out, ASR in
- [ ] Stage 4 — deep config: VAD thresholds + endpointing tuning in config
- [ ] Stage 5 — bot performance: per-turn latency logs, transcripts, metrics

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   (then paste your API key into .env)
python src/main.py
```
