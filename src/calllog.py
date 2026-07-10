"""Call transcripts + per-turn latency logging (bot performance evidence).

One .jsonl file per call in logs/: first line is call metadata, every
following line is one turn with its stage timings in milliseconds.
"""

import json
import time
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"


def start_call(customer_name, mode):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"call_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    _append(path, {"customer": customer_name, "mode": mode, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    return path


def log_turn(path, user_text, reply, retrieved, timings, events=None):
    record = {
        "caller": user_text,
        "maya": reply,
        "retrieved": retrieved,
        "timings": timings,
    }
    if events:
        record["events"] = events
    _append(path, record)


def end_call(path, disposition):
    """Final line of every call: the outcome, for recovery analytics."""
    _append(path, {"disposition": disposition})


def _append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
