"""Bot performance report: aggregates logs/*.jsonl into per-call and
per-stage latency stats.

Run:  .venv\\Scripts\\python.exe src\\report.py
"""

import json
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"

STAGES = ["asr_ms", "retrieval_ms", "llm_ms", "tts_ms"]


def main():
    files = sorted(LOG_DIR.glob("call_*.jsonl"))
    if not files:
        print("No call logs yet. Make a call first: python src/main.py")
        return

    all_turns = []
    dispositions = {}
    print("Calls:")
    for path in files:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = lines[0]
        disposition = "(none recorded)"
        if len(lines) > 1 and "disposition" in lines[-1]:
            disposition = lines[-1]["disposition"]
            lines = lines[:-1]
        dispositions[disposition] = dispositions.get(disposition, 0) + 1
        turns = [l for l in lines[1:] if "timings" in l]
        all_turns.extend(turns)
        totals = [sum(t["timings"].values()) for t in turns] or [0]
        print(
            f"  {path.name}  {meta['customer']:<14} {meta['mode']:<6} "
            f"turns={len(turns):<3} avg={sum(totals)/len(totals):.0f}ms  "
            f"outcome={disposition}"
        )

    print("\nCall outcomes:")
    for outcome, count in sorted(dispositions.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome:<22} {count}")

    print(f"\nPer-stage latency across {len(all_turns)} turns:")
    for stage in STAGES:
        values = [t["timings"][stage] for t in all_turns if stage in t["timings"]]
        if values:
            print(
                f"  {stage:<13} avg={sum(values)/len(values):>6.0f}ms   "
                f"min={min(values):>5}ms   max={max(values):>5}ms"
            )

    total_values = [sum(t["timings"].values()) for t in all_turns]
    if total_values:
        print(f"\nFull turn (caller stops talking -> reply done): avg={sum(total_values)/len(total_values):.0f}ms")


if __name__ == "__main__":
    main()
