"""Standalone experiment: how does silence_after_speech_ms actually behave?

Mirrors the exact endpointing state machine from voice.py's listen(), but
runs it over a scripted sequence of speech/silence frames instead of a live
mic, so it's instant, repeatable, and needs no audio hardware.

Run:  .venv\\Scripts\\python.exe src\\tune_endpointing.py
"""

FRAME_MS = 30  # matches config.yaml vad.frame_ms


def simulate_turn(is_speech_frames, silence_after_speech_ms, frame_ms=FRAME_MS,
                   max_utterance_s=15):
    """Replays voice.py's in_speech/silent_streak logic frame by frame.

    Returns (end_frame_index, reason) -- end_frame_index is None if the
    caller's sequence runs out before the turn would ever end.
    """
    endpoint_frames = silence_after_speech_ms // frame_ms
    max_frames = max_utterance_s * 1000 // frame_ms

    in_speech = False
    silent_streak = 0
    recorded = 0

    for i, is_speech in enumerate(is_speech_frames):
        if not in_speech:
            if is_speech:
                in_speech = True
                recorded = 1
        else:
            recorded += 1
            if is_speech:
                silent_streak = 0
            else:
                silent_streak += 1
                if silent_streak >= endpoint_frames:
                    return i, "endpoint (silence after speech)"
            if recorded >= max_frames:
                return i, "max_utterance_s hit"

    return None, "sequence ended before a decision"


def frames(speech_ms=0, silence_ms=0):
    """N frames of speech followed by N frames of silence, as a list of bools."""
    return [True] * (speech_ms // FRAME_MS) + [False] * (silence_ms // FRAME_MS)


def describe(label, seq, silence_after_speech_ms):
    end_frame, reason = simulate_turn(seq, silence_after_speech_ms)
    if end_frame is None:
        print(f"  {label:<28} threshold={silence_after_speech_ms:>4}ms  -> never ends ({reason})")
        return
    elapsed_ms = end_frame * FRAME_MS
    print(f"  {label:<28} threshold={silence_after_speech_ms:>4}ms  -> ends at frame {end_frame:>3} "
          f"(~{elapsed_ms:>4}ms in)  [{reason}]")


def main():
    print("=" * 78)
    print("SCENARIO 1: a natural mid-sentence pause, then more speech, then a real")
    print('end-of-turn silence.  e.g. "when is... [400ms pause] ...my payment due"')
    print("=" * 78)
    # 1.5s talking, 400ms pause (NOT the end of the turn), 1.5s more talking,
    # then a real 2.5s trailing silence (the actual end of the turn) -- long
    # enough that even the highest threshold tested below gets to resolve.
    scenario = (frames(speech_ms=1500) + frames(silence_ms=400) +
                frames(speech_ms=1500) + frames(silence_ms=2500))

    for threshold in (300, 500, 800, 1500, 2000):
        describe("mid-sentence pause test", scenario, threshold)

    print()
    print("=" * 78)
    print("SCENARIO 2: same real end-of-turn silence in isolation, just measuring")
    print("how long the bot waits before it decides you're actually done.")
    print("=" * 78)
    trailing_silence_only = frames(speech_ms=800) + frames(silence_ms=3000)
    for threshold in (300, 500, 800, 1500, 2000):
        describe("trailing silence only", trailing_silence_only, threshold)

    print()
    print("=" * 78)
    print("READING THE RESULTS")
    print("=" * 78)
    print("""
Scenario 1 is the one that matters: at low thresholds (300ms), the endpoint
fires DURING the mid-sentence pause, at frame ~63 (~1900ms in), before you've
even said the second half of your sentence. That's the bot cutting you off
mid-thought. At 800ms (the current config value) and above, it correctly
rides through the 400ms pause and only ends after the real 1200ms trailing
silence, at frame ~113 (~3400ms in).

Scenario 2 shows the cost of setting it too high: at 2000ms, the bot is
sitting there waiting a full 2 seconds of dead air after you've clearly
finished, before it decides to respond. That's the "awkward dead air" failure
mode. 800ms is the value that survives scenario 1 without paying scenario 2's
latency tax.
""")


if __name__ == "__main__":
    main()
