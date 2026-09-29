"""Config loading and the end-of-turn decision, kept free of audio deps so the simulator can use them."""

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"

ENDPOINT = "endpoint (silence after speech)"
MAX_UTTERANCE = "max_utterance_s hit"
NO_SPEECH = "max_wait_s hit before any speech"


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Endpointer:
    """Fed one VAD verdict per frame; says when the caller's turn is over."""

    def __init__(self, frame_ms, silence_after_speech_ms, max_utterance_s, max_wait_s,
                 pre_speech_buffer_ms=0):
        self.endpoint_frames = silence_after_speech_ms // frame_ms
        self.max_frames = max_utterance_s * 1000 // frame_ms
        self.max_wait_frames = max_wait_s * 1000 // frame_ms
        self.pre_frames = pre_speech_buffer_ms // frame_ms
        self.in_speech = False
        self.speech_frames = 0
        self.silent_streak = 0
        self.recorded = 0  # frames kept, including the pre-speech buffer
        self.waited = 0

    def step(self, is_speech):
        """Returns None to keep listening, else the reason the turn ended."""
        if not self.in_speech:
            self.waited += 1
            if is_speech:
                self.in_speech = True
                self.speech_frames = 1
                self.recorded = min(self.waited, self.pre_frames)
                return None
            return NO_SPEECH if self.waited >= self.max_wait_frames else None

        self.recorded += 1
        if is_speech:
            self.speech_frames += 1
            self.silent_streak = 0
        else:
            self.silent_streak += 1
            if self.silent_streak >= self.endpoint_frames:
                return ENDPOINT
        if self.recorded >= self.max_frames:
            return MAX_UTTERANCE
        return None


def demo():
    ep = dict(frame_ms=30, silence_after_speech_ms=90, max_utterance_s=1, max_wait_s=1,
              pre_speech_buffer_ms=60)

    e = Endpointer(**ep)
    turn = [False, True, True, False, False, False]
    assert [e.step(s) for s in turn] == [None, None, None, None, None, ENDPOINT]
    assert e.speech_frames == 2

    e = Endpointer(**ep)
    assert [e.step(False) for _ in range(33)][-1] == NO_SPEECH

    e = Endpointer(**ep)
    while (reason := e.step(True)) is None:
        pass
    assert reason == MAX_UTTERANCE and e.recorded == 33

    print("endpointing ok")


if __name__ == "__main__":
    demo()
