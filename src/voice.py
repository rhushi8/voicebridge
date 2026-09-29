"""All tunables live in config.yaml."""

import asyncio
import collections
import tempfile
import time
from pathlib import Path

import edge_tts
import numpy as np
import sounddevice as sd
import webrtcvad

from endpointing import NO_SPEECH, Endpointer

SAMPLE_RATE = 16000


def load_asr_model(cfg):
    from faster_whisper import WhisperModel

    return WhisperModel(cfg["asr"]["model_size"], device="cpu", compute_type="int8")


def speak(text, cfg):
    tts = cfg["tts"]
    mp3_path = Path(tempfile.gettempdir()) / f"maya_{time.time_ns()}.mp3"

    async def _synth():
        communicate = edge_tts.Communicate(
            text,
            voice=tts["voice"],
            rate=tts["rate"],
            pitch=tts["pitch"],
            volume=tts["volume"],
        )
        await communicate.save(str(mp3_path))

    asyncio.run(_synth())

    import pygame

    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(str(mp3_path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()
    mp3_path.unlink(missing_ok=True)


def listen(cfg, asr_model):
    """Waits for speech, ends the turn after silence_after_speech_ms of quiet."""
    vad_cfg, ep = cfg["vad"], cfg["endpointing"]
    vad = webrtcvad.Vad(vad_cfg["aggressiveness"])
    frame_ms = vad_cfg["frame_ms"]
    frame_samples = SAMPLE_RATE * frame_ms // 1000

    turn = Endpointer(
        frame_ms,
        ep["silence_after_speech_ms"],
        ep["max_utterance_s"],
        ep["max_wait_s"],
        ep["pre_speech_buffer_ms"],
    )
    pre_buffer = collections.deque(maxlen=turn.pre_frames)
    recorded = []

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples
    ) as stream:
        while True:
            frame, _ = stream.read(frame_samples)
            frame = bytes(frame)
            was_in_speech = turn.in_speech
            reason = turn.step(vad.is_speech(frame, SAMPLE_RATE))

            if was_in_speech:
                recorded.append(frame)
            else:
                pre_buffer.append(frame)
                if turn.in_speech:
                    recorded.extend(pre_buffer)
            if reason:
                break

    if reason == NO_SPEECH or turn.speech_frames < ep["min_speech_ms"] // frame_ms:
        return ""

    audio = np.frombuffer(b"".join(recorded), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = asr_model.transcribe(
        audio, language=cfg["asr"]["language"], beam_size=cfg["asr"]["beam_size"]
    )
    return " ".join(s.text.strip() for s in segments).strip()
