"""The voice layer: Maya's mouth (TTS), ears (ASR), and turn-taking sense
(VAD + endpointing).

All tunable behavior lives in config.yaml -- this module just reads it.
"""

import asyncio
import collections
import tempfile
import time
from pathlib import Path

import edge_tts
import numpy as np
import sounddevice as sd
import webrtcvad
import yaml

CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
SAMPLE_RATE = 16000


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_asr_model(cfg):
    from faster_whisper import WhisperModel

    return WhisperModel(cfg["asr"]["model_size"], device="cpu", compute_type="int8")


def speak(text, cfg):
    """Synthesize text with the configured voice/prosody and play it."""
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
    """Record the caller until they finish their turn, then transcribe.

    The turn-taking state machine:
      waiting  -- no speech yet; keep a small rolling pre-speech buffer so the
                  first syllable isn't lost. Give up after max_wait_s.
      speaking -- collect audio. Every silent frame counts toward the endpoint;
                  silence_after_speech_ms of quiet means the caller is done.
    """
    vad_cfg, ep = cfg["vad"], cfg["endpointing"]
    vad = webrtcvad.Vad(vad_cfg["aggressiveness"])
    frame_ms = vad_cfg["frame_ms"]
    frame_samples = SAMPLE_RATE * frame_ms // 1000

    endpoint_frames = ep["silence_after_speech_ms"] // frame_ms
    min_speech_frames = ep["min_speech_ms"] // frame_ms
    max_frames = ep["max_utterance_s"] * 1000 // frame_ms
    max_wait_frames = ep["max_wait_s"] * 1000 // frame_ms
    pre_buffer = collections.deque(maxlen=ep["pre_speech_buffer_ms"] // frame_ms)

    recorded = []
    speech_frames = 0
    silent_streak = 0
    in_speech = False
    frames_waited = 0

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples
    ) as stream:
        while True:
            frame, _ = stream.read(frame_samples)
            frame = bytes(frame)
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not in_speech:
                pre_buffer.append(frame)
                frames_waited += 1
                if is_speech:
                    in_speech = True
                    recorded.extend(pre_buffer)
                    speech_frames = 1
                elif frames_waited >= max_wait_frames:
                    return ""
            else:
                recorded.append(frame)
                if is_speech:
                    speech_frames += 1
                    silent_streak = 0
                else:
                    silent_streak += 1
                    if silent_streak >= endpoint_frames:
                        break
                if len(recorded) >= max_frames:
                    break

    if speech_frames < min_speech_frames:
        return ""

    audio = np.frombuffer(b"".join(recorded), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = asr_model.transcribe(
        audio, language=cfg["asr"]["language"], beam_size=cfg["asr"]["beam_size"]
    )
    return " ".join(s.text.strip() for s in segments).strip()
