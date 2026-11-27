"""Streaming sentence-chunked Kokoro TTS with Browser WAV streaming, optional Thor playback & Barge-in."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import wave

import numpy as np

logger = logging.getLogger("egogaze.tts")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;\n])")


class StreamingSentenceChunker:
    """Buffers streaming LLM/VLM tokens and yields speakable clauses/sentences early for low TTFA."""

    def __init__(self, min_first_chars: int = 6, min_chunk_chars: int = 12) -> None:
        self.min_first_chars = min_first_chars
        self.min_chunk_chars = min_chunk_chars
        self._buf = ""
        self._emitted_first = False

    def push(self, token: str) -> List[str]:
        if not token:
            return []
        self._buf += token
        out: List[str] = []

        if not self._emitted_first and len(self._buf) >= self.min_first_chars:
            m = re.search(r"[，,。！？；!?;]", self._buf)
            if m and m.end() >= self.min_first_chars:
                chunk = self._buf[: m.end()].strip()
                self._buf = self._buf[m.end() :]
                if chunk:
                    out.append(chunk)
                    self._emitted_first = True

        parts = _SENTENCE_SPLIT_RE.split(self._buf)
        if len(parts) > 1:
            for p in parts[:-1]:
                s = p.strip()
                if s:
                    out.append(s)
                    self._emitted_first = True
            self._buf = parts[-1]
        return out

    def flush(self) -> Optional[str]:
        rem = self._buf.strip()
        self._buf = ""
        self._emitted_first = False
        return rem or None


def _samples_to_wav_b64(samples: np.ndarray, sample_rate: int) -> str:
    arr = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


class StreamingTTSWorker:
    """Synthesizes Kokoro TTS on Jetson Thor and streams WAV audio to Mac browser (and/or local Thor speaker)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("tts_enabled", True))
        self.play_local = bool(cfg.get("tts_play_local", False))
        self.model_dir = Path(cfg.get("tts_model_dir", ""))
        self.num_threads = int(cfg.get("tts_threads", 2))
        self.sid = int(cfg.get("tts_sid", 18))
        self.speed = float(cfg.get("tts_speed", 1.08))
        self.provider = str(cfg.get("tts_provider", "cpu"))
        self.target_sr = cfg.get("tts_target_samplerate", 24000)
        self.output_device = cfg.get("tts_output_device", None)

        self._tts = None
        self._model_sr: int = 24000
        self._playback_sr: int = int(self.target_sr or 24000)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._barge_in = threading.Event()
        self._speaking = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_first_audio: Optional[Callable[[str, float], None]] = None
        self._on_wav_ready: Optional[Callable[[str, str, str], None]] = None
        self.available = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def start(
        self,
        on_first_audio: Optional[Callable[[str, float], None]] = None,
        on_wav_ready: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self._on_first_audio = on_first_audio
        self._on_wav_ready = on_wav_ready
        if not self.enabled:
            return
        self._init_model()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="StreamingTTSWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.interrupt()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def interrupt(self) -> None:
        """Barge-in: immediately abort current speech and drain queued sentences."""
        self._barge_in.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def enqueue_sentence(self, utterance_id: str, text: str, is_first: bool = False) -> None:
        if not self.available or not text.strip():
            return
        # Strip bracketed markers and stage directions like [1] or （笑） for natural speech
        clean = re.sub(r"\[[^\]]*\]|（[^）]*）|\([^)]*\)", "", text).strip()
        if not clean:
            return
        self._barge_in.clear()
        self._q.put((utterance_id, clean, is_first))

    def _init_model(self) -> None:
        if not self.model_dir.exists():
            logger.info("Kokoro TTS model dir not found (%s); TTS disabled", self.model_dir)
            return
        try:
            import sherpa_onnx

            lexicons = [
                str(self.model_dir / f)
                for f in ("lexicon-us-en.txt", "lexicon-zh.txt")
                if (self.model_dir / f).exists()
            ]
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(self.model_dir / "model.onnx"),
                        voices=str(self.model_dir / "voices.bin"),
                        tokens=str(self.model_dir / "tokens.txt"),
                        data_dir=str(self.model_dir / "espeak-ng-data"),
                        lexicon=",".join(lexicons),
                    ),
                    provider=self.provider,
                    debug=False,
                    num_threads=self.num_threads,
                ),
                rule_fsts="",
                max_num_sentences=1,
            )
            if not cfg.validate():
                logger.warning("Kokoro TTS config invalid at %s", self.model_dir)
                return
            self._tts = sherpa_onnx.OfflineTts(cfg)
            self._model_sr = self._tts.sample_rate
            self._playback_sr = int(self.target_sr or self._model_sr)
            self.available = True
            logger.info("Kokoro TTS loaded (sr=%d, browser_stream=True, local_play=%s)", self._model_sr, self.play_local)
        except Exception as exc:
            logger.warning("Kokoro TTS unavailable (%s)", exc)

    def _play_local_samples(self, samples: np.ndarray) -> None:
        try:
            from scipy import signal
            import sounddevice as sd

            arr = np.asarray(samples, dtype=np.float32)
            if self._playback_sr != self._model_sr and len(arr) > 0:
                n = round(len(arr) * float(self._playback_sr) / self._model_sr)
                arr = signal.resample(arr, n).astype(np.float32)

            step = max(int(self._playback_sr * 0.05), 256)
            kwargs: Dict[str, Any] = {"samplerate": self._playback_sr, "channels": 1, "dtype": "float32"}
            if self.output_device is not None:
                kwargs["device"] = self.output_device

            with sd.OutputStream(**kwargs) as stream:
                for i in range(0, len(arr), step):
                    if self._barge_in.is_set() or self._stop.is_set():
                        break
                    stream.write(arr[i : i + step].reshape(-1, 1))
        except Exception as exc:
            logger.warning("Local TTS playback failed: %s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                utterance_id, text, is_first = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop.is_set() or self._barge_in.is_set() or self._tts is None:
                continue

            self._speaking.set()
            try:
                audio = self._tts.generate(text, sid=self.sid, speed=self.speed)
                if self._stop.is_set() or self._barge_in.is_set():
                    continue
                if len(audio.samples) > 0:
                    if is_first and self._on_first_audio is not None:
                        self._on_first_audio(utterance_id, time.monotonic_ns())
                    # Stream WAV to Mac browser speakers via SSE
                    if self._on_wav_ready is not None:
                        wav_b64 = _samples_to_wav_b64(audio.samples, audio.sample_rate)
                        self._on_wav_ready(utterance_id, text, wav_b64)
                    if self.play_local:
                        self._play_local_samples(audio.samples)
            except Exception as exc:
                logger.warning("TTS synthesis failed: %s", exc)
            finally:
                self._speaking.clear()
