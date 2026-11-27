"""Upgraded Streaming ASR (Zipformer Bilingual ONNX + NCNN fallback) + Silero VAD with Pre-Roll & AGC.

Fixes:
1. 0.5s audio pre-roll ring buffer flushed on speech start so beginning syllables are never clipped.
2. Proper Silero VAD queue draining (`while not vad.empty(): vad.pop()`) + reset after each utterance.
3. Trailing silence flush (0.4s) before Zipformer final readout so short 2-4 char commands decode accurately.
4. Live `last_rms` telemetry exposed to Web Console VU meter for both Thor USB mic and Mac browser mic.
"""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional
import uuid

import numpy as np

from egogaze.core.schemas import SpeechUtterance
from egogaze.sensors.live_devices import discover_audio_input_device

logger = logging.getLogger("egogaze.asr")


class StreamingASRWorker:
    """Supports both Thor USB microphone and Mac Browser streamed PCM audio through the same Zipformer+VAD core."""

    def __init__(self, speech_cfg: Dict[str, Any], sensor_cfg: Dict[str, Any], demo_mode: bool = False) -> None:
        self.speech_cfg = speech_cfg
        self.sensor_cfg = sensor_cfg
        self.demo_mode = demo_mode or not bool(speech_cfg.get("enabled", True))
        self.enable_thor_mic = bool(speech_cfg.get("enable_thor_mic", True)) and not demo_mode

        self.asr_engine_pref = str(speech_cfg.get("asr_engine", "zipformer_onnx"))
        self.zipformer_dir = Path(speech_cfg.get("zipformer_model_dir", ""))
        self.ncnn_dir = Path(speech_cfg.get("asr_model_dir", ""))
        self.vad_path = Path(speech_cfg.get("vad_model_path", ""))
        self.vad_threshold = float(speech_cfg.get("vad_threshold", 0.25))
        self.agc_target_rms = float(speech_cfg.get("agc_target_rms", 0.10))
        self.agc_max_gain = float(speech_cfg.get("agc_max_gain", 12.0))
        self.hotwords = speech_cfg.get("hotwords_file", "")
        self.num_threads = int(speech_cfg.get("asr_threads", 4))
        self.silence_chunks_end = int(speech_cfg.get("silence_chunks_end", 10))

        self.input_sr = int(sensor_cfg.get("mic_samplerate", 44100))
        self.target_sr = int(sensor_cfg.get("target_samplerate", 16000))
        self.chunk_frames = int(sensor_cfg.get("audio_chunk_frames", 1024))
        self.mic_name = str(sensor_cfg.get("mic_name", "UGREEN"))

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._audio_q: queue.Queue = queue.Queue(maxsize=256)

        self._backend_kind = "none"  # zipformer_onnx | sherpa_ncnn
        self._onnx_recognizer = None
        self._onnx_stream = None
        self._ncnn_recognizer = None
        self._vad = None
        self.available = False
        self.browser_mic_active = False
        self.last_rms: float = 0.0
        self.is_speaking: bool = False

        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_final: Optional[Callable[[SpeechUtterance], None]] = None
        self._on_barge_in: Optional[Callable[[], None]] = None

    @property
    def backend_name(self) -> str:
        return self._backend_kind

    def start(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[SpeechUtterance], None],
        on_barge_in: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_barge_in = on_barge_in
        if not bool(self.speech_cfg.get("enabled", True)):
            return
        if not self._init_models():
            return
        self._stop.clear()
        t_proc = threading.Thread(target=self._process_loop, name="ASRProcessLoop", daemon=True)
        t_proc.start()
        self._threads = [t_proc]
        if self.enable_thor_mic:
            t_mic = threading.Thread(target=self._thor_mic_loop, name="ThorMicCapture", daemon=True)
            t_mic.start()
            self._threads.append(t_mic)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    def feed_browser_pcm16(self, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
        """Accept 16-bit signed mono PCM chunks streamed from the Mac browser microphone."""
        if not self.available or not pcm_bytes:
            return
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != self.target_sr and arr.size > 0:
            from scipy import signal

            n = max(1, int(len(arr) * self.target_sr / float(sample_rate)))
            arr = signal.resample(arr, n).astype(np.float32)
        # Slice into uniform 512-sample (32ms) chunks matching Silero VAD window size
        step = 512
        now_ns = time.monotonic_ns()
        for i in range(0, len(arr), step):
            sub = arr[i : i + step]
            if sub.size < 160:
                continue
            try:
                self._audio_q.put_nowait((sub, now_ns, "mac_mic"))
            except queue.Full:
                break

    def _apply_agc(self, chunk: np.ndarray) -> np.ndarray:
        """Gentle Automatic Gain Control so distant speech is picked up cleanly without amplifying silence."""
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-10))
        self.last_rms = round(rms, 5)
        if rms < 0.0012:
            return chunk
        gain = min(self.agc_max_gain, self.agc_target_rms / max(rms, 1e-4))
        gain = max(1.0, gain)
        return np.tanh(chunk * gain).astype(np.float32)

    def _init_models(self) -> bool:
        if not self.vad_path.exists():
            logger.info("VAD model not found at %s; ASR disabled", self.vad_path)
            return False

        try:
            import sherpa_onnx

            vad_cfg = sherpa_onnx.VadModelConfig()
            vad_cfg.silero_vad.model = str(self.vad_path)
            vad_cfg.silero_vad.threshold = self.vad_threshold
            vad_cfg.silero_vad.min_silence_duration = 0.35
            vad_cfg.silero_vad.min_speech_duration = 0.08
            vad_cfg.sample_rate = self.target_sr
            self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30)
        except Exception as exc:
            logger.warning("Silero VAD init failed: %s", exc)
            return False

        # 1. Prefer Sherpa-ONNX Streaming Zipformer Bilingual
        if self.asr_engine_pref == "zipformer_onnx" and self.zipformer_dir.exists():
            try:
                import sherpa_onnx

                enc = self.zipformer_dir / "encoder-epoch-99-avg-1.int8.onnx"
                dec = self.zipformer_dir / "decoder-epoch-99-avg-1.int8.onnx"
                joi = self.zipformer_dir / "joiner-epoch-99-avg-1.int8.onnx"
                tok = self.zipformer_dir / "tokens.txt"
                if enc.exists() and dec.exists() and joi.exists() and tok.exists():
                    self._onnx_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                        tokens=str(tok),
                        encoder=str(enc),
                        decoder=str(dec),
                        joiner=str(joi),
                        num_threads=self.num_threads,
                        sample_rate=self.target_sr,
                        feature_dim=80,
                        decoding_method="greedy_search",
                        provider="cpu",
                    )
                    self._onnx_stream = self._onnx_recognizer.create_stream()
                    self._backend_kind = "zipformer_onnx"
                    self.available = True
                    logger.info(
                        "StreamingASRWorker loaded Zipformer-ONNX Bilingual + Silero VAD (thr=%.2f)",
                        self.vad_threshold,
                    )
                    return True
            except Exception as exc:
                logger.warning("Zipformer-ONNX init failed (%s); falling back to Sherpa-NCNN", exc)

        # 2. Fallback: Sherpa-NCNN Conv-Emformer
        if self.ncnn_dir.exists():
            try:
                import sherpa_ncnn

                hw = self.hotwords if (self.hotwords and Path(self.hotwords).exists()) else ""
                self._ncnn_recognizer = sherpa_ncnn.Recognizer(
                    tokens=str(self.ncnn_dir / "tokens.txt"),
                    encoder_param=str(self.ncnn_dir / "encoder_jit_trace-pnnx.ncnn.param"),
                    encoder_bin=str(self.ncnn_dir / "encoder_jit_trace-pnnx.ncnn.bin"),
                    decoder_param=str(self.ncnn_dir / "decoder_jit_trace-pnnx.ncnn.param"),
                    decoder_bin=str(self.ncnn_dir / "decoder_jit_trace-pnnx.ncnn.bin"),
                    joiner_param=str(self.ncnn_dir / "joiner_jit_trace-pnnx.ncnn.param"),
                    joiner_bin=str(self.ncnn_dir / "joiner_jit_trace-pnnx.ncnn.bin"),
                    decoding_method="modified_beam_search",
                    num_threads=self.num_threads,
                    hotwords_file=hw,
                )
                self._backend_kind = "sherpa_ncnn"
                self.available = True
                logger.info("StreamingASRWorker loaded Sherpa-NCNN fallback + Silero VAD")
                return True
            except Exception as exc:
                logger.warning("Sherpa-NCNN init failed: %s", exc)
        return False

    def _accept_and_decode(self, chunk: np.ndarray) -> str:
        if self._backend_kind == "zipformer_onnx":
            self._onnx_stream.accept_waveform(self.target_sr, chunk)
            while self._onnx_recognizer.is_ready(self._onnx_stream):
                self._onnx_recognizer.decode_stream(self._onnx_stream)
            return self._onnx_recognizer.get_result(self._onnx_stream).strip()
        if self._backend_kind == "sherpa_ncnn":
            self._ncnn_recognizer.accept_waveform(self.target_sr, chunk)
            return self._ncnn_recognizer.text.strip()
        return ""

    def _reset_decoder(self) -> None:
        if self._backend_kind == "zipformer_onnx" and self._onnx_recognizer is not None:
            self._onnx_recognizer.reset(self._onnx_stream)
        elif self._backend_kind == "sherpa_ncnn" and self._ncnn_recognizer is not None:
            self._ncnn_recognizer.reset()
        # CRITICAL: Always drain finished VAD speech segments so internal 30s buffer never locks up!
        if self._vad is not None:
            while not self._vad.empty():
                self._vad.pop()

    def _thor_mic_loop(self) -> None:
        try:
            from scipy import signal
            import sounddevice as sd
        except Exception as exc:
            logger.warning("Audio libs missing (%s); Thor mic disabled", exc)
            return

        dev_idx = discover_audio_input_device(self.mic_name)
        kwargs: Dict[str, Any] = {"samplerate": self.input_sr, "channels": 1, "dtype": "float32"}
        if dev_idx is not None:
            kwargs["device"] = dev_idx

        # Accumulate into 512-sample (32ms) chunks at 16kHz for optimal Silero VAD accuracy
        resample_buf = np.zeros(0, dtype=np.float32)

        try:
            with sd.InputStream(**kwargs) as stream:
                logger.info("Thor USB mic opened (device=%s, sr=%d->%d)", dev_idx, self.input_sr, self.target_sr)
                while not self._stop.is_set():
                    data, _ = stream.read(self.chunk_frames)
                    if self.browser_mic_active:
                        continue
                    mono_ns = time.monotonic_ns()
                    chunk = np.mean(data, axis=1, dtype=np.float32) if data.ndim == 2 else data.astype(np.float32)
                    if self.input_sr != self.target_sr:
                        n = int(len(chunk) * self.target_sr / self.input_sr)
                        chunk = signal.resample(chunk, n).astype(np.float32)
                    resample_buf = np.concatenate([resample_buf, chunk])
                    while resample_buf.size >= 512:
                        frame512 = resample_buf[:512].copy()
                        resample_buf = resample_buf[512:]
                        try:
                            self._audio_q.put_nowait((frame512, mono_ns, "thor_mic"))
                        except queue.Full:
                            pass
        except Exception as exc:
            logger.warning("Thor mic stream stopped: %s", exc)

    def _process_loop(self) -> None:
        speaking = False
        silence_count = 0
        speech_start_ns = 0
        last_partial = ""
        active_source = "voice"
        # 0.55s pre-roll buffer (17 * 32ms) so beginning syllables before VAD trigger are preserved!
        pre_roll: Deque[np.ndarray] = deque(maxlen=17)

        while not self._stop.is_set():
            try:
                raw_chunk, mono_ns, source = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            chunk = self._apply_agc(raw_chunk)
            self._vad.accept_waveform(chunk)
            vad_speech = self._vad.is_speech_detected()
            # Also drain completed VAD segments proactively to prevent buffer accumulation
            while not self._vad.empty():
                self._vad.pop()

            if vad_speech and not speaking:
                speaking = True
                self.is_speaking = True
                silence_count = 0
                speech_start_ns = mono_ns
                active_source = source
                if self._on_barge_in:
                    self._on_barge_in()
                # Flush pre-roll into recognizer FIRST so no initial words are lost!
                for pr in pre_roll:
                    self._accept_and_decode(pr)
                pre_roll.clear()
                partial = self._accept_and_decode(chunk)
                if partial and partial != last_partial:
                    last_partial = partial
                    if self._on_partial:
                        self._on_partial(partial)
                continue

            if not speaking:
                pre_roll.append(chunk)
                continue

            # Currently inside an active speech utterance
            partial = self._accept_and_decode(chunk)
            if partial and partial != last_partial:
                last_partial = partial
                if self._on_partial:
                    self._on_partial(partial)

            if vad_speech:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >= self.silence_chunks_end:
                    speaking = False
                    self.is_speaking = False
                    silence_count = 0
                    # Flush 0.4s trailing zeros so Zipformer emits final characters cleanly
                    final_text = self._accept_and_decode(np.zeros(int(self.target_sr * 0.4), dtype=np.float32))
                    final_ns = time.monotonic_ns()
                    self._reset_decoder()
                    pre_roll.clear()
                    last_partial = ""
                    if self._on_partial:
                        self._on_partial("")
                    if len(final_text) >= 2 and self._on_final:
                        logger.info("ASR Final (%s): %r", active_source, final_text)
                        utt = SpeechUtterance(
                            utterance_id=uuid.uuid4().hex[:8],
                            text=final_text,
                            is_final=True,
                            speech_start_mono_ns=speech_start_ns,
                            speech_end_mono_ns=mono_ns,
                            final_mono_ns=final_ns,
                            source=active_source,
                            vad_ms=round((mono_ns - speech_start_ns) / 1e6, 1),
                            asr_ms=round((final_ns - mono_ns) / 1e6, 1),
                        )
                        self._on_final(utt)
