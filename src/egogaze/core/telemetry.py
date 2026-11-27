"""Real-time FPS and end-to-end latency telemetry tracking."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque, Dict, List

from egogaze.core.schemas import PipelineLatencyTrace


class RateMeter:
    """Sliding-window event rate (Hz / FPS) meter."""

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_s = window_s
        self._ticks: Deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        with self._lock:
            self._ticks.append(t)
            cutoff = t - self._window_s
            while self._ticks and self._ticks[0] < cutoff:
                self._ticks.popleft()

    @property
    def hz(self) -> float:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window_s
            while self._ticks and self._ticks[0] < cutoff:
                self._ticks.popleft()
            if len(self._ticks) < 2:
                return float(len(self._ticks)) / max(self._window_s, 1e-6)
            span = max(self._ticks[-1] - self._ticks[0], 1e-3)
            return (len(self._ticks) - 1) / span


class TelemetryCollector:
    """Aggregates module rates and recent latency traces for Web dashboard."""

    def __init__(self, max_traces: int = 30) -> None:
        self.gaze_rate = RateMeter()
        self.vision_rate = RateMeter()
        self.scene_cam_rate = RateMeter()
        self._traces: Deque[PipelineLatencyTrace] = deque(maxlen=max_traces)
        self._lock = threading.Lock()

    def record_trace(self, trace: PipelineLatencyTrace) -> None:
        with self._lock:
            self._traces.append(trace)

    def recent_traces(self) -> List[Dict]:
        with self._lock:
            return [t.model_dump() for t in self._traces]

    def summary(self) -> Dict[str, float]:
        return {
            "gaze_hz": round(self.gaze_rate.hz, 1),
            "vision_fps": round(self.vision_rate.hz, 1),
            "camera_fps": round(self.scene_cam_rate.hz, 1),
        }
