"""Thread-safe monotonic-timestamped ring buffer for multimodal alignment."""

from __future__ import annotations

from collections import deque
import threading
from typing import Callable, Deque, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class TemporalRingBuffer(Generic[T]):
    """Stores timestamped sensor/perception items ordered by monotonic nanoseconds."""

    def __init__(
        self,
        retention_ns: int = 4_000_000_000,
        max_items: int = 512,
        ts_getter: Optional[Callable[[T], int]] = None,
    ) -> None:
        self._retention_ns = retention_ns
        self._max_items = max_items
        self._ts_getter = ts_getter or (lambda item: int(getattr(item, "capture_mono_ns")))
        self._buf: Deque[Tuple[int, T]] = deque()
        self._lock = threading.RLock()

    def append(self, item: T, mono_ns: Optional[int] = None) -> None:
        ts = int(mono_ns if mono_ns is not None else self._ts_getter(item))
        with self._lock:
            self._buf.append((ts, item))
            cutoff = ts - self._retention_ns
            while len(self._buf) > self._max_items or (self._buf and self._buf[0][0] < cutoff):
                self._buf.popleft()

    def latest(self) -> Optional[T]:
        with self._lock:
            return self._buf[-1][1] if self._buf else None

    def query_window(self, start_ns: int, end_ns: int) -> List[T]:
        with self._lock:
            return [item for ts, item in self._buf if start_ns <= ts <= end_ns]

    def nearest(self, target_ns: int, max_delta_ns: Optional[int] = None) -> Tuple[Optional[T], Optional[int]]:
        with self._lock:
            if not self._buf:
                return None, None
            best_ts, best_item = min(self._buf, key=lambda pair: abs(pair[0] - target_ns))
            delta = abs(best_ts - target_ns)
            if max_delta_ns is not None and delta > max_delta_ns:
                return None, delta
            return best_item, delta

    def snapshot(self) -> List[T]:
        with self._lock:
            return [item for _, item in self._buf]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
