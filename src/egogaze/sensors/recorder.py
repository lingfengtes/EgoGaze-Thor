"""Synchronized multimodal session recorder & deterministic offline replayer."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("egogaze.recorder")


class SessionRecorder:
    """Records synchronized scene frames (MP4) + structured events (JSONL) with monotonic timestamps."""

    def __init__(self, base_dir: str | Path = "data/recordings") -> None:
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self._active = False
        self._session_dir: Optional[Path] = None
        self._events_fp = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._frame_idx = 0
        self._t0_ns = 0

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    def start(self, tag: str = "session") -> Path:
        with self._lock:
            if self._active and self._session_dir:
                return self._session_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = self.base_dir / f"{stamp}_{tag}"
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._events_fp = (self._session_dir / "events.jsonl").open("w", encoding="utf-8")
            self._video_writer = None
            self._frame_idx = 0
            self._t0_ns = time.monotonic_ns()
            self._active = True
            logger.info("Started multimodal recording: %s", self._session_dir)
            return self._session_dir

    def write_frame(self, frame_bgr: np.ndarray, mono_ns: int) -> None:
        with self._lock:
            if not self._active or self._session_dir is None:
                return
            if self._video_writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(str(self._session_dir / "scene.mp4"), fourcc, 20.0, (w, h))
            self._video_writer.write(frame_bgr)
            self._write_jsonl({"type": "frame", "frame_idx": self._frame_idx, "rel_mono_ns": mono_ns - self._t0_ns})
            self._frame_idx += 1

    def write_event(self, event_type: str, payload: Dict[str, Any], mono_ns: Optional[int] = None) -> None:
        with self._lock:
            if not self._active:
                return
            t_ns = time.monotonic_ns() if mono_ns is None else mono_ns
            self._write_jsonl({"type": event_type, "rel_mono_ns": t_ns - self._t0_ns, "payload": payload})

    def _write_jsonl(self, obj: Dict[str, Any]) -> None:
        if self._events_fp:
            self._events_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._events_fp.flush()

    def stop(self) -> Optional[Path]:
        with self._lock:
            if not self._active:
                return None
            self._active = False
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            if self._events_fp is not None:
                self._events_fp.close()
                self._events_fp = None
            out = self._session_dir
            logger.info("Stopped multimodal recording: %s (%d frames)", out, self._frame_idx)
            return out


class SessionReplayer:
    """Loads a recorded session directory for deterministic offline evaluation & replay."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.events_path = self.session_dir / "events.jsonl"
        self.video_path = self.session_dir / "scene.mp4"

    def load_events(self) -> List[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: List[Dict[str, Any]] = []
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def iter_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        if not self.video_path.exists():
            return
        cap = cv2.VideoCapture(str(self.video_path))
        idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()
