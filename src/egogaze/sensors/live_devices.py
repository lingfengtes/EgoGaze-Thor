"""Hardware device discovery & hot-plug-resilient multi-camera capture for Jetson Thor.

Key reliability features:
1. Pure-sysfs `/sys/class/video4linux/video*/{name,index}` discovery (<0.5ms, immune to missing /dev/video0).
2. Automatic USB hot-plug recovery: when head/cable movement causes a USB UVC disconnect and Linux
   re-enumerates /dev/videoX to new indices, each camera thread immediately releases the stale fd,
   re-discovers the new /dev/videoX node, and re-opens the stream seamlessly (~1s recovery).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("egogaze.sensors")


def _sysfs_discover_camera(target_name: str) -> Optional[str]:
    """Fast, zero-subprocess V4L2 capture node discovery via /sys/class/video4linux."""
    root = Path("/sys/class/video4linux")
    if not root.exists():
        return None

    matches_idx0: List[Tuple[int, str]] = []
    matches_other: List[Tuple[int, str]] = []

    for node in root.glob("video*"):
        suffix = node.name.removeprefix("video")
        if not suffix.isdigit():
            continue
        dev_num = int(suffix)
        try:
            name = (node / "name").read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if target_name not in name:
            continue
        dev_path = f"/dev/video{dev_num}"
        try:
            idx = (node / "index").read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            idx = ""
        if idx == "0":
            matches_idx0.append((dev_num, dev_path))
        else:
            matches_other.append((dev_num, dev_path))

    if matches_idx0:
        matches_idx0.sort(key=lambda x: x[0])
        return matches_idx0[0][1]
    if matches_other:
        matches_other.sort(key=lambda x: x[0])
        return matches_other[0][1]
    return None


def _is_video_capture_node(device_path: str) -> bool:
    try:
        res = subprocess.run(
            ["v4l2-ctl", "--device", device_path, "--all"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        txt = res.stdout + res.stderr
        return "Device Caps" in txt and "Video Capture" in txt
    except Exception:
        return False


def discover_v4l2_camera(target_name: str, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve current /dev/videoX dynamically (sysfs primary -> v4l2-ctl secondary -> fallback)."""
    sysfs_path = _sysfs_discover_camera(target_name)
    if sysfs_path:
        return sysfs_path
    try:
        res = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=3)
        if res.stdout.strip():
            for block in res.stdout.strip().split("\n\n"):
                lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
                if not lines or target_name not in lines[0]:
                    continue
                candidates = [ln for ln in lines[1:] if ln.startswith("/dev/video")]
                for cand in candidates:
                    if _is_video_capture_node(cand):
                        return cand
                if candidates:
                    return candidates[0]
    except Exception:
        pass
    if fallback and Path(fallback).exists():
        return fallback
    return None


def _dev_node_exists(device_path: Optional[str]) -> bool:
    if not device_path:
        return False
    if device_path.startswith("/dev/video"):
        name = Path(device_path).name
        return Path(f"/sys/class/video4linux/{name}").exists() and Path(device_path).exists()
    return Path(device_path).exists()


def discover_audio_input_device(target_name: str) -> Optional[int]:
    """Find sounddevice input index matching substring."""
    try:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0 and target_name.lower() in dev.get("name", "").lower():
                return idx
    except Exception as exc:
        logger.debug("Audio discovery fallback: %s", exc)
    return None


def _open_v4l2(device: str, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    arg: Any = device
    if isinstance(device, str) and device.startswith("/dev/video"):
        suffix = device.removeprefix("/dev/video")
        if suffix.isdigit():
            arg = int(suffix)
    cap = cv2.VideoCapture(arg, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Prevent indefinite kernel block on USB disconnect if OpenCV build supports read timeout
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
    return cap


def generate_demo_scene_frame(width: int = 960, height: int = 540, frame_id: int = 0) -> np.ndarray:
    """Render a clean synthetic desktop scene for hardware-free Demo mode."""
    img = np.full((height, width, 3), (26, 29, 36), dtype=np.uint8)
    desk_y = int(height * 0.34)
    cv2.rectangle(img, (0, desk_y), (width, height), (42, 46, 56), -1)
    cv2.line(img, (0, desk_y), (width, desk_y), (70, 78, 92), 2)

    t = time.monotonic()
    dx = int(12 * math.sin(t * 0.8))

    cv2.rectangle(img, (int(width * 0.14), int(height * 0.42)), (int(width * 0.27), int(height * 0.68)), (185, 110, 40), -1)
    cv2.putText(img, "BLUE CUP", (int(width * 0.15), int(height * 0.56)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
    cv2.rectangle(img, (int(width * 0.34), int(height * 0.28)), (int(width * 0.65), int(height * 0.66)), (95, 102, 114), -1)
    cv2.rectangle(img, (int(width * 0.36), int(height * 0.31)), (int(width * 0.63), int(height * 0.56)), (30, 38, 48), -1)
    cv2.putText(img, "LAPTOP SCREEN", (int(width * 0.41), int(height * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 2)
    cv2.rectangle(img, (int(width * 0.70) + dx, int(height * 0.44)), (int(width * 0.83) + dx, int(height * 0.70)), (55, 65, 195), -1)
    cv2.putText(img, "RED CUP", (int(width * 0.71) + dx, int(height * 0.58)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
    cv2.rectangle(img, (int(width * 0.18), int(height * 0.71)), (int(width * 0.44), int(height * 0.90)), (50, 175, 215), -1)
    cv2.putText(img, "DEEP LEARNING BOOK", (int(width * 0.20), int(height * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 2)
    cv2.rectangle(img, (int(width * 0.68), int(height * 0.72)), (int(width * 0.79), int(height * 0.89)), (22, 24, 28), -1)
    cv2.rectangle(img, (int(width * 0.69), int(height * 0.74)), (int(width * 0.78), int(height * 0.87)), (48, 56, 68), 2)
    cv2.putText(img, "PHONE", (int(width * 0.70), int(height * 0.81)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 200, 210), 1)
    cv2.putText(img, f"DEMO DESKTOP SCENE | frame #{frame_id}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 165, 185), 1, cv2.LINE_AA)
    return img


def generate_demo_eye_frame(label: str, gaze_xy: Tuple[float, float]) -> np.ndarray:
    img = np.full((180, 240, 3), (20, 22, 28), dtype=np.uint8)
    cx = int(120 + (gaze_xy[0] - 0.5) * 70)
    cy = int(90 + (gaze_xy[1] - 0.5) * 45)
    cv2.ellipse(img, (120, 90), (88, 46), 0, 0, 360, (190, 195, 205), 2)
    cv2.circle(img, (cx, cy), 22, (75, 105, 140), -1)
    cv2.circle(img, (cx, cy), 10, (15, 15, 18), -1)
    cv2.circle(img, (cx - 5, cy - 5), 3, (255, 255, 255), -1)
    cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 210), 1, cv2.LINE_AA)
    return img


class MultiCameraRig:
    """Thread-safe capture manager with automatic USB hot-plug recovery for Left, Right, and Scene cameras."""

    def __init__(self, cfg: Dict[str, Any], demo_mode: bool = False) -> None:
        self.cfg = cfg
        self.demo_mode = demo_mode
        self._scene_lock = threading.Lock()
        self._eye_lock = threading.Lock()
        self._stop = threading.Event()
        self._reconnect_scene = threading.Event()
        self._reconnect_eyes = threading.Event()
        self._threads: List[threading.Thread] = []

        self._scene_frame: Optional[np.ndarray] = None
        self._scene_mono_ns: int = 0
        self._scene_frame_id: int = 0
        self.scene_reconnecting: bool = False

        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None
        self._eye_mono_ns: int = 0
        self._eye_frame_id: int = 0
        self.eyes_reconnecting: bool = False

        self._demo_gaze_xy: Tuple[float, float] = (0.5, 0.5)
        self.active_paths: Dict[str, Optional[str]] = {"left": None, "right": None, "scene": None}
        self.reconnect_count: int = 0

    def set_demo_gaze(self, xy: Tuple[float, float]) -> None:
        self._demo_gaze_xy = xy

    def request_reconnect(self) -> None:
        """Manually or programmatically force all USB cameras to release and re-discover /dev/videoX."""
        logger.info("Camera reconnect requested")
        self._reconnect_scene.set()
        self._reconnect_eyes.set()

    def start(self) -> None:
        self._stop.clear()
        if self.demo_mode:
            t = threading.Thread(target=self._run_demo, name="DemoCamRig", daemon=True)
            t.start()
            self._threads = [t]
            return

        t_scene = threading.Thread(target=self._run_scene_cam, name="SceneCamThread", daemon=True)
        t_eyes = threading.Thread(target=self._run_eye_cams, name="EyeCamsThread", daemon=True)
        t_scene.start()
        t_eyes.start()
        self._threads = [t_scene, t_eyes]

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.5)

    def get_latest_scene(self) -> Tuple[Optional[np.ndarray], int, int]:
        with self._scene_lock:
            if self._scene_frame is None:
                return None, 0, 0
            return self._scene_frame.copy(), self._scene_mono_ns, self._scene_frame_id

    def get_latest_eyes(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
        with self._eye_lock:
            if self._left_frame is None or self._right_frame is None:
                return None, None, 0, 0
            return self._left_frame.copy(), self._right_frame.copy(), self._eye_mono_ns, self._eye_frame_id

    def _run_demo(self) -> None:
        frame_id = 0
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()
            s_frame = generate_demo_scene_frame(960, 540, frame_id)
            l_frame = generate_demo_eye_frame("EYE_L (DEMO)", self._demo_gaze_xy)
            r_frame = generate_demo_eye_frame("EYE_R (DEMO)", self._demo_gaze_xy)
            with self._scene_lock:
                self._scene_frame = s_frame
                self._scene_mono_ns = now_ns
                self._scene_frame_id = frame_id
            with self._eye_lock:
                self._left_frame = l_frame
                self._right_frame = r_frame
                self._eye_mono_ns = now_ns
                self._eye_frame_id = frame_id
            frame_id += 1
            time.sleep(1.0 / 25.0)

    def _run_scene_cam(self) -> None:
        cam_name = self.cfg.get("camera_scene_name", "USB2.0 camera")
        fallback = self.cfg.get("fallback_scene")
        w = int(self.cfg.get("scene_width", 1280))
        h = int(self.cfg.get("scene_height", 720))
        fps = int(self.cfg.get("scene_fps", 30))

        cap_s: Optional[cv2.VideoCapture] = None
        cur_path: Optional[str] = None
        frame_id = 0
        consec_fails = 0
        last_node_check = 0.0

        try:
            while not self._stop.is_set():
                # 1. Handle manual reconnect or dead handle -> release & re-discover
                if self._reconnect_scene.is_set():
                    self._reconnect_scene.clear()
                    if cap_s is not None:
                        cap_s.release()
                        cap_s = None
                    cur_path = None
                    consec_fails = 0
                    time.sleep(0.35)

                if cap_s is None:
                    self.scene_reconnecting = True
                    p_s = discover_v4l2_camera(cam_name, fallback)
                    if p_s:
                        cap_s = _open_v4l2(p_s, w, h, fps)
                        if cap_s is not None:
                            cur_path = p_s
                            self.active_paths["scene"] = p_s
                            self.scene_reconnecting = False
                            consec_fails = 0
                            logger.info("Scene camera connected on %s (%dx%d@%d)", p_s, w, h, fps)
                        else:
                            time.sleep(0.5)
                            continue
                    else:
                        time.sleep(0.5)
                        continue

                # 2. Periodic sysfs check: detect USB hot-unplug immediately before V4L2 stalls
                now = time.monotonic()
                if now - last_node_check >= 0.5:
                    last_node_check = now
                    if not _dev_node_exists(cur_path):
                        logger.warning("Scene camera node %s vanished (USB disconnect); releasing to auto-recover...", cur_path)
                        cap_s.release()
                        cap_s = None
                        cur_path = None
                        self.scene_reconnecting = True
                        self.reconnect_count += 1
                        time.sleep(0.6)
                        continue

                ok_s, fs = cap_s.read()
                if ok_s and fs is not None and fs.size > 0:
                    consec_fails = 0
                    now_ns = time.monotonic_ns()
                    with self._scene_lock:
                        self._scene_frame = fs
                        self._scene_mono_ns = now_ns
                        self._scene_frame_id = frame_id
                    frame_id += 1
                else:
                    consec_fails += 1
                    if consec_fails >= 5:
                        logger.warning("Scene camera %s failed %d consecutive reads; reconnecting...", cur_path, consec_fails)
                        cap_s.release()
                        cap_s = None
                        cur_path = None
                        self.scene_reconnecting = True
                        self.reconnect_count += 1
                        consec_fails = 0
                        time.sleep(0.6)
                    else:
                        time.sleep(0.02)
        finally:
            if cap_s is not None:
                cap_s.release()

    def _run_eye_cams(self) -> None:
        name_l = self.cfg.get("camera_left_name", "Camera_L")
        name_r = self.cfg.get("camera_right_name", "Camera_R")
        fb_l = self.cfg.get("fallback_left")
        fb_r = self.cfg.get("fallback_right")
        w = int(self.cfg.get("eye_width", 320))
        h = int(self.cfg.get("eye_height", 240))
        fps = int(self.cfg.get("eye_fps", 30))

        cap_l: Optional[cv2.VideoCapture] = None
        cap_r: Optional[cv2.VideoCapture] = None
        path_l: Optional[str] = None
        path_r: Optional[str] = None
        frame_id = 0
        fails_l = 0
        fails_r = 0
        last_node_check = 0.0

        def _release_both() -> None:
            nonlocal cap_l, cap_r, path_l, path_r, fails_l, fails_r
            if cap_l is not None:
                cap_l.release()
                cap_l = None
            if cap_r is not None:
                cap_r.release()
                cap_r = None
            path_l = path_r = None
            fails_l = fails_r = 0

        try:
            while not self._stop.is_set():
                if self._reconnect_eyes.is_set():
                    self._reconnect_eyes.clear()
                    _release_both()
                    self.eyes_reconnecting = True
                    time.sleep(0.4)

                # Open any missing eye camera dynamically
                if cap_l is None:
                    pl = discover_v4l2_camera(name_l, fb_l)
                    if pl and pl != path_r:
                        cap_l = _open_v4l2(pl, w, h, fps)
                        if cap_l is not None:
                            path_l = pl
                            self.active_paths["left"] = pl
                            fails_l = 0
                            logger.info("Left eye camera connected on %s", pl)

                if cap_r is None:
                    pr = discover_v4l2_camera(name_r, fb_r)
                    if pr and pr != path_l:
                        cap_r = _open_v4l2(pr, w, h, fps)
                        if cap_r is not None:
                            path_r = pr
                            self.active_paths["right"] = pr
                            fails_r = 0
                            logger.info("Right eye camera connected on %s", pr)

                if cap_l is None or cap_r is None:
                    self.eyes_reconnecting = True
                    time.sleep(0.5)
                    continue

                self.eyes_reconnecting = False

                # Periodic sysfs node existence check
                now = time.monotonic()
                if now - last_node_check >= 0.5:
                    last_node_check = now
                    if not _dev_node_exists(path_l) or not _dev_node_exists(path_r):
                        logger.warning("Eye camera node vanished (L=%s, R=%s); releasing pair to auto-recover...", path_l, path_r)
                        _release_both()
                        self.eyes_reconnecting = True
                        self.reconnect_count += 1
                        time.sleep(0.6)
                        continue

                ok_l, fl = cap_l.read()
                ok_r, fr = cap_r.read()

                if ok_l and ok_r and fl is not None and fr is not None and fl.size > 0 and fr.size > 0:
                    fails_l = fails_r = 0
                    now_ns = time.monotonic_ns()
                    with self._eye_lock:
                        self._left_frame = fl
                        self._right_frame = fr
                        self._eye_mono_ns = now_ns
                        self._eye_frame_id = frame_id
                    frame_id += 1
                else:
                    if not ok_l:
                        fails_l += 1
                    if not ok_r:
                        fails_r += 1
                    if fails_l >= 5 or fails_r >= 5:
                        logger.warning("Eye camera read failures (L=%d, R=%d); reconnecting pair...", fails_l, fails_r)
                        _release_both()
                        self.eyes_reconnecting = True
                        self.reconnect_count += 1
                        time.sleep(0.6)
                    else:
                        time.sleep(0.02)
        finally:
            _release_both()
