"""Main EgoGaze-Thor Multimodal Pipeline Orchestrator."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np

from egogaze.core.ring_buffer import TemporalRingBuffer
from egogaze.core.schemas import (
    DialogueEvent,
    GazeSample,
    GazeState,
    GroundingResult,
    PipelineLatencyTrace,
    RouteType,
    SceneSnapshot,
    SpeechUtterance,
)
from egogaze.core.telemetry import TelemetryCollector
from egogaze.fusion.grounding import ProbabilisticGroundingEngine
from egogaze.perception.fom_prompt import build_foveated_som_prompt
from egogaze.perception.gaze_engine import GazeEngine
from egogaze.perception.vision_fast import FastVisionEngine
from egogaze.reasoning.llm_client import UnifiedLLMClient
from egogaze.reasoning.router import FastSlowRouter
from egogaze.reasoning.vlm_client import EdgeVLMClient
from egogaze.sensors.live_devices import MultiCameraRig
from egogaze.sensors.recorder import SessionRecorder
from egogaze.speech.asr_stream import StreamingASRWorker
from egogaze.speech.tts_stream import StreamingSentenceChunker, StreamingTTSWorker

logger = logging.getLogger("egogaze.pipeline")


class EgoGazePipeline:
    """Coordinates live/demo sensors, GazeEngine, FastVision, Probabilistic Grounding, VLM/LLM & TTS."""

    def __init__(self, cfg: Dict[str, Any], mode: Optional[str] = None) -> None:
        self.cfg = cfg
        self.mode = mode or cfg.get("system", {}).get("mode", "live")
        is_demo = (self.mode == "demo")

        self.telemetry = TelemetryCollector()
        self.gaze_buffer: TemporalRingBuffer[GazeSample] = TemporalRingBuffer(retention_ns=4_000_000_000)
        self.scene_buffer: TemporalRingBuffer[SceneSnapshot] = TemporalRingBuffer(retention_ns=4_000_000_000)

        self.cameras = MultiCameraRig(cfg.get("sensors", {}), demo_mode=is_demo)
        self.gaze_engine = GazeEngine(cfg.get("gaze", {}), demo_mode=is_demo)
        self.vision_engine = FastVisionEngine(cfg.get("vision", {}), demo_mode=is_demo)
        self.grounding_engine = ProbabilisticGroundingEngine(cfg.get("fusion", {}))
        self.router = FastSlowRouter()
        self.llm_client = UnifiedLLMClient(cfg.get("reasoning", {}))
        self.vlm_client = EdgeVLMClient(cfg.get("reasoning", {}).get("vlm", {}))
        self.recorder = SessionRecorder(cfg.get("system", {}).get("recordings_dir", "data/recordings"))
        self.asr_worker = StreamingASRWorker(cfg.get("speech", {}), cfg.get("sensors", {}), demo_mode=is_demo)
        self.tts_worker = StreamingTTSWorker(cfg.get("speech", {}))

        # Rendering & Modality Signal Controls
        self.show_heatmap = True
        self.show_som = True
        self.box_mode = "all"  # all | gaze_only | off
        self.enable_gaze = True
        self.enable_vision = True
        self.enable_voice = True
        self.fusion_mode = "probabilistic"  # probabilistic | heuristic_baseline | speech_only | gaze_only
        self._gaze_trail: Deque[Tuple[float, float, GazeState]] = deque(maxlen=40)
        self._track_trail: Deque[Tuple[int, int]] = deque(maxlen=36)
        self._last_locked_obj: Optional[Any] = None
        self._locked_miss_frames: int = 0
        self._memory_scene: Optional[SceneSnapshot] = None
        self._diff_added_classes: set[str] = set()
        self._latest_rendered_scene: Optional[np.ndarray] = None
        self._latest_scene_jpeg: Optional[bytes] = None
        self._scene_seq: int = 0
        self._latest_raw_scene: Optional[np.ndarray] = None
        self._latest_eyes_strip: Optional[np.ndarray] = None
        self._latest_eyes_jpeg: Optional[bytes] = None
        self._eyes_seq: int = 0
        self._latest_gaze: Optional[GazeSample] = None
        self._latest_scene: Optional[SceneSnapshot] = None
        self._latest_grounding: Optional[GroundingResult] = None
        self._partial_asr: str = ""
        self._dialogue_history: Deque[DialogueEvent] = deque(maxlen=30)
        self._utterance_start_ns: Dict[str, int] = {}
        self._utterance_ttfa_ms: Dict[str, float] = {}

        # SSE subscribers
        self._sub_lock = threading.Lock()
        self._event_listeners: List[Callable[[Dict[str, Any]], None]] = []

        self._render_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    @property
    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def add_event_listener(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._sub_lock:
            self._event_listeners.append(cb)

    def remove_event_listener(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._sub_lock:
            if cb in self._event_listeners:
                self._event_listeners.remove(cb)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {"type": event_type, "ts": time.time(), "data": data}
        with self._sub_lock:
            listeners = list(self._event_listeners)
        for cb in listeners:
            try:
                cb(payload)
            except Exception:
                pass

    def start(self) -> None:
        self._stop.clear()
        self.cameras.start()
        self.tts_worker.start(
            on_first_audio=self._on_tts_first_audio,
            on_wav_ready=self._on_tts_wav_ready,
        )
        self.asr_worker.start(
            on_partial=self._on_asr_partial,
            on_final=self._on_utterance,
            on_barge_in=self._on_barge_in,
        )

        t_gaze = threading.Thread(target=self._gaze_loop, name="GazeLoop", daemon=True)
        t_vis = threading.Thread(target=self._vision_loop, name="VisionLoop", daemon=True)
        self._threads = [t_gaze, t_vis]
        for t in self._threads:
            t.start()
        logger.info(
            "EgoGazePipeline started (mode=%s, gaze=%s, vision=%s, asr=%s)",
            self.mode,
            self.gaze_engine.backend_name,
            self.vision_engine.backend_name,
            self.asr_worker.backend_name,
        )

    def stop(self) -> None:
        self._stop.set()
        self.recorder.stop()
        self.asr_worker.stop()
        self.tts_worker.stop()
        self.cameras.stop()
        for t in self._threads:
            t.join(timeout=2.0)

    def _on_barge_in(self) -> None:
        if self.cfg.get("speech", {}).get("barge_in_enabled", True):
            self.tts_worker.interrupt()
            self._emit_event("barge_in", {"interrupted": True})

    def _on_tts_first_audio(self, utterance_id: str, mono_ns: int) -> None:
        t0 = self._utterance_start_ns.get(utterance_id)
        if t0 is not None:
            self._utterance_ttfa_ms[utterance_id] = round((mono_ns - t0) / 1e6, 1)

    def _on_tts_wav_ready(self, utterance_id: str, text: str, wav_b64: str) -> None:
        self._emit_event("tts_audio", {"utterance_id": utterance_id, "text": text, "wav_b64": wav_b64})

    def feed_browser_audio(self, pcm16_bytes: bytes, sample_rate: int = 16000) -> None:
        self.asr_worker.feed_browser_pcm16(pcm16_bytes, sample_rate)

    def set_browser_mic_active(self, active: bool) -> None:
        self.asr_worker.browser_mic_active = active

    def _on_asr_partial(self, text: str) -> None:
        if not self.enable_voice:
            return
        self._partial_asr = text
        self._emit_event("asr_partial", {"text": text})

    def submit_text_query(self, text: str, source: str = "text") -> GroundingResult:
        """Entry point from Web UI or CLI to inject a query through the exact same fusion/reasoning path."""
        now_ns = time.monotonic_ns()
        utt = SpeechUtterance(
            utterance_id=uuid.uuid4().hex[:8],
            text=text.strip(),
            is_final=True,
            speech_start_mono_ns=now_ns - 600_000_000,
            speech_end_mono_ns=now_ns,
            final_mono_ns=now_ns,
            source=source,
        )
        return self._handle_utterance_sync(utt)

    def _on_utterance(self, utt: SpeechUtterance) -> None:
        if not self.enable_voice:
            return
        threading.Thread(target=self._handle_utterance_sync, args=(utt,), daemon=True).start()

    def _handle_utterance_sync(self, utt: SpeechUtterance) -> GroundingResult:
        self._partial_asr = ""
        self._utterance_start_ns[utt.utterance_id] = time.monotonic_ns()

        user_ev = DialogueEvent(
            utterance_id=utt.utterance_id,
            role="user",
            text=utt.text,
            timestamp=time.time(),
            meta={"source": utt.source},
        )
        self._dialogue_history.append(user_ev)
        self._emit_event("dialogue", user_ev.model_dump())

        # 1. Temporal window alignment for gaze & scene (respects live modality toggles!)
        if self.enable_gaze:
            w_start = utt.speech_start_mono_ns - self.grounding_engine.pre_window_ns
            w_end = utt.speech_end_mono_ns + self.grounding_engine.post_window_ns
            gaze_window = self.gaze_buffer.query_window(w_start, w_end)
            if not gaze_window and self._latest_gaze is not None:
                gaze_window = [self._latest_gaze]
        else:
            gaze_window = []

        if self.enable_vision:
            scene, _ = self.scene_buffer.nearest(utt.speech_end_mono_ns)
            if scene is None:
                scene = self._latest_scene
        else:
            scene = None

        # 2. Probabilistic Referential Grounding (respects live fusion_mode selector for real-time ablation demos!)
        grounding = self.grounding_engine.ground(utt, scene, gaze_window, mode=self.fusion_mode)
        self._latest_grounding = grounding
        self.recorder.write_event("utterance_grounding", grounding.model_dump(), utt.final_mono_ns)
        self._emit_event("grounding", grounding.model_dump())

        # 3. Fast-Slow Routing & Execution
        route = self.router.route(grounding)
        if route == RouteType.SYSTEM1_FAST:
            self._execute_system1(utt, grounding, scene)
        elif route == RouteType.SYSTEM2_VLM_FSOM:
            self._execute_system2_vlm(utt, grounding, scene)
        else:
            self._execute_dialogue_llm(utt, grounding, scene)

        return grounding

    def lock_object_at_point(self, x_norm: float, y_norm: float) -> Optional[Dict[str, Any]]:
        """Click-to-lock / unlock in live mode: locks object containing normalized (x, y), or unlocks."""
        scene = self._latest_scene
        if not scene or not scene.objects:
            self.vision_engine.set_locked_track(None)
            self._track_trail.clear()
            self._last_locked_obj = None
            return None

        w, h = float(max(scene.width, 1)), float(max(scene.height, 1))
        px, py = x_norm * w, y_norm * h
        hits = [
            o for o in scene.objects
            if o.bbox_xyxy[0] <= px <= o.bbox_xyxy[2] and o.bbox_xyxy[1] <= py <= o.bbox_xyxy[3]
        ]
        if not hits:
            self.vision_engine.set_locked_track(None)
            self._track_trail.clear()
            self._last_locked_obj = None
            return None
        # Choose smallest containing box (most specific object)
        chosen = min(hits, key=lambda o: o.area_norm)
        self.vision_engine.set_locked_track(chosen.track_id)
        self._track_trail.clear()
        self._last_locked_obj = chosen
        self._locked_miss_frames = 0
        return chosen.model_dump()

    def memorize_current_scene(self) -> str:
        scene = self._latest_scene
        if not scene or not scene.objects:
            return "当前画面没有检测到物体，无法建立桌面记忆快照。"
        self._memory_scene = scene.model_copy(deep=True)
        self._diff_added_classes.clear()
        counts: Dict[str, int] = {}
        for o in scene.objects:
            counts[o.class_name] = counts.get(o.class_name, 0) + 1
        summary = "、".join(f"{v}个{k}" for k, v in counts.items())
        return f"已记住当前桌面场景（共 {len(scene.objects)} 个物体：{summary}）。现在可以移动或增减物品，再问我“桌面有什么变化”。"

    def diff_current_scene(self) -> str:
        if self._memory_scene is None:
            return "还没有记住桌面基准，请先说“记住桌面”或点击“记住桌面”按钮。"
        cur = self._latest_scene
        if cur is None:
            return "当前视觉画面未就绪。"

        def _counts(snap: SceneSnapshot) -> Dict[str, int]:
            c: Dict[str, int] = {}
            for o in snap.objects:
                c[o.class_name] = c.get(o.class_name, 0) + 1
            return c

        old_c = _counts(self._memory_scene)
        new_c = _counts(cur)
        added = []
        removed = []
        self._diff_added_classes.clear()
        for k in sorted(set(old_c) | set(new_c)):
            diff = new_c.get(k, 0) - old_c.get(k, 0)
            if diff > 0:
                added.append(f"{diff}个{k}")
                self._diff_added_classes.add(k)
            elif diff < 0:
                removed.append(f"{-diff}个{k}")

        # Check significant movement of common single-instance classes
        moved = []
        for k in set(old_c) & set(new_c):
            if old_c[k] == 1 and new_c[k] == 1:
                o1 = next(o for o in self._memory_scene.objects if o.class_name == k)
                o2 = next(o for o in cur.objects if o.class_name == k)
                dist = ((o1.center_norm[0] - o2.center_norm[0]) ** 2 + (o1.center_norm[1] - o2.center_norm[1]) ** 2) ** 0.5
                if dist > 0.12:
                    moved.append(k)

        parts = []
        if added:
            parts.append("新增了 " + "、".join(added))
        if removed:
            parts.append("拿走了 " + "、".join(removed))
        if moved:
            parts.append("移动了 " + "、".join(moved))
        if not parts:
            return "与记忆快照相比，桌面物体基本没有明显变化。"
        return "检测到桌面变化：" + "；".join(parts) + "。"

    def _execute_system1(
        self,
        utt: SpeechUtterance,
        grounding: GroundingResult,
        scene: Optional[SceneSnapshot],
    ) -> None:
        t0 = time.perf_counter_ns()
        hint = grounding.cues.intent_hint
        if hint == "memorize_scene":
            reply = self.memorize_current_scene()
        elif hint == "diff_scene":
            reply = self.diff_current_scene()
        elif hint == "stop_track":
            self.vision_engine.set_locked_track(None)
            self._track_trail.clear()
            self._last_locked_obj = None
            reply = "已解除当前目标锁定。"
        elif hint == "count":
            n = len(scene.objects) if scene else 0
            if grounding.cues.target_classes and scene:
                matched = [o for o in scene.objects if o.class_name.lower() in grounding.cues.target_classes]
                reply = f"当前画面中检测到 {len(matched)} 个 {', '.join(grounding.cues.target_classes)}，总计 {n} 个物体。"
            else:
                reply = f"当前主视角共检测到 {n} 个桌面物体。"
        else:
            if grounding.selected_track_id is not None:
                self.vision_engine.set_locked_track(grounding.selected_track_id)
                self._track_trail.clear()
                self._locked_miss_frames = 0
                reply = (
                    f"已锁定目标 [{grounding.selected_som_id}] {grounding.selected_class}"
                    f"（融合置信度 {grounding.selected_prob * 100:.0f}%，眼动权重 {grounding.gaze_reliability:.2f}）。"
                )
            else:
                reply = "当前视野中未找到匹配的目标物体，请注视目标或提供更明确的特征。"

        self.tts_worker.enqueue_sentence(utt.utterance_id, reply, is_first=True)
        total_ms = round((time.perf_counter_ns() - t0) / 1e6, 2)

        ev = DialogueEvent(
            utterance_id=utt.utterance_id,
            role="assistant",
            text=reply,
            route=RouteType.SYSTEM1_FAST,
            backend="probabilistic_grounding",
            timestamp=time.time(),
        )
        self._dialogue_history.append(ev)
        self._emit_event("dialogue", ev.model_dump())

        trace = PipelineLatencyTrace(
            utterance_id=utt.utterance_id,
            utterance_text=utt.text,
            route=RouteType.SYSTEM1_FAST,
            backend="probabilistic_grounding",
            vad_ms=utt.vad_ms,
            asr_ms=utt.asr_ms,
            fusion_ms=grounding.fusion_ms,
            ttft_ms=total_ms,
            ttfa_ms=self._utterance_ttfa_ms.get(utt.utterance_id, 0.0),
            total_ms=round(grounding.fusion_ms + total_ms, 2),
            timestamp=time.time(),
        )
        self.telemetry.record_trace(trace)
        self._emit_event("latency", trace.model_dump())

    def _execute_system2_vlm(
        self,
        utt: SpeechUtterance,
        grounding: GroundingResult,
        scene: Optional[SceneSnapshot],
    ) -> None:
        if grounding.selected_track_id is not None:
            self.vision_engine.set_locked_track(grounding.selected_track_id)

        with self._render_lock:
            raw_frame = self._latest_raw_scene.copy() if self._latest_raw_scene is not None else None

        if raw_frame is None:
            self._execute_dialogue_llm(utt, grounding, scene)
            return

        vlm_cfg = self.cfg.get("reasoning", {}).get("vlm", {})
        pack = build_foveated_som_prompt(
            raw_frame,
            scene,
            grounding,
            full_frame_max_width=int(vlm_cfg.get("full_frame_max_width", 768)),
            roi_size=int(vlm_cfg.get("foveated_roi_size", 448)),
        )

        # Immediately push the F-SoM visual prompt thumbnail to Web UI while VLM streams
        self._emit_event(
            "fsom_prompt",
            {
                "utterance_id": utt.utterance_id,
                "fsom_b64": pack.fsom_b64,
                "roi_b64": pack.roi_b64,
                "scene_legend": pack.scene_legend,
            },
        )

        token_iter, metrics = self.vlm_client.stream_reason(utt.text, pack)
        chunker = StreamingSentenceChunker()
        full_reply = ""
        first_sentence = True
        for delta in token_iter:
            full_reply += delta
            self._emit_event(
                "dialogue",
                DialogueEvent(
                    utterance_id=utt.utterance_id,
                    role="assistant",
                    text=full_reply,
                    route=RouteType.SYSTEM2_VLM_FSOM,
                    backend=metrics.get("backend"),
                    is_partial=True,
                    timestamp=time.time(),
                ).model_dump(),
            )
            for sent in chunker.push(delta):
                self.tts_worker.enqueue_sentence(utt.utterance_id, sent, is_first=first_sentence)
                first_sentence = False

        tail = chunker.flush()
        if tail:
            self.tts_worker.enqueue_sentence(utt.utterance_id, tail, is_first=first_sentence)

        final_ev = DialogueEvent(
            utterance_id=utt.utterance_id,
            role="assistant",
            text=full_reply,
            route=RouteType.SYSTEM2_VLM_FSOM,
            backend=metrics.get("backend"),
            is_partial=False,
            fsom_image_b64=pack.fsom_b64,
            roi_image_b64=pack.roi_b64,
            timestamp=time.time(),
        )
        self._dialogue_history.append(final_ev)
        self._emit_event("dialogue", final_ev.model_dump())

        trace = PipelineLatencyTrace(
            utterance_id=utt.utterance_id,
            utterance_text=utt.text,
            route=RouteType.SYSTEM2_VLM_FSOM,
            backend=str(metrics.get("backend", "qwen3_vl_4b")),
            vad_ms=utt.vad_ms,
            asr_ms=utt.asr_ms,
            fusion_ms=grounding.fusion_ms,
            ttft_ms=float(metrics.get("ttft_ms", 0.0)),
            ttfa_ms=self._utterance_ttfa_ms.get(utt.utterance_id, 0.0),
            total_ms=round(grounding.fusion_ms + float(metrics.get("total_ms", 0.0)), 2),
            timestamp=time.time(),
        )
        self.telemetry.record_trace(trace)
        self._emit_event("latency", trace.model_dump())

    def _execute_dialogue_llm(
        self,
        utt: SpeechUtterance,
        grounding: GroundingResult,
        scene: Optional[SceneSnapshot],
    ) -> None:
        now_dt = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_str = f"{now_dt.year}年{now_dt.month}月{now_dt.day}日 {weekdays[now_dt.weekday()]} {now_dt.strftime('%H:%M')}"
        scene_summary = ", ".join(f"[{o.som_id}]{o.color} {o.class_name}" for o in (scene.objects if scene else []))
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是运行在 Jetson Thor 上的实时语音助手 EgoGaze-Thor。当前真实时间：{time_str}。"
                    "请用简洁自然的口语化中文直接回答用户（1到2句话即可），"
                    "禁止输出括号注释、思考过程、提示语或重复句子。"
                ),
            },
            {
                "role": "user",
                "content": f"桌面感知物体：{scene_summary or '无'}；注视目标：{grounding.selected_class or '无'}。\n用户说：{utt.text}",
            },
        ]
        token_iter, metrics = self.llm_client.stream_chat(messages)
        chunker = StreamingSentenceChunker()
        full_reply = ""
        first_sentence = True
        for delta in token_iter:
            full_reply += delta
            self._emit_event(
                "dialogue",
                DialogueEvent(
                    utterance_id=utt.utterance_id,
                    role="assistant",
                    text=full_reply,
                    route=RouteType.DIALOGUE_LLM,
                    backend=metrics.get("backend"),
                    is_partial=True,
                    timestamp=time.time(),
                ).model_dump(),
            )
            for sent in chunker.push(delta):
                self.tts_worker.enqueue_sentence(utt.utterance_id, sent, is_first=first_sentence)
                first_sentence = False

        tail = chunker.flush()
        if tail:
            self.tts_worker.enqueue_sentence(utt.utterance_id, tail, is_first=first_sentence)

        final_ev = DialogueEvent(
            utterance_id=utt.utterance_id,
            role="assistant",
            text=full_reply,
            route=RouteType.DIALOGUE_LLM,
            backend=metrics.get("backend"),
            is_partial=False,
            timestamp=time.time(),
        )
        self._dialogue_history.append(final_ev)
        self._emit_event("dialogue", final_ev.model_dump())

        trace = PipelineLatencyTrace(
            utterance_id=utt.utterance_id,
            utterance_text=utt.text,
            route=RouteType.DIALOGUE_LLM,
            backend=str(metrics.get("backend", self.llm_client.active_name)),
            vad_ms=utt.vad_ms,
            asr_ms=utt.asr_ms,
            fusion_ms=grounding.fusion_ms,
            ttft_ms=float(metrics.get("ttft_ms", 0.0)),
            ttfa_ms=self._utterance_ttfa_ms.get(utt.utterance_id, 0.0),
            total_ms=round(grounding.fusion_ms + float(metrics.get("total_ms", 0.0)), 2),
            timestamp=time.time(),
        )
        self.telemetry.record_trace(trace)
        self._emit_event("latency", trace.model_dump())

    def _gaze_loop(self) -> None:
        target_hz = float(self.cfg.get("gaze", {}).get("target_hz", 25.0))
        period = 1.0 / max(target_hz, 1.0)
        last_eye_id = -1
        while not self._stop.is_set():
            t0 = time.monotonic()
            lf, rf, eye_ns, fid = self.cameras.get_latest_eyes()
            if lf is not None and rf is not None and (fid != last_eye_id or self.mode == "demo"):
                last_eye_id = fid
                sample = self.gaze_engine.estimate(lf, rf, eye_ns or time.monotonic_ns(), fid)
                self._latest_gaze = sample
                self.gaze_buffer.append(sample)
                self._gaze_trail.append((sample.filtered_xy[0], sample.filtered_xy[1], sample.state))
                self.telemetry.gaze_rate.tick()
                self.recorder.write_event("gaze", sample.model_dump(), sample.capture_mono_ns)

                # Side-by-side eye preview strip (pre-encode once)
                strip = np.hstack([cv2.resize(lf, (180, 135)), cv2.resize(rf, (180, 135))])
                ok_e, buf_e = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                with self._render_lock:
                    self._latest_eyes_strip = strip
                    if ok_e:
                        self._latest_eyes_jpeg = buf_e.tobytes()
                        self._eyes_seq += 1
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    def _vision_loop(self) -> None:
        target_fps = float(self.cfg.get("vision", {}).get("target_fps", 20.0))
        period = 1.0 / max(target_fps, 1.0)
        quality = int(self.cfg.get("web", {}).get("jpeg_quality", 72))
        max_w = int(self.cfg.get("web", {}).get("stream_max_width", 960))
        last_scene_id = -1
        last_frame_rx_time = time.monotonic()
        stale_banner_sent = False
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame, scene_ns, fid = self.cameras.get_latest_scene()
            if frame is not None and (fid != last_scene_id or self.mode == "demo"):
                last_scene_id = fid
                last_frame_rx_time = time.monotonic()
                stale_banner_sent = False
                self.telemetry.scene_cam_rate.tick()
                gaze = self._latest_gaze if self.enable_gaze else None
                if self.enable_vision:
                    try:
                        snap = self.vision_engine.process_frame(frame, gaze, scene_ns or time.monotonic_ns(), fid)
                    except Exception as exc:
                        logger.error("Vision inference error (passing raw frame through): %s", exc)
                        h, w = frame.shape[:2]
                        snap = SceneSnapshot(
                            frame_id=fid,
                            capture_mono_ns=scene_ns or time.monotonic_ns(),
                            timestamp=time.time(),
                            width=w,
                            height=h,
                            objects=[],
                        )
                else:
                    h, w = frame.shape[:2]
                    snap = SceneSnapshot(
                        frame_id=fid,
                        capture_mono_ns=scene_ns or time.monotonic_ns(),
                        timestamp=time.time(),
                        width=w,
                        height=h,
                        objects=[],
                    )
                self._latest_scene = snap
                self.scene_buffer.append(snap)
                self.telemetry.vision_rate.tick()

                if self.recorder.is_recording:
                    self.recorder.write_frame(frame, snap.capture_mono_ns)
                    self.recorder.write_event("scene", snap.model_dump(), snap.capture_mono_ns)

                rendered = self._render_overlay(frame, snap, gaze)
                # Resize & JPEG-encode once in worker thread so FastAPI never blocks
                rh, rw = rendered.shape[:2]
                stream_img = (
                    cv2.resize(rendered, (max_w, int(rh * max_w / rw)), interpolation=cv2.INTER_AREA)
                    if rw > max_w
                    else rendered
                )
                ok_s, buf_s = cv2.imencode(".jpg", stream_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                with self._render_lock:
                    self._latest_raw_scene = frame
                    self._latest_rendered_scene = rendered
                    if ok_s:
                        self._latest_scene_jpeg = buf_s.tobytes()
                        self._scene_seq += 1
            elif self.mode != "demo" and (time.monotonic() - last_frame_rx_time) > 2.2:
                # Auto-trigger USB recovery if scene camera stalled without raising a read error
                if not self.cameras.scene_reconnecting and (time.monotonic() - last_frame_rx_time) > 3.5:
                    self.cameras.request_reconnect()
                    last_frame_rx_time = time.monotonic() - 2.3
                if not stale_banner_sent:
                    stale_banner_sent = True
                    banner = np.full((540, 960, 3), (18, 22, 30), dtype=np.uint8)
                    cv2.putText(banner, "USB Camera Reconnecting (Hot-Plug Recovery)...", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 200, 255), 2, cv2.LINE_AA)
                    cv2.putText(banner, "Auto-rediscovering /dev/videoX nodes on Jetson Thor", (165, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 175, 205), 1, cv2.LINE_AA)
                    ok_b, buf_b = cv2.imencode(".jpg", banner, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok_b:
                        with self._render_lock:
                            self._latest_scene_jpeg = buf_b.tobytes()
                            self._scene_seq += 1
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    @staticmethod
    def _draw_hud_brackets(
        canvas: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int = 3,
    ) -> None:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        arm = max(14, min(36, int(min(bw, bh) * 0.24)))
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + arm, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + arm), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - arm, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + arm), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + arm, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - arm), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - arm, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - arm), color, thickness, cv2.LINE_AA)

    def _render_overlay(
        self,
        frame_bgr: np.ndarray,
        scene: SceneSnapshot,
        gaze: Optional[GazeSample],
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # Identify locked object (30-frame coasting + persistent appearance Re-ID search banner!)
        locked_obj = next((o for o in scene.objects if o.is_locked), None)
        coasting = False
        searching_label: Optional[str] = None
        if locked_obj is not None:
            self._last_locked_obj = locked_obj
            self._locked_miss_frames = 0
            lx1, ly1, lx2, ly2 = locked_obj.bbox_xyxy
            self._track_trail.append(((lx1 + lx2) // 2, (ly1 + ly2) // 2))
        elif self.vision_engine.locked_track_id is not None:
            self._locked_miss_frames += 1
            sig = self.vision_engine.locked_signature
            if self._last_locked_obj is not None and self._locked_miss_frames <= 30:
                locked_obj = self._last_locked_obj
                coasting = True
            elif sig is not None:
                searching_label = f"SEARCHING TARGET: {sig.class_name} ({sig.color}) ..."
        else:
            self._last_locked_obj = None
            self._track_trail.clear()

        if searching_label:
            cv2.rectangle(canvas, (14, 14), (420, 46), (25, 32, 48), -1)
            cv2.rectangle(canvas, (14, 14), (420, 46), (0, 200, 255), 1)
            cv2.putText(canvas, searching_label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 2, cv2.LINE_AA)

        # Subtle spotlight vignette when a target is actively locked (makes tracking POP visually)
        if locked_obj is not None:
            dimmed = (canvas.astype(np.float32) * 0.78).astype(np.uint8)
            lx1, ly1, lx2, ly2 = locked_obj.bbox_xyxy
            lx1, ly1 = max(0, lx1), max(0, ly1)
            lx2, ly2 = min(w, lx2), min(h, ly2)
            dimmed[ly1:ly2, lx1:lx2] = canvas[ly1:ly2, lx1:lx2]
            canvas = dimmed

        # 1. Fast 1/4-resolution decaying gaze heatmap trail (only when gaze modality is enabled!)
        if self.enable_gaze and self.show_heatmap and self._gaze_trail:
            sw, sh = max(80, w // 4), max(45, h // 4)
            heat = np.zeros((sh, sw), dtype=np.float32)
            pts = list(self._gaze_trail)
            n_pts = len(pts)
            for i, (gx, gy, st) in enumerate(pts):
                if st == GazeState.INVALID:
                    continue
                weight = ((i + 1) / float(n_pts)) ** 1.5 * (1.0 if st == GazeState.FIXATION else 0.3)
                px, py = int(gx * sw), int(gy * sh)
                cv2.circle(heat, (px, py), max(5, int(min(sw, sh) * 0.055)), weight, -1)
            if float(heat.max()) > 0.01:
                heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=5)
                heat_u8 = np.clip((heat / (float(heat.max()) + 1e-6)) * 255, 0, 255).astype(np.uint8)
                heat_full = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
                colored = cv2.applyColorMap(heat_full, cv2.COLORMAP_JET)
                mask = (heat_full > 35)[..., None]
                canvas = np.where(mask, cv2.addWeighted(canvas, 0.68, colored, 0.32, 0), canvas)

        # 2. Detection boxes governed cleanly by 3-mode selector: all | gaze_only | off
        gaze_px = (
            (gaze.filtered_xy[0] * w, gaze.filtered_xy[1] * h)
            if (gaze is not None and gaze.state != GazeState.INVALID)
            else None
        )
        if self.box_mode != "off":
            for obj in scene.objects:
                if locked_obj is not None and obj.track_id == locked_obj.track_id:
                    continue
                x1, y1, x2, y2 = obj.bbox_xyxy
                gaze_in_box = False
                if gaze_px is not None:
                    pad_x, pad_y = int(w * 0.03), int(h * 0.03)
                    gaze_in_box = (x1 - pad_x <= gaze_px[0] <= x2 + pad_x) and (y1 - pad_y <= gaze_px[1] <= y2 + pad_y)
                is_gaze_hover = gaze_in_box or obj.gaze_dwell_ms >= 140.0
                is_new_diff = obj.class_name in self._diff_added_classes

                if self.box_mode == "gaze_only" and not is_gaze_hover and not is_new_diff:
                    continue

                if is_new_diff:
                    color = (220, 80, 255)  # Magenta for newly added scene-diff objects
                elif is_gaze_hover:
                    color = (255, 220, 60)  # Cyan for gaze-fixated hover focus
                else:
                    color = (90, 215, 135)

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2 if is_gaze_hover else 1)
                prefix = f"[{obj.som_id}] " if self.show_som else ""
                suffix = " · NEW" if is_new_diff else (f" · {int(obj.gaze_dwell_ms)}ms" if is_gaze_hover else "")
                label = f"{prefix}{obj.class_name} ({obj.color}) {obj.confidence:.2f}{suffix}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                by1 = max(0, y1 - th - 8)
                cv2.rectangle(canvas, (x1, by1), (min(w - 1, x1 + tw + 8), by1 + th + 8), color, -1)
                cv2.putText(canvas, label, (x1 + 4, by1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (12, 16, 22), 1, cv2.LINE_AA)

                if obj.gaze_dwell_ms > 20.0:
                    ratio = min(1.0, obj.gaze_dwell_ms / 800.0)
                    bar_x2 = int(x1 + (x2 - x1) * ratio)
                    cv2.rectangle(canvas, (x1, max(0, y2 - 4)), (bar_x2, y2), (255, 220, 60), -1)

        # 3. Cinematic HUD Tracking Overlay + 36-frame Motion Trajectory Trail
        if locked_obj is not None:
            trail = list(self._track_trail)
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    alpha = i / float(len(trail))
                    t_color = (int(60 + 195 * (1 - alpha)), int(215 * alpha), int(255 * alpha))
                    cv2.line(canvas, trail[i - 1], trail[i], t_color, max(1, int(3 * alpha)), cv2.LINE_AA)

            lx1, ly1, lx2, ly2 = locked_obj.bbox_xyxy
            hud_color = (255, 200, 60) if coasting else (0, 215, 255)
            self._draw_hud_brackets(canvas, (lx1, ly1, lx2, ly2), hud_color, thickness=3)
            # Subtle inner hairline box + center reticle
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), hud_color, 1)
            cx, cy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
            cv2.circle(canvas, (cx, cy), 5, hud_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 14, hud_color, 1, cv2.LINE_AA)

            status_tag = "REACQUIRING" if coasting else "LOCKED"
            hud_label = f"{status_tag} [{locked_obj.som_id}] {locked_obj.class_name} | {locked_obj.color} | v={locked_obj.velocity_norm:.2f}"
            (tw, th), _ = cv2.getTextSize(hud_label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            by1 = max(0, ly1 - th - 12)
            cv2.rectangle(canvas, (lx1, by1), (min(w - 1, lx1 + tw + 12), by1 + th + 12), hud_color, -1)
            cv2.putText(canvas, hud_label, (lx1 + 6, by1 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (10, 14, 20), 2, cv2.LINE_AA)

        # 4. Real-time gaze reticle + dwell progress arc (only when gaze modality is enabled!)
        if self.enable_gaze and gaze is not None and gaze.state != GazeState.INVALID:
            gx, gy = int(gaze.filtered_xy[0] * w), int(gaze.filtered_xy[1] * h)
            ret_color = (0, 90, 255) if gaze.state == GazeState.FIXATION else (255, 200, 60)
            r = max(14, int(min(w, h) * 0.036))
            cv2.circle(canvas, (gx, gy), r, ret_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 4, ret_color, -1, cv2.LINE_AA)
            if gaze.state == GazeState.FIXATION and gaze.dwell_ms > 50:
                sweep = int(min(360.0, (gaze.dwell_ms / 1000.0) * 360.0))
                cv2.ellipse(canvas, (gx, gy), (r + 5, r + 5), -90, 0, sweep, (0, 220, 255), 2, cv2.LINE_AA)
            state_txt = f"{gaze.state.value} {int(gaze.dwell_ms)}ms"
            cv2.putText(canvas, state_txt, (min(w - 140, gx + r + 8), max(20, gy + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ret_color, 1, cv2.LINE_AA)

        return canvas

    def get_rendered_scene_jpeg(self, quality: int = 72) -> Optional[bytes]:
        with self._render_lock:
            return self._latest_scene_jpeg

    def get_scene_jpeg_with_seq(self) -> Tuple[Optional[bytes], int]:
        with self._render_lock:
            return self._latest_scene_jpeg, self._scene_seq

    def get_eyes_strip_jpeg(self, quality: int = 70) -> Optional[bytes]:
        with self._render_lock:
            return self._latest_eyes_jpeg

    def get_eyes_jpeg_with_seq(self) -> Tuple[Optional[bytes], int]:
        with self._render_lock:
            return self._latest_eyes_jpeg, self._eyes_seq

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "telemetry": self.telemetry.summary(),
            "gaze_backend": self.gaze_engine.backend_name,
            "vision_backend": self.vision_engine.backend_name,
            "llm_backend": self.llm_client.active_name,
            "llm_backends": list(self.llm_client.backends.keys()),
            "vlm_backend": self.vlm_client.backend_name,
            "tts_sid": self.tts_worker.sid,
            "calibrated": self.gaze_engine.calibrator.is_calibrated,
            "calibration_rmse": self.gaze_engine.calibrator.train_rmse,
            "calibration_samples": len(self.gaze_engine.calibrator.samples),
            "recording": self.recorder.is_recording,
            "show_heatmap": self.show_heatmap,
            "show_som": self.show_som,
            "box_mode": self.box_mode,
            "enable_gaze": self.enable_gaze,
            "enable_vision": self.enable_vision,
            "enable_voice": self.enable_voice,
            "fusion_mode": self.fusion_mode,
            "has_memory": self._memory_scene is not None,
            "gaze": self._latest_gaze.model_dump() if self._latest_gaze else None,
            "scene": self._latest_scene.model_dump() if self._latest_scene else None,
            "grounding": self._latest_grounding.model_dump() if self._latest_grounding else None,
            "partial_asr": self._partial_asr,
            "dialogue": [d.model_dump() for d in self._dialogue_history],
            "latency_traces": self.telemetry.recent_traces(),
        }
