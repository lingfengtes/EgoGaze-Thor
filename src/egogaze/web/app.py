"""FastAPI server exposing MJPEG live feeds, SSE telemetry stream, calibration, recording & query APIs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import queue
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np

from egogaze.pipeline import EgoGazePipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _placeholder_jpeg(text: str, width: int = 640, height: int = 360) -> bytes:
    img = np.full((height, width, 3), (18, 22, 30), dtype=np.uint8)
    cv2.putText(img, text, (32, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 170, 210), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else b""


class QueryRequest(BaseModel):
    text: str


class DemoGazeRequest(BaseModel):
    x: float
    y: float


class CalibSampleRequest(BaseModel):
    target_x: float
    target_y: float


class ToggleRequest(BaseModel):
    show_heatmap: Optional[bool] = None
    show_som: Optional[bool] = None
    box_mode: Optional[str] = None
    enable_gaze: Optional[bool] = None
    enable_vision: Optional[bool] = None
    enable_voice: Optional[bool] = None
    fusion_mode: Optional[str] = None
    llm_backend: Optional[str] = None
    tts_sid: Optional[int] = None


def create_app(pipeline: EgoGazePipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pipeline.start()
        try:
            yield
        finally:
            pipeline.stop()

    app = FastAPI(title="EgoGaze-Thor Telemetry Console", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            return HTMLResponse("<h1>EgoGaze-Thor UI not found</h1>", status_code=404)
        return FileResponse(html_path)

    @app.get("/api/state")
    async def get_state() -> Dict[str, Any]:
        return pipeline.snapshot_state()

    @app.post("/api/query")
    async def post_query(req: QueryRequest) -> Dict[str, Any]:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty query")
        res = await asyncio.to_thread(pipeline.submit_text_query, req.text.strip(), "text")
        return res.model_dump()

    @app.post("/api/demo/gaze")
    async def set_demo_gaze(req: DemoGazeRequest) -> Dict[str, Any]:
        xy = (max(0.0, min(1.0, req.x)), max(0.0, min(1.0, req.y)))
        pipeline.gaze_engine.set_demo_target(xy)
        pipeline.cameras.set_demo_gaze(xy)
        return {"ok": True, "gaze": xy}

    @app.post("/api/controls")
    async def update_controls(req: ToggleRequest) -> Dict[str, Any]:
        if req.show_heatmap is not None:
            pipeline.show_heatmap = req.show_heatmap
        if req.show_som is not None:
            pipeline.show_som = req.show_som
        if req.box_mode in ("all", "gaze_only", "off"):
            pipeline.box_mode = req.box_mode
        if req.enable_gaze is not None:
            pipeline.enable_gaze = req.enable_gaze
        if req.enable_vision is not None:
            pipeline.enable_vision = req.enable_vision
        if req.enable_voice is not None:
            pipeline.enable_voice = req.enable_voice
        if req.fusion_mode in ("probabilistic", "heuristic_baseline", "speech_only", "gaze_only"):
            pipeline.fusion_mode = req.fusion_mode
        if req.llm_backend is not None:
            pipeline.llm_client.switch_backend(req.llm_backend)
        if req.tts_sid is not None and 0 <= req.tts_sid <= 52:
            pipeline.tts_worker.sid = int(req.tts_sid)
        return {
            "show_heatmap": pipeline.show_heatmap,
            "show_som": pipeline.show_som,
            "box_mode": pipeline.box_mode,
            "enable_gaze": pipeline.enable_gaze,
            "enable_vision": pipeline.enable_vision,
            "enable_voice": pipeline.enable_voice,
            "fusion_mode": pipeline.fusion_mode,
            "llm_backend": pipeline.llm_client.active_name,
            "tts_sid": pipeline.tts_worker.sid,
        }

    @app.post("/api/track/click")
    async def click_track(req: DemoGazeRequest) -> Dict[str, Any]:
        locked = pipeline.lock_object_at_point(req.x, req.y)
        return {"locked": locked}

    @app.post("/api/cameras/reconnect")
    async def reconnect_cameras() -> Dict[str, Any]:
        pipeline.cameras.request_reconnect()
        return {"ok": True, "paths": pipeline.cameras.active_paths}

    @app.post("/api/record/start")
    async def start_record() -> Dict[str, Any]:
        path = pipeline.recorder.start("web")
        return {"recording": True, "path": str(path)}

    @app.post("/api/record/stop")
    async def stop_record() -> Dict[str, Any]:
        path = pipeline.recorder.stop()
        return {"recording": False, "path": str(path) if path else None}

    @app.post("/api/calibration/clear")
    async def calib_clear() -> Dict[str, Any]:
        pipeline.gaze_engine.calibrator.clear_samples()
        return {"samples": 0}

    @app.post("/api/calibration/sample")
    async def calib_sample(req: CalibSampleRequest) -> Dict[str, Any]:
        raw_xy = pipeline.gaze_engine.last_raw_xy
        pipeline.gaze_engine.calibrator.add_sample(raw_xy, (req.target_x, req.target_y))
        return {
            "samples": len(pipeline.gaze_engine.calibrator.samples),
            "raw_xy": raw_xy,
            "target_xy": (req.target_x, req.target_y),
        }

    @app.post("/api/calibration/fit")
    async def calib_fit() -> Dict[str, Any]:
        try:
            rmse = pipeline.gaze_engine.calibrator.fit()
            return {"ok": True, "rmse": round(rmse, 4), "calibrated": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/calibration/reset")
    async def calib_reset() -> Dict[str, Any]:
        pipeline.gaze_engine.calibrator.reset()
        return {"ok": True, "calibrated": False}

    @app.post("/api/audio/pcm")
    async def ingest_browser_pcm(request: Request, sr: int = 16000) -> Dict[str, Any]:
        """Receive 16-bit signed mono PCM audio chunks from Mac browser microphone."""
        body = await request.body()
        if body:
            pipeline.feed_browser_audio(body, sample_rate=sr)
        return {"ok": True, "bytes": len(body)}

    @app.post("/api/audio/mic_mode")
    async def set_mic_mode(req: Dict[str, Any]) -> Dict[str, Any]:
        active = bool(req.get("browser_mic", False))
        pipeline.set_browser_mic_active(active)
        return {"browser_mic": active}

    @app.get("/stream/scene")
    async def stream_scene(request: Request):
        wait_jpg = _placeholder_jpeg("EgoGaze-Thor: Waiting for scene camera...")

        async def _mjpeg():
            last_seq = -1
            sent_placeholder = False
            while not pipeline.is_stopped:
                if await request.is_disconnected():
                    break
                jpg, seq = pipeline.get_scene_jpeg_with_seq()
                if jpg and seq != last_seq:
                    last_seq = seq
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                elif jpg is None and not sent_placeholder:
                    sent_placeholder = True
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + wait_jpg + b"\r\n"
                await asyncio.sleep(0.025)

        return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/eyes")
    async def stream_eyes(request: Request):
        wait_jpg = _placeholder_jpeg("Waiting for eye cams...", 360, 135)

        async def _mjpeg():
            last_seq = -1
            sent_placeholder = False
            while not pipeline.is_stopped:
                if await request.is_disconnected():
                    break
                jpg, seq = pipeline.get_eyes_jpeg_with_seq()
                if jpg and seq != last_seq:
                    last_seq = seq
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                elif jpg is None and not sent_placeholder:
                    sent_placeholder = True
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + wait_jpg + b"\r\n"
                await asyncio.sleep(0.08)

        return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/events")
    async def sse_events(request: Request):
        """Server-Sent Events pushing real-time telemetry, grounding breakdowns, dialogue & F-SoM packs."""
        ev_q: queue.Queue = queue.Queue(maxsize=64)

        def _on_ev(payload: Dict[str, Any]) -> None:
            try:
                ev_q.put_nowait(payload)
            except queue.Full:
                try:
                    ev_q.get_nowait()
                    ev_q.put_nowait(payload)
                except queue.Empty:
                    pass

        pipeline.add_event_listener(_on_ev)

        async def _stream():
            last_tick = 0.0
            try:
                # Send initial full state snapshot
                yield f"data: {json.dumps({'type': 'snapshot', 'data': pipeline.snapshot_state()}, ensure_ascii=False)}\n\n"
                while not pipeline.is_stopped:
                    if await request.is_disconnected():
                        break
                    # Flush discrete events first
                    while True:
                        try:
                            item = ev_q.get_nowait()
                            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                        except queue.Empty:
                            break
                    # Periodic 5Hz telemetry heartbeat (gaze state, FPS, scene count)
                    now = time.monotonic()
                    if now - last_tick >= 0.2:
                        last_tick = now
                        tick = {
                            "type": "tick",
                            "data": {
                                "telemetry": pipeline.telemetry.summary(),
                                "gaze": pipeline._latest_gaze.model_dump() if pipeline._latest_gaze else None,
                                "object_count": len(pipeline._latest_scene.objects) if pipeline._latest_scene else 0,
                                "recording": pipeline.recorder.is_recording,
                                "calibrated": pipeline.gaze_engine.calibrator.is_calibrated,
                                "mic_rms": pipeline.asr_worker.last_rms,
                                "asr_speaking": pipeline.asr_worker.is_speaking,
                            },
                        }
                        yield f"data: {json.dumps(tick, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.05)
            finally:
                pipeline.remove_event_listener(_on_ev)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app
