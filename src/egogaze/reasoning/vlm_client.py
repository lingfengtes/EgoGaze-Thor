"""System 2 Edge VLM client supporting Ollama (qwen3-vl:8b), OpenAI-compatible VLM HTTP, and local HF.

Consumes Foveated Set-of-Mark (F-SoM) annotated full frames + optional high-resolution foveated ROI crops
to answer open-world visual questions ("我正看着的是什么？", "手的数字是几？", "读一下瓶子上的字").
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from egogaze.core.schemas import GroundingResult, SceneSnapshot
from egogaze.perception.fom_prompt import FoveatedPromptPack

logger = logging.getLogger("egogaze.vlm")


class EdgeVLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.mode = cfg.get("mode", "auto")  # ollama | http | local_hf | auto
        self.ollama_url = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = cfg.get("ollama_model", "qwen3-vl:8b")
        self.base_url = cfg.get("base_url", "http://127.0.0.1:8002").rstrip("/")
        self.model_id = cfg.get("model_id", "/model/Qwen3-VL-4B-Instruct")
        self.hf_model_name = cfg.get("hf_model_name", "Qwen/Qwen3-VL-4B-Instruct")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 160))

        self._hf_lock = threading.Lock()
        self._hf_model = None
        self._hf_processor = None
        self._resolved_backend = f"ollama:{self.ollama_model}"

    @property
    def backend_name(self) -> str:
        return self._resolved_backend

    def _ollama_model_ready(self) -> bool:
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.ollama_model in n or n.startswith(self.ollama_model.split(":")[0]) for n in names)
        except Exception:
            return False

    def _http_online(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=1.5)
            return r.status_code == 200
        except Exception:
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=1.5)
                return r.status_code == 200
            except Exception:
                return False

    def _ensure_local_hf(self) -> bool:
        if self._hf_model is not None:
            return True
        model_path = Path(self.model_id)
        if not model_path.exists():
            return False
        with self._hf_lock:
            if self._hf_model is not None:
                return True
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                logger.info("Loading local VLM from %s ...", model_path)
                self._hf_processor = AutoProcessor.from_pretrained(str(model_path))
                self._hf_model = AutoModelForImageTextToText.from_pretrained(
                    str(model_path),
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self._hf_model.eval()
                self._resolved_backend = f"local_hf:{model_path.name}"
                return True
            except Exception as exc:
                logger.warning("Failed to load local VLM: %s", exc)
                return False

    def _system_prompt(self) -> str:
        return (
            "你是运行在 Jetson Thor 上的第一人称视觉语音助手 EgoGaze-Thor。\n"
            "第一张图是带有红色十字注视点（表示用户视线焦点）和检测参考框的主视角画面；"
            "第二张图（如有）是用户注视目标的高清局部放大图。\n"
            "【最重要规则】：当用户问“我正看着什么/这是什么”时，必须严格以【红色十字注视点正下方】以及【第二张局部放大图中心】的物体为准！"
            "如果红色十字不在绿色检测框内，说明用户看的是绿框以外的物体，绝对不要误答绿框里的物体。\n"
            "请仔细观察图像真实细节（包括书本、纸张、手势、手指比出的数字、物品材质与文字等），"
            "用简洁自然的口语化中文直接回答（1~2句话），禁止输出括号注释或重复句子。"
        )

    def stream_reason(
        self,
        user_query: str,
        prompt_pack: FoveatedPromptPack,
    ) -> Tuple[Iterator[str], Dict[str, Any]]:
        metrics: Dict[str, Any] = {"backend": self._resolved_backend, "ttft_ms": 0.0, "total_ms": 0.0}

        def _gen() -> Iterator[str]:
            t0 = time.perf_counter_ns()
            first = True

            # 1. Native Ollama Vision API (qwen3-vl:8b on Jetson Thor GPU — fast, zero-crash, multi-image!)
            if self.mode in ("ollama", "auto") and self._ollama_model_ready():
                self._resolved_backend = f"ollama:{self.ollama_model}"
                metrics["backend"] = self._resolved_backend
                images = [prompt_pack.fsom_b64]
                if prompt_pack.roi_b64:
                    images.append(prompt_pack.roi_b64)
                payload = {
                    "model": self.ollama_model,
                    "stream": True,
                    "keep_alive": "30m",
                    "messages": [
                        {"role": "system", "content": self._system_prompt()},
                        {
                            "role": "user",
                            "content": (
                                f"场景检测参考:\n{prompt_pack.scene_legend}\n\n"
                                f"请根据图片直接回答用户问题：{user_query}"
                            ),
                            "images": images,
                        },
                        {"role": "assistant", "content": "<think>\n\n</think>\n"},
                    ],
                    "options": {
                        "num_ctx": 4096,
                        "temperature": 0.4,
                        "top_p": 0.85,
                        "repeat_penalty": 1.15,
                        "num_predict": self.max_new_tokens,
                    },
                }
                try:
                    with httpx.stream("POST", f"{self.ollama_url}/api/chat", json=payload, timeout=45.0) as resp:
                        resp.raise_for_status()
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            chunk = json.loads(raw_line)
                            if chunk.get("done"):
                                break
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                if first:
                                    metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                                    first = False
                                yield delta
                    metrics["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                    return
                except Exception as exc:
                    logger.warning("Ollama VLM stream failed: %s", exc)

            # 2. OpenAI-compatible VLM HTTP server
            if self.mode in ("http", "auto") and self._http_online():
                self._resolved_backend = f"http:{self.base_url}"
                metrics["backend"] = self._resolved_backend
                content: List[Dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prompt_pack.fsom_b64}"}}
                ]
                if prompt_pack.roi_b64:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prompt_pack.roi_b64}"}})
                content.append({"type": "text", "text": f"场景检测:\n{prompt_pack.scene_legend}\n用户问题: {user_query}"})
                payload = {
                    "model": self.hf_model_name,
                    "messages": [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": content}],
                    "max_tokens": self.max_new_tokens,
                    "temperature": 0.3,
                    "stream": True,
                }
                try:
                    with httpx.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload, timeout=30.0) as resp:
                        resp.raise_for_status()
                        for raw_line in resp.iter_lines():
                            line = raw_line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                if first:
                                    metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                                    first = False
                                yield delta
                    metrics["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                    return
                except Exception as exc:
                    logger.warning("VLM HTTP stream failed: %s", exc)

            # 3. Local HF Transformers fallback
            if self.mode in ("local_hf", "auto") and self._ensure_local_hf():
                metrics["backend"] = self._resolved_backend
                try:
                    import cv2
                    from PIL import Image
                    from transformers import TextIteratorStreamer

                    images = [Image.fromarray(cv2.cvtColor(prompt_pack.fsom_bgr, cv2.COLOR_BGR2RGB))]
                    if prompt_pack.roi_bgr is not None:
                        images.append(Image.fromarray(cv2.cvtColor(prompt_pack.roi_bgr, cv2.COLOR_BGR2RGB)))
                    hf_msgs = [
                        {
                            "role": "user",
                            "content": [{"type": "image"} for _ in images]
                            + [{"type": "text", "text": f"{self._system_prompt()}\n场景:\n{prompt_pack.scene_legend}\n问题：{user_query}"}],
                        }
                    ]
                    prompt_text = self._hf_processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
                    inputs = self._hf_processor(text=[prompt_text], images=images, return_tensors="pt")
                    inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
                    streamer = TextIteratorStreamer(self._hf_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    t_gen = threading.Thread(
                        target=self._hf_model.generate,
                        kwargs={**inputs, "max_new_tokens": self.max_new_tokens, "do_sample": False, "streamer": streamer},
                        daemon=True,
                    )
                    t_gen.start()
                    for piece in streamer:
                        if piece:
                            if first:
                                metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                                first = False
                            yield piece
                    t_gen.join(timeout=2.0)
                    metrics["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                    return
                except Exception as exc:
                    logger.warning("Local HF VLM error: %s", exc)

            # 4. Graceful fallback while Ollama model finishes downloading
            self._resolved_backend = "vlm_downloading_fallback"
            metrics["backend"] = self._resolved_backend
            metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
            yield f"端侧视觉模型 {self.ollama_model} 正在后台加载中，当前基于场景图识别到：{prompt_pack.scene_legend.splitlines()[0]}"
            metrics["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)

        return _gen(), metrics

    def locate_track_target(
        self,
        utterance_text: str,
        target_hint: str,
        frame_bgr: np.ndarray,
        scene: Optional[SceneSnapshot],
        grounding: Optional[GroundingResult] = None,
        max_width: int = 640,
    ) -> Dict[str, Any]:
        """System 2 VLM visual verification for open-world / misclassified tracking targets.

        When YOLO misclassifies an object (e.g. apple -> orange) or misses an open-world target completely,
        renders a clean Set-of-Mark numbered frame and asks qwen3-vl:8b to identify either:
          1) the exact SoM box ID (`som_id`) that truly contains the requested target, OR
          2) a normalized bounding box (`bbox_xyxy`) so OpenWorldPatchTracker can lock onto it.
        """
        t0 = time.perf_counter_ns()
        h, w = frame_bgr.shape[:2]

        # Build clean Set-of-Mark verification frame (large crisp [1], [2]... tags so VLM reads IDs accurately)
        annotated = frame_bgr.copy()
        legend_lines: List[str] = []
        valid_som_ids: set[int] = set()
        if scene and scene.objects:
            for obj in scene.objects:
                valid_som_ids.add(obj.som_id)
                x1, y1, x2, y2 = obj.bbox_xyxy
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 235, 130), 2)
                tag = f"[{obj.som_id}]"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                by1 = max(0, y1 - th - 10)
                cv2.rectangle(annotated, (x1, by1), (min(w - 1, x1 + tw + 10), by1 + th + 10), (0, 215, 255), -1)
                cv2.putText(annotated, tag, (x1 + 5, by1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (12, 16, 22), 2, cv2.LINE_AA)
                legend_lines.append(
                    f"[{obj.som_id}] det_label={obj.class_name}, color={obj.color}, pos=({obj.center_norm[0]:.2f},{obj.center_norm[1]:.2f})"
                )

        if grounding and grounding.gaze_centroid and grounding.gaze_reliability >= 0.30:
            gx, gy = int(grounding.gaze_centroid[0] * w), int(grounding.gaze_centroid[1] * h)
            cv2.circle(annotated, (gx, gy), 16, (0, 50, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (gx - 22, gy), (gx + 22, gy), (0, 50, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (gx, gy - 22), (gx, gy + 22), (0, 50, 255), 2, cv2.LINE_AA)

        if w > max_width:
            scale = max_width / float(w)
            send_img = cv2.resize(annotated, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            send_img = annotated
        ok, enc = cv2.imencode(".jpg", send_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        fsom_b64 = base64.b64encode(enc.tobytes()).decode("ascii") if ok else ""

        if not self.enabled or not (self.mode in ("ollama", "auto") and self._ollama_model_ready()):
            return {
                "status": "unavailable",
                "found": False,
                "som_id": None,
                "bbox_xyxy": None,
                "label": target_hint,
                "label_en": None,
                "reason": "VLM backend unavailable",
                "fsom_b64": fsom_b64,
                "latency_ms": round((time.perf_counter_ns() - t0) / 1e6, 2),
                "backend": "unavailable",
            }

        legend_str = "\n".join(legend_lines) if legend_lines else "(当前检测器未检出任何物体框)"
        sys_prompt = (
            "你是第一人称视觉目标定位器。用户想追踪画面中的某个目标物体。\n"
            "图中绿色框标有醒目黄色编号 [1], [2], ...。\n"
            "【极其重要】：检测器给出的 det_label 经常误识别（例如把苹果标成 orange 或 sports ball，把笔记本电脑标成 tv，把本子标成 book），"
            "你必须【完全以图像真实视觉内容】为准，不要被错误的 det_label 误导！\n"
            "只输出一行合法 JSON（不要 Markdown 代码块、不要多余解释）：\n"
            "1) 若某个编号框正好框住了用户要找的物体：\n"
            '{"found": true, "som_id": 编号整数, "bbox_1000": null, "label": "中文物体名", "label_en": "english_name", "reason": "10字以内原因"}\n'
            "2) 若画面里确实有该物体，但没有任何编号框框住它，请给出它在图像中的边界框（0~1000归一化坐标 [x1,y1,x2,y2]）：\n"
            '{"found": true, "som_id": null, "bbox_1000": [x1, y1, x2, y2], "label": "中文物体名", "label_en": "english_name", "reason": "10字以内原因"}\n'
            "3) 若画面里确实没有该物体：\n"
            '{"found": false, "som_id": null, "bbox_1000": null, "label": null, "label_en": null, "reason": "画面中未看到"}'
        )
        user_prompt = (
            f"用户完整指令：{utterance_text}\n"
            f"需要定位的目标：{target_hint}\n"
            f"当前编号框列表（det_label 可能有误，请看图核实）：\n{legend_str}\n"
            "请输出 JSON："
        )
        payload = {
            "model": self.ollama_model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt, "images": [fsom_b64]},
                {"role": "assistant", "content": "<think>\n\n</think>\n"},
            ],
            "options": {
                "num_ctx": 4096,
                "temperature": 0.1,
                "top_p": 0.85,
                "num_predict": 110,
            },
        }
        try:
            resp = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=25.0)
            resp.raise_for_status()
            raw_txt = resp.json().get("message", {}).get("content", "").strip()
            latency_ms = round((time.perf_counter_ns() - t0) / 1e6, 2)
            parsed = self._parse_locate_json(raw_txt, valid_som_ids, w, h)
            parsed["fsom_b64"] = fsom_b64
            parsed["latency_ms"] = latency_ms
            parsed["backend"] = f"ollama:{self.ollama_model}"
            parsed["status"] = "ok"
            logger.info(
                "VLM track locate (%s, %.0fms): found=%s som_id=%s bbox=%s label=%s raw=%s",
                target_hint,
                latency_ms,
                parsed["found"],
                parsed["som_id"],
                parsed["bbox_xyxy"],
                parsed["label"],
                raw_txt[:160],
            )
            return parsed
        except Exception as exc:
            logger.warning("VLM locate_track_target failed: %s", exc)
            return {
                "status": "error",
                "found": False,
                "som_id": None,
                "bbox_xyxy": None,
                "label": target_hint,
                "label_en": None,
                "reason": str(exc),
                "fsom_b64": fsom_b64,
                "latency_ms": round((time.perf_counter_ns() - t0) / 1e6, 2),
                "backend": f"ollama:{self.ollama_model}",
            }

    @staticmethod
    def _parse_locate_json(
        raw_txt: str,
        valid_som_ids: set[int],
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_txt, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        data: Dict[str, Any] = {}
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {}

        found = bool(data.get("found", False))
        som_id: Optional[int] = None
        raw_som = data.get("som_id")
        if raw_som is not None:
            m_id = re.search(r"\d+", str(raw_som))
            if m_id:
                cand_id = int(m_id.group(0))
                if cand_id in valid_som_ids:
                    som_id = cand_id
                    found = True

        bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
        raw_box = data.get("bbox_1000") or data.get("bbox_norm") or data.get("bbox")
        if som_id is None and isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            try:
                vals = [float(v) for v in raw_box]
                scale = 1.0 if max(vals) <= 1.05 else 1000.0
                x1 = int(max(0, min(img_w - 2, vals[0] / scale * img_w)))
                y1 = int(max(0, min(img_h - 2, vals[1] / scale * img_h)))
                x2 = int(max(0, min(img_w - 1, vals[2] / scale * img_w)))
                y2 = int(max(0, min(img_h - 1, vals[3] / scale * img_h)))
                if x2 - x1 >= 16 and y2 - y1 >= 16:
                    bbox_xyxy = (x1, y1, x2, y2)
                    found = True
            except Exception:
                bbox_xyxy = None

        if som_id is None and bbox_xyxy is None:
            found = False

        label = data.get("label")
        label_en = data.get("label_en")
        if isinstance(label_en, str):
            label_en = re.sub(r"[^a-zA-Z0-9_\- ]+", "", label_en).strip()[:24] or None
        else:
            label_en = None
        return {
            "found": found,
            "som_id": som_id,
            "bbox_xyxy": bbox_xyxy,
            "label": str(label).strip() if label else None,
            "label_en": label_en,
            "reason": str(data.get("reason", "")).strip(),
        }
