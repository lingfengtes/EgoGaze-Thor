"""System 2 Edge VLM client supporting Ollama (qwen3-vl:8b), OpenAI-compatible VLM HTTP, and local HF.

Consumes Foveated Set-of-Mark (F-SoM) annotated full frames + optional high-resolution foveated ROI crops
to answer open-world visual questions ("我正看着的是什么？", "手的数字是几？", "读一下瓶子上的字").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

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
