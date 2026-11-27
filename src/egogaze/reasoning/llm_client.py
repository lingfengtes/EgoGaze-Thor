"""Unified streaming LLM client for Jetson Thor TensorRT-Edge-LLM C++ daemon & OpenAI endpoints.

Key reliability features:
1. Sentence-buffered deduplication & meta-note stripping (prevents NVFP4 repetition loops from reaching UI/TTS).
2. Always drains the C++ edgellm_daemon SSE response cleanly to [DONE] even after suppressing duplicates,
   preventing cpp-httplib from destroying the daemon's joinable std::thread mid-stream.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

logger = logging.getLogger("egogaze.llm")

_SENTENCE_END_RE = re.compile(r"[。！？!?;\n]")
_META_MARKERS = ("（注：", "(注:", "（若需", "(若需", "（根据实际", "（说明：", "【注")


def _normalize_for_dedup(s: str) -> str:
    return re.sub(r"[\s，,。！？!?;；：:\"'“”‘’()（）\[\]]+", "", s).strip().lower()


def _is_duplicate_clause(candidate: str, seen_norms: List[str]) -> bool:
    norm = _normalize_for_dedup(candidate)
    if len(norm) < 2:
        return False
    for prev in seen_norms:
        if norm == prev:
            return True
        # Catch short repeating tail phrases like "星期五。" after "今天星期五。"
        if len(norm) >= 2 and (prev.endswith(norm) or norm in prev):
            return True
        # Catch restarts where the model starts repeating the exact opening of an earlier sentence
        prefix_len = min(len(norm), len(prev))
        if prefix_len >= 6 and (norm.startswith(prev[:prefix_len]) or prev.startswith(norm[:prefix_len])):
            return True
    return False


class UnifiedLLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.backends: Dict[str, Dict[str, Any]] = cfg.get("llm_backends", {})
        self.active_name: str = cfg.get("active_llm_backend", "edgellm_nvfp4")

    @property
    def active_config(self) -> Dict[str, Any]:
        return self.backends.get(
            self.active_name,
            {"type": "edgellm_daemon", "base_url": "http://127.0.0.1:8000", "model": "Qwen3-8B-NVFP4", "timeout_s": 15.0},
        )

    def switch_backend(self, name: str) -> bool:
        if name in self.backends:
            self.active_name = name
            logger.info("Switched LLM backend to %s (%s)", name, self.active_config.get("model"))
            return True
        return False

    def health_check(self, name: Optional[str] = None) -> Dict[str, Any]:
        key = name or self.active_name
        bcfg = self.backends.get(key, self.active_config)
        base = bcfg["base_url"].rstrip("/")
        path = "/health" if bcfg.get("type") == "edgellm_daemon" else "/v1/models"
        try:
            r = httpx.get(f"{base}{path}", timeout=2.0)
            return {"backend": key, "model": bcfg.get("model"), "online": r.status_code == 200, "status": r.status_code}
        except Exception as exc:
            return {"backend": key, "model": bcfg.get("model"), "online": False, "error": str(exc)}

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 80,
        temperature: float = 0.65,
    ) -> Tuple[Iterator[str], Dict[str, Any]]:
        """Return (deduplicated_chunk_iterator, metrics_dict). Always drains SSE cleanly to [DONE]."""
        bcfg = self.active_config
        btype = bcfg.get("type", "openai")
        base = bcfg["base_url"].rstrip("/")
        timeout = float(bcfg.get("timeout_s", 15.0))
        metrics: Dict[str, Any] = {
            "backend": self.active_name,
            "model": bcfg.get("model", "unknown"),
            "ttft_ms": 0.0,
            "total_ms": 0.0,
        }

        if btype == "ollama":
            url = f"{base}/api/chat"
            ollama_msgs = list(messages)
            # Prefill empty think block so hybrid Qwen3/Qwen3-VL skips thinking (~200ms TTFT!)
            if "qwen3" in str(bcfg.get("model", "")).lower():
                ollama_msgs.append({"role": "assistant", "content": "<think>\n\n</think>\n"})
            payload = {
                "model": bcfg.get("model", "qwen3-vl:8b"),
                "messages": ollama_msgs,
                "stream": True,
                "keep_alive": "30m",
                "options": {
                    "num_ctx": 4096,
                    "temperature": temperature,
                    "top_p": 0.85,
                    "top_k": 20,
                    "repeat_penalty": 1.15,
                    "num_predict": max_tokens,
                },
            }
        elif btype == "edgellm_daemon":
            url = f"{base}/v1/chat/completions/stream"
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.85,
                "top_k": 20,
                "enable_thinking": False,
            }
        else:
            url = f"{base}/v1/chat/completions"
            payload = {
                "model": bcfg.get("model", "Qwen3-4B-fp8"),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.85,
                "stream": True,
                "enable_thinking": False,
            }

        def _gen() -> Iterator[str]:
            t0 = time.perf_counter_ns()
            first = True
            pending = ""
            seen_norms: List[str] = []
            suppressed = False
            emitted_chars = 0

            try:
                with httpx.stream("POST", url, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if btype == "ollama":
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("done"):
                                break
                            if suppressed:
                                continue
                            delta = chunk.get("message", {}).get("content", "")
                        else:
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            if suppressed:
                                continue
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if "error" in chunk:
                                logger.error("LLM stream error: %s", chunk["error"])
                                suppressed = True
                                continue
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if not delta:
                            continue
                        if first:
                            metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                            first = False

                        pending += delta

                        # Cut off meta-commentary notes cleanly
                        for marker in _META_MARKERS:
                            idx = pending.find(marker)
                            if idx != -1:
                                head = pending[:idx].strip()
                                if head and not _is_duplicate_clause(head, seen_norms):
                                    seen_norms.append(_normalize_for_dedup(head))
                                    emitted_chars += len(head)
                                    yield head
                                pending = ""
                                suppressed = True
                                break
                        if suppressed:
                            continue

                        # Emit sentence-by-sentence with duplicate detection
                        while True:
                            m = _SENTENCE_END_RE.search(pending)
                            if not m:
                                # Also catch mid-sentence restart repetition early
                                if len(pending) >= 12 and _is_duplicate_clause(pending, seen_norms):
                                    pending = ""
                                    suppressed = True
                                break
                            clause = pending[: m.end()].strip()
                            pending = pending[m.end() :]
                            if not clause:
                                continue
                            if _is_duplicate_clause(clause, seen_norms):
                                logger.info("Suppressed duplicate LLM sentence: %r", clause)
                                suppressed = True
                                pending = ""
                                break
                            seen_norms.append(_normalize_for_dedup(clause))
                            emitted_chars += len(clause)
                            yield clause
                            if emitted_chars >= 120:
                                suppressed = True
                                pending = ""
                                break

                # Flush any remaining valid non-duplicate tail
                tail = pending.strip()
                if tail and not suppressed and not _is_duplicate_clause(tail, seen_norms):
                    yield tail

            except Exception as exc:
                logger.warning("LLM backend %s unreachable (%s); using structured fallback reply", self.active_name, exc)
                if first:
                    metrics["ttft_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
                yield f"[离线回退 | {self.active_name} 未启动] 请运行 start_llm_thor.sh 启动 Thor 端侧模型。"
            finally:
                metrics["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)

        return _gen(), metrics
