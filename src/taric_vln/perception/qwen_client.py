from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from taric_vln.config import TaricConfig
from taric_vln.perception.deepseek_client import (
    DeepSeekAPIError,
    build_vlm_prompt,
    extract_json_object,
    format_http_error,
    image_to_data_url,
)
from taric_vln.perception.vision_adapter import observation_from_result
from taric_vln.types import CameraIntrinsics, VLMObservation


class QwenVisionClient:
    """DashScope/Qwen OpenAI-compatible vision API adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 2,
        min_interval_s: float = 0.0,
        max_tokens: int | None = 512,
        fallback_on_error: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY", "")
        self.model = model or os.getenv("QWEN_VISION_MODEL", "qwen-vl-plus")
        self.base_url = (
            base_url
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.min_interval_s = min_interval_s
        self.max_tokens = max_tokens
        self.fallback_on_error = fallback_on_error
        self._last_call_time = 0.0

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
        camera: CameraIntrinsics | None = None,
        previous_state: dict[str, Any] | None = None,
        config: TaricConfig | None = None,
    ) -> VLMObservation:
        camera = camera or CameraIntrinsics()
        config = config or TaricConfig()
        try:
            if not self.api_key:
                raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not set.")
            payload = self._build_payload(Path(image_path), instruction, camera, previous_state, config)
            started_at = time.time()
            response = self._post_json(payload)
            latency_s = time.time() - started_at
            content = response["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            parsed["_api_meta"] = {
                "latency_s": latency_s,
                "model": self.model,
                "usage": response.get("usage", {}),
            }
            return observation_from_result(parsed, camera, config)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            return VLMObservation(
                exploration_phrase=instruction,
                goal_phrase=instruction,
                visible=False,
                focus_pixel=None,
                tile_scores=[],
                cue_bearing_rad=0.0,
                traversability_scores=[0.0] * config.heading_sectors,
                confidence=0.0,
                error=str(exc),
            )

    def simple_image_json(self, image_path: str | Path, instruction: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not set.")
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only strict compact JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(Path(image_path))}},
                        {
                            "type": "text",
                            "text": (
                                "Briefly inspect this image for a navigation task. "
                                f"Instruction: {instruction}. "
                                "Return JSON with keys ok, visible_goal, short_description."
                            ),
                        },
                    ],
                },
            ],
        }
        self._set_max_tokens(payload, 128)
        response = self._post_json(payload)
        content = response["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        parsed["_api_meta"] = {"model": self.model, "usage": response.get("usage", {})}
        return parsed

    def text_only_json(self, instruction: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not set.")
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only strict JSON."},
                {
                    "role": "user",
                    "content": (
                        "This is a Qwen vision endpoint connectivity diagnostic. "
                        f"Instruction: {instruction}. "
                        "Return JSON with keys ok, model_role, note."
                    ),
                },
            ],
        }
        self._set_max_tokens(payload, 128)
        response = self._post_json(payload)
        content = response["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        parsed["_api_meta"] = {"model": self.model, "usage": response.get("usage", {})}
        return parsed

    def payload_size_bytes(
        self,
        image_path: str | Path,
        instruction: str,
        camera: CameraIntrinsics | None = None,
        previous_state: dict[str, Any] | None = None,
        config: TaricConfig | None = None,
    ) -> int:
        camera = camera or CameraIntrinsics()
        config = config or TaricConfig()
        payload = self._build_payload(Path(image_path), instruction, camera, previous_state, config)
        return len(json.dumps(payload).encode("utf-8"))

    def _build_payload(
        self,
        image_path: Path,
        instruction: str,
        camera: CameraIntrinsics,
        previous_state: dict[str, Any] | None,
        config: TaricConfig,
    ) -> dict[str, Any]:
        prompt = build_vlm_prompt(instruction, camera, previous_state, config)
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the vision-language navigation perception frontend. "
                        "Return only strict JSON matching the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
        self._set_max_tokens(payload)
        return payload

    def _set_max_tokens(self, payload: dict[str, Any], override: int | None = None) -> None:
        max_tokens = override if override is not None else self.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._respect_rate_limit()
            request = Request(url, data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = DeepSeekAPIError(format_http_error(exc))
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 8.0))
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"Qwen vision API request failed: {last_error}")

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call_time = time.time()
