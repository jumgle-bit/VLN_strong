from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from taric_vln.config import TaricConfig
from taric_vln.geometry import pixel_to_bearing
from taric_vln.types import CameraIntrinsics, VLMObservation


class DeepSeekAPIError(RuntimeError):
    """Raised when the DeepSeek HTTP API rejects a request."""


class DeepSeekVLMClient:
    """OpenAI-compatible DeepSeek client for image-conditioned JSON outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 2,
        min_interval_s: float = 0.0,
        cache_dir: str | Path | None = ".taric_cache/deepseek",
        fallback_on_error: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.min_interval_s = min_interval_s
        self.fallback_on_error = fallback_on_error
        self._last_call_time = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        image_path = Path(image_path)
        try:
            cached = self._read_cache(image_path, instruction, camera, previous_state, config)
            if cached is not None:
                return self._observation_from_payload(cached, camera, config)

            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set.")

            payload = self._build_payload(image_path, instruction, camera, previous_state, config)
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
            self._write_cache(image_path, instruction, camera, previous_state, config, parsed)
            return self._observation_from_payload(parsed, camera, config)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            return self._fallback_observation(str(exc), instruction, config)

    def text_only_json(
        self,
        instruction: str,
        config: TaricConfig | None = None,
    ) -> dict[str, Any]:
        """Small diagnostic request that does not include an image."""

        config = config or TaricConfig()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "Return only strict JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        "This is a text-only DeepSeek API diagnostic for a VLN project. "
                        f"Instruction: {instruction}\n"
                        "Return JSON with keys: ok, model_role, note. "
                        f"The navigation config has {config.heading_sectors} heading sectors."
                    ),
                },
            ],
        }
        response = self._post_json(payload)
        content = response["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        parsed["_api_meta"] = {"model": self.model, "usage": response.get("usage", {})}
        return parsed

    def _build_payload(
        self,
        image_path: Path,
        instruction: str,
        camera: CameraIntrinsics,
        previous_state: dict[str, Any] | None,
        config: TaricConfig,
    ) -> dict[str, Any]:
        prompt = build_vlm_prompt(instruction, camera, previous_state, config)
        return {
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
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                },
            ],
        }

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
        raise RuntimeError(f"DeepSeek API request failed: {last_error}")

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call_time = time.time()

    def _observation_from_payload(
        self, payload: dict[str, Any], camera: CameraIntrinsics, config: TaricConfig
    ) -> VLMObservation:
        focus = parse_focus_pixel(payload.get("focus_pixel"))
        bearing = parse_float(payload.get("cue_bearing_rad"), default=None)
        if bearing is None:
            bearing_deg = parse_float(payload.get("cue_bearing_deg"), default=None)
            bearing = 0.0 if bearing_deg is None else bearing_deg * 3.141592653589793 / 180.0
        if focus is not None:
            bearing = pixel_to_bearing(focus, camera)

        scores = parse_float_list(payload.get("traversability_scores"), config.heading_sectors)
        tile_scores = parse_float_list(payload.get("tile_scores"), None)
        return VLMObservation(
            exploration_phrase=str(payload.get("exploration_phrase", "")),
            goal_phrase=str(payload.get("goal_phrase", "")),
            visible=bool(payload.get("visible", False)),
            focus_pixel=focus,
            tile_scores=tile_scores,
            cue_bearing_rad=float(bearing),
            traversability_scores=scores,
            confidence=float(parse_float(payload.get("confidence"), default=0.0) or 0.0),
            raw=payload,
        )

    def _fallback_observation(
        self, error: str, instruction: str, config: TaricConfig
    ) -> VLMObservation:
        return VLMObservation(
            exploration_phrase=instruction,
            goal_phrase=instruction,
            visible=False,
            focus_pixel=None,
            tile_scores=[],
            cue_bearing_rad=0.0,
            traversability_scores=[0.0] * config.heading_sectors,
            confidence=0.0,
            error=error,
        )

    def _cache_key(
        self,
        image_path: Path,
        instruction: str,
        camera: CameraIntrinsics,
        previous_state: dict[str, Any] | None,
        config: TaricConfig,
    ) -> str:
        h = hashlib.sha256()
        h.update(image_path.read_bytes())
        h.update(instruction.encode("utf-8"))
        h.update(json.dumps(camera.to_dict(), sort_keys=True).encode("utf-8"))
        h.update(json.dumps(previous_state or {}, sort_keys=True).encode("utf-8"))
        h.update(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8"))
        h.update(self.model.encode("utf-8"))
        return h.hexdigest()

    def _read_cache(
        self,
        image_path: Path,
        instruction: str,
        camera: CameraIntrinsics,
        previous_state: dict[str, Any] | None,
        config: TaricConfig,
    ) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{self._cache_key(image_path, instruction, camera, previous_state, config)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(
        self,
        image_path: Path,
        instruction: str,
        camera: CameraIntrinsics,
        previous_state: dict[str, Any] | None,
        config: TaricConfig,
        payload: dict[str, Any],
    ) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{self._cache_key(image_path, instruction, camera, previous_state, config)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class MockVLMClient:
    def __init__(self, observations: list[VLMObservation] | None = None) -> None:
        self.observations = observations or []
        self.calls = 0

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
        camera: CameraIntrinsics | None = None,
        previous_state: dict[str, Any] | None = None,
        config: TaricConfig | None = None,
    ) -> VLMObservation:
        config = config or TaricConfig()
        if self.calls < len(self.observations):
            obs = self.observations[self.calls]
        else:
            obs = VLMObservation(
                exploration_phrase=instruction,
                goal_phrase=instruction,
                visible=False,
                focus_pixel=None,
                tile_scores=[],
                cue_bearing_rad=0.0,
                traversability_scores=[1.0] * config.heading_sectors,
                confidence=1.0,
            )
        self.calls += 1
        return obs


def build_vlm_prompt(
    instruction: str,
    camera: CameraIntrinsics,
    previous_state: dict[str, Any] | None,
    config: TaricConfig,
) -> str:
    return (
        "Task: outdoor vision-language navigation perception.\n"
        f"Instruction: {instruction}\n"
        f"Camera: {json.dumps(camera.to_dict(), sort_keys=True)}\n"
        f"Previous state: {json.dumps(previous_state or {}, sort_keys=True)}\n"
        f"Use {config.heading_sectors} left-to-right heading sectors covering "
        f"+/-{config.max_bearing_deg} degrees. Index 0 is image-left.\n"
        "Return strict compact JSON with exactly these keys and no other keys:\n"
        "{\n"
        '  "exploration_phrase": "coarse navigation phrase",\n'
        '  "goal_phrase": "specific target phrase",\n'
        '  "visible": true,\n'
        '  "focus_pixel": [u, v] or null,\n'
        '  "tile_scores": [numbers for coarse target saliency],\n'
        '  "cue_bearing_deg": number, \n'
        '  "traversability_scores": [10 numbers from 0 to 1],\n'
        '  "confidence": number from 0 to 1\n'
        "}\n"
        "Do not include camera, previous_state, image URLs, target descriptions, path descriptions, "
        "Markdown fences, comments, or explanatory text. "
        "If the goal is not confidently visible, set visible=false and focus_pixel=null, "
        "but still provide an exploration cue and traversability scores. "
        "Traversability score 1 means safe and easy to drive, 0 means blocked or unsafe."
    )


def image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def format_http_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = body.strip()
    if len(body) > 2000:
        body = body[:2000] + "...<truncated>"
    if body:
        return f"HTTP {exc.code} {exc.reason}: {body}"
    return f"HTTP {exc.code} {exc.reason}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def parse_focus_pixel(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "u" in value and "v" in value:
            return (float(value["u"]), float(value["v"]))
        if "x" in value and "y" in value:
            return (float(value["x"]), float(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_float_list(value: Any, expected_len: int | None) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        parsed = parse_float(item, default=None)
        if parsed is not None:
            result.append(float(parsed))
    if expected_len is not None:
        result = result[:expected_len]
        if result and len(result) < expected_len:
            result.extend([result[-1]] * (expected_len - len(result)))
    return result
