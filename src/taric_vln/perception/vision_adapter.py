from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

from taric_vln.config import TaricConfig
from taric_vln.geometry import pixel_to_bearing
from taric_vln.perception.deepseek_client import parse_float, parse_float_list, parse_focus_pixel
from taric_vln.types import CameraIntrinsics, VLMObservation


class PythonVisionClient:
    """Load a user-provided Python vision adapter.

    The adapter can be a class, an object with `analyze(...)`, or a function.
    Set it with `TARIC_VISION_ADAPTER=module.path:ClassName` or pass a `.py`
    file path such as `/home/user/my_adapter.py:MyAdapter`.
    """

    def __init__(self, adapter: str | None = None) -> None:
        self.adapter_ref = adapter or os.getenv("TARIC_VISION_ADAPTER", "")
        if not self.adapter_ref:
            raise RuntimeError("Set TARIC_VISION_ADAPTER or pass adapter='module:Class'.")
        self.adapter = load_symbol(self.adapter_ref)
        if isinstance(self.adapter, type):
            self.adapter = self.adapter()

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
        if hasattr(self.adapter, "analyze"):
            result = self.adapter.analyze(
                image_path=str(image_path),
                instruction=instruction,
                camera=camera.to_dict(),
                previous_state=previous_state or {},
                config=config.to_dict(),
            )
        else:
            result = self.adapter(
                image_path=str(image_path),
                instruction=instruction,
                camera=camera.to_dict(),
                previous_state=previous_state or {},
                config=config.to_dict(),
            )
        return observation_from_result(result, camera, config)


class CommandVisionClient:
    """Call an external command that reads JSON from stdin and returns JSON."""

    def __init__(self, command: str | None = None, timeout_s: float = 60.0) -> None:
        self.command = command or os.getenv("TARIC_VISION_COMMAND", "")
        if not self.command:
            raise RuntimeError("Set TARIC_VISION_COMMAND or pass command='...'.")
        self.timeout_s = timeout_s

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
        payload = {
            "image_path": str(image_path),
            "instruction": instruction,
            "camera": camera.to_dict(),
            "previous_state": previous_state or {},
            "config": config.to_dict(),
        }
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Vision command failed with code {completed.returncode}: {completed.stderr.strip()}"
            )
        return observation_from_result(json.loads(completed.stdout), camera, config)


def load_symbol(ref: str) -> Any:
    module_ref, _, symbol_name = ref.rpartition(":")
    if not module_ref or not symbol_name:
        raise ValueError("Adapter reference must be 'module:Symbol' or '/path/file.py:Symbol'.")

    module_path = Path(module_ref)
    if module_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import adapter file: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return getattr(module, symbol_name)


def observation_from_result(
    result: VLMObservation | dict[str, Any],
    camera: CameraIntrinsics,
    config: TaricConfig,
) -> VLMObservation:
    if isinstance(result, VLMObservation):
        return result
    if not isinstance(result, dict):
        raise TypeError("Vision adapter must return VLMObservation or dict.")

    focus = parse_focus_pixel(result.get("focus_pixel"))
    bearing = parse_float(result.get("cue_bearing_rad"), default=None)
    if bearing is None:
        bearing_deg = parse_float(result.get("cue_bearing_deg"), default=None)
        bearing = 0.0 if bearing_deg is None else bearing_deg * 3.141592653589793 / 180.0
    if focus is not None:
        bearing = pixel_to_bearing(focus, camera)

    return VLMObservation(
        exploration_phrase=str(result.get("exploration_phrase", "")),
        goal_phrase=str(result.get("goal_phrase", "")),
        visible=bool(result.get("visible", False)),
        focus_pixel=focus,
        tile_scores=parse_float_list(result.get("tile_scores"), None),
        cue_bearing_rad=float(bearing),
        traversability_scores=parse_float_list(
            result.get("traversability_scores"), config.heading_sectors
        ),
        confidence=float(parse_float(result.get("confidence"), default=0.0) or 0.0),
        raw=result,
        error=result.get("error"),
    )
