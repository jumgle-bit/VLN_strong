from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taric_vln.config import TaricConfig
from taric_vln.perception import DeepSeekVLMClient
from taric_vln.types import CameraIntrinsics


def generate_pseudolabels(
    manifest_rows: list[dict[str, Any]],
    output_path: str | Path,
    client: DeepSeekVLMClient,
    config: TaricConfig | None = None,
) -> None:
    config = config or TaricConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(manifest_rows):
            camera = CameraIntrinsics.from_dict(row.get("camera"))
            observation = client.analyze(
                image_path=row["image_path"],
                instruction=str(row.get("instruction", "")),
                camera=camera,
                previous_state={"row_index": idx},
                config=config,
            )
            label = {
                "source": row,
                "label": observation.to_dict(),
            }
            f.write(json.dumps(label, ensure_ascii=True) + "\n")
