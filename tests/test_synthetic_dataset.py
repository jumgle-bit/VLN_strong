from __future__ import annotations

import json
from pathlib import Path


def test_synthetic_30_dataset_manifest_and_images_exist() -> None:
    manifest = Path("data/episodes/synthetic_30/manifest.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    episode_ids = {row["episode_id"] for row in rows}

    assert len(rows) == 240
    assert len(episode_ids) == 30
    assert all(Path(row["image_path"]).exists() for row in rows)
    assert all("cue_available_gt" in row for row in rows)
