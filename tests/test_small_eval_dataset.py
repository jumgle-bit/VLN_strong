from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def test_small_eval_dataset_manifest_and_images_exist() -> None:
    manifest = Path("data/episodes/small_eval/manifest.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    episode_counts = Counter(row["episode_id"] for row in rows)

    assert len(rows) == 120
    assert len(episode_counts) == 10
    assert set(episode_counts.values()) == {12}
    assert all(Path(row["image_path"]).exists() for row in rows)
    assert all("cue_available_gt" in row for row in rows)
    assert any(not row["cue_available_gt"] for row in rows)
    assert all(row["image_path"].startswith("data/episodes/small_eval/images/") for row in rows)
