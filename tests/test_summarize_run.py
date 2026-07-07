from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_summarize_run_cli() -> None:
    row = {
        "episode_id": "e1",
        "success": True,
        "spl": 1.0,
        "final_distance_m": 0.0,
        "path_length_m": 3.0,
        "shortest_path_m": 3.0,
        "fail_at_cue_free": False,
        "cfsr_10": False,
        "cfsr_25": False,
        "cfsr_50": False,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.jsonl"
        path.write_text(json.dumps({"metrics": row}) + "\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/summarize_run.py", "--input", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["sr"] == 1.0
