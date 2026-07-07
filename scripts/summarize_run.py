from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.eval import EpisodeMetrics, summarize_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    metrics = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metric = row.get("metrics", row)
        if metric:
            metrics.append(metrics_from_dict(metric))

    summary = summarize_metrics(metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def metrics_from_dict(data: dict) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=str(data["episode_id"]),
        success=bool(data["success"]),
        spl=float(data["spl"]),
        final_distance_m=float(data["final_distance_m"]),
        path_length_m=float(data["path_length_m"]),
        shortest_path_m=float(data["shortest_path_m"]),
        fail_at_cue_free=bool(data["fail_at_cue_free"]),
        cfsr_10=bool(data["cfsr_10"]),
        cfsr_25=bool(data["cfsr_25"]),
        cfsr_50=bool(data["cfsr_50"]),
    )


if __name__ == "__main__":
    main()
