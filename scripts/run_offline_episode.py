from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.perception import CueExtractor, DeepSeekVLMClient, MockVLMClient
from taric_vln.sim.offline_runner import OfflineEpisodeRunner, group_by_episode, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    grounder = TraversabilityGrounder(config)
    client = MockVLMClient() if args.mock else DeepSeekVLMClient()
    extractor = CueExtractor(client, config=config, grounder=grounder)
    runner = OfflineEpisodeRunner(extractor, config=config, grounder=grounder)

    rows = load_manifest(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for _episode_id, steps in group_by_episode(rows).items():
            record = runner.run_steps(steps)
            f.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
