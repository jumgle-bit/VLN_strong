from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.perception import (
    CommandVisionClient,
    CueExtractor,
    DeepSeekVLMClient,
    MockVLMClient,
    PythonVisionClient,
)
from taric_vln.sim.offline_runner import OfflineEpisodeRunner, group_by_episode, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--python-adapter", default=None)
    parser.add_argument("--command-adapter", default=None)
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    grounder = TraversabilityGrounder(config)
    choices = [args.mock, bool(args.python_adapter), bool(args.command_adapter)]
    if sum(1 for enabled in choices if enabled) > 1:
        parser.error("Use only one of --mock, --python-adapter, or --command-adapter.")
    if args.mock:
        client = MockVLMClient()
    elif args.python_adapter:
        client = PythonVisionClient(args.python_adapter)
    elif args.command_adapter:
        client = CommandVisionClient(args.command_adapter)
    else:
        client = DeepSeekVLMClient()
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
