from __future__ import annotations

import argparse
import json

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.perception import DeepSeekTextPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    plan = DeepSeekTextPlanner().plan(args.instruction, config=config)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
