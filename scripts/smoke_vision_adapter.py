from __future__ import annotations

import argparse
import json

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.perception import CommandVisionClient, PythonVisionClient
from taric_vln.types import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--python-adapter", default=None)
    parser.add_argument("--command-adapter", default=None)
    args = parser.parse_args()

    if bool(args.python_adapter) == bool(args.command_adapter):
        parser.error("Pass exactly one of --python-adapter or --command-adapter.")

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    client = (
        PythonVisionClient(args.python_adapter)
        if args.python_adapter
        else CommandVisionClient(args.command_adapter)
    )
    observation = client.analyze(
        image_path=args.image,
        instruction=args.instruction,
        camera=CameraIntrinsics(),
        previous_state={},
        config=config,
    )
    print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
