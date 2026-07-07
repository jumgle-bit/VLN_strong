from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.perception import QwenVisionClient
from taric_vln.types import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    client = QwenVisionClient(
        model=args.model,
        base_url=args.base_url,
        fallback_on_error=False,
    )
    try:
        observation = client.analyze(
            image_path=args.image,
            instruction=args.instruction,
            camera=CameraIntrinsics(),
            previous_state={},
            config=config,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
