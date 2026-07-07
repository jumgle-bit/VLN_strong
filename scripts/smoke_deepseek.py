from __future__ import annotations

import argparse
import json

from taric_vln.config import TaricConfig
from taric_vln.perception import DeepSeekVLMClient
from taric_vln.types import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    client = DeepSeekVLMClient(fallback_on_error=False)
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
