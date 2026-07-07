from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import add_repo_src_to_path

add_repo_src_to_path()

from taric_vln.config import TaricConfig
from taric_vln.perception import DeepSeekVLMClient
from taric_vln.types import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Send a no-image diagnostic request to verify key/model/base_url.",
    )
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    client = DeepSeekVLMClient(fallback_on_error=False)
    try:
        if args.text_only:
            payload = client.text_only_json(args.instruction, config=config)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if not args.image:
            parser.error("--image is required unless --text-only is used")

        observation = client.analyze(
            image_path=args.image,
            instruction=args.instruction,
            camera=CameraIntrinsics(),
            previous_state={},
            config=config,
        )
        print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        message = str(exc)
        print(message, file=sys.stderr)
        if "HTTP 400" in message and "image" in message.lower():
            print(
                "\nThe text API is reachable, but this endpoint appears to reject image input. "
                "Run the same command with --text-only to confirm the key/model first.",
                file=sys.stderr,
            )
        elif "HTTP 400" in message:
            print(
                "\nHTTP 400 means DeepSeek rejected the request shape. "
                "Try --text-only. If --text-only works but image mode fails, use a vision-capable "
                "endpoint/model or replace this client with a VLM adapter.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
