from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Send a no-image request to verify key/model/base_url first.",
    )
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    client = QwenVisionClient(
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout,
        retries=args.retries,
        fallback_on_error=False,
    )
    try:
        print(
            f"Qwen endpoint={client.base_url} model={client.model} timeout={args.timeout}s retries={args.retries}",
            file=sys.stderr,
        )
        if args.diagnose:
            payload = client.text_only_json(args.instruction)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        image_path = Path(args.image)
        if not image_path.exists():
            raise RuntimeError(f"Image not found: {image_path}")
        payload_size = client.payload_size_bytes(
            image_path,
            args.instruction,
            camera=CameraIntrinsics(),
            previous_state={},
            config=config,
        )
        print(
            f"image_size={image_path.stat().st_size} bytes request_payload={payload_size} bytes",
            file=sys.stderr,
        )
        observation = client.analyze(
            image_path=image_path,
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
