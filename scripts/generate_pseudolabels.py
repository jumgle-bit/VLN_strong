from __future__ import annotations

import argparse

from taric_vln.config import TaricConfig
from taric_vln.perception import DeepSeekVLMClient
from taric_vln.sim.offline_runner import load_manifest
from taric_vln.training import generate_pseudolabels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = TaricConfig.from_json(args.config) if args.config else TaricConfig()
    rows = load_manifest(args.manifest)
    generate_pseudolabels(
        rows,
        output_path=args.output,
        client=DeepSeekVLMClient(),
        config=config,
    )


if __name__ == "__main__":
    main()
