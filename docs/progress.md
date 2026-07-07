# Reproduction Progress

This document tracks the TARIC/VLN reproduction plan at the current engineering checkpoint.

## Completed

- Read and summarized the TARIC paper goal: outdoor VLN under semantic-cue interruptions, using traversability-aware heading grounding and 3D cue memory.
- Created and pushed the GitHub project at `git@github.com:jumgle-bit/VLN_strong.git`.
- Set up a runnable Python package with tests, docs, scripts, and sample data.
- Implemented the core TARIC scaffold:
  - cue extraction interface
  - traversability-aware heading grounding
  - particle-based 3D cue memory
  - offline episode runner
  - SR/SPL/Fail@CF/CFSR metrics
  - ROS Noetic bridge skeleton
- Verified DeepSeek API access and confirmed this endpoint is text-only.
- Added `DeepSeekTextPlanner` for instruction decomposition.
- Added custom vision model adapters for Python and command-line integrations.
- Added Qwen/DashScope vision API support.
- Confirmed Qwen image access with the synthetic `library_entrance.png` smoke-test image.
- Added nonstandard/truncated Qwen JSON recovery for required TARIC fields.
- Added a generated 30-episode synthetic VLN dataset for small-scale offline pipeline validation.

## Current Stage

You are now at the start of offline VLN episode replay. The perception frontends are usable, and the next task is to build image-sequence manifests with poses and goals.

Approximate progress: 40-45%.

## Not Yet Completed

- Build real or simulated VLN datasets with image sequences, camera intrinsics, robot poses, goal positions, and cue-availability labels.
- Run 30 short offline episodes and inspect per-step outputs.
- Build a simulator or use an existing outdoor simulator/ROS bag source for larger scale evaluation.
- Generate pseudolabels from Qwen for visible gate, tile scoring, and traversability sectors.
- Train or distill lightweight local modules.
- Run paper-scale evaluation: 100 episodes per scene, 3 random seeds, 600-1000m routes.
- Integrate with real ROS topics or robot hardware.

## Next Milestone

Run the included 30-episode synthetic dataset end to end:

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/synthetic_30/manifest.jsonl \
  --output outputs/synthetic_30_run.jsonl \
  --qwen

python scripts/summarize_run.py --input outputs/synthetic_30_run.jsonl
```

After this works, replace the synthetic manifest with real sequences collected from simulation, video, or ROS bags.
