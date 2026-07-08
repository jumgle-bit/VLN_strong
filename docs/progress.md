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
- Added a simulated `small_eval` dataset with 10 episodes, 12 steps per episode, and 120 generated outdoor VLN frames.
- Verified `small_eval` with the mock offline runner and summary metrics.
- Added a configurable ROS Noetic bridge for TurtleBot3 `waffle_pi` topics and Qwen/mock vision backends.
- Added a TurtleBot3 commander that converts `/taric/executable_heading` into conservative `/cmd_vel` commands.
- Added a ROS episode recorder for TurtleBot3 Gazebo image, camera-info, and odometry topics.

## Current Stage

You now have two offline replay datasets:

- `data/episodes/synthetic_30/manifest.jsonl`: 30 short engineering-validation episodes.
- `data/episodes/small_eval/manifest.jsonl`: 10 simulated evaluation episodes for the first controlled small-scale experiment.

Approximate progress: 50%.

## Not Yet Completed

- Inspect per-step Qwen outputs on `small_eval`.
- Smoke test the ROS bridge in Gazebo with `TARIC_VISION_BACKEND=mock`.
- Run the TurtleBot3 commander at low speed and verify `/cmd_vel`.
- Switch the ROS bridge to `TARIC_VISION_BACKEND=qwen` after mock topics are stable.
- Record `data/episodes/gazebo_small_eval/manifest.jsonl` from Gazebo using `taric_vln.ros.episode_recorder`.
- Build a simulator or use an existing outdoor simulator/ROS bag source for larger scale evaluation.
- Generate pseudolabels from Qwen for visible gate, tile scoring, and traversability sectors.
- Train or distill lightweight local modules.
- Run paper-scale evaluation: 100 episodes per scene, 3 random seeds, 600-1000m routes.
- Integrate with real ROS topics or robot hardware.

## Next Milestone

Run the simulated `small_eval` dataset end to end with the mock client:

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_mock_run.jsonl \
  --mock

python scripts/summarize_run.py --input outputs/small_eval_mock_run.jsonl
```

Then run a small Qwen pass on `small_eval`, inspect the JSON outputs, and generate pseudolabels for lightweight module training.
