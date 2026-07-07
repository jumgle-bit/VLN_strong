# TARIC VLN DeepSeek Reproduction Scaffold

This repository implements a runnable engineering scaffold for reproducing the TARIC outdoor VLN pipeline with a DeepSeek VLM frontend.

It includes:

- OpenAI-compatible `DeepSeekVLMClient` with image-input diagnostics, JSON parsing, retries, rate limiting, local response cache, and fallback behavior.
- `DeepSeekTextPlanner` for text-only instruction decomposition when the DeepSeek endpoint does not support image input.
- `QwenVisionClient` for DashScope/Qwen vision API calls.
- Python and command-line adapters for plugging in your own vision model.
- Cue extraction, visibility gating, traversability-aware heading grounding, and 3D cue memory.
- Offline episode runner and evaluation metrics for SR, SPL, Fail@CF, and CFSR.
- Pseudolabel generation utilities for distilling API outputs into local training data.
- ROS Noetic bridge skeleton for later robot integration.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## DeepSeek API Smoke Test

Set your API key and model name:

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Run one image through the VLM frontend:

```bash
python scripts/smoke_deepseek.py \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

First verify the API key/model/base URL with a text-only request:

```bash
python scripts/smoke_deepseek.py \
  --text-only \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

If `--text-only` works but the image request returns HTTP 400, the DeepSeek endpoint is rejecting image input. In that case, keep DeepSeek for text planning and replace `DeepSeekVLMClient` with a vision-capable model adapter that returns the same `VLMObservation` interface.

For the confirmed text-only DeepSeek path, run:

```bash
python scripts/plan_instruction.py \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

To import your own vision model, see `docs/vision_adapter.md` and run:

```bash
python scripts/smoke_vision_adapter.py \
  --python-adapter /home/mystery/models/my_vln_adapter.py:MyVisionAdapter \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

For a Qwen/DashScope vision API, set `DASHSCOPE_API_KEY` and run:

```bash
python scripts/smoke_qwen_vision.py \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

## Offline Episode Replay

Create a JSONL manifest where each line is a step:

```json
{"episode_id":"demo","image_path":"data/samples/0001.jpg","instruction":"Find the pavilion.","pose":{"x":0,"y":0,"z":0,"yaw":0},"camera":{"width":640,"height":480,"fx":550,"fy":550,"cx":320,"cy":240},"goal_position":{"x":30,"y":4,"z":0},"cue_available_gt":true}
```

Run:

```bash
python scripts/run_offline_episode.py --manifest data/samples/demo.jsonl --output outputs/demo_run.jsonl --mock
```

Remove `--mock` to use the DeepSeek API.

## Pseudolabel Generation

```bash
python scripts/generate_pseudolabels.py \
  --manifest data/samples/demo.jsonl \
  --output outputs/pseudolabels.jsonl
```

These labels are intended for lightweight distillation tasks such as tile scoring, visible gating, and traversability classification. DeepSeek itself is not trained by this project.

## ROS Bridge

The bridge is intentionally import-safe on machines without ROS. On Ubuntu 20.04 with ROS Noetic:

```bash
rosrun taric_vln ros_bridge.py
```

The bridge subscribes to camera and odometry topics and publishes executable heading plus debug signals. The low-level controller or local planner should still enforce robot safety.
