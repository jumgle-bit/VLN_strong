# Reproduction Workflow

## Stage 1: Smoke Tests

1. Run unit tests locally.
2. Run `scripts/smoke_deepseek.py` on one image.
3. Confirm image input works and JSON is parseable.

If DeepSeek rejects image input with `unknown variant image_url, expected text`, use it only for text planning:

```bash
python scripts/plan_instruction.py --instruction "Find the red library entrance. Follow the path and avoid grass."
```

Then connect a separate vision-capable adapter to the `VLMObservation` interface.

## Stage 2: Offline Replay

Create JSONL manifests from image sequences or ROS bags. Each row should include image path, instruction, pose, camera intrinsics, and optional ground-truth goal/cue fields.

Run:

```bash
python scripts/run_offline_episode.py --manifest data/samples/demo.jsonl --output outputs/demo_run.jsonl --mock
```

Use `--mock` first, then remove it for real DeepSeek calls.

## Stage 3: Pseudolabels

Use DeepSeek to label images for:

- visible gate
- target tile scorer
- traversability sector classifier

```bash
python scripts/generate_pseudolabels.py --manifest data/samples/demo.jsonl --output outputs/pseudolabels.jsonl
```

DeepSeek is not trained. The labels are for lightweight local models or manual analysis.

## Stage 4: Simulation Evaluation

Start with 30 short episodes, then scale to 100 episodes per scene and 3 seeds.

Report:

- SR
- SPL
- Fail@CF
- CFSR@10/25/50

Compare at least:

- frame-only DeepSeek
- 2D memory
- full TARIC with traversability grounding and 3D memory

## Stage 5: ROS Integration

Use `taric_vln.ros.ros_bridge` as the integration skeleton. It subscribes to camera, camera info, and odometry, then publishes executable heading and debug messages.
