# Simulated Small-Eval VLN Dataset

This is a simulated small evaluation dataset for the TARIC/VLN engineering pipeline.

It is intended for the first offline evaluation stage when no real outdoor environment, robot, ROS bag, or simulator export is available yet. The images are generated with simple outdoor navigation scenes: paved paths, grass regions, target buildings, visible target signs, and cue-interruption frames where the target cue is occluded.

- Episodes: 10
- Steps per episode: 12
- Total manifest rows: 120
- Image size: 640x360
- Generator seed: 260531121

This dataset is useful for:

- Verifying manifest parsing.
- Testing offline episode execution.
- Checking cue-free memory behavior.
- Recording SR, SPL, Fail@CF, and CFSR metrics.
- Doing a small Qwen API run before scaling to larger simulated data.

This dataset is not a paper-quality benchmark and should not be reported as a TARIC reproduction result.

Run:

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_mock_run.jsonl \
  --mock

python scripts/summarize_run.py --input outputs/small_eval_mock_run.jsonl
```

After the mock run works, run a small Qwen API subset or the full dataset:

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_qwen_run.jsonl \
  --qwen

python scripts/summarize_run.py --input outputs/small_eval_qwen_run.jsonl
```
