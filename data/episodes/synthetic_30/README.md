# Synthetic 30-Episode VLN Dataset

This is a small synthetic dataset for engineering validation of the TARIC/VLN pipeline.

- Episodes: 30
- Steps per episode: 8
- Total manifest rows: 240
- Image size: 640x360
- Generator seed: 31121

The scenes contain simplified outdoor paths, grass, buildings, target signs, and cue-interruption steps. This dataset is for smoke testing and pipeline validation only; it is not a paper-quality benchmark.

Run:

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/synthetic_30/manifest.jsonl \
  --output outputs/synthetic_30_run.jsonl \
  --qwen

python scripts/summarize_run.py --input outputs/synthetic_30_run.jsonl
```
