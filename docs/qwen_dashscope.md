# Qwen Vision API

This project can call a Qwen vision API through the OpenAI-compatible DashScope/Bailian endpoint.

## Environment

Do not put the API key in code or commit it to Git.

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export QWEN_VISION_MODEL="qwen-vl-plus"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

You can also use `qwen-vl-max` if your account has access.

## Smoke Test

```bash
python scripts/smoke_qwen_vision.py \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

Expected output is the normalized `VLMObservation` JSON:

```json
{
  "visible": true,
  "focus_pixel": [320.0, 240.0],
  "tile_scores": [0.1, 1.0, 0.2],
  "cue_bearing_rad": 0.0,
  "traversability_scores": [0.7, 0.8, 0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0],
  "confidence": 0.8
}
```

## Offline Episode

```bash
python scripts/run_offline_episode.py \
  --manifest data/samples/demo.jsonl \
  --output outputs/demo_run.jsonl \
  --qwen
```

Override model or endpoint:

```bash
python scripts/run_offline_episode.py \
  --manifest data/samples/demo.jsonl \
  --output outputs/demo_run.jsonl \
  --qwen \
  --qwen-model qwen-vl-max \
  --qwen-base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```
