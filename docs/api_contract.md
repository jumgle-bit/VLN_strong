# TARIC DeepSeek API Contract

`DeepSeekVLMClient` expects an OpenAI-compatible chat completions endpoint with image input. The model must return strict JSON.

## Input

- RGB image path
- Navigation instruction
- Camera intrinsics
- Previous state containing step index and memory debug state

## Required Output

```json
{
  "exploration_phrase": "coarse navigation phrase",
  "goal_phrase": "specific target phrase",
  "visible": true,
  "focus_pixel": [320, 240],
  "tile_scores": [0.1, 0.8, 0.2],
  "cue_bearing_deg": 0.0,
  "traversability_scores": [0.7, 0.8, 0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0],
  "confidence": 0.8
}
```

The Python interface normalizes this into `VLMObservation`. If `focus_pixel` is present, bearing is recomputed from camera intrinsics to keep geometry consistent.

## Fallback Behavior

If the API fails, times out, or returns invalid JSON, the default client returns:

- `visible=false`
- forward exploration bearing
- all-zero traversability scores
- an `error` string

The navigation stack can still continue using 3D memory if it has already been initialized.
