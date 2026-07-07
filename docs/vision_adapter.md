# Vision Model Adapter

DeepSeek is confirmed text-only in this project. To use your own vision model, wrap it so it returns the common `VLMObservation` JSON fields.

## Required Output

Your model adapter should return a Python dict:

```json
{
  "exploration_phrase": "follow the path",
  "goal_phrase": "red library entrance",
  "visible": true,
  "focus_pixel": [320, 240],
  "tile_scores": [0.1, 1.0, 0.2],
  "cue_bearing_deg": 0.0,
  "traversability_scores": [0.7, 0.8, 0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0],
  "confidence": 0.8
}
```

If `focus_pixel` is present, the project recomputes bearing from camera intrinsics. `traversability_scores` are left-to-right heading-sector scores from 0 to 1.

## Python Adapter

Create a file outside the repo, for example `~/models/my_vln_adapter.py`:

```python
class MyVisionAdapter:
    def __init__(self):
        # Load your model here.
        pass

    def analyze(self, image_path, instruction, camera, previous_state, config):
        # Run your model here and return the required dict.
        return {
            "exploration_phrase": instruction,
            "goal_phrase": instruction,
            "visible": False,
            "focus_pixel": None,
            "tile_scores": [],
            "cue_bearing_deg": 0.0,
            "traversability_scores": [1.0] * int(config["heading_sectors"]),
            "confidence": 0.5,
        }
```

Smoke test:

```bash
python scripts/smoke_vision_adapter.py \
  --python-adapter /home/mystery/models/my_vln_adapter.py:MyVisionAdapter \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

Offline replay:

```bash
python scripts/run_offline_episode.py \
  --manifest data/samples/demo.jsonl \
  --output outputs/demo_run.jsonl \
  --python-adapter /home/mystery/models/my_vln_adapter.py:MyVisionAdapter
```

## Command Adapter

If your model already has a command-line runner, make it read JSON from stdin and print the required JSON to stdout.

Input JSON contains:

- `image_path`
- `instruction`
- `camera`
- `previous_state`
- `config`

Smoke test:

```bash
python scripts/smoke_vision_adapter.py \
  --command-adapter "python /home/mystery/models/run_model.py" \
  --image data/samples/example.jpg \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```
