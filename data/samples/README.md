# Sample Data

Put local smoke-test images and JSONL manifests here.

Example manifest row:

```json
{"episode_id":"demo","image_path":"data/samples/0001.jpg","instruction":"Find the pavilion.","pose":{"x":0,"y":0,"z":0,"yaw":0},"camera":{"width":640,"height":480,"fx":550,"fy":550,"cx":320,"cy":240},"goal_position":{"x":30,"y":4,"z":0},"cue_available_gt":true}
```

Images and large generated datasets should not be committed unless they are small curated examples.
