# Gazebo Small-Eval VLN Dataset

This dataset is recorded from TurtleBot3 Gazebo topics using:

```bash
/usr/bin/python3 -m taric_vln.ros.episode_recorder
```

Episodes `gazebo_002` through `gazebo_010` were recorded with the controlled batch workflow:

```bash
bash scripts/record_gazebo_small_eval.sh
```

The batch workflow resets the TurtleBot3 pose near the open center of the Gazebo world before each episode, then drives a slow short arc to avoid the wall-collision behavior seen with a constant mock heading.

Manifest:

```text
data/episodes/gazebo_small_eval/manifest.jsonl
```

Images:

```text
data/episodes/gazebo_small_eval/images
```

This is a simulated dataset for offline TARIC/VLN evaluation. It is not a paper-scale benchmark.
