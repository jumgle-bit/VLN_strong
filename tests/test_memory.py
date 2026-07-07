from __future__ import annotations

import random

from taric_vln.config import TaricConfig
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.memory import CueMemory3D
from taric_vln.types import CameraIntrinsics, Pose3D


def test_memory_initializes_from_two_visible_focus_points_and_reads_out() -> None:
    config = TaricConfig(particle_count=50, min_triangulation_baseline_m=0.5)
    memory = CueMemory3D(config, rng=random.Random(1))
    camera = CameraIntrinsics()

    assert memory.update_visible(Pose3D(0.0, 0.0, 0.0, 0.0), (320.0, 240.0), camera) is False
    assert memory.update_visible(Pose3D(1.0, 0.0, 0.0, 0.0), (300.0, 240.0), camera) is True
    assert memory.ready is True

    readout = memory.readout(
        Pose3D(2.0, 0.0, 0.0, 0.0),
        [1.0] * config.heading_sectors,
        TraversabilityGrounder(config),
    )

    assert readout.memory_ready is True
    assert readout.executable_bearing_rad is not None
    assert readout.goal_mean is not None
