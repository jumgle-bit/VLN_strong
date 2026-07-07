from __future__ import annotations

from taric_vln.eval import compute_episode_metrics, summarize_metrics
from taric_vln.types import Point3D, Pose3D


def test_episode_metrics_success_and_summary() -> None:
    poses = [Pose3D(0.0, 0.0), Pose3D(2.0, 0.0), Pose3D(4.0, 0.0)]
    metrics = compute_episode_metrics(
        "e1",
        poses,
        Point3D(5.0, 0.0),
        cue_available=[True, False, False],
        success_radius_m=2.0,
    )
    summary = summarize_metrics([metrics])

    assert metrics.success is True
    assert summary["sr"] == 1.0
    assert summary["episodes"] == 1.0
