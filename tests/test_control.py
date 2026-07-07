from __future__ import annotations

from taric_vln.control import HeadingController


def test_heading_controller_slows_down_for_large_turns() -> None:
    controller = HeadingController(max_speed_mps=1.0, max_angular_z_rad_s=0.5)

    straight = controller.command(0.0)
    turning = controller.command(1.0)

    assert straight.speed_mps > turning.speed_mps
    assert turning.angular_z_rad_s == 0.5
