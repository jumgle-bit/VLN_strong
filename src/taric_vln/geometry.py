from __future__ import annotations

import math

from taric_vln.config import TaricConfig
from taric_vln.types import CameraIntrinsics, Pose3D, Point3D


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def pixel_to_bearing(pixel: tuple[float, float], camera: CameraIntrinsics) -> float:
    u, _v = pixel
    return math.atan2(camera.cx - float(u), camera.fx)


def bearing_to_pixel(bearing_rad: float, camera: CameraIntrinsics) -> tuple[float, float]:
    u = camera.cx - camera.fx * math.tan(bearing_rad)
    return (u, camera.cy)


def sector_centers(config: TaricConfig) -> list[float]:
    n = config.heading_sectors
    if n <= 1:
        return [0.0]
    span = 2.0 * config.max_bearing_rad
    # Index 0 is leftmost in the image, so it has positive bearing.
    return [config.max_bearing_rad - i * span / (n - 1) for i in range(n)]


def bearing_to_world_direction(pose: Pose3D, bearing_rad: float) -> tuple[float, float]:
    heading = pose.yaw + bearing_rad
    return (math.cos(heading), math.sin(heading))


def world_bearing_from_pose(pose: Pose3D, point: Point3D) -> float:
    dx = point.x - pose.x
    dy = point.y - pose.y
    return wrap_angle(math.atan2(dy, dx) - pose.yaw)


def distance_2d(a: Pose3D | Point3D, b: Pose3D | Point3D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def project_world_point_to_pixel(
    point: Point3D, pose: Pose3D, camera: CameraIntrinsics
) -> tuple[float, float] | None:
    bearing = world_bearing_from_pose(pose, point)
    if abs(bearing) > math.radians(89.0):
        return None
    return bearing_to_pixel(bearing, camera)


def triangulate_rays_2d(
    pose_a: Pose3D, bearing_a: float, pose_b: Pose3D, bearing_b: float
) -> Point3D:
    ax, ay = pose_a.x, pose_a.y
    bx, by = pose_b.x, pose_b.y
    dax, day = bearing_to_world_direction(pose_a, bearing_a)
    dbx, dby = bearing_to_world_direction(pose_b, bearing_b)

    det = dax * (-dby) - day * (-dbx)
    if abs(det) < 1e-6:
        mid_x = (ax + bx) / 2.0
        mid_y = (ay + by) / 2.0
        return Point3D(mid_x + 10.0 * dax, mid_y + 10.0 * day, 0.0)

    rhs_x = bx - ax
    rhs_y = by - ay
    t = (rhs_x * (-dby) - rhs_y * (-dbx)) / det
    u = (dax * rhs_y - day * rhs_x) / det

    pa_x = ax + t * dax
    pa_y = ay + t * day
    pb_x = bx + u * dbx
    pb_y = by + u * dby
    return Point3D((pa_x + pb_x) / 2.0, (pa_y + pb_y) / 2.0, 0.0)
