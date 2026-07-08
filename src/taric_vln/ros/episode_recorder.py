from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from taric_vln.types import CameraIntrinsics, Point3D, Pose3D


@dataclass(frozen=True)
class EpisodeRecorderSettings:
    image_topic: str = "/camera/rgb/image_raw"
    camera_info_topic: str = "/camera/rgb/camera_info"
    odom_topic: str = "/odom"
    output_root: str = "data/episodes/gazebo_small_eval"
    episode_id: str = ""
    instruction: str = ""
    record_interval_s: float = 1.0
    max_frames: int = 50
    cue_available_gt: bool = True
    goal_x: float | None = None
    goal_y: float | None = None
    goal_z: float = 0.0
    goal_offset_x: float = 3.0
    goal_offset_y: float = 0.0
    shortest_path_m: float | None = None
    image_ext: str = "jpg"

    @classmethod
    def from_env(cls) -> "EpisodeRecorderSettings":
        return cls(
            image_topic=os.getenv("TARIC_IMAGE_TOPIC", cls.image_topic),
            camera_info_topic=os.getenv("TARIC_CAMERA_INFO_TOPIC", cls.camera_info_topic),
            odom_topic=os.getenv("TARIC_ODOM_TOPIC", cls.odom_topic),
            output_root=os.getenv("TARIC_RECORD_ROOT", cls.output_root),
            episode_id=os.getenv("TARIC_EPISODE_ID", default_episode_id()),
            instruction=os.getenv("TARIC_INSTRUCTION", cls.instruction),
            record_interval_s=float(
                os.getenv("TARIC_RECORD_INTERVAL_S", cls.record_interval_s)
            ),
            max_frames=int(os.getenv("TARIC_RECORD_MAX_FRAMES", cls.max_frames)),
            cue_available_gt=parse_bool(
                os.getenv("TARIC_CUE_AVAILABLE_GT"),
                default=cls.cue_available_gt,
            ),
            goal_x=parse_optional_float(os.getenv("TARIC_GOAL_X")),
            goal_y=parse_optional_float(os.getenv("TARIC_GOAL_Y")),
            goal_z=float(os.getenv("TARIC_GOAL_Z", cls.goal_z)),
            goal_offset_x=float(os.getenv("TARIC_GOAL_OFFSET_X", cls.goal_offset_x)),
            goal_offset_y=float(os.getenv("TARIC_GOAL_OFFSET_Y", cls.goal_offset_y)),
            shortest_path_m=parse_optional_float(os.getenv("TARIC_SHORTEST_PATH_M")),
            image_ext=os.getenv("TARIC_RECORD_IMAGE_EXT", cls.image_ext).lstrip("."),
        )


class EpisodeRecorder:
    def __init__(self) -> None:
        import rospy

        self.rospy = rospy
        self.settings = EpisodeRecorderSettings.from_env()
        self.output_root = Path(self.settings.output_root)
        self.image_dir = self.output_root / "images"
        self.manifest_path = self.output_root / "manifest.jsonl"
        self.latest_camera = CameraIntrinsics()
        self.latest_pose = Pose3D(0.0, 0.0, 0.0, 0.0)
        self.initial_pose: Pose3D | None = None
        self.goal_position: Point3D | None = None
        self.shortest_path_m: float | None = None
        self.last_recorded_at = 0.0
        self.step_index = 0
        self._manifest_file = None

    def start(self) -> None:
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image

        rospy.init_node("taric_episode_recorder")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._write_readme()
        self._manifest_file = self.manifest_path.open("a", encoding="utf-8")

        rospy.loginfo("TARIC recorder episode_id=%s", self.settings.episode_id)
        rospy.loginfo("TARIC recorder output root=%s", self.output_root)
        rospy.loginfo("TARIC recorder image topic=%s", self.settings.image_topic)
        rospy.loginfo("TARIC recorder camera info topic=%s", self.settings.camera_info_topic)
        rospy.loginfo("TARIC recorder odom topic=%s", self.settings.odom_topic)
        if not self.settings.instruction:
            rospy.logwarn("TARIC_INSTRUCTION is empty; manifest rows will have an empty instruction.")

        rospy.Subscriber(
            self.settings.camera_info_topic,
            CameraInfo,
            self.on_camera_info,
            queue_size=1,
        )
        rospy.Subscriber(self.settings.odom_topic, Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber(self.settings.image_topic, Image, self.on_image, queue_size=1)
        rospy.on_shutdown(self.close)
        rospy.spin()

    def on_camera_info(self, msg) -> None:
        self.latest_camera = CameraIntrinsics(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(msg.K[0]),
            fy=float(msg.K[4]),
            cx=float(msg.K[2]),
            cy=float(msg.K[5]),
        )

    def on_odom(self, msg) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        p = msg.pose.pose.position
        self.latest_pose = Pose3D(float(p.x), float(p.y), float(p.z), yaw)
        if self.initial_pose is None:
            self.initial_pose = self.latest_pose
            self.goal_position = self._resolve_goal_position(self.initial_pose)
            self.shortest_path_m = self._resolve_shortest_path(self.initial_pose, self.goal_position)

    def on_image(self, msg) -> None:
        if self._manifest_file is None:
            return
        now = time.time()
        if now - self.last_recorded_at < self.settings.record_interval_s:
            return
        self.last_recorded_at = now

        image_path = self._save_ros_image(msg)
        goal = self.goal_position or self._resolve_goal_position(self.latest_pose)
        shortest_path = self.shortest_path_m or self._resolve_shortest_path(self.latest_pose, goal)
        row = {
            "episode_id": self.settings.episode_id,
            "step_id": self.step_index,
            "image_path": str(image_path).replace("\\", "/"),
            "instruction": self.settings.instruction,
            "pose": self.latest_pose.to_dict(),
            "camera": self.latest_camera.to_dict(),
            "goal_position": goal.to_dict(),
            "shortest_path_m": round(shortest_path, 3),
            "cue_available_gt": self.settings.cue_available_gt,
            "ros_stamp": msg.header.stamp.to_sec() if msg.header.stamp else None,
        }
        self._manifest_file.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        self._manifest_file.flush()
        self.rospy.loginfo(
            "Recorded %s step %d/%d",
            self.settings.episode_id,
            self.step_index + 1,
            self.settings.max_frames,
        )
        self.step_index += 1
        if self.step_index >= self.settings.max_frames:
            self.rospy.loginfo("Reached TARIC_RECORD_MAX_FRAMES=%d", self.settings.max_frames)
            self.rospy.signal_shutdown("episode recording complete")

    def close(self) -> None:
        if self._manifest_file is not None:
            self._manifest_file.close()
            self._manifest_file = None

    def _save_ros_image(self, msg) -> Path:
        from cv_bridge import CvBridge
        import cv2

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_name = (
            f"{self.settings.episode_id}_step_{self.step_index:04d}.{self.settings.image_ext}"
        )
        path = self.image_dir / image_name
        cv2.imwrite(str(path), image)
        return path

    def _resolve_goal_position(self, pose: Pose3D) -> Point3D:
        if self.settings.goal_x is not None and self.settings.goal_y is not None:
            return Point3D(self.settings.goal_x, self.settings.goal_y, self.settings.goal_z)
        return Point3D(
            pose.x + self.settings.goal_offset_x,
            pose.y + self.settings.goal_offset_y,
            self.settings.goal_z,
        )

    def _resolve_shortest_path(self, pose: Pose3D, goal: Point3D) -> float:
        if self.settings.shortest_path_m is not None:
            return self.settings.shortest_path_m
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        dz = goal.z - pose.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _write_readme(self) -> None:
        readme_path = self.output_root / "README.md"
        if readme_path.exists():
            return
        readme_path.write_text(
            f"""# Gazebo Small-Eval VLN Dataset

This dataset is recorded from TurtleBot3 Gazebo topics using:

```bash
/usr/bin/python3 -m taric_vln.ros.episode_recorder
```

Manifest:

```text
{self.manifest_path.as_posix()}
```

Images:

```text
{self.image_dir.as_posix()}
```

This is a simulated dataset for offline TARIC/VLN evaluation. It is not a paper-scale benchmark.
""",
            encoding="utf-8",
        )


def default_episode_id() -> str:
    return "gazebo_" + time.strftime("%Y%m%d_%H%M%S")


def parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    EpisodeRecorder().start()


if __name__ == "__main__":
    main()
