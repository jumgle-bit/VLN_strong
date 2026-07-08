from __future__ import annotations

from dataclasses import dataclass
import math
import os
import tempfile
import time

from taric_vln.config import TaricConfig
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.memory import CueMemory3D
from taric_vln.perception import CueExtractor, DeepSeekVLMClient, MockVLMClient, QwenVisionClient
from taric_vln.types import CameraIntrinsics, Pose3D


@dataclass(frozen=True)
class RosBridgeSettings:
    image_topic: str = "/camera/rgb/image_raw"
    camera_info_topic: str = "/camera/rgb/camera_info"
    odom_topic: str = "/odom"
    heading_topic: str = "/taric/executable_heading"
    memory_topic: str = "/taric/debug/memory"
    traversability_topic: str = "/taric/debug/traversability"
    status_topic: str = "/taric/debug/status"
    vision_backend: str = "qwen"
    instruction: str = ""
    frame_interval_s: float = 5.0
    qwen_timeout_s: float = 90.0
    qwen_retries: int = 0
    qwen_max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "RosBridgeSettings":
        return cls(
            image_topic=os.getenv("TARIC_IMAGE_TOPIC", cls.image_topic),
            camera_info_topic=os.getenv("TARIC_CAMERA_INFO_TOPIC", cls.camera_info_topic),
            odom_topic=os.getenv("TARIC_ODOM_TOPIC", cls.odom_topic),
            heading_topic=os.getenv("TARIC_HEADING_TOPIC", cls.heading_topic),
            memory_topic=os.getenv("TARIC_MEMORY_TOPIC", cls.memory_topic),
            traversability_topic=os.getenv("TARIC_TRAVERSABILITY_TOPIC", cls.traversability_topic),
            status_topic=os.getenv("TARIC_STATUS_TOPIC", cls.status_topic),
            vision_backend=os.getenv("TARIC_VISION_BACKEND", cls.vision_backend).strip().lower(),
            instruction=os.getenv("TARIC_INSTRUCTION", cls.instruction),
            frame_interval_s=float(os.getenv("TARIC_FRAME_INTERVAL_S", cls.frame_interval_s)),
            qwen_timeout_s=float(os.getenv("TARIC_QWEN_TIMEOUT_S", cls.qwen_timeout_s)),
            qwen_retries=int(os.getenv("TARIC_QWEN_RETRIES", cls.qwen_retries)),
            qwen_max_tokens=int(os.getenv("TARIC_QWEN_MAX_TOKENS", cls.qwen_max_tokens)),
        )


def build_vlm_client(settings: RosBridgeSettings):
    if settings.vision_backend == "qwen":
        return QwenVisionClient(
            timeout_s=settings.qwen_timeout_s,
            retries=settings.qwen_retries,
            max_tokens=settings.qwen_max_tokens,
            fallback_on_error=True,
        )
    if settings.vision_backend == "deepseek":
        return DeepSeekVLMClient()
    if settings.vision_backend == "mock":
        return MockVLMClient()
    raise ValueError(
        "TARIC_VISION_BACKEND must be one of: qwen, deepseek, mock "
        f"(got {settings.vision_backend!r})"
    )


class RosBridge:
    def __init__(self) -> None:
        import rospy
        from std_msgs.msg import Float32, String

        self.rospy = rospy
        self.Float32 = Float32
        self.String = String
        self.settings = RosBridgeSettings.from_env()
        self.config = TaricConfig()
        self.grounder = TraversabilityGrounder(self.config)
        self.memory = CueMemory3D(self.config)
        self.extractor = CueExtractor(
            build_vlm_client(self.settings),
            config=self.config,
            grounder=self.grounder,
        )
        self.latest_pose = Pose3D(0.0, 0.0, 0.0, 0.0)
        self.latest_camera = CameraIntrinsics()
        self._last_processed_at = 0.0
        self._processing = False
        self.heading_pub = rospy.Publisher(self.settings.heading_topic, Float32, queue_size=1)
        self.memory_pub = rospy.Publisher(self.settings.memory_topic, String, queue_size=1)
        self.trav_pub = rospy.Publisher(self.settings.traversability_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.settings.status_topic, String, queue_size=1)

    def start(self) -> None:
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image

        rospy.init_node("taric_vln_bridge")
        rospy.loginfo("TARIC bridge backend=%s", self.settings.vision_backend)
        rospy.loginfo("TARIC image topic=%s", self.settings.image_topic)
        rospy.loginfo("TARIC camera info topic=%s", self.settings.camera_info_topic)
        rospy.loginfo("TARIC odom topic=%s", self.settings.odom_topic)
        if not self.settings.instruction:
            rospy.logwarn("TARIC_INSTRUCTION is empty; navigation perception will be weak.")
        rospy.Subscriber(
            self.settings.camera_info_topic,
            CameraInfo,
            self.on_camera_info,
            queue_size=1,
        )
        rospy.Subscriber(self.settings.odom_topic, Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber(self.settings.image_topic, Image, self.on_image, queue_size=1)
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

    def on_image(self, msg) -> None:
        import json

        now = time.time()
        if self._processing:
            return
        if now - self._last_processed_at < self.settings.frame_interval_s:
            return
        self._processing = True
        self._last_processed_at = now
        try:
            image_path = self._save_ros_image(msg)
            cue = self.extractor.extract(
                image_path=image_path,
                instruction=self.settings.instruction,
                camera=self.latest_camera,
                previous_state={
                    "memory": self.memory.snapshot(),
                    "pose": self.latest_pose.to_dict(),
                    "stamp": msg.header.stamp.to_sec() if msg.header.stamp else None,
                },
            )
            command = cue.executable_bearing_rad
            if cue.visible_after_gate and cue.observation.focus_pixel is not None:
                self.memory.update_visible(
                    self.latest_pose, cue.observation.focus_pixel, self.latest_camera
                )
            elif self.memory.ready:
                readout = self.memory.readout(
                    self.latest_pose,
                    cue.observation.traversability_scores,
                    self.grounder,
                )
                if readout.executable_bearing_rad is not None:
                    command = readout.executable_bearing_rad
            self.heading_pub.publish(self.Float32(command))
            self.memory_pub.publish(self.String(json.dumps(self.memory.snapshot())))
            self.trav_pub.publish(self.String(json.dumps(cue.observation.traversability_scores)))
            self.status_pub.publish(
                self.String(
                    json.dumps(
                        {
                            "visible": cue.visible_after_gate,
                            "heading_rad": command,
                            "selected_sector": cue.selected_sector,
                            "traversability_status": cue.traversability_status,
                            "vision_error": cue.observation.error,
                        }
                    )
                )
            )
        except Exception as exc:
            self.rospy.logwarn("TARIC bridge failed: %s", exc)
        finally:
            self._processing = False

    def _save_ros_image(self, msg) -> str:
        from cv_bridge import CvBridge
        import cv2

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = os.path.join(tempfile.gettempdir(), "taric_latest.jpg")
        cv2.imwrite(path, image)
        return path


def main() -> None:
    RosBridge().start()


if __name__ == "__main__":
    main()
