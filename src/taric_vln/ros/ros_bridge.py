from __future__ import annotations

import math
import os
import tempfile

from taric_vln.config import TaricConfig
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.memory import CueMemory3D
from taric_vln.perception import CueExtractor, DeepSeekVLMClient
from taric_vln.types import CameraIntrinsics, Pose3D


class RosBridge:
    def __init__(self) -> None:
        import rospy
        from std_msgs.msg import Float32, String

        self.rospy = rospy
        self.Float32 = Float32
        self.String = String
        self.config = TaricConfig()
        self.grounder = TraversabilityGrounder(self.config)
        self.memory = CueMemory3D(self.config)
        self.extractor = CueExtractor(
            DeepSeekVLMClient(),
            config=self.config,
            grounder=self.grounder,
        )
        self.latest_pose = Pose3D(0.0, 0.0, 0.0, 0.0)
        self.latest_camera = CameraIntrinsics()
        self.instruction = os.getenv("TARIC_INSTRUCTION", "")
        self.heading_pub = rospy.Publisher("/taric/executable_heading", Float32, queue_size=1)
        self.memory_pub = rospy.Publisher("/taric/debug/memory", String, queue_size=1)
        self.trav_pub = rospy.Publisher("/taric/debug/traversability", String, queue_size=1)

    def start(self) -> None:
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image

        rospy.init_node("taric_vln_bridge")
        rospy.Subscriber("/camera/camera_info", CameraInfo, self.on_camera_info, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.on_image, queue_size=1)
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

        try:
            image_path = self._save_ros_image(msg)
            cue = self.extractor.extract(
                image_path=image_path,
                instruction=self.instruction,
                camera=self.latest_camera,
                previous_state={"memory": self.memory.snapshot()},
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
        except Exception as exc:
            self.rospy.logwarn("TARIC bridge failed: %s", exc)

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
