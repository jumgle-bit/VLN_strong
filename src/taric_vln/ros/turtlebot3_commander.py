from __future__ import annotations

from dataclasses import dataclass
import os
import time

from taric_vln.control import HeadingController


@dataclass(frozen=True)
class TurtleBot3CommanderSettings:
    heading_topic: str = "/taric/executable_heading"
    cmd_vel_topic: str = "/cmd_vel"
    control_hz: float = 10.0
    command_timeout_s: float = 2.0
    max_speed_mps: float = 0.18
    max_angular_z_rad_s: float = 0.8
    yaw_gain: float = 1.0

    @classmethod
    def from_env(cls) -> "TurtleBot3CommanderSettings":
        return cls(
            heading_topic=os.getenv("TARIC_HEADING_TOPIC", cls.heading_topic),
            cmd_vel_topic=os.getenv("TARIC_CMD_VEL_TOPIC", cls.cmd_vel_topic),
            control_hz=float(os.getenv("TARIC_CONTROL_HZ", cls.control_hz)),
            command_timeout_s=float(
                os.getenv("TARIC_COMMAND_TIMEOUT_S", cls.command_timeout_s)
            ),
            max_speed_mps=float(os.getenv("TARIC_MAX_SPEED_MPS", cls.max_speed_mps)),
            max_angular_z_rad_s=float(
                os.getenv("TARIC_MAX_ANGULAR_Z_RAD_S", cls.max_angular_z_rad_s)
            ),
            yaw_gain=float(os.getenv("TARIC_YAW_GAIN", cls.yaw_gain)),
        )


class TurtleBot3Commander:
    def __init__(self) -> None:
        import rospy
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Float32

        self.rospy = rospy
        self.Twist = Twist
        self.Float32 = Float32
        self.settings = TurtleBot3CommanderSettings.from_env()
        self.controller = HeadingController(
            max_speed_mps=self.settings.max_speed_mps,
            max_angular_z_rad_s=self.settings.max_angular_z_rad_s,
            yaw_gain=self.settings.yaw_gain,
        )
        self.latest_heading_rad: float | None = None
        self.latest_heading_at = 0.0
        self.cmd_pub = rospy.Publisher(self.settings.cmd_vel_topic, Twist, queue_size=1)

    def start(self) -> None:
        import rospy

        rospy.init_node("taric_turtlebot3_commander")
        rospy.loginfo("TARIC commander heading topic=%s", self.settings.heading_topic)
        rospy.loginfo("TARIC commander cmd_vel topic=%s", self.settings.cmd_vel_topic)
        rospy.Subscriber(
            self.settings.heading_topic,
            self.Float32,
            self.on_heading,
            queue_size=1,
        )
        timer_period = 1.0 / max(self.settings.control_hz, 1e-6)
        rospy.Timer(rospy.Duration(timer_period), self.on_timer)
        rospy.spin()

    def on_heading(self, msg) -> None:
        self.latest_heading_rad = float(msg.data)
        self.latest_heading_at = time.time()

    def on_timer(self, _event) -> None:
        twist = self.Twist()
        if self.latest_heading_rad is not None:
            age_s = time.time() - self.latest_heading_at
            if age_s <= self.settings.command_timeout_s:
                command = self.controller.command(self.latest_heading_rad, source="ros")
                twist.linear.x = command.speed_mps
                twist.angular.z = command.angular_z_rad_s
        self.cmd_pub.publish(twist)


def main() -> None:
    TurtleBot3Commander().start()


if __name__ == "__main__":
    main()
