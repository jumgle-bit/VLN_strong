from __future__ import annotations

import argparse
import math
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset TurtleBot3 in Gazebo and drive a short controlled arc."
    )
    parser.add_argument("--model", default="turtlebot3_waffle_pi")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.02)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--duration", type=float, default=36.0)
    parser.add_argument("--speed", type=float, default=0.045)
    parser.add_argument("--angular", type=float, default=0.10)
    parser.add_argument("--curve-period", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    import rospy
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState
    from geometry_msgs.msg import Twist

    args = parse_args()
    rospy.init_node("taric_episode_driver", anonymous=True)
    pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=1)

    rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    state = ModelState()
    state.model_name = args.model
    state.reference_frame = "world"
    state.pose.position.x = args.x
    state.pose.position.y = args.y
    state.pose.position.z = args.z
    state.pose.orientation.z = math.sin(args.yaw / 2.0)
    state.pose.orientation.w = math.cos(args.yaw / 2.0)
    set_state(state)

    publish_stop(pub)
    started_at = time.time()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and time.time() - started_at < args.duration:
        elapsed = time.time() - started_at
        twist = Twist()
        twist.linear.x = args.speed
        sign = 1.0 if int(elapsed // args.curve_period) % 2 == 0 else -1.0
        twist.angular.z = sign * args.angular
        pub.publish(twist)
        rate.sleep()

    publish_stop(pub)


def publish_stop(pub) -> None:
    from geometry_msgs.msg import Twist

    zero = Twist()
    for _ in range(10):
        pub.publish(zero)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
