# ROS Noetic + TurtleBot3 Gazebo Integration

This guide assumes Ubuntu 20.04, ROS Noetic, Gazebo, RViz, and TurtleBot3 are already installed.

The confirmed TurtleBot3 `waffle_pi` topics are:

```text
/camera/rgb/image_raw
/camera/rgb/camera_info
/odom
/cmd_vel
```

## Important Python Rule

Do not start Gazebo or ROS nodes from the project `.venv`. ROS Noetic uses system Python packages such as `rospy`, `rospkg`, and `cv_bridge`.

Use:

```bash
source /opt/ros/noetic/setup.bash
```

For TARIC Python modules, expose the repo through `PYTHONPATH`:

```bash
cd ~/VLN_strong
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Required ROS Packages

If imports fail, install:

```bash
sudo apt update
sudo apt install -y \
  python3-rospkg \
  python3-catkin-pkg \
  python3-empy \
  python3-numpy \
  python3-yaml \
  python3-opencv \
  ros-noetic-cv-bridge \
  ros-noetic-xacro
```

Verify the system Python imports before starting the bridge:

```bash
python3 -c "import numpy, cv2, cv_bridge, rospkg; print('ROS Python deps OK')"
```

If this fails while your shell prompt shows `(.venv)`, run `deactivate` first and retry. ROS bridge nodes should use system Python, not the project virtual environment.

## Terminal 1: Start Gazebo

```bash
source /opt/ros/noetic/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

Check topics in another terminal:

```bash
source /opt/ros/noetic/setup.bash
rostopic list | grep -E "camera|odom|cmd_vel"
```

## Terminal 2: Smoke Test TARIC Bridge Without API

Start with the mock backend. This checks image subscription, camera info, odometry, and TARIC heading publication without spending API calls.

```bash
cd ~/VLN_strong
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

export TARIC_VISION_BACKEND=mock
export TARIC_INSTRUCTION="Find the red library entrance. Follow the path and avoid grass."
export TARIC_FRAME_INTERVAL_S=2

python3 -m taric_vln.ros.ros_bridge
```

In another terminal:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /taric/executable_heading
rostopic echo /taric/debug/status
```

If these topics publish values, the ROS bridge is wired correctly.

## Terminal 3: Convert TARIC Heading to TurtleBot3 cmd_vel

Start conservatively:

```bash
cd ~/VLN_strong
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

export TARIC_MAX_SPEED_MPS=0.12
export TARIC_MAX_ANGULAR_Z_RAD_S=0.6
export TARIC_COMMAND_TIMEOUT_S=2

python3 -m taric_vln.ros.turtlebot3_commander
```

Inspect velocity commands:

```bash
rostopic echo /cmd_vel
```

The commander publishes zero velocity if no fresh TARIC heading is received within `TARIC_COMMAND_TIMEOUT_S`.

## Terminal 2: Switch Bridge to Qwen

After the mock bridge works, stop it and restart with Qwen:

```bash
cd ~/VLN_strong
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

export DASHSCOPE_API_KEY="<your_dashscope_key>"
export QWEN_VISION_MODEL="qwen-vl-plus"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

export TARIC_VISION_BACKEND=qwen
export TARIC_INSTRUCTION="Find the target building. Stay on the paved path and avoid grass."
export TARIC_FRAME_INTERVAL_S=5
export TARIC_QWEN_TIMEOUT_S=90
export TARIC_QWEN_RETRIES=0
export TARIC_QWEN_MAX_TOKENS=1024

python3 -m taric_vln.ros.ros_bridge
```

Keep `TARIC_FRAME_INTERVAL_S` at 5 seconds or higher while debugging. Qwen is a cloud API; do not call it at camera frame rate.

## Environment Variables

Bridge:

```text
TARIC_VISION_BACKEND=qwen|mock|deepseek
TARIC_INSTRUCTION=...
TARIC_IMAGE_TOPIC=/camera/rgb/image_raw
TARIC_CAMERA_INFO_TOPIC=/camera/rgb/camera_info
TARIC_ODOM_TOPIC=/odom
TARIC_HEADING_TOPIC=/taric/executable_heading
TARIC_FRAME_INTERVAL_S=5
TARIC_QWEN_TIMEOUT_S=90
TARIC_QWEN_RETRIES=0
TARIC_QWEN_MAX_TOKENS=1024
```

Commander:

```text
TARIC_CMD_VEL_TOPIC=/cmd_vel
TARIC_MAX_SPEED_MPS=0.12
TARIC_MAX_ANGULAR_Z_RAD_S=0.6
TARIC_YAW_GAIN=1.0
TARIC_CONTROL_HZ=10
TARIC_COMMAND_TIMEOUT_S=2
```

## Current Limitations

- The ROS bridge sends a single RGB frame to the vision backend at a throttled interval.
- The command node is a conservative heading-to-velocity adapter, not a full obstacle-avoiding local planner.
- Qwen is still the vision frontend; DeepSeek remains text-only for this project.
- For paper-scale evaluation, record Gazebo episodes into manifests and evaluate offline before running long closed-loop tests.
