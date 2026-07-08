#!/usr/bin/env bash
set -eo pipefail

cd "${TARIC_REPO_ROOT:-$HOME/VLN_strong}"
mkdir -p outputs/ros_logs

source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TARIC_RECORD_ROOT="${TARIC_RECORD_ROOT:-data/episodes/gazebo_small_eval}"
export TARIC_INSTRUCTION="${TARIC_INSTRUCTION:-Find the target building. Stay on the paved path and avoid grass.}"
export TARIC_RECORD_INTERVAL_S="${TARIC_RECORD_INTERVAL_S:-1}"
export TARIC_RECORD_MAX_FRAMES="${TARIC_RECORD_MAX_FRAMES:-30}"
export TARIC_GOAL_Z="${TARIC_GOAL_Z:-0.0}"
export TARIC_SHORTEST_PATH_M="${TARIC_SHORTEST_PATH_M:-1.0}"
export TARIC_CUE_AVAILABLE_GT="${TARIC_CUE_AVAILABLE_GT:-true}"

# episode_id x y yaw goal_x goal_y speed angular
EPISODES=(
  "gazebo_002 0.00 0.00 0.00 1.00 0.00 0.045 0.10"
  "gazebo_003 0.00 0.00 1.57 0.00 1.00 0.045 0.10"
  "gazebo_004 0.00 0.00 3.14 -1.00 0.00 0.045 0.10"
  "gazebo_005 0.00 0.00 -1.57 0.00 -1.00 0.045 0.10"
  "gazebo_006 0.25 0.00 0.78 0.95 0.70 0.040 0.09"
  "gazebo_007 -0.25 0.00 2.36 -0.95 0.70 0.040 0.09"
  "gazebo_008 0.00 0.25 -0.78 0.70 -0.45 0.040 0.09"
  "gazebo_009 0.00 -0.25 -2.36 -0.70 -0.95 0.040 0.09"
  "gazebo_010 0.15 0.15 0.35 1.09 0.49 0.035 0.08"
)

stop_robot() {
  rostopic pub -1 /cmd_vel geometry_msgs/Twist '{}' >/tmp/taric_stop_cmd.log 2>&1 || true
}

stop_robot

for spec in "${EPISODES[@]}"; do
  read -r episode_id start_x start_y start_yaw goal_x goal_y speed angular <<< "$spec"
  echo "=== Recording ${episode_id} ==="

  export TARIC_EPISODE_ID="$episode_id"
  export TARIC_GOAL_X="$goal_x"
  export TARIC_GOAL_Y="$goal_y"

  /usr/bin/python3 scripts/drive_turtlebot3_episode.py \
    --x "$start_x" \
    --y "$start_y" \
    --yaw "$start_yaw" \
    --duration 36 \
    --speed "$speed" \
    --angular "$angular" \
    > "outputs/ros_logs/driver_${episode_id}.log" 2>&1 &
  driver_pid=$!

  sleep 2
  /usr/bin/python3 -m taric_vln.ros.episode_recorder \
    > "outputs/ros_logs/recorder_${episode_id}.log" 2>&1

  wait "$driver_pid" || true
  stop_robot
  sleep 2
done

echo "Gazebo batch recording complete."
