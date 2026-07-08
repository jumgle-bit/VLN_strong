# Ubuntu 20.04 Setup

This project is designed so the core pipeline can run without a local GPU. Use a GPU cloud server only for heavy simulation, local model distillation, or large batch experiments.

## Base Packages

```bash
sudo apt update
sudo apt install -y git git-lfs python3.10 python3.10-venv python3-pip tmux ffmpeg libgl1 libglib2.0-0
```

If Ubuntu 20.04 does not provide Python 3.10 through your package sources, install Miniconda or Mambaforge and create a Python 3.10 environment.

## Install Project

```bash
git clone git@github.com:jumgle-bit/VLN_strong.git
cd VLN_strong
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## DeepSeek API

Do not commit your API key. Put it in your shell environment:

```bash
export DEEPSEEK_API_KEY="your_key_here"
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Then run:

```bash
python scripts/smoke_deepseek.py --image data/samples/example.jpg --instruction "Find the red library entrance."
```

## Optional ROS Noetic

Install ROS Noetic only on the machine that will connect to a robot or ROS bag:

```bash
sudo apt install -y ros-noetic-desktop-full ros-noetic-cv-bridge ros-noetic-nav-msgs ros-noetic-sensor-msgs python3-numpy python3-opencv
```

The ROS bridge publishes heading/debug topics and leaves final safety to the robot's local planner or native controller.
