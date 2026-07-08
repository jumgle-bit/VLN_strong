# VLN / TARIC 复现项目交接说明

## 1. 项目定位

本项目用于复现论文 `TARIC: Memory-Augmented Traversability-Aware Outdoor VLN under Interrupted Semantic Cues` 的工程流程。当前目标不是完全复刻论文表格数值，而是先掌握 VLN 从视觉语言理解、方向决策、3D 记忆、离线评估到后续 ROS 部署的一整套工程闭环。

仓库地址：

```bash
git@github.com:jumgle-bit/VLN_strong.git
```

当前技术路线：

- DeepSeek：只用于文本规划和指令拆解。已测试当前 DeepSeek 接口不接受图片输入。
- Qwen / DashScope：作为视觉语言 API，用于图片理解和 TARIC 单帧语义 cue 提取。
- TARIC 后端：仓库内自研，包括 cue extraction、traversability grounding、3D cue memory、offline episode runner 和 metrics。
- 训练策略：不训练 DeepSeek 或 Qwen 本体，只做数据集构建、伪标签生成、轻量模块训练或蒸馏。

## 2. 当前已经完成

### 论文理解

已经明确 TARIC 的核心问题：

- 户外长距离 VLN 中，目标语义线索会被遮挡、消失或离开视野。
- 普通 VLN agent 在 cue-free 阶段容易回头、震荡或乱走。
- TARIC 通过三部分缓解问题：
  - 语义 cue 提取。
  - traversability-aware heading grounding。
  - world-aligned 3D cue memory。

### 工程框架

当前仓库已经建立完整 Python 工程：

```text
src/taric_vln/
  perception/      # DeepSeek、Qwen、视觉适配器、cue extractor
  grounding/       # traversability-aware heading grounding
  memory/          # 3D cue memory
  sim/             # offline episode runner
  eval/            # SR / SPL / Fail@CF / CFSR metrics
  control/         # heading -> simple command
  ros/             # ROS Noetic bridge skeleton

scripts/
  smoke_deepseek.py
  smoke_qwen_vision.py
  smoke_vision_adapter.py
  plan_instruction.py
  run_offline_episode.py
  summarize_run.py
  generate_pseudolabels.py
  generate_synthetic_episodes.py

data/
  samples/
  episodes/synthetic_30/

docs/
  progress.md
  qwen_dashscope.md
  vision_adapter.md
  reproduction_workflow.md
  ubuntu20_setup.md
```

### API 接入状态

DeepSeek：

- 文本 API 已可用。
- 图片输入已测试失败，错误含义是接口只接受 text，不接受 image_url。
- 已实现 `DeepSeekTextPlanner`，用于指令规划，不再作为视觉前端。

Qwen / DashScope：

- 图片输入已打通。
- 已实现 `QwenVisionClient`。
- 已处理 Qwen 返回非严格 JSON 或输出被截断时的字段恢复问题。
- 推荐正式视觉 smoke test 使用 `--timeout 90 --max-tokens 1024`。

### 数据和评估

已经生成并提交 30 个 synthetic episode：

```text
data/episodes/synthetic_30/manifest.jsonl
data/episodes/synthetic_30/images/
```

规模：

- 30 episodes。
- 每个 episode 8 step。
- 共 240 张 synthetic VLN 图片。
- 共 240 条 manifest 记录。
- 包含 `cue_available_gt=false` 的中断线索阶段。

这些数据是工程测试数据，不是论文级真实实验数据。它们用于验证代码、指标和闭环是否能跑通。

已跑通 demo episode，并得到过：

```json
{
  "episodes": 1.0,
  "sr": 1.0,
  "spl": 1.0,
  "fail_at_cf": 0.0
}
```

30 episode mock 流程也已跑通，说明离线 runner、memory、grounding 和 metrics 可以闭环。

## 3. 新同学快速上手

### 3.1 克隆仓库

```bash
git clone git@github.com:jumgle-bit/VLN_strong.git
cd VLN_strong
```

如果没有配置 GitHub SSH，可以先用 HTTPS 克隆：

```bash
git clone https://github.com/jumgle-bit/VLN_strong.git
cd VLN_strong
```

### 3.2 Ubuntu 20.04 基础依赖

先确认 Python 版本：

```bash
python3 --version
```

项目要求 Python 3.10 或更高。如果系统没有 Python 3.10，可以安装：

```bash
sudo apt update
sudo apt install -y software-properties-common git curl build-essential
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

创建虚拟环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3.3 配置 API key

不要把 API key 写进代码、Markdown、JSON 或 Git commit。只在本机终端设置环境变量。

Qwen / DashScope：

```bash
export DASHSCOPE_API_KEY="<your_qwen_or_dashscope_key>"
export QWEN_VISION_MODEL="qwen-vl-plus"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

DeepSeek：

```bash
export DEEPSEEK_API_KEY="<your_deepseek_key>"
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

如果想长期保存，可以写入自己机器上的 `~/.bashrc` 或 `~/.zshrc`，但不要提交到仓库。

### 3.4 运行测试

```bash
pytest
```

如果只是先确认脚本能启动：

```bash
python scripts/smoke_qwen_vision.py --help
python scripts/run_offline_episode.py --help
python scripts/summarize_run.py --help
```

## 4. 推荐运行命令

### 4.1 DeepSeek 文本诊断

```bash
python scripts/smoke_deepseek.py \
  --text-only \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

注意：不要用 DeepSeek 跑图片 smoke test。当前测试结论是该接口不支持图片输入。

### 4.2 Qwen 视觉诊断

先跑无图诊断：

```bash
python scripts/smoke_qwen_vision.py \
  --diagnose \
  --image data/samples/library_entrance.png \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

再跑简单图片诊断：

```bash
python scripts/smoke_qwen_vision.py \
  --simple-image \
  --timeout 30 \
  --retries 0 \
  --image data/samples/library_entrance.png \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

最后跑完整 TARIC JSON：

```bash
python scripts/smoke_qwen_vision.py \
  --timeout 90 \
  --retries 0 \
  --max-tokens 1024 \
  --image data/samples/library_entrance.png \
  --instruction "Find the red library entrance. Follow the path and avoid grass."
```

### 4.3 离线 demo episode

```bash
python scripts/run_offline_episode.py \
  --manifest data/samples/demo.jsonl \
  --output outputs/demo_run.jsonl \
  --qwen

python scripts/summarize_run.py --input outputs/demo_run.jsonl
```

### 4.4 30 episode synthetic 数据

先用 mock 跑，避免 API 成本：

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/synthetic_30/manifest.jsonl \
  --output outputs/synthetic_30_mock_run.jsonl \
  --mock

python scripts/summarize_run.py --input outputs/synthetic_30_mock_run.jsonl
```

确认流程没有问题后，再少量抽样用 Qwen 跑。不要一开始对 240 帧全部调用 API。

## 5. Manifest 数据格式

每一行是一个 step，JSONL 格式。核心字段如下：

```json
{
  "episode_id": "route_001",
  "step_id": 0,
  "image_path": "data/episodes/route_001/0001.jpg",
  "instruction": "Find the red library entrance. Follow the paved path and avoid grass.",
  "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
  "camera": {"width": 960, "height": 540, "fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 270.0},
  "goal_position": {"x": 30.0, "y": 0.0, "z": 0.0},
  "shortest_path_m": 30.0,
  "cue_available_gt": true
}
```

后续要复现论文，关键就是把 synthetic manifest 替换为真实或仿真的轨迹数据。

## 6. 当前进度判断

当前完成度约 50%。

已经完成：

- TARIC 工程骨架。
- DeepSeek 文本规划接入。
- Qwen 视觉 API 接入。
- 单帧视觉 smoke test。
- demo episode 离线闭环。
- 30 episode synthetic 数据集。
- SR、SPL、Fail@CF、CFSR 指标统计。
- ROS bridge 初始骨架。

还没有完成：

- 真实户外或仿真环境数据集。
- 论文级规模评估。
- 轻量视觉模块训练。
- 伪标签清洗和人工校验。
- 与真实机器人或 ROS bag 的完整联调。
- 与论文 baseline 的系统对比。

## 7. 下一步工作计划

### 阶段 A：真实 small_eval 数据

目标：制作第一个真实小规模评估集。

建议规格：

- 5 到 10 个真实 episode。
- 每个 episode 10 到 30 帧。
- 图片可以来自校园拍摄、仿真截图或 ROS bag 抽帧。
- 每帧需要填写 pose、camera、goal_position、shortest_path_m 和 cue_available_gt。

输出文件建议：

```text
data/episodes/small_eval/manifest.jsonl
data/episodes/small_eval/images/
```

运行：

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_qwen_run.jsonl \
  --qwen

python scripts/summarize_run.py --input outputs/small_eval_qwen_run.jsonl
```

### 阶段 B：伪标签生成

用 Qwen 给真实 small_eval 生成伪标签：

```bash
python scripts/generate_pseudolabels.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_pseudolabels.jsonl
```

人工检查重点：

- `visible` 是否正确。
- `focus_pixel` 是否落在目标区域。
- `tile_scores` 是否指向目标方向。
- `traversability_scores` 是否避开草地、墙体、障碍物。
- `cue_bearing_deg` 是否符合视觉直觉。

### 阶段 C：轻量模块训练

训练目标不是训练 Qwen 或 DeepSeek，而是用伪标签训练小模块：

- visible gate。
- tile scorer。
- traversability sector classifier。

后续可以比较三种配置：

- Qwen-only。
- 轻量模型-only。
- Qwen + 轻量模型 fallback。

### 阶段 D：仿真或真实长距离评估

接近论文规模时，需要：

- 30 到 100 episodes per scene。
- 每条路线 200 到 1000m。
- 3 个 random seed。
- 统计 SR、SPL、Fail@CF、CFSR@10、CFSR@25、CFSR@50。
- 对比 frame-only、2D memory、full TARIC。

### 阶段 E：ROS / 机器人部署

已有 `src/taric_vln/ros/ros_bridge.py` 初始骨架。后续需要接真实 topic：

- 订阅 `/camera/rgb/image_raw`。
- 订阅 `/camera/camera_info`。
- 订阅 `/odom`。
- 发布 `/taric/executable_heading`。
- 发布 `/taric/debug/memory`。
- 发布 `/taric/debug/traversability`。

底层运动控制建议交给机器人已有 local planner 或安全控制器，不要让 VLM 直接控制电机。

## 8. 接手时优先看哪些文件

先看：

- `README.md`
- `docs/progress.md`
- `docs/qwen_dashscope.md`
- `docs/vision_adapter.md`
- `docs/reproduction_workflow.md`
- `docs/handoff.md`

再看代码：

- `src/taric_vln/perception/qwen_client.py`
- `src/taric_vln/perception/cue_extractor.py`
- `src/taric_vln/grounding/traversability.py`
- `src/taric_vln/memory/cue_memory.py`
- `src/taric_vln/sim/offline_runner.py`
- `src/taric_vln/eval/metrics.py`

最后看脚本：

- `scripts/smoke_qwen_vision.py`
- `scripts/run_offline_episode.py`
- `scripts/summarize_run.py`
- `scripts/generate_pseudolabels.py`

## 9. 重要注意事项

- 不要把任何 API key 上传 GitHub。
- 不要把 synthetic_30 当成论文结果，它只是工程闭环数据。
- DeepSeek 当前只走文本，不走图片。
- Qwen 图片请求可能慢，完整 TARIC JSON 建议 `--timeout 90 --max-tokens 1024`。
- 大规模调用 Qwen 前先抽样，否则容易浪费 API 费用。
- 如果后续上云 GPU，优先用于仿真、轻量模型训练和批量数据处理，不用于训练闭源 VLM。

当前最关键的下一步是制作真实 `small_eval` 数据集。完成它以后，项目才真正进入论文复现实验阶段。

## 10. 2026-07-08 更新：仿真 small_eval 已完成

由于目前没有真实环境，项目已先生成一个完全仿真的 `small_eval` 数据集：

```text
data/episodes/small_eval/manifest.jsonl
data/episodes/small_eval/images/
```

规模：

- 10 episodes。
- 每个 episode 12 step。
- 共 120 张仿真户外 VLN 图片。
- 包含 paved path、grass、target building、target sign 和 cue-interruption frame。
- `episode_id` 使用 `small_eval_001` 到 `small_eval_010`。

已完成验证：

```bash
python scripts/run_offline_episode.py \
  --manifest data/episodes/small_eval/manifest.jsonl \
  --output outputs/small_eval_mock_run.jsonl \
  --mock

python scripts/summarize_run.py --input outputs/small_eval_mock_run.jsonl
```

mock 汇总结果：

```json
{
  "episodes": 10.0,
  "sr": 1.0,
  "spl": 0.999399660463863,
  "fail_at_cf": 0.0,
  "mean_final_distance_m": 0.63
}
```

新的下一步不是再制作 small_eval，而是：

1. 用 Qwen 跑 `data/episodes/small_eval/manifest.jsonl`。
2. 检查每一步的视觉 JSON 输出是否合理。
3. 用 `scripts/generate_pseudolabels.py` 生成伪标签。
4. 基于伪标签开始训练 visible gate、tile scorer 和 traversability classifier。
