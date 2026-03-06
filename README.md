<h1 align="center">🤖 VLA Arm Deployment</h1>

<p align="center">
  <a href="https://isaac-sim.github.io/IsaacLab/"><img alt="IsaacLab" src="https://img.shields.io/badge/IsaacLab-1.0+-blue.svg"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

<p align="center">
  <b>Teleoperation Recording and VLA Model Deployment Toolkit</b><br>
  <i>基于 IsaacLab 的遥操作录制和视觉-语言-动作 (VLA) 模型部署工具集</i>
</p>

<p align="center">
  <a href="#-功能特性">功能特性</a> •
  <a href="#-安装">安装</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-工作流程">工作流程</a> •
  <a href="#-项目结构">项目结构</a>
</p>

---

## ✨ 功能特性

- 🎮 **键盘遥操作录制** - 支持双摄像头实时录制机械臂操作演示
- 📊 **数据格式转换** - HDF5 到 LeRobot v3.0 格式无缝转换
- 🤖 **PI0.5 模型支持** - 完整的 VLA 模型微调和部署流程
- 🔄 **Mimic 数据增强** - 支持子任务边界标注和自动生成增强数据
- 📦 **开源友好** - 相对路径设计，便于版本控制和社区共享

## 📋 前置要求

- **IsaacLab** >= 1.0 ([安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- **Python** >= 3.10
- **CUDA** >= 11.8 (用于 GPU 加速)
- **Conda** 或 **venv** (推荐)

## 🚀 安装

### 1. 克隆仓库

```bash
cd ~
git clone https://github.com/yourusername/vla-arm-deployment.git
cd vla-arm-deployment
```

### 2. 创建 Conda 环境

```bash
# 创建环境
conda create -n vla python=3.10
conda activate vla

# 安装基础依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e ".[dev]"
```

### 3. 配置 IsaacLab 路径

确保 IsaacLab 已安装，并在 `~/IsaacLab` 目录下可用。

## 🎯 快速开始

### 录制演示数据

在 `~/IsaacLab` 目录下运行：

```bash
cd ~/IsaacLab
source ~/miniforge3/bin/activate isaac  # 或你的 conda 环境

# 我的本地资产路径，可更换为你的实际路径
export ISAACLAB_ASSETS_DIR=/home/intern/vla-arm-deployment/assets/Isaac/IsaacLab

# 录制 Franka 放置任务
./isaaclab.sh -p ../vla-arm-deployment/scripts/record_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ../vla-arm-deployment/datasets/franka_place_bin.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

### 分析数据集

```bash
python scripts/analyze_dataset.py datasets/franka_place_bin.hdf5 --mode full
```

### 数据集标注和增强
```bash
# (1) 标注子任务边界
./isaaclab.sh -p ../vla-arm-deployment/scripts/annotate_demos_local.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_file ../vla-arm-deployment/datasets/franka_place_bin_v1.hdf5 \
  --output_file ../vla-arm-deployment/datasets/franka_place_bin_v2_annotated.hdf5

# (2) 生成扩增数据
./isaaclab.sh -p ../vla-arm-deployment/scripts/generate_dataset_local.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_file ../vla-arm-deployment/datasets/franka_place_bin_3camera_annotated.hdf5 \
  --output_file ../vla-arm-deployment/datasets/franka_place_bin_3camera_augmented.hdf5 \
  --generation_num_trials 200

# (3) 提取所有相机的视频
python3 /home/intern/vla-arm-deployment/scripts/extract_hdf5_videos.py \
  --input /home/intern/vla-arm-deployment/datasets/franka_place_bin_3camera_augmented.hdf5 \
  --output ./videos
```

### 转换为 LeRobot 格式

```bash
python scripts/convert_to_lerobot.py \
  --input datasets/franka_place_bin.hdf5 \
  --output datasets/lerobot/franka_place_bin \
  --repo-id local/franka_place_bin \
  --task "pick up cubes and place them into the blue bin"
```

## 📖 工作流程

完整的从录制到部署的工作流程：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. 录制演示    │────▶│  2. Mimic增强   │────▶│ 3. 格式转换     │
│  (Teleop)       │     │  (Data Aug)     │     │  (LeRobot)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  6. 部署模型    │◀────│  5. 微调训练    │◀────│ 4. 计算统计量   │
│  (Inference)    │     │  (Training)     │     │  (Stats)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

详细步骤请查看 [完整工作流程指南](docs/workflow.md)。

## 🎮 键盘控制

### 基本控制

| 按键 | 功能 |
|------|------|
| `F9` | 开始/暂停录制 |
| `E`  | **保存当前 episode**（标记为成功） |
| `R`  | 重置环境（不保存） |
| `F8` | 清空累计输入 |

### 末端执行器控制 (IJLK 布局)

| 按键 | 方向 |
|------|------|
| `I / K` | +X / -X |
| `J / L` | +Y / -Y |
| `U / O` | +Z / -Z |
| `P` | 夹爪开/关 |

## 📁 项目结构

```
vla-arm-deployment/
├── 📂 scripts/               # 可执行脚本
│   ├── record_demos.py      # 遥操作录制
│   ├── convert_to_lerobot.py # 格式转换
│   ├── deploy_vla.py        # 模型部署
│   └── analyze_dataset.py   # 数据集分析
│
├── 📂 src/vla_arm_deployment/  # Python 包
│   ├── __init__.py
│   └── utils/               # 工具函数
│       ├── __init__.py
│       └── dataset_utils.py
│
├── 📂 tasks/                # IsaacLab 任务配置
│   ├── franka/              # Franka 机械臂任务
│   ├── ur10_gripper/        # UR10 吸盘任务
│   └── ur_10e/              # UR10e 部署任务
│
├── 📂 datasets/             # 数据集目录 (gitignored)
│   ├── *.hdf5              # 原始 HDF5 数据
│   └── lerobot/            # LeRobot 格式数据
│
├── 📂 configs/              # 配置文件
├── 📂 docs/                 # 文档
├── 📂 examples/             # 示例代码
├── 📂 tests/                # 测试代码
│
├── 📝 README.md            # 本文件
├── 📝 LICENSE              # MIT 许可证
├── 📝 requirements.txt     # Python 依赖
├── 📝 pyproject.toml       # 项目配置
├── 📝 Makefile             # 常用命令
└── 📝 .gitignore           # Git 忽略规则
```

## 🛠️ 可用任务

| 任务 ID | 描述 | 推荐 |
|---------|------|------|
| `Isaac-Stack-Cube-Franka-IK-Rel-v0` | Franka 堆叠方块 | ⭐⭐⭐ |
| `Isaac-Place-Bin-Franka-IK-Rel-v0` | Franka 放入容器 | ⭐⭐⭐ |
| `Isaac-Lift-Cube-Franka-IK-Rel-v0` | Franka 抓取方块 | ⭐⭐ |
| `Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0` | UR10 吸盘堆叠 | ⭐⭐ |

## 🧪 开发

### 代码格式化

```bash
make format  # 使用 ruff 格式化
make lint    # 代码检查
```

### 运行测试

```bash
make test
```

### 清理生成文件

```bash
make clean
```

## ⚠️ 重要提醒

### 录制启动逻辑

- 启动后录制默认**开启**（`running_recording_instance = True`）
- 闲置时段的数据也会被记录到 HDF5
- **建议**：加载完成后按 `F9` 暂停，准备好后再开始录制
- 四元数在代码里写 `(-x, w, y, z)` 在isaac sim里是 `(w, x, y, z)`

### HDF5 文件覆盖警告

- **每次运行都会覆盖同名文件！** 使用 `h5py.File(path, "w")` 模式
- 第二次运行同样的 `--dataset_file` 命令会**丢失**第一次的数据
- **建议**：每次录制使用不同的文件名，或使用 `make analyze` 检查后再继续

### 数据采集建议

**目标**：为后续 Mimic 数据增强提供高质量原始数据

- **每条演示**：确保完整流程（抓取 → 放入 → 重复），至少 200-400 步
- **采集数量**：建议 15-30 条完整演示/文件（太少则增强效果有限）
- **多样化方向**：
  - 方块位置随机（已配置）
  - 抓取角度变化（不同方向接近方块）
  - 允许少量失败/调整（增加鲁棒性）
- **不要刻意随机化抓取顺序**：让模型自然学会抓最近的方块
- **录制节奏**：不要过快，保持自然操作节奏

### Mimic 的完整流程是：
- | 步骤 | 做什么 | 关键性 |
- |------|--------|--------|
- | 1. 场景随机化 | cube 随机位置、机械臂初始姿态随机 | ⭐⭐⭐ |
- | 2. 源 demo 选择 | 从标注池找最相似的历史片段 | ⭐⭐ |
- | 3. 空间变换 | 把动作映射到新 cube 位置 | ⭐⭐⭐⭐⭐ |
- | 4. 动作插值 | 平滑过渡到目标 | ⭐⭐ |
- | 5. 噪声添加 | 加 3cm 扰动（只是最后一步） | ⭐ |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

## 🙏 致谢

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) - NVIDIA 的机器人学习框架
- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face 的机器人学习库
- [PI0.5](https://github.com/your-pi05-repo) - VLA 模型

---

<p align="center">
  Made with ❤️ by VLA Arm Deployment Contributors
</p>