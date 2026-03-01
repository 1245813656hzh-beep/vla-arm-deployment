# UR10_VLA - 键盘遥操作录制演示数据

本目录提供一个基于 IsaacLab `scripts/tools/record_demos.py` 改过的录制脚本：`record_demos.py`。
用途是用人类键盘遥操作采集 manipulation 任务的演示数据，并导出为 HDF5（episodes）。

## 项目结构

```
UR10_VLA/
├── record_demos.py        # 主录制脚本（键盘遥操作 + HDF5 导出）
├── analyze_dataset.py     # HDF5 数据集分析工具
├── Task/                  # 本地任务配置（优先于系统版本加载）
│   ├── franka/            # Franka 机械臂任务配置
│   │   ├── stack_ik_rel_env_cfg.py        # Stack 任务 IK 配置（含双摄像头 + 放宽成功条件）
│   │   ├── place_bin_ik_rel_env_cfg.py    # Place-into-bin 任务 IK 配置（双摄像头，无成功条件）
│   │   ├── stack_joint_pos_env_cfg.py     # Stack 任务关节位置配置
│   │   └── ...
│   ├── ur10_gripper/      # UR10 吸盘任务配置
│   └── ur_10e/            # UR10e 部署任务配置
└── README.md
```

## 运行前准备

- IsaacLab 仓库在 `~/IsaacLab`
- Conda 环境 `isaac` 已安装

在 `~/IsaacLab` 下执行：

```bash
source ~/miniforge3/bin/activate isaac
```

## 启动命令模板

建议在 `~/IsaacLab` 目录运行：

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task <ENV_ID> \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/<name>.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

## 全流程：录制 → Mimic 增强 → LeRobot → PI0.5 → 部署

下面以 **Franka Place-into-Bin** 为例，给出完整流程。

### 1) 录制原始演示（HDF5）

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_place_bin.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

说明：该任务 **无自动成功条件**，用 **E 键**手动保存 episode。

### 2) Mimic 数据增强（标注 + 生成）

Mimic 需要 3 步：

```bash
# (1) 标注子任务边界
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_dataset ./datasets/franka_place_bin.hdf5

# (2) 生成扩增数据
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_dataset <annotated_dataset.hdf5> \
  --num_trials 200
```

### 3) 转换为 LeRobot v3.0 数据集

```bash
python convert_to_lerobot.py \
  --input ./datasets/franka_place_bin.hdf5 \
  --output ./datasets/lerobot/franka_place_bin \
  --repo-id local/franka_place_bin \
  --task "pick up cubes and place them into the blue bin" \
  --fps 30
```

### 4) 为 PI0.5 计算 quantile stats（必须）

PI0.5 使用 **QUANTILES** 归一化（需要 `q01/q99` 等），转换后需补齐统计信息。

```bash
python /home/intern/copy_openarm_huang/openarms/OpenArm/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id local/franka_place_bin \
  --root ./datasets/lerobot/franka_place_bin
```

### 5) 微调 PI0.5（LeRobot 训练）

```bash
cd /home/intern/copy_openarm_huang/openarms/OpenArm/lerobot

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/franka_place_bin \
  --dataset.root=/home/intern/UR10_VLA/datasets/lerobot/franka_place_bin \
  --policy.type=pi05 \
  --policy.use_amp=false \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=30000
```

训练完成后，模型会输出到 `outputs/train/.../checkpoints/<step>/pretrained_model`。

### 6) 部署到 Isaac Sim（闭环推理）

已提供脚本：`deploy_vla.py`

```bash
./isaaclab.sh -p ../UR10_VLA/deploy_vla.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --policy_path /path/to/pretrained_model \
  --dataset_repo_id local/franka_place_bin \
  --dataset_root ../UR10_VLA/datasets/lerobot/franka_place_bin \
  --device cuda:0 \
  --enable_cameras
```

参数说明：

- `--task`：环境 ID（任务名）
- `--dataset_file`：输出 HDF5 文件路径（会创建/覆盖）
- `--device cuda:0`：仿真用 GPU；部分任务（如吸盘 suction）可能需要 `--device cpu`
- `--teleop_space task`：末端（SE(3) delta pose）遥操作
- `--enable_cameras`：启用摄像头（录制图像数据需要此参数）
- 键盘布局默认使用 `ijkl`，避免与 Isaac Sim viewport 的 WASD 导航快捷键冲突
- 如果 `UR10_VLA/Task` 下存在本地任务配置，会优先加载本地版本（方便二次开发）

## 可用任务与示例

### 1) Franka Stack：堆叠方块（主要任务）

三个方块堆叠任务，配置了双摄像头（table_cam + wrist_cam），成功条件 XY 阈值已放宽至 0.1。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_stack.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

### 2) Franka Place-into-Bin：将方块放入容器（推荐任务）

蓝色分拣容器在桌面中央，2-3 个方块散落在容器外。使用键盘控制 Franka 把方块逐个抓起放入容器。
无自动成功条件，用 **E 键**手动保存 episode。配置了双摄像头（table_cam + wrist_cam）。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_place_bin.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

### 3) Franka Lift：抓起方块

适合练习抓取+抬升，难度较低。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_lift.hdf5 \
  --device cuda:0
```

### 4) UR10 Long Suction Stack：UR10 吸盘堆叠

使用吸盘末端执行器。该类任务通常需要 CPU 仿真（否则可能报错/不稳定）。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/ur10_suction_stack.hdf5 \
  --device cpu
```

### 5) UR10e Deploy Reach（ROS inference 环境）

偏部署/推理风格的 reach 环境；可能没有显式 `success` 终止条件。
如果没有 `success`，录制脚本会导出所有 episode。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Deploy-Reach-UR10e-ROS-Inference-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/ur10e_reach_ros.hdf5 \
  --device cuda:0
```

## 键盘按键说明

### 录制控制

| 按键 | 功能 |
|------|------|
| `F9` | 开始/暂停录制（切换开关） |
| `E`  | **保存当前 episode 到 HDF5（标记为成功）并重置环境** |
| `R`  | 重置环境（丢弃当前 episode，不保存） |
| `F8` | 清空当前按键累计的位移/旋转输入（不重置环境） |

### 末端平移（ijkl 布局）

| 按键 | 方向 |
|------|------|
| `I / K` | +X / -X |
| `J / L` | +Y / -Y |
| `U / O` | +Z / -Z |

### 末端旋转

| 按键 | 方向 |
|------|------|
| `N / M` | +roll / -roll |
| `T / G` | +pitch / -pitch |
| `Y / B` | +yaw / -yaw |

### 夹爪

| 按键 | 功能 |
|------|------|
| `P` | 开/关切换 |

## 录制流程说明

1. 启动脚本后自动开始录制，确保 Isaac Sim viewport 获得焦点
2. 用键盘控制机械臂完成任务
3. **结束一条演示**有两种方式：
   - **自动成功**：任务达到成功条件并保持连续 `--num_success_steps`（默认 10）步后，自动导出
   - **手动保存**：按 `E` 键手动将当前 episode 标记为成功并导出到 HDF5
4. 按 `R` 可丢弃当前 episode 并重置（不保存）
5. 按 `F9` 可暂停/恢复录制

## 摄像头配置

Franka Stack 任务配置了两个 200x200 RGB 摄像头：

| 摄像头 | 位置 | 用途 |
|--------|------|------|
| `table_cam` | 桌面前方俯瞰视角 (1.0, 0.0, 0.6) | 全局观察 |
| `wrist_cam` | 机械臂腕部 (panda_hand) | 近距离观察 |

需要 `--enable_cameras` 参数才能启用摄像头数据录制。

## 成功条件

Franka Stack 任务的成功条件（本地配置已修改）：

- 三个方块垂直堆叠（cube_3 底部, cube_2 中间, cube_1 顶部）
- **XY 对齐阈值：0.1**（原始默认 0.04，已放宽 2.5 倍）
- 高度差：约 0.0468m（每层）
- 高度误差阈值：0.005m
- 夹爪必须打开

## HDF5 数据集结构

每条 demo 包含以下数据：

```
/data/demo_N/
├── actions                 # (T, 7) 操作指令 [dx, dy, dz, droll, dpitch, dyaw, gripper]
├── processed_actions       # (T-1, 8) 环境处理后的动作
├── obs/                    # 观测数据
│   ├── joint_pos           # (T, 9) 关节位置
│   ├── joint_vel           # (T, 9) 关节速度
│   ├── eef_pos             # (T, 3) 末端位置
│   ├── eef_quat            # (T, 4) 末端姿态四元数
│   ├── gripper_pos         # (T, 2) 夹爪位置
│   ├── cube_positions      # (T, 9) 三个方块的位置
│   ├── cube_orientations   # (T, 12) 三个方块的朝向
│   ├── object              # (T, 39) 综合物体状态
│   ├── table_cam           # (T, 200, 200, 3) 桌面摄像头 RGB 图像
│   └── wrist_cam           # (T, 200, 200, 3) 腕部摄像头 RGB 图像
├── states/                 # 物理状态（用于回放/重置）
│   ├── articulation/robot/ # 机器人关节位置/速度/根位姿
│   └── rigid_object/       # 各方块位姿/速度
└── initial_state/          # episode 初始状态
```

## 数据集分析工具

使用 `analyze_dataset.py` 检查录制的 HDF5 文件：

```bash
# 查看顶层结构
python analyze_dataset.py /path/to/dataset.hdf5 --mode top

# 查看完整结构
python analyze_dataset.py /path/to/dataset.hdf5 --mode full
```

## 常见问题

- **觉得卡顿**：可以降低控制频率，例如 `--step_hz 20`（默认 30）
- **按键冲突**：默认已用 `ijkl` 并避开了 `H` 等常见热键；仍冲突时确保 viewport 聚焦
- **Overriding environment 警告**：正常现象，本地任务注册时覆盖系统版本导致，不影响功能
- **摄像头图像全黑**：确认加了 `--enable_cameras` 参数
- **demo 只有 1 步**：说明在重置后立刻按了 E 键，注意先操作几步再保存
- **PI0.5 报 stats 缺失**：请先运行 `augment_dataset_quantile_stats.py` 生成 `q01/q99` 等统计量
