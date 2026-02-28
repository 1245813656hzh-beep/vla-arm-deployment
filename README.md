# UR10_VLA - 键盘遥操作录制演示数据

本目录提供一个基于 IsaacLab `scripts/tools/record_demos.py` 改过的录制脚本：`record_demos.py`。
用途是用人类键盘遥操作采集 manipulation 任务的演示数据，并导出为 HDF5（episodes）。

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
  --device cuda:0
```

参数说明：

- `--task`：环境 ID（任务名）
- `--dataset_file`：输出 HDF5 文件路径（会创建/覆盖）
- `--device cuda:0`：仿真用 GPU；部分任务（如吸盘 suction）可能需要 `--device cpu`
- `--teleop_space task`：末端（SE(3) delta pose）遥操作
- 键盘布局默认使用 `ijkl`，避免与 Isaac Sim viewport 的 WASD 导航快捷键冲突（通常不需要额外参数）
- 如果 `UR10_VLA/Task` 下存在本地任务配置，会优先加载本地版本（方便二次开发）

## 可用任务与示例

### 1) Franka Lift：抓起方块

适合练习抓取+抬升，难度较低。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_lift.hdf5 \
  --device cuda:0
```

### 2) Franka Stack：堆叠方块

比 Lift 更难（接触更多、需要精确放置）。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/franka_stack.hdf5 \
  --device cuda:0
```

### 3) UR10 Long Suction Stack：UR10 吸盘堆叠

使用吸盘末端执行器。该类任务通常需要 CPU 仿真（否则可能报错/不稳定）。

```bash
./isaaclab.sh -p ../UR10_VLA/record_demos.py \
  --task Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ./datasets/ur10_suction_stack.hdf5 \
  --device cpu
```

### 4) UR10e Deploy Reach（ROS inference 环境）

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

## 如何开始/结束录制

- 启动脚本后会自动开始录制，无需额外按键
- 确保 Isaac Sim 窗口/viewport 获得焦点（鼠标点一下 viewport），然后直接用键盘控制机械臂完成任务
- `F9`：开始/暂停录制（切换开关）
- **结束一条演示**：
  - 任务达到成功条件并保持连续 `--num_success_steps`（默认 10）步后，会自动结束本条并导出到 `--dataset_file`
  - 或按 `R` 重置环境，提前结束当前演示并开始下一条

## 录制内容说明

- 录制器使用 `ActionStateRecorder`，会保存**动作(action)** 与**状态(state)**
- 当 `--teleop_space task` 时，记录的 action 是末端 **SE(3) delta pose**（不是关节命令）
- 状态里通常包含关节位置/速度等机器人状态（具体字段取决于任务环境配置）

## 键盘按键说明

通用：

- `F9`：开始/暂停录制（切换开关）
- `R`：重置环境 / 重新开始本条录制

### 布局 1：默认 `ijkl`（推荐，减少与 viewport 冲突）

末端平移：

- `I/K`：+X / -X
- `J/L`：+Y / -Y
- `U/O`：+Z / -Z

末端旋转：

- `N/M`：+roll / -roll
- `T/G`：+pitch / -pitch
- `Y/B`：+yaw / -yaw

夹爪：

- `P`：开/关切换

设备输入清零（仅清空末端 delta 命令，不重置环境）：

- `F8`：清空当前按键累计的位移/旋转输入

环境重置：

- `R`：重置环境 / 重新开始本条录制

## 常见问题

- 觉得卡顿：可以把控制频率降一点，例如加上 `--step_hz 20`（默认 30）。
- 按键冲突：默认已用 `ijkl` 并避开了 `H` 等常见热键；仍冲突时，确保 viewport 聚焦。
