# 完整工作流程指南

本文档详细介绍从录制演示数据到部署 VLA 模型的完整流程。

## 工作流程概览

```
录制演示 → Mimic 增强 → 格式转换 → 计算统计量 → 微调训练 → 部署推理
```

## 1. 录制演示数据

### 1.1 基础录制

```bash
cd ~/IsaacLab
./isaaclab.sh -p ../vla-arm-deployment/scripts/record_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --teleop_space task \
  --dataset_file ../vla-arm-deployment/datasets/franka_place_bin.hdf5 \
  --device cuda:0 \
  --enable_cameras
```

### 1.2 关键操作

- **F9**: 暂停/开始录制
- **E**: 保存当前 episode（标记为成功）
- **R**: 重置环境（不保存）

### 1.3 验证数据

```bash
python scripts/analyze_dataset.py datasets/franka_place_bin.hdf5 --mode full
```

## 2. Mimic 数据增强

### 2.1 标注子任务边界

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_dataset ../vla-arm-deployment/datasets/franka_place_bin.hdf5
```

### 2.2 生成增强数据

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0 \
  --input_dataset datasets/annotated_franka_place_bin.hdf5 \
  --num_trials 200
```

## 3. 转换为 LeRobot 格式

```bash
python scripts/convert_to_lerobot.py \
  --input datasets/franka_place_bin.hdf5 \
  --output datasets/lerobot/franka_place_bin \
  --repo-id local/franka_place_bin \
  --task "pick up cubes and place them into the blue bin" \
  --fps 30
```

## 4. 计算 Quantile 统计量

PI0.5 需要 quantile 统计量进行归一化：

```bash
python /path/to/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id local/franka_place_bin \
  --root datasets/lerobot/franka_place_bin
```

## 5. 微调 PI0.5 模型

```bash
cd /path/to/lerobot

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/franka_place_bin \
  --dataset.root=/path/to/vla-arm-deployment/datasets/lerobot/franka_place_bin \
  --policy.type=pi05 \
  --policy.use_amp=false \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=30000
```

模型将保存到：`outputs/train/.../checkpoints/`

## 6. 部署推理

```bash
./isaaclab.sh -p ../vla-arm-deployment/scripts/deploy_vla.py \
  --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
  --policy_path /path/to/checkpoint/pretrained_model \
  --dataset_repo_id local/franka_place_bin \
  --dataset_root ../vla-arm-deployment/datasets/lerobot/franka_place_bin \
  --device cuda:0 \
  --enable_cameras
```

## 故障排除

### HDF5 文件损坏

如果 HDF5 文件只有几十字节：
- 确保按 **E** 键保存，而不是 **R**
- 不要强制退出程序
- 使用绝对路径确保文件正确写入

### 录制卡顿

```bash
# 降低帧率
--step_hz 20  # 默认 30
```

### 内存不足

```bash
# 使用 CPU
--device cpu
```