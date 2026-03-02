# 常见问题解答 (FAQ)

## 录制相关问题

### Q: 为什么我的 HDF5 文件只有 96 字节？

**A**: 这通常是因为：
1. 没有按 **E** 键保存 episode
2. 按了 **R** 重置（这会丢弃当前录制）
3. 程序在保存前崩溃了

**解决方法**：确保完成操作后按 **E** 键保存。

### Q: 录制时有很多静止帧怎么办？

**A**: 
- 启动后按 **F9** 暂停录制
- 准备好后再按 **F9** 开始
- 这样可以避免录制闲置时段的数据

### Q: 第二次运行会覆盖第一次的数据吗？

**A**: **会！** IsaacLab 使用 `h5py.File(path, "w")` 模式，会完全清空同名文件。

**建议**：每次录制使用不同的文件名，例如：
```bash
--dataset_file datasets/franka_place_bin_v2.hdf5
```

## 环境问题

### Q: 出现 "No module named 'isaaclab'" 错误？

**A**: 确保：
1. 在 `~/IsaacLab` 目录下运行
2. 已激活正确的 conda 环境：`conda activate isaac`
3. IsaacLab 已正确安装

### Q: 摄像头图像全黑？

**A**: 确保添加了 `--enable_cameras` 参数。

### Q: 录制非常卡顿？

**A**: 降低控制频率：
```bash
--step_hz 20  # 默认为 30
```

## 数据转换问题

### Q: LeRobot 转换失败？

**A**: 检查 HDF5 文件是否完整：
```bash
python scripts/analyze_dataset.py datasets/your_file.hdf5 --mode top
```

### Q: PI0.5 训练报错缺少 stats？

**A**: 必须先运行 quantile stats 计算脚本：
```bash
python augment_dataset_quantile_stats.py \
  --repo-id your_repo_id \
  --root datasets/lerobot/your_dataset
```

## 部署问题

### Q: 模型推理时报错？

**A**: 检查：
1. 模型路径是否正确
2. 数据集路径是否正确（需要 stats）
3. 摄像头是否启用 (`--enable_cameras`)

### Q: 如何修改任务描述？

**A**: 在 `deploy_vla.py` 中使用 `--task_description` 参数：
```bash
--task_description "pick up the red cube"
```