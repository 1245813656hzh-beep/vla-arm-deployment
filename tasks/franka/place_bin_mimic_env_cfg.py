# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic environment config for the place-cubes-into-bin task (Franka, IK Rel).

Defines 6 subtasks:
  grasp_1 → place_1 → grasp_2 → place_2 → grasp_3 → place_3
Each grasp picks up a cube; each place releases it into the blue sorting bin.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import place_bin_observations
from isaaclab.utils import configclass

from .place_bin_ik_rel_env_cfg import FrankaPlaceBinEnvCfg


@configclass
class FrankaPlaceBinIKRelMimicEnvCfg(FrankaPlaceBinEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for Franka place-cubes-into-bin IK Rel env."""

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # ── datagen config ──────────────────────────────────────────────
        self.datagen_config.name = "demo_src_place_bin_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # ── subtask configs (only cube_3 pick/place) ───────────────────
        subtask_configs = []
        # 1) Grasp cube_3
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp cube 3",
                next_subtask_description="Place cube 3 into the bin",
            )
        )

        # 2) Place cube_3 into bin
        subtask_configs.append(
            SubTaskConfig(
                object_ref="blue_sorting_bin",
                subtask_term_signal="place_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube 3 into the bin",
            )
        )

        self.subtask_configs["franka"] = subtask_configs

        # ── success termination (required for Mimic datagen) ───────────
        self.terminations.success = DoneTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_3"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )

# 1. object_ref="blue_sorting_bin"
# 作用：指定这个子任务要交互的目标物体  
# 含义：机械臂需要移动到 blue_sorting_bin（蓝色分拣桶）附近  
# 对 place 任务：表示要把物体放到这个 bin 里
# ---
# 2. subtask_term_signal="place_3"
# 作用：定义子任务完成的信号名称  
# 含义：当环境返回的 place_3 信号为 True 时，表示子任务完成  
# 在哪里定义：在 ObservationsCfg.SubtaskCfg 里：
# place_3 = ObsTerm(func=place_bin_observations.object_in_bin, ...)
# ---
# 3. subtask_term_offset_range=(0, 0)
# 作用：控制子任务终止信号的触发时机偏移  
# 格式：(min_offset, max_offset)，单位是 timestep  
# 含义：
# - (0, 0) = 严格按照信号触发时刻终止
# - (-5, 5) = 在信号触发前 5 步到后 5 步之间随机选择终止点
# 用途：增加时间维度上的随机性
# ---
# 4. selection_strategy="nearest_neighbor_object"
# 作用：选择源 demo 的策略  
# 可选值：
# - "nearest_neighbor_object"：选择物体位置最接近的源 demo
# - "random"：随机选择
# - "sequential"：顺序选择
# 含义：从标注的数据池中，找 cube_3 初始位置最接近新场景的那个 demo
# ---
# 5. selection_strategy_kwargs={"nn_k": 3}
# 作用：选择策略的参数  
# 含义：
# - "nn_k": 3 = 从最近的 3 个候选 demo 中随机选 1 个
# - 增大 k 值会增加多样性，但可能降低相似度
# ---
# 6. action_noise=0.01 ⭐
# 作用：动作噪声大小  
# 含义：给每个动作添加 ±1cm 的随机扰动（6D 位姿空间）  
# 影响：
# - 0.0 = 完全复现源 demo，无噪声
# - 0.03 = 3cm 噪声，更随机但可能抖动
# - 0.01 = 1cm 噪声，平衡平滑与多样性
# ---
# 7. num_interpolation_steps=5 ⭐
# 作用：插值平滑步数  
# 含义：从当前姿态到目标姿态之间，用 5 步线性插值过渡  
# 影响：
# - 0 = 直接跳转到目标，可能突变
# - 5 = 5 步平滑过渡，减少抖动
# - 10 = 更平滑但执行时间更长
# ---
# 8. num_fixed_steps=0
# 作用：子任务开始前固定执行的步数  
# 含义：在正式开始子任务动作前，先保持固定动作多少步  
# 用途：比如在 grasp 前停顿一下观察
# ---
# 9. apply_noise_during_interpolation=False
# 作用：是否在插值过程中也添加噪声  
# 含义：
# - False = 只在最终目标动作加噪，插值过程保持平滑
# - True = 每步插值都加噪，更随机但可能更抖