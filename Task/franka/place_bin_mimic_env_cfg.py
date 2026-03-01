# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic environment config for the place-cubes-into-bin task (Franka, IK Rel).

Defines 6 subtasks:
  grasp_1 → place_1 → grasp_2 → place_2 → grasp_3 → place_3
Each grasp picks up a cube; each place releases it into the blue sorting bin.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
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

        # ── subtask configs (6 subtasks) ────────────────────────────────
        subtask_configs = []

        # 1) Grasp cube_1
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp cube 1",
                next_subtask_description="Place cube 1 into the bin",
            )
        )

        # 2) Place cube_1 into bin
        subtask_configs.append(
            SubTaskConfig(
                object_ref="blue_sorting_bin",
                subtask_term_signal="place_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube 1 into the bin",
                next_subtask_description="Grasp cube 2",
            )
        )

        # 3) Grasp cube_2
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp cube 2",
                next_subtask_description="Place cube 2 into the bin",
            )
        )

        # 4) Place cube_2 into bin
        subtask_configs.append(
            SubTaskConfig(
                object_ref="blue_sorting_bin",
                subtask_term_signal="place_2",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube 2 into the bin",
                next_subtask_description="Grasp cube 3",
            )
        )

        # 5) Grasp cube_3
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp cube 3",
                next_subtask_description="Place cube 3 into the bin",
            )
        )

        # 6) Place cube_3 into bin (final subtask — no term signal needed)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="blue_sorting_bin",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube 3 into the bin",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
