# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom subtask observation functions for the place-into-bin task.

Provides `object_in_bin()` — checks if a cube has been placed inside the
sorting bin (XY within bin boundaries, Z at bin floor level, gripper open).
"""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_in_bin(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    bin_cfg: SceneEntityCfg,
    xy_threshold: float = 0.06,
    z_min: float = 0.015,
    z_max: float = 0.08,
) -> torch.Tensor:
    """Check if an object has been placed inside the bin and the gripper is open.

    The signal is True when ALL of:
    1. The cube's XY position is within ``xy_threshold`` of the bin center.
    2. The cube's Z position (relative to bin bottom) is between ``z_min`` and ``z_max``.
    3. The gripper is open (object has been released).

    Args:
        env: The environment instance.
        robot_cfg: Robot articulation config.
        object_cfg: The cube rigid object config.
        bin_cfg: The sorting bin rigid object config.
        xy_threshold: Max XY distance from bin center to count as "inside".
        z_min: Minimum height above bin origin to count as inside (filters table-level).
        z_max: Maximum height above bin origin to count as inside.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]
    sorting_bin: RigidObject = env.scene[bin_cfg.name]

    cube_pos = cube.data.root_pos_w
    bin_pos = sorting_bin.data.root_pos_w

    pos_diff = cube_pos - bin_pos
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_diff = pos_diff[:, 2]

    # Cube is spatially inside the bin
    in_bin = torch.logical_and(xy_dist < xy_threshold, z_diff > z_min)
    in_bin = torch.logical_and(in_bin, z_diff < z_max)

    # Gripper must be open (object released)
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        in_bin = torch.logical_and(suction_cup_is_open.squeeze(-1), in_bin)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Only parallel gripper is supported"
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                env.device
            )
            in_bin = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    open_val,
                    atol=1e-4,
                    rtol=1e-4,
                ),
                in_bin,
            )
            in_bin = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    open_val,
                    atol=1e-4,
                    rtol=1e-4,
                ),
                in_bin,
            )

    return in_bin
