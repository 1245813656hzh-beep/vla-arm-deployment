# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Place cubes into a bin task — Franka IK Relative control with cameras.

Task: Pick up cubes scattered on the table and place them into a blue sorting bin.
No success condition — use E key to manually save episodes.
"""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import (
    GripperRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import (
    Se3RelRetargeterCfg,
)
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import bin_stack_joint_pos_env_cfg
from . import place_bin_observations

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values and camera images."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        table_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination observations for Mimic data generation.

        These signals let Mimic annotate/detect when each subtask completes:
          - grasp_N: cube_N is grasped (EE close + gripper closed)
          - place_N: cube_N is in the bin (XY/Z check + gripper open)
        """

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        place_1 = ObsTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_1"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        place_2 = ObsTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_2"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )
        grasp_3 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaPlaceBinEnvCfg(bin_stack_joint_pos_env_cfg.FrankaBinStackEnvCfg):
    """Place cubes into bin — IK relative control, dual cameras, no success condition."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent (sets up bin, cubes, events, etc.)
        super().__post_init__()

        # Remove ALL termination conditions (no success, no timeout, no dropping)
        self.terminations.success = None
        self.terminations.time_out = None
        self.terminations.cube_1_dropping = None
        self.terminations.cube_2_dropping = None
        self.terminations.cube_3_dropping = None

        # Override cube reset events: ALL 3 cubes spawn outside the bin
        # Remove the old event that put cube_1 inside the bin
        self.events.reset_cube_1_pose = None
        # Randomize all 3 cubes together outside the bin
        # Cubes spawn BEHIND the bin (away from robot), within table_cam view
        # Bin is at x=0.4, spawn cubes at x=0.58-0.65 (0.18-0.25m behind bin)
        self.events.reset_cube_pose = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.58, 0.65),  # Behind bin, visible to table_cam at (1.0, 0.0, 0.6)
                    "y": (-0.20, 0.20),  # Spread left and right
                    "z": (0.0203, 0.0203),
                    "yaw": (-1.0, 1.0, 0),
                },
                "min_separation": 0.12,
                "asset_cfgs": [
                    SceneEntityCfg("cube_1"),
                    SceneEntityCfg("cube_2"),
                    SceneEntityCfg("cube_3"),
                ],
            },
        )

        # Set Franka with stiffer PD controller for better IK tracking
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set IK relative control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.8,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Wrist camera (mounted on panda_hand)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # Table camera (overhead view) - original position
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.6),  # Original position
                rot=(0.27060, -0.65328, -0.65328, 0.27060),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["table_cam", "wrist_cam"]

        # Teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.08,
                    rot_sensitivity=0.08,
                    sim_device=self.sim.device,
                ),
            }
        )
