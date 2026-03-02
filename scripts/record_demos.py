# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib
import sys

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/dataset.hdf5",
    help="File path to export recorded demos. Relative to project root (e.g., datasets/my_data.hdf5)",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--teleop_space",
    type=str,
    default="auto",
    choices=["auto", "task", "joint"],
    help=(
        "Teleoperation command space. "
        "'task' uses end-effector delta pose (SE(3)); 'joint' uses joint-space actions; "
        "'auto' switches to task-space IK when the environment arm action is joint-space."
    ),
)

parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Keyboard teleop depends on Kit appwindow input APIs and won't work in headless mode.
if getattr(args_cli, "headless", False) and args_cli.teleop_device.lower() == "keyboard":
    parser.error(
        "Keyboard teleop requires a GUI. Remove --headless or choose a non-keyboard teleop device."
    )

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import gymnasium as gym
import logging
import os
import time
import torch
import numpy as np

import carb

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import (
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import (
    InstructionDisplay,
    show_subtask_instructions,
)

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


def register_local_tasks(task_root: str) -> None:
    if not os.path.isdir(task_root):
        return

    if task_root not in sys.path:
        sys.path.insert(0, task_root)

    from gymnasium.envs.registration import registry as gym_registry

    local_task_ids = {
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "Isaac-Place-Bin-Franka-IK-Rel-v0",
        "Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0",
        "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
        "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
    }

    for task_id in local_task_ids:
        if task_id in gym_registry:
            del gym_registry[task_id]

    for module_name in ("franka", "ur10_gripper", "ur_10e"):
        try:
            __import__(module_name)
        except Exception as exc:
            logger.warning(f"Failed to import local task module '{module_name}': {exc}")


class MappedSe3Keyboard(Se3Keyboard):
    """Se3Keyboard variant with remappable key layout.

    This is used to avoid collisions with Isaac Sim viewport WASD navigation/hotkeys.
    """

    def __init__(self, cfg: Se3KeyboardCfg):
        super().__init__(cfg)

    def _create_key_bindings(self):
        # Alternate mapping to avoid viewport WASD conflicts.
        # Move: I/K (x), J/L (y), U/O (z)
        # Rotate: N/M (x), T/G (y), Y/B (z)
        # Gripper toggle: P
        # Device reset (clear delta commands): F8
        self._device_reset_key = "F8"
        self._gripper_key = "P"
        self._pos_keys = ["I", "K", "J", "L", "U", "O"]
        # NOTE: Avoid "H" since Isaac Sim commonly binds it to visibility/hide actions.
        self._rot_keys = ["N", "M", "T", "G", "Y", "B"]
        self._INPUT_KEY_MAPPING = {
            "P": True,
            "I": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "K": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "J": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "L": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "U": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "N": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "M": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "Y": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "B": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        # event.input may be a carb.input enum (with .name) or a plain string
        raw = event.input
        key_name = raw.name if hasattr(raw, "name") else str(raw)
        key_name = key_name.upper()
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name == getattr(self, "_device_reset_key", "L"):
                self.reset()
            if key_name == self._gripper_key:
                self._close_gripper = not self._close_gripper
            elif key_name in self._pos_keys:
                self._delta_pos += self._INPUT_KEY_MAPPING[key_name]
            elif key_name in self._rot_keys:
                self._delta_rot += self._INPUT_KEY_MAPPING[key_name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._pos_keys:
                self._delta_pos -= self._INPUT_KEY_MAPPING[key_name]
            elif key_name in self._rot_keys:
                self._delta_rot -= self._INPUT_KEY_MAPPING[key_name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()
        return True


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # If the environment provides a keyboard teleop device, replace it with the remapped
    # keyboard (default: ijkl) to avoid Isaac Sim viewport hotkey conflicts.
    if (
        getattr(args_cli, "teleop_device", "").lower() == "keyboard"
        and hasattr(env_cfg, "teleop_devices")
        and hasattr(env_cfg.teleop_devices, "devices")
        and "keyboard" in env_cfg.teleop_devices.devices
    ):
        kb_cfg = env_cfg.teleop_devices.devices["keyboard"]
        # Override the device class used by the factory.
        setattr(kb_cfg, "class_type", MappedSe3Keyboard)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # Only export episodes explicitly marked as successful (via 'E' key).
    # This prevents accidental exports when pressing 'R' to reset.
    export_all_episodes = False

    # Optionally switch joint-space arm action to task-space IK for end-effector teleop.
    def _arm_action_is_joint_space(cfg: object) -> bool:
        cls = getattr(cfg, "class_type", None)
        name = getattr(cls, "__name__", "") if cls is not None else ""
        return name in {
            "JointPositionAction",
            "RelativeJointPositionAction",
            "JointVelocityAction",
            "JointEffortAction",
        }

    arm_action_cfg = getattr(env_cfg.actions, "arm_action", None)

    want_task_space = args_cli.teleop_space == "task"
    if args_cli.teleop_space == "auto" and _arm_action_is_joint_space(arm_action_cfg):
        want_task_space = True

    # Only rewrite the environment action when it is joint-space.
    # Many tasks (e.g. Franka IK tasks) already define a correct task-space IK action.
    if want_task_space and _arm_action_is_joint_space(arm_action_cfg):
        # Prefer the command generator body as the end-effector reference (common in reach tasks).
        body_name = "ee_link"
        if (
            hasattr(env_cfg, "commands")
            and hasattr(env_cfg.commands, "ee_pose")
            and hasattr(env_cfg.commands.ee_pose, "body_name")
        ):
            body_name = env_cfg.commands.ee_pose.body_name

        # Reuse joint selection from the existing action if available.
        joint_names = [".*"]
        if arm_action_cfg is not None and hasattr(arm_action_cfg, "joint_names"):
            joint_names = arm_action_cfg.joint_names

        env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=joint_names,
            body_name=body_name,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_ALL
        if export_all_episodes
        else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(
    callbacks: dict[str, Callable], expected_action_dim: int, sim_device: str
) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_interface = None
    try:
        if (
            hasattr(env_cfg, "teleop_devices")
            and args_cli.teleop_device in env_cfg.teleop_devices.devices
        ):
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                # Se3Keyboard can optionally append a gripper term (7th element). Match env action dim.
                # Use an alternate key layout to avoid conflicts with Isaac Sim viewport hotkeys.
                cfg = Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    gripper_term=(expected_action_dim == 7),
                    sim_device=sim_device,
                )
                teleop_interface = MappedSe3Keyboard(cfg)
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(
                        pos_sensitivity=0.05,
                        rot_sensitivity=0.05,
                        sim_device=sim_device,
                    )
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, handtracking")
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        import omni.ui as ui

        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(
    env: gym.Env, success_term: object | None, success_step_count: int
) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
) -> int:
    """Handle resetting the environment.

    Resets the environment and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    # NOTE: Do not call env.recorder_manager.reset() here.
    # env.reset() triggers recorder_manager.record_pre_reset(), which concludes and exports
    # the current episode (depending on export mode). Resetting the recorder first would
    # discard the buffered episode and lead to empty datasets for tasks without explicit success.
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr
    instruction_display = None
    toggle_recording_key = "F9"

    def build_label_text(recorded_count: int, is_running: bool) -> str:
        status = "ON" if is_running else "PAUSED"
        return f"Recorded {recorded_count} successful demonstrations. Recording: {status} ({toggle_recording_key} to toggle)."

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")
        if instruction_display is not None:
            instruction_display.show_demo(
                build_label_text(current_recorded_demo_count, running_recording_instance)
            )

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")
        if instruction_display is not None:
            instruction_display.show_demo(
                build_label_text(current_recorded_demo_count, running_recording_instance)
            )

    def toggle_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = not running_recording_instance
        state = "started" if running_recording_instance else "paused"
        print(f"Recording {state}")
        if instruction_display is not None:
            instruction_display.show_demo(
                build_label_text(current_recorded_demo_count, running_recording_instance)
            )

    def export_and_reset():
        """Export current episode to HDF5 (marked as success) and reset the environment."""
        nonlocal should_reset_recording_instance
        try:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Episode manually exported as SUCCESS and saved to HDF5.")
        except Exception as exc:
            print(f"Failed to export episode: {exc}")
        should_reset_recording_instance = True

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "E": export_and_reset,
        toggle_recording_key: toggle_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    expected_action_dim = getattr(env.action_manager, "total_action_dim", None)
    if expected_action_dim is None:
        expected_action_dim = int(env.action_space.shape[0])
    teleop_interface = setup_teleop_device(
        teleoperation_callbacks, expected_action_dim, str(env.device)
    )
    teleop_interface.add_callback("R", reset_recording_instance)

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    label_text = build_label_text(current_recorded_demo_count, running_recording_instance)
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Get keyboard command
            action = teleop_interface.advance()
            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(
                env, success_term, success_step_count
            )
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = build_label_text(
                    current_recorded_demo_count, running_recording_instance
                )
                print(label_text)

            # Check if we've reached the desired number of demos
            if (
                args_cli.num_demos > 0
                and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
            ):
                label_text = (
                    f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                )
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(
                    env, success_step_count, instruction_display, label_text
                )
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Register local task configs (if present)
    local_task_root = os.path.join(os.path.dirname(__file__), "..", "tasks")
    register_local_tasks(local_task_root)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(
        f"Recording session completed with {current_recorded_demo_count} successful demonstrations"
    )
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
