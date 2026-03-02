# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Deploy a LeRobot PI0.5 (pi05) policy in Isaac Sim for closed-loop inference.

This script loads a fine-tuned PI0.5 policy and runs it in a chosen IsaacLab task.
It uses image + state observations by default:
  - observation.state = [eef_pos(3), eef_quat(4)]
  - observation.images.table_cam
  - observation.images.wrist_cam

Example:
    ./isaaclab.sh -p ../vla-arm-deployment/deploy_vla.py \
      --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
      --policy_path /path/to/pi05/checkpoint/pretrained_model \
      --dataset_repo_id local/franka_place_bin \
      --dataset_root ../vla-arm-deployment/datasets/lerobot/franka_place_bin \
      --device cuda:0 \
      --enable_cameras
"""

# Standard library imports
import argparse
import contextlib
import sys
import time
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Deploy a LeRobot PI0.5 policy in Isaac Sim."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--policy_path",
    type=str,
    required=True,
    help="Path to a PI0.5 checkpoint directory (contains config.json + model.safetensors).",
)
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="local/franka_place_bin",
    help="LeRobot dataset repo_id used for stats and features.",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="./datasets/lerobot/franka_place_bin",
    help="Local root directory of the LeRobot dataset.",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="pick up cubes and place them into the blue bin",
    help="Task string fed to PI0.5 language input.",
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=30,
    help="Environment stepping rate in Hz.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Max steps before exit (0 = run until window closed).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Third-party imports
import gymnasium as gym
import logging
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.tasks.utils.parse_cfg import parse_env_cfg

logger = logging.getLogger(__name__)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def create_environment_config() -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as exc:
        logger.error(f"Failed to parse environment configuration: {exc}")
        sys.exit(1)

    # Disable recording/export
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # Keep episodes running until user closes the app
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Ensure policy observation terms are not concatenated
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False

    return env_cfg


def load_policy_and_processors():
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    except ImportError as exc:
        print("ERROR: lerobot is not installed or not on PYTHONPATH.")
        print("Install it with: pip install lerobot")
        print("Or from source: cd /path/to/lerobot && pip install -e .")
        raise exc

    dataset_root = Path(args_cli.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    policy_path = Path(args_cli.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")

    # Load dataset metadata (features + stats)
    dataset_meta = LeRobotDatasetMetadata(
        args_cli.dataset_repo_id, root=dataset_root, revision=None
    )
    if dataset_meta.stats is None:
        raise ValueError(
            "Dataset stats not found. Make sure meta/stats.json exists in the dataset root."
        )

    # Load policy config
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    if args_cli.device:
        policy_cfg.device = args_cli.device

    # Create policy
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)

    # Create processors (try loading from policy directory, fallback to fresh)
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=str(policy_path),
            dataset_stats=dataset_meta.stats,
        )
    except Exception:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg, pretrained_path=None, dataset_stats=dataset_meta.stats
        )

    return policy, preprocessor, postprocessor


def build_policy_batch(obs_policy: dict) -> dict:
    """Create a LeRobot batch dict from IsaacLab policy observations.

    Required keys:
      - observation.state
      - observation.images.table_cam
      - observation.images.wrist_cam
      - task
    """
    eef_pos = obs_policy["eef_pos"][0]
    eef_quat = obs_policy["eef_quat"][0]
    state = torch.cat([eef_pos, eef_quat], dim=0)

    def _to_image_tensor(img: torch.Tensor) -> torch.Tensor:
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.0
        elif img.dtype != torch.float32:
            img = img.to(torch.float32)
        return img

    table_cam = _to_image_tensor(obs_policy["table_cam"][0])
    wrist_cam = _to_image_tensor(obs_policy["wrist_cam"][0])

    return {
        "observation.state": state,
        "observation.images.table_cam": table_cam,
        "observation.images.wrist_cam": wrist_cam,
        "task": args_cli.task_description,
    }


def main() -> None:
    if not args_cli.enable_cameras:
        raise ValueError("--enable_cameras is required for image + state policy input.")

    env_cfg = create_environment_config()
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    policy, preprocessor, postprocessor = load_policy_and_processors()
    policy.eval()
    policy.reset()

    env.sim.reset()
    obs, *_ = env.reset()

    step_count = 0
    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            obs_policy = obs["policy"]
            batch = build_policy_batch(obs_policy)
            proc_batch = preprocessor(batch)

            action = policy.select_action(proc_batch)
            action = postprocessor(action)
            action = torch.clamp(action, -1.0, 1.0)

            # Convert to numpy for env.step
            action_np = action.cpu().numpy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]

            obs, reward, terminated, truncated, info = env.step(action_np)

            done = bool(terminated[0] or truncated[0])
            if done:
                env.sim.reset()
                obs, *_ = env.reset()
                policy.reset()

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    main()
