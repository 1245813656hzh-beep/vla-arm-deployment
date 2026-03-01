#!/usr/bin/env python3
"""Convert IsaacLab HDF5 demonstration dataset to LeRobot v3.0 format.

Usage:
    python convert_to_lerobot.py \
        --input ./datasets/franka_place_bin.hdf5 \
        --output ./datasets/lerobot/franka_place_bin \
        --repo-id user/franka_place_bin \
        --task "pick up cubes and place them into the blue bin" \
        --fps 30 \
        --use-videos

    # Minimal (images stored in parquet, no video encoding):
    python convert_to_lerobot.py \
        --input ./datasets/franka_place_bin.hdf5 \
        --output ./datasets/lerobot/franka_place_bin \
        --repo-id user/franka_place_bin

    # Only convert successful episodes:
    python convert_to_lerobot.py \
        --input ./datasets/franka_place_bin.hdf5 \
        --output ./datasets/lerobot/franka_place_bin \
        --repo-id user/franka_place_bin \
        --success-only
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab HDF5 demos to LeRobot v3.0 dataset format."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to IsaacLab HDF5 dataset file."
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output root directory for the LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/franka_place_bin",
        help="HuggingFace-style repo ID (e.g., user/dataset_name). Default: local/franka_place_bin",
    )
    parser.add_argument(
        "--task",
        default="pick up cubes and place them into the blue bin",
        help="Task description string embedded in each frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency in Hz. Must match --step_hz used during recording. Default: 30",
    )
    parser.add_argument(
        "--robot-type",
        default="franka",
        help="Robot type string. Default: franka",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        default=False,
        help="Store images as mp4 video files (smaller, slower). "
        "Default: store as images in parquet (larger, faster).",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        default=False,
        help="Only convert episodes marked as success (E key). Default: convert all non-empty episodes.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of async image writer threads. Default: 4",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push the converted dataset to HuggingFace Hub after conversion.",
    )
    return parser.parse_args()


def discover_episodes(
    h5_data: h5py.Group, success_only: bool
) -> list[tuple[str, int, bool]]:
    """Find valid (non-empty) demo episodes in the HDF5 file.

    Returns:
        List of (demo_key, num_samples, success) tuples.
    """
    episodes = []
    for key in sorted(h5_data.keys()):
        if not key.startswith("demo_"):
            continue
        ep = h5_data[key]
        n = int(ep.attrs.get("num_samples", 0))
        success = bool(ep.attrs.get("success", False))
        if n == 0:
            continue
        if success_only and not success:
            continue
        episodes.append((key, n, success))
    return episodes


def discover_features(ep: h5py.Group) -> dict:
    """Auto-discover observation and action features from a sample episode.

    Returns:
        LeRobot features dict (without DEFAULT_FEATURES).
    """
    features = {}

    # --- Actions ---
    actions = ep["actions"]
    action_dim = actions.shape[1]
    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"][:action_dim]
    # Pad names if action_dim > 7
    while len(action_names) < action_dim:
        action_names.append(f"action_{len(action_names)}")

    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": action_names,
    }

    # --- State observations ---
    obs = ep["obs"]

    # EEF pose = eef_pos (3) + eef_quat (4) = 7-dim state
    if "eef_pos" in obs and "eef_quat" in obs:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }

    # Joint positions
    if "joint_pos" in obs:
        joint_dim = obs["joint_pos"].shape[1]
        features["observation.joint_pos"] = {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": [f"joint_{i}" for i in range(joint_dim)],
        }

    # Joint velocities
    if "joint_vel" in obs:
        jvel_dim = obs["joint_vel"].shape[1]
        features["observation.joint_vel"] = {
            "dtype": "float32",
            "shape": (jvel_dim,),
            "names": [f"joint_vel_{i}" for i in range(jvel_dim)],
        }

    # Gripper position
    if "gripper_pos" in obs:
        grip_dim = obs["gripper_pos"].shape[1]
        features["observation.gripper_pos"] = {
            "dtype": "float32",
            "shape": (grip_dim,),
            "names": [f"finger_{i}" for i in range(grip_dim)],
        }

    # --- Image observations ---
    image_keys = []
    for obs_key in obs.keys():
        ds = obs[obs_key]
        if isinstance(ds, h5py.Dataset) and ds.ndim == 4 and ds.dtype == np.uint8:
            image_keys.append(obs_key)

    for img_key in image_keys:
        ds = obs[img_key]
        h, w, c = ds.shape[1], ds.shape[2], ds.shape[3]
        features[f"observation.images.{img_key}"] = {
            "dtype": "video"
            if False
            else "image",  # will be overridden by --use-videos
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }

    return features, image_keys


def convert(args: argparse.Namespace) -> None:
    """Main conversion logic."""
    # Lazy import — lerobot may not be installed on all machines
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("ERROR: lerobot is not installed.")
        print("Install it with: pip install lerobot")
        print("Or from source: cd /path/to/lerobot && pip install -e .")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_root = Path(args.output)

    # Open HDF5 — try SWMR mode first (allows reading while Isaac Sim is still writing)
    try:
        h5f = h5py.File(str(input_path), "r", swmr=True, libver="latest")
    except Exception:
        h5f = h5py.File(str(input_path), "r")

    with h5f:
        h5_data = h5f["data"]

        # Discover episodes
        episodes = discover_episodes(h5_data, args.success_only)
        if not episodes:
            print("ERROR: No valid episodes found in the dataset.")
            if args.success_only:
                print("Hint: try without --success-only to include all episodes.")
            sys.exit(1)

        print(f"Found {len(episodes)} valid episodes:")
        total_frames = 0
        for demo_key, n, success in episodes:
            status = "success" if success else "no-success"
            print(f"  {demo_key}: {n} frames ({status})")
            total_frames += n
        print(f"Total frames to convert: {total_frames}")

        # Discover features from first episode
        first_ep = h5_data[episodes[0][0]]
        features, image_keys = discover_features(first_ep)

        # Override image dtype based on --use-videos flag
        for img_key in image_keys:
            feat_key = f"observation.images.{img_key}"
            features[feat_key]["dtype"] = "video" if args.use_videos else "image"

        print(f"\nFeatures discovered:")
        for feat_name, feat_info in features.items():
            print(
                f"  {feat_name}: shape={feat_info['shape']}, dtype={feat_info['dtype']}"
            )

        # Create LeRobot dataset
        has_images = len(image_keys) > 0
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=str(output_root),
            robot_type=args.robot_type,
            use_videos=args.use_videos and has_images,
            image_writer_threads=args.image_writer_threads if has_images else 0,
        )

        # Convert each episode
        for ep_idx, (demo_key, num_frames, success) in enumerate(episodes):
            ep = h5_data[demo_key]
            obs = ep["obs"]

            print(
                f"\nConverting {demo_key} ({num_frames} frames)...", end="", flush=True
            )

            # Pre-load arrays for speed
            actions_arr = ep["actions"][:].astype(np.float32)

            eef_pos_arr = (
                obs["eef_pos"][:].astype(np.float32) if "eef_pos" in obs else None
            )
            eef_quat_arr = (
                obs["eef_quat"][:].astype(np.float32) if "eef_quat" in obs else None
            )
            joint_pos_arr = (
                obs["joint_pos"][:].astype(np.float32) if "joint_pos" in obs else None
            )
            joint_vel_arr = (
                obs["joint_vel"][:].astype(np.float32) if "joint_vel" in obs else None
            )
            gripper_pos_arr = (
                obs["gripper_pos"][:].astype(np.float32)
                if "gripper_pos" in obs
                else None
            )

            # Pre-load images (may be large)
            image_arrays = {}
            for img_key in image_keys:
                image_arrays[img_key] = obs[
                    img_key
                ]  # Keep as HDF5 dataset, read per-frame

            for i in range(num_frames):
                frame = {
                    "action": actions_arr[i],
                    "task": args.task,
                }

                # EEF state (pos + quat concatenated)
                if eef_pos_arr is not None and eef_quat_arr is not None:
                    frame["observation.state"] = np.concatenate(
                        [eef_pos_arr[i], eef_quat_arr[i]], axis=0
                    )

                if joint_pos_arr is not None:
                    frame["observation.joint_pos"] = joint_pos_arr[i]

                if joint_vel_arr is not None:
                    frame["observation.joint_vel"] = joint_vel_arr[i]

                if gripper_pos_arr is not None:
                    frame["observation.gripper_pos"] = gripper_pos_arr[i]

                # Images — read per-frame from HDF5 (avoids loading all images into RAM)
                for img_key in image_keys:
                    # HDF5 stores (H, W, C) uint8, LeRobot expects the same for add_frame
                    frame[f"observation.images.{img_key}"] = image_arrays[img_key][i]

                dataset.add_frame(frame)

            dataset.save_episode()
            print(f" done (episode {ep_idx})")

        # Finalize — flush parquet writers, compute global stats
        print("\nFinalizing dataset (computing stats, encoding videos)...")
        dataset.finalize()

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Frames:   {total_frames}")
    print(f"  Output:   {output_root}")
    print(f"  Repo ID:  {args.repo_id}")
    print(f"{'=' * 60}")

    if args.push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        dataset.push_to_hub(tags=["sim", "isaac-sim", "franka", "manipulation"])
        print("Push complete!")


if __name__ == "__main__":
    convert(parse_args())
