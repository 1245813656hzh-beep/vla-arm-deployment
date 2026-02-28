#!/usr/bin/env python3
import argparse
from collections import defaultdict
from typing import Any

import h5py
import numpy as np


def summarize_dataset(ds: h5py.Dataset) -> str:
    shape = ds.shape
    dtype = ds.dtype
    chunks = ds.chunks
    return f"shape={shape}, dtype={dtype}, chunks={chunks}"


def safe_example(ds: h5py.Dataset) -> str:
    try:
        if ds.size == 0:
            return "empty"
        value = ds[0]
        if isinstance(value, np.ndarray):
            return f"example={value.flatten()[:5].tolist()}"
        return f"example={value}"
    except Exception as exc:
        return f"example=unavailable ({exc})"


def walk_group(group: h5py.Group, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else f"/{name}"
        if isinstance(item, h5py.Group):
            lines.append(f"GROUP {path}")
            lines.extend(walk_group(item, path))
        else:
            info = summarize_dataset(item)
            example = safe_example(item)
            lines.append(f"DATA  {path}  {info}  {example}")
    return lines


def find_episode_groups(root: h5py.File) -> list[h5py.Group]:
    episodes = []
    for name, item in root.items():
        if isinstance(item, h5py.Group) and name.startswith("episode"):
            episodes.append(item)
    return episodes


def summarize_episodes(root: h5py.File) -> list[str]:
    episodes = find_episode_groups(root)
    lines = [f"episodes_found={len(episodes)}"]
    for idx, ep in enumerate(episodes[:5]):
        episode_path = f"/{ep.name}"
        lines.append(f"episode[{idx}] path={episode_path}")
        lines.extend(walk_group(ep, episode_path))
    if len(episodes) > 5:
        lines.append("(only first 5 episodes shown)")
    return lines


def summarize_top_level(root: h5py.File) -> list[str]:
    lines = ["top_level:"]
    lines.extend(walk_group(root))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect IsaacLab demo dataset hdf5 contents."
    )
    parser.add_argument("dataset", help="Path to dataset hdf5 file")
    parser.add_argument("--mode", choices=["top", "episodes"], default="episodes")
    args = parser.parse_args()

    try:
        with h5py.File(args.dataset, "r") as root:
            if args.mode == "top":
                lines = summarize_top_level(root)
            else:
                lines = summarize_episodes(root)
            for line in lines:
                print(line)
    except OSError as exc:
        message = str(exc)
        if "truncated file" in message:
            print("ERROR: Dataset file looks truncated or incomplete.")
            print("- Make sure recording finished cleanly and the app is closed.")
            print("- Try a different dataset file or re-record a short demo.")
            print(
                "- If you still need this file, check its size and confirm it is not 0 bytes."
            )
        else:
            print(f"ERROR: Failed to open dataset: {message}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
