"""Wrapper for IsaacLab Mimic generate_dataset with local task registry.

This script registers local tasks in ../tasks before invoking IsaacLab's
generate_dataset.py, so custom environments are available in the Gym registry.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _register_local_tasks() -> None:
    """Import local task modules to register Gym environments."""
    project_root = Path(__file__).resolve().parents[1]
    task_root = project_root / "tasks"

    if task_root.is_dir():
        sys.path.insert(0, str(task_root))

        for module_name in ("franka", "ur10_gripper", "ur_10e"):
            try:
                __import__(module_name)
            except Exception as exc:  # pragma: no cover - best effort import
                print(f"[WARN] Failed to import local task module '{module_name}': {exc}")


def main() -> None:
    _register_local_tasks()

    # Ensure cameras are enabled for Mimic data generation
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    # Resolve IsaacLab generate_dataset script path
    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_script = (
        repo_root
        / "IsaacLab"
        / "scripts"
        / "imitation_learning"
        / "isaaclab_mimic"
        / "generate_dataset.py"
    )

    if not isaaclab_script.exists():
        raise FileNotFoundError(
            f"generate_dataset.py not found at {isaaclab_script}. Set your working directory to IsaacLab and use -p."
        )

    runpy.run_path(str(isaaclab_script), run_name="__main__")


if __name__ == "__main__":
    main()
