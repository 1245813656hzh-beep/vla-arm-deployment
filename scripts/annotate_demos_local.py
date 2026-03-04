"""Wrapper for IsaacLab Mimic annotate_demos with local task registry.

This script registers local tasks in ../tasks before invoking IsaacLab's
annotate_demos.py, so custom environments (e.g., Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0)
are available in the Gym registry.
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

    # Ensure cameras are enabled for Mimic annotation
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    # Resolve IsaacLab annotate_demos script path
    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_script = (
        repo_root
        / "IsaacLab"
        / "scripts"
        / "imitation_learning"
        / "isaaclab_mimic"
        / "annotate_demos.py"
    )

    if not isaaclab_script.exists():
        raise FileNotFoundError(
            f"annotate_demos.py not found at {isaaclab_script}. Set your working directory to IsaacLab and use -p."
        )

    runpy.run_path(str(isaaclab_script), run_name="__main__")


if __name__ == "__main__":
    main()
