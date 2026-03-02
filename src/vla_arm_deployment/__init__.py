"""VLA Arm Deployment: Teleoperation Recording and VLA Model Deployment Toolkit.

This package provides tools for:
- Recording demonstrations via teleoperation
- Converting datasets to LeRobot format
- Training and deploying PI0.5 policies
"""

__version__ = "0.1.0"
__author__ = "VLA Arm Deployment Contributors"
__email__ = "your-email@example.com"

from . import utils

__all__ = ["utils", "__version__"]