"""Dataset utility functions."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_hdf5_dataset(file_path: str) -> h5py.File:
    """Load an HDF5 dataset file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        h5py.File object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file is corrupted
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        return h5py.File(file_path, 'r')
    except OSError as e:
        raise OSError(f"Failed to open HDF5 file (may be corrupted): {e}")


def analyze_dataset_structure(file_path: str) -> Dict[str, Any]:
    """Analyze the structure of an HDF5 dataset.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing dataset information
    """
    with load_hdf5_dataset(file_path) as f:
        info = {
            "episodes": [],
            "total_episodes": 0,
            "total_steps": 0,
        }
        
        if 'data' in f:
            episodes = list(f['data'].keys())
            info["episodes"] = episodes
            info["total_episodes"] = len(episodes)
            
            for ep_name in episodes:
                ep_data = f['data'][ep_name]
                if 'actions' in ep_data:
                    info["total_steps"] += ep_data['actions'].shape[0]
                    
    return info


def get_episode_count(file_path: str) -> int:
    """Get the number of episodes in a dataset.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Number of episodes
    """
    info = analyze_dataset_structure(file_path)
    return info["total_episodes"]


def validate_dataset(file_path: str) -> bool:
    """Validate that an HDF5 dataset is not corrupted.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with load_hdf5_dataset(file_path) as f:
            # Check for required keys
            if 'data' not in f:
                return False
            
            # Check that episodes have required fields
            for ep_name in f['data'].keys():
                ep = f['data'][ep_name]
                if 'actions' not in ep:
                    return False
                    
        return True
    except Exception:
        return False