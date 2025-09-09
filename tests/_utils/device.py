"""Device utilities for tests."""

import torch


def select_most_accurate_gpu_device() -> str:
    """Select the most accurate GPU device available.
    
    Returns:
        str: Device string - "mps" if available on Apple Silicon, 
             otherwise "cuda" for NVIDIA GPUs.
    """
    return "mps" if torch.backends.mps.is_available() else "cuda"
