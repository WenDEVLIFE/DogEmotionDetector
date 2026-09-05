"""Reproducibility utilities for consistent random seeds."""

import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow.
    
    Args:
        seed: Seed value for random number generators.
        
    Note:
        This function sets seeds for reproducibility but does not guarantee
        completely deterministic results across different hardware/software
        configurations due to non-deterministic GPU operations.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set Python hash seed for reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Set TensorFlow deterministic operations (may impact performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # For TensorFlow 2.x, set eager execution to ensure reproducibility
    # (eager mode is default in TF 2.x)
    print(f"Random seeds set to {seed} for reproducibility.")