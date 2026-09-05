"""GPU setup and configuration utilities."""

import os
from typing import Optional

import tensorflow as tf


def setup_gpu(memory_growth: bool = True) -> None:
    """Configure GPU memory growth to prevent out-of-memory errors.
    
    Args:
        memory_growth: If True, enable memory growth for GPU devices.
            If False, TensorFlow will allocate all GPU memory at once.
            
    Note:
        If no GPU is available, this function does nothing and the system
        will use CPU for computations.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU devices found. Using CPU.")
        return
    
    print(f"Found {len(gpus)} GPU device(s): {[gpu.name for gpu in gpus]}")
    
    if memory_growth:
        # Enable memory growth for each GPU
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu.name}.")
            except RuntimeError as e:
                # Memory growth must be set before GPUs are initialized
                print(f"Could not set memory growth for {gpu.name}: {e}")
    else:
        # Set virtual memory limit (optional)
        # You can adjust this based on your GPU memory
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        # )
        print("Memory growth disabled. TensorFlow will allocate all GPU memory.")
    
    # Verify GPU configuration
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Logical GPU devices: {len(logical_gpus)}")