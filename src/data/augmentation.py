"""Data augmentation utilities for training data."""

from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation(config: Dict[str, Any] = None) -> keras.Sequential:
    """Build augmentation pipeline for training data.

    Applies: random horizontal flip, random rotation (±20°),
    random brightness (±0.2). Only applied to training set.

    Args:
        config: Optional configuration dictionary. Currently unused but
                reserved for future augmentation parameters.

    Returns:
        tf.keras.Sequential model that can be called on image batches.
    """
    # Build augmentation pipeline
    augmentation = keras.Sequential([
        # Random horizontal flip with 50% probability
        layers.RandomFlip("horizontal", seed=42),
        
        # Random rotation within ±20° (converted to radians)
        layers.RandomRotation(factor=0.0556, seed=42),  # 20° = 0.0556 turns
        
        # Random brightness adjustment within ±0.2
        layers.RandomBrightness(factor=0.2, seed=42),
    ], name="data_augmentation")
    
    return augmentation