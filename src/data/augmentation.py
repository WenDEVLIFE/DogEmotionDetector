"""Data augmentation utilities for training data."""

from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation(config: Dict[str, Any] = None) -> keras.Sequential:
    """Build augmentation pipeline for training data.

    Applies: random horizontal flip, random rotation (±15°),
    random brightness (±0.15). Only applied to training set.
    """
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomRotation(factor=0.0417, seed=42),
        layers.RandomBrightness(factor=0.15, seed=42),
    ], name="data_augmentation")
    
    return augmentation
