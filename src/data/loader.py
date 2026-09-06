"""Data loading utilities for dog emotion detection.

Uses PIL/Pillow for image reading (via tf.py_function) to ensure
cross-platform compatibility, including Windows Python accessing
WSL filesystem paths where TensorFlow's C++ file backend fails.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def load_dataset(
    root_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Load images from class-organized directories.

    Uses PIL for image reading (via tf.py_function) to work across
    platforms, including Windows Python accessing WSL filesystems.

    Args:
        root_dir: Path to directory containing class subdirectories.
        image_size: (height, width) tuple, default (224, 224).
        batch_size: Number of images per batch.

    Returns:
        tf.data.Dataset yielding (image_batch, label_batch).

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If any class directory is empty.
    """
    root_path = Path(root_dir)

    # Validate root directory
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {root_dir}")

    # Resolve to absolute path for cross-platform PIL compatibility
    root_abs = str(root_path.resolve())
    logger.info(f"Dataset root (absolute): {root_abs}")

    # Discover class subdirectories (sorted for deterministic ordering)
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found in {root_dir}")

    class_names = [d.name for d in class_dirs]
    logger.info(f"Found classes: {class_names}")

    # Validate non-empty directories and collect absolute file paths + labels
    all_paths: list[str] = []
    all_labels: list[int] = []

    for class_idx, class_dir in enumerate(class_dirs):
        image_files = sorted(
            str(f.resolve())
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_files:
            raise ValueError(f"Empty class directory: {class_dir.name}")
        logger.info(f"  {class_dir.name}: {len(image_files)} images")
        all_paths.extend(image_files)
        all_labels.extend([class_idx] * len(image_files))

    if not all_paths:
        raise ValueError("No images found in any class directory.")

    total = len(all_paths)
    logger.info(f"Total images: {total}")

    # Build tf.data.Dataset from file paths and labels
    paths_tensor = tf.constant(all_paths, dtype=tf.string)
    labels_tensor = tf.constant(all_labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    # --- Image loading function using PIL via tf.py_function -----------
    h, w = image_size

    def _load_image_pil(path: tf.Tensor, label: tf.Tensor):
        """Read an image with PIL and return (float32_tensor, label)."""
        def _read(path_tensor):
            # Extract actual string from EagerTensor
            path_str = path_tensor.numpy().decode("utf-8")
            try:
                img = Image.open(path_str).convert("RGB")
                img = img.resize((w, h), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
                # MobileNetV2 expects [-1, 1] input range
                arr = (arr / 127.5) - 1.0
                return arr
            except Exception as exc:
                warnings.warn(f"Skipping corrupted image {path_str}: {exc}")
                return np.zeros((h, w, 3), dtype=np.float32)

        img = tf.py_function(_read, [path], tf.float32)
        img.set_shape((h, w, 3))
        return img, label

    # Map loading across parallel threads
    dataset = dataset.map(_load_image_pil, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter out corrupted images (all-zero sentinel)
    dataset = dataset.filter(
        lambda img, lbl: tf.reduce_sum(tf.abs(img)) > 0.0
    )

    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=min(total, 10000), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Attach metadata for downstream use
    dataset.class_names = class_names
    dataset.num_classes = len(class_names)
    dataset.total_images = total

    return dataset
