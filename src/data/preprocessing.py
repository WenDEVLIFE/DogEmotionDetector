"""Data preprocessing utilities for stratified splitting."""

import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: tf.data.Dataset,
    splits: List[float] = None,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Stratified train/val/test split preserving class proportions.

    Collects all images and labels from the already-loaded dataset into
    memory, uses sklearn train_test_split for index-level stratification,
    then rebuilds tf.data.Dataset objects from the split arrays.

    Args:
        dataset: tf.data.Dataset yielding (image_batch, label_batch) tuples.
                 Typically returned by load_dataset().
        splits: List of three floats summing to 1.0 [train, val, test].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_ds, val_ds, test_ds) tf.data.Dataset objects,
        each yielding (image_batch, label_batch).

    Raises:
        ValueError: If splits don't sum to 1.0 or dataset is empty.
    """
    if splits is None:
        splits = [0.8, 0.1, 0.1]

    # ── Validate splits ──────────────────────────────────────────────
    if len(splits) != 3:
        raise ValueError(f"Splits must have exactly 3 elements, got {len(splits)}")

    total = sum(splits)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    # ── Collect all images and labels into NumPy arrays ──────────────
    logger.info("Collecting images from dataset into memory…")
    all_images = []
    all_labels = []
    for images, labels in dataset.unbatch().batch(1):
        all_images.append(images.numpy().squeeze(0))
        all_labels.append(labels.numpy().item())

    if not all_images:
        raise ValueError("Dataset is empty — nothing to split.")

    all_images = np.array(all_images)  # (N, H, W, 3)
    all_labels = np.array(all_labels)  # (N,)
    logger.info(f"Collected {len(all_labels)} images, shape {all_images.shape}")

    # ── Stratified split using sklearn ───────────────────────────────
    indices = np.arange(len(all_labels))
    val_test_ratio = splits[1] + splits[2]

    idx_train, idx_val_test = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=seed,
    )

    relative_val_ratio = splits[1] / val_test_ratio
    val_test_labels = all_labels[idx_val_test]

    # Guard: if any class has fewer than 2 samples in val_test set,
    # fall back to non-stratified split for the second split.
    unique, counts = np.unique(val_test_labels, return_counts=True)
    min_class_count = counts.min() if len(counts) > 0 else 0
    can_stratify = min_class_count >= 2

    idx_val, idx_test = train_test_split(
        idx_val_test,
        test_size=1 - relative_val_ratio,
        stratify=val_test_labels if can_stratify else None,
        random_state=seed,
    )

    logger.info(
        f"Split sizes — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}"
    )

    # ── Determine batch size from the original dataset ───────────────
    batch_size = 32
    if hasattr(dataset, "_batch_size"):
        try:
            batch_size = int(dataset._batch_size.numpy())
        except Exception:
            pass

    # ── Helper: build a tf.data.Dataset from index array ─────────────
    def _create_dataset(sub_indices: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        """Build a batched, prefetched dataset from a subset of indices."""
        sub_images = tf.constant(all_images[sub_indices], dtype=tf.float32)
        sub_labels = tf.constant(all_labels[sub_indices], dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((sub_images, sub_labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(sub_indices), 1000), seed=seed)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _create_dataset(idx_train, shuffle=True)
    val_ds = _create_dataset(idx_val, shuffle=False)
    test_ds = _create_dataset(idx_test, shuffle=False)

    return train_ds, val_ds, test_ds
