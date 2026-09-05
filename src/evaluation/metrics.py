"""Model evaluation and metrics for dog emotion detection."""

from __future__ import annotations

import logging
import sys
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def evaluate(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    threshold: float = 0.60,
) -> Dict[str, object]:
    """Compute per-class metrics and macro F1.

    Runs the model on the test dataset, collects predictions and true labels,
    then computes accuracy, per-class F1, macro F1, and confusion matrix.
    Raises SystemExit(1) if macro F1 is below the threshold.

    Args:
        model: Trained tf.keras.Model.
        test_ds: Test tf.data.Dataset yielding (image_batch, label_batch).
        class_names: List of class names for the confusion matrix.
        threshold: Minimum macro F1 to pass. Defaults to 0.60.

    Returns:
        Dict with keys: macro_f1, per_class_f1, confusion_matrix, accuracy

    Raises:
        SystemExit(1): If macro_f1 < threshold.
    """
    # ── Collect predictions and true labels ────────────────────────────
    logger.info("Running model predictions on test set...")
    y_pred_probs = model.predict(test_ds, verbose=0)  # (N, num_classes)

    y_true = []
    for _images, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true, axis=0)

    # Convert to class indices (int32 for sklearn compatibility)
    y_pred = np.argmax(y_pred_probs, axis=1).astype(np.int32)  # (N,)
    y_true = (
        np.argmax(y_true, axis=1).astype(np.int32)
        if y_true.ndim > 1
        else y_true.astype(np.int32)
    )

    logger.info(f"Evaluated {len(y_true)} samples")

    # ── Accuracy ───────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)

    # ── Per-class F1 ──────────────────────────────────────────────────
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # ── Macro F1 ──────────────────────────────────────────────────────
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # ── Print results ──────────────────────────────────────────────────
    _print_results(accuracy, macro_f1, f1_per_class, cm, class_names)

    # ── Threshold check ───────────────────────────────────────────────
    if macro_f1 < threshold:
        logger.error(
            f"Macro F1 ({macro_f1:.4f}) is below threshold ({threshold:.2f}). "
            f"Failing evaluation."
        )
        sys.exit(1)

    logger.info(f"Macro F1 ({macro_f1:.4f}) meets threshold ({threshold:.2f}). Passing.")

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {name: float(f1) for name, f1 in zip(class_names, f1_per_class)},
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
    }


def _print_results(
    accuracy: float,
    macro_f1: float,
    per_class_f1: np.ndarray,
    cm: np.ndarray,
    class_names: List[str],
) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print("-" * 50)
    print("  Per-class F1:")
    for name, f1 in zip(class_names, per_class_f1):
        print(f"    {name:<12s} {f1:.4f}")
    print("-" * 50)
    print("  Confusion Matrix (rows=true, cols=pred):")
    header = "  {:>12s}".format("") + "".join(f"{name:>12s}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  {:>12s}".format(name) + "".join(f"{cm[i, j]:>12d}" for j in range(len(class_names)))
        print(row)
    print("=" * 50 + "\n")
