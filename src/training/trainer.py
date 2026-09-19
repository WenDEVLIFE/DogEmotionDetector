"""Training loop with callbacks for dog emotion detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

logger = logging.getLogger(__name__)


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Dict[str, Any],
    output_dir: str = "models",
) -> tf.keras.callbacks.History:
    """Train model with configured callbacks.

    Single-phase training with Adam optimizer and sparse categorical crossentropy.
    """
    training_cfg = config.get("training", {})
    epochs = training_cfg.get("epochs", 150)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 15)
    reduce_lr_patience = training_cfg.get("reduce_lr_patience", 7)
    reduce_lr_factor = training_cfg.get("reduce_lr_factor", 0.5)

    if train_ds is None:
        raise ValueError("Training dataset cannot be None")
    if val_ds is None:
        raise ValueError("Validation dataset cannot be None")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoint.keras"
    logger.info(f"Model checkpoint will be saved to: {checkpoint_path}")

    learning_rate = training_cfg.get("learning_rate", 0.001)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"Model compiled with Adam optimizer (lr={learning_rate})")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            verbose=1,
        ),
    ]

    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    logger.info(f"Reduce LR patience: {reduce_lr_patience}, factor: {reduce_lr_factor}")

    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Training completed successfully")
        return history

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e
