"""MobileNetV2 transfer learning model for dog emotion classification."""

from __future__ import annotations

import tensorflow as tf


def build_model(config: dict) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model.

    Architecture:
        MobileNetV2(ImageNet weights, include_top=False)
        -> GlobalAveragePooling2D
        -> Dense(128, relu)
        -> Dropout(0.3)
        -> Dense(5, softmax)

    Args:
        config: Dict with keys: base_model, frozen_layers, dense_units, dropout, num_classes

    Returns:
        Compiled tf.keras.Model
    """
    # Extract config values
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    base_model_name = model_cfg["base_model"]
    frozen_layers = model_cfg["frozen_layers"]
    dense_units = model_cfg["dense_units"]
    dropout = model_cfg["dropout"]
    num_classes = data_cfg["num_classes"]
    learning_rate = training_cfg["learning_rate"]
    image_size = data_cfg["image_size"]

    # Input layer
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    # Base model: MobileNetV2 with ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze the first N layers
    base_model.trainable = True
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False

    # Forward pass through base model (training=True so BN uses batch stats)
    x = base_model(inputs, training=True)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
