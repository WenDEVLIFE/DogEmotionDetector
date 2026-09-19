"""Transfer learning model for dog emotion classification."""

from __future__ import annotations

import tensorflow as tf

BACKBONES = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
}


def _preprocess_inputs(inputs: tf.Tensor, backbone_name: str) -> tf.Tensor:
    """Apply backbone-specific input preprocessing."""
    if backbone_name == "EfficientNetV2B0":
        return tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    # MobileNetV2 expects [-1, 1]
    return inputs / 127.5 - 1.0


def build_model(config: dict) -> tf.keras.Model:
    """Build transfer learning model with configurable backbone.

    Architecture:
        Backbone(ImageNet weights, include_top=False)
        -> GlobalAveragePooling2D
        -> Dense(256, relu)
        -> Dropout
        -> Dense(5, softmax)
    """
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    backbone_name = model_cfg["base_model"]
    frozen_layers = model_cfg["frozen_layers"]
    dense_units = model_cfg["dense_units"]
    dropout = model_cfg["dropout"]
    weight_decay = model_cfg.get("weight_decay", 0.0)
    num_classes = data_cfg["num_classes"]
    learning_rate = training_cfg["learning_rate"]
    image_size = data_cfg["image_size"]
    label_smoothing = training_cfg.get("label_smoothing", 0.0)

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    backbone_cls = BACKBONES[backbone_name]
    base_model = backbone_cls(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = True
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False

    x = _preprocess_inputs(inputs, backbone_name)
    x = base_model(x, training=True)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    return model
