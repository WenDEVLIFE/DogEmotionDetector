"""TFLite conversion and benchmarking for dog emotion detection."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def export_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantize: bool = False,
) -> str:
    """Convert Keras model to TFLite format.

    Args:
        model: Trained Keras model.
        output_path: Path for .tflite file.
        quantize: If True, apply dynamic range quantization.

    Returns:
        Path to exported .tflite file.

    Raises:
        RuntimeError: If conversion fails.
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting Keras model to TFLite (quantize={quantize})...")

    try:
        # Create converter from the Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply dynamic range quantization if requested
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Dynamic range quantization enabled")

        # Convert the model
        tflite_model = converter.convert()

        # Write the TFLite model to file
        Path(output_path).write_bytes(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        logger.info(f"TFLite model exported to {output_path} ({size_mb:.2f} MB)")
        return output_path

    except Exception as e:
        raise RuntimeError(f"TFLite conversion failed: {e}") from e


def benchmark_tflite(
    tflite_path: str,
    test_image: np.ndarray,
    iterations: int = 100,
) -> Dict[str, object]:
    """Benchmark TFLite model inference speed.

    Runs the TFLite model on the provided test image for the specified
    number of iterations and reports latency statistics.

    Args:
        tflite_path: Path to the .tflite model file.
        test_image: Input image array with shape (1, H, W, 3), dtype float32.
        iterations: Number of inference iterations. Defaults to 100.

    Returns:
        Dict with keys: prediction (np.ndarray), avg_latency_ms (float),
        max_latency_ms (float)

    Raises:
        RuntimeError: If benchmark fails.
        FileNotFoundError: If tflite_path does not exist.
    """
    if not Path(tflite_path).exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    logger.info(f"Benchmarking TFLite model: {tflite_path} ({iterations} iterations)")

    try:
        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure test image matches expected input shape and dtype
        expected_shape = input_details[0]["shape"]
        if test_image.shape != tuple(expected_shape):
            test_image = np.resize(test_image, expected_shape).astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], test_image)

        # Warm up run
        interpreter.invoke()

        # Benchmark inference
        latencies = []
        for _ in range(iterations):
            start = time.time()
            interpreter.invoke()
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Get final prediction
        prediction = interpreter.get_tensor(output_details[0]["index"])

        avg_latency_ms = float(np.mean(latencies))
        max_latency_ms = float(np.max(latencies))

        logger.info(
            f"Benchmark complete: avg={avg_latency_ms:.2f}ms, max={max_latency_ms:.2f}ms"
        )

        return {
            "prediction": prediction,
            "avg_latency_ms": avg_latency_ms,
            "max_latency_ms": max_latency_ms,
        }

    except Exception as e:
        raise RuntimeError(f"TFLite benchmark failed: {e}") from e
