"""Dog Emotion Detector — main entry point.

Orchestrates the full pipeline:
    Load config → Set seeds → Setup GPU → Load data → Split →
    Augment → Build model → Train → Evaluate → Export TFLite → Benchmark

Usage:
    python main.py train --config configs/config.yaml
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dog_emotion_detector")


def cmd_train(args: argparse.Namespace) -> None:
    """Execute the full train → evaluate → export pipeline."""
    # ── 1. Load config ──────────────────────────────────────────────────
    from src.utils.config import load_config

    logger.info("Step 1/11: Loading configuration from %s", args.config)
    config = load_config(args.config)
    logger.info("Config loaded: seed=%s, epochs=%s, classes=%s",
                config.get("seed"),
                config.get("training", {}).get("epochs"),
                config.get("data", {}).get("class_names"))

    # ── 2. Set reproducibility seeds ───────────────────────────────────
    from src.utils.reproducibility import set_seeds

    seed = config.get("seed", 42)
    logger.info("Step 2/11: Setting random seeds to %d", seed)
    set_seeds(seed)

    # ── 3. Setup GPU ───────────────────────────────────────────────────
    from src.utils.gpu import setup_gpu

    logger.info("Step 3/11: Configuring GPU")
    setup_gpu()

    # ── 4. Load dataset ────────────────────────────────────────────────
    from src.data.loader import load_dataset

    data_cfg = config.get("data", {})
    root_dir = data_cfg.get("root", "datasets/dog_emotion_classification")
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    batch_size = data_cfg.get("batch_size", 32)

    logger.info("Step 4/11: Loading dataset from %s", root_dir)
    t0 = time.time()
    dataset = load_dataset(
        root_dir=root_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    logger.info("Dataset loaded in %.1fs — %d images, %d classes: %s",
                time.time() - t0,
                dataset.total_images,
                dataset.num_classes,
                dataset.class_names)

    # ── 5. Split dataset ───────────────────────────────────────────────
    from src.data.preprocessing import split_dataset

    splits = data_cfg.get("splits", [0.8, 0.1, 0.1])
    logger.info("Step 5/11: Splitting dataset (splits=%s)", splits)
    train_ds, val_ds, test_ds = split_dataset(dataset, splits=splits, seed=seed)
    logger.info("Splits — train: %d batches, val: %d batches, test: %d batches",
                train_ds.cardinality().numpy(),
                val_ds.cardinality().numpy(),
                test_ds.cardinality().numpy())

    # ── 6. Build and apply augmentation ─────────────────────────────────
    from src.data.augmentation import build_augmentation

    logger.info("Step 6/11: Building augmentation pipeline")
    augmentation = build_augmentation(config)
    logger.info("Augmentation layers: %s", [l.name for l in augmentation.layers])

    # Apply augmentation ONLY to training data (not val/test)
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    logger.info("Augmentation applied to training dataset")

    # ── 7. Build model ─────────────────────────────────────────────────
    from src.model.architecture import build_model

    logger.info("Step 7/11: Building MobileNetV2 transfer learning model")
    model = build_model(config)
    logger.info("Model built — %d total params, %d trainable",
                model.count_params(),
                sum(tf.size(v).numpy() for v in model.trainable_variables))

    # ── 8. Train model ─────────────────────────────────────────────────
    from src.training.trainer import train

    export_cfg = config.get("export", {})
    output_dir = export_cfg.get("output_dir", "models")

    logger.info("Step 8/11: Training model")
    t0 = time.time()
    history = train(model, train_ds, val_ds, config, output_dir=output_dir)
    train_time = time.time() - t0
    logger.info("Training completed in %.1fs — best epoch: %d",
                train_time,
                len(history.history.get("val_loss", [])))

    # ── 9. Evaluate model ──────────────────────────────────────────────
    from src.evaluation.metrics import evaluate

    class_names = data_cfg.get("class_names", [])
    logger.info("Step 9/11: Evaluating model on test set")
    metrics = evaluate(model, test_ds, class_names, threshold=0.60)
    logger.info("Evaluation — accuracy: %.4f, macro F1: %.4f",
                metrics["accuracy"], metrics["macro_f1"])

    # ── 10. Export TFLite ──────────────────────────────────────────────
    from src.export.converter import export_tflite

    quantize = export_cfg.get("quantize", False)
    tflite_path = str(Path(output_dir) / "model.tflite")

    logger.info("Step 10/11: Exporting TFLite model (quantize=%s)", quantize)
    export_tflite(model, tflite_path, quantize=quantize)

    # ── 11. Benchmark TFLite ───────────────────────────────────────────
    from src.export.converter import benchmark_tflite
    import numpy as np

    logger.info("Step 11/11: Benchmarking TFLite model")
    test_image = np.random.rand(1, image_size[0], image_size[1], 3).astype(np.float32)
    benchmark = benchmark_tflite(tflite_path, test_image, iterations=100)
    logger.info("Benchmark — avg latency: %.2fms, max latency: %.2fms",
                benchmark["avg_latency_ms"], benchmark["max_latency_ms"])

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Dataset:       {dataset.total_images} images, {dataset.num_classes} classes")
    print(f"  Model:         MobileNetV2 transfer learning")
    print(f"  Train time:    {train_time:.1f}s")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  TFLite export: {tflite_path}")
    print(f"  Benchmark:     {benchmark['avg_latency_ms']:.2f}ms avg")
    print("=" * 60 + "\n")


def main() -> None:
    """Parse CLI arguments and dispatch to subcommands."""
    parser = argparse.ArgumentParser(
        prog="main",
        description="Dog Emotion Detector — Detect emotions in dogs from images",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train subcommand ───────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train", help="Train, evaluate, and export the model"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "train": cmd_train,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Lazy import to keep --help fast
    import tensorflow as tf  # noqa: F401 — ensure TF is available before CLI runs
    main()
