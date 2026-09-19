# Dog Emotion Detector

Detect emotions in dogs from images using transfer learning with TensorFlow.

Classifies dog images into 5 emotion categories: **alert**, **angry**, **frown**, **happy**, **relax**.

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 69.9% |
| Macro F1 | 69.2% |
| Model size | 24 MB |
| Inference time | ~22ms (CPU) |

**Per-class F1:**
- happy: 0.89
- relax: 0.79
- frown: 0.61
- angry: 0.53
- alert: 0.58

## Installation

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Train the model

```bash
python main.py train --config configs/config.yaml
```

This runs the full pipeline: load data → split → augment → build model → train → evaluate → export TFLite → benchmark.

### Configuration

Edit `configs/config.yaml` to adjust training parameters:

```yaml
model:
  base_model: EfficientNetV2B0
  frozen_layers: 120
  dense_units: 256
  dropout: 0.3
  weight_decay: 0.01

training:
  epochs: 150
  learning_rate: 0.0003
  label_smoothing: 0.1
  early_stopping_patience: 20
```

## Project Structure

```
DogEmotionDetector/
├── configs/
│   └── config.yaml          # Training configuration
├── datasets/
│   └── dog_emotion_classification/
│       ├── alert/           # 1,865 images
│       ├── angry/           # 1,865 images
│       ├── frown/           # 1,865 images
│       ├── happy/           # 1,865 images
│       └── relax/           # 1,865 images
├── models/
│   ├── model.tflite         # Exported TFLite model
│   └── checkpoint.keras     # Training checkpoint
├── src/
│   ├── data/
│   │   ├── augmentation.py  # Data augmentation pipeline
│   │   ├── loader.py        # PIL-based image loading
│   │   └── preprocessing.py # Train/val/test splitting
│   ├── evaluation/
│   │   └── metrics.py       # F1, accuracy, confusion matrix
│   ├── export/
│   │   └── converter.py     # TFLite conversion & benchmarking
│   ├── model/
│   │   └── architecture.py  # Multi-backbone transfer learning
│   ├── training/
│   │   └── trainer.py       # Training loop with callbacks
│   └── utils/
│       ├── config.py        # YAML config loading
│       ├── gpu.py           # GPU setup (XLA, CUDA)
│       └── reproducibility.py # Seed setting
├── main.py                  # CLI entry point
└── pyproject.toml           # Package metadata
```

## Dataset

Uses the [Dog Emotions - 5 Classes](https://www.kaggle.com/datasets/dougandrade/dog-emotions-5-classes) dataset by Doug Andrade with 9,325 images across 5 classes (~1,865 each).

## Architecture

- **Backbone:** EfficientNetV2B0 (ImageNet pretrained)
- **Head:** GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(5, softmax)
- **Optimization:** Adam optimizer, label smoothing (0.1), early stopping, reduce-LR-on-plateau

## Export & Deployment

The trained model is exported to TFLite format for mobile deployment:

```bash
# Model is exported automatically during training
# Output: models/model.tflite (24 MB)
```

Benchmark results (100 iterations, CPU):
- Average latency: ~22ms
- Max latency: ~28ms

## License

MIT
