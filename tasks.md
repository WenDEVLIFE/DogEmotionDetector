# DogEmotionDetector - Implementation Tasks

## Phase 1: Foundation

- [x] 1.1 Create `pyproject.toml` — project metadata, dependencies, tool configs
- [x] 1.2 Create `configs/config.yaml` — default configuration
- [x] 1.3 Create `src/__init__.py` — package root
- [x] 1.4 Create `src/utils/__init__.py` — utils package
- [x] 1.5 Create `src/utils/config.py` — config loading
- [x] 1.6 Create `src/utils/reproducibility.py` — seed management
- [x] 1.7 Create `src/utils/gpu.py` — GPU setup
- [x] 1.8 Create `src/data/__init__.py` — data package (empty for now)
- [x] 1.9 Create `src/model/__init__.py` — model package (empty for now)
- [x] 1.10 Create `src/training/__init__.py` — training package (empty for now)
- [x] 1.11 Create `src/evaluation/__init__.py` — evaluation package (empty for now)
- [x] 1.12 Create `src/export/__init__.py` — export package (empty for now)
- [x] 1.13 Create `src/cli.py` — CLI with --config flag

## Phase 2: Data Pipeline

- [x] 2.1 Create `src/data/loader.py` — PIL-based load_dataset function
- [x] 2.2 Create `src/data/preprocessing.py` — stratified split_dataset function
- [x] 2.3 Create `src/data/augmentation.py` — build_augmentation function
- [x] 2.4 Update `src/data/__init__.py` — export public API

## Phase 3: Model Architecture

- [x] 3.1 Create `src/model/architecture.py` — MobileNetV2 transfer learning model

## Phase 4: Training Loop

- [x] 4.1 Create `src/training/trainer.py` — train function with callbacks
- [x] 4.2 Update `src/training/__init__.py` — export train function

## Phase 5: Evaluation

- [x] 5.1 Create `src/evaluation/metrics.py` — per-class metrics, macro F1, confusion matrix
- [x] 5.2 Update `src/evaluation/__init__.py` — export evaluate function

## Phase 6: TFLite Export

- [x] 6.1 Create `src/export/converter.py` — export_tflite and benchmark_tflite functions
- [x] 6.2 Update `src/export/__init__.py` — export public API

## Phase 7: Integration

- [x] 7.1 Rewrite `main.py` — full pipeline orchestration with CLI subcommands