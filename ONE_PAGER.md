# Radar Human Activity Classification — One Pager

## Goal
Build a reproducible radar human activity classification pipeline that converts raw radar `.dat` files into three image representations and trains a three-stream CNN to recognise six activities.

## Implementation

### 1. Radar pre-processing
- `preprocessing.py` reads raw `.dat` files and parses MATLAB-style complex IQ values.
- The pipeline computes:
  - Range-Time image via MTI filtering on range-bin time series.
  - Range-Doppler image via Doppler FFT of MTI-filtered range bins.
  - Spectrogram image by summing STFT outputs across selected range bins.
- All images are normalised to `[0, 1]` and resized to `64x64`.

### 2. Dataset and labels
- `dataset.py` scans the dataset folders listed in `config.py`.
- Labels are extracted from the first character of each filename (`K=1..6`).
- The dataset is split into train/val/test sets with stratification per class.
- Each sample becomes a tensor of shape `[3, 64, 64]` with channels:
  1. Spectrogram
  2. Range-Time
  3. Range-Doppler

### 3. Model architecture
- `cnn_model.py` implements a three-stream CNN with late fusion.
- Each stream has four convolutional blocks and a fully connected head.
- The three stream embeddings are concatenated and passed through a classifier head.
- Output is a softmax logits vector over six activity classes.

### 4. Training and evaluation
- `train.py` implements the training loop, checkpoint saving, and periodic validation.
- `evaluate.py` computes loss, accuracy, precision, recall, F1, and confusion matrices.
- `utils.py` provides logging, plotting, checkpointing, and metric utilities.

## Reproducibility
- Central settings live in `config.py`.
- Default output path is `outputs/`; change it via environment variables.
- The training script saves the best checkpoint as `outputs/checkpoints/best_model.pth`.

## Deliverable status
- `preprocessing.py` is now isolated and documented for standalone use.
- The codebase includes a runnable training and evaluation pipeline.
- Documentation files are added for project summary and model details.

## Next steps
- Place the dataset under `RADAR_DATASET_ROOT`.
- Run `python train.py` to generate model weights.
- Run `python evaluate.py` to validate results and save confusion matrix figures.
