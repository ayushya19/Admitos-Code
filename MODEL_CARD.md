# Model Card — Radar Activity Classifier

## Model name
Three-Stream Radar Activity Classifier

## Purpose
Classify six human activities from radar micro-Doppler data using three complementary radar representations.

## Inputs
- Tensor of shape `[B, 3, 64, 64]`
- Channels:
  - `0` → Spectrogram
  - `1` → Range-Time
  - `2` → Range-Doppler

## Outputs
- Logits vector of shape `[B, 6]`
- Class mapping:
  - `0` Walking
  - `1` Sitting down
  - `2` Standing up
  - `3` Pick up object
  - `4` Drink water
  - `5` Fall

## Architecture
- Three separate 4-block StreamCNN encoders.
- Each stream performs:
  - 2 × Conv-BN-ReLU
  - MaxPool
  - Repeat across 4 blocks
  - AdaptiveAvgPool
  - Dense projection to 256-d feature vector
- Late fusion by concatenating stream outputs into `768-d` vector.
- Classifier head: 768 → 384 → 192 → 6.

## Training
- Loss: weighted cross entropy
- Optimizer: Adam
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: `32`
- Epochs: `50`
- Scheduler: StepLR with `step_size=10`, `gamma=0.5`
- Validation every `5` epochs

## Data split
- Train: `80%`
- Validation: `10%` of train
- Test: remaining `20%`

## Checkpoints and outputs
- Checkpoints: `outputs/checkpoints/best_model.pth`
- Logs: `outputs/logs/train.log`, `outputs/logs/evaluate.log`
- Results: `outputs/results/` with confusion matrices and training curves

## Usage
- Train: `python train.py`
- Evaluate: `python evaluate.py`

## Notes
- `preprocessing.py` is the recommended module for standalone radar pre-processing and can be used for debugging or generating visualisations outside the PyTorch dataset.
