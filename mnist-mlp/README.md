# mnist-mlp

A simple multi-layer perceptron (MLP) for MNIST handwritten digit classification. Used as the project's "hello world" baseline before introducing convolutional architectures.

## Architecture

A 3-layer fully-connected network:

| Layer | Shape | Activation |
|-------|-------|------------|
| Flatten | `784` | — |
| Linear | `784 → 256` | ReLU |
| Linear | `256 → 128` | ReLU |
| Linear | `128 → 10` | (logits) |

Defined in [`model.py`](model.py) (`MLP` class).

## Specs (latest released run)

Numbers below are sourced from [`assets/metrics.json`](assets/metrics.json) — regenerate via `demo.py` and copy on each release.

| Item | Value |
|------|-------|
| Input | 28×28 grayscale image |
| Output | one of 0–9 |
| Parameter count | ~235K |
| Model file size | 0.92 MB |
| Test accuracy (10,000 samples) | **97.60%** |
| Single-sample inference (CPU, Apple Silicon) | 16.4 µs |
| Single-sample inference (MPS, Apple Silicon) | 99.2 µs |
| Released run ID | `20260509_115652` |

> Test environment: Python 3.14, PyTorch 2.11, macOS / Apple Silicon. MPS overhead exceeds compute for a model this small; CPU is faster here.

## Environment

Uses the **shared venv at the repo root**. Activate before running:

```bash
source ../.venv/bin/activate   # adjust path to wherever the shared venv lives
```

Required packages: `torch`, `torchvision` (training/eval) and `matplotlib`, `numpy` (showcase). All project-wide.

## Run

Training (downloads MNIST into `input/` on first run):

```bash
python train.py                            # defaults: 5 epochs, bs=64, lr=1e-3
python train.py --epochs 10 --lr 5e-4      # override
```

Each run is tagged with a timestamp `RUN_ID` of the form `yyyyMMdd_HHmmss` (start time). It writes:

- `model/<RUN_ID>.pt` — trained weights
- `output/<RUN_ID>/history.json` — per-epoch loss/accuracy

Device auto-detects in this order: CUDA → MPS (Apple Silicon) → CPU. Override with `--device cpu`.

Evaluation:

```bash
python eval.py                                    # uses ./latest.pt (the released model)
python eval.py --weights model/20260509_115652.pt # use a specific run's weights
```

Generate showcase artifacts (sample predictions, training curve, confusion matrix, inference timing → all under `output/<RUN_ID>/showcase/`):

```bash
python demo.py --weights model/<RUN_ID>.pt --run-id <RUN_ID>
python demo.py --weights model/<RUN_ID>.pt --run-id <RUN_ID> --device cpu   # for CPU latency
```

## Data

- MNIST is downloaded automatically by `torchvision` into `input/MNIST/`
- `input/` is gitignored — every clone re-downloads on first run

## Outputs

| Path | Tracked? | Contents |
|------|----------|----------|
| `latest.pt` | yes | the **released** model (manually copied from `model/`) |
| `assets/*.png`, `assets/metrics.json` | yes | the **released** showcase, manually copied from `output/<RUN_ID>/showcase/` |
| `model/<RUN_ID>.pt` | no | every training run's weights, named by start timestamp |
| `output/<RUN_ID>/history.json` | no | per-epoch `train_loss` / `test_acc` for that run |
| `output/<RUN_ID>/showcase/` | no | plots + `metrics.json` from `demo.py` |

## Hyperparameters

| Name | Default | Flag |
|------|---------|------|
| Epochs | 5 | `--epochs` |
| Batch size | 64 | `--batch-size` |
| Learning rate | 1e-3 | `--lr` |
| Optimizer | Adam | (hardcoded) |
| Loss | CrossEntropy | (hardcoded) |

## Releasing a run

Training and demo do **not** auto-promote anything. To publish a chosen run:

```bash
cp model/<RUN_ID>.pt latest.pt
cp output/<RUN_ID>/showcase/*.png assets/
cp output/<RUN_ID>/showcase/metrics.json assets/
# then update the "Specs" table above and OVERVIEW.md image references
```
