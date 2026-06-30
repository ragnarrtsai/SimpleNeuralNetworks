# youbike-transformer

A small Transformer encoder that predicts each Taipei YouBike station's `available_rent_bikes` 30 minutes ahead, given the station's recent history and time-of-day. The model is trained to predict the **delta** from the current bike count (residual target); the absolute prediction is recovered as `current_bikes + model_output`.

## Architecture

```
seq (B, 7) — bike fractions at t-60min … t
       │
       ▼
Linear(1 → d_model)   +   PosEmbedding(7, d_model)
       │
       ▼
TransformerEncoder × num_layers (multi-head self-attention)
       │
       ▼
take last token (B, d_model)
       │
       ├── concat with static (B, 7): lat, lng, total, sin_h, cos_h, sin_dow, cos_dow
       ▼
Linear(d_model + 7 → 64) → GELU → Linear(64 → 1)
       │
       ▼
predicted delta (B,)
```

Defined in [`model.py`](model.py) (`YouBikeTransformer` class). Defaults: `d_model=64`, `nhead=4`, `num_layers=2`, `dim_feedforward=128`, `dropout=0.1`.

## Specs (latest released run)

> Numbers below are sourced from [`assets/metrics.json`](assets/metrics.json) — regenerate via `demo.py` and copy on each release.

| Item | Value |
|------|-------|
| Input | static (7) + sequence (7 × 1) |
| Output | one scalar (delta from current; absolute = current + delta) |
| Parameter count | TODO |
| Model file size | TODO |
| Test MAE (bikes) | **TODO** |
| Baseline MAE (predict-current) | TODO |
| Single-sample inference (CPU, Apple Silicon) | TODO µs |
| Released run ID | TODO |

## Environment

Uses the **shared venv at the repo root**. Activate before running:

```bash
source ../.venv/bin/activate
```

Required packages: `torch`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `ipykernel` (for the notebook).

## Data

YouBike historical archive — clone once into `input/`:

```bash
cd input
git clone --depth=1 https://github.com/tses89214/youbike-historical-data.git
```

One-time ~1.8 GB shallow clone covering 2024-05-03 → 2025-06-22 (416 days of 10-minute snapshots, ~1700 stations).

- `data/slots/<YYYY-MM-DD>.csv` — per-day station status (sno, total, available_rent_bikes, available_return_bikes, infoTime)
- `data/sites/<YYYY-MM-DD>.csv` — station metadata (sno, sna, lat, lng, district, …)

`input/` is gitignored; nothing from the dataset is committed.

For inference on **today's** state, the official open API still works without a token:

```
https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json
```

Schema matches the historical archive (same field names + `Quantity` corresponds to `total`).

## Run

Training is a notebook with tqdm progress bars + per-epoch loss table + training-results analysis:

```bash
jupyter lab train.ipynb        # or open in VSCode
```

Key knobs in the first config cell:

| Name | Default | Meaning |
|------|---------|---------|
| `N_DAYS` | 30 | how many recent days of slots to load |
| `LAG_STEPS` | 6 | past timesteps (× 10 min) → sequence length = 7 |
| `HORIZON_STEP` | 3 | predict this many steps (× 10 min) ahead |
| `SAMPLE_FRAC` | 0.05 | random subsample of windows (memory) |
| `TEST_FRAC` | 0.2 | last fraction of time range becomes test |
| `D_MODEL` | 64 | Transformer embedding dim |
| `NHEAD` | 4 | attention heads |
| `NUM_LAYERS` | 2 | encoder layers |
| `DIM_FF` | 128 | feed-forward dim inside each encoder layer |
| `DROPOUT` | 0.1 | |
| `EPOCHS` | 20 | training epochs |
| `BATCH_SIZE` | 1024 | |
| `LR` | 1e-3 | Adam learning rate, with `CosineAnnealingLR` schedule |

Each run captures `RUN_ID = yyyyMMdd_HHmmss` at the top and writes:

- `model/<RUN_ID>.pt` — trained weights
- `output/<RUN_ID>/history.json` — per-epoch `train_loss` / `test_mae` / `test_rmse` / `lr`

Re-evaluate later:

```bash
python eval.py                                       # uses ./latest.pt on last 7 days
python eval.py --weights model/20260514_113042.pt --days 14
```

Generate showcase artifacts:

```bash
python demo.py --weights model/<RUN_ID>.pt --run-id <RUN_ID>
```

This writes plots and `metrics.json` under `output/<RUN_ID>/showcase/`.

Device auto-detects CUDA → MPS → CPU. Override with `--device cpu`.

## Outputs

| Path | Tracked? | Contents |
|------|----------|----------|
| `latest.pt` | yes | the **released** model (manually copied from `model/`) |
| `assets/*.png`, `assets/metrics.json` | yes | the **released** showcase, manually copied from `output/<RUN_ID>/showcase/` |
| `model/<RUN_ID>.pt` | no | every training run's weights |
| `output/<RUN_ID>/history.json` | no | per-epoch metrics |
| `output/<RUN_ID>/showcase/` | no | plots + `metrics.json` from `demo.py` |
| `input/youbike-historical-data/` | no | cloned dataset |

## Hyperparameters

See the config cell at the top of `train.ipynb`. Loss is `SmoothL1Loss` (Huber); optimizer is `Adam`; scheduler is `CosineAnnealingLR(T_max=EPOCHS)`.

## Releasing a run

```bash
cp model/<RUN_ID>.pt latest.pt
python demo.py --weights latest.pt --run-id <RUN_ID>
cp output/<RUN_ID>/showcase/*.png assets/
cp output/<RUN_ID>/showcase/metrics.json assets/
# then update the "Specs" table above and OVERVIEW.md image references
```
