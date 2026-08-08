# Privacy-Preserving Federated Intrusion Detection System for Simulated Industrial IoT (CNN-GRU)

A federated intrusion detection system for Industrial IoT network traffic. A CNN-GRU
classifier is trained two ways on the Edge-IIoTset dataset — once centrally as a
baseline, and once through a simulated federated learning setup where raw traffic never
leaves the client — and the two are compared on identical held-out data. SHAP
attributions explain individual predictions, and a read-only Streamlit dashboard
presents every artifact the pipeline produces.

The design is frozen in `project-docs/SDS.md`. This README documents the system as it
actually exists on disk, verified against the real file tree.

## What the system does

- **Preprocessing** — loads the raw Edge-IIoTset CSV, drops identifier and leakage-prone
  columns, derives multi-class and binary labels, removes exact duplicate rows, splits
  train/test once, then fits every encoder and the scaler on the **training partition
  only** before transforming the full dataset.
- **Centralized baseline** — trains the CNN-GRU on the entire training split with class
  weighting, early stopping, and best-weight checkpointing.
- **Federated simulation** — shards the training split IID across virtual clients, runs
  FedAvg through Flower's simulation backend, and captures the aggregated global model
  from the strategy rather than from any single client.
- **Evaluation** — scores both models on the same held-out test set and writes a
  side-by-side comparison, confusion matrices, classification reports, convergence and
  training-curve plots, and a model-size table.
- **Explainability** — computes SHAP GradientExplainer attributions over a persisted
  background sample and writes summary and bar plots.
- **Dashboard** — four read-only Streamlit tabs (Project Overview, Predictions, FL vs.
  Centralized, Explainability). The dashboard performs no training.

## Architecture

Single CNN-GRU model definition, used unchanged by both the centralized and federated
paths (`src/models/cnn_gru.py`):

```
Input(num_features, 1)
  -> Conv1D(64, kernel_size=3, padding="same", activation="relu")
  -> MaxPooling1D(pool_size=2)
  -> GRU(64, return_sequences=False)
  -> Dropout(0.3)
  -> Dense(128, activation="relu")
  -> Dense(num_classes, activation="softmax")
```

Compiled with Adam (`learning_rate=0.001`) and `categorical_crossentropy`.
`num_classes` is always derived at runtime from `outputs/artifacts/class_mapping.json`,
never hardcoded. `padding="same"` and `return_sequences=False` are locked values.

## Requirements

- Python 3.10
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- `configs/config.yaml` selects a Parquet processed-data file by default, which requires
  a Parquet engine:

  ```bash
  pip install pyarrow
  ```

  `pyarrow` is not currently pinned in `requirements.txt`. Either install it as above, or
  set `paths.processed_data_file` to `edge_iiotset_processed.csv` to use the CSV path with
  no extra dependency.

- Architecture-diagram export (`*_architecture.png`) additionally needs `pydot` and a
  system Graphviz install. This step is best-effort: if either is missing, the pipeline
  logs a warning and continues.

## Dataset

Place the raw Edge-IIoTset CSV at:

```
data/raw/edge_iiotset.csv
```

The filename comes from `paths.raw_data_file` in `configs/config.yaml`. This file is
user-supplied and is never written by any script.

## Run order

Run every command from the project root. Each script sets the global seed first, then
creates one timestamped log file under `logs/`.

```bash
# 1. Preprocess: encode, scale, split, and persist all artifacts
python experiments/run_preprocessing.py

# 2. Centralized baseline
python experiments/run_centralized.py

# 3. Federated simulation (Flower + FedAvg)
python experiments/run_federated.py

# 4. Evaluate and compare both models on the shared test set
python experiments/run_evaluation.py

# 5. SHAP explainability
python experiments/run_explainability.py

# 6. Dashboard (read-only; requires the artifacts above)
streamlit run dashboard/app.py
```

Steps 2 and 3 both depend only on step 1, so their order relative to each other does not
matter. Step 4 requires both. Steps 4 and 5 must precede step 6 for all four tabs to
render data.

### Run profiles

`src/utils/config_loader.py` resolves a run mode from the `IIOT_MODE` environment
variable, defaulting to `FULL_EXPERIMENT` (`configs/config.yaml` used as-is).

```bash
# Reduced-memory profile for a 16 GB Windows laptop
set IIOT_MODE=LAPTOP_MODE      # Windows cmd
python experiments/run_federated.py
```

`LAPTOP_MODE` overlays `configs/config_laptop.yaml`, which changes only the federated
settings, `training.batch_size`, and Ray runtime limits. Preprocessing, the model
architecture, and the evaluation pipeline are identical across profiles.

## Configuration

`configs/config.yaml` is the single source of truth. No script hardcodes a value that
belongs in config. Top-level sections: `seed`, `paths`, `dataset`, `model`, `training`,
`federated`, `explainability`, `dashboard`.

Values used for the results currently in `outputs/`:

| Setting | Value |
|---|---|
| `seed` | 42 |
| `dataset.test_size` | 0.2 |
| `dataset.drop_duplicates` | true |
| `training.batch_size` | 128 |
| `training.centralized_epochs` | 10 |
| `training.use_class_weights` | true |
| `federated.num_clients` | 2 |
| `federated.num_rounds` | 5 |
| `federated.partition_strategy` | iid |
| `explainability.background_size` | 100 |

Note that `configs/config.yaml` currently sets `num_clients: 2` and `num_rounds: 5`,
while SDS Section 8 specifies 4 clients and 15 rounds. The committed federated artifacts
reflect the 2-client / 5-round run.

## Repository layout

```
iiot-fl-ids-project/
├── README.md
├── requirements.txt
├── requirements_colab.txt
├── configs/
│   ├── config.yaml                  # production config (FULL_EXPERIMENT)
│   └── config_laptop.yaml           # LAPTOP_MODE overlay
├── data/
│   ├── raw/edge_iiotset.csv         # user-supplied
│   └── processed/                   # generated (.parquet and .csv)
├── logs/                            # one timestamped log per script run
├── outputs/
│   ├── artifacts/                   # encoders, scaler, split indices,
│   │                                # feature names, class mapping, seed,
│   │                                # SHAP background
│   ├── models/                      # .h5 models, summaries, diagrams
│   ├── results/                     # metrics CSVs, plots, reports, timings
│   └── shap_plots/                  # SHAP summary and bar plots
├── src/
│   ├── utils/                       # config_loader, logger, seed, dataio
│   ├── preprocessing/               # load_dataset, encode_normalize
│   ├── partitioning/                # partition_data (IID sharding)
│   ├── models/                      # cnn_gru (the only model builder)
│   ├── federated/                   # client_app, server_app (SavingFedAvg)
│   ├── centralized/                 # train_baseline
│   ├── evaluation/                  # metrics, compare_fl_vs_centralized
│   └── explainability/              # shap_utils
├── dashboard/app.py                 # 4-tab read-only Streamlit app
├── experiments/                     # 5 orchestration entry points
├── tests/                           # pytest suite
└── project-docs/                    # SDS, AI Coding Contract, phase prompts,
                                     # compliance audit
```

`src/utils/dataio.py` and `configs/config_laptop.yaml` are additions beyond the SDS
Section 9 tree: `dataio.py` centralizes processed-data read/write so the storage format
is a config choice, and the laptop overlay exists to keep the federated simulation inside
16 GB of RAM on Windows.

## Outputs

After a full run:

| Directory | Contents |
|---|---|
| `outputs/artifacts/` | `scaler.pkl`, `label_encoder.pkl`, `categorical_encoder.pkl`, `train_indices.pkl`, `test_indices.pkl`, `feature_names.pkl`, `class_mapping.json`, `random_seed.txt`, `shap_background.npy` |
| `outputs/models/` | `centralized_best_model.h5`, `centralized_last_model.h5`, `centralized_model_summary.txt`, `centralized_architecture.png`, `federated_global_model.h5`, `federated_model_summary.txt`, `federated_architecture.png` |
| `outputs/results/` | `centralized_history.csv`, `centralized_training_time.txt`, `federated_history.csv`, `federated_training_time.txt`, `comparison_table.csv`, `model_size_comparison.csv`, `convergence_plot.png`, `centralized_confusion_matrix.png`, `federated_confusion_matrix.png`, `centralized_classification_report.txt`, `federated_classification_report.txt`, `centralized_training_curves.png`, `class_distribution.png` |
| `outputs/shap_plots/` | `shap_summary_plot.png`, `shap_bar_plot.png` |

Processed data: 1,945,449 rows after de-duplication (1,556,359 train / 389,090 test),
88 features, 15 attack classes.

## Results

From `outputs/results/comparison_table.csv`, both models scored on the same 389,090-row
held-out test set:

| Metric | Centralized | Federated |
|---|---|---|
| Accuracy | 0.9564 | 0.8861 |
| F1 (macro) | 0.8075 | 0.5266 |
| F1 (weighted) | 0.9540 | 0.8829 |
| Loss | 0.0995 | 0.2291 |
| Training time (s) | 898.1 | 4757.6 |
| Latency (ms/sample) | 0.101 | 0.095 |
| Model size (MB) | 0.446 | 0.167 |

Both models share the same 35,471-parameter architecture. The federated model trades
roughly 7 accuracy points for the privacy property that raw traffic never leaves the
client, and the gap is widest on macro-F1 — the rare attack classes are what federation
costs most. Federated accuracy rises across rounds (0.720 → 0.886 over 5 rounds; see
`federated_history.csv`).

## Reproducibility

- `seed: 42` is set in `configs/config.yaml` and applied by `src/utils/seed.py` to Python
  `random`, NumPy, and TensorFlow before any model is constructed. The value is recorded
  to `outputs/artifacts/random_seed.txt` on every run.
- The train/test split is computed once, stratified, and persisted as
  `train_indices.pkl` / `test_indices.pkl`. Re-runs reuse it unless
  `dataset.force_reprocess: true`.
- IID partitioning uses a local `RandomState(seed)`, so shard boundaries do not depend on
  how much other random work preceded them.
- `PYTHONHASHSEED` is set at runtime, which CPython applies only to child processes. For
  strict hash determinism, set it in the shell before launching:

  ```bash
  set PYTHONHASHSEED=42
  ```

  No numerical result in this pipeline depends on hash ordering.
- Determinism is scoped to a CPU-only target environment. TensorFlow op-level determinism
  is not enabled, so GPU runs may differ in low-order decimal places.

## Tests

```bash
pytest tests/ -v
```

123 tests, all passing. Coverage: `test_config_loader.py`, `test_preprocessing.py`
(including train/test leakage regression tests), `test_partitioning.py`,
`test_models.py` (layer-by-layer architecture assertions), and `test_federated_loop.py`
(a 2-client reduced-config simulation smoke test that never touches the production
config).

`tests/test_metrics.py` and `tests/test_shap_utils.py` are specified in SDS Section 24
but are not present in `tests/`. The evaluation and explainability modules they would
cover are implemented and have produced all their artifacts, but they carry no automated
test coverage.

## Documentation

| Document | Purpose |
|---|---|
| `project-docs/SDS.md` | Frozen software design specification (Revision 2.1) |
| `project-docs/AI_Coding_Contract.md` | Locked values and invariants |
| `project-docs/Phase_Implementation_Prompts.md` | Phase-by-phase implementation plan |
| `project-docs/COMPLIANCE_AUDIT.md` | Read-only compliance audit findings |
