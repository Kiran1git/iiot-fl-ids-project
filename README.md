# Privacy-Preserving Federated Intrusion Detection for Industrial IoT (CNN-GRU)

A federated intrusion detection system for Industrial IoT network traffic. A lightweight
CNN-GRU classifier is trained two ways on the Edge-IIoTset dataset — once centrally as a
baseline, and once through a simulated federated setup where raw traffic never leaves the
client — and both are scored on the same held-out test set. SHAP attributions explain the
model globally and per prediction, and a read-only Streamlit dashboard exposes every
artifact the pipeline produces, including live prediction on a user-uploaded CSV.

This README documents the system **as implemented and executed**, verified against the
files in this repository. The frozen design specification is `project-docs/SDS.md`; where
the executed configuration differs from it, the executed values are stated here.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Preprocessing | `src/preprocessing/` | Loads the raw CSV, drops identifier and leakage-prone columns, derives multi-class and binary labels, removes exact duplicate rows, splits train/test once (stratified), fits encoders and the scaler on the **training partition only**, then transforms the full dataset. |
| Partitioning | `src/partitioning/` | Shards the training split IID across virtual clients using a local `RandomState(seed)`. |
| Centralized baseline | `src/centralized/` | Trains the CNN-GRU on the full training split with balanced class weights, early stopping, and best-weight checkpointing. |
| Federated simulation | `src/federated/` | Runs Flower + FedAvg over Ray. `SavingFedAvg` captures the aggregated global parameters from the strategy, not from any single client. |
| Evaluation | `src/evaluation/` | Scores both models on the identical test set; writes the comparison table, confusion matrices, classification reports, convergence and training-curve plots, and a model-size table. |
| Explainability | `src/explainability/` | SHAP `GradientExplainer` over a persisted background sample; global summary and bar plots, plus a single-prediction entry point for the dashboard. |
| Inference | `src/inference/predict.py` | Read-only user-facing path: uploaded CSV → persisted artifacts → federated global model → class, confidence, Normal/Attack. Fits nothing and writes nothing. |
| Dashboard | `dashboard/app.py` | Five read-only tabs. Performs no training. |

### Leakage prevention

Two classes of leakage are handled explicitly, and both are covered by regression tests:

- **Fit-on-train-only.** Every encoder, the `MinMaxScaler`, and the label encoder are fit
  on the training partition and then applied to the test partition. The split is computed
  once and persisted as `train_indices.pkl` / `test_indices.pkl`.
- **Identifier and near-label columns are dropped** via `dataset.drop_columns`, including
  `frame.time` (a per-packet timestamp that is effectively a row ID, and correlates with
  the label because Edge-IIoTset was captured attack-by-attack), `tcp.srcport`,
  `http.request.uri.query`, host/IP columns, and raw payload columns. Exact duplicate rows
  are removed **before** the split, so byte-identical training rows cannot appear in test.

## Data and label space

| Property | Value |
|---|---|
| Dataset | Edge-IIoTset (user-supplied raw CSV) |
| Rows after de-duplication | 1,945,449 |
| Train / test split | 1,556,359 / 389,090 (80/20, stratified) |
| Model input features | 88 |
| Attack classes | 15 |
| Required raw upload columns | 48 |

The 15 classes, from `outputs/artifacts/class_mapping.json`: `Backdoor`, `DDoS_HTTP`,
`DDoS_ICMP`, `DDoS_TCP`, `DDoS_UDP`, `Fingerprinting`, `MITM`, `Normal`, `Password`,
`Port_Scanning`, `Ransomware`, `SQL_injection`, `Uploading`, `Vulnerability_scanner`,
`XSS`. `num_classes` is always derived from this file at runtime, never hardcoded.

## Model

One model definition (`src/models/cnn_gru.py`), used unchanged by both training paths:

```
Input(88, 1)
  -> Conv1D(64, kernel_size=3, padding="same", activation="relu")
  -> MaxPooling1D(pool_size=2)
  -> GRU(64, return_sequences=False)
  -> Dropout(0.3)
  -> Dense(128, activation="relu")
  -> Dense(15, activation="softmax")
```

Adam (`learning_rate=0.001`), `categorical_crossentropy`, **35,471 parameters** — small
enough for repeated federated parameter exchange.

## Executed configuration

`configs/config.yaml` is the single source of truth; `src/utils/config_loader.py` resolves
a run profile from the `IIOT_MODE` environment variable and deep-merges an overlay.

| Setting | Executed value |
|---|---|
| `seed` | 42 |
| `dataset.test_size` | 0.2 |
| `dataset.drop_duplicates` | true |
| `training.centralized_epochs` | 10 (early stopping on `val_loss`, patience 3) |
| `training.batch_size` | 128 |
| `training.use_class_weights` | true |
| **`federated.num_clients`** | **2** |
| **`federated.num_rounds`** | **5** |
| **`federated.local_epochs`** | **1** |
| `federated.partition_strategy` | iid |
| `federated.client_val_split` | 0.2 |
| `explainability.background_size` / `sample_size` | 100 / 100 |

The committed federated artifacts are from a **2-client × 5-round × 1-local-epoch** run.
The SDS's own reference YAML block specifies 4 clients × 15 rounds × 2 local epochs; that
configuration exhausted memory on the 16 GB Windows target (Ray's memory-mapped object
store fails with `ERROR_NO_SYSTEM_RESOURCES` around round 5). The reduction is documented
in `configs/config_laptop.yaml`, which also serialises clients to one concurrent Ray
actor. FedAvg is mathematically unaffected by serialising independent clients within a
round.


## Results

Both models scored on the same 389,090-row held-out test set
(`outputs/results/comparison_table.csv`):

| Metric | Centralized | Federated |
|---|---|---|
| Accuracy | 0.9564 | 0.8861 |
| Precision (macro) | 0.8358 | 0.5866 |
| Recall (macro) | 0.8333 | 0.6073 |
| F1 (macro) | 0.8075 | 0.5261 |
| F1 (weighted) | 0.9540 | 0.8829 |
| Loss | 0.0995 | 0.2291 |
| Training time (s) | 898.1 | 4757.6 |
| Latency (ms/sample) | 0.101 | 0.095 |
| Model size (MB) | 0.446 | 0.167 |

Federated accuracy across rounds (`outputs/results/federated_history.csv`): 0.720 → 0.720
→ 0.720 → 0.814 → 0.886. Round 3 shows a loss spike (1.34) before convergence resumes.

The federated model gives up roughly 7 accuracy points for the privacy property that raw
traffic never leaves the client. The gap is widest on macro-F1 (0.81 → 0.53), meaning the
rare attack classes are what federation costs most under this reduced configuration.

### Evaluation artifacts

Per-model confusion matrices (`centralized_confusion_matrix.png`,
`federated_confusion_matrix.png`), full per-class precision/recall/F1 classification
reports, `convergence_plot.png`, `centralized_training_curves.png`,
`class_distribution.png`, `comparison_table.csv`, and `model_size_comparison.csv` — all
under `outputs/results/`.

## Explainability

- **Global** — `experiments/run_explainability.py` builds a `GradientExplainer` from a
  100-row background sample drawn with a seeded `RandomState`, persists it to
  `outputs/artifacts/shap_background.npy`, and writes `shap_summary_plot.png` and
  `shap_bar_plot.png`.
- **Per prediction** — `explain_single_prediction` returns one SHAP array per class
  (15 arrays of shape `(1, 88)`). `get_top_feature_contributions` ranks features using the
  **predicted class's** signed row, not a mean across classes, so the sign of each
  contribution ("increases" / "decreases") stays meaningful.
- The dashboard never re-samples the background; it always loads the persisted
  `shap_background.npy`, so an explanation cannot drift from the run it describes.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Five read-only tabs:

1. **Project Overview** — project statistics, class distribution, federated architecture
   diagram.
2. **Dataset & Model Explorer** — dataset preview, shape and class count, test-split
   sample selector, prediction with per-class probabilities.
3. **FL vs. Centralized** — comparison table, model-size table, convergence plot, both
   confusion matrices side by side.
4. **Explainability** — global SHAP summary plot plus per-sample top contributing
   features for a selected test row.
5. **Live Prediction** — upload and score your own CSV (below).

### Live Prediction workflow

`CSV upload → validation → preprocessing → prediction → confidence → Normal/Attack → CSV download`

- **Upload** — a raw Edge-IIoTset-format CSV. The expander lists the 48 required columns.
- **Validation** — a missing required column rejects the whole file with a message naming
  what is missing. A row with an unparseable numeric cell is dropped and reported by its
  1-based line number, so one bad line does not cost the rest of the upload. Extra columns
  (including `Attack_type` / `Attack_label` if left in) are ignored and reported.
- **Preprocessing** — the same persisted encoders, the same scaler, the same reshape used
  in training. Features are reindexed onto the persisted 88-column order before reaching
  the model, so the upload's own column order is irrelevant.
- **Prediction** — the federated global model
  (`federated_global_model_keras215_v2.h5`) produces per-row `predicted_class`,
  `confidence`, and `status` (`Normal` / `Attack`), plus an attack count and a predicted
  class distribution chart.
- **Download** — the full result table as CSV.

### Live SHAP

Below the results, select any predicted row to see its explanation: signed SHAP
contributions toward that row's predicted class, ranked by absolute magnitude, with a
downloadable CSV of the contributions. The explanation runs against the exact tensor the
model scored (`predict_dataframe(..., return_features=True)`), not a recomputed one.

## Installation

Python 3.10.

```bash
pip install -r requirements.txt
```

Pinned in `requirements.txt`: `tensorflow==2.15.1`, `flwr[simulation]==1.8.0`,
`ray==2.6.3`, `shap==0.44.1`, plus scikit-learn, pandas, numpy, streamlit, matplotlib,
pyyaml, pytest. `requirements_colab.txt` is the Colab variant.

Two optional extras:

- **Parquet** — `configs/config.yaml` defaults to a `.parquet` processed file, which needs
  a Parquet engine (`pip install pyarrow`). It is not pinned in `requirements.txt`; the
  alternative is setting `paths.processed_data_file` to `edge_iiotset_processed.csv`.
- **Architecture diagrams** — `*_architecture.png` export needs `pydot` and system
  Graphviz. Best-effort: the pipeline logs a warning and continues without them.

Place the raw dataset at `data/raw/edge_iiotset.csv` (filename from
`paths.raw_data_file`). It is user-supplied and never written by any script.

## Run order

From the project root. Each script seeds first, then writes one timestamped log to
`logs/`.

```bash
python experiments/run_preprocessing.py   # 1. encode, scale, split, persist artifacts
python experiments/run_centralized.py     # 2. centralized baseline
python experiments/run_federated.py       # 3. federated simulation (Flower + FedAvg)
python experiments/run_evaluation.py      # 4. compare both on the shared test set
python experiments/run_explainability.py  # 5. SHAP global explanations
streamlit run dashboard/app.py            # 6. dashboard (read-only)
```

Steps 2 and 3 each depend only on step 1, so their relative order does not matter. Step 4
needs both. Steps 4 and 5 must precede step 6 for every tab to render data.

Reduced-memory profile:

```bash
set IIOT_MODE=LAPTOP_MODE      # Windows cmd
python experiments/run_federated.py
```

`LAPTOP_MODE` overlays `configs/config_laptop.yaml`, changing only the federated settings,
`training.batch_size`, and Ray limits. Preprocessing, the architecture, and the evaluation
pipeline are identical across profiles.

## Models and artifacts

```
outputs/
├── artifacts/     scaler.pkl, label_encoder.pkl, categorical_encoder.pkl,
│                  train_indices.pkl, test_indices.pkl, feature_names.pkl,
│                  class_mapping.json, random_seed.txt, shap_background.npy
├── models/        centralized_best_model.h5, centralized_last_model.h5,
│                  federated_global_model.h5,
│                  federated_global_model_keras215_v2.h5,
│                  centralized_model_summary.txt, federated_model_summary.txt,
│                  centralized_architecture.png, federated_architecture.png
├── results/       comparison_table.csv, model_size_comparison.csv,
│                  centralized_history.csv, federated_history.csv,
│                  *_training_time.txt, *_confusion_matrix.png,
│                  *_classification_report.txt, convergence_plot.png,
│                  centralized_training_curves.png, class_distribution.png
└── shap_plots/    shap_summary_plot.png, shap_bar_plot.png
```

`federated_global_model_keras215_v2.h5` is the checkpoint the inference path and dashboard
load; the filename is declared once, as
`FEDERATED_GLOBAL_MODEL_FILENAME` in `src/inference/predict.py`, so the two cannot drift
onto different weights. `federated_global_model.h5` is the original training output,
retained as the provenance of the results above.

## Repository layout

```
iiot-fl-ids-project/
├── README.md, requirements.txt, requirements_colab.txt
├── configs/          config.yaml (FULL_EXPERIMENT), config_laptop.yaml (LAPTOP_MODE)
├── data/             raw/ (user-supplied), processed/ (generated)
├── logs/             one timestamped log per script run
├── outputs/          artifacts/, models/, results/, shap_plots/
├── src/
│   ├── utils/        config_loader, logger, seed, dataio
│   ├── preprocessing/ load_dataset, encode_normalize
│   ├── partitioning/ partition_data (IID sharding)
│   ├── models/       cnn_gru (the only model builder)
│   ├── centralized/  train_baseline
│   ├── federated/    client_app, server_app (SavingFedAvg)
│   ├── evaluation/   metrics, compare_fl_vs_centralized
│   ├── explainability/ shap_utils
│   └── inference/    predict (read-only user-facing path)
├── dashboard/app.py  5-tab read-only Streamlit app
├── experiments/      5 orchestration entry points
├── tests/            7 pytest modules
└── project-docs/     SDS, AI Coding Contract, phase prompts, compliance audit
```

## Reproducibility

- `seed: 42` is applied by `src/utils/seed.py` to Python `random`, NumPy, TensorFlow, and
  `PYTHONHASHSEED` before any model is built, and recorded to
  `outputs/artifacts/random_seed.txt` on every run.
- The stratified train/test split is computed once and reused from the persisted index
  files unless `dataset.force_reprocess: true`.
- IID partitioning and SHAP background sampling each use a local `RandomState(seed)`, so
  their draws do not depend on how much other random work preceded them.
- `PYTHONHASHSEED` set at runtime only affects child processes; set it in the shell
  (`set PYTHONHASHSEED=42`) for strict hash determinism. No numerical result here depends
  on hash ordering.
- Determinism is scoped to a CPU-only target. TensorFlow op-level determinism is not
  enabled, so GPU runs may differ in low-order decimals.

## Tests

```bash
pytest tests/ -q
```

**179 tests, all passing.** Modules: `test_config_loader.py`, `test_preprocessing.py`
(including train/test leakage regression tests), `test_partitioning.py`, `test_models.py`
(layer-by-layer architecture assertions), `test_federated_loop.py` (a 2-client
reduced-config simulation smoke test that never touches the production config),
`test_inference.py` (upload validation and prediction contracts, including against the
persisted artifacts), and `test_live_prediction_shap.py` (end-to-end explanation coverage
across every class and feature).

`tests/test_metrics.py` and `tests/test_shap_utils.py` are named in SDS Section 24 but are
not present. Those modules are implemented and have produced all their artifacts, but they
carry no dedicated automated coverage beyond the SHAP paths exercised by
`test_live_prediction_shap.py`.

## Limitations

- **Reduced federated configuration.** The executed run is 2 clients × 5 rounds × 1 local
  epoch, driven by a 16 GB Windows memory ceiling rather than by a modelling choice. The
  federated numbers should be read as a lower bound: more clients, more rounds, and more
  local epochs would be expected to narrow the gap to the centralized baseline.
- **Rare-class performance.** Macro-F1 of 0.53 federated versus 0.81 centralized means
  minority attack classes are detected substantially less reliably under federation, even
  with balanced class weights.
- **Simulated federation.** Flower's Ray simulation runs all clients in one process on one
  machine. There is no real network partition, no client dropout, no stragglers, and no
  communication cost measurement.
- **IID partitioning only.** Shards are drawn IID. Real IIoT deployments are non-IID by
  nature, which is typically harder for FedAvg.
- **No formal privacy guarantee.** The privacy property is architectural — raw traffic
  stays on the client — not cryptographic. No differential privacy, secure aggregation, or
  gradient-inversion defence is implemented.
- **Single dataset, single architecture.** Results are specific to Edge-IIoTset and to
  this one CNN-GRU; no cross-dataset or cross-architecture generalisation is claimed.
- **CPU-only determinism.** See Reproducibility above.

## Documentation

| Document | Purpose |
|---|---|
| `project-docs/SDS.md` | Frozen software design specification (Revision 2.1) |
| `project-docs/AI_Coding_Contract.md` | Locked values and invariants |
| `project-docs/Phase_Implementation_Prompts.md` | Phase-by-phase implementation plan |
| `project-docs/COMPLIANCE_AUDIT.md` | Read-only compliance audit findings |
