# Software Design Specification
## Revision 2.1
## Privacy-Preserving Federated Intrusion Detection System for Simulated Industrial IoT (CNN-GRU)

**Status:** DESIGN FROZEN — IMPLEMENTATION READY

**Document type:** Implementation-grade Software Design Specification
**Intended reader:** An AI coding agent (Claude Code, Codex, Cursor, Antigravity, Gemini CLI, or equivalent) implementing this project with zero additional context beyond this document.
**Rule for the implementing agent:** Every instruction in this document is written to have exactly one valid implementation. Where a decision point could still plausibly be read two ways, the most literal, most restrictive interpretation applies. The agent must never introduce an architecture, library, dataset, version, or design pattern not explicitly named in this document. This document is the single authoritative source for this project — no separate addendum, patch, or errata document exists or should be consulted.

---

## 0. Document Control

| Field | Value |
|---|---|
| Project | IIoT Federated Intrusion Detection System (CNN-GRU) |
| Base reference | Bhavsar et al., "FL-IDS: Federated Learning-Based Intrusion Detection System Using Edge Devices for Transportation IoT," IEEE Access, 2024 |
| Mode | Simulation-only (no physical hardware, no Docker) |
| Architecture status | LOCKED — CNN-GRU, Flower, FedAvg, Edge-IIoTset, SHAP, Streamlit are final |
| Revision | 2.1 — supersedes Revision 2.0. All inconsistencies identified during the Revision 2.0 architecture consistency audit are resolved in place. This is the single authoritative specification. |

---

## 1. Project Objective

Build a simulation-only Industrial IoT intrusion detection system that:
1. Trains a single, shared **CNN-GRU** architecture to perform **multi-class** classification of IIoT network traffic (Normal + attack categories).
2. Trains this architecture two ways: (a) **Federated Learning** via Flower's Simulation Engine with **FedAvg** across 4 virtual clients holding an **IID** partition of the data, and (b) **Centralized Learning** on the full dataset, as a comparison baseline.
3. Produces a rigorous, side-by-side **Federated vs. Centralized** performance comparison.
4. Explains model predictions using **SHAP**.
5. Presents all results through a **read-only Streamlit dashboard** that performs no training.

This objective is final. Do not modify it.

---

## 2. Research Gap (Why This Project Differs From the Reference Paper)

The reference paper (FL-IDS) established that federated learning is viable for IoT intrusion detection using physical edge hardware (Raspberry Pi clients, Jetson Xavier server) in the **transportation** domain, using **Logistic Regression** and a **PCC-filtered CNN** on NSL-KDD and Car-Hacking datasets.

This project's novelty and scope difference:

| Dimension | Reference Paper | This Project | Why the Change |
|---|---|---|---|
| Domain | Transportation IoT (CAVs) | Industrial IoT (IIoT) | Different threat landscape and dataset ecosystem; IIoT has distinct attack categories (e.g., industrial protocol exploitation) not present in transportation datasets |
| Deployment | Physical edge hardware (Raspberry Pi, Jetson Xavier) | Pure simulation (Flower Simulation Engine, virtual clients) | Removes hardware dependency entirely, enabling reproducibility on any standard machine and isolating the *algorithmic* contribution from *deployment engineering* concerns |
| Feature engineering | Manual PCC (Pearson Correlation Coefficient) feature selection feeding the CNN | No manual feature selection; raw normalized/encoded features feed the model directly | The CNN-GRU hybrid is designed to let the CNN layer learn spatial feature importance implicitly, removing a hand-tuned preprocessing step and reducing pipeline fragility |
| Model comparison | Logistic Regression (LR-IDS) vs. PCC-CNN (two competing architectures) | Single CNN-GRU architecture used in both Federated and Centralized modes | The project's comparison axis is **training paradigm** (federated vs. centralized), not **model family** — this isolates the effect of federation itself on a fixed architecture |
| Temporal modeling | None (CNN only; no recurrent component) | GRU layer added after CNN | Adds capacity to model dependencies across the feature sequence beyond purely local (convolutional) patterns — this is the primary architectural novelty relative to the reference paper |
| Explainability | Not present | SHAP integration required | Adds transparency into which features drive each classification decision — absent from the reference work entirely |
| Presentation layer | Not present | Streamlit dashboard | Adds an operator-facing visualization layer absent from the reference work |
| Explicit training-paradigm comparison | Implicit (paper reports centralized and federated numbers in the same tables but does not frame it as the central research question) | Explicit, first-class deliverable (`compare_fl_vs_centralized.py`) | This project treats "does federation cost us accuracy, and if so how much" as a primary, directly answered question |

**Stated novelty of this project:** A hardware-free, fully reproducible demonstration that a CNN-GRU hybrid model can be trained under a federated paradigm on IIoT traffic data with accuracy competitive to centralized training, while additionally providing per-prediction explainability — none of which the reference paper addresses.

### 2.1 Source Dataset Column Reference (Authoritative)

The raw Edge-IIoTset CSV supplies the classification target under two fixed, literal native column names. The implementing agent must treat these as ground truth and must not substitute, guess, or auto-detect alternatives:

- **`Attack_type`** — native multi-class attack-category column (string labels, including the literal value `"Normal"` for benign traffic). This is the sole source column for the derived `label` column used for model training.
- **`Attack_label`** — native binary column (`0`/`1`). This is the source for the derived `binary_label` column. It is a direct, near-perfect proxy for the multi-class target and constitutes label leakage if left in the feature set.

Both native columns (`Attack_type`, `Attack_label`) are removed from the DataFrame **exclusively by `create_labels`** (Section 14.1). `drop_identifier_columns` (Section 14.1) is responsible only for the unrelated high-cardinality/identifier columns listed in `config["dataset"]["drop_columns"]`, and that list does **not** include `Attack_label` — this avoids any ordering ambiguity between the two functions and gives `create_labels` sole, unambiguous ownership of native-column removal and of the binary-agreement validation described in Section 14.1.

`config["dataset"]["target_column"]` refers to this native source column (`"Attack_type"`), never to the derived `label` column. The derived `label` and `binary_label` columns are created once, by `create_labels`, and are never read back from raw source data afterward.

---

## 3. Design Rationale (Why Each Major Decision Was Made)

| Decision | Rationale |
|---|---|
| **Edge-IIoTset dataset** | Purpose-built, recent (2022), publicly available IIoT/IoT dataset with realistic multi-class attack categories (DoS/DDoS, information gathering, MITM, injection, malware) generated from a real testbed topology — directly relevant to the IIoT domain this project targets. |
| **CNN-GRU architecture** | Conv1D layers extract local patterns across the feature vector (analogous to the reference paper's CNN component); GRU layers add a lightweight recurrent mechanism to model dependencies across the feature sequence with fewer parameters than an LSTM, keeping the model "lightweight" as required. |
| **Flower framework** | Directly reusable simulation engine (`flwr.simulation.start_simulation`) that requires no physical devices, no network configuration, and no container orchestration to simulate multiple federated clients on a single machine. |
| **FedAvg strategy** | The standard, most well-understood federated aggregation algorithm (McMahan et al., 2017); using it (rather than a novel aggregation scheme) keeps the project's contribution focused on the federated-vs-centralized comparison rather than on aggregation-algorithm research. |
| **IID partitioning (not Non-IID)** | Isolates the effect of federation itself (communication rounds, weight averaging) from the confounding effect of data heterogeneity across clients. Non-IID introduces a second independent variable that would make the Federated vs. Centralized comparison harder to interpret cleanly. |
| **SHAP explainability** | Model-agnostic, theoretically grounded (Shapley values from cooperative game theory) explanation method with a Keras-native `GradientExplainer` implementation, avoiding the need for a separate surrogate model. |
| **Centralized comparison baseline** | Without a centralized baseline, there is no reference point to determine whether federation incurs an accuracy or convergence cost — this baseline is what makes the federated results interpretable rather than an isolated number. |
| **Simulation-only (no hardware, no Docker)** | Removes deployment engineering as a variable, keeps the project reproducible on any machine meeting the stated minimum specs (Section 6), and matches the explicit, final scope decision made for this project. |
| **Streamlit dashboard** | Fastest path to a Python-native, no-separate-frontend interactive presentation layer; appropriate for a project of this scope and team size. |
| **Multi-class (not binary) classification** | The project's stated goal is to "detect and classify" attacks, which requires per-attack-category labels, not merely a normal/attack binary signal. |
| **Fixed random seed (42) everywhere** | Guarantees that every run — preprocessing, partitioning, model initialization, training, SHAP sampling — is exactly reproducible, which is required for the Federated vs. Centralized comparison to be scientifically valid (both paths must be evaluated under identical, reproducible conditions). |
| **Train/test split performed before fitting any encoder or scaler** | Fitting a scaler/encoder on the full dataset (including test rows) leaks test-set distributional information into the transformation applied to training data, which would make the reported test-set metrics optimistic and non-representative of true generalization. Splitting first and fitting transformers on the training partition only ensures the test set is genuinely held out. |
| **`create_labels` as sole owner of native-column removal** | Having a single function own both the derivation of `label`/`binary_label` and the removal of the native `Attack_type`/`Attack_label` columns eliminates any ambiguity about pipeline step ordering and guarantees the binary-agreement validation (Section 14.1) always has both columns available when it runs. |
| **Exact pinned dependency versions (Section 7)** | Flower's public API changed shape across 1.x releases (client/strategy-based simulation vs. app-based simulation), and TensorFlow's default Keras version changed in a way that breaks SHAP's `GradientExplainer`. Pinning exact versions — including Flower's `ray` transitive dependency — removes any chance that a "compatible" open-ended install silently breaks either the federated simulation loop or the explainability pipeline. |
| **Separate fit-time and evaluate-time metrics aggregation functions** | `FLClient.fit()` returns both `loss` and `accuracy` in its metrics dict (sourced from local training history), while `FLClient.evaluate()` returns only `accuracy` in its metrics dict because Flower's simulation engine already aggregates evaluation loss natively from each client's returned `(loss, num_examples, metrics)` tuple. Using one aggregation function for both callback sites would require a `loss` key that does not exist in the evaluate-time metrics dict, so two distinct functions are required to avoid a runtime `KeyError`. |
| **Persisted training-time and SHAP-background artifacts** | `generate_comparison_table` and the dashboard's per-prediction explainer both need data produced mid-run (wall-clock training duration; a representative background sample) that is not otherwise saved anywhere. Persisting both as first-class artifacts keeps every downstream script a pure consumer of `outputs/`, consistent with the rest of the artifact-flow design, rather than requiring any script to recompute or hold in-memory state across process boundaries. |

---

## 4. Explicit Assumptions

The implementing agent must treat the following as ground truth. No alternative interpretation is permitted.

1. The Edge-IIoTset dataset is treated as **static** — it is downloaded once, preprocessed once, and never streamed or updated during the project's lifetime.
2. Each row (sample) in the dataset is treated as an **independent observation** — no cross-row temporal relationship is assumed at the dataset level (i.e., row order carries no meaning and may be shuffled freely).
3. `Conv1D` layers treat each sample's **feature vector as a pseudo-sequence** (features arranged along a 1D axis), not as a genuine time series. There is no actual temporal/packet-order signal being modeled by the CNN component.
4. The `GRU` layer models **dependencies across the feature dimension** of a single sample (i.e., relationships between different feature values within one row), **not** temporal dependencies across multiple packets or multiple time steps. This is an architectural choice, not a claim that the dataset contains genuine sequential/streaming structure.
5. No physical hardware, embedded device, or network interface exists anywhere in this project. All "clients" are Python objects running in a single process via Flower's Simulation Engine.
6. All 4 virtual clients are assumed to have **equal computational capability** (simulation environment) — there is no modeling of heterogeneous client compute/battery/bandwidth constraints.
7. The dataset, once preprocessed, is assumed to fit in memory on a machine with 8GB RAM. If the raw Edge-IIoTset file(s) exceed available memory during initial loading, the implementing agent must load via chunked `pandas.read_csv(..., chunksize=...)` and concatenate, but the **processed, feature-selected/encoded output** is assumed to fit in memory for training.
8. "Best model" (Section 15) is defined strictly as the model checkpoint with the **lowest validation loss** observed during training, not the highest validation accuracy, unless explicitly stated otherwise for a specific script.
9. The raw source columns `Attack_type` and `Attack_label` (Section 2.1) are always present in the raw CSV under exactly those names; the derived `label`/`binary_label` columns are computed once during preprocessing and are the only label representations used anywhere downstream.
10. `num_classes` is never a hardcoded literal anywhere in `src/`. It is always derived at runtime as `len(class_mapping)`, where `class_mapping` is loaded from `outputs/artifacts/class_mapping.json`.
11. This project targets CPU-only execution on the minimum hardware specified in Section 6. No GPU-specific nondeterminism is assumed, provisioned for, or accounted for anywhere in this document; full run-to-run determinism is required (Section 23).

---

## 5. Explicit Limitations and Out-of-Scope Items

The following are **deliberately excluded** from this project. The implementing agent must not add them, even partially, even as "bonus" functionality.

| Excluded Item | Reason for Exclusion |
|---|---|
| Physical hardware deployment | Project is simulation-only by final design decision (Section 1). |
| Docker / containerization | Adds deployment engineering complexity irrelevant to the algorithmic comparison this project studies. |
| Non-IID client data partitioning | Would introduce a confounding variable into the Federated vs. Centralized comparison (Section 3). |
| Adversarial federated learning (poisoning, Byzantine clients) | Out of scope — this project studies standard FedAvg behavior only, not robustness. |
| Secure aggregation | Out of scope — no cryptographic protocol is implemented; model weights are aggregated in plaintext within the simulation. |
| Differential privacy | Out of scope — no noise injection or privacy-budget accounting is implemented. |
| Encrypted gradients / homomorphic encryption | Out of scope — not required for a simulation-only academic project. |
| Concept drift handling | Out of scope — the dataset is static (Assumption 1); no mechanism for detecting or adapting to distributional shift over time is implemented. |
| Online / continual learning | Out of scope — training occurs in fixed, offline rounds/epochs against a fixed dataset. |
| Streaming inference | Out of scope — inference in this project is always performed on a pre-loaded batch of test samples, never on a live/streaming feed. |
| Real packet capture / live network monitoring | Out of scope — all data originates from the static Edge-IIoTset CSV files. |
| Edge device latency measurement | Out of scope — no physical device exists to measure. |
| Communication bandwidth analysis | Out of scope — Flower's simulation transport is in-process; real network bandwidth is not modeled or measured. |
| Flower's app-based simulation API (`ClientApp`/`ServerApp`/`run_simulation`) | Out of scope — this project uses the pinned `flwr==1.8.0` client-function/strategy-based `start_simulation` API exclusively (Section 7, Section 14.6). Do not upgrade Flower or switch APIs. |
| Arbitrary optimizer selection | Out of scope — Adam is the only supported optimizer (Section 6); `config["training"]["optimizer"]` is documentation-only, not a live switch. |
| GPU-specific execution paths or nondeterminism handling | Out of scope — the project's minimum hardware target is CPU-only (Section 6); no GPU code path, tolerance, or fallback logic is implemented. |

If asked to extend this project in any of the above directions, the implementing agent must treat that as a **separate future work item**, not part of this SDS's deliverables.

---

## 6. Final Locked Technical Decisions

| Parameter | Value | Configurable via `config.yaml`? |
|---|---|---|
| Dataset | Edge-IIoTset | `dataset.raw_path` |
| Classification type | Multi-class (attack category as target) | N/A (fixed) |
| Native target column (raw CSV) | `Attack_type` | `dataset.target_column` |
| Native binary column (raw CSV, always dropped by `create_labels`) | `Attack_label` | `dataset.raw_binary_column` |
| Categorical/numeric column classification rule | `"dtype_object"` — the only implemented rule; columns of pandas dtype `object`/`category` are categorical, all other non-label numeric dtypes (including `bool`) are numeric | `dataset.categorical_column_rule` |
| Model | CNN-GRU: `Conv1D(64, kernel_size=3, padding='same', activation='relu') → MaxPooling1D(pool_size=2) → GRU(64, return_sequences=False) → Dropout(0.3) → Dense(128, activation='relu') → Dense(num_classes, activation='softmax')` | `model.*` block (layer sizes only; padding, `return_sequences`, and layer order are fixed, not configurable) |
| `num_classes` | **Not a fixed integer.** Always derived at runtime as `len(class_mapping)`, where `class_mapping` is the object persisted to `outputs/artifacts/class_mapping.json` during preprocessing. Every script that builds or evaluates a model must derive `num_classes` this way; it must never be hardcoded (e.g., never literally `15`) anywhere in `src/`. | N/A (derived, not configured) |
| Optimizer | Adam (the only supported optimizer; `training.optimizer` in config is documentation-only and does not switch optimizer classes) | `training.learning_rate=0.001` |
| Loss function | `categorical_crossentropy` | `training.loss` |
| Centralized epochs | 10 | `training.centralized_epochs` |
| Centralized validation split | 0.2 (carved by Keras from the end of the training array passed to `.fit()`) | `training.validation_split` |
| Early stopping (centralized only) | `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` | `training.early_stopping_patience` |
| FL local epochs per round | 2 | `federated.local_epochs` |
| FL communication rounds | 15 | `federated.num_rounds` |
| Number of virtual clients | 4 | `federated.num_clients` |
| Client data split | IID, equal-size, seeded random partition of the **training split only** (see Section 14.3) | `federated.partition_strategy = "iid"` |
| Client evaluation data | The single shared global test split (identical across all 4 clients — never partitioned per client; see Section 14.3) | N/A (fixed) |
| FL strategy | Custom `SavingFedAvg` subclass of `flwr.server.strategy.FedAvg` (Section 14.6), `min_available_clients = min_fit_clients = min_evaluate_clients = 4` in the production run (see Section 19 for the reduced-client smoke-test exception) | `federated.min_clients` |
| Batch size | 64 (both centralized and federated local training) | `training.batch_size` |
| Train/test split | 80/20, stratified by multi-class label, performed **before** any encoder/scaler is fit (Section 14.2) | `dataset.test_size = 0.2` |
| SHAP explainer | `shap.GradientExplainer` | `explainability.method` |
| SHAP sample size | 100 randomly sampled test instances, seeded | `explainability.sample_size = 100` |
| SHAP background sample size | `min(100, len(X_train))` randomly sampled training instances, seeded, persisted to `outputs/artifacts/shap_background.npy` | `explainability.background_size = 100` |
| Dashboard framework | Streamlit, single app, 4 tabs | N/A (fixed) |
| Global random seed | 42 | `seed = 42` (top-level config key) |
| Minimum hardware | Windows 10/11, Intel i5, 8GB RAM, CPU-only | N/A (documentation only) |

**Rule:** No numeric hyperparameter, file path, or class count may be hardcoded inside any `src/` module. Every module reads these values from a loaded `config.yaml` dictionary passed in by the calling script (see Section 8), or derives `num_classes` from `class_mapping.json` as specified above.

---

## 7. Tech Stack (Fixed, Exact Versions Pinned)

```
python==3.10.*
tensorflow==2.15.1        # last release with Keras 2 as default; required for shap.GradientExplainer compatibility
flwr[simulation]==1.8.0   # pinned to the client_fn/strategy/start_simulation API used in Section 14.6
ray==2.6.3                # exact transitive dependency of flwr[simulation]==1.8.0; pinned explicitly so no open-ended
                           # install can silently substitute a different Ray release under the simulation engine
scikit-learn>=1.3,<1.5
pandas>=2.0,<2.3
numpy>=1.24,<1.27
shap==0.44.1              # known-compatible with tensorflow==2.15.1 / Keras 2 GradientExplainer path
streamlit>=1.30,<1.40
matplotlib>=3.7,<3.10
pyyaml>=6.0
pytest>=7.4
```

No substitution of any library in this list is permitted (e.g., do not substitute PyTorch for TensorFlow, do not substitute Dash for Streamlit). No version other than the exact pins above is permitted for `tensorflow`, `flwr`, `ray`, and `shap` — these four are pinned exactly because their APIs are not compatible across versions in ways that would silently break Section 14.6 (Flower simulation loop) or Section 14.10 (SHAP explainability). If the exact pinned `tensorflow==2.15.1` cannot be installed on the implementing agent's platform, the agent must stop and flag this as a build-blocking condition rather than silently installing a newer TensorFlow release — a newer release defaults to Keras 3, which is incompatible with the locked SHAP explainer.

---

## 8. Configuration System

**Rule:** ALL configurable values live in exactly one file: `configs/config.yaml`. No script may define a hyperparameter, file path, or magic number inline. Every script begins by loading this file via `src/utils/config_loader.py::load_config()`.

**Path construction rule:** Every file path built from two or more `config["paths"]` components must be constructed via `os.path.join(...)`. String concatenation with a literal `"/"` or `"\\"` is not permitted anywhere in `src/`, `experiments/`, or `dashboard/` — this guarantees identical behavior across the Windows target environment (Section 6) and any POSIX development machine.

### `configs/config.yaml` (create with exactly these keys)

```yaml
seed: 42

paths:
  raw_data_dir: "data/raw"
  raw_data_file: "edge_iiotset.csv"
  processed_data_dir: "data/processed"
  processed_data_file: "edge_iiotset_processed.csv"
  artifacts_dir: "outputs/artifacts"
  models_dir: "outputs/models"
  results_dir: "outputs/results"
  shap_plots_dir: "outputs/shap_plots"
  logs_dir: "logs"

dataset:
  name: "Edge-IIoTset"
  target_column: "Attack_type"          # native raw-CSV column (multi-class)
  raw_binary_column: "Attack_label"     # native raw-CSV column (binary) — removed exclusively by create_labels
  derived_label_column: "label"         # created by create_labels from target_column
  derived_binary_column: "binary_label" # created by create_labels from raw_binary_column
  normal_class_value: "Normal"          # exact string match used to derive binary_label == 0
  test_size: 0.2
  drop_columns:
    - "ip.src_host"
    - "ip.dst_host"
    - "arp.src.proto_ipv4"
    - "arp.dst.proto_ipv4"
    - "http.file_data"
    - "http.request.full_uri"
    - "tcp.options"
    - "tcp.payload"
    - "dns.qry.name.len"
    - "mqtt.msg"
  categorical_column_rule: "dtype_object"   # the only implemented rule (Section 14.2); passed explicitly into
                                             # infer_feature_column_types so the config key is never unused
  force_reprocess: false

model:
  conv_filters: 64
  conv_kernel_size: 3
  conv_padding: "same"
  conv_activation: "relu"
  pool_size: 2
  gru_units: 64
  gru_return_sequences: false
  dropout_rate: 0.3
  dense_units: 128
  dense_activation: "relu"
  output_activation: "softmax"

training:
  optimizer: "adam"   # documentation-only; Adam is the only implemented optimizer (Section 6)
  learning_rate: 0.001
  loss: "categorical_crossentropy"
  batch_size: 64
  centralized_epochs: 10
  validation_split: 0.2
  early_stopping_patience: 3
  early_stopping_monitor: "val_loss"

federated:
  num_clients: 4
  num_rounds: 15
  local_epochs: 2
  partition_strategy: "iid"
  min_available_clients: 4
  min_fit_clients: 4
  min_evaluate_clients: 4

explainability:
  method: "GradientExplainer"
  sample_size: 100
  background_size: 100

dashboard:
  title: "IIoT Federated Intrusion Detection Dashboard"
```

**Note on `drop_columns`:** This list contains only genuine identifier/high-cardinality columns. It deliberately does **not** include `Attack_label` — that native column is removed exclusively by `create_labels` (Section 14.1, Section 2.1), never by `drop_identifier_columns`, so there is exactly one function in the codebase responsible for native-column removal.

### `src/utils/config_loader.py`

**Function: `load_config`**
- **Purpose:** Load and parse `configs/config.yaml` into a Python dictionary, and validate required top-level keys exist.
- **Arguments:** `config_path: str = "configs/config.yaml"`
- **Returns:** `dict` — parsed YAML content.
- **Raises:** `FileNotFoundError` if `config_path` does not exist; `KeyError` if any of the top-level keys (`seed`, `paths`, `dataset`, `model`, `training`, `federated`, `explainability`, `dashboard`) is missing.
- **Reads:** `configs/config.yaml`.
- **Writes:** Nothing.
- **Dependencies:** `yaml` (PyYAML).
- **Example usage:**
  ```python
  config = load_config("configs/config.yaml")
  seed = config["seed"]
  ```

---

## 9. Folder Structure (Exact — Create Precisely This Layout)

```
iiot-fl-ids-project/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── edge_iiotset.csv                # user-provided; not generated by code
│   └── processed/
│       └── edge_iiotset_processed.csv      # generated by preprocessing pipeline
├── logs/
│   ├── preprocessing_<timestamp>.log
│   ├── centralized_training_<timestamp>.log
│   ├── federated_training_<timestamp>.log
│   ├── evaluation_<timestamp>.log
│   └── explainability_<timestamp>.log
├── outputs/
│   ├── artifacts/
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   ├── categorical_encoder.pkl
│   │   ├── train_indices.pkl
│   │   ├── test_indices.pkl
│   │   ├── feature_names.pkl
│   │   ├── class_mapping.json
│   │   ├── random_seed.txt
│   │   └── shap_background.npy
│   ├── models/
│   │   ├── centralized_best_model.h5
│   │   ├── centralized_last_model.h5
│   │   ├── centralized_model_summary.txt
│   │   ├── centralized_architecture.png
│   │   ├── federated_global_model.h5
│   │   ├── federated_model_summary.txt
│   │   └── federated_architecture.png
│   ├── results/
│   │   ├── centralized_history.csv
│   │   ├── centralized_training_time.txt
│   │   ├── federated_history.csv
│   │   ├── federated_training_time.txt
│   │   ├── comparison_table.csv
│   │   ├── convergence_plot.png
│   │   ├── centralized_confusion_matrix.png
│   │   ├── federated_confusion_matrix.png
│   │   ├── centralized_classification_report.txt
│   │   ├── federated_classification_report.txt
│   │   ├── centralized_training_curves.png
│   │   ├── class_distribution.png
│   │   └── model_size_comparison.csv
│   └── shap_plots/
│       ├── shap_summary_plot.png
│       └── shap_bar_plot.png
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── seed.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── load_dataset.py
│   │   └── encode_normalize.py
│   ├── partitioning/
│   │   ├── __init__.py
│   │   └── partition_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_gru.py
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client_app.py
│   │   └── server_app.py
│   ├── centralized/
│   │   ├── __init__.py
│   │   └── train_baseline.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── compare_fl_vs_centralized.py
│   └── explainability/
│       ├── __init__.py
│       └── shap_utils.py
├── dashboard/
│   └── app.py
├── experiments/
│   ├── run_preprocessing.py
│   ├── run_centralized.py
│   ├── run_federated.py
│   ├── run_evaluation.py
│   └── run_explainability.py
└── tests/
    ├── __init__.py
    ├── test_config_loader.py
    ├── test_preprocessing.py
    ├── test_partitioning.py
    ├── test_models.py
    ├── test_federated_loop.py
    ├── test_metrics.py
    └── test_shap_utils.py
```

### Directory Content Rules

| Directory | Contains | Naming Convention | Overwrite Rule |
|---|---|---|---|
| `data/raw/` | User-supplied Edge-IIoTset CSV. Never written by code. | Exactly `edge_iiotset.csv` (as set in `config.yaml paths.raw_data_file`) | Never overwritten by any script. |
| `data/processed/` | Output of the preprocessing pipeline: the full dataset (train and test rows together), with categorical encoding, scaling, and derived labels applied, using transformation parameters fit on the training partition only (Section 14.2). | Exactly `edge_iiotset_processed.csv` | Overwritten only if `dataset.force_reprocess: true` in config, OR the file does not yet exist. If it exists and `force_reprocess: false`, the preprocessing script must log a message and skip regeneration, reusing the existing file. |
| `outputs/artifacts/` | Fitted encoders/scalers (fit on the training partition only), saved train/test indices, feature name list, class mapping, seed record, persisted SHAP background sample. | Fixed filenames as listed in Section 9 tree. | Same rule as processed data: never regenerate train/test split if `train_indices.pkl`/`test_indices.pkl` already exist, unless `force_reprocess: true`. `shap_background.npy` is regenerated only when `experiments/run_explainability.py` is re-run. |
| `outputs/models/` | Trained Keras models (`.h5`), model summaries (`.txt`), architecture diagrams (`.png`). | Fixed filenames per tree above; distinguished by `centralized_` / `federated_` prefix. | Overwritten on every full training run (deterministic due to fixed seed, so overwriting produces identical results). |
| `outputs/results/` | CSV metrics, PNG plots, text reports, and training-time records from training and evaluation runs. | Fixed filenames per tree above. | `centralized_training_time.txt` is overwritten on every centralized training run; `federated_training_time.txt` is overwritten on every federated training run; all other files in this directory are overwritten on every evaluation run. |
| `outputs/shap_plots/` | SHAP visualization PNGs. | Fixed filenames per tree above. | Overwritten on every explainability run. |
| `logs/` | One log file per script execution. | `<fixed_script_name>_<YYYYMMDD_HHMMSS>.log`, where `<fixed_script_name>` is the literal string assigned in Section 13's naming table (not a Python module name). | **Never overwritten** — every run appends a new timestamped file, preserving full run history for audit purposes. |

---

## 10. Exact Artifact Flow

```
data/raw/edge_iiotset.csv
        │
        ▼  (src/preprocessing/load_dataset.py :: load_raw_dataset)
In-memory DataFrame (raw)
        │
        ▼  (src/preprocessing/load_dataset.py :: drop_identifier_columns)
DataFrame with identifier/high-cardinality columns removed (Attack_type, Attack_label still present)
        │
        ▼  (src/preprocessing/load_dataset.py :: create_labels)
DataFrame with label + binary_label created; Attack_type and Attack_label both removed
(create_labels is the sole owner of native-column removal — see Section 2.1, 14.1)
        │
        ▼  (src/preprocessing/encode_normalize.py :: infer_feature_column_types, using
        │   config["dataset"]["categorical_column_rule"])
categorical_columns, numeric_columns
        │
        ▼  (src/preprocessing/encode_normalize.py :: split_train_test)
train_indices, test_indices computed on the RAW (unencoded, unscaled) labeled DataFrame
        │
        ▼  (src/preprocessing/encode_normalize.py :: fit_categorical_encoder, fit_scaler, fit_label_encoder — each fit ONLY on df.loc[train_indices])
outputs/artifacts/{scaler.pkl, label_encoder.pkl, categorical_encoder.pkl}
        │
        ▼  (apply fitted encoder/scaler to the FULL dataset — train rows and test rows alike)
data/processed/edge_iiotset_processed.csv
        +
outputs/artifacts/{feature_names.pkl, class_mapping.json, train_indices.pkl, test_indices.pkl}
        │
        ▼  (experiments/run_preprocessing.py, after run_preprocessing_pipeline returns, calls
        │   src/evaluation/metrics.py :: generate_class_distribution_plot)
outputs/results/class_distribution.png
        │
        ├────────────────────────────────────────────┐
        ▼ (centralized path)                          ▼ (federated path)
src/centralized/train_baseline.py            src/partitioning/partition_data.py
(via prepare_model_ready_data;                (partitions ONLY the train_indices rows
 wraps training in time.time() timing)         into 4 IID client shards)
        │                                              │
        ▼                                              ▼
outputs/models/centralized_*.h5              src/federated/server_app.py
outputs/results/centralized_history.csv      (orchestrates client_app.py via
outputs/results/centralized_training_time.txt  flwr.simulation.start_simulation with
        │                                     a SavingFedAvg strategy; each client
        │                                     evaluates on the SAME shared global
        │                                     test split, never a per-client split;
        │                                     wraps start_simulation in time.time() timing)
        │                                              │
        │                                              ▼
        │                                   outputs/models/federated_global_model.h5
        │                                   (built by capturing SavingFedAvg.latest_parameters
        │                                    and loading them into a fresh build_cnn_gru model)
        │                                   outputs/results/federated_history.csv
        │                                   outputs/results/federated_training_time.txt
        │                                              │
        └──────────────────────┬───────────────────────┘
                                ▼
                src/evaluation/compare_fl_vs_centralized.py :: run_evaluation_pipeline
                (evaluates centralized_best_model.h5 and federated_global_model.h5
                 on the identical shared test split; loss always computed via
                 model.evaluate(), never independently recomputed; reads both
                 *_training_time.txt files; calls generate_confusion_matrix,
                 generate_classification_report, and generate_training_curves_plot
                 from src/evaluation/metrics.py)
                                │
                                ▼
        outputs/results/{comparison_table.csv, model_size_comparison.csv,
                          convergence_plot.png, centralized_training_curves.png,
                          *_confusion_matrix.png, *_classification_report.txt}
                                │
                                ▼
                src/explainability/shap_utils.py :: generate_shap_explanations
                (explains federated_global_model.h5; squeezes the trailing
                 channel dimension before any SHAP plotting call; persists
                 the seeded background sample used to build the explainer)
                                │
                                ▼
        outputs/shap_plots/{shap_summary_plot.png, shap_bar_plot.png}
                +
        outputs/artifacts/shap_background.npy
                                │
                                ▼
                        dashboard/app.py
                (reads ONLY from outputs/ — never trains, never
                 recomputes SHAP background sampling, never touches
                 data/raw and touches data/processed only to display
                 or evaluate individual sample rows; may call
                 prepare_model_ready_data and explain_single_prediction
                 as read-only, non-training utility functions — Section 11)
```

**Rule:** Every arrow above represents a strict producer → consumer relationship. A downstream script must never regenerate what an upstream script already produced; it must load the upstream script's saved artifact.

---

## 11. Module Dependency Graph (Import Rules)

The implementing agent must enforce these import directions. Any import that violates this graph is a defect.

```
src/utils/            → imported by: everything. Imports: nothing from src/.
src/preprocessing/    → imports: src/utils/. Imported by: src/partitioning/, src/centralized/, src/federated/,
                         src/evaluation/, src/explainability/, experiments/, dashboard/ (encode_normalize.py::
                         prepare_model_ready_data only — a pure, read-only tensor-preparation function; dashboard/
                         must never import run_preprocessing_pipeline or any other pipeline-execution function
                         from this package).
src/partitioning/     → imports: src/utils/. Imported by: src/federated/, experiments/.
src/models/           → imports: src/utils/. Imported by: src/centralized/, src/federated/, src/evaluation/,
                         src/explainability/, experiments/.
src/centralized/      → imports: src/utils/, src/preprocessing/, src/models/. Imported by: experiments/.
src/federated/        → imports: src/utils/, src/preprocessing/, src/partitioning/, src/models/. Imported by: experiments/.
src/evaluation/       → imports: src/utils/, src/preprocessing/ (for prepare_model_ready_data only), src/models/.
                         Imported by: experiments/, dashboard/ (metrics/plot-loading and plot-generation functions
                         only; no training-related imports permitted in dashboard/).
src/explainability/   → imports: src/utils/, src/preprocessing/ (for prepare_model_ready_data only), src/models/.
                         Imported by: experiments/, dashboard/ (shap_utils.py::explain_single_prediction only —
                         a pure, read-only single-sample explanation function).
dashboard/            → imports: src/utils/, src/evaluation/ (metrics/plot loading and plot-generation functions),
                         src/preprocessing/encode_normalize.py::prepare_model_ready_data,
                         src/explainability/shap_utils.py::explain_single_prediction.
                         MUST NOT import src/centralized/, src/federated/, src/partitioning/, or any of the
                         following functions from any module: run_preprocessing_pipeline, train_centralized_model,
                         run_federated_simulation, partition_iid, generate_shap_explanations.
experiments/          → imports: everything under src/. This is the only layer permitted to orchestrate
                         cross-module calls, including calling src/evaluation/metrics.py plot-generation
                         functions from experiments/run_preprocessing.py and experiments/run_evaluation.py.
```

**Forbidden:** Circular imports of any kind. `src/models/cnn_gru.py` must never import from `src/centralized/` or `src/federated/` (dependency flows one direction: models are consumed by training modules, never the reverse).

---

## 12. Reproducibility and Seeding

### `src/utils/seed.py`

**Function: `set_global_seed`**
- **Purpose:** Set every relevant random seed in the process to guarantee deterministic behavior across preprocessing, model initialization, training, partitioning, and SHAP sampling.
- **Arguments:** `seed: int = 42`
- **Returns:** `None`
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing.
- **Writes:** `outputs/artifacts/random_seed.txt` (writes the seed value used, as plain text, for audit purposes).
- **Dependencies:** `random` (Python stdlib), `numpy`, `tensorflow`.
- **Side effects (must set all of the following):**
  1. `random.seed(seed)`
  2. `numpy.random.seed(seed)`
  3. `tensorflow.random.set_seed(seed)`
  4. `os.environ["PYTHONHASHSEED"] = str(seed)`
- **Example usage:**
  ```python
  from src.utils.seed import set_global_seed
  set_global_seed(config["seed"])
  ```
- **Rule:** `set_global_seed()` must be called as the **first executable statement** in every script under `experiments/`, before any data loading, model building, or partitioning occurs.

### Additional Seed Propagation Requirements

| Location | Requirement |
|---|---|
| `train_test_split` (scikit-learn) | Must pass `random_state=config["seed"]` explicitly. |
| `partition_iid` | Must accept and use `seed` parameter, defaulting to `config["seed"]`, for shuffling before splitting. |
| Flower simulation | Must set `numpy`/`tensorflow` seeds inside each simulated client process as well, since Flower simulation may spawn separate Python processes/threads — call `set_global_seed()` inside `client_fn` before model instantiation. |
| SHAP sampling (test instances) | The random sample of 100 test instances must be drawn using `numpy.random.RandomState(config["seed"])`, not the global unseeded random state, to guarantee the exact same 100 samples are chosen on every run. |
| SHAP sampling (background instances) | The random sample of `min(100, len(X_train))` background instances must also be drawn using `numpy.random.RandomState(config["seed"])`, guaranteeing `outputs/artifacts/shap_background.npy` is byte-for-byte identical across runs. |
| Keras model weight initialization | Relies on `tensorflow.random.set_seed()` having already been called (see `set_global_seed`); no additional per-layer seeding is required. |

---

## 13. Logging Requirements

### `src/utils/logger.py`

**Function: `get_logger`**
- **Purpose:** Create and return a configured `logging.Logger` instance that writes to both console (INFO level) and a timestamped log file (DEBUG level) under `logs/`.
- **Arguments:** `name: str` — a fixed literal identifier for the calling script (see naming table below; this is **not** `__name__` of the calling module), `logs_dir: str` (from `config["paths"]["logs_dir"]`)
- **Returns:** `logging.Logger`
- **Raises:** `OSError` if `logs_dir` cannot be created.
- **Reads:** Nothing.
- **Writes:** Creates `logs_dir` if it does not exist; creates a new log file `logs/<name>_<YYYYMMDD_HHMMSS>.log` on each call.
- **Dependencies:** `logging` (stdlib), `datetime`, `os`.
- **Example usage:**
  ```python
  from src.utils.logger import get_logger
  logger = get_logger("centralized_training", config["paths"]["logs_dir"])
  logger.info("Starting centralized training run")
  ```

### Fixed Logger Names (must match Section 9's log filenames exactly)

| Script | `name` argument passed to `get_logger` |
|---|---|
| `experiments/run_preprocessing.py` | `"preprocessing"` |
| `experiments/run_centralized.py` | `"centralized_training"` |
| `experiments/run_federated.py` | `"federated_training"` |
| `experiments/run_evaluation.py` | `"evaluation"` |
| `experiments/run_explainability.py` | `"explainability"` |

Modules under `src/` that need a logger (e.g., inside `train_baseline.py`) must receive an already-constructed logger passed down from the calling `experiments/run_*.py` script rather than constructing their own — no module may introduce a distinct log-file name beyond the five listed above. This applies equally to `src/evaluation/metrics.py`'s plot-generation functions when called from `experiments/run_preprocessing.py` (uses the `"preprocessing"` logger) and from `experiments/run_evaluation.py` (uses the `"evaluation"` logger).

### Mandatory Log Contents Per Script

Every script under `experiments/` must log, at minimum, the following events (at `INFO` level unless noted):

| Event | Required Fields |
|---|---|
| Script start | Script name, timestamp, loaded config summary (all hyperparameters relevant to this script) |
| Dataset loaded | Row count, column count, class distribution (counts per class) |
| Preprocessing complete | Final feature count, train set size, test set size |
| Model built | Full `model.summary()` output (logged at `DEBUG` level, full text) |
| Training epoch/round complete (per epoch for centralized, per round for federated) | Epoch/round number, training loss, training accuracy, validation loss, validation accuracy (or federated equivalent: aggregated loss/accuracy) |
| Training complete | Total training time (seconds — the same value persisted to `*_training_time.txt`), final best metric value, path where model was saved |
| Evaluation complete | All computed metrics (Section 16), path where each report/plot was saved |
| Any exception | Full stack trace, logged at `ERROR` level, followed by the script raising the exception (never silently swallowed) |
| Script end | Timestamp, total script wall-clock duration |

---

## 14. Module Specifications and Function Contracts

Every function below must be implemented **exactly** as specified — exact name, exact argument names/types, exact return type. The implementing agent must not rename, merge, or split these functions.

### 14.1 `src/preprocessing/load_dataset.py`

**Function: `load_raw_dataset`**
- **Purpose:** Read the raw Edge-IIoTset CSV file into a pandas DataFrame.
- **Arguments:** `raw_path: str`
- **Returns:** `pandas.DataFrame`
- **Raises:** `FileNotFoundError` if `raw_path` does not exist.
- **Reads:** File at `raw_path`.
- **Writes:** Nothing.
- **Dependencies:** `pandas`, `src.utils.logger`.
- **Example usage:** `df = load_raw_dataset(os.path.join(config["paths"]["raw_data_dir"], config["paths"]["raw_data_file"]))`

**Function: `drop_identifier_columns`**
- **Purpose:** Remove columns listed in `config["dataset"]["drop_columns"]` (identifier/high-cardinality columns only — see the note in Section 8) from the DataFrame.
- **Arguments:** `df: pandas.DataFrame`, `columns_to_drop: list[str]`
- **Returns:** `pandas.DataFrame` (new DataFrame with columns removed; original is not mutated in place — use `df.drop(columns=..., errors="ignore")`)
- **Raises:** Nothing (uses `errors="ignore"` so missing columns do not raise).
- **Reads:** Nothing.
- **Writes:** Nothing.
- **Dependencies:** `pandas`.
- **Example usage:** `df = drop_identifier_columns(df, config["dataset"]["drop_columns"])`
- **Note:** This function never drops `Attack_type` or `Attack_label` — `config["dataset"]["drop_columns"]` deliberately excludes `Attack_label` (Section 8). Both native label-source columns are consumed and removed exclusively by `create_labels` below, which must always run after this function in the pipeline (Section 14.2).

**Function: `create_labels`**
- **Purpose:** Produce the derived `label` (multi-class) and `binary_label` (binary) columns from the native source columns, and remove both native source columns from the returned DataFrame so they cannot leak into the model's features. This function is the single, sole owner of native-column removal in the pipeline.
- **Arguments:** `df: pandas.DataFrame` (must still contain both `target_column` and `raw_binary_column` — this function is always called immediately after `drop_identifier_columns`, which never removes either of them), `target_column: str` (= `config["dataset"]["target_column"]`, i.e. `"Attack_type"`), `raw_binary_column: str` (= `config["dataset"]["raw_binary_column"]`, i.e. `"Attack_label"`), `normal_class_value: str` (= `config["dataset"]["normal_class_value"]`, i.e. `"Normal"`)
- **Returns:** `pandas.DataFrame` with:
  - `label` = exact copy of `df[target_column]` (string class names, unmodified).
  - `binary_label` = `0` where `df[target_column] == normal_class_value` (exact string equality, case-sensitive), else `1`. This is derived independently from `target_column`, not copied from `raw_binary_column` — `raw_binary_column` is read only to validate agreement (see below).
  - `target_column` (`Attack_type`) and `raw_binary_column` (`Attack_label`) are both dropped from the returned DataFrame after `label`/`binary_label` are created.
- **Validation performed inside the function:** Since `raw_binary_column` is guaranteed present at this point in the pipeline (Section 14.2's fixed execution order), the function always asserts that the derived `binary_label` agrees with `df[raw_binary_column]` for every row. If they disagree for any row, log a `WARNING` with the mismatch count (do not raise — the target-derived value is authoritative) before dropping `raw_binary_column`.
- **Raises:** `KeyError` if `target_column` or `raw_binary_column` is not present in `df`.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `pandas`, `numpy`, `src.utils.logger`.
- **Example usage:** `df = create_labels(df, config["dataset"]["target_column"], config["dataset"]["raw_binary_column"], config["dataset"]["normal_class_value"])`

### 14.2 `src/preprocessing/encode_normalize.py`

**Function: `infer_feature_column_types`**
- **Purpose:** Deterministically classify the DataFrame's remaining feature columns (after `drop_identifier_columns` and `create_labels` have run, so `label`/`binary_label` exist and `Attack_type`/`Attack_label` are gone) into categorical and numeric groups. This is the single, authoritative place this classification happens — no other module may re-derive it independently.
- **Arguments:** `df: pandas.DataFrame`, `label_columns: list[str] = ["label", "binary_label"]`, `rule: str = "dtype_object"` (sourced from `config["dataset"]["categorical_column_rule"]`)
- **Returns:** `tuple[list[str], list[str]]` — `(categorical_columns, numeric_columns)`.
- **Classification rule (exact, no discretion — the only implemented value of `rule` is `"dtype_object"`; any other value raises):**
  - `categorical_columns` = every column not in `label_columns` whose pandas dtype is `object` or `category`.
  - `numeric_columns` = every column not in `label_columns` whose dtype is any numeric dtype (`int*`, `float*`, `bool` treated as numeric).
- **Raises:** `ValueError` if `rule` is not `"dtype_object"`.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `pandas`.
- **Example usage:** `categorical_columns, numeric_columns = infer_feature_column_types(df, ["label", "binary_label"], config["dataset"]["categorical_column_rule"])`

**Function: `fit_categorical_encoder`**
- **Purpose:** Fit a `sklearn.preprocessing.OrdinalEncoder` on all categorical columns, using **only the training-partition rows** (`df.loc[train_indices]`, where `train_indices` comes from `split_train_test`, called beforehand — see the exact pipeline order below). `OrdinalEncoder` (not `OneHotEncoder`) is used to keep feature count fixed and Conv1D-compatible.
- **Arguments:** `df: pandas.DataFrame` (already sliced to training rows only by the caller), `categorical_columns: list[str]`
- **Returns:** `tuple[pandas.DataFrame, sklearn.preprocessing.OrdinalEncoder]` — the encoded DataFrame (training rows only) and the fitted encoder.
- **Raises:** `ValueError` if `categorical_columns` is empty and categorical columns exist in `df` but were not passed.
- **Reads:** Nothing. **Writes:** Nothing (saving to disk is handled by `run_preprocessing_pipeline`).
- **Dependencies:** `sklearn.preprocessing.OrdinalEncoder`, `pandas`.

**Function: `fit_scaler`**
- **Purpose:** Fit a `sklearn.preprocessing.MinMaxScaler` on all numeric feature columns (excluding `label` and `binary_label`), using **only the training-partition rows**.
- **Arguments:** `df: pandas.DataFrame` (already sliced to training rows only by the caller), `numeric_columns: list[str]`
- **Returns:** `tuple[pandas.DataFrame, sklearn.preprocessing.MinMaxScaler]`
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `sklearn.preprocessing.MinMaxScaler`, `pandas`.

**Function: `fit_label_encoder`**
- **Purpose:** Fit a `sklearn.preprocessing.LabelEncoder` on the `label` column of the **training-partition rows only**, to produce integer class indices, and build a `class_mapping` dict (`{str(class_index): class_name}` — keys are stringified integers, since JSON object keys are always strings).
- **Arguments:** `labels: pandas.Series` (the `label` column, sliced to training rows only by the caller)
- **Returns:** `tuple[numpy.ndarray, sklearn.preprocessing.LabelEncoder, dict]` — encoded integer labels (for the training rows passed in), fitted encoder, class mapping dict.
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `sklearn.preprocessing.LabelEncoder`.

**Function: `split_train_test`**
- **Purpose:** Perform a stratified 80/20 train/test split on the labeled DataFrame — **before** any categorical encoding or scaling has been fit or applied — indexed by row index (not by re-shuffling values), so indices can be persisted and reused.
- **Arguments:** `df: pandas.DataFrame` (post `drop_identifier_columns` + `create_labels`, pre-encoding/pre-scaling), `label_column: str` (`"label"`), `test_size: float`, `seed: int`
- **Returns:** `tuple[numpy.ndarray, numpy.ndarray]` — `train_indices`, `test_indices` (row index arrays into `df`).
- **Raises:** `ValueError` if any class has fewer than 2 samples (stratification requirement).
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `sklearn.model_selection.train_test_split`.
- **Example usage:**
  ```python
  train_idx, test_idx = split_train_test(df, "label", config["dataset"]["test_size"], config["seed"])
  ```

**Function: `prepare_model_ready_data`**
- **Purpose:** The single, shared function that converts the processed (encoded, scaled) tabular DataFrame plus a set of row indices into the exact tensors the CNN-GRU model consumes. This is the only place this reshape/one-hot conversion logic may live; `train_baseline.py`, `client_app.py`, `server_app.py`, `compare_fl_vs_centralized.py`, `shap_utils.py`, and `dashboard/app.py` must all call this function rather than re-implementing it.
- **Arguments:** `df: pandas.DataFrame` (the processed DataFrame, or a row-subset of it), `indices: numpy.ndarray` (rows to select), `feature_columns: list[str]` (from `feature_names.pkl`, in the fixed persisted order), `label_encoder: sklearn.preprocessing.LabelEncoder`, `num_classes: int` (derived from `class_mapping.json` per Section 6)
- **Returns:** `tuple[numpy.ndarray, numpy.ndarray]` — `X`, `y`:
  - `X`: shape `(len(indices), len(feature_columns), 1)` — `df.loc[indices, feature_columns].values.reshape(-1, len(feature_columns), 1)`.
  - `y`: shape `(len(indices), num_classes)` — integer-encoded via `label_encoder.transform(df.loc[indices, "label"])`, then one-hot encoded via `tensorflow.keras.utils.to_categorical(..., num_classes=num_classes)`.
- **Raises:** `ValueError` if any value in `df.loc[indices, "label"]` is unseen by `label_encoder` (i.e., not present in `class_mapping.json`).
- **Reads:** Nothing (pure function over its arguments). **Writes:** Nothing.
- **Dependencies:** `numpy`, `tensorflow.keras.utils.to_categorical`.
- **Example usage:**
  ```python
  X_train, y_train = prepare_model_ready_data(df, train_indices, feature_columns, label_encoder, num_classes)
  X_test,  y_test  = prepare_model_ready_data(df, test_indices,  feature_columns, label_encoder, num_classes)
  ```

**Function: `run_preprocessing_pipeline`**
- **Purpose:** Top-level orchestrator that runs the full, correctly-ordered preprocessing pipeline and persists every required artifact.
- **Arguments:** `config: dict`
- **Returns:** `None`
- **Raises:** Propagates any exception raised by the functions it calls, after logging the error at `ERROR` level.
- **Reads:** `data/raw/edge_iiotset.csv` (or configured path).
- **Exact execution order (fixed — must not be reordered):**
  1. `load_raw_dataset`
  2. `drop_identifier_columns` (removes only the identifier/high-cardinality columns in `config["dataset"]["drop_columns"]`; `Attack_type` and `Attack_label` are untouched here)
  3. `create_labels` (reads `Attack_type` and `Attack_label`, both still present; produces `label`/`binary_label`; removes both native columns — the sole point in the pipeline where this removal happens)
  4. `infer_feature_column_types` (on the full labeled DataFrame, using `config["dataset"]["categorical_column_rule"]`, to obtain `categorical_columns`/`numeric_columns`)
  5. `split_train_test` (on the full labeled, **unencoded, unscaled** DataFrame) → `train_indices`, `test_indices`
  6. `fit_categorical_encoder(df.loc[train_indices], categorical_columns)` → fitted encoder
  7. `fit_scaler(df.loc[train_indices], numeric_columns)` → fitted scaler
  8. `fit_label_encoder(df.loc[train_indices, "label"])` → fitted label encoder, `class_mapping`
  9. Apply the fitted encoder and scaler (fit in steps 6–7) to transform the **full** dataset — both `train_indices` and `test_indices` rows — producing the final encoded/scaled DataFrame.
  10. Persist the single transformed DataFrame as `data/processed/edge_iiotset_processed.csv` (contains both train and test rows, transformed using train-only-fit parameters).
  11. Persist all artifacts: `scaler.pkl`, `label_encoder.pkl`, `categorical_encoder.pkl`, `train_indices.pkl`, `test_indices.pkl`, `feature_names.pkl` (the fixed, ordered list of feature column names used to build `X`), `class_mapping.json`.
- **Dependencies:** All functions in this file, `src.preprocessing.load_dataset`, `src.utils.logger`, `src.utils.seed`, `pickle`, `json`.
- **Overwrite behavior:** If `config["dataset"]["force_reprocess"]` is `False` AND `data/processed/edge_iiotset_processed.csv` already exists, log an `INFO` message stating preprocessing was skipped, and return immediately without re-executing the pipeline.
- **Rationale for this exact order:** Fitting the categorical encoder, scaler, and label encoder on the full dataset before splitting would leak test-set distributional information into the training-time transformation, producing optimistic, non-representative test metrics. Splitting first and fitting all transformers on `train_indices` only guarantees a genuinely held-out test set. Running `create_labels` after `drop_identifier_columns` (rather than before, or interleaved) guarantees `create_labels` always receives a DataFrame in which both native label-source columns are still present, so its internal binary-agreement validation is always reachable.
- **Orchestration note:** `experiments/run_preprocessing.py` calls `run_preprocessing_pipeline(config)` and then, as a separate step, calls `src.evaluation.metrics.generate_class_distribution_plot` (Section 14.8) to produce `outputs/results/class_distribution.png`. This plot-generation call lives in the `experiments/` orchestration layer (which is permitted to import across all of `src/`) rather than inside `run_preprocessing_pipeline` itself, keeping `src/preprocessing/` free of any dependency on `src/evaluation/` per the import graph in Section 11.

### 14.3 `src/partitioning/partition_data.py`

**Function: `partition_iid`**
- **Purpose:** Split a DataFrame into `num_clients` equal-size, non-overlapping, randomly shuffled shards (IID partitioning). This function is applied **only** to the training-split portion of the processed data (the rows identified by `train_indices.pkl`) — it must never be applied to the test split.
- **Arguments:** `df: pandas.DataFrame`, `num_clients: int`, `seed: int`
- **Returns:** `list[pandas.DataFrame]` — length `num_clients`, each a disjoint subset of `df`, index reset (`reset_index(drop=True)`) on each shard.
- **Raises:** `ValueError` if `num_clients < 1` or `num_clients > len(df)`.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `pandas`, `numpy`.
- **Example usage:** `client_shards = partition_iid(df.loc[train_indices], config["federated"]["num_clients"], config["seed"])`
- **Guarantee to enforce:** `sum(len(shard) for shard in client_shards) == len(df)` and no row index appears in more than one shard.

**Federated client test-data semantics (authoritative — applies to Section 14.5):** Every `FLClient` instance receives its own disjoint training shard from `partition_iid` above, plus the **same shared global test split** (`test_indices.pkl` rows) — identical across all 4 clients, never a per-client partition of the test set. This guarantees that the per-round aggregated federated metric, every client's local `evaluate()` call, and the final `run_evaluation_pipeline` evaluation are all computed against the exact same held-out data used for the centralized baseline, so the federated-vs-centralized comparison differs only in how the model was trained, not in what it was evaluated against.

### 14.4 `src/models/cnn_gru.py`

**Function: `build_cnn_gru`**
- **Purpose:** Construct and compile the CNN-GRU Keras model per the exact architecture in Section 6.
- **Arguments:** `input_shape: tuple[int, int]` (`(num_features, 1)`), `num_classes: int` (derived from `class_mapping.json` per Section 6 — never hardcoded), `model_config: dict` (the `config["model"]` sub-dict), `training_config: dict` (the `config["training"]` sub-dict)
- **Returns:** `tensorflow.keras.Model` — already compiled with optimizer, loss, and `["accuracy"]` metric.
- **Raises:** `ValueError` if `num_classes < 2`.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `tensorflow.keras`.
- **Exact layer sequence to implement (fixed — padding and `return_sequences` are not configurable, only the sizes below are):**
  ```
  Input(shape=input_shape)
  Conv1D(filters=model_config["conv_filters"], kernel_size=model_config["conv_kernel_size"],
         padding=model_config["conv_padding"], activation=model_config["conv_activation"])
  MaxPooling1D(pool_size=model_config["pool_size"])
  GRU(units=model_config["gru_units"], return_sequences=model_config["gru_return_sequences"])
  Dropout(rate=model_config["dropout_rate"])
  Dense(units=model_config["dense_units"], activation=model_config["dense_activation"])
  Dense(units=num_classes, activation=model_config["output_activation"])
  ```
  `padding="same"` guarantees the Conv1D output sequence length equals `num_features` regardless of `kernel_size`, so the downstream `MaxPooling1D`/`GRU` shapes never depend on an unstated padding choice. `return_sequences=False` guarantees `GRU` emits a single final-state vector of shape `(batch, gru_units)` — no `Flatten` layer is used or needed, since `Dropout`/`Dense` operate directly on this 2D tensor.
- **Compile call:** `model.compile(optimizer=Adam(learning_rate=training_config["learning_rate"]), loss=training_config["loss"], metrics=["accuracy"])` — Adam is the only optimizer ever instantiated here; `training_config["optimizer"]` is not read or branched on.
- **Example usage:**
  ```python
  model = build_cnn_gru(input_shape=(num_features, 1), num_classes=num_classes, model_config=config["model"], training_config=config["training"])
  ```

**Function: `get_model_summary_string`**
- **Purpose:** Capture `model.summary()` output as a string (for logging and for saving to `*_model_summary.txt`).
- **Arguments:** `model: tensorflow.keras.Model`
- **Returns:** `str`
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** Nothing (caller writes the returned string to file).
- **Dependencies:** `io.StringIO`, `tensorflow.keras`.

**Function: `save_model_architecture_diagram`**
- **Purpose:** Save a visual diagram of the model architecture.
- **Arguments:** `model: tensorflow.keras.Model`, `output_path: str`
- **Returns:** `None`
- **Raises:** Logs a `WARNING` (does not raise) if `pydot`/`graphviz` are unavailable — architecture diagram generation is best-effort, not build-blocking.
- **Reads:** Nothing. **Writes:** PNG file at `output_path`.
- **Dependencies:** `tensorflow.keras.utils.plot_model`.

### 14.5 `src/federated/client_app.py`

**Class: `FLClient(flwr.client.NumPyClient)`**
- **Purpose:** Wrap one virtual client's local model and data shard for Flower's simulation engine.
- **`__init__` arguments:** `client_id: int`, `train_data: pandas.DataFrame` (this client's disjoint shard from `partition_iid`), `test_data: pandas.DataFrame` (the shared global test split — identical for every client; see Section 14.3), `config: dict`
- **Method: `get_parameters(self, config: dict) -> list[numpy.ndarray]`** — returns current local model weights.
- **Method: `fit(self, parameters: list[numpy.ndarray], config: dict) -> tuple[list[numpy.ndarray], int, dict]`**
  - Sets local model weights from `parameters`.
  - Builds `X_train, y_train` via `prepare_model_ready_data` on `self.train_data`.
  - Trains locally for `federated.local_epochs` epochs, batch size `training.batch_size`.
  - Returns `(updated_weights, num_training_examples, metrics_dict)` where `metrics_dict = {"loss": ..., "accuracy": ...}` taken from the final local epoch's `history.history` values. **Both keys are always present** — this dict is consumed exclusively by `weighted_average_fit` (Section 14.6), which requires both.
- **Method: `evaluate(self, parameters: list[numpy.ndarray], config: dict) -> tuple[float, int, dict]`**
  - Sets local model weights from `parameters`.
  - Builds `X_test, y_test` via `prepare_model_ready_data` on `self.test_data` (the shared global test split).
  - Evaluates via `model.evaluate(X_test, y_test)`.
  - Returns `(loss, num_test_examples, {"accuracy": accuracy_value})`. **Only `accuracy` is present in this dict** — `loss` is returned separately as the first tuple element, and Flower's simulation engine aggregates it natively (into `history.losses_distributed`) without going through a custom metrics-aggregation function. This dict is consumed exclusively by `weighted_average_eval` (Section 14.6), which reads only `accuracy`.
- **Raises:** Propagates any Keras training/evaluation exception after logging.
- **Reads:** Nothing beyond in-memory `train_data`/`test_data` passed at construction.
- **Writes:** Nothing (client does not persist models directly; only the server persists the final global model).
- **Dependencies:** `flwr.client.NumPyClient`, `src.models.cnn_gru`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.utils.seed`.
- **Critical requirement:** `set_global_seed(config["seed"])` must be called inside this client's training setup (see Section 12) before model instantiation, since Flower simulation may run clients in separate processes.

**Function: `client_fn`**
- **Purpose:** Factory function required by Flower's simulation engine to instantiate a client given a client ID.
- **Arguments:** `cid: str` (Flower passes client ID as a string)
- **Returns:** `flwr.client.Client` (an `FLClient` instance, converted via `.to_client()`)
- **Raises:** `IndexError` if `int(cid)` exceeds the number of pre-loaded partitions.
- **Reads:** Pre-partitioned client train shards (computed once via `partition_iid`, passed via closure — never re-read from disk on every call) and the single shared test split (also passed via closure).
- **Writes:** Nothing.
- **Dependencies:** `FLClient`, `src.partitioning.partition_data.partition_iid`.

### 14.6 `src/federated/server_app.py`

**Class: `SavingFedAvg(flwr.server.strategy.FedAvg)`**
- **Purpose:** A minimal `FedAvg` subclass that captures the most recently aggregated global model parameters, since neither Flower's `start_simulation` return value nor a stock `FedAvg` instance exposes the final round's aggregated parameters after the simulation ends.
- **Implementation (exact):**
  ```python
  class SavingFedAvg(fl.server.strategy.FedAvg):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.latest_parameters = None

      def aggregate_fit(self, server_round, results, failures):
          aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
          if aggregated_parameters is not None:
              self.latest_parameters = aggregated_parameters
          return aggregated_parameters, aggregated_metrics
  ```

**Function: `get_strategy`**
- **Purpose:** Construct the `SavingFedAvg` strategy object with all parameters from `config["federated"]`.
- **Arguments:** `config: dict`
- **Returns:** `SavingFedAvg` (never a plain `flwr.server.strategy.FedAvg` — the plain class does not expose final parameters and must not be used directly).
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `SavingFedAvg` (defined in this module).
- **Required strategy parameters:** `min_available_clients`, `min_fit_clients`, `min_evaluate_clients` all set to `config["federated"]["num_clients"]` (production value: 4; see Section 19 for the smoke test's independently-constructed reduced config); `fit_metrics_aggregation_fn` set to `weighted_average_fit`; `evaluate_metrics_aggregation_fn` set to `weighted_average_eval` (both below — these are two distinct functions, never the same function reused for both callback sites).

**Function: `weighted_average_fit`**
- **Purpose:** The example-count-weighted aggregation formula used for `fit_metrics_aggregation_fn`. Operates on `FLClient.fit()`'s metrics dicts, which always contain both `"loss"` and `"accuracy"` (Section 14.5).
- **Arguments:** `metrics: list[tuple[int, dict]]` (Flower passes a list of `(num_examples, metrics_dict)` pairs, one per participating client)
- **Returns:** `dict` — `{"accuracy": weighted_accuracy, "loss": weighted_loss}`
- **Exact formula:**
  ```python
  def weighted_average_fit(metrics: list[tuple[int, dict]]) -> dict:
      total_examples = sum(num_examples for num_examples, _ in metrics)
      weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
      weighted_loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
      return {"accuracy": weighted_accuracy, "loss": weighted_loss}
  ```
- **Raises:** Nothing under normal operation. **Reads/Writes:** Nothing.

**Function: `weighted_average_eval`**
- **Purpose:** The example-count-weighted aggregation formula used for `evaluate_metrics_aggregation_fn`. Operates on `FLClient.evaluate()`'s metrics dicts, which contain only `"accuracy"` (Section 14.5) — `evaluate`-time loss aggregation is handled natively by Flower via each client's returned `loss` tuple element and requires no custom function.
- **Arguments:** `metrics: list[tuple[int, dict]]`
- **Returns:** `dict` — `{"accuracy": weighted_accuracy}`
- **Exact formula:**
  ```python
  def weighted_average_eval(metrics: list[tuple[int, dict]]) -> dict:
      total_examples = sum(num_examples for num_examples, _ in metrics)
      weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
      return {"accuracy": weighted_accuracy}
  ```
- **Raises:** Nothing under normal operation. **Reads/Writes:** Nothing.

**Function: `run_federated_simulation`**
- **Purpose:** Top-level orchestrator: partitions data, launches the Flower simulation for `config["federated"]["num_rounds"]` rounds using the pinned `flwr==1.8.0` API, captures the final aggregated global model, times the run, and persists results.
- **Arguments:** `config: dict`
- **Returns:** `None`
- **Exact simulation call (fixed — do not use a different Flower entry point):**
  ```python
  import flwr as fl
  import time

  strategy = get_strategy(config)
  start_time = time.time()
  history = fl.simulation.start_simulation(
      client_fn=client_fn,
      num_clients=config["federated"]["num_clients"],
      config=fl.server.ServerConfig(num_rounds=config["federated"]["num_rounds"]),
      strategy=strategy,
  )
  training_time_seconds = time.time() - start_time
  ```
  This is the legacy client-function/strategy-based simulation API, pinned at `flwr==1.8.0` (Section 7). The newer app-based `flwr.simulation.run_simulation(server_app=..., client_app=..., num_supernodes=...)` entry point is explicitly out of scope (Section 5) and must not be used.
- **Global-weight capture (mandatory, not optional):** After `start_simulation` returns:
  1. Convert `strategy.latest_parameters` to NumPy arrays via `flwr.common.parameters_to_ndarrays`.
  2. Build a fresh model via `build_cnn_gru(...)` using the same `input_shape`/`num_classes`/`model_config`/`training_config` as every other model in this project.
  3. Call `model.set_weights(ndarrays)`.
  4. Save this model as `outputs/models/federated_global_model.h5` — this is the only source of the saved federated global model; a per-client local model must never be saved under this filename.
- **Reads:** `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/train_indices.pkl`, `outputs/artifacts/test_indices.pkl`, `outputs/artifacts/class_mapping.json` (for `num_classes`).
- **Writes:**
  - `outputs/models/federated_global_model.h5`
  - `outputs/models/federated_model_summary.txt`
  - `outputs/models/federated_architecture.png`
  - `outputs/results/federated_history.csv` (columns: `round, aggregated_loss, aggregated_accuracy`, populated from `history.metrics_distributed` / `history.losses_distributed` returned by `start_simulation`)
  - `outputs/results/federated_training_time.txt` (plain text, a single floating-point value: `training_time_seconds` as measured above)
- **Raises:** Propagates any Flower simulation exception after logging at `ERROR` level.
- **Dependencies:** `flwr.simulation.start_simulation`, `flwr.common.parameters_to_ndarrays`, `get_strategy`, `client_fn`, `src.models.cnn_gru`, `src.utils.logger`, `src.utils.seed`, `time`.

### 14.7 `src/centralized/train_baseline.py`

**Function: `train_centralized_model`**
- **Purpose:** Train the CNN-GRU model on the full training split (no partitioning), with early stopping, time the run, and persist all required artifacts.
- **Arguments:** `config: dict`
- **Returns:** `None`
- **Raises:** Propagates any Keras training exception after logging at `ERROR` level.
- **Reads:** `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/train_indices.pkl`, `outputs/artifacts/test_indices.pkl`, `outputs/artifacts/feature_names.pkl`, `outputs/artifacts/label_encoder.pkl`, `outputs/artifacts/class_mapping.json`.
- **Data source (exact):**
  ```python
  X_train, y_train = prepare_model_ready_data(df, train_indices, feature_columns, label_encoder, num_classes)
  X_test,  y_test  = prepare_model_ready_data(df, test_indices,  feature_columns, label_encoder, num_classes)
  ```
  `validation_split=config["training"]["validation_split"]` is applied by Keras internally to `(X_train, y_train)` — no separate held-out validation file is produced.
- **Writes:**
  - `outputs/models/centralized_best_model.h5` (lowest `val_loss` checkpoint, via `ModelCheckpoint(save_best_only=True, monitor="val_loss")`)
  - `outputs/models/centralized_last_model.h5` (final-epoch model state)
  - `outputs/models/centralized_model_summary.txt`
  - `outputs/models/centralized_architecture.png`
  - `outputs/results/centralized_history.csv` (columns: `epoch, loss, accuracy, val_loss, val_accuracy`)
  - `outputs/results/centralized_training_time.txt` (plain text, a single floating-point value: total wall-clock seconds measured via `time.time()` wrapping the `.fit()` call)
- **Exact training call specification (no ambiguity permitted):**
  ```
  Instantiate build_cnn_gru() from src/models/cnn_gru.py using config["model"] and config["training"], with num_classes derived from class_mapping.json.
  Record start_time = time.time() immediately before calling .fit().
  Fit on X_train, y_train with:
      epochs = config["training"]["centralized_epochs"]
      batch_size = config["training"]["batch_size"]
      validation_split = config["training"]["validation_split"]
      callbacks = [
          EarlyStopping(monitor=config["training"]["early_stopping_monitor"],
                        patience=config["training"]["early_stopping_patience"],
                        restore_best_weights=True),
          ModelCheckpoint(filepath="outputs/models/centralized_best_model.h5",
                           monitor=config["training"]["early_stopping_monitor"],
                           save_best_only=True)
      ]
  Record training_time_seconds = time.time() - start_time immediately after .fit() returns, and
  write it to outputs/results/centralized_training_time.txt.
  After fit() completes, save the in-memory model (which holds restored best weights
  due to restore_best_weights=True) as outputs/models/centralized_last_model.h5.
  ```
- **Authoritative model for comparison:** All downstream comparison/evaluation code (`run_evaluation_pipeline`, `comparison_table.csv`, `model_size_comparison.csv`) uses **`centralized_best_model.h5`** exclusively as "the" centralized model. `centralized_last_model.h5` is retained on disk for audit/debugging purposes only and is never read by any evaluation or dashboard code.
- **Dependencies:** `tensorflow.keras.callbacks.EarlyStopping`, `tensorflow.keras.callbacks.ModelCheckpoint`, `src.models.cnn_gru`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.utils.logger`, `src.utils.seed`, `time`.

### 14.8 `src/evaluation/metrics.py`

**Function: `compute_all_metrics`**
- **Purpose:** Compute the full metric suite for a set of predictions against ground truth.
- **Arguments:** `y_true: numpy.ndarray`, `y_pred: numpy.ndarray`, `class_names: list[str]`
- **Returns:** `dict` with keys: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `loss` (this project always has a compiled Keras model available at every call site, so `loss` is always populated by the caller from `model.evaluate()` — see Section 14.9 — and is never left as `None` in practice; the parameter exists for API generality only).
- **Raises:** `ValueError` if `y_true` and `y_pred` differ in length.
- **Reads:** Nothing. **Writes:** Nothing.
- **Dependencies:** `sklearn.metrics` (`accuracy_score`, `precision_recall_fscore_support`).

**Function: `generate_confusion_matrix`**
- **Purpose:** Compute and save a confusion matrix plot.
- **Arguments:** `y_true: numpy.ndarray`, `y_pred: numpy.ndarray`, `class_names: list[str]`, `output_path: str`
- **Returns:** `numpy.ndarray` (the raw confusion matrix)
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** PNG at `output_path`.
- **Dependencies:** `sklearn.metrics.confusion_matrix`, `matplotlib`.

**Function: `generate_classification_report`**
- **Purpose:** Compute and save the full per-class precision/recall/F1 report as text.
- **Arguments:** `y_true: numpy.ndarray`, `y_pred: numpy.ndarray`, `class_names: list[str]`, `output_path: str`
- **Returns:** `str` (the report text)
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** Text file at `output_path`.
- **Dependencies:** `sklearn.metrics.classification_report`.

**Function: `generate_class_distribution_plot`**
- **Purpose:** Plot a bar chart of sample counts per attack category (per unique value of the `label` column) from the labeled dataset.
- **Arguments:** `df: pandas.DataFrame` (the labeled DataFrame — the in-memory result of `run_preprocessing_pipeline`'s labeling step, or `data/processed/edge_iiotset_processed.csv` reloaded), `label_column: str` (`"label"`), `output_path: str`
- **Returns:** `None`
- **Raises:** `KeyError` if `label_column` is not present in `df`.
- **Reads:** Nothing (operates on the DataFrame passed in). **Writes:** PNG at `output_path`.
- **Dependencies:** `pandas`, `matplotlib`.
- **Called from:** `experiments/run_preprocessing.py`, immediately after `run_preprocessing_pipeline(config)` returns (Section 14.2's orchestration note). Output path: `outputs/results/class_distribution.png`.

**Function: `generate_training_curves_plot`**
- **Purpose:** Plot the centralized model's per-epoch loss and accuracy (train and validation) from its saved training history.
- **Arguments:** `history_csv_path: str` (`outputs/results/centralized_history.csv`), `output_path: str`
- **Returns:** `None`
- **Raises:** `FileNotFoundError` if `history_csv_path` does not exist.
- **Reads:** `outputs/results/centralized_history.csv`. **Writes:** PNG at `output_path`.
- **Dependencies:** `pandas`, `matplotlib`.
- **Called from:** `experiments/run_evaluation.py`, via `run_evaluation_pipeline` (Section 14.9). Output path: `outputs/results/centralized_training_curves.png`.

### 14.9 `src/evaluation/compare_fl_vs_centralized.py`

**Function: `evaluate_model_on_test_set`**
- **Purpose:** Load a trained model and evaluate it on the shared held-out test set, measuring both accuracy metrics and timing metrics.
- **Arguments:** `model_path: str`, `X_test: numpy.ndarray`, `y_test: numpy.ndarray`, `class_names: list[str]`
- **Exact computation (fixed — no alternative method permitted):**
  ```python
  model = tensorflow.keras.models.load_model(model_path)
  loss_value, accuracy_value = model.evaluate(X_test, y_test, verbose=0)
  start = time.time()
  y_pred_probs = model.predict(X_test, verbose=0)
  inference_time_seconds = time.time() - start
  y_pred = numpy.argmax(y_pred_probs, axis=1)
  y_true = numpy.argmax(y_test, axis=1)   # y_test is one-hot, from prepare_model_ready_data
  metrics = compute_all_metrics(y_true, y_pred, class_names)
  metrics["loss"] = loss_value   # always sourced from model.evaluate(), never recomputed independently
  ```
  If `metrics["accuracy"]` (from `compute_all_metrics`) diverges from `accuracy_value` (from `model.evaluate`) by more than `1e-6`, log a `WARNING` — both are expected to agree up to floating-point rounding.
- **Returns:** `dict` with keys: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `loss`, `inference_time_seconds`, `prediction_latency_ms_per_sample` (`inference_time_seconds / len(X_test) * 1000`), `model_size_mb` (file size of `model_path` in megabytes, `os.path.getsize(model_path) / (1024*1024)`), `y_pred` (`numpy.ndarray`, for downstream confusion-matrix/classification-report generation), `y_true` (`numpy.ndarray`, same purpose).
- **Raises:** `FileNotFoundError` if `model_path` does not exist.
- **Reads:** Model file at `model_path`.
- **Writes:** Nothing (caller persists results).
- **Dependencies:** `tensorflow.keras.models.load_model`, `src.evaluation.metrics.compute_all_metrics`, `time`, `os`, `numpy`.

**Function: `generate_comparison_table`**
- **Purpose:** Build the final side-by-side comparison table from both models' evaluation results and both models' persisted training-time files. The "Centralized" row always evaluates `outputs/models/centralized_best_model.h5` (Section 14.7); the "Federated" row always evaluates `outputs/models/federated_global_model.h5` (Section 14.6).
- **Arguments:** `centralized_metrics: dict` (from `evaluate_model_on_test_set`), `federated_metrics: dict` (from `evaluate_model_on_test_set`), `centralized_training_time_seconds: float` (read from `outputs/results/centralized_training_time.txt` by the caller), `federated_training_time_seconds: float` (read from `outputs/results/federated_training_time.txt` by the caller), `output_path: str`
- **Returns:** `pandas.DataFrame` (also saved to disk)
- **Required columns:** `Model` (`"Centralized"` / `"Federated"`), `Accuracy`, `Precision_Macro`, `Recall_Macro`, `F1_Macro`, `Precision_Weighted`, `Recall_Weighted`, `F1_Weighted`, `Loss`, `Training_Time_Seconds`, `Inference_Time_Seconds`, `Prediction_Latency_ms_per_sample`, `Model_Size_MB`.
- **Raises:** Nothing under normal operation.
- **Reads:** Nothing. **Writes:** CSV at `output_path`.
- **Dependencies:** `pandas`.

**Function: `generate_model_size_comparison`**
- **Purpose:** Produce the `model_size_comparison.csv` deliverable (Section 9/24), comparing the on-disk size and parameter count of the two authoritative models.
- **Arguments:** `centralized_model_path: str` (path to `centralized_best_model.h5`), `federated_model_path: str` (path to `federated_global_model.h5`), `output_path: str`
- **Returns:** `pandas.DataFrame` with exactly two rows and columns: `Model` (`"Centralized"` / `"Federated"`), `Model_Size_MB` (`os.path.getsize(path) / (1024*1024)`), `Parameter_Count` (`model.count_params()` after loading each model via `tensorflow.keras.models.load_model`).
- **Raises:** `FileNotFoundError` if either path does not exist.
- **Reads:** Both `.h5` files (loaded only to call `count_params()`). **Writes:** CSV at `output_path`.
- **Dependencies:** `tensorflow.keras.models.load_model`, `pandas`, `os`.
- **Called from:** `run_evaluation_pipeline` (below), immediately after `generate_comparison_table`.

**Function: `generate_convergence_plot`**
- **Purpose:** Plot federated per-round accuracy alongside centralized per-epoch accuracy on the same figure for visual comparison.
- **Arguments:** `federated_history_path: str`, `centralized_history_path: str`, `output_path: str`
- **Returns:** `None`
- **Raises:** `FileNotFoundError` if either history CSV is missing.
- **Reads:** Both history CSVs. **Writes:** PNG at `output_path`.
- **Dependencies:** `pandas`, `matplotlib`.

**Function: `run_evaluation_pipeline`**
- **Purpose:** The single top-level orchestrator for the entire evaluation stage, called by `experiments/run_evaluation.py`. Owns every call required to produce all Section 9/17/24 evaluation-stage deliverables, so no evaluation-stage logic is left implicit or unassigned to any function contract in this document.
- **Arguments:** `config: dict`
- **Returns:** `None`
- **Exact sequence of operations (fixed):**
  1. Load `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/test_indices.pkl`, `outputs/artifacts/feature_names.pkl`, `outputs/artifacts/label_encoder.pkl`, `outputs/artifacts/class_mapping.json`; derive `num_classes = len(class_mapping)`.
  2. Build `X_test, y_test` via `prepare_model_ready_data(df, test_indices, feature_columns, label_encoder, num_classes)`.
  3. Call `evaluate_model_on_test_set("outputs/models/centralized_best_model.h5", X_test, y_test, class_names)` → `centralized_metrics`.
  4. Call `evaluate_model_on_test_set("outputs/models/federated_global_model.h5", X_test, y_test, class_names)` → `federated_metrics`.
  5. Read `outputs/results/centralized_training_time.txt` and `outputs/results/federated_training_time.txt` as floats.
  6. Call `generate_comparison_table(centralized_metrics, federated_metrics, centralized_training_time_seconds, federated_training_time_seconds, "outputs/results/comparison_table.csv")`.
  7. Call `generate_model_size_comparison("outputs/models/centralized_best_model.h5", "outputs/models/federated_global_model.h5", "outputs/results/model_size_comparison.csv")`.
  8. Call `generate_convergence_plot("outputs/results/federated_history.csv", "outputs/results/centralized_history.csv", "outputs/results/convergence_plot.png")`.
  9. Call `src.evaluation.metrics.generate_training_curves_plot("outputs/results/centralized_history.csv", "outputs/results/centralized_training_curves.png")`.
  10. Call `src.evaluation.metrics.generate_confusion_matrix(centralized_metrics["y_true"], centralized_metrics["y_pred"], class_names, "outputs/results/centralized_confusion_matrix.png")`.
  11. Call `src.evaluation.metrics.generate_confusion_matrix(federated_metrics["y_true"], federated_metrics["y_pred"], class_names, "outputs/results/federated_confusion_matrix.png")`.
  12. Call `src.evaluation.metrics.generate_classification_report(centralized_metrics["y_true"], centralized_metrics["y_pred"], class_names, "outputs/results/centralized_classification_report.txt")`.
  13. Call `src.evaluation.metrics.generate_classification_report(federated_metrics["y_true"], federated_metrics["y_pred"], class_names, "outputs/results/federated_classification_report.txt")`.
- **Raises:** Propagates any exception raised by the functions it calls, after logging at `ERROR` level.
- **Reads/Writes:** The union of every read/write listed in the functions it calls, above.
- **Dependencies:** All functions in this file, `src.evaluation.metrics`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.utils.logger`, `src.utils.seed`.
- **Example usage:** `experiments/run_evaluation.py` calls `set_global_seed(config["seed"])` first, then `run_evaluation_pipeline(config)`.

### 14.10 `src/explainability/shap_utils.py`

**Function: `generate_shap_explanations`**
- **Purpose:** Sample a seeded background set from the training data, sample a seeded set of test instances, compute SHAP values for those test instances, save summary/bar plots, and persist the background sample for reuse by the dashboard.
- **Arguments:** `model: tensorflow.keras.Model`, `X_train: numpy.ndarray` (shape `(n_train, num_features, 1)`, the full training tensor produced via `prepare_model_ready_data`), `test_samples_full: numpy.ndarray` (shape `(n_test, num_features, 1)`, the full test tensor produced via `prepare_model_ready_data`), `class_names: list[str]`, `output_dir: str`, `background_output_path: str` (`outputs/artifacts/shap_background.npy`), `seed: int`, `background_size: int` (= `config["explainability"]["background_size"]`), `sample_size: int` (= `config["explainability"]["sample_size"]`)
- **Background-sampling requirement:** Select `min(background_size, len(X_train))` background instances via `numpy.random.RandomState(seed).choice(len(X_train), size=min(background_size, len(X_train)), replace=False)`. Persist the selected background array (shape `(min(background_size, len(X_train)), num_features, 1)`) to `background_output_path` via `numpy.save`. This is the same background array used to construct the `shap.GradientExplainer` below, guaranteeing the persisted file and the explainer used to produce `shap_summary_plot.png`/`shap_bar_plot.png` are identical.
- **Test-sampling requirement:** Select the `sample_size` test instances via a **separate, independently-seeded** draw: `numpy.random.RandomState(seed).choice(len(test_samples_full), size=sample_size, replace=False)` — this must be reproducible across runs, and must not use the unseeded global random state. (Using the same `RandomState(seed)` constructor for both the background and test draws, called as two independent `.choice(...)` invocations in this fixed order — background first, test second — guarantees both draws are exactly reproducible across runs.)
- **Multi-class output shape handling (fixed procedure — required before any plotting call):** With input shape `(n, num_features, 1)`, `shap.GradientExplainer` returns, for multi-class output, a `list[numpy.ndarray]` of length `num_classes`, each shaped `(n, num_features, 1)`. Before plotting:
  1. Squeeze the trailing channel dimension on every per-class array: `shap_values_2d = [sv.squeeze(axis=-1) for sv in shap_values]`.
  2. Squeeze the input sample array identically: `test_samples_2d = test_samples.squeeze(axis=-1)`.
  3. Pass `shap_values_2d` and `test_samples_2d` (both now 2D, `(n, num_features)`) to `shap.summary_plot(...)`, together with `feature_names` loaded from `feature_names.pkl`.
  4. For the bar plot, aggregate mean absolute SHAP value per feature across all classes (`numpy.mean([numpy.abs(sv) for sv in shap_values_2d], axis=0)`, then mean over samples) before calling `shap.summary_plot(..., plot_type="bar")`.
- **Returns:** `list[numpy.ndarray]` — the per-class SHAP values in their original (pre-squeeze) 3D shape, so callers other than the plotting routines still receive shapes consistent with the model's input.
- **Raises:** Propagates any SHAP runtime exception after logging.
- **Reads:** Nothing beyond passed-in arrays. **Writes:** `outputs/shap_plots/shap_summary_plot.png`, `outputs/shap_plots/shap_bar_plot.png`, `outputs/artifacts/shap_background.npy`.
- **Dependencies:** `shap.GradientExplainer`, `matplotlib`, `numpy`, `numpy.random.RandomState`.

**Function: `explain_single_prediction`**
- **Purpose:** Compute and return SHAP values for exactly one sample, for use by the dashboard's per-prediction explanation view. Applies the identical squeeze steps as `generate_shap_explanations` before returning, so the dashboard receives a 2D `(num_features,)`-aligned array ready for direct bar-chart rendering. Never re-samples or regenerates the background distribution — the caller (`dashboard/app.py`) is responsible for loading the persisted `outputs/artifacts/shap_background.npy` and constructing `explainer` from it before calling this function.
- **Arguments:** `model: tensorflow.keras.Model`, `explainer: shap.GradientExplainer` (constructed by the caller from the persisted `shap_background.npy` array and `model`), `sample: numpy.ndarray` (shape `(1, num_features, 1)`, matching a single model input)
- **Returns:** SHAP values for the given sample, squeezed to 2D per-class structure consistent with `generate_shap_explanations`'s plotting representation.
- **Raises:** Propagates any SHAP runtime exception.
- **Reads:** Nothing. **Writes:** Nothing (dashboard handles rendering).
- **Dependencies:** `shap.GradientExplainer`.

---

## 15. Model Persistence Rules

| Rule | Detail |
|---|---|
| Best model definition | The checkpoint with the lowest `val_loss` observed during training (centralized path only — federated path saves only the final round's aggregated global model, captured explicitly via `SavingFedAvg.latest_parameters` per Section 14.6, since Flower's simulation does not natively checkpoint per-round). |
| Last model definition | The model state at the final completed epoch (centralized) or final completed round (federated). |
| Authoritative model for comparison | `centralized_best_model.h5` and `federated_global_model.h5` are the only two models read by any evaluation, comparison, explainability, or dashboard code. `centralized_last_model.h5` exists for audit purposes only. |
| Format | All models saved in Keras native `.h5` format via `model.save(path)`. |
| Training history | Every training run must produce a CSV with one row per epoch (centralized) or per round (federated), containing at minimum loss and accuracy (train and validation/aggregated as applicable), plus a companion `*_training_time.txt` recording total wall-clock training duration in seconds. |
| Model summary | `get_model_summary_string()` output written to a `.txt` file alongside every saved model. |
| Architecture diagram | `save_model_architecture_diagram()` output written to a `.png` file alongside every saved model; failures in diagram generation must be logged as `WARNING`, not treated as a fatal error. |
| Optimizer state | Since `model.save(...)` in `.h5` format includes optimizer state by default in TensorFlow/Keras, no separate optimizer-state file is required — this is satisfied automatically by the standard save call. |

---

## 16. Evaluation Requirements

Every evaluation run (centralized and federated, evaluated independently on the same held-out test set) must compute and report:

- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- Macro F1-score
- Weighted F1-score
- Loss (categorical cross-entropy on the test set, always sourced from `model.evaluate()` per Section 14.9)
- Confusion Matrix (saved as PNG)
- Full Classification Report (per-class precision/recall/F1, saved as text)
- Training Time (seconds, measured via `time.time()` wrapping the `.fit()` call for the centralized path, or wrapping `start_simulation(...)` for the federated path — persisted to `*_training_time.txt` by each respective training script, per Section 14.6/14.7, and read back by `run_evaluation_pipeline`, Section 14.9)
- Inference Time (seconds, measured via `time.time()` wrapping the `.predict()` call on the full test set)
- Prediction Latency (milliseconds per sample, derived from inference time)
- Model Size (megabytes, derived from the saved `.h5` file's size on disk) and Parameter Count (via `model.count_params()`)

ROC/PR curves are **not required** for this multi-class setup unless the implementing agent also computes one-vs-rest curves per class — this is optional and, if implemented, must be clearly labeled as a per-class one-vs-rest curve, not a binary ROC curve.

---

## 17. Visualization Requirements

The following plots are mandatory deliverables. Each has exactly one owning function and one calling script — no plot in this table has an ambiguous or alternative source.

| Plot | Owning Function | Called From | Output File |
|---|---|---|---|
| Class distribution (bar chart of sample counts per attack category) | `src/evaluation/metrics.py::generate_class_distribution_plot` | `experiments/run_preprocessing.py`, immediately after `run_preprocessing_pipeline` returns | `outputs/results/class_distribution.png` |
| Centralized training curves (loss/accuracy vs. epoch, train and validation) | `src/evaluation/metrics.py::generate_training_curves_plot` | `experiments/run_evaluation.py`, via `run_evaluation_pipeline` | `outputs/results/centralized_training_curves.png` |
| Federated convergence plot (aggregated accuracy vs. round) overlaid with centralized (accuracy vs. epoch) | `src/evaluation/compare_fl_vs_centralized.py::generate_convergence_plot` | `experiments/run_evaluation.py`, via `run_evaluation_pipeline` | `outputs/results/convergence_plot.png` |
| Confusion matrix — centralized model | `src/evaluation/metrics.py::generate_confusion_matrix` | `experiments/run_evaluation.py`, via `run_evaluation_pipeline` | `outputs/results/centralized_confusion_matrix.png` |
| Confusion matrix — federated model | `src/evaluation/metrics.py::generate_confusion_matrix` | `experiments/run_evaluation.py`, via `run_evaluation_pipeline` | `outputs/results/federated_confusion_matrix.png` |
| SHAP summary plot (beeswarm, global feature importance) | `src/explainability/shap_utils.py::generate_shap_explanations` | `experiments/run_explainability.py` | `outputs/shap_plots/shap_summary_plot.png` |
| SHAP bar plot (mean absolute SHAP value per feature) | `src/explainability/shap_utils.py::generate_shap_explanations` | `experiments/run_explainability.py` | `outputs/shap_plots/shap_bar_plot.png` |

---

## 18. Dashboard Specification (Read-Only Enforcement)

**Absolute rule:** `dashboard/app.py` must NEVER call any training function (`train_centralized_model`, `run_federated_simulation`), NEVER call `partition_iid`, NEVER call `run_preprocessing_pipeline`, and NEVER call `generate_shap_explanations`. It reads exclusively from `outputs/` and `data/processed/` (for display purposes only, e.g., showing a sample row), and may call only the two read-only utility functions explicitly permitted by Section 11: `prepare_model_ready_data` and `explain_single_prediction`.

### Tab 1: Overview
- Load and display `outputs/results/class_distribution.png`.
- Display basic dataset stats (row count, feature count, number of classes) read from `outputs/artifacts/feature_names.pkl` and `outputs/artifacts/class_mapping.json` (`num_classes = len(class_mapping)`).
- **If missing:** Display `st.error("Preprocessing artifacts not found. Run experiments/run_preprocessing.py first.")` and halt rendering of this tab only (other tabs must still attempt to render independently).

### Tab 2: Predictions
- Load `outputs/models/federated_global_model.h5` via `tensorflow.keras.models.load_model` (cache with `@st.cache_resource`).
- Load test set via `outputs/artifacts/test_indices.pkl` applied to `data/processed/edge_iiotset_processed.csv`, converted via `prepare_model_ready_data`.
- Provide a `st.selectbox` or `st.slider` to pick a test sample index.
- Display predicted class (via `class_mapping.json` reverse lookup, casting keys to `int`) vs. true class.
- **If missing artifacts:** Display `st.error("Trained model or test data not found. Run experiments/run_federated.py and experiments/run_preprocessing.py first.")`.

### Tab 3: FL vs. Centralized
- Load and display `outputs/results/comparison_table.csv` as a `st.dataframe`.
- Load and display `outputs/results/model_size_comparison.csv` as a `st.dataframe`.
- Load and display `outputs/results/convergence_plot.png`.
- Load and display both confusion matrix PNGs side by side (`st.columns(2)`).
- **If missing:** Display `st.error("Comparison results not found. Run experiments/run_evaluation.py first.")`.

### Tab 4: Explainability
- Load the persisted background sample from `outputs/artifacts/shap_background.npy` via `numpy.load` (cache with `@st.cache_resource`), and construct a `shap.GradientExplainer(model, background)` instance from it and the cached model from Tab 2 (also cached with `@st.cache_resource`) — this explainer is never rebuilt from a freshly re-sampled background; it always uses the exact array persisted by `experiments/run_explainability.py`.
- For the sample selected in Tab 2 (use `st.session_state` to share selection across tabs), call `explain_single_prediction(model, explainer, sample)` (Section 14.10).
- Render the resulting SHAP values as a bar chart of top contributing features for that specific prediction.
- Also display the precomputed `outputs/shap_plots/shap_summary_plot.png` as the "global" explanation view.
- **If missing:** Display `st.error("SHAP artifacts not found. Run experiments/run_explainability.py first.")`.

---

## 19. Testing Requirements

Every listed test file must exist and pass before the project is considered complete.

| Test File | Required Test Cases |
|---|---|
| `tests/test_config_loader.py` | Loads a valid config successfully; raises `FileNotFoundError` on missing file; raises `KeyError` on a config missing a required top-level key. |
| `tests/test_preprocessing.py` | `load_raw_dataset` raises `FileNotFoundError` on bad path; `drop_identifier_columns` removes only the configured identifier columns, ignores missing ones without error, and never removes `Attack_type`/`Attack_label`; `create_labels` produces both `label` and `binary_label` columns with no NaNs, and both `Attack_type` and `Attack_label` are absent from the returned DataFrame; `infer_feature_column_types` correctly separates a mixed dummy DataFrame into categorical/numeric lists using `rule="dtype_object"`, excludes `label`/`binary_label` from both, and raises `ValueError` for any other `rule` value; `split_train_test` produces non-overlapping index sets whose union covers the full dataset; `prepare_model_ready_data` produces `X` of shape `(n, num_features, 1)` and one-hot `y` of shape `(n, num_classes)` for a small dummy processed DataFrame; encoders/scaler fit only on a training slice produce different parameters than if fit on the full dataset (regression test guarding against the leakage bug the pipeline order in Section 14.2 exists to prevent). |
| `tests/test_partitioning.py` | `partition_iid` produces exactly `num_clients` shards; shard sizes sum to original dataset size; no row index appears in more than one shard; raises `ValueError` on `num_clients < 1`. |
| `tests/test_models.py` | `build_cnn_gru` returns a compiled `tf.keras.Model`; a forward pass on a dummy batch of shape `(batch_size, num_features, 1)` produces output shape `(batch_size, num_classes)`; raises `ValueError` on `num_classes < 2`; confirms `Conv1D` uses `padding="same"` and `GRU` uses `return_sequences=False` (via inspecting layer config). |
| `tests/test_federated_loop.py` | Constructs an independent, reduced-scale copy of `config` for this test only — `federated.num_clients = 2` and `federated.min_available_clients = federated.min_fit_clients = federated.min_evaluate_clients = 2` — so that `get_strategy`'s minimum-client requirements are satisfiable by a 2-client simulation; the production `config.yaml` (`num_clients = 4`) is never used by this test. Using this reduced config, a 2-round, 2-client smoke-test simulation (small subset of data, not the full pipeline) built via `get_strategy(reduced_config)` and `fl.simulation.start_simulation` completes without raising an exception, produces a non-empty `federated_history.csv`-equivalent in-memory result, and confirms `strategy.latest_parameters` is not `None` after the run. |
| `tests/test_metrics.py` | `compute_all_metrics` returns all required dict keys; raises `ValueError` on mismatched `y_true`/`y_pred` lengths; known-input sanity check (e.g., perfect predictions yield `accuracy == 1.0`); `generate_class_distribution_plot` raises `KeyError` when `label_column` is absent and produces a non-empty PNG on valid input; `generate_training_curves_plot` raises `FileNotFoundError` on a missing history CSV and produces a non-empty PNG on valid input. |
| `tests/test_shap_utils.py` | `generate_shap_explanations` runs on a small dummy model/dataset without raising; output plots are created on disk at the expected paths; `outputs/artifacts/shap_background.npy`-equivalent output array is created with shape `(min(background_size, len(X_train)), num_features, 1)`; confirms the squeeze step produces 2D arrays of shape `(n, num_features)` before any plotting call; `explain_single_prediction`, given an explainer built from a persisted background array (not a freshly-sampled one), returns SHAP values without raising. |

Test categories to include across the above files, as applicable per module: **unit tests** (single function, isolated inputs), **integration tests** (multiple functions/modules working together, e.g., preprocessing → partitioning), **smoke tests** (does the federated simulation run end-to-end at all, even with trivial data, using the reduced-client test config described above).

---

## 20. Coding Standards

- All public functions must have full Python type hints on arguments and return values.
- All public functions must have Google-style project-docstrings (Purpose, Args, Returns, Raises sections at minimum).
- Code must conform to PEP8 (enforceable via `flake8` or `black`, implementing agent's choice of tool, but the resulting code must pass standard PEP8 checks).
- No duplicated logic: if the same preprocessing/model-building/metric-computation logic is needed in two places, it must be extracted into a shared function in the appropriate `src/` module — never copy-pasted. In particular, tensor preparation (reshape + one-hot encoding) lives exclusively in `prepare_model_ready_data` (Section 14.2), feature-column classification lives exclusively in `infer_feature_column_types` (Section 14.2), native-label-column derivation and removal lives exclusively in `create_labels` (Section 14.1), and plot generation for each named plot in Section 17 lives exclusively in its one listed owning function — no call site may reimplement any of these.
- No global mutable variables. Configuration and state must be passed explicitly as function arguments/return values.
- Single Responsibility Principle: each function does exactly one job as described in its contract in Section 14; orchestration (calling multiple functions in sequence) belongs only in `run_*` functions or `experiments/` scripts, never mixed into a low-level function.
- All file paths built from `config["paths"]` components use `os.path.join(...)`, never string concatenation (Section 8).
- No circular imports (see Section 11's dependency graph).
- No unused configuration keys: every key present in `configs/config.yaml` must be read by at least one function contract in this document (e.g., `dataset.categorical_column_rule` is read by `infer_feature_column_types`, Section 14.2).

---

## 21. Implementation Order (Strict Dependency-Ordered Milestones)

Each milestone lists required inputs, expected outputs, and validation checks. **No milestone may begin before all its listed dependencies are complete and validated.**

### Milestone 0 — Project Scaffolding
- **Depends on:** Nothing.
- **Inputs:** None.
- **Tasks:** Create the exact folder structure (Section 9); create `requirements.txt` with the exact pinned versions (Section 7, including `ray==2.6.3`); create `configs/config.yaml` (Section 8) with all keys populated exactly as specified.
- **Expected outputs:** Empty, importable project skeleton.
- **Validation checks:** `python -c "from src.utils.config_loader import load_config; print(load_config())"` succeeds and prints the full config dict.
- **Failure condition:** Any import error, or missing config key.
- **Recovery step:** Fix folder/file structure to exactly match Section 9 before proceeding.

### Milestone 1 — Utilities Layer
- **Depends on:** Milestone 0.
- **Inputs:** `configs/config.yaml`.
- **Tasks:** Implement `src/utils/config_loader.py`, `src/utils/logger.py`, `src/utils/seed.py` exactly per Section 8/12/13 contracts, including the fixed logger-name table in Section 13.
- **Expected outputs:** Three working utility modules.
- **Validation checks:** `tests/test_config_loader.py` passes; calling `set_global_seed(42)` twice in a row produces identical `numpy.random.rand(5)` output both times; `get_logger("preprocessing", logs_dir)` creates a timestamped file named `preprocessing_<timestamp>.log` under `logs/`.
- **Failure condition:** Any of the above checks fail.
- **Recovery step:** Fix the specific utility function; re-run validation checks before proceeding.

### Milestone 2 — Preprocessing Pipeline
- **Depends on:** Milestone 1. Requires `data/raw/edge_iiotset.csv` to be manually placed by the user before this milestone can run.
- **Inputs:** Raw Edge-IIoTset CSV, `config.yaml`.
- **Tasks:** Implement all functions in `src/preprocessing/load_dataset.py` and `src/preprocessing/encode_normalize.py` per Section 14.1–14.2 — **in the exact 11-step order specified in `run_preprocessing_pipeline`'s contract** (drop identifiers, then create labels, then split before fit); implement `experiments/run_preprocessing.py` as the orchestration entry point calling `run_preprocessing_pipeline(config)` followed by `src.evaluation.metrics.generate_class_distribution_plot(...)`.
- **Expected outputs:** `data/processed/edge_iiotset_processed.csv`, all artifacts listed in Section 14.2's write list, and `outputs/results/class_distribution.png`.
- **Validation checks:** `tests/test_preprocessing.py` passes; processed CSV has no NaN values; neither `Attack_type` nor `Attack_label` appears in the processed CSV's columns; `class_mapping.json` has one entry per unique attack category in the raw data; re-running `experiments/run_preprocessing.py` a second time (with `force_reprocess: false`) logs a skip message and does not regenerate the output file; the scaler/encoder parameters are verifiably fit on `train_indices` rows only (e.g., fitting on the full dataset would produce different `MinMaxScaler` min/max values than fitting on the training slice alone — this must not match); `outputs/results/class_distribution.png` exists and is non-zero-byte.
- **Failure condition:** Any NaN in processed output; class count mismatch; leaked native columns present in processed CSV; re-run does not skip as expected; transformer parameters match a full-dataset fit rather than a train-only fit; missing class distribution plot.
- **Recovery step:** Fix the specific preprocessing function; delete `data/processed/` and `outputs/artifacts/` contents; re-run from scratch.

### Milestone 3 — Model Definition
- **Depends on:** Milestone 2 (needs known feature count for `input_shape` and `class_mapping.json` for `num_classes`).
- **Inputs:** `outputs/artifacts/feature_names.pkl`, `outputs/artifacts/class_mapping.json`, `config.yaml`.
- **Tasks:** Implement `src/models/cnn_gru.py` per Section 14.4, with `num_classes` always derived from `class_mapping.json`, never hardcoded.
- **Expected outputs:** A working `build_cnn_gru` function.
- **Validation checks:** `tests/test_models.py` passes; `model.summary()` shows the exact layer sequence specified in Section 6/14.4, including `padding="same"` on `Conv1D` and `return_sequences=False` on `GRU`; parameter count is logged and is small enough to qualify as "lightweight" (soft guideline: under 500K parameters).
- **Failure condition:** Output shape mismatch; wrong layer order; missing padding/return_sequences settings; test failures.
- **Recovery step:** Fix architecture in `cnn_gru.py`; re-run tests.

### Milestone 4 — Centralized Baseline Training
- **Depends on:** Milestone 3.
- **Inputs:** Processed data, train/test indices, model builder, `prepare_model_ready_data`.
- **Tasks:** Implement `src/centralized/train_baseline.py` per Section 14.7 (including the `time.time()`-based training-time capture and `centralized_training_time.txt` write); implement `experiments/run_centralized.py`.
- **Expected outputs:** `outputs/models/centralized_best_model.h5`, `centralized_last_model.h5`, `centralized_history.csv`, `outputs/results/centralized_training_time.txt`, model summary, architecture diagram.
- **Validation checks:** Test-set accuracy exceeds a reasonable baseline threshold (implementing agent should log the achieved accuracy; no specific numeric acceptance threshold is imposed beyond "training must complete without error and produce a non-trivial accuracy, i.e., substantially above random-guess baseline for the number of classes present"); `centralized_history.csv` has one row per completed epoch (accounting for early stopping possibly halting before `centralized_epochs`); `centralized_training_time.txt` contains a single positive float.
- **Failure condition:** Training crashes; accuracy at or near random-guess level; missing output files.
- **Recovery step:** Check data preprocessing correctness first (most likely cause), then model architecture, then hyperparameters.

### Milestone 5 — Data Partitioning
- **Depends on:** Milestone 2.
- **Inputs:** Processed training data (`train_indices` rows only).
- **Tasks:** Implement `src/partitioning/partition_data.py` per Section 14.3.
- **Expected outputs:** A working `partition_iid` function.
- **Validation checks:** `tests/test_partitioning.py` passes.
- **Failure condition:** Overlapping shards; incorrect shard count; size mismatch.
- **Recovery step:** Fix shuffling/splitting logic; re-run tests.

### Milestone 6 — Federated Client/Server Wiring
- **Depends on:** Milestones 3, 5.
- **Inputs:** Model builder, partitioned client train shards, the shared global test split.
- **Tasks:** Implement `src/federated/client_app.py` and `src/federated/server_app.py` per Section 14.5–14.6, including the `SavingFedAvg` strategy subclass and the two distinct aggregation functions `weighted_average_fit` and `weighted_average_eval`.
- **Expected outputs:** Working `FLClient`, `client_fn`, `get_strategy`, `SavingFedAvg`, `weighted_average_fit`, `weighted_average_eval`.
- **Validation checks:** `tests/test_federated_loop.py` (2-round, 2-client smoke test on a small data subset, using an independently-constructed reduced config with `num_clients = min_available_clients = min_fit_clients = min_evaluate_clients = 2`, per Section 19) passes without error and confirms `strategy.latest_parameters` is populated after the run; confirms neither aggregation function raises a `KeyError` during the smoke test.
- **Failure condition:** Simulation crashes; weights not properly exchanged (aggregated accuracy stays at random-guess level across all rounds, indicating aggregation is not functioning); `latest_parameters` remains `None`; `KeyError` raised inside either aggregation function.
- **Recovery step:** Verify `get_parameters`/`fit`/`evaluate` correctly set and return weights; verify `FLClient.fit()`'s metrics dict always contains both `loss` and `accuracy` while `FLClient.evaluate()`'s metrics dict contains only `accuracy`; verify `get_strategy` assigns `weighted_average_fit` to `fit_metrics_aggregation_fn` and `weighted_average_eval` to `evaluate_metrics_aggregation_fn` (never the same function for both); verify the test's reduced config's `min_*_clients` matches its own `num_clients` (2, not the production value of 4); verify `aggregate_fit` is being invoked (not bypassed).

### Milestone 7 — Full Federated Training Run
- **Depends on:** Milestone 6.
- **Inputs:** Full processed dataset, all federated config values (production `config.yaml`, `num_clients = 4`).
- **Tasks:** Run `run_federated_simulation(config)` for the full 15 rounds, 4 clients via `experiments/run_federated.py`, including the global-weight capture and save procedure and the `federated_training_time.txt` write (Section 14.6).
- **Expected outputs:** `outputs/models/federated_global_model.h5`, `outputs/results/federated_history.csv` with 15 rows, `outputs/results/federated_training_time.txt`.
- **Validation checks:** Aggregated accuracy in `federated_history.csv` shows a generally increasing trend across rounds (allowing for normal round-to-round noise); final round accuracy is meaningfully above random-guess baseline; `federated_global_model.h5`, when loaded and evaluated on the shared test set, produces accuracy consistent with (not radically different from) the final round's aggregated evaluation accuracy; `federated_training_time.txt` contains a single positive float.
- **Failure condition:** Accuracy does not improve across rounds at all; fewer than 15 rows in history file; the saved global model's standalone evaluation accuracy is wildly inconsistent with the simulation's own reported final-round accuracy (a strong signal that the wrong weights were captured/saved); missing or malformed training-time file.
- **Recovery step:** Re-check Milestone 6's smoke test still passes; check local epoch count and learning rate are being applied correctly per client; re-verify the `SavingFedAvg.aggregate_fit` capture and the `parameters_to_ndarrays` → `model.set_weights` → save sequence.

### Milestone 8 — Evaluation and Comparison
- **Depends on:** Milestones 4, 7.
- **Inputs:** Both trained models (`centralized_best_model.h5`, `federated_global_model.h5`), shared test set, both `*_training_time.txt` files.
- **Tasks:** Implement `src/evaluation/metrics.py` (including `generate_class_distribution_plot` and `generate_training_curves_plot`) and `src/evaluation/compare_fl_vs_centralized.py` (including `run_evaluation_pipeline`) per Section 14.8–14.9; implement `experiments/run_evaluation.py` as a thin wrapper calling `set_global_seed(config["seed"])` then `run_evaluation_pipeline(config)`.
- **Expected outputs:** All files listed in Section 17's evaluation-related rows, plus `comparison_table.csv`, `model_size_comparison.csv`, both confusion matrix PNGs, both classification report text files, and `centralized_training_curves.png`.
- **Validation checks:** `tests/test_metrics.py` passes; `comparison_table.csv` has exactly 2 rows (Centralized, Federated) with all required columns populated (no NaN values, including a non-null `Loss` sourced from `model.evaluate()` and a non-null `Training_Time_Seconds` sourced from the persisted `*_training_time.txt` files); `model_size_comparison.csv` has exactly 2 rows with populated `Model_Size_MB` and `Parameter_Count`.
- **Failure condition:** Missing metrics; NaN values in either comparison table; missing plots or reports.
- **Recovery step:** Verify both models load correctly and produce predictions on the same test set shape; verify loss is being read from `model.evaluate()` and not left uncomputed; verify both training-time files exist before `run_evaluation_pipeline` attempts to read them (i.e., Milestones 4 and 7 both completed).

### Milestone 9 — Explainability Integration
- **Depends on:** Milestone 7 (uses the federated global model as the primary explained model) and Milestone 2 (needs processed test/train data).
- **Inputs:** Federated global model, full training tensor and full test tensor (via `prepare_model_ready_data`).
- **Tasks:** Implement `src/explainability/shap_utils.py` per Section 14.10, including the seeded background-sampling and persistence step (`shap_background.npy`) and the mandatory squeeze steps before plotting; implement `experiments/run_explainability.py`.
- **Expected outputs:** `outputs/shap_plots/shap_summary_plot.png`, `shap_bar_plot.png`, `outputs/artifacts/shap_background.npy`.
- **Validation checks:** `tests/test_shap_utils.py` passes; all three output files exist and are non-zero-byte after running; SHAP values are confirmed squeezed to `(n, num_features)` before being passed to `shap.summary_plot`; `shap_background.npy` has shape `(min(background_size, len(X_train)), num_features, 1)`.
- **Failure condition:** SHAP computation crashes or times out; empty/corrupt output files; a shape-mismatch error from `shap.summary_plot` due to a skipped squeeze step; missing or malformed background-array file.
- **Recovery step:** Reduce `explainability.sample_size` or `explainability.background_size` if computation is too slow; confirm the pinned `tensorflow==2.15.1`/`shap==0.44.1`/`ray==2.6.3` combination (Section 7) is actually installed, since this is the most likely source of a `GradientExplainer` incompatibility; verify the squeeze steps are applied before any plotting call.

### Milestone 10 — Dashboard
- **Depends on:** Milestones 2, 4, 7, 8, 9 (dashboard reads artifacts from all of them).
- **Inputs:** All `outputs/` artifacts, including `shap_background.npy`.
- **Tasks:** Implement `dashboard/app.py` per Section 18's 4-tab specification, using only the imports permitted by Section 11 (`prepare_model_ready_data` and `explain_single_prediction`, plus `src/evaluation/` plot/data loading).
- **Expected outputs:** A running Streamlit app.
- **Validation checks:** `streamlit run dashboard/app.py` launches without error; all 4 tabs render real data (assuming all prior milestones' outputs exist); manually deleting one artifact file causes only that tab to show the specified `st.error` message, without crashing the rest of the app; Tab 4's `GradientExplainer` is built from the persisted `shap_background.npy` file, never from a fresh in-dashboard sample.
- **Failure condition:** App crashes on launch; a tab renders blank instead of showing data or an error message; Tab 4 attempts to re-sample a background set instead of loading the persisted one.
- **Recovery step:** Verify all file paths referenced in `dashboard/app.py` exactly match `config["paths"]` values (constructed via `os.path.join`); verify `@st.cache_resource` usage is not causing stale-model issues after retraining.

### Milestone 11 — Full Test Suite and Documentation
- **Depends on:** All previous milestones.
- **Inputs:** Entire codebase.
- **Tasks:** Ensure every test file listed in Section 19 exists and passes; write `README.md` documenting exact run order (`experiments/run_preprocessing.py` → `run_centralized.py` → `run_federated.py` → `run_evaluation.py` → `run_explainability.py` → `streamlit run dashboard/app.py`).
- **Expected outputs:** `pytest` exits with code 0; complete `README.md`.
- **Validation checks:** `pytest tests/` passes with zero failures.
- **Failure condition:** Any test failure.
- **Recovery step:** Fix the specific failing module; do not mark the project complete until all tests pass.

---

## 22. AI Implementation Safeguards (Explicit De-Ambiguation Examples)

Wherever this document could be read multiple ways, the following literal instructions apply. Where Section 14's function contracts and Section 21's milestone tasks already state something precisely, those take precedence — this section exists to close remaining gaps.

- Do not substitute `LSTM` for `GRU` anywhere, even if it "seems similar." The specification says GRU. Implement GRU.
- Do not add PCC (Pearson Correlation) feature selection anywhere in the pipeline. This project explicitly excludes it (Section 1, Section 2).
- Do not implement Logistic Regression anywhere, even as an "extra" baseline. This project explicitly excludes it.
- Do not add Docker, Kubernetes, or any containerization. Use `flwr.simulation.start_simulation` only, at the pinned `flwr==1.8.0` version (Section 7, Section 14.6).
- Do not implement Non-IID partitioning. `partition_iid` must always be strictly IID, equal-size, seeded random shards, applied only to the training split.
- Do not add a database (SQL or NoSQL) anywhere in this project. All persistence is via flat files (`.csv`, `.pkl`, `.json`, `.h5`, `.png`, `.txt`, `.npy`) as specified.
- Do not compute or display ROC curves unless explicitly extended by the user later — they are optional and out of the default scope (Section 16).
- Do not hardcode `num_classes` anywhere (e.g., never write the literal `15`). Always derive it as `len(class_mapping)` from `class_mapping.json` (Section 6).
- Do not fit `fit_categorical_encoder`, `fit_scaler`, or `fit_label_encoder` on anything other than the `train_indices` slice of the DataFrame. Fitting on the full dataset (train+test) before splitting is a data-leakage bug, not an acceptable shortcut (Section 14.2).
- Do not remove `Attack_type` or `Attack_label` inside `drop_identifier_columns`. Both native columns are removed exclusively inside `create_labels`, which always runs immediately after `drop_identifier_columns` in the pipeline (Section 14.1, Section 14.2).
- Do not reuse a single metrics-aggregation function for both `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn` in the Flower strategy. Use `weighted_average_fit` (reads `loss` and `accuracy`) for fit-time aggregation and `weighted_average_eval` (reads only `accuracy`) for evaluate-time aggregation (Section 14.6).
- Do not upgrade `tensorflow`, `flwr`, `ray`, or `shap` beyond the exact versions pinned in Section 7, even if a newer version appears to "just work" — the pinned versions are the only combination verified compatible with this SDS's `GradientExplainer` and Flower-simulation requirements.
- Do not use Flower's `run_simulation(server_app=..., client_app=..., num_supernodes=...)` app-based API. Use `fl.simulation.start_simulation(client_fn=..., num_clients=..., config=..., strategy=...)` exclusively (Section 14.6).
- Do not save a per-client local model as `federated_global_model.h5`. The saved federated model must always be built by loading `SavingFedAvg.latest_parameters` into a freshly constructed `build_cnn_gru` model (Section 14.6).
- Do not treat `config["training"]["optimizer"]` as a live switch. Adam is hardcoded in `build_cnn_gru`'s compile call; the config key is documentation-only (Section 6, Section 14.4).
- Do not reimplement the tensor-preparation (reshape + one-hot) or feature-column-classification logic at more than one call site. Use `prepare_model_ready_data` and `infer_feature_column_types` exclusively (Section 14.2, Section 20).
- Do not have `dashboard/app.py` re-sample a SHAP background set. It must always load the persisted `outputs/artifacts/shap_background.npy` (Section 14.10, Section 18).
- Do not leave `outputs/results/centralized_training_curves.png`, the two confusion matrices, or the two classification reports without a named calling script. All are produced by `run_evaluation_pipeline` (Section 14.9), called from `experiments/run_evaluation.py`.
- Do not use the production `config.yaml`'s `federated.num_clients = 4` for the federated smoke test in `tests/test_federated_loop.py`. Construct an independent reduced-scale config copy with `num_clients = min_available_clients = min_fit_clients = min_evaluate_clients = 2` for that test only (Section 19).
- When the specification says "save," it means write to the exact file path given in Section 9's folder tree — never invent a different filename or location.
- When a function contract in Section 14 lists specific argument names and types, those exact names must be used in the implementation signature — do not rename `df` to `data`, do not rename `seed` to `random_state`, etc.
- The dashboard (Section 18) must be built with the assumption that Milestones 2–9 may not yet have been run when it is first launched — every tab must independently handle missing artifacts gracefully via `st.error`, never via an unhandled exception that crashes the whole app.

---

## 23. Project Acceptance Criteria

This project is considered **complete** only when ALL of the following are true simultaneously:

1. `pytest tests/` passes with zero failures (Section 19/21 Milestone 11).
2. `experiments/run_preprocessing.py` runs end-to-end and produces all artifacts listed in Section 14.2 plus `outputs/results/class_distribution.png`, with encoders/scaler verifiably fit on the training partition only (Milestone 2 validation checks).
3. `experiments/run_centralized.py` runs end-to-end and produces a trained model with test-set accuracy meaningfully above random-guess baseline for the number of classes present, plus `outputs/results/centralized_training_time.txt`.
4. `experiments/run_federated.py` runs end-to-end for the full 15 rounds and 4 clients, producing a global model (built from `SavingFedAvg.latest_parameters`, per Section 14.6), a 15-row history file, and `outputs/results/federated_training_time.txt`, with a generally increasing accuracy trend across rounds.
5. `experiments/run_evaluation.py` produces a complete `comparison_table.csv` and `model_size_comparison.csv` with all required columns (Section 14.9) populated for both Centralized and Federated rows, plus both confusion matrices, both classification reports, and `centralized_training_curves.png`.
6. `experiments/run_explainability.py` produces both SHAP plots and `outputs/artifacts/shap_background.npy` without error, with the required squeeze steps applied before plotting.
7. `streamlit run dashboard/app.py` launches and all 4 tabs render correctly when all prior artifacts exist, with Tab 4's explainer built from the persisted background array.
8. Every file listed in the Deliverables Checklist (Section 24) exists on disk.
9. Re-running the entire pipeline from scratch (with `force_reprocess: true` once, then `false` thereafter) with the same `config.yaml` produces byte-for-byte identical `outputs/artifacts/*.pkl` and `outputs/artifacts/shap_background.npy` files, and numerically identical evaluation metrics. Full bitwise determinism is required on the project's CPU-only target environment (Section 6, Assumption 11) given the fixed seed (Section 12).

---

## 24. Deliverables Checklist

**Configuration & Code**
- [ ] `configs/config.yaml`
- [ ] `requirements.txt` (with exact pinned versions per Section 7, including `ray==2.6.3`)
- [ ] All files listed in Section 9's folder tree under `src/`, `dashboard/`, `experiments/`, `tests/`

**Data Artifacts**
- [ ] `data/processed/edge_iiotset_processed.csv`
- [ ] `outputs/artifacts/scaler.pkl`
- [ ] `outputs/artifacts/label_encoder.pkl`
- [ ] `outputs/artifacts/categorical_encoder.pkl`
- [ ] `outputs/artifacts/train_indices.pkl`
- [ ] `outputs/artifacts/test_indices.pkl`
- [ ] `outputs/artifacts/feature_names.pkl`
- [ ] `outputs/artifacts/class_mapping.json`
- [ ] `outputs/artifacts/random_seed.txt`
- [ ] `outputs/artifacts/shap_background.npy`

**Trained Models**
- [ ] `outputs/models/centralized_best_model.h5`
- [ ] `outputs/models/centralized_last_model.h5`
- [ ] `outputs/models/centralized_model_summary.txt`
- [ ] `outputs/models/centralized_architecture.png`
- [ ] `outputs/models/federated_global_model.h5`
- [ ] `outputs/models/federated_model_summary.txt`
- [ ] `outputs/models/federated_architecture.png`

**Results & Reports**
- [ ] `outputs/results/centralized_history.csv`
- [ ] `outputs/results/centralized_training_time.txt`
- [ ] `outputs/results/federated_history.csv`
- [ ] `outputs/results/federated_training_time.txt`
- [ ] `outputs/results/comparison_table.csv`
- [ ] `outputs/results/convergence_plot.png`
- [ ] `outputs/results/centralized_confusion_matrix.png`
- [ ] `outputs/results/federated_confusion_matrix.png`
- [ ] `outputs/results/centralized_classification_report.txt`
- [ ] `outputs/results/federated_classification_report.txt`
- [ ] `outputs/results/centralized_training_curves.png`
- [ ] `outputs/results/class_distribution.png`
- [ ] `outputs/results/model_size_comparison.csv`

**Explainability**
- [ ] `outputs/shap_plots/shap_summary_plot.png`
- [ ] `outputs/shap_plots/shap_bar_plot.png`

**Dashboard**
- [ ] Working `dashboard/app.py` with all 4 tabs functioning per Section 18

**Testing**
- [ ] `tests/test_config_loader.py` (passing)
- [ ] `tests/test_preprocessing.py` (passing)
- [ ] `tests/test_partitioning.py` (passing)
- [ ] `tests/test_models.py` (passing)
- [ ] `tests/test_federated_loop.py` (passing)
- [ ] `tests/test_metrics.py` (passing)
- [ ] `tests/test_shap_utils.py` (passing)

**Documentation**
- [ ] `README.md` with exact run instructions in order
- [ ] This SDS document retained in the repository for reference (e.g., `project-docs/SDS.md`)

---

*End of Software Design Specification, Revision 2.1. This document is final, design-frozen, and implementation-ready. It is the single authoritative source for this project — no separate addendum, patch, or errata document exists or should be consulted. No further architectural decisions remain to be made.*