# Implementation Prompts — Privacy-Preserving Federated Intrusion Detection System (CNN-GRU)

**Source documents (FROZEN, authoritative):** Software Design Specification Rev 2.1, Master Implementation Blueprint v1, AI Coding Contract.
**Note:** All 18 prompts below are derived mechanically from the AI Coding Contract (Part B, per-phase) plus the Global Invariants (Part A) and Absolute Prohibitions (Part C) that govern every phase. Nothing has been redesigned, renamed, or reinterpreted. Every prompt is standalone — issue it to a coding agent in isolation, one phase at a time.

**Global invariants that apply to every phase below (Part A / Part C, not repeated in full per phase):**
- SDS Rev 2.1 architecture is frozen; no library/dataset/model/framework substitutions.
- No phase may begin before its Required Previous Phases have passed their Validation Gate.
- No phase may touch a file outside its own exhaustive allow-list.
- No literals in `src/` — every hyperparameter/path/class-count comes from `configs/config.yaml` or is derived at runtime.
- All multi-component paths use `os.path.join(...)` — never string concatenation.
- `configs/config.yaml` is edited only by explicit human instruction, never autonomously.
- The Part C "Absolute Prohibitions" table (architecture, folder structure, function names, config keys, output filenames, model architecture, seed policy, training pipeline, evaluation pipeline) may never be changed by any agent at any phase.

---

## PHASE 1 — Project Scaffolding & Configuration

**Objective:** Create the complete project directory tree and the populated `configs/config.yaml`, `requirements.txt`, and skeleton `README.md`, exactly matching SDS Section 9 (folder tree) and SDS Section 8 (config keys).

**Files allowed to CREATE:** `configs/config.yaml`, `requirements.txt`, `README.md` (skeleton), all directories and empty `__init__.py` files per SDS Section 9.

**Files allowed to MODIFY:** None (nothing pre-exists).

**Files allowed to READ:** None.

**Files FORBIDDEN (must never be touched):** N/A — this is the first phase; nothing pre-exists to forbid.

**Functions to implement:** None (scaffolding only).

**Required imports:** None (no code yet).

**Forbidden imports:** N/A.

**Exact implementation requirements:**
- Directory tree must match SDS Section 9 exactly — no extra, missing, or renamed directories.
- `configs/config.yaml` must contain every key specified in SDS Section 8, written verbatim, with no keys added or omitted.
- `requirements.txt` must pin the exact library versions required by the frozen architecture (including `flwr==1.8.0`).
- `README.md` is a skeleton only at this phase — full content is deferred to Phase 18.
- Do not create any file, function, or config key not explicitly named in SDS Section 8/9.

**Validation checklist:**
- [ ] `pip install -r requirements.txt` succeeds.
- [ ] `yaml.safe_load(configs/config.yaml)` parses without error.
- [ ] Directory tree matches SDS Section 9 exactly (structural diff = zero).
- [ ] `config.yaml` keys match SDS Section 8 exactly (no missing/extra keys).

**Unit tests:** None required at this phase.

**Integration tests:** None required at this phase.

**Expected outputs:** Fully scaffolded, empty directory tree; populated `configs/config.yaml`; `requirements.txt`; skeleton `README.md`.

**Acceptance criteria:** Directory tree and `config.yaml` are byte-for-byte structurally identical to SDS Sections 9 and 8.

**Stop conditions:** Stop immediately if any directory or config key is missing or malformed — do not proceed to Phase 2 in that state.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 2 — Utilities Layer

**Objective:** Implement the shared config-loading, logging, and seed-setting utilities that every later phase depends on.

**Files allowed to CREATE:** `src/utils/config_loader.py`, `src/utils/logger.py`, `src/utils/seed.py`, `tests/test_config_loader.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `configs/config.yaml` (at runtime).

**Files that are READ ONLY:** `configs/config.yaml`.

**Files FORBIDDEN:** Any file outside `src/utils/` and `tests/test_config_loader.py`.

**Functions to implement:** `load_config`, `get_logger`, `set_global_seed`. These three must never be duplicated — no other module may implement its own config-loading, logging-setup, or seed-setting logic anywhere in the project.

**Required imports:** `yaml`, `logging`, `datetime`, `os`, `random`, `numpy`, `tensorflow`.

**Forbidden imports:** Any `src/` submodule other than `src/utils/` itself.

**Exact implementation requirements:**
- `load_config`, `get_logger`, `set_global_seed` function signatures must match SDS Section 8/12/13 exactly.
- `get_logger` must follow the fixed logger-name table from the SDS.
- Only `seed` and `paths.logs_dir` config keys may be read at this phase.

**Validation checklist:**
- [ ] Three-case validation in `test_config_loader.py` passes.
- [ ] `set_global_seed(42)` is idempotent across repeated calls.
- [ ] `get_logger` produces filenames matching the required pattern.

**Unit tests:** `tests/test_config_loader.py` — full pass required.

**Integration tests:** None at this phase (no callers yet).

**Expected outputs:** `outputs/artifacts/random_seed.txt`, `logs/<fixed_name>_<timestamp>.log`.

**Acceptance criteria:** Function signatures match SDS Section 8/12/13 exactly; fixed logger-name table honored.

**Stop conditions:** Stop if any of the three functions fails validation — no downstream phase may import a broken utility.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 3 — Preprocessing: Load, Drop, Label

**Objective:** Implement raw dataset loading, identifier-column dropping, and label derivation from the raw Edge-IIoTset CSV.

**Files allowed to CREATE:** `src/preprocessing/load_dataset.py`, `src/preprocessing/__init__.py`, `tests/test_preprocessing.py` (load/drop/label cases only).

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `data/raw/edge_iiotset.csv` (at runtime).

**Files that are READ ONLY:** `data/raw/edge_iiotset.csv`, `src/utils/*`.

**Files FORBIDDEN:** `data/raw/edge_iiotset.csv` (write-forbidden — read-only forever), anything under `outputs/`, `data/processed/`.

**Functions to implement:** `load_raw_dataset`, `drop_identifier_columns`, `create_labels`. `create_labels` is the sole owner of native-column (`Attack_type`, `Attack_label`) removal anywhere in the codebase, for every future phase — this must never be duplicated.

**Required imports:** `pandas`, `numpy`, `src.utils.logger`.

**Forbidden imports:** `src.preprocessing.encode_normalize` (does not exist yet — no forward references), `src.models`, `src.federated`, `src.centralized`, `src.evaluation`, `src.explainability`.

**Exact implementation requirements:**
- `label` and `binary_label` are derived exactly once, inside this phase's preprocessing functions — no later phase may re-derive them.
- Only these config keys may be read: `paths.raw_data_dir`, `paths.raw_data_file`, `dataset.drop_columns`, `dataset.target_column`, `dataset.raw_binary_column`, `dataset.normal_class_value`.
- `data/raw/edge_iiotset.csv` is never written, modified, or regenerated.

**Validation checklist:**
- [ ] `FileNotFoundError` raised on a bad path.
- [ ] `Attack_type` and `Attack_label` are absent from the DataFrame after `create_labels`.
- [ ] Zero NaNs in `label` and `binary_label`.
- [ ] A binary-agreement mismatch produces a WARNING, not an exception.

**Unit tests:** Load/drop/label test cases in `tests/test_preprocessing.py`.

**Integration tests:** None at this phase.

**Expected outputs:** No persisted artifacts — in-memory labeled DataFrame only.

**Logging required:** Row/column counts after load; column list after drop; label distribution after `create_labels` — via `get_logger("preprocessing", ...)`.

**Acceptance criteria:** Matches SDS Section 14.1 function contracts exactly; native-column-removal ownership confirmed sole to `create_labels`.

**Stop conditions:** Stop if any validation check fails — Phase 4 cannot proceed without a correctly labeled DataFrame.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 4 — Preprocessing: Encode, Normalize, Split, Orchestrate

**Objective:** Implement feature-type inference, encoding, scaling, label encoding, the single train/test split, the full preprocessing orchestration pipeline, and the class-distribution plot.

**Files allowed to CREATE:** `src/preprocessing/encode_normalize.py`, `experiments/run_preprocessing.py`, remainder of `tests/test_preprocessing.py`, `src/evaluation/metrics.py` (created early — `generate_class_distribution_plot` only, the single explicit cross-boundary exception per SDS Section 14.2), `src/evaluation/__init__.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** Output of Phase 3 functions (in-memory), `configs/config.yaml`.

**Files that are READ ONLY:** `src/preprocessing/load_dataset.py` (consumed, not modified), `src/utils/*`.

**Files FORBIDDEN:** `src/preprocessing/load_dataset.py` (Phase 3's file — read-only from here on), `data/raw/edge_iiotset.csv`.

**Functions to implement:** `infer_feature_column_types`, `fit_categorical_encoder`, `fit_scaler`, `fit_label_encoder`, `split_train_test`, `prepare_model_ready_data`, `run_preprocessing_pipeline`, `generate_class_distribution_plot`. `prepare_model_ready_data` and `infer_feature_column_types` are the sole implementations for the entire remaining project — every later phase imports these, never reimplements them.

**Required imports:** `pandas`, `numpy`, `sklearn.preprocessing.{OrdinalEncoder, MinMaxScaler, LabelEncoder}`, `sklearn.model_selection.train_test_split`, `tensorflow.keras.utils.to_categorical`, `pickle`, `json`, `matplotlib` (plot function only), `src.preprocessing.load_dataset`, `src.utils.*`.

**Forbidden imports:** `src.models`, `src.federated`, `src.centralized`, `src.explainability`.

**Exact implementation requirements:**
- `infer_feature_column_types` classifies columns exclusively via rule `"dtype_object"` — no other function may independently reclassify columns.
- Any encoder/scaler/label encoder is fit only on `df.loc[train_indices]` — never on the full dataset before the split (leakage defect).
- The train/test split is computed exactly once, persisted as `train_indices.pkl` / `test_indices.pkl`, and reused everywhere downstream — no phase re-splits the data.
- The 11-step processing order must match SDS Section 14.2 exactly.
- `force_reprocess: false` must cause a re-run to skip already-processed output.
- Only these config keys may be read: `dataset.*`, `paths.processed_data_dir`, `paths.processed_data_file`, `paths.artifacts_dir`, `paths.results_dir`.

**Validation checklist:**
- [ ] 11-step order matches SDS Section 14.2 exactly.
- [ ] Leakage-regression test: train-only-fit params differ from full-dataset-fit params.
- [ ] Zero NaNs in processed data.
- [ ] Native columns (`Attack_type`, `Attack_label`) absent from processed output.
- [ ] `force_reprocess: false` re-run skips reprocessing.

**Unit tests:** Full `tests/test_preprocessing.py` pass.

**Integration tests:** SDS Milestone 2 validation checks (Section 21).

**Expected outputs:** `data/processed/edge_iiotset_processed.csv`; `outputs/artifacts/{scaler.pkl, label_encoder.pkl, categorical_encoder.pkl, train_indices.pkl, test_indices.pkl, feature_names.pkl, class_mapping.json}`; `outputs/results/class_distribution.png`.

**Logging required:** Feature count, train/test sizes, per SDS Section 13's mandatory contents — via `get_logger("preprocessing", ...)`.

**Acceptance criteria:** SDS Milestone 2 validation checks (Section 21) pass in full.

**Stop conditions:** Stop on any failure; delete any partial `data/processed/` / `outputs/artifacts/` output before retrying — never leave a half-written processed CSV in place.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 5 — CNN-GRU Model Definition

**Objective:** Implement the single, frozen CNN-GRU model-building function used by both the centralized and federated paths.

**Files allowed to CREATE:** `src/models/cnn_gru.py`, `src/models/__init__.py`, `tests/test_models.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `outputs/artifacts/feature_names.pkl`, `outputs/artifacts/class_mapping.json` (test-time only).

**Files that are READ ONLY:** `outputs/artifacts/*` (test fixtures only, never modified).

**Files FORBIDDEN:** Anything under `src/preprocessing/`, `src/centralized/`, `src/federated/`.

**Functions to implement:** `build_cnn_gru`, `get_model_summary_string`, `save_model_architecture_diagram`. `build_cnn_gru` is the sole model-construction function for both centralized and federated paths, for the entire remaining project — it must never be duplicated.

**Required imports:** `tensorflow.keras`, `io`, `src.utils.*`.

**Forbidden imports:** `src.preprocessing`, `src.centralized`, `src.federated` (per SDS Section 11, the models layer must not depend on training layers).

**Exact implementation requirements:**
- Exactly one architecture: the 7-layer sequence specified in SDS Section 14.4, exactly as written — no layer added, removed, or reordered.
- `Conv1D.padding` is always `"same"`; `GRU.return_sequences` is always `False` — both are hard-coded, never configurable.
- `num_classes` is never hardcoded — always `len(class_mapping)` read from `outputs/artifacts/class_mapping.json`.
- Adam is the only optimizer ever instantiated. `config["training"]["optimizer"]` is documentation-only and must never be read as a live branch.
- Only these config keys may be read: `model.*`, `training.learning_rate`, `training.loss`.

**Validation checklist:**
- [ ] Compiled model is returned.
- [ ] Output shape is correct for the given `num_classes`.
- [ ] `ValueError` raised on `num_classes < 2`.
- [ ] `padding="same"` and `return_sequences=False` confirmed via layer-config inspection.

**Unit tests:** `tests/test_models.py` — full pass.

**Integration tests:** SDS Milestone 3 validation checks.

**Expected outputs:** None persisted by this phase (pure function of arguments; callers in later phases persist outputs).

**Acceptance criteria:** SDS Milestone 3 validation checks pass in full.

**Stop conditions:** Stop on any validation failure — no training phase may proceed on an unverified architecture.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 6 — Centralized Baseline Training

**Objective:** Implement and run centralized CNN-GRU training with early stopping and best-model checkpointing.

**Files allowed to CREATE:** `src/centralized/train_baseline.py`, `src/centralized/__init__.py`, `experiments/run_centralized.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/{train_indices.pkl, test_indices.pkl, feature_names.pkl, label_encoder.pkl, class_mapping.json}`.

**Files that are READ ONLY:** All of the above; `src/models/cnn_gru.py`; `src/preprocessing/encode_normalize.py`.

**Files FORBIDDEN:** `src/preprocessing/*`, `src/models/*`, `outputs/artifacts/*` (consumed, never rewritten by this phase).

**Functions to implement:** `train_centralized_model`. The training-time measurement pattern (`time.time()` wrapping only `.fit()`) must never be reimplemented differently in Phase 10's federated timing.

**Required imports:** `tensorflow.keras.callbacks.{EarlyStopping, ModelCheckpoint}`, `time`, `pandas`, `pickle`, `json`, `src.models.cnn_gru`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.utils.*`.

**Forbidden imports:** `src.federated`, `src.partitioning`, `src.evaluation`, `src.explainability`.

**Exact implementation requirements:**
- Always use `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` and `ModelCheckpoint(save_best_only=True, monitor="val_loss")`, sourced from config.
- `centralized_best_model.h5` (lowest `val_loss`) is the only model that any other phase may load — `centralized_last_model.h5` is audit-only.
- `time.time()` wraps only the `.fit()` call — never surrounding I/O or plotting.
- `validation_split` is applied by Keras internally; no separate validation file is produced or read.
- Only these config keys may be read: `training.*`, `paths.models_dir`, `paths.results_dir`.

**Validation checklist:**
- [ ] Accuracy meaningfully above random-guess baseline.
- [ ] History row count ≤ `centralized_epochs`.
- [ ] `centralized_best_model.h5` confirmed as the lowest-`val_loss` checkpoint, distinct from `centralized_last_model.h5`'s save call.

**Unit tests:** None new for this phase (no dedicated `tests/` file exists per SDS Section 19; covered by `test_models.py` plus this phase's own validation checklist).

**Integration tests:** SDS Milestone 4 validation checks (Section 21).

**Expected outputs:** `outputs/models/{centralized_best_model.h5, centralized_last_model.h5, centralized_model_summary.txt, centralized_architecture.png}`; `outputs/results/{centralized_history.csv, centralized_training_time.txt}`.

**Logging required:** Per-epoch metrics, total training time, best model path — via `get_logger("centralized_training", ...)`.

**Acceptance criteria:** SDS Milestone 4 validation checks pass in full.

**Stop conditions:** Stop on any failure — do not let Phase 12 evaluate a broken or absent model. On recovery, check preprocessing correctness first, then architecture, then hyperparameters (SDS Section 21 Milestone 4 order).

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 7 — Data Partitioning (IID)

**Objective:** Implement the IID client-sharding function used to build federated learning client shards from the training split only.

**Files allowed to CREATE:** `src/partitioning/partition_data.py`, `src/partitioning/__init__.py`, `tests/test_partitioning.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** None at build time (pure function; test fixtures are synthetic).

**Files that are READ ONLY:** N/A.

**Files FORBIDDEN:** `src/centralized/*`, `outputs/models/*`, `outputs/results/*` (this phase touches no persisted artifacts at all).

**Functions to implement:** `partition_iid`. This is the sole sharding function for the project — it must never be applied to the shared test split anywhere downstream.

**Required imports:** `pandas`, `numpy`, `src.utils.*` (optional, logging only).

**Forbidden imports:** `src.preprocessing`, `src.models`, `src.centralized`, `src.federated` (per SDS Section 11, the partitioning layer imports only `src.utils`).

**Exact implementation requirements:**
- Federated client shards are drawn only from `train_indices` rows — the shared test split is never partitioned per client.
- Only these config keys may be read: `federated.num_clients`, `seed`.
- No shards persisted to disk — in-memory only.

**Validation checklist:**
- [ ] Exact shard count matches `num_clients`.
- [ ] Full row-count coverage across all shards.
- [ ] No overlapping indices between shards.
- [ ] `ValueError` raised on `num_clients < 1`.

**Unit tests:** `tests/test_partitioning.py` — full pass.

**Integration tests:** SDS Milestone 5 validation checks.

**Expected outputs:** None persisted (in-memory shards only).

**Acceptance criteria:** SDS Milestone 5 validation checks pass in full.

**Stop conditions:** Stop on any validation failure — Phase 8's `client_fn` cannot be built on unverified sharding.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 8 — Federated Client

**Objective:** Implement the Flower `NumPyClient` and its factory function for the federated learning simulation.

**Files allowed to CREATE:** `src/federated/client_app.py`, `src/federated/__init__.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** None directly (data is received via constructor/closure, supplied by Phase 9's orchestrator).

**Files that are READ ONLY:** N/A at this phase.

**Files FORBIDDEN:** `src/partitioning/*`, `src/models/*`, `src/preprocessing/*`.

**Functions to implement:** `FLClient` (class, with `get_parameters`, `fit`, `evaluate`), `client_fn`. These are the sole client implementation — no per-phase alternate client class may be introduced.

**Required imports:** `flwr.client.NumPyClient`, `tensorflow`, `src.models.cnn_gru`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.partitioning.partition_data.partition_iid`, `src.utils.seed`.

**Forbidden imports:** `src.federated.server_app` (does not exist yet — no forward references), `src.evaluation`, `src.explainability`, `src.centralized`.

**Exact implementation requirements:**
- Clients never persist models to disk (per SDS Section 14.5).
- `set_global_seed` must be called before model instantiation inside client setup.
- Only these config keys may be read: `federated.local_epochs`, `training.batch_size`, `seed`.

**Validation checklist:**
- [ ] `fit()`'s metrics dict always contains both `"loss"` and `"accuracy"`.
- [ ] `evaluate()`'s metrics dict contains only `"accuracy"`.
- [ ] `set_global_seed` confirmed called before model instantiation.

**Unit tests:** No standalone client-only test file exists per SDS Section 19 — this phase is exercised indirectly by `tests/test_federated_loop.py` in Phase 9.

**Integration tests:** Deferred to Phase 9's smoke test.

**Expected outputs:** None persisted (clients never save models).

**Acceptance criteria:** Class/function signatures match SDS Section 14.5 exactly.

**Stop conditions:** Stop on any validation failure — do not proceed to Phase 9's strategy wiring on an unverified client contract.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 9 — Federated Server, Strategy, Simulation Wiring (Smoke Test)

**Objective:** Implement the FedAvg-derived saving strategy and its aggregation functions, and prove the federated loop works end-to-end at reduced scale via a smoke test.

**Files allowed to CREATE:** `src/federated/server_app.py`, `tests/test_federated_loop.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** None (the test uses synthetic in-memory data, not the real processed CSV).

**Files that are READ ONLY:** `src/federated/client_app.py` (consumed, not modified).

**Files FORBIDDEN:** `configs/config.yaml` (the test must construct its own independent reduced config copy in-memory — never write to or mutate the production file), `src/federated/client_app.py`.

**Functions to implement:** `SavingFedAvg` (class), `get_strategy`, `weighted_average_fit`, `weighted_average_eval`. (`run_federated_simulation`'s signature is stubbed here; its full body is deferred to Phase 10.) `SavingFedAvg` is the sole strategy class — a plain `FedAvg` must never be instantiated directly for production use anywhere. `weighted_average_fit` and `weighted_average_eval` are two distinct functions and must never be collapsed into one; their assignment to `fit_metrics_aggregation_fn` vs. `evaluate_metrics_aggregation_fn` must never be swapped.

**Required imports:** `flwr` (`flwr.server.strategy.FedAvg`, `flwr.simulation.start_simulation`, `flwr.server.ServerConfig`, `flwr.common.parameters_to_ndarrays`), `time`, `src.federated.client_app`.

**Forbidden imports:** `flwr.simulation.run_simulation` / `ClientApp` / `ServerApp` (the app-based Flower API — permanently forbidden project-wide, not just this phase). The pinned API is `flwr==1.8.0`'s client-function/strategy-based `start_simulation` API only.

**Exact implementation requirements:**
- Every simulated client evaluates against the identical shared global test split — never a per-client test partition.
- The test's reduced config (`num_clients = min_available_clients = min_fit_clients = min_evaluate_clients = 2`) must be constructed independently and never derived by mutating the loaded production config in place.

**Validation checklist:**
- [ ] 2-round/2-client simulation completes without exception.
- [ ] `strategy.latest_parameters is not None` after the run.
- [ ] Neither aggregation function raises `KeyError`.

**Unit tests:** `tests/test_federated_loop.py` — this file is simultaneously the unit test and the smoke test for this phase; full pass required.

**Integration tests:** SDS Milestone 6 validation checks.

**Expected outputs:** None yet (synthetic-data smoke test only).

**Logging required:** Round-by-round aggregated metrics during the smoke test.

**Acceptance criteria:** SDS Milestone 6 validation checks pass in full.

**Stop conditions:** Stop on any failure. The expensive full 15-round/4-client run in Phase 10 must never be attempted before this smoke test is green. On recovery, verify weight exchange, verify aggregation-function-to-callback assignment is not swapped, and verify the reduced config's `min_*_clients` matches its own `num_clients` (SDS Section 21 Milestone 6).

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 10 — Full Federated Training Run

**Objective:** Extend Phase 9's stub into the full production-scale federated simulation (4 clients, 15 rounds) and persist the resulting global model.

**Files allowed to CREATE:** `experiments/run_federated.py`.

**Files allowed to MODIFY:** `src/federated/server_app.py` — add the full `run_federated_simulation` body ONLY; no other function in this file may be altered.

**Files allowed to READ:** `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/{train_indices.pkl, test_indices.pkl, class_mapping.json}`.

**Files that are READ ONLY:** `src/federated/client_app.py`, `src/partitioning/partition_data.py`, `src/models/cnn_gru.py`.

**Files FORBIDDEN:** `src/federated/client_app.py`, `src/partitioning/partition_data.py`, `tests/test_federated_loop.py` (Phase 9's smoke test must remain untouched and still passing after this phase).

**Functions to implement:** `run_federated_simulation` (full body — extends Phase 9's stub; no second simulation entry point may be introduced).

**Required imports:** `flwr`, `time` — no new imports beyond what Phase 9 already established.

**Forbidden imports:** Same as Phase 9 — the app-based Flower API is permanently forbidden.

**Exact implementation requirements:**
- `federated_global_model.h5` is always built from `SavingFedAvg.latest_parameters`, converted via `parameters_to_ndarrays`, loaded into a fresh `build_cnn_gru` instance — never a saved per-client local model.
- Production federated runs use `num_clients = min_available_clients = min_fit_clients = min_evaluate_clients = 4`.
- Only production config values may be read: `federated.num_clients` (=4), `federated.num_rounds` (=15), `federated.local_epochs` (=2) — never the test's reduced values.

**Validation checklist:**
- [ ] 15 rows present in the federated history.
- [ ] Increasing accuracy trend across rounds.
- [ ] Saved model's standalone evaluation is consistent with the simulation's own final-round reported accuracy.
- [ ] Positive float present in the training-time file.

**Unit tests:** None new (Phase 9's smoke test already covers the logic at reduced scale; this phase is scale-execution, not new design).

**Integration tests:** SDS Milestone 7 validation checks.

**Expected outputs:** `outputs/models/{federated_global_model.h5, federated_model_summary.txt, federated_architecture.png}`; `outputs/results/{federated_history.csv, federated_training_time.txt}`.

**Logging required:** Per-round aggregated metrics, total training time, saved model path — via `get_logger("federated_training", ...)`.

**Acceptance criteria:** SDS Milestone 7 validation checks pass in full.

**Stop conditions:** Stop on any failure — do not proceed to Phase 12 evaluation with an unverified or absent federated model. On recovery, re-check Phase 9's smoke test still passes, verify local epoch/learning-rate application, and re-verify the `parameters_to_ndarrays → set_weights → save` sequence (SDS Section 21 Milestone 7).

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 11 — Evaluation Metrics Module

**Objective:** Finalize the evaluation/metrics module (started early in Phase 4) by adding the remaining metrics and plotting functions.

**Files allowed to CREATE:** `tests/test_metrics.py`. (`src/evaluation/metrics.py` and `src/evaluation/__init__.py` already exist from Phase 4's early-created stub — this phase extends that file, it does not re-create it.)

**Files allowed to MODIFY:** `src/evaluation/metrics.py` (add the remaining functions beyond `generate_class_distribution_plot` only).

**Files allowed to READ:** `outputs/results/centralized_history.csv` (test-time), arbitrary caller-supplied arrays/DataFrames.

**Files that are READ ONLY:** `outputs/results/centralized_history.csv`.

**Files FORBIDDEN:** `src/centralized/*`, `src/federated/*` (per SDS Section 11, the evaluation layer does not import training layers).

**Functions to implement:** `compute_all_metrics`, `generate_confusion_matrix`, `generate_classification_report`, `generate_training_curves_plot`. (`generate_class_distribution_plot` already exists from Phase 4 — do not reimplement, only verify it still conforms.) All five metrics/plot functions in this file are the sole implementations for their named plot/report, per SDS Section 17's ownership table — no second implementation of any of these may exist anywhere, including inside `dashboard/app.py`.

**Required imports:** `sklearn.metrics.{accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report}`, `matplotlib`, `pandas`.

**Forbidden imports:** `src.centralized`, `src.federated`.

**Exact implementation requirements:**
- No config keys are read directly — these are pure functions over caller-supplied arguments.
- Test-set loss elsewhere in the project is always sourced from `model.evaluate()`, never independently recomputed from predictions (this constraint governs callers of this module in Phase 12).

**Validation checklist:**
- [ ] All 8 dict keys present in `compute_all_metrics`.
- [ ] `ValueError` raised on length mismatch between inputs.
- [ ] Perfect-prediction sanity check passes.
- [ ] `KeyError` / `FileNotFoundError` raised correctly by the two plot functions on bad input.

**Unit tests:** `tests/test_metrics.py` — full pass.

**Integration tests:** None new at this phase (exercised in Phase 12's pipeline).

**Expected outputs:** None owned by this module directly (paths supplied by callers in Phase 4 / Phase 12).

**Acceptance criteria:** `test_metrics.py` green; a grep check confirms no duplicated plot logic exists anywhere else in the repo.

**Stop conditions:** Stop on any failure — Phase 12 cannot build a correct comparison pipeline on unverified metrics functions.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 12 — Evaluation & Comparison Pipeline

**Objective:** Build the full evaluation pipeline that loads both trained models, evaluates them on the shared test set, and produces the centralized-vs-federated comparison artifacts.

**Files allowed to CREATE:** `src/evaluation/compare_fl_vs_centralized.py`, `experiments/run_evaluation.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `outputs/models/{centralized_best_model.h5, federated_global_model.h5}`, `outputs/results/{centralized_training_time.txt, federated_training_time.txt, federated_history.csv, centralized_history.csv}`, `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/{test_indices.pkl, feature_names.pkl, label_encoder.pkl, class_mapping.json}`.

**Files that are READ ONLY:** All files listed above, plus `src/evaluation/metrics.py`.

**Files FORBIDDEN:** `outputs/models/*` (loaded only, never rewritten by evaluation), `src/centralized/*`, `src/federated/*`.

**Functions to implement:** `evaluate_model_on_test_set`, `generate_comparison_table`, `generate_model_size_comparison`, `generate_convergence_plot`, `run_evaluation_pipeline`. These five are the sole comparison-pipeline implementation for the entire project.

**Required imports:** `tensorflow.keras.models.load_model`, `numpy`, `pandas`, `time`, `os`, `src.evaluation.metrics`, `src.preprocessing.encode_normalize.prepare_model_ready_data`.

**Forbidden imports:** `src.centralized.train_baseline`, `src.federated.server_app`, `src.federated.client_app` — evaluation must never re-trigger training.

**Exact implementation requirements:**
- Test-set loss is always sourced from `model.evaluate()` — never independently recomputed from predictions.
- The 12-step sequence must match SDS Section 14.9 exactly.
- `comparison_table.csv` always has exactly two rows: `Centralized` (from `centralized_best_model.h5`) and `Federated` (from `federated_global_model.h5`), with all 13 columns populated and no NaNs.
- Only these config keys may be read: `paths.models_dir`, `paths.results_dir`.

**Validation checklist:**
- [ ] 12-step sequence matches SDS Section 14.9 exactly.
- [ ] 2-row comparison table with all 13 columns populated, zero NaNs.
- [ ] 2-row model-size comparison table.

**Unit tests:** Underlying functions covered by `test_metrics.py` (Phase 11); full-pipeline validation via this phase's own checklist (no separate dedicated test file per SDS Section 19).

**Integration tests:** SDS Milestone 8 validation checks.

**Expected outputs:** `outputs/results/{comparison_table.csv, model_size_comparison.csv, convergence_plot.png, centralized_training_curves.png, centralized_confusion_matrix.png, federated_confusion_matrix.png, centralized_classification_report.txt, federated_classification_report.txt}`.

**Logging required:** All computed metrics for both models, every saved report/plot path — via `get_logger("evaluation", ...)`.

**Acceptance criteria:** SDS Milestone 8 validation checks pass in full.

**Stop conditions:** Stop on any failure — Phase 15's dashboard Tab 3 must never render against incomplete comparison output. On recovery, verify both models load/predict on identically shaped data, and verify both training-time files exist before running this pipeline (SDS Section 21 Milestone 8).

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 13 — SHAP Explainability

**Objective:** Implement SHAP-based explainability for the federated global model — background generation and single-prediction explanation.

**Files allowed to CREATE:** `src/explainability/shap_utils.py`, `src/explainability/__init__.py`, `experiments/run_explainability.py`, `tests/test_shap_utils.py`.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `outputs/models/federated_global_model.h5`, `data/processed/edge_iiotset_processed.csv`, `outputs/artifacts/{train_indices.pkl, test_indices.pkl, feature_names.pkl, label_encoder.pkl, class_mapping.json}`.

**Files that are READ ONLY:** All of the above.

**Files FORBIDDEN:** `outputs/models/federated_global_model.h5` (loaded only, never rewritten by this phase), `src/federated/*`, `src/centralized/*`.

**Functions to implement:** `generate_shap_explanations`, `explain_single_prediction`. Both must never be duplicated — `explain_single_prediction` in particular must never be reimplemented inside `dashboard/app.py` (Phase 15 imports it, does not rebuild it).

**Required imports:** `shap.GradientExplainer`, `numpy` (`numpy.random.RandomState`), `matplotlib`, `src.preprocessing.encode_normalize.prepare_model_ready_data`, `src.models.cnn_gru` (for loading only).

**Forbidden imports:** `src.centralized`, `src.federated` — the explainability layer never re-triggers training.

**Exact implementation requirements:**
- `numpy.random.RandomState(seed)` is used for both the SHAP background and test-sample draws, background first, in that fixed order, on every run.
- `shap_background.npy` is written only by `generate_shap_explanations` and read only by scripts constructing a `shap.GradientExplainer` from it — never regenerated by a reader.
- `shap_background.npy` shape must be `(min(background_size, len(X_train)), num_features, 1)`.
- Only these config keys may be read: `explainability.method`, `explainability.sample_size`, `explainability.background_size`, `seed`.

**Validation checklist:**
- [ ] All 3 output files are non-zero-byte.
- [ ] Squeeze steps confirmed applied before any plotting call.
- [ ] `shap_background.npy` shape matches `(min(background_size, len(X_train)), num_features, 1)`.
- [ ] `explain_single_prediction` works from a persisted (not freshly-sampled) background.

**Unit tests:** `tests/test_shap_utils.py` — full pass.

**Integration tests:** SDS Milestone 9 validation checks.

**Expected outputs:** `outputs/shap_plots/{shap_summary_plot.png, shap_bar_plot.png}`; `outputs/artifacts/shap_background.npy`.

**Logging required:** Background/test sample sizes, output file paths — via `get_logger("explainability", ...)`.

**Acceptance criteria:** SDS Milestone 9 validation checks pass in full.

**Stop conditions:** Stop on any failure — Phase 15's dashboard Tab 4 must never attempt to build an explainer from a missing or unverified background array. On recovery, reduce sample/background sizes if slow, confirm pinned `tensorflow`/`shap`/`ray` versions, and verify squeeze-step ordering (SDS Section 21 Milestone 9).

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 14 — Dashboard: Tabs 1 & 2

**Objective:** Build the first two tabs of the read-only Streamlit dashboard (overview / data and model exploration tabs per SDS Section 18).

**Files allowed to CREATE:** `dashboard/app.py` (Tabs 1–2 content), `dashboard/__init__.py` if required.

**Files allowed to MODIFY:** None.

**Files allowed to READ:** `outputs/results/class_distribution.png`, `outputs/artifacts/{feature_names.pkl, class_mapping.json, test_indices.pkl}`, `outputs/models/federated_global_model.h5`, `data/processed/edge_iiotset_processed.csv`.

**Files that are READ ONLY:** All of the above — the dashboard never writes to any of them.

**Files FORBIDDEN:** `src/centralized/*`, `src/federated/*`, `src/partitioning/*` (forbidden as imports, not just as writes); any file under `outputs/` or `data/` is read-only for the entire dashboard, permanently.

**Functions to implement:** Tab 1 and Tab 2 rendering logic (script-level Streamlit code; no new `src/` functions). `prepare_model_ready_data` must be imported, never reimplemented inline in the dashboard.

**Required imports:** `streamlit`, `tensorflow.keras.models.load_model`, `numpy`, `pandas`, `src.preprocessing.encode_normalize.prepare_model_ready_data`.

**Forbidden imports:** `src.centralized.*`, `src.federated.*`, `src.partitioning.*`, and specifically the functions `run_preprocessing_pipeline`, `train_centralized_model`, `run_federated_simulation`, `partition_iid`, `generate_shap_explanations` — permanently, for the life of the dashboard, not just this phase. `dashboard/app.py` never trains, never partitions, never runs preprocessing, and never resamples a SHAP background set — it is strictly read-only against `outputs/` and `data/processed/`.

**Exact implementation requirements:**
- Only these config keys may be read: `dashboard.title`, `paths.*`.
- Every tab independently guards against missing artifacts via `st.error(...)` and continues rendering the other tabs — no tab's failure may crash the app shell.

**Validation checklist:**
- [ ] Both tabs render with real data.
- [ ] Each tab independently shows `st.error(...)` on its own missing artifact without crashing the other tab or the app shell.
- [ ] Forbidden-import grep check passes.

**Unit tests:** No dedicated pytest file (SDS Section 19 does not list one); manual/visual validation per this phase's checklist is the required testing method.

**Integration tests:** Matches SDS Section 18's Tab 1/Tab 2 bullet points exactly.

**Expected outputs:** None (dashboard is display-only).

**Acceptance criteria:** Matches SDS Section 18's Tab 1/Tab 2 bullet points exactly.

**Stop conditions:** Stop on any failure — Phase 15 must not extend a dashboard shell whose Tabs 1–2 don't already independently fail-safe.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 15 — Dashboard: Tabs 3 & 4

**Objective:** Extend the dashboard with the comparison-results tab and the SHAP explainability tab.

**Files allowed to CREATE:** None new.

**Files allowed to MODIFY:** `dashboard/app.py` — add Tabs 3–4 only; Tabs 1–2 code from Phase 14 must not be altered except for the minimal `st.session_state` wiring needed to share the selected sample index.

**Files allowed to READ:** `outputs/results/{comparison_table.csv, model_size_comparison.csv, convergence_plot.png, centralized_confusion_matrix.png, federated_confusion_matrix.png}`, `outputs/artifacts/shap_background.npy`, `outputs/shap_plots/shap_summary_plot.png`.

**Files that are READ ONLY:** All of the above.

**Files FORBIDDEN:** Same forbidden-import list as Phase 14, permanently; `outputs/artifacts/shap_background.npy` (read-only — never regenerated by the dashboard).

**Functions to implement:** Tab 3 and Tab 4 rendering logic (script-level). `explain_single_prediction` must be imported from `src/explainability/shap_utils.py`, never reimplemented; the four metrics/plot-loading calls reuse Phase 11/12's saved files, never recompute them.

**Required imports:** `streamlit`, `pandas`, `numpy`, `shap.GradientExplainer`, `src.explainability.shap_utils.explain_single_prediction`.

**Forbidden imports:** Same permanent list as Phase 14, plus `src.explainability.shap_utils.generate_shap_explanations` specifically — the dashboard may import `explain_single_prediction` only, never the background-generating function.

**Exact implementation requirements:**
- Only `paths.*` config keys may be read.
- Tab 4's `GradientExplainer` must be confirmed built from the persisted `shap_background.npy`, never freshly resampled.
- Each of the 4 tabs must independently guard against missing artifacts.
- `st.session_state` correctly shares the selected sample between Tab 2 and Tab 4.

**Validation checklist:**
- [ ] All 4 tabs render correctly together.
- [ ] Tab 4's `GradientExplainer` confirmed built from the persisted background file.
- [ ] Each of the 4 tabs independently artifact-guarded.
- [ ] `st.session_state` correctly shares the selected sample between Tab 2 and Tab 4.

**Unit tests:** No dedicated pytest file; manual/visual validation via the full 4-tab artifact-removal test matrix.

**Integration tests:** SDS Milestone 10 validation checks.

**Expected outputs:** None (dashboard is display-only).

**Acceptance criteria:** SDS Milestone 10 validation checks pass in full; matches every bullet point in SDS Section 18.

**Stop conditions:** Stop on any failure — do not consider Phase 16/17 testing complete while the dashboard can crash on a missing artifact. On recovery, verify all paths via `os.path.join(config["paths"][...], ...)` and verify `@st.cache_resource` isn't serving a stale model reference.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 16 — Core Test Suite Completion (Config, Preprocessing, Partitioning, Models)

**Objective:** Fill any remaining test-case gaps in the four foundational test files against the SDS Section 19 test-case table.

**Files allowed to CREATE:** None new.

**Files allowed to MODIFY:** `tests/test_config_loader.py`, `tests/test_preprocessing.py`, `tests/test_partitioning.py`, `tests/test_models.py` — gap-fill only, no rewrites of already-passing cases.

**Files allowed to READ:** All `src/utils/`, `src/preprocessing/`, `src/partitioning/`, `src/models/` modules.

**Files that are READ ONLY:** All `src/` modules referenced above — this phase does not modify production code, only test code (unless a genuine defect is found, in which case the fix routes back to the owning phase's file, not patched ad hoc here).

**Files FORBIDDEN:** Any file outside the four named test files; no production `src/` file may be modified from this phase.

**Functions to implement:** None new in `src/`.

**Required imports:** `pytest`, plus whatever the four test files already import.

**Forbidden imports:** N/A beyond the existing per-module rules.

**Exact implementation requirements:**
- Every test case named in SDS Section 19's table for these four files must be present and passing — cross-checked line by line, not assumed from "pytest passes."
- Any genuine `src/` defect discovered must be routed back to its owning phase (2, 3, 4, 5, or 7) for the fix — never patched directly in this phase.

**Validation checklist:**
- [ ] Every SDS Section 19 test case for `test_config_loader.py` present and passing.
- [ ] Every SDS Section 19 test case for `test_preprocessing.py` present and passing.
- [ ] Every SDS Section 19 test case for `test_partitioning.py` present and passing.
- [ ] Every SDS Section 19 test case for `test_models.py` present and passing.

**Unit tests:** `pytest tests/test_config_loader.py tests/test_preprocessing.py tests/test_partitioning.py tests/test_models.py -v` — full pass required.

**Integration tests:** None new (test-only phase).

**Expected outputs:** None.

**Acceptance criteria:** 100% of SDS Section 19's listed cases for these four files present and green.

**Stop conditions:** Stop on any failure; add the missing case and fix the underlying function via its owning phase — never patch the production function inside this phase.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 17 — Remaining Test Suite Completion & Full Regression Pass

**Objective:** Fill any remaining test-case gaps in the three remaining test files and run the complete 7-file test suite together for the first time.

**Files allowed to CREATE:** None new.

**Files allowed to MODIFY:** `tests/test_federated_loop.py`, `tests/test_metrics.py`, `tests/test_shap_utils.py` — gap-fill only.

**Files allowed to READ:** All `src/` modules.

**Files that are READ ONLY:** All `src/` modules; any discovered defect routes back to its owning phase, not patched here.

**Files FORBIDDEN:** Any file outside the three named test files, plus the four files already finalized in Phase 16.

**Functions to implement:** None new in `src/`.

**Required imports:** `pytest`, plus whatever the three test files already import.

**Forbidden imports:** N/A beyond existing per-module rules.

**Exact implementation requirements:**
- The federated test must continue to construct its own independent reduced config, per Global Invariant A.5.6 — no config key changes are permitted here.
- The full case list from SDS Section 19 for all 3 files must be implemented.
- `pytest tests/` (all 7 files together) must exit 0 with zero failures — the 7 test files named in SDS Section 19 are the complete test suite; no 8th top-level test file may be added.

**Validation checklist:**
- [ ] Full SDS Section 19 case list present for `test_federated_loop.py`, `test_metrics.py`, `test_shap_utils.py`.
- [ ] `pytest tests/` (all 7 files together) exits 0 with zero failures.

**Unit tests:** Each of the three files individually, plus the full suite.

**Integration tests:** `pytest tests/` full-suite pass — this is the first point in the project the entire suite runs as one command.

**Expected outputs:** None.

**Acceptance criteria:** `pytest tests/` exits 0; matches SDS Section 21 Milestone 11 and Section 23 acceptance criterion 1.

**Stop conditions:** Stop on any failure; isolate the failing file, fix via the owning phase, then re-run the *full* suite (not just the failing file). Use `pytest`'s `tmp_path` fixture to avoid cross-file fixture collisions.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.

---

## PHASE 18 — Documentation & Final Project Validation

**Objective:** Produce the final `README.md` and perform a full, direct-inspection audit of the entire project against the SDS's acceptance criteria and deliverables checklist.

**Files allowed to CREATE:** `README.md` (final version).

**Files allowed to MODIFY:** None in `src/`, `experiments/`, `dashboard/`, `tests/` — any defect discovered during this final audit routes back to its owning phase (1–17) for a proper fix, never patched ad hoc inside this documentation phase.

**Files allowed to READ:** Entire repository; full `outputs/` tree.

**Files that are READ ONLY:** Everything.

**Files FORBIDDEN:** All production code — this is a documentation-and-audit-only phase; nothing in `src/`, `experiments/`, `dashboard/`, or `tests/` may be modified.

**Functions to implement:** None.

**Required imports:** N/A (documentation).

**Forbidden imports:** N/A.

**Exact implementation requirements:**
- All 9 SDS Section 23 acceptance criteria (criterion 9 as patched by Global Invariant A.10.4's determinism clarification) must be re-verified directly, not assumed.
- Every file in SDS Section 24's Deliverables Checklist must be confirmed present on disk by direct inspection, not by reading source code and assuming.
- `README.md` must describe the actually-implemented system, verified against the real file tree — not copied from SDS prose.

**Validation checklist:**
- [ ] All 9 SDS Section 23 acceptance criteria (as patched) re-verified directly.
- [ ] Every item in SDS Section 24's Deliverables Checklist confirmed present on disk.
- [ ] `README.md` verified against the real file tree.

**Unit tests:** N/A (documentation phase — relies on the full suite already having passed in Phase 17).

**Integration tests:** Full run-order sequence (Phases 1–17's pipelines) executed end-to-end at least once as the final reproducibility check.

**Expected outputs:** `README.md` (final version).

**Acceptance criteria:** All SDS Section 23 (as patched) and Section 24 items satisfied; `README.md` describes the actually-implemented system, verified against the real file tree.

**Stop conditions:** Stop and route any unchecked item back to its owning phase — this phase does not close out the project until every item is checked. This is the terminal phase of the project.

---

STOP AFTER THIS PHASE.
Do not implement any later phase.
Do not modify files owned by previous phases.
Wait for the next prompt.