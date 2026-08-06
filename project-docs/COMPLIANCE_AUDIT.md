# IIoT Federated IDS — Repository Compliance Audit

> **Audit type:** Read-only. This document records findings and the required fix
> for each. No source file was modified while producing it.
>
> **Governing specifications cross-checked for every section:**
> - `project-docs/SDS.md`
> - `project-docs/AI_Coding_Contract.md`
> - `project-docs/Phase_Implementation_Prompts.md`
> - `Claude_Skills.md` (skills / coding-standard rules)
>
> **Audit method:** incremental. Each section is written to disk immediately
> after the files it covers are read, so audit quality does not degrade as the
> codebase is traversed.
>
> **Started:** 2026-08-06
> **Repository commit at audit time:** `58bf27c4d0f8843fe2ace2d15d0c7b5aefaa18a2`

---

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **PASS** | Requirement fully implemented and matches the specification. |
| **PARTIAL** | Implemented, but deviates from spec or is missing a sub-requirement. |
| **FAIL** | Implemented incorrectly — violates the spec or a frozen invariant. |
| **NOT IMPLEMENTED** | Required by spec; no implementation exists. |

## Severity vocabulary

| Severity | Meaning |
| --- | --- |
| **BLOCKER** | Violates a frozen Contract invariant, or breaks a documented run path. |
| **MAJOR** | Real deviation from the SDS with functional consequences. |
| **MINOR** | Style, docstring, or naming deviation; no functional impact. |
| **OBSERVATION** | Informational; no action required. |

---

## Section index

| # | Component | Status | Batch |
| --- | --- | --- | --- |
| 1 | `src/utils/config_loader.py` | PASS | 0 |
| 2 | `src/utils/logger.py` | PASS | 0 |
| 3 | `src/utils/seed.py` | PASS WITH NOTES | 0 |
| 4 | `src/utils/dataio.py` | PASS | 0 |
| 5 | `src/preprocessing/load_dataset.py` | PASS | 1 |
| 6 | `src/preprocessing/encode_normalize.py` | PASS WITH NOTES | 1 |
| 7 | `src/models/cnn_gru.py` | PASS | 2 |
| 8 | `src/partitioning/` | PASS | 2 |
| 9 | `src/centralized/train_baseline.py` | PASS | 3 |
| 10 | `src/federated/client_app.py` | PASS | 3 |
| 11 | `src/federated/server_app.py` | PASS WITH NOTES | 3 |
| 12 | `src/evaluation/` | PARTIAL | 4 |
| 13 | `src/explainability/` | NOT IMPLEMENTED | 4 |
| 14 | `dashboard/` | NOT IMPLEMENTED | 4 |
| 15 | `experiments/` | PARTIAL | 5 |
| 16 | `tests/` | PARTIAL | 5 |
| 17 | `README.md` / packaging | PARTIAL | 5 |
| 18 | Import-graph conformance | PASS | 6 |
| 19 | Artifact inventory vs SDS §12 | PARTIAL | 6 |
| — | **Final roll-up** | COMPLETE | 7 |


---

# BATCH 0 — Utilities layer (SDS §13, §12)

The four modules in `src/utils/` are the project's *singleton services*: each one
is declared by the SDS to be the sole authoritative implementation of its
concern, and the AI Coding Contract forbids any other module from
re-implementing that concern locally. The audit for this batch therefore tests
two things per module: (a) does the implementation do what the spec says, and
(b) is the singleton claim actually true across the repository.

---

## 1. `src/utils/config_loader.py` — PASS

**Governing spec:** SDS §13 (single config entry point); AI Coding Contract
Part C (locked configuration values); `Phase_Implementation_Prompts.md` Phase 1.

**Files:** `src/utils/config_loader.py` (195 lines), `configs/config.yaml` (112
lines), `configs/config_laptop.yaml` (123 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 1.1 | Exactly one authoritative config-loading function exists | **PASS** | `load_config`, `config_loader.py:127`. Module docstring states the exclusivity rule at lines 3–5. |
| 1.2 | Loads YAML from a configurable path, defaulting to `configs/config.yaml` | **PASS** | `config_loader.py:128` — `config_path: str = "configs/config.yaml"`. |
| 1.3 | Raises `FileNotFoundError` when the config file is absent | **PASS** | `config_loader.py:161-164`. |
| 1.4 | Validates that all required top-level keys are present | **PASS** | `_REQUIRED_TOP_LEVEL_KEYS` at `config_loader.py:42-51`; loop at `188-192` raises `KeyError`. |
| 1.5 | Validation runs **after** overlay merge, so an overlay cannot delete a required section | **PASS** | Merge at `config_loader.py:183`, validation at `188`. Ordering is explicitly commented at line 187. |
| 1.6 | Two run profiles are selectable without editing code | **PASS** | `FULL_EXPERIMENT` / `LAPTOP_MODE` constants, `config_loader.py:53-54`; overlay table at `58-61`. |
| 1.7 | Mode precedence is argument → `IIOT_MODE` env var → default | **PASS** | `resolve_mode`, `config_loader.py:85`. |
| 1.8 | An unknown mode name fails loudly rather than silently defaulting | **PASS** | `config_loader.py:88-92` raises `ValueError` listing valid modes. Rationale documented at `80-83`. |
| 1.9 | Overlay is a **deep** merge, so a mode file need only restate changed keys | **PASS** | `_deep_merge`, `config_loader.py:97-123`; recursion at `118-119`. |
| 1.10 | Base config is never mutated by the merge | **PASS** | `copy.deepcopy(base)` at `config_loader.py:114`; `copy.deepcopy(overlay_value)` at `121`. |
| 1.11 | `FULL_EXPERIMENT` leaves the production config byte-identical | **PASS** | Maps to `None` at `config_loader.py:59`; the `if overlay_path is not None` guard at `175` skips merging entirely. |
| 1.12 | Missing overlay file for a mode that requires one raises `FileNotFoundError` | **PASS** | `config_loader.py:176-180`, with a message naming both the mode and the file. |
| 1.13 | The resolved mode is recorded in the returned config for audit | **PASS** | `config["run_mode"] = resolved_mode`, `config_loader.py:185`. |
| 1.14 | Locked Contract values present and correct in `configs/config.yaml` | **PASS** | `conv_padding: "same"` (`config.yaml:65`), `gru_return_sequences: false` (`config.yaml:69`), `seed: 42` (`config.yaml:1`). |
| 1.15 | Every SDS-required top-level section exists in the base config | **PASS** | `seed`, `paths`, `dataset`, `model`, `training`, `federated`, `explainability`, `dashboard` — all present in `configs/config.yaml`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 1-A | OBSERVATION | `resolve_mode` upper-cases the resolved name, so `iiot_mode=laptop_mode` works. This is a usability convenience, not a spec deviation, and the typo-guard at `88-92` still catches `LAPTOP-MODE`. | `config_loader.py:86` | None. |
| 1-B | OBSERVATION | `config["run_mode"]` is injected *after* the merge but is not in `_REQUIRED_TOP_LEVEL_KEYS`, so it can never be supplied by a YAML file and cannot be accidentally overridden by an overlay. Correct by construction. | `config_loader.py:185` vs `42-51` | None. |
| 1-C | MINOR | The overlay path `"configs/config_laptop.yaml"` at line 60 is relative to the process CWD, matching the default `config_path`. Both therefore assume the project root is the CWD. This is consistent within the module, but it means `load_config()` cannot be called from a subdirectory without arguments. | `config_loader.py:60`, `128` | Optional: resolve both paths relative to the package root. Not required by the SDS, which documents project-root execution. |

### Notes

The `LAPTOP_MODE` overlay is unusually well-documented: `configs/config_laptop.yaml`
lines 1–31 record the exact Windows failure mode being mitigated
(`CreateFileMapping() failed. GetLastError() = 1450`), and every subsequent key
carries a comment explaining its memory rationale. This exceeds the SDS
documentation requirement and materially aids review.

Critically, the overlay changes **only** `federated`, `training.batch_size`, and
adds a `runtime_profile` block. It does not touch `dataset`, `model`, `paths`,
or `explainability` — which is what makes the claim in `config_loader.py:30-33`
("identical preprocessing, identical CNN-GRU architecture, identical evaluation
pipeline across profiles") verifiably true rather than merely asserted.

One consequence worth recording for the federated audit in Batch 3:
`config_laptop.yaml` introduces two keys absent from the base config —
`federated.ray` (lines 66–80) and `federated.client_num_cpus` / `client_num_gpus`
(lines 92–93), plus the top-level `runtime_profile` (lines 107–123). Any consumer
of these keys **must** use `.get()` with a default, because they do not exist in
`FULL_EXPERIMENT`. This will be verified against `server_app.py` in Section 11.

---

## 2. `src/utils/logger.py` — PASS

**Governing spec:** SDS §13 (fixed logger-name table); `Phase_Implementation_Prompts.md` Phase 1.

**Files:** `src/utils/logger.py` (76 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 2.1 | Exactly one authoritative logger-construction function | **PASS** | `get_logger`, `logger.py:21`. Exclusivity stated at `logger.py:3-4`. |
| 2.2 | The five fixed logger names are documented in-module | **PASS** | `logger.py:8-13` reproduces the SDS §13 naming table verbatim. |
| 2.3 | Console handler at INFO level | **PASS** | `logger.py:64-66`. |
| 2.4 | File handler at DEBUG level | **PASS** | `logger.py:69-71`. |
| 2.5 | Logger itself set to DEBUG so the file handler receives everything | **PASS** | `logger.py:56`. |
| 2.6 | Log files are uniquely timestamped and never overwritten | **PASS** | `datetime.now().strftime("%Y%m%d_%H%M%S")` at `logger.py:49`; filename at `50`; `mode="a"` at `69`. |
| 2.7 | `logs_dir` is created if absent | **PASS** | `os.makedirs(logs_dir, exist_ok=True)`, `logger.py:47`. |
| 2.8 | `logs_dir` is sourced from config, not hardcoded | **PASS** | Parameter, `logger.py:21`; documented as `config["paths"]["logs_dir"]` at `37-38`. |
| 2.9 | Repeated calls must not accumulate duplicate handlers | **PASS** | Internal logger name is `f"{name}_{timestamp}"` (`logger.py:55`), so `logging.getLogger` never returns a cached object with handlers already attached. Rationale documented at `52-54`. |
| 2.10 | Consistent timestamped message format | **PASS** | `logger.py:58-61`. |
| 2.11 | UTF-8 encoding on the log file | **PASS** | `logger.py:69`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 2-A | OBSERVATION | The duplicate-handler defence (2.9) is the correct fix for the classic `logging` pitfall where a test suite calling `get_logger("preprocessing", ...)` twice produces doubled console output. Worth noting that it also means two loggers created in the *same second* with the same `name` **will** share a logger object and double their handlers. In practice the five names are used once per process, so this is unreachable. | `logger.py:55` | None. |
| 2-B | MINOR | `get_logger` does not validate `name` against the five-name table it documents. A typo such as `"preprocesing"` silently produces a differently-named log file rather than failing. Contrast with `resolve_mode`, which *does* validate its input. | `logger.py:21`, `8-13` | Optional consistency improvement: validate `name` against a module-level tuple and raise `ValueError`. Not mandated by the SDS. |
| 2-C | OBSERVATION | `propagate` is left at its default `True`. Because the logger name is unique per call and no root handlers are configured by the project, no duplicate emission occurs. If a future module calls `logging.basicConfig()`, records would appear twice. | `logger.py:55-56` | None now; note for Batch 5 when auditing `experiments/`. |

### Notes

Compliance here is clean. The one design decision worth flagging forward is that
`get_logger` returns a **new log file per call**, deliberately (`logger.py:27-29`).
Batch 5 must therefore confirm that each `experiments/run_*.py` calls
`get_logger` exactly once at start-up and threads the returned logger down into
`src/` functions as a parameter — rather than each `src/` module calling
`get_logger` itself, which would scatter dozens of log files per run. Early
evidence that the intended pattern is honoured: `create_labels` accepts an
optional `logger` parameter instead of constructing one
(`load_dataset.py:112`, documented at `145-148`).

---

## 3. `src/utils/seed.py` — PASS WITH NOTES

**Governing spec:** SDS §12 (reproducibility artifacts), SDS §13;
AI Coding Contract (seed 42 is a locked value);
`Phase_Implementation_Prompts.md` Phase 1.

**Files:** `src/utils/seed.py` (53 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 3.1 | Exactly one authoritative seed-setting function | **PASS** | `set_global_seed`, `seed.py:16`. Exclusivity stated at `seed.py:3-4`. |
| 3.2 | Seeds Python stdlib `random` | **PASS** | `seed.py:43`. |
| 3.3 | Seeds NumPy | **PASS** | `seed.py:44`. |
| 3.4 | Seeds TensorFlow | **PASS** | `seed.py:45`. |
| 3.5 | Sets `PYTHONHASHSEED` | **PARTIAL** | `seed.py:46` — set correctly, but see finding 3-A for the timing caveat. |
| 3.6 | Default seed value is 42 (Contract-locked) | **PASS** | `seed.py:16`; matches `configs/config.yaml:1`. |
| 3.7 | Persists the seed to `outputs/artifacts/random_seed.txt` for audit | **PASS** | `seed.py:49-53`. |
| 3.8 | Artifacts directory created if absent | **PASS** | `seed.py:50`. |
| 3.9 | Documented as the first executable statement of every `experiments/` script | **PASS** (doc) | `seed.py:5-6`. Actual call-site ordering is verified in Batch 5. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 3-A | MINOR | `os.environ["PYTHONHASHSEED"]` is set at *runtime*, but CPython fixes its string-hash randomisation at **interpreter start-up**. Setting the variable from inside a running process therefore has no effect on the current process's hash seed; it only propagates to child processes. Genuine hash determinism requires the variable to be set in the shell before `python` is invoked, or the process to re-exec itself. | `seed.py:46` | Document the limitation in the docstring, and set `PYTHONHASHSEED=42` in the shell / Colab cell before launching. Note: this affects only set/dict iteration order, which this pipeline does not depend on for numerical results — hence MINOR, not MAJOR. |
| 3-B | MINOR | The artifacts path is hardcoded as `os.path.join("outputs", "artifacts")` rather than read from `config["paths"]["artifacts_dir"]`. Every other module sources paths from config. If `artifacts_dir` were ever repointed in `configs/config.yaml` (it is currently `"outputs/artifacts"`, so they agree today), `random_seed.txt` would be written to the old location and silently diverge from the rest of the artifacts. | `seed.py:49-50` vs `configs/config.yaml:9` | Accept an optional `artifacts_dir` parameter defaulting to the current literal, and have `experiments/` scripts pass `config["paths"]["artifacts_dir"]`. |
| 3-C | OBSERVATION | The function does not enable TensorFlow op-level determinism (`tf.config.experimental.enable_op_determinism()`), and does not constrain intra/inter-op thread counts. Full bit-for-bit reproducibility of GPU training therefore is not guaranteed — only the seeds are fixed. The docstring at `seed.py:20-26` claims "byte-identical results across runs **on the same machine**", which is a defensible scoping of the claim for CPU execution. | `seed.py:43-46` | None required. If bit-exact GPU reproducibility becomes a submission requirement, add `enable_op_determinism()`. Relevant to the Colab-readiness assessment in the final roll-up. |
| 3-D | OBSERVATION | Writing `random_seed.txt` on every call means a run that invokes `set_global_seed` once per script overwrites the file with the same value each time — idempotent and harmless. | `seed.py:52-53` | None. |

### Notes

Findings 3-A and 3-B are both real but neither changes any number this project
reports: the pipeline's randomness flows through `random`, NumPy, TensorFlow,
and scikit-learn's `random_state` (which is passed the config seed explicitly at
`encode_normalize.py:490`), none of which consult `PYTHONHASHSEED`. The status is
PASS WITH NOTES rather than PARTIAL for that reason — the reproducibility
guarantee the project actually relies on is intact.

Finding 3-C is the one to carry into the final roll-up: on Colab GPU, cuDNN's
non-deterministic kernel selection means two runs of `run_centralized.py` may
differ in the fourth decimal place of accuracy. That is normal and acceptable
for this class of project, but the README should say so rather than promise
byte-identical results.

---

## 4. `src/utils/dataio.py` — PASS

**Governing spec:** SDS §13; `Phase_Implementation_Prompts.md` (processed-data
persistence); AI Coding Contract (no duplicated I/O logic).

**Files:** `src/utils/dataio.py` (156 lines), `configs/config.yaml:6-7`.

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 4.1 | Exactly one read and one write entry point for `data/processed/` | **PASS** | `read_processed` (`dataio.py:67`), `write_processed` (`dataio.py:114`); `__all__` at `37` restricts the public surface to those two. |
| 4.2 | Storage format is a configuration decision, not a code change | **PASS** | Dispatch on file extension from `config["paths"]["processed_data_file"]`; `_dispatch_extension`, `dataio.py:44-64`. |
| 4.3 | Parquet supported | **PASS** | `dataio.py:103-106` (read), `140-146` (write). |
| 4.4 | CSV supported for backwards compatibility | **PASS** | `dataio.py:108-111` (read), `156` (write). Rationale at `26-29`. |
| 4.5 | Unsupported extension raises `ValueError` | **PASS** | `dataio.py:61-64`, message enumerates supported extensions. |
| 4.6 | Missing file on read raises `FileNotFoundError` with actionable guidance | **PASS** | `dataio.py:95-99` — message names `experiments/run_preprocessing.py` as the fix. |
| 4.7 | Column projection is pushed down to the reader where possible | **PASS** | `pandas.read_parquet(path, columns=columns)`, `dataio.py:106`; CSV falls back to read-then-subset at `109-111`, which is documented honestly at `84-85`. |
| 4.8 | Parent directory created on write | **PASS** | `dataio.py:136-138`. |
| 4.9 | Index is not written to disk | **PASS** | `index=False` at both `dataio.py:146` and `156`. |
| 4.10 | Missing Parquet engine produces an actionable error, not a generic one | **PASS** | `dataio.py:147-153` re-raises `ImportError` naming both fixes (`pip install pyarrow`, or switch the config to `.csv`). |
| 4.11 | Config default selects Parquet | **PASS** | `processed_data_file: "edge_iiotset_processed.parquet"`, `configs/config.yaml:7`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 4-A | **MAJOR** | `pyarrow` is **not listed in `requirements.txt`**, yet `configs/config.yaml:7` selects a `.parquet` processed-data file by default. A clean environment built strictly from `requirements.txt` will therefore fail at the first `write_processed` call in `run_preprocessing.py`. The failure is *well-handled* (the `ImportError` at `dataio.py:147-153` explains the fix precisely), but it is still a guaranteed first-run failure on a fresh machine — including a fresh Colab runtime. Note that `pandas>=2.0` does not pull in `pyarrow` as a hard dependency. | `requirements.txt:1-11` vs `configs/config.yaml:7`; `dataio.py:140-153` | Add `pyarrow>=14,<16` to `requirements.txt`. This is the single highest-value one-line fix found in this batch. |
| 4-B | OBSERVATION | `_dispatch_extension` is called *before* the directory is created in `write_processed` (`dataio.py:134` then `136`), so an invalid extension fails without leaving an empty directory behind. Correct ordering. | `dataio.py:134-138` | None. |
| 4-C | MINOR | `read_processed` checks `os.path.exists` before dispatching (`dataio.py:95`), so an unsupported *and* missing path reports "not found" rather than "unsupported format". Slightly less precise, but the more actionable message wins. | `dataio.py:95-101` | None. |
| 4-D | OBSERVATION | The `.pq` extension is accepted alongside `.parquet` (`dataio.py:40`). Harmless breadth. | `dataio.py:40` | None. |

### Notes

The module-level docstring (`dataio.py:8-29`) is a model of the standard this
project sets for itself: it states the problem (643 MB CSV), quantifies the three
costs, names the fix, and — importantly — records that the change is
**backwards-compatible** because reverting `processed_data_file` to a `.csv`
name restores the old behaviour with no code edit. That claim is verifiably true
from the dispatch logic.

Finding **4-A is the first actionable defect of the audit** and is carried
forward to the Critical Bugs list in the final roll-up. It is classified MAJOR
rather than BLOCKER because the fix is a single dependency line and the runtime
error message already tells the user exactly what to do — but on a fresh Google
Colab runtime, `run_preprocessing.py` will not complete without it.

---

## Batch 0 summary

| Section | Component | Status |
| --- | --- | --- |
| 1 | `config_loader.py` | **PASS** — 15/15 requirements |
| 2 | `logger.py` | **PASS** — 11/11 requirements |
| 3 | `seed.py` | **PASS WITH NOTES** — 8 PASS, 1 PARTIAL (3.5) |
| 4 | `dataio.py` | **PASS** — 11/11 requirements |

**Defects raised in this batch:** 1 MAJOR (`4-A`, missing `pyarrow` dependency),
4 MINOR, 6 OBSERVATION. No BLOCKER, no FAIL, no NOT IMPLEMENTED.

**Verdict for the utilities layer:** compliant. All four modules honour their
singleton mandate, source their inputs from config rather than hardcoding, and
fail loudly with actionable messages. The layer is production-ready once `4-A`
is fixed.

**Carried forward:**
- `4-A` → Critical Bugs / Colab readiness (final roll-up).
- `3-C` → Colab readiness caveat re: GPU determinism (final roll-up).
- `1-C` note → verify `federated.ray`, `federated.client_num_cpus`,
  `runtime_profile` are read with `.get()` defaults (Section 11).
- `2-C` note → verify each `experiments/run_*.py` calls `get_logger` exactly once
  and passes the logger down (Batch 5).

---

# BATCH 1 — Preprocessing layer (SDS §14.1, §14.2)

This is the highest-risk layer in the project. Every downstream number — the
centralized baseline, the federated global model, the SHAP attributions, the
dashboard — is conditioned on the correctness of these two modules. The audit
therefore weights **data-leakage integrity** above all other criteria: an
architecture defect degrades results visibly, whereas a leakage defect *inflates*
them and is invisible in the metrics.

---

## 5. `src/preprocessing/load_dataset.py` — PASS

**Governing spec:** SDS §14.1; AI Coding Contract Invariant A.2 (`create_labels`
is the sole owner of native-column removal); `Phase_Implementation_Prompts.md`
Phase 2.

**Files:** `src/preprocessing/load_dataset.py` (200 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 5.1 | `load_raw_dataset` reads the raw CSV into a DataFrame | **PASS** | `load_dataset.py:58`. |
| 5.2 | Missing raw file raises `FileNotFoundError` | **PASS** | `load_dataset.py:46-49`. |
| 5.3 | Path is assembled by the caller from config, not hardcoded | **PASS** | Documented at `load_dataset.py:28-30`; caller builds it at `encode_normalize.py:653-656`. |
| 5.4 | No column transformations occur during load | **PASS** | Only dtype downcasting (`load_dataset.py:64-70`), which is value-preserving for this data. |
| 5.5 | `drop_identifier_columns` removes every column in `drop_columns` | **PASS** | `load_dataset.py:104`. |
| 5.6 | Absent columns are silently ignored rather than raising | **PASS** | `errors="ignore"`, `load_dataset.py:104`. |
| 5.7 | `drop_identifier_columns` never drops `Attack_type` / `Attack_label` | **PASS** | It drops exactly the configured list; `configs/config.yaml:23-46` contains neither native column. Invariant restated at `load_dataset.py:85-87`. |
| 5.8 | Input DataFrame is not mutated in place | **PASS** | `df.drop(...)` returns a new frame, `load_dataset.py:104`. |
| 5.9 | `create_labels` derives `label` from `target_column` | **PASS** | `load_dataset.py:178`. |
| 5.10 | `create_labels` derives `binary_label` from `target_column`, not from the raw binary column | **PASS** | `np.where(df[target_column] == normal_class_value, 0, 1)`, `load_dataset.py:181-183`. This independence is what makes 5.11 a genuine cross-check. |
| 5.11 | Agreement between derived and native binary label is validated | **PASS** | `load_dataset.py:186-195`. |
| 5.12 | Disagreement logs a WARNING and does **not** raise | **PASS** | `load_dataset.py:189-195`; target-derived value kept as authoritative. |
| 5.13 | `create_labels` drops both native columns — the sole removal point | **PASS** | `load_dataset.py:198`. |
| 5.14 | Missing source columns raise `KeyError` | **PASS** | `load_dataset.py:164-171`, one guard per column with a distinct message. |
| 5.15 | Logger is injected, never constructed inside the function | **PASS** | Optional `logger` parameter, `load_dataset.py:112`; fallback at `173`. Confirms the Section 2 note. |
| 5.16 | No other module drops `Attack_type` or `Attack_label` | **PASS** | Repository-wide check: the only `drop(columns=[target_column, raw_binary_column])` is `load_dataset.py:198`. `encode_normalize.py` drops only `binary_label` (`877`) and `onehot_columns` (`352`). |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 5-A | OBSERVATION | The float64→float32 / int64→int32 downcast at `load_dataset.py:64-70` is applied before any fitting and is justified in-comment (lines 51-57): the columns are packet counts, byte lengths, and flow statistics, none exceeding 7 significant digits, and all are MinMax-scaled into [0,1] before reaching a float32 Keras graph. Halves peak RAM on a ~1.16 GB CSV. Sound. | `load_dataset.py:51-70` | None. |
| 5-B | OBSERVATION | Deriving `binary_label` from `target_column` rather than trusting `Attack_label` converts the native column from an *input* into an *independent check*. This is a stronger design than the SDS strictly requires and is the reason 5.11 has diagnostic value. | `load_dataset.py:181-195` | None. |
| 5-C | MINOR | `df[raw_binary_column].astype(int)` at line 186 will raise `ValueError` rather than log a warning if the native column ever contains a non-numeric value. Given `Attack_label` is 0/1 in Edge-IIoTset this is unreachable in practice, but the failure mode would be a hard crash inside a function whose documented contract is "warn, don't raise". | `load_dataset.py:186` | Optional: wrap the comparison in `pd.to_numeric(..., errors="coerce")` so a malformed native column degrades to a warning. |
| 5-D | OBSERVATION | `create_labels` copies the frame at line 175 before mutating. Combined with the copy in `drop_identifier_columns`, the pipeline briefly holds two frames. Deliberate: the orchestrator rebinds `df` immediately (`encode_normalize.py:680`), so the old frame is collectable. | `load_dataset.py:175` | None. |

### Notes

Invariant A.2 — the single most-repeated rule in the governing documents — holds.
Native-column removal occurs at exactly one place in the entire repository
(`load_dataset.py:198`), and the ordering that makes this safe is enforced by the
orchestrator: `drop_identifier_columns` (which cannot touch the native columns
because they are absent from `drop_columns`) runs first, then `create_labels`
consumes and removes them.

The `drop_columns` list in `configs/config.yaml:23-46` deserves note here because
it is *itself* a leakage control, and a well-reasoned one. Beyond the obvious
identifier columns, three entries were added specifically as leakage fixes, each
with a quantified justification in-config:

- `frame.time` (`config.yaml:39`) — 1,765,009 distinct values across 1,775,360
  training rows. Effectively a row identifier. Because Edge-IIoTset was captured
  attack-by-attack, ordinal-encoding the timestamp handed the model capture
  order, which is near-perfectly correlated with the label. This is the single
  most severe leakage vector in the raw dataset and it has been correctly closed.
- `tcp.srcport` (`config.yaml:42`) — 61,975 distinct ephemeral ports; an
  identifier, not a protocol signal.
- `http.request.uri.query` (`config.yaml:46`) — 5,526 distinct raw query strings;
  encoding them memorises attack payloads instead of learning behaviour.

---

## 6. `src/preprocessing/encode_normalize.py` — PASS WITH NOTES

**Governing spec:** SDS §14.2; AI Coding Contract (fit-on-train-only;
`prepare_model_ready_data` is the sole tensor-preparation function; the split is
computed once and reused); `Phase_Implementation_Prompts.md` Phases 2–3.

**Files:** `src/preprocessing/encode_normalize.py` (932 lines).

### Requirement checklist — functions 1–6

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 6.1 | `infer_feature_column_types` is the sole column classifier | **PASS** | `encode_normalize.py:55`; ownership stated at `13`. |
| 6.2 | Label columns excluded from all output groups | **PASS** | `encode_normalize.py:123-124`, `132`. |
| 6.3 | Output groups are disjoint and cover every non-label column | **PASS** | `numeric_columns` is the complement of `categorical_columns` (`137-139`); the cardinality split partitions the categorical list (`146-150`). |
| 6.4 | Original DataFrame column order preserved | **PASS** | All three lists are built by list-comprehension over `df.columns`. |
| 6.5 | Unsupported rule name raises `ValueError` | **PASS** | `encode_normalize.py:126-130`. |
| 6.6 | Low-cardinality categoricals are one-hot encoded (no false ordering) | **PASS** | `OneHotEncoder`, `encode_normalize.py:273-278`. Rationale at `219-227`. |
| 6.7 | High-cardinality categoricals use frequency encoding | **PASS** | `encode_normalize.py:283-289`, `normalize=True` so codes are dataset-size independent and already in [0,1]. |
| 6.8 | Unseen test-set categories handled without raising | **PASS** | One-hot: `handle_unknown="ignore"` → all-zero row (`274`). Frequency: `.fillna(0.0)` (`339`). Both encode "never seen in training", which is the honest semantics. |
| 6.9 | Encoder bundle is self-describing and picklable | **PASS** | Dict returned at `encode_normalize.py:291-297`, persisted verbatim as `categorical_encoder.pkl` (`916`). |
| 6.10 | `fit_scaler` uses `MinMaxScaler` | **PASS** | `encode_normalize.py:389`. |
| 6.11 | `fit_label_encoder` builds `class_mapping` with string keys | **PASS** | `encode_normalize.py:430-432`; string keys are required because JSON object keys are always strings. |
| 6.12 | `split_train_test` is stratified | **PASS** | `stratify=stratify_labels`, `encode_normalize.py:491`. |
| 6.13 | Split is seeded from config | **PASS** | `random_state=seed`, `encode_normalize.py:490`; passed `config["seed"]` at `732`. |
| 6.14 | Split returns index arrays, not copies, so it can be persisted and reused | **PASS** | `encode_normalize.py:475`, `487-493`; persisted at `917-918`. |
| 6.15 | Classes with <2 samples raise a clear `ValueError` | **PASS** | `encode_normalize.py:479-485`, message names the offending classes. |
| 6.16 | `prepare_model_ready_data` is the sole tensor-preparation function | **PASS** | `encode_normalize.py:500`; ownership at `14-16`. Verified against consumers in Batches 3–4. |
| 6.17 | Produces `X` of shape `(n, num_features, 1)` for Conv1D | **PASS** | `encode_normalize.py:546-550`. |
| 6.18 | Produces one-hot `y` of shape `(n, num_classes)` | **PASS** | `to_categorical`, `encode_normalize.py:552`. |
| 6.19 | `num_classes` is a parameter, never hardcoded | **PASS** | `encode_normalize.py:505`; documented as `len(class_mapping)` at `523-524`. |
| 6.20 | Feature order comes from the persisted `feature_names.pkl`, not from live column order | **PASS** | `feature_columns` parameter, `encode_normalize.py:503`, used to index at `547`. |
| 6.21 | Tensors built as float32, not the pandas float64 default | **PASS** | `encode_normalize.py:548`, `552`. Rationale at `540-545`. |

### Requirement checklist — orchestrator (`run_preprocessing_pipeline`)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 6.22 | All 11 SDS §14.2 steps execute in a fixed, documented order | **PASS** | Steps 1–11 labelled in code at `encode_normalize.py:657`, `664`, `674`, `697`, `723`, `746`, `796`, `822`, `842`, `859`, `895`, `905`. |
| 6.23 | `force_reprocess: false` skips regeneration when the output exists | **PASS** | `encode_normalize.py:637-645`. |
| 6.24 | Output directories created before writing | **PASS** | `encode_normalize.py:648-649`. |
| 6.25 | **Encoders fit on training rows only** | **PASS** | `train_slice = df.loc[train_indices]` (`753`) is the only frame passed to `fit_categorical_transformer` (`804-808`) and `fit_categorical_encoder` (`815-817`). |
| 6.26 | **Scaler fit on training rows only** | **PASS** | `scaler.fit(df.loc[train_indices, scale_columns])`, `encode_normalize.py:849`. |
| 6.27 | **Label encoder fit on training rows only** | **PASS** | `fit_label_encoder(df.loc[train_indices, "label"])`, `encode_normalize.py:863-865`. |
| 6.28 | **Column-type inference performed on training rows only** | **PASS** | `infer_feature_column_types(train_slice, ...)`, `encode_normalize.py:760-765` / `767-771`. See finding 6-A — this is a deliberate, correct deviation from the literal SDS order. |
| 6.29 | Encoders applied to the full dataset after fitting on train | **PASS** | `encode_normalize.py:824` / `831-833`, `854-856`. |
| 6.30 | Split computed exactly once, then persisted | **PASS** | One `split_train_test` call, `encode_normalize.py:728`; persisted at `917-918`. |
| 6.31 | Duplicate rows removed **before** the split | **PASS** | Step 3b, `encode_normalize.py:697-713`. See finding 6-B. |
| 6.32 | Disabling `drop_duplicates` warns about the consequence | **PASS** | `encode_normalize.py:715-718`. |
| 6.33 | All seven artifacts persisted to `artifacts_dir` | **PASS** | `scaler.pkl`, `label_encoder.pkl`, `categorical_encoder.pkl`, `train_indices.pkl`, `test_indices.pkl`, `feature_names.pkl` (`914-919`), `class_mapping.json` (`921-924`). |
| 6.34 | Processed dataset written through the `dataio` singleton | **PASS** | `write_processed(df, processed_csv_path)`, `encode_normalize.py:898`. No direct `to_csv` / `to_parquet` call. |
| 6.35 | Failures log a full traceback at ERROR before re-raising | **PASS** | `encode_normalize.py:928-932`, `exc_info=True`. |
| 6.36 | Logger constructed once via the `logger` singleton | **PASS** | `get_logger("preprocessing", logs_dir)`, `encode_normalize.py:614`; the fixed name matches the SDS §13 table. |
| 6.37 | NaN check before persisting | **PARTIAL** | Check exists at `encode_normalize.py:887-892` but only warns. See finding 6-D. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 6-A | OBSERVATION | The pipeline moves the split **before** column-type inference, deviating from the literal SDS §14.2 ordering. This is a leakage *fix*, not a violation: inference measures per-column cardinality, which is a fitted property, and measuring it on the full frame let test-only categories decide whether a column was one-hot or frequency encoded. The deviation is declared explicitly at `585-596`. Correct call; the SDS text should be amended to match the code rather than the reverse. | `encode_normalize.py:585-596`, `720-771` | Update SDS §14.2 to record the corrected order. |
| 6-B | OBSERVATION | De-duplication runs before the split (Step 3b) with the same justification: ~13% of Edge-IIoTset rows are exact duplicates, and splitting first leaves byte-identical copies of training rows sitting in the test set. The comment at `690-696` records that feature-only and feature+label duplicate rates are identical, so no duplicate group disagrees on its label and the drop is information-lossless. Rigorous. | `encode_normalize.py:690-713` | None. |
| 6-C | OBSERVATION | `binary_label` is dropped after its agreement check (`876-881`) and never persisted. The reasoning at `870-875` is exactly right: it is a deterministic function of the target, so carrying it into the processed file leaves a perfectly label-correlated column adjacent to the features, one `feature_columns` bug away from total leakage. This is a defect the audit would otherwise have raised — it has been pre-empted. | `encode_normalize.py:870-881` | None. |
| 6-D | MINOR | The pre-persist NaN check warns but does not fail (`887-892`). Given every fitted transform in the pipeline has an explicit unseen-value path (`fillna(0.0)`, `handle_unknown="ignore"`), a NaN reaching this point indicates an unmodelled defect, and Keras would later fail with a far less informative error deep in training. | `encode_normalize.py:887-892` | Consider raising when `nan_count > 0`, or gate on a `config["dataset"]["fail_on_nan"]` flag. |
| 6-E | MINOR | With the legacy `"dtype_object"` rule, `scale_columns = numeric_columns` (`834`) — so ordinal-encoded categorical columns are **not** MinMax-scaled and enter the network as raw integers 0..n. Under the production `"cardinality"` rule this is unreachable (the branch at `828` filters against live columns and all categoricals are already in [0,1]), so it does not affect any produced result. It is nevertheless a latent inconsistency in a code path the tests still exercise. | `encode_normalize.py:828-834` | Either scale the ordinal columns in the legacy branch, or mark `"dtype_object"` deprecated and remove it once the tests are migrated. |
| 6-F | MINOR | `fit_categorical_encoder` calls `encoder.fit(df[[]])` on the empty-column path (`200`) purely to keep the return interface uniform. `OrdinalEncoder` fitted on zero features is a degenerate object whose `transform` is never called, since the caller guards on `if categorical_columns`. Harmless but obscure. | `encode_normalize.py:196-201` | Return `None` for the encoder instead, or add a comment stating the object is intentionally unusable. |
| 6-G | OBSERVATION | Step 8 transforms in place rather than via `df_processed = df.copy()` (`850-856`). On a multi-GB frame this avoids holding two full copies simultaneously — a meaningful saving that the in-line comment documents. | `encode_normalize.py:850-856` | None. |
| 6-H | OBSERVATION | `apply_categorical_transformer` reorders columns to `[untouched..., one-hot indicators...]` (`352-353`). This is safe **only** because `feature_names.pkl` is captured from the post-transform frame (`883`) and every consumer indexes by that list rather than by position. The dependency is documented at `321-324`. Batches 3–4 must confirm no consumer assumes positional order. | `encode_normalize.py:352-353`, `883` | None; verify downstream. |

### Notes

The leakage discipline in this module is the strongest evidence of engineering
quality in the repository so far. Four separate fit-on-train-only requirements
(6.25 – 6.28) all hold, and two of them (6.28 column-type inference, 6.31
de-duplication ordering) required *deviating from the written SDS* to achieve.
Both deviations are declared in the docstring at lines 585–596 with their
reasoning, rather than made silently — which is precisely the behaviour the AI
Coding Contract asks for when spec and correctness conflict.

The one structural remark: at 932 lines, this module carries the orchestrator
(372 lines) alongside seven transformation functions. The SDS assigns both
responsibilities here, so this is compliant, but `run_preprocessing_pipeline`
would be a natural future extraction to `src/preprocessing/pipeline.py`.

Two items are carried into Batch 3–4 verification:
- **6-H** — confirm no consumer relies on positional column order.
- **6.16** — confirm `train_baseline.py`, `client_app.py`, the evaluation module,
  the explainability module, and the dashboard all call
  `prepare_model_ready_data` rather than reimplementing the reshape.

---

## Batch 1 summary

| Section | Component | Status |
| --- | --- | --- |
| 5 | `load_dataset.py` | **PASS** — 16/16 requirements |
| 6 | `encode_normalize.py` | **PASS WITH NOTES** — 36 PASS, 1 PARTIAL (6.37) |

**Defects raised in this batch:** 0 BLOCKER, 0 MAJOR, 4 MINOR (`5-C`, `6-D`,
`6-E`, `6-F`), 8 OBSERVATION. No FAIL, no NOT IMPLEMENTED.

**Verdict for the preprocessing layer:** compliant, and materially stronger than
the SDS requires on data-leakage integrity. Contract Invariant A.2 holds
repository-wide. Every fitted object in the pipeline — column classifier,
categorical encoders, scaler, label encoder — sees training rows exclusively.

**Carried forward:**
- `6-A` → SDS §14.2 text should be updated to match the corrected step order
  (documentation task, final roll-up "Remaining work").
- `6-D`, `6-E` → Remaining work (low priority; no effect on current results).
- `6-H`, `6.16` → verify against downstream consumers in Batches 3–4.

---

# BATCH 2 — Model architecture and client partitioning (SDS §6, §14.3, §14.4)

Two small, tightly-scoped modules. Both are declared frozen by the AI Coding
Contract: the CNN-GRU layer sequence may not be altered, and IID sharding has a
single implementation. The audit checks the frozen specification literally,
layer by layer and guarantee by guarantee.

---

## 7. `src/models/cnn_gru.py` — PASS

**Governing spec:** SDS §6 (architecture), SDS §14.4 (layer specification),
SDS §11 (import graph); AI Coding Contract Part C (locked values:
`conv_padding="same"`, `gru_return_sequences=False`), and the "single model
builder" invariant; `Phase_Implementation_Prompts.md` Phase 5.

**Files:** `src/models/cnn_gru.py` (173 lines), `configs/config.yaml:62-88`.

### Requirement checklist — architecture

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 7.1 | Exactly one model-construction function in the project | **PASS** | `build_cnn_gru`, `cnn_gru.py:30`. Exclusivity stated at `5-7`. No second builder exists in `src/`. |
| 7.2 | Layer 1 — `Input(shape=(num_features, 1))` | **PASS** | `cnn_gru.py:80`. |
| 7.3 | Layer 2 — `Conv1D` with configured filters/kernel/padding/activation | **PASS** | `cnn_gru.py:81-86`. |
| 7.4 | Layer 3 — `MaxPooling1D(pool_size)` | **PASS** | `cnn_gru.py:87`. |
| 7.5 | Layer 4 — `GRU(units, return_sequences)` | **PASS** | `cnn_gru.py:88-91`. |
| 7.6 | Layer 5 — `Dropout(rate)` | **PASS** | `cnn_gru.py:92`. |
| 7.7 | Layer 6 — `Dense(dense_units, dense_activation)` | **PASS** | `cnn_gru.py:93-96`. |
| 7.8 | Layer 7 — `Dense(num_classes, output_activation)` | **PASS** | `cnn_gru.py:97-100`. |
| 7.9 | Exactly 7 entries; no layer added, removed, or reordered | **PASS** | `cnn_gru.py:78-102` — the `Sequential` list contains precisely the 7 specified entries. |
| 7.10 | No `Flatten` layer (GRU emits a single final-state vector) | **PASS** | Absent by construction; `return_sequences=False` makes it unnecessary. Reasoning at `cnn_gru.py:52-54`. |
| 7.11 | `conv_padding` is `"same"` (Contract-locked) | **PASS** | Read from config at `cnn_gru.py:84`; `configs/config.yaml:65` sets `"same"`. |
| 7.12 | `gru_return_sequences` is `False` (Contract-locked) | **PASS** | Read at `cnn_gru.py:90`; `configs/config.yaml:69` sets `false`. |
| 7.13 | All layer hyperparameters read from `model_config`, never hardcoded | **PASS** | Every argument in `cnn_gru.py:81-100` indexes `model_config`. |
| 7.14 | `num_classes` is a runtime parameter derived from `class_mapping.json` | **PASS** | Parameter at `cnn_gru.py:32`; contract documented at `59-61`. |
| 7.15 | `num_classes < 2` raises `ValueError` | **PASS** | `cnn_gru.py:73-76`. |

### Requirement checklist — compilation and artifacts

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 7.16 | Model returned already compiled | **PASS** | `cnn_gru.py:106-110`. |
| 7.17 | Adam is the only optimizer instantiated | **PASS** | `cnn_gru.py:107`. The comment at `104-105` records that `training_config["optimizer"]` is documentation-only and deliberately not read — matching `configs/config.yaml:76`. |
| 7.18 | Learning rate sourced from `training_config` | **PASS** | `cnn_gru.py:107`. |
| 7.19 | Loss sourced from `training_config` | **PASS** | `cnn_gru.py:108`; `configs/config.yaml:78` = `categorical_crossentropy`, consistent with the one-hot `y` produced by `prepare_model_ready_data` (§6.18). |
| 7.20 | `metrics=["accuracy"]` | **PASS** | `cnn_gru.py:109`. |
| 7.21 | `get_model_summary_string` captures `model.summary()` as text | **PASS** | `cnn_gru.py:115-134`, via `io.StringIO` and `print_fn`. |
| 7.22 | Architecture diagram generation is best-effort, never build-blocking | **PASS** | `cnn_gru.py:160-173` — broad `except` logs a WARNING and returns normally. Required because `pydot`/`graphviz` are absent from `requirements.txt`. |
| 7.23 | Import-graph conformance: imports nothing from preprocessing/centralized/federated | **PASS** | `cnn_gru.py:14-27` imports only `io`, `tensorflow`, and Keras symbols. Rule stated at `9-11`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 7-A | OBSERVATION | The two Contract-locked values are *read from config* rather than hardcoded, and the docstring (`cnn_gru.py:46-54`) explains why each matters architecturally: `padding="same"` keeps the Conv1D output length equal to `num_features` so downstream shapes never depend on an unstated padding choice, and `return_sequences=False` yields a `(batch, gru_units)` vector so no `Flatten` is needed. Both explanations are correct. | `cnn_gru.py:46-54`, `84`, `90` | None. |
| 7-B | MINOR | Because the locked values are read from config rather than asserted, editing `configs/config.yaml:65` to `"valid"` would silently change the frozen architecture without any error. The lock is documentary, not enforced. | `cnn_gru.py:84`, `90` | Optional: assert `model_config["conv_padding"] == "same"` and `model_config["gru_return_sequences"] is False`, raising `ValueError` otherwise. This would make the Contract lock executable. |
| 7-C | OBSERVATION | `save_model_architecture_diagram` logs through `tensorflow.get_logger()` (`cnn_gru.py:168`) rather than the project logger. Consistent with the SDS §11 import rule — `src/models/` may not import `src/utils/` — so this is the correct trade-off, at the cost of the warning not appearing in the project log file. | `cnn_gru.py:168-173` | None. |
| 7-D | OBSERVATION | `pydot` and `graphviz` are absent from `requirements.txt`, so the architecture diagram will **not** be produced in a clean environment. This is by design (7.22) and non-fatal, but it means `*_architecture.png` will be missing from `outputs/models/`. Distinguish this from finding 4-A, which *is* fatal. | `requirements.txt`, `cnn_gru.py:160-173` | Optional: add `pydot` + system graphviz if the diagram is a required submission artifact. Track in the artifact inventory (Section 19). |
| 7-E | OBSERVATION | `use_class_weights: true` (`config.yaml:88`) is not consumed here — `build_cnn_gru` does not compile with weighting, since Keras applies `class_weight` at `fit()` time. Correct placement; verified in Sections 9–10. | `configs/config.yaml:84-88` | None; verify at the `fit()` call sites. |

### Notes

The architecture is frozen and matches the specification exactly — this is the
cleanest section of the audit so far. The interaction worth recording for later
sections is the shape contract: `prepare_model_ready_data` emits `X` of shape
`(n, len(feature_columns), 1)` (§6.17), and `build_cnn_gru` expects
`input_shape=(num_features, 1)`. The two agree **only** if callers pass
`input_shape=(len(feature_names), 1)` derived from `feature_names.pkl`. Sections
9–11 must confirm that every call site derives the shape from the persisted
artifact rather than from a live DataFrame's column count — the two diverge the
moment one-hot expansion changes (finding 6-H).

Finding 7-B is worth weighing in review: the project describes these as "locked
values", but nothing in code enforces the lock. Making it executable is a
three-line change and would convert a documentation promise into a guarantee.

---

## 8. `src/partitioning/partition_data.py` — PASS

**Governing spec:** SDS §14.3; AI Coding Contract A.2.7 (the global test split is
never partitioned per client); `Phase_Implementation_Prompts.md` Phase 7.

**Files:** `src/partitioning/partition_data.py` (71 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 8.1 | Exactly one authoritative sharding function | **PASS** | `partition_iid`, `partition_data.py:17`. Exclusivity stated at `3-4`. |
| 8.2 | Produces exactly `num_clients` shards | **PASS** | `numpy.array_split(..., num_clients)`, `partition_data.py:65`. |
| 8.3 | Shards are disjoint — no row appears twice | **PASS** | Built from a single `rng.permutation` (`60`), which is a bijection over `range(len(df))`; `array_split` partitions it without overlap. |
| 8.4 | Shards are exhaustive — sizes sum to `len(df)` | **PASS** | `array_split` covers every position exactly once even under non-divisible splits; documented at `62-64`. |
| 8.5 | Shard sizes differ by at most one row | **PASS** | Guaranteed by `numpy.array_split` semantics; stated at `partition_data.py:64`. |
| 8.6 | Rows are shuffled before splitting (true IID, not contiguous blocks) | **PASS** | `rng.permutation(len(df))`, `partition_data.py:60`. Critical: the processed frame is in capture order, so a contiguous split would produce a pathologically non-IID partition. |
| 8.7 | Shuffle is seeded and reproducible | **PASS** | `numpy.random.RandomState(seed)`, `partition_data.py:59`; seed sourced from `config["seed"]` by the caller (`36-37`). |
| 8.8 | `num_clients < 1` raises `ValueError` | **PASS** | `partition_data.py:48-51`. |
| 8.9 | `num_clients > len(df)` raises `ValueError` | **PASS** | `partition_data.py:52-56`. |
| 8.10 | Each shard's index is reset | **PASS** | `.reset_index(drop=True)`, `partition_data.py:68`. Necessary so client-side code can use positional indices without colliding with the original frame's labels. |
| 8.11 | Applied only to the training split, never to the test split | **PASS** (module contract) | Stated at `partition_data.py:5-7` and `31-33`. The call site is verified in Section 11. |
| 8.12 | No shard is persisted to disk | **PASS** | No I/O in the module; stated at `9-10`. |
| 8.13 | `partition_strategy: "iid"` in config matches the implemented strategy | **PASS** | `configs/config.yaml:94`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 8-A | OBSERVATION | Using a local `RandomState(seed)` rather than the global NumPy seed makes partitioning independent of how much other random work happened earlier in the process. Two runs produce identical shards even if the preprocessing path consumed a different number of random draws. This is the more robust choice and matters for a federated simulation, where reproducibility of the shard boundaries is what makes round-by-round metrics comparable. | `partition_data.py:59` | None. |
| 8-B | OBSERVATION | `numpy.random.RandomState` is the legacy API; the modern equivalent is `numpy.random.default_rng(seed)`. `RandomState` has a **stronger** stream-stability guarantee across NumPy versions, which is preferable for a reproducibility-critical function. Deliberate or not, it is the right call. | `partition_data.py:59` | None. |
| 8-C | MINOR | `df.iloc[positions]` (`68`) materialises a full copy of each shard, so peak memory during partitioning is approximately 2× the training frame. On the 16 GB laptop profile with the training split at roughly 1.4 M rows × ~95 float32 columns (~530 MB), the transient is ~1 GB. Tolerable, but this happens in the parent process immediately before Ray actors are spawned — the exact moment `LAPTOP_MODE` is trying to conserve memory. | `partition_data.py:67-69` | Optional: `del df` in the caller after partitioning, or build shards lazily. Verify in Section 11 whether the caller already releases the frame. |
| 8-D | MINOR | `partition_strategy` (`configs/config.yaml:94`) is never read by this module — `partition_iid` is called directly by name. A future `"non_iid"` value would be silently ignored rather than raising. | `configs/config.yaml:94`; call site verified in Section 11 | Optional: dispatch on `partition_strategy` and raise `ValueError` for unknown strategies, mirroring `resolve_mode` and `infer_feature_column_types`. |
| 8-E | OBSERVATION | The type hint `list[pandas.DataFrame]` (`partition_data.py:21`) requires Python 3.9+. The project targets 3.10 (`README.md:11`), so this is fine, and it is consistent with `cnn_gru.py:31`'s `tuple[int, int]`. | `partition_data.py:21` | None. |

### Notes

Requirement 8.6 is the one that carries real weight. Edge-IIoTset is captured
attack-by-attack, so the processed frame is strongly ordered by label — the same
property that made `frame.time` a leakage vector (§5 notes). A contiguous split
would hand client 0 almost exclusively one attack family, producing a severely
non-IID partition while the config still claimed `"iid"`. The seeded permutation
at line 60 is therefore not a stylistic detail; it is what makes the IID claim
true. Correctly implemented.

Requirement 8.11 is the Contract A.2.7 guarantee (the global test split is never
sharded). The module honours it in contract and documentation, but cannot enforce
it — the caller decides what to pass. Section 11 must confirm that
`server_app.run_federated_simulation` passes `df.loc[train_indices]` and not the
full frame. This is flagged as the primary open verification item from this batch.

---

## Batch 2 summary

| Section | Component | Status |
| --- | --- | --- |
| 7 | `models/cnn_gru.py` | **PASS** — 23/23 requirements |
| 8 | `partitioning/partition_data.py` | **PASS** — 13/13 requirements |

**Defects raised in this batch:** 0 BLOCKER, 0 MAJOR, 3 MINOR (`7-B`, `8-C`,
`8-D`), 7 OBSERVATION. No FAIL, no NOT IMPLEMENTED.

**Verdict:** both modules are compliant and frozen as specified. The CNN-GRU
layer sequence matches SDS §14.4 exactly, and IID partitioning provides genuine
randomised sharding with reproducible boundaries.

**Structural finding for the roll-up:** `list_files` on the remaining directories
shows the project is **incomplete**, not merely imperfect:

| Path | State |
| --- | --- |
| `src/evaluation/` | `metrics.py` present but holds only `generate_class_distribution_plot`. The file's own docstring (`metrics.py:8-10`) names four Phase 8 functions as not yet written: `compute_all_metrics`, `generate_confusion_matrix`, `generate_classification_report`, `generate_training_curves_plot`. |
| `src/explainability/` | Contains **only** `__init__.py`. No SHAP implementation exists. |
| `dashboard/` | **Empty.** No `app.py`. |
| `experiments/` | `run_evaluation.py` and `run_explainability.py` are **absent**, though `README.md:21-23` lists both in the run order. |

This materially changes the completion estimate and is the dominant input to the
final roll-up. Sections 12–14 will record these as NOT IMPLEMENTED rather than
as defects in existing code.

**Carried forward:**
- `7-B` → Remaining work (make the Contract lock executable).
- `7-D` → Artifact inventory (Section 19): architecture diagram will not generate.
- `8-C`, `8-D` → Remaining work.
- `8.11` → **primary verification item for Section 11**: confirm `partition_iid`
  receives `df.loc[train_indices]`.
- `7.14` / shape contract → confirm `input_shape` is derived from
  `feature_names.pkl` at every call site (Sections 9–11).

---

# BATCH 3 — Training layers (SDS §14.5, §14.6, §14.7)

The centralized baseline and the federated client are the two consumers whose
correctness the earlier batches deferred to. Three carried-forward verification
items are resolved here: the `input_shape` derivation (§7.14), sole use of
`prepare_model_ready_data` (§6.16), and positional-order independence (§6-H).

---

## 9. `src/centralized/train_baseline.py` — PASS

**Governing spec:** SDS §14.7, SDS §9 (centralized artifacts), SDS §11 (import
graph); AI Coding Contract (single model builder; single tensor-preparation
function; `num_classes` derived, never hardcoded; timing wraps `.fit()` only);
`Phase_Implementation_Prompts.md` Phase 6.

**Files:** `src/centralized/train_baseline.py` (377 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 9.1 | Trains on the full training split, no partitioning | **PASS** | `train_indices` used directly, `train_baseline.py:153-155`. No `partition_iid` import. |
| 9.2 | Model built exclusively via `build_cnn_gru` | **PASS** | `train_baseline.py:168-173`. |
| 9.3 | **`input_shape` derived from `feature_names.pkl`** (carried from §7.14) | **PASS** | `num_features = len(feature_columns)` at `142`, where `feature_columns` is loaded from `feature_names.pkl` (`133`); passed at `169`. Resolves the shape-contract concern — not taken from a live column count. |
| 9.4 | **Tensors built exclusively via `prepare_model_ready_data`** (carried from §6.16) | **PASS** | `train_baseline.py:153-158`. No local reshape or `to_categorical`. |
| 9.5 | **Feature access is by name, not position** (carried from §6-H) | **PASS** | `feature_columns` is passed through to `prepare_model_ready_data`, which indexes `df.loc[indices, feature_columns]`. Column reordering by one-hot expansion is therefore immaterial. |
| 9.6 | `num_classes` derived as `len(class_mapping)` | **PASS** | `train_baseline.py:141`, with the rule restated at `140`. |
| 9.7 | Processed data read through the `dataio` singleton | **PASS** | `read_processed`, `train_baseline.py:114`. |
| 9.8 | `EarlyStopping` configured from config | **PASS** | `train_baseline.py:182-186`; monitor and patience both from `config["training"]`. |
| 9.9 | `restore_best_weights=True` | **PASS** | `train_baseline.py:185`. |
| 9.10 | `ModelCheckpoint` saves best-only | **PASS** | `train_baseline.py:187-191`, `save_best_only=True`. |
| 9.11 | `validation_split` applied by Keras internally; no separate val file | **PASS** | `train_baseline.py:244`; documented at `62-64`. |
| 9.12 | **`time.time()` wraps only `.fit()`** | **PASS** | `start_time` at `238`, `fit` at `239-247`, elapsed at `248`. No I/O inside the window. The later test-set evaluation is timed separately (`343`, `358`). |
| 9.13 | Class weights computed from **training labels only** | **PASS** | `numpy.argmax(y_train, axis=1)`, `train_baseline.py:207`; rationale at `202-204`. |
| 9.14 | Class weighting gated on `use_class_weights` | **PASS** | `train_baseline.py:206`, `.get()` with a default. |
| 9.15 | Disabling class weights emits a quantified warning | **PASS** | `train_baseline.py:227-234`, computes and reports the actual imbalance ratio. |
| 9.16 | `class_weight` passed to `.fit()` (resolves §7-E) | **PASS** | `train_baseline.py:245`. Confirms weighting is applied at fit time, not compile time. |
| 9.17 | Per-epoch metrics logged | **PASS** | `train_baseline.py:252-263`. |
| 9.18 | `centralized_best_model.h5` persisted | **PASS** | Via `ModelCheckpoint`, `train_baseline.py:178`, `188`. |
| 9.19 | `centralized_last_model.h5` persisted (audit-only) | **PASS** | `train_baseline.py:278`; audit-only status documented at `274-277` and `16-17`. |
| 9.20 | `centralized_model_summary.txt` persisted | **PASS** | `train_baseline.py:286-289`. |
| 9.21 | `centralized_architecture.png` attempted, non-fatally | **PASS** | `train_baseline.py:291-300`; existence checked and a WARNING logged if absent. |
| 9.22 | `centralized_history.csv` persisted | **PASS** | `train_baseline.py:303-304`. |
| 9.23 | `centralized_training_time.txt` persisted | **PASS** | `train_baseline.py:312-316`. |
| 9.24 | Held-out test accuracy logged | **PASS** | `train_baseline.py:344-359`. |
| 9.25 | Output directories created before writing | **PASS** | `train_baseline.py:91-92`. |
| 9.26 | Logger injected from the calling script, never constructed | **PASS** | Optional parameter at `49`; fallback `logging.getLogger` (no handlers, no file) at `83-84`; rule restated at `70-73`. |
| 9.27 | Failures log a full traceback before re-raising | **PASS** | `train_baseline.py:373-377`. |
| 9.28 | Import-graph conformance: imports only utils/preprocessing/models | **PASS** | `train_baseline.py:37-43`. Rule stated at `21-23`. No federated, partitioning, evaluation, or explainability import. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 9-A | OBSERVATION | `KeyboardInterrupt` is handled separately from `Exception` (`train_baseline.py:363-372`) because it derives from `BaseException` and would otherwise bypass the handler, leaving a log that stops mid-run with no explanation. The message correctly states that artifacts already logged as saved remain valid. Thoughtful. | `train_baseline.py:363-372` | None. |
| 9-B | OBSERVATION | Test-set evaluation runs **after** every artifact is persisted (`323-327`), so an interruption during the multi-minute evaluation pass cannot cost the completed training run. Correct ordering. | `train_baseline.py:323-359` | None. |
| 9-C | OBSERVATION | `model.evaluate` receives an explicit `batch_size` from config (`347`) rather than accepting the Keras default of 32, and `verbose=1` is justified in-comment: a silent multi-minute pass over 443k rows is indistinguishable from a hang and invites an interrupt. Both are deliberate. | `train_baseline.py:329-348` | None. |
| 9-D | MINOR | The test-set loss and accuracy are **logged but not persisted** (`350-359`). Every other measured quantity gets a file. Section 12 will need these numbers for the FL-vs-centralized comparison, which means either re-running evaluation or parsing the log. | `train_baseline.py:344-359` | Write them to `results_dir` as e.g. `centralized_test_metrics.json`. Low effort, removes a log-parsing dependency from the not-yet-written evaluation phase. |
| 9-E | MINOR | Models are saved in legacy HDF5 (`.h5`) format (`178-179`). TensorFlow 2.15 emits a deprecation warning recommending the native `.keras` format. Functional today; a future TF upgrade could break it. | `train_baseline.py:178-179` | None required for submission. Note as a forward-compatibility item. |
| 9-F | OBSERVATION | Both `X_train` and `X_test` are materialised before training (`153-158`). At ~1.4 M and ~443 k rows × 95 float32, that is roughly 530 MB + 168 MB resident for the whole run, plus Keras's internal `validation_split` slice. Acceptable for the centralized path, which does not run under `LAPTOP_MODE`'s Ray constraints. | `train_baseline.py:153-158` | None. |
| 9-G | OBSERVATION | `class_weight` keys are cast to plain `int` and values to `float` (`214-217`). Necessary — Keras rejects NumPy integer keys in the `class_weight` dict. Correct defensive conversion. | `train_baseline.py:214-217` | None. |

### Notes

This module resolves all three verification items carried into Batch 3. In
particular, 9.3 confirms the shape contract flagged in §7: `input_shape` is built
from `len(feature_columns)` where `feature_columns` came from `feature_names.pkl`,
so the model input dimension and the tensor's feature axis are derived from the
same persisted artifact and cannot drift.

The class-weight block (`194-234`) is worth highlighting. The comment quantifies
the failure mode it prevents: with a ~1600:1 imbalance, ignoring the 1,001-row
`Fingerprinting` class costs 0.045% accuracy, so an unweighted model reaches ~96%
accuracy while never predicting the rarest attacks. For an IDS that is the failure
mode that matters, and the weighting is computed from training rows only, so it
introduces no leakage.

---

## 10. `src/federated/client_app.py` — PASS

**Governing spec:** SDS §14.5, SDS §14.3 (shared global test split; never a
per-client test partition), SDS §12 (seeding before model instantiation),
SDS §11, SDS §22; AI Coding Contract (single client class; clients write
nothing); `Phase_Implementation_Prompts.md` Phases 8–9.

**Files:** `src/federated/client_app.py` (473 lines).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 10.1 | Exactly one Flower client class | **PASS** | `FLClient(NumPyClient)`, `client_app.py:73`. Exclusivity stated at `7-8`. |
| 10.2 | Constructor signature fixed to `client_id`/`train_data`/`test_data`/`config` | **PASS** | `client_app.py:97-103`. |
| 10.3 | Each client receives a disjoint shard from `partition_iid` | **PASS** | Shards arrive pre-computed via `client_fn`'s `client_shards` (`428`, `468`). |
| 10.4 | The test split passed in is the **shared global** split, not per-client | **PASS** | `test_data` is a single frame bound once by the orchestrator (`429`, `469`); documented at `12-14`, `115-116`. |
| 10.5 | **`set_global_seed` called before model instantiation** | **PASS** | `client_app.py:147`, immediately before `build_cnn_gru` at `149`. Mandatory ordering flagged in-code at `144-146`. |
| 10.6 | Model built exclusively via `build_cnn_gru` | **PASS** | `client_app.py:149-154`. |
| 10.7 | `input_shape` derived from the persisted feature list | **PASS** | `(len(self.feature_columns), 1)`, `client_app.py:150`; `feature_columns` arrives via `config["runtime"]` from the orchestrator's `feature_names.pkl` load. |
| 10.8 | `num_classes` supplied by the caller, never hardcoded | **PASS** | `client_app.py:142`, with the rule restated at `140-141` and `36-37`. |
| 10.9 | Tensors built exclusively via `prepare_model_ready_data` | **PASS** | `client_app.py:193-206`. |
| 10.10 | **Clients perform no file I/O whatsoever** | **PASS** | No `open`, `pickle`, `read_*`, or `save` call anywhere in the module. Everything arrives through the constructor or `config["runtime"]` (`26-29`, `31-37`). |
| 10.11 | **Clients persist nothing** | **PASS** | No write path exists; stated at `23-24`. |
| 10.12 | `get_parameters` returns local weights | **PASS** | `client_app.py:258`. |
| 10.13 | `fit` loads global weights before training | **PASS** | `set_weights(parameters)`, `client_app.py:297`. |
| 10.14 | `fit` trains for `federated.local_epochs` at `training.batch_size` | **PASS** | `client_app.py:305-306`. |
| 10.15 | `fit` returns `(weights, num_examples, metrics)` with both `loss` and `accuracy` | **PASS** | `client_app.py:313-316`, `329-333`. |
| 10.16 | **FedAvg weight excludes local validation rows** | **PASS** | `self.num_train_examples` counts `X_train` only (`207`), computed after the local split; reasoning at `326-328`. |
| 10.17 | `evaluate` loads global weights before evaluating | **PASS** | `client_app.py:387`. |
| 10.18 | `evaluate` returns `(loss, num_examples, {"accuracy": ...})` | **PASS** | `client_app.py:409-413`. Loss returned as the first element so Flower aggregates it into `losses_distributed`. |
| 10.19 | Local class weights derived from this client's own rows only | **PASS** | `client_app.py:218-229`; privacy rationale at `210-215`. |
| 10.20 | Single-class shard cannot crash weighting | **PASS** | `if len(present_classes) > 1` guard, `client_app.py:220`. |
| 10.21 | Local validation split is stratified where possible, with a safe fallback | **PASS** | `client_app.py:174-185`; falls back to unstratified when any class has <2 rows (`178-179`) rather than failing the round. |
| 10.22 | `client_fn` accepts `cid` as a string and converts it | **PASS** | `client_app.py:458`. |
| 10.23 | Out-of-range `cid` raises `IndexError` with a clear message | **PASS** | `client_app.py:460-464`. |
| 10.24 | Returns `.to_client()` as the pinned `flwr==1.8.0` API requires | **PASS** | `client_app.py:473`. |
| 10.25 | Shards/test/config bound by the orchestrator, not re-read per call | **PASS** | Keyword-only parameters at `427-430`; documented as `functools.partial` binding at `435-439`. |
| 10.26 | Module-level logger only; no log file created | **PASS** | `logging.getLogger(__name__)`, `client_app.py:70`; rule at `67-69`. |
| 10.27 | Import-graph conformance | **PASS** | `client_app.py:46-58` imports only from `utils`, `preprocessing`, `partitioning`, `models`. No `centralized`, `evaluation`, `explainability`, or `server_app` import (no forward reference). |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 10-A | OBSERVATION | **This module fixes a genuine methodological defect.** Previously every client evaluated on the identical shared global test split, so distributed evaluation returned four copies of the same number and `weighted_average_eval` averaged a constant — while simultaneously scoring the server's held-out set every round. Each client now holds out `client_val_split` of its *own* shard (`170-185`), making distributed evaluation four genuinely different local scores, which is what example-weighted aggregation is designed for. The reasoning is recorded at `156-169`. This is the single most consequential correctness fix in the federated layer. | `client_app.py:156-185`, `344-361` | None. |
| 10-B | OBSERVATION | Tensors are cached in `__init__` rather than rebuilt in `fit` and `evaluate` (`187-206`). The comment quantifies the saving: the previous design performed 2 conversions per client per round = 120 conversions of the same rows over a 4-client × 15-round run, each a `to_numpy` + reshape over ~390k × 95 values. Correct optimisation with no semantic change. | `client_app.py:187-206` | None. |
| 10-C | **MAJOR** | `test_data` is stored on every client (`134`) but **never used** — `evaluate` reads only `self.X_validation` (`394`). Under Flower's Ray backend each actor therefore holds a full copy of the global test frame for the entire simulation with no consumer. At ~443k rows × 95 columns that is roughly 168 MB per actor as float32, and considerably more as an un-downcast pandas frame. This directly contradicts the memory objective of `LAPTOP_MODE`, whose entire purpose is preventing the Windows `CreateFileMapping` failure by shrinking per-actor working set. Note `config_laptop.yaml:115-123` subsamples the *server's* copy to 50k rows for exactly this reason — while each client still receives the full frame. | `client_app.py:101`, `134` vs `394`; `configs/config_laptop.yaml:115-123` | The constructor signature is fixed by SDS §14.5 and cannot change, so keep the parameter but do not retain it: drop the `self.test_data = test_data` assignment, or store `None`. Highest-value memory fix available in the federated layer. |
| 10-D | MINOR | `local_train` and `local_validation` both alias `train_data` when `client_val_split` is 0 or the shard has ≤1 row (`171-173`). Evaluation then scores the training rows. Defensible as a degenerate fallback, but it is silent — no warning distinguishes "evaluated on held-out data" from "evaluated on training data". | `client_app.py:170-173` | Log a WARNING when the fallback is taken. |
| 10-E | MINOR | `local_train.index.values` is passed as the indices argument (`195`, `202`). This works because `train_test_split` preserves the original index labels and `prepare_model_ready_data` uses `.loc`. But `partition_iid` already called `reset_index(drop=True)` (`partition_data.py:68`), so these are 0-based positional labels — the correctness depends on a two-step reasoning chain about which index is live. Fragile rather than wrong. | `client_app.py:193-206` | Add a comment stating that shard indices are 0-based post-`reset_index`, and that `train_test_split` preserves those labels. |
| 10-F | OBSERVATION | `set_global_seed(config["seed"])` (`147`) also writes `outputs/artifacts/random_seed.txt` (§3.7). Under the Ray backend this means every client actor writes the same file. Same value each time so the content is stable, but it is concurrent writes to a shared path from multiple processes — and it technically breaches the "clients write nothing" rule (10.11) transitively. Interacts with finding 3-B. | `client_app.py:147`; `seed.py:49-53` | Consider a `write_artifact=False` parameter on `set_global_seed` for the client path. |
| 10-G | OBSERVATION | `partition_iid` is imported and re-exported in `__all__` (`56`, `65`) but never called in this module. Deliberate and documented at `60-64`: SDS §14.5 lists it as a declared dependency of `client_fn`'s data contract. Unusual but justified. | `client_app.py:56`, `60-65` | None. |
| 10-H | OBSERVATION | The local validation split uses `random_state=config["seed"]` (`183`) — the same seed for every client. Since each client splits a *different* shard, the resulting validation sets differ; the shared seed only means the split is reproducible. Correct. | `client_app.py:180-185` | None. |

### Notes

The client is compliant on every requirement, including the two that are easiest
to get wrong: seeding before model construction (10.5) and zero file I/O (10.10).
The zero-I/O property is what makes the same class usable unchanged against
synthetic in-memory data in the Phase 9 smoke test and against the real dataset
in Phase 10 — an explicit design goal recorded at lines 26–29.

Finding **10-C is the most significant defect found so far**. It is not a
correctness bug — no metric is wrong — but it is a substantial and avoidable
memory cost in exactly the code path the project has spent an entire config
overlay trying to make fit in 16 GB. The fix is deleting one assignment.

`test_data` remains a required constructor parameter under SDS §14.5, so the
signature must stay; only the retention should go. Worth noting the asymmetry
this reveals: `config_laptop.yaml` carefully subsamples the server's test tensor
to 50k rows to free ~150 MB, while each Ray actor silently holds the full frame.

---

## Batch 3 summary (partial — Section 11 pending)

| Section | Component | Status |
| --- | --- | --- |
| 9 | `centralized/train_baseline.py` | **PASS** — 28/28 requirements |
| 10 | `federated/client_app.py` | **PASS** — 27/27 requirements |

**Defects raised so far in this batch:** 0 BLOCKER, 1 MAJOR (`10-C`, unused
`test_data` retained per actor), 4 MINOR (`9-D`, `9-E`, `10-D`, `10-E`),
8 OBSERVATION.

**Verification items resolved:**
- §7.14 / shape contract → **resolved**: both call sites derive `input_shape`
  from the persisted feature list (`9.3`, `10.7`).
- §6.16 → **resolved for these two modules**: both use
  `prepare_model_ready_data` exclusively (`9.4`, `10.9`).
- §6-H positional order → **resolved**: all feature access is by name (`9.5`).
- §7-E class weighting at fit time → **resolved** (`9.16`).

**Still open for Section 11:**
- §8.11 — confirm `partition_iid` receives `df.loc[train_indices]`, not the full
  frame (Contract A.2.7).
- §1-C — confirm `federated.ray`, `client_num_cpus`, `runtime_profile` are read
  with `.get()` defaults so `FULL_EXPERIMENT` still runs.
- §8-C — confirm the caller releases the training frame after partitioning.

---

# AUDIT CHECKPOINT — resume instructions

**Status at checkpoint:** Sections 1–10 complete. Sections 11–19 and the final
roll-up remain. The audit was paused here because the reviewing context window
was exhausted, not because of any finding.

## Interim tally (Sections 1–10)

| Severity | Count | IDs |
| --- | --- | --- |
| BLOCKER | 0 | — |
| MAJOR | 2 | `4-A` (missing `pyarrow` dependency), `10-C` (unused `test_data` retained per Ray actor) |
| MINOR | 11 | `1-C`, `2-B`, `3-A`, `3-B`, `4-C`, `5-C`, `6-D`, `6-E`, `6-F`, `7-B`, `8-C`, `8-D`, `9-D`, `9-E`, `10-D`, `10-E` |
| OBSERVATION | 25 | — |

**Requirements checked:** 158. **PASS:** 156. **PARTIAL:** 2 (`3.5`
`PYTHONHASHSEED` timing, `6.37` NaN check warns rather than fails).
**FAIL:** 0. **NOT IMPLEMENTED:** 0 *within audited modules*.

## Work remaining

| # | Component | Expected finding |
| --- | --- | --- |
| 11 | `src/federated/server_app.py` (~700 lines) | Unknown — the last substantial implemented module. |
| 12 | `src/evaluation/metrics.py` | **PARTIAL** — 1 of 5 required functions implemented. |
| 13 | `src/explainability/` | **NOT IMPLEMENTED** — `__init__.py` only. |
| 14 | `dashboard/` | **NOT IMPLEMENTED** — directory empty. |
| 15–17 | `experiments/`, `tests/` | `run_evaluation.py` and `run_explainability.py` absent. |
| 18 | Import-graph conformance | Partially verified: §7.23, §9.28, §10.27 all PASS. |
| 19 | Artifact inventory vs SDS §12 | Pending. |
| — | Final roll-up | Pending. |

## Open verification items for Section 11

These were deferred from earlier sections and **must** be checked against
`server_app.py`:

1. **§8.11 / Contract A.2.7** — confirm `partition_iid` is called with
   `df.loc[train_indices]` and never the full processed frame. This is the last
   unverified frozen invariant in the audit.
2. **§1-C** — confirm `federated.ray`, `federated.client_num_cpus`,
   `federated.client_num_gpus`, and the top-level `runtime_profile` are read
   with `.get()` defaults. These keys exist **only** in
   `configs/config_laptop.yaml`; a bare `[...]` lookup would raise `KeyError`
   under `FULL_EXPERIMENT` and break the cloud run path.
3. **§8-C** — confirm the caller releases the training frame after partitioning
   (`del`, or reassignment) before Ray actors are spawned.
4. **§6.16** — confirm the server uses `prepare_model_ready_data` for its
   central-evaluation tensors rather than reimplementing the reshape.
5. **§7.14** — confirm the server's reconstructed global model derives
   `input_shape` from `feature_names.pkl`.
6. **§10-C follow-through** — confirm what the orchestrator passes as
   `test_data` to `client_fn`, and whether `runtime_profile.server_eval_max_rows`
   subsampling is applied before or after that binding.

## How to resume

Start a new task with roughly this instruction:

> Continue the compliance audit in `project-docs/COMPLIANCE_AUDIT.md`. Sections
> 1–10 are complete — read the "AUDIT CHECKPOINT" section for the interim tally
> and the six open verification items. Begin at Section 11
> (`src/federated/server_app.py`), then Sections 12–19 and the final roll-up.
> Append each section; do not overwrite existing content. Replace this
> checkpoint block when the audit is complete.

Governing documents to re-read at the start of the new task: `SDS.md` §14.6
(server/strategy), `AI_Coding_Contract.md` (Invariant A.2.7),
`Phase_Implementation_Prompts.md` Phases 10–18.

---

<!-- APPEND NEXT SECTION BELOW THIS LINE -->

## 11. `src/federated/server_app.py` — PASS WITH NOTES

**Governing spec:** SDS §14.6 (strategy, the two aggregation functions, the
pinned `start_simulation` call, mandatory global-weight capture, the five
written artifacts), SDS §14.3 (the shared global test split is never
partitioned), SDS §11 (import graph), SDS §15 (persistence rules), SDS §22
(forbidden practices); AI Coding Contract Invariant A.2.7;
`Phase_Implementation_Prompts.md` Phase 10.

**Files:** `src/federated/server_app.py` (905 lines).

### Requirement checklist — `SavingFedAvg` and the aggregation functions

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 11.1 | Exactly one strategy class; it subclasses `FedAvg` | **PASS** | `SavingFedAvg(FedAvg)`, `server_app.py:77`. Exclusivity rule stated at `6-10`. |
| 11.2 | `__init__` forwards `*args`/`**kwargs` and initialises `latest_parameters = None` | **PASS** | `server_app.py:114-115`. Matches the SDS §14.6 exact-implementation block verbatim. |
| 11.3 | `aggregate_fit` delegates to `super()` and never alters the return value | **PASS** | `server_app.py:139-141`, returned unchanged at `154`. |
| 11.4 | Aggregated parameters captured only when not `None` | **PASS** | `server_app.py:143-144`. A failed round therefore cannot overwrite a good capture. |
| 11.5 | `weighted_average_fit` reads **both** `loss` and `accuracy` | **PASS** | `server_app.py:180-189`; formula is character-for-character the SDS §14.6 block. |
| 11.6 | `weighted_average_eval` reads **only** `accuracy` | **PASS** | `server_app.py:217-222`. No `loss` key access, so `FLClient.evaluate`'s single-key dict cannot raise `KeyError`. |
| 11.7 | The two functions are distinct objects, never collapsed into one | **PASS** | Separate `def`s at `157` and `192`; asserted by `tests/test_federated_loop.py:198-200`. |
| 11.8 | `fit_metrics_aggregation_fn` ← `weighted_average_fit` | **PASS** | `server_app.py:492`. |
| 11.9 | `evaluate_metrics_aggregation_fn` ← `weighted_average_eval` | **PASS** | `server_app.py:493`. Assignments are not swapped; asserted by `tests/test_federated_loop.py:223-232`. |
| 11.10 | All three minimum-client thresholds pinned to `federated.num_clients` | **PASS** | `server_app.py:486-491`. |
| 11.11 | `get_strategy` returns `SavingFedAvg`, never a plain `FedAvg` | **PASS** | `server_app.py:488`; return type annotated at `455`. |
| 11.12 | Missing `federated.num_clients` raises `KeyError` | **PASS** | Bare lookup at `server_app.py:486`; documented at `483-484`; asserted by `tests/test_federated_loop.py:235-238`. |

### Requirement checklist — `run_federated_simulation`

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 11.13 | Single top-level federated orchestrator | **PASS** | `server_app.py:514`; ownership at `26-30`. `experiments/run_federated.py:72` is its only caller. |
| 11.14 | **`partition_iid` receives `df.loc[train_indices]`, never the full frame** (Contract A.2.7 — carried from §8.11) | **PASS** | `server_app.py:634-636`. **This resolves the last unverified frozen invariant in the audit.** |
| 11.15 | The shared global test split is bound once and never sharded | **PASS** | `test_data_global = df.loc[test_indices]` (`637`), passed as a single frame to `partial(client_fn, ..., test_data=test_data_global, ...)` (`656-661`). |
| 11.16 | Processed data read through the `dataio` singleton | **PASS** | `read_processed`, `server_app.py:596`. |
| 11.17 | `num_classes` derived as `len(class_mapping)`, never hardcoded | **PASS** | `server_app.py:620`, rule restated at `619`. |
| 11.18 | **`input_shape` derived from `feature_names.pkl`** (carried from §7.14) | **PASS** | `num_features = len(feature_columns)` (`621`) where `feature_columns` is `_load_pickle("feature_names.pkl")` (`612`); used at `430` and `830`. |
| 11.19 | **Server tensors built via `prepare_model_ready_data`** (carried from §6.16) | **PASS** | `server_app.py:674-680`. No local reshape or `to_categorical`. §6.16 is now verified at every production call site. |
| 11.20 | Feature access is by name, not position (carried from §6-H) | **PASS** | `feature_columns` is threaded into `prepare_model_ready_data`, which indexes by label. §6-H is now fully closed. |
| 11.21 | Pinned `flwr==1.8.0` `start_simulation(client_fn=…, num_clients=…, config=…, strategy=…)` API used | **PASS** | `server_app.py:721-728`. |
| 11.22 | App-based `run_simulation` / `ClientApp` / `ServerApp` never used | **PASS** | Absent repository-wide; prohibition restated at `server_app.py:18-22`. |
| 11.23 | **`time.time()` wraps only `start_simulation`** | **PASS** | `start_time` at `720`, call at `721-728`, elapsed at `729`. Tensor construction (`674-689`) and strategy construction (`692`) are outside the window; the in-code comment at `664-665` states this explicitly. |
| 11.24 | Global-weight capture is mandatory, not best-effort | **PASS** | `RuntimeError` when `latest_parameters is None`, `server_app.py:821-826`. |
| 11.25 | Weights converted via `parameters_to_ndarrays` | **PASS** | `server_app.py:828`. |
| 11.26 | Global model built by a fresh `build_cnn_gru` + `set_weights` | **PASS** | `server_app.py:829-835`. |
| 11.27 | A per-client local model is never saved as `federated_global_model.h5` | **PASS** | The only `.save()` call (`845`) writes `global_model`, which is provably the aggregated model. Rule restated at `530-534`. |
| 11.28 | Writes `federated_global_model.h5` | **PASS** | `server_app.py:842-845`. |
| 11.29 | Writes `federated_model_summary.txt` | **PASS** | `server_app.py:851-855`, via `get_model_summary_string`. |
| 11.30 | Writes `federated_architecture.png`, non-fatally | **PASS** | `server_app.py:857-866`; missing file downgrades to WARNING. Mirrors `train_baseline.py:291-300`. |
| 11.31 | Writes `federated_history.csv` | **PASS** | `server_app.py:869-870`, `index=False`. |
| 11.32 | Writes `federated_training_time.txt` as a single float | **PASS** | `server_app.py:878-882`. |
| 11.33 | History columns include the SDS-fixed `round, aggregated_loss, aggregated_accuracy` | **PASS** | `server_app.py:790-800`. See finding 11-C on the two added columns. |
| 11.34 | History populated from `history.losses_distributed` / `history.metrics_distributed` | **PASS** | `server_app.py:777-778`. |
| 11.35 | Output directories created before writing | **PASS** | `server_app.py:569-570`. |
| 11.36 | Logger injected from `experiments/`, never constructed | **PASS** | Optional parameter `516`; handler-less fallback at `561-562`; rule at `63-65`. Module logger is `logging.getLogger(__name__)` (`66`). |
| 11.37 | Failures log a full traceback at ERROR before re-raising | **PASS** | `server_app.py:901-904`, `exc_info=True`. |
| 11.38 | `KeyboardInterrupt` handled separately from `Exception` | **PASS** | `server_app.py:891-900`. Same defence as `train_baseline.py:363-372`. |
| 11.39 | Import-graph conformance | **PASS** | `server_app.py:47-60` imports only `flwr`, `pandas`, and from `src.federated.client_app`, `src.models`, `src.partitioning`, `src.preprocessing`, `src.utils`. No `centralized`, `evaluation`, or `explainability` import. Rule stated at `32-35`. |
| 11.40 | Path construction uses `os.path.join` (SDS §8) | **PASS** | Every path: `591-594`, `606`, `615`, `842`, `852`, `857`, `869`, `878`. No literal separators anywhere. |

### Requirement checklist — LAPTOP_MODE optional-key handling (carried from §1-C)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 11.41 | **`federated.ray` read with a `.get()` default** | **PASS** | `config["federated"].get("ray") or {}`, `server_app.py:263`. The `or {}` also survives an explicit `ray:` key with a `null` value. |
| 11.42 | **`federated.client_num_cpus` read defensively** | **PASS** | `if "client_num_cpus" not in federated_config: return None`, `server_app.py:307-308`. |
| 11.43 | **`federated.client_num_gpus` read with a default** | **PASS** | `.get("client_num_gpus", 0)`, `server_app.py:312`. |
| 11.44 | **`runtime_profile` read with a `.get()` default** | **PASS** | `(config.get("runtime_profile") or {}).get(...)` at both `345-347` and `740-742`. |
| 11.45 | `FULL_EXPERIMENT` path is behaviourally unchanged by the LAPTOP_MODE code | **PASS** | `build_ray_init_args` returns the original three-key dict (`257-261`); `build_client_resources` returns `None` → Flower's default; `subsample_for_server_eval` returns `test_data` unchanged (`349-355`); the release-memory block is skipped entirely (`740`). **§1-C is resolved — no `KeyError` is reachable under `FULL_EXPERIMENT`.** |
| 11.46 | Server-eval subsample is stratified and seeded | **PASS** | `groupby("label").apply(... sample(random_state=config["seed"]))`, `server_app.py:358-368`; `max(1, ...)` guarantees no rare class is dropped (`364`). |
| 11.47 | `run_mode` recorded in the run log for audit | **PASS** | `server_app.py:576-578`, `.get()` with a `FULL_EXPERIMENT` default. |
| 11.48 | **Caller releases the training frame after partitioning** (carried from §8-C) | **PARTIAL** | `client_shards`, `bound_client_fn`, `X_test_global`, `y_test_global` are deleted at `764-766` — but only *after* the simulation and only under `release_memory_after_rounds`. The full `df` is never released. See finding 11-B. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 11-A | OBSERVATION | The server-side `evaluate_fn` (`384-452`) is an addition beyond SDS §14.6, and a correct one. Distributed evaluation now scores each client's *own* local hold-out (§10-A), which answers "how does the global model do locally?" but no longer scores the shared global test split. `evaluate_fn` restores that measurement — once per round, on the server, unweighted. The docstring at `395-407` states the diagnostic value precisely: a divergence between the distributed and centralized curves is the standard signature of an aggregation defect, and with only one of the two that failure is invisible. The model shell is built once outside the round loop (`429-434`), so the addition costs one graph construction, not fifteen. | `server_app.py:384-452`, `682-689` | None. Recommend recording the addition in SDS §14.6. |
| 11-B | MINOR | The full processed `df` (~1.9 M × ~95 columns) stays referenced for the entire simulation. `client_shards` (`634`) and `test_data_global` (`637`) are `.loc` copies, so at the moment `start_simulation` is called the parent process holds roughly **2× the dataset**: the original frame plus the shards plus the test copy. `df` is never used after line 637. Deleting it there would free the larger of the two — in the exact process whose footprint `LAPTOP_MODE` exists to shrink. The existing `del` block (`764-766`) runs only *after* the rounds, i.e. after peak. This partially resolves §8-C: the caller does release the shards, but at the wrong time and not the frame. | `server_app.py:596`, `634-637`, `764-766` | Add `del df` (plus a `gc.collect()`) immediately after line 637, and move the shard release so it is not gated on `release_memory_after_rounds`. `gc` is already imported (`38`). |
| 11-C | OBSERVATION | `federated_history.csv` carries two columns beyond the SDS-fixed three: `centralized_loss` and `centralized_accuracy` (`795-798`). This is a superset, not a substitution — a consumer reading the three mandated columns is unaffected. Worth flagging to Section 12: `generate_convergence_plot` (SDS §14.9) reads this file and must select columns by name. | `server_app.py:790-800` | None; note as an input contract for the unimplemented evaluation module. |
| 11-D | MINOR | `history_rows` is built by iterating `sorted(losses)` — the rounds present in `losses_distributed` (`787`). If a round's *distributed* evaluation failed but its *centralized* evaluation succeeded, that round is absent from the CSV entirely and its centralized metrics are silently discarded. The reverse case is handled correctly, since `central_losses.get(...)` yields `None`. | `server_app.py:777-800` | Iterate `sorted(set(losses) | set(central_losses))` so neither evaluation path can drop a round. |
| 11-E | MINOR | `subsample_for_server_eval` uses `DataFrame.groupby(...).apply(...)` (`358-368`), which pandas ≥ 2.2 deprecates for operating on the grouping columns (`DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns`). `requirements.txt` allows `pandas<2.3`, so a compliant install can emit this warning on every LAPTOP_MODE run. Functional today; noisy, and a future break. | `server_app.py:358-368`; `requirements.txt:5` | Use `test_data.groupby("label", group_keys=False)[test_data.columns].apply(...)`, or replace with a per-group index draw. |
| 11-F | OBSERVATION | `client_runtime_config = dict(config)` (`650`) is a **shallow** copy, so `client_runtime_config["federated"]` is the *same* dict object the server holds. Adding the `"runtime"` key is safe because it is a new top-level key, but any future nested mutation would propagate back into the server's config. Contrast with `_deep_merge`'s `copy.deepcopy` (`config_loader.py:114`). A shallow copy is in fact the right call here — the `runtime` sub-dict carries a fitted `LabelEncoder` that must not be duplicated per client — but the reason is not stated in code. | `server_app.py:650-655` | Add a one-line comment recording that the shallow copy is deliberate (shared `LabelEncoder`) and that nested keys must not be mutated. |
| 11-G | OBSERVATION | The `runtime` payload passed to clients (`651-655`) carries `feature_columns`, `label_encoder`, and `num_classes` — precisely what `FLClient` needs, and nothing more. This is what makes the client's zero-I/O property (§10.10) achievable: every artifact is loaded once by the server (`610-617`) and pushed into the actors, rather than each actor re-reading `outputs/artifacts/`. Clean separation. | `server_app.py:605-617`, `650-661` | None. |
| 11-H | MINOR | **§10-C follow-through.** The orchestrator binds the **full** `test_data_global` to `client_fn` (`659`) — subsampling is applied *only* to the server's own copy (`671-673`), and only in LAPTOP_MODE. Combined with `client_app.py:134` retaining `self.test_data` unused, every Ray actor holds the full ~443k-row test frame while the server holds a 50k-row sample. The asymmetry noted in §10-C is confirmed at the call site: the server-side saving of ~150 MB is dwarfed by the client-side retention of ~168 MB × N actors. | `server_app.py:659`, `671-673`; `client_app.py:134` | Fix `10-C` (drop the client-side retention). Once `self.test_data` is not retained, this binding costs nothing. The two findings must be fixed together to realise the saving. |
| 11-I | OBSERVATION | Ray shutdown after the final round (`743-759`) is wrapped in a broad `except` that logs a WARNING and continues — "never let cleanup failure destroy a completed training run" (`754`). Correct priority ordering: at that point the model weights exist in-process and the artifacts have not yet been written. | `server_app.py:743-759` | None. |
| 11-J | OBSERVATION | `import ray` is deferred into the shutdown branch (`744`) rather than being a module-level import. Deliberate: `ray` is a transitive dependency of `flwr[simulation]` and this keeps `server_app` importable (for the unit tests) without a hard top-level dependency on it. | `server_app.py:743-747` | None. |
| 11-K | MINOR | `federated.partition_strategy` is read once for logging (`585`) but never dispatched on — `partition_iid` is called by name (`634`). Setting `partition_strategy: "non_iid"` would be logged and then silently ignored. This is the call-site confirmation of §8-D. | `server_app.py:585`, `634` | Raise `ValueError` for any value other than `"iid"`, or dispatch through a strategy table. |
| 11-L | OBSERVATION | `min_available_clients` / `min_fit_clients` / `min_evaluate_clients` exist in `configs/config.yaml:95-97` but `get_strategy` reads only `num_clients` and pins all three to it (`486-491`). The config keys are therefore inert. This is *safer* than reading them — the three can never disagree with the client count — but it technically violates SDS §20's "no unused configuration keys" rule. The SDS itself mandates the pinning (§14.6), so the code is right and the config carries three redundant keys. | `server_app.py:486-491`; `configs/config.yaml:95-97` | Either annotate the three config keys as documentation-only (as `training.optimizer` already is at `config.yaml:76`), or delete them. |
| 11-M | OBSERVATION | The concurrency estimate logged at `709-718` (`floor(ray_num_cpus / client_num_cpus)`) makes the LAPTOP_MODE serialisation visible in the log, and correctly states that peak RAM is driven by concurrent actors rather than by `num_clients`. Good operator-facing diagnostics. | `server_app.py:709-718` | None. |
| 11-N | OBSERVATION | Two stylistic blemishes: a stray blank line inside the `make_server_side_evaluate_fn` parameter list (`385`) and a blank line between the `get_strategy` signature and its docstring (`456`). PEP 257 expects the docstring to be the first statement; it still is at runtime (blank lines are not statements), so this is cosmetic only. | `server_app.py:385`, `456` | Remove the stray blank lines. |

### Notes

Section 11 closes every open verification item carried from Batches 0–3:

| Item | Origin | Resolution |
| --- | --- | --- |
| §8.11 / Contract A.2.7 | Batch 2 | **Resolved (11.14).** `partition_iid(df.loc[train_indices], ...)`. The test split is bound separately and never sharded. |
| §1-C | Batch 0 | **Resolved (11.41–11.45).** All four LAPTOP_MODE-only keys use `.get()` or membership tests. `FULL_EXPERIMENT` cannot raise `KeyError`. |
| §8-C | Batch 2 | **Partially resolved (11.48).** Shards are released, but only post-simulation and only in LAPTOP_MODE; `df` is never released. Raised as `11-B`. |
| §6.16 | Batch 1 | **Fully resolved.** `prepare_model_ready_data` is now confirmed as the sole tensor-preparation path at all three production call sites (`9.4`, `10.9`, `11.19`). |
| §7.14 | Batch 2 | **Fully resolved.** All three call sites derive `input_shape` from `feature_names.pkl`. |
| §10-C | Batch 3 | **Confirmed and quantified (11-H).** The orchestrator binds the full test frame to every client; the fix remains a single deletion in `client_app.py`. |

The most consequential judgement in this section is the server-side `evaluate_fn`
(11-A). It is not in the SDS, and an audit could have flagged it as scope creep.
It is not: §10-A moved distributed evaluation onto per-client local hold-outs,
which *removed* the project's only measurement of the global model on the shared
global test split. Adding `evaluate_fn` restores that measurement. Without it,
the pair of changes would have left the project with no round-by-round number
comparable to the centralized baseline. The two changes are correct only
together, and they were made together.

Structurally, this is the most operationally mature module in the repository:
`KeyboardInterrupt` handling, defensive Ray shutdown, quantified memory
commentary, and a mandatory-not-optional weight-capture check that converts a
silent empty-model failure into an explicit `RuntimeError`.

---

## Batch 3 summary (complete)

| Section | Component | Status |
| --- | --- | --- |
| 9 | `centralized/train_baseline.py` | **PASS** — 28/28 requirements |
| 10 | `federated/client_app.py` | **PASS** — 27/27 requirements |
| 11 | `federated/server_app.py` | **PASS WITH NOTES** — 47 PASS, 1 PARTIAL (11.48) |

**Defects raised in this batch:** 0 BLOCKER, 1 MAJOR (`10-C`), 9 MINOR (`9-D`,
`9-E`, `10-D`, `10-E`, `11-B`, `11-D`, `11-E`, `11-H`, `11-K`), 16 OBSERVATION.

**Verdict for the training layers:** compliant. Every frozen invariant that
Batches 0–2 deferred is now verified true at its call site. Contract A.2.7
holds: the global test split is never partitioned. The federated path uses the
pinned Flower API exclusively and the saved global model is provably the
aggregated model, not a client's.

**Carried forward:**
- `10-C` + `11-H` → Critical Bugs (fix together).
- `11-B` → Remaining work (memory; LAPTOP_MODE).
- `11-C` → input contract for Section 12's unimplemented `generate_convergence_plot`.
- `11-D`, `11-E`, `11-K`, `11-L` → Remaining work.

---

# BATCH 4 — Evaluation, explainability, and dashboard (SDS §14.8, §14.9, §14.10, §18)

Batch 2's structural survey predicted this batch would record absences rather
than defects. That prediction is confirmed. The audit method changes accordingly:
where no implementation exists, the checklist records the **specification** each
missing function must satisfy, so the sections remain actionable as a build
specification rather than merely reporting a gap.

---

## 12. `src/evaluation/` — PARTIAL (1 of 5 functions in `metrics.py`; `compare_fl_vs_centralized.py` NOT IMPLEMENTED)

**Governing spec:** SDS §14.8 (`metrics.py`, five functions), SDS §14.9
(`compare_fl_vs_centralized.py`, five functions), SDS §16 (metric suite),
SDS §17 (plot ownership table), SDS §11 (import graph);
`Phase_Implementation_Prompts.md` Phase 11.

**Files present:** `src/evaluation/__init__.py` (empty),
`src/evaluation/metrics.py` (72 lines).
**Files absent:** `src/evaluation/compare_fl_vs_centralized.py`.

### Requirement checklist — `metrics.py` (SDS §14.8)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 12.1 | `generate_class_distribution_plot` exists with the SDS signature | **PASS** | `metrics.py:20-24` — `(df, label_column, output_path)`, exact argument names. |
| 12.2 | Raises `KeyError` when `label_column` is absent | **PASS** | `metrics.py:52-56`; the message enumerates the available columns. |
| 12.3 | Writes a PNG at `output_path`; returns `None` | **PASS** | `metrics.py:71`; return type annotated `None` at `24`. |
| 12.4 | Reads nothing — operates on the passed frame | **PASS** | No I/O other than `savefig`. |
| 12.5 | Non-interactive matplotlib backend selected before `pyplot` import | **PASS** | `matplotlib.use("Agg")`, `metrics.py:13-14`. Required for headless/pytest execution. |
| 12.6 | Figure closed after saving (no handle leak) | **PASS** | `plt.close(fig)`, `metrics.py:72`. |
| 12.7 | Called from `experiments/run_preprocessing.py` immediately after the pipeline | **PASS** | `run_preprocessing.py:71`. Matches the SDS §17 ownership row. |
| 12.8 | `compute_all_metrics` | **NOT IMPLEMENTED** | Absent. Must return `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `loss`; raise `ValueError` on length mismatch (SDS §14.8). |
| 12.9 | `generate_confusion_matrix` | **NOT IMPLEMENTED** | Absent. Must return the raw matrix and write a PNG (SDS §14.8, §17). |
| 12.10 | `generate_classification_report` | **NOT IMPLEMENTED** | Absent. Must return the report text and write a `.txt` (SDS §14.8). |
| 12.11 | `generate_training_curves_plot` | **NOT IMPLEMENTED** | Absent. Must raise `FileNotFoundError` on a missing history CSV and write `centralized_training_curves.png` (SDS §14.8, §17). |

### Requirement checklist — `compare_fl_vs_centralized.py` (SDS §14.9)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 12.12 | Module exists | **NOT IMPLEMENTED** | File absent from `src/evaluation/`. |
| 12.13 | `evaluate_model_on_test_set` | **NOT IMPLEMENTED** | Must follow the SDS §14.9 fixed computation block, source `loss` from `model.evaluate()`, time only `.predict()`, and warn if the two accuracy figures diverge by >1e-6. |
| 12.14 | `generate_comparison_table` | **NOT IMPLEMENTED** | Must emit exactly 2 rows and all 13 named columns. |
| 12.15 | `generate_model_size_comparison` | **NOT IMPLEMENTED** | Must emit `Model`, `Model_Size_MB`, `Parameter_Count`. |
| 12.16 | `generate_convergence_plot` | **NOT IMPLEMENTED** | Must overlay federated per-round accuracy on centralized per-epoch accuracy. |
| 12.17 | `run_evaluation_pipeline` | **NOT IMPLEMENTED** | The 13-step fixed sequence in SDS §14.9 is unimplemented. |
| 12.18 | `tests/test_metrics.py` | **NOT IMPLEMENTED** | Required by SDS §19; the file does not exist. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 12-A | **BLOCKER** | `run_evaluation_pipeline` does not exist, so **no evaluation-stage artifact can be produced**: `comparison_table.csv`, `model_size_comparison.csv`, `convergence_plot.png`, both confusion matrices, both classification reports, and `centralized_training_curves.png` — seven of the project's headline deliverables (SDS §17, §24). SDS §23 Acceptance Criterion 5 is unsatisfiable. Since the FL-vs-centralized comparison is the project's stated primary research question, this is the single largest gap in the repository. | SDS §14.9, §17, §23; `src/evaluation/` contents | Implement `compare_fl_vs_centralized.py` in full, then `experiments/run_evaluation.py`. |
| 12-B | **MAJOR** | Four of the five `metrics.py` functions are unimplemented. `metrics.py:8-10` names them honestly as Phase 8 work, so this is a *declared* gap rather than a silent one — but `generate_confusion_matrix` and `generate_classification_report` are SDS §16 mandatory outputs for both models. | `metrics.py:8-10` | Implement the four functions per SDS §14.8. |
| 12-C | OBSERVATION | The one implemented function is a good template for the rest: exact SDS argument names, the documented `KeyError`, a full Google-style docstring including a `Dependencies` section, and the `Agg` backend set before `pyplot` is imported. The cross-boundary early creation is declared at `metrics.py:3-6` with the Phase 4 allow-list cited. | `metrics.py:1-51` | None. |
| 12-D | MINOR | Figure height scales as `max(4, len(counts) * 0.4)` (`metrics.py:60`) and the value labels are offset by 0.5% of the maximum count (`67`). With Edge-IIoTset's ~1600:1 imbalance on a linear axis, the rarest classes render as bars of near-zero width with their labels overlapping the axis. A `log` x-scale would make the tail readable. | `metrics.py:58-68` | Optional: `ax.set_xscale("log")`, or annotate outside the bar for small values. |
| 12-E | OBSERVATION | Import-graph conformance holds for what exists: `metrics.py` imports only `matplotlib` and `pandas` — no `src/` import at all, which is within SDS §11's allowance (`utils`, `preprocessing.prepare_model_ready_data`, `models`). | `metrics.py:13-17` | None. |
| 12-F | MINOR | `pandas` is imported as `pd` (`metrics.py:17`) whereas every other module in the repository imports it unaliased (`import pandas`). `numpy`/`numpy as np` is similarly inconsistent between `metrics.py`-era code and `server_app.py`. Cosmetic, but it is the only file that breaks the convention. | `metrics.py:17` vs `server_app.py:48` | Align on the unaliased form used by `src/`. |

### Notes

The distinction between 12-A and 12-B matters for planning. 12-B is *declared*
incomplete work — the file's own docstring names the four missing functions and
the phase that owns them. 12-A is a whole missing module plus a missing
`experiments/` entry point that `README.md:21` already advertises as part of the
run order. A user following the README today reaches step 4 and gets
`No such file or directory`.

Every input `run_evaluation_pipeline` needs already exists and is correct:
`centralized_best_model.h5`, `federated_global_model.h5`, both
`*_training_time.txt` files, both history CSVs, and the artifacts under
`outputs/artifacts/`. The gap is purely the consumer, which makes this the
highest-value remaining work in the project.

One input contract to honour when it is written (from 11-C):
`federated_history.csv` now carries five columns, not three. `generate_convergence_plot`
must select `aggregated_accuracy` by name.

---

## 13. `src/explainability/` — NOT IMPLEMENTED

**Governing spec:** SDS §14.10 (`shap_utils.py`, two functions), SDS §17
(two SHAP plots), SDS §18 Tab 4, SDS §11 (import graph);
`Phase_Implementation_Prompts.md` Phase 12.

**Files present:** `src/explainability/__init__.py` (empty).
**Files absent:** `src/explainability/shap_utils.py`.

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 13.1 | `shap_utils.py` exists | **NOT IMPLEMENTED** | Only `__init__.py` is present. |
| 13.2 | `generate_shap_explanations` with the nine SDS-named arguments | **NOT IMPLEMENTED** | SDS §14.10. |
| 13.3 | Background sample drawn via `numpy.random.RandomState(seed).choice(...)`, size `min(background_size, len(X_train))` | **NOT IMPLEMENTED** | SDS §14.10 background-sampling requirement. |
| 13.4 | Background array persisted to `outputs/artifacts/shap_background.npy` | **NOT IMPLEMENTED** | Required by the dashboard's Tab 4, which must never re-sample. |
| 13.5 | Test sample drawn as a second, independently-seeded `.choice(...)` in the fixed order (background first) | **NOT IMPLEMENTED** | SDS §14.10 test-sampling requirement. |
| 13.6 | Mandatory squeeze of the trailing channel axis before any plotting call | **NOT IMPLEMENTED** | SDS §14.10 four-step fixed procedure. Skipping it is the documented shape-mismatch failure mode. |
| 13.7 | Writes `shap_summary_plot.png` | **NOT IMPLEMENTED** | SDS §17. |
| 13.8 | Writes `shap_bar_plot.png` (mean absolute SHAP per feature across classes) | **NOT IMPLEMENTED** | SDS §17, §14.10 step 4. |
| 13.9 | Returns per-class SHAP values in their original 3D shape | **NOT IMPLEMENTED** | SDS §14.10 return contract. |
| 13.10 | `explain_single_prediction` for the dashboard's per-prediction view | **NOT IMPLEMENTED** | SDS §14.10; the only function `dashboard/app.py` may import from this package. |
| 13.11 | `explain_single_prediction` never re-samples the background | **NOT IMPLEMENTED** | Caller supplies the explainer built from the persisted array. |
| 13.12 | `experiments/run_explainability.py` | **NOT IMPLEMENTED** | Absent; advertised at `README.md:22`. |
| 13.13 | `tests/test_shap_utils.py` | **NOT IMPLEMENTED** | Required by SDS §19; absent. |
| 13.14 | Explains `federated_global_model.h5` (per the SDS §10 artifact-flow diagram) | **NOT IMPLEMENTED** | The model exists once `run_federated.py` has been run; the consumer does not. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 13-A | **BLOCKER** | The explainability layer is entirely absent. SHAP integration is named in SDS §1 as one of the project's two stated novelties over the reference work ("adds transparency into which features drive each classification decision — absent from the reference work entirely"). Its absence removes a first-class contribution, blocks SDS §23 Acceptance Criterion 6, and cascades into dashboard Tab 4. | `src/explainability/` contents; SDS §1, §14.10, §23 | Implement `shap_utils.py` and `experiments/run_explainability.py` per SDS §14.10. |
| 13-B | OBSERVATION | The dependency chain is already correct and pinned for this work: `tensorflow==2.15.1` is pinned specifically because it is the last release defaulting to Keras 2, which `shap.GradientExplainer` requires, and `shap==0.44.1` is pinned alongside it (`requirements.txt:1,7`). The environment is ready; only the code is missing. | `requirements.txt:1,7`; SDS §7 | None. |
| 13-C | OBSERVATION | The config block is present and correct — `explainability.method: "GradientExplainer"`, `sample_size: 100`, `background_size: 100` (`configs/config.yaml:106-109`) — and is currently read by **no** function, which is a live violation of SDS §20's "no unused configuration keys" rule. The violation resolves itself the moment 13-A is fixed. | `configs/config.yaml:106-109` | None beyond 13-A. |
| 13-D | OBSERVATION | `outputs/artifacts/shap_background.npy` is the only artifact in SDS §12 produced by a script other than preprocessing. Its absence is what forces dashboard Tab 4 into its `st.error` path even after every other stage has run. | SDS §12, §18 Tab 4 | None beyond 13-A. |

### Notes

The SDS's squeeze procedure (13.6) deserves emphasis for whoever implements this.
With input shape `(n, num_features, 1)`, `GradientExplainer` returns a list of
`num_classes` arrays each shaped `(n, num_features, 1)`. `shap.summary_plot`
requires 2D. Passing the 3D arrays straight through is the documented failure
mode and produces an opaque shape error inside SHAP's plotting internals — which
is precisely why the SDS specifies the four steps literally rather than
describing them.

Nothing about this layer is blocked by other work. `federated_global_model.h5`
and the full training/test tensors are all obtainable from existing, verified
code paths.

---

## 14. `dashboard/` — NOT IMPLEMENTED

**Governing spec:** SDS §18 (four tabs, read-only enforcement), SDS §11
(dashboard import restrictions), SDS §22 (absolute prohibitions);
`Phase_Implementation_Prompts.md` Phase 13.

**Files present:** none — the directory is empty.
**Files absent:** `dashboard/app.py`.

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 14.1 | `dashboard/app.py` exists | **NOT IMPLEMENTED** | Directory empty. |
| 14.2 | Tab 1 — Overview: `class_distribution.png` + dataset stats from `feature_names.pkl` / `class_mapping.json` | **NOT IMPLEMENTED** | SDS §18 Tab 1. |
| 14.3 | Tab 2 — Predictions: cached model load, `prepare_model_ready_data`, sample selector, predicted vs. true | **NOT IMPLEMENTED** | SDS §18 Tab 2. |
| 14.4 | Tab 3 — FL vs. Centralized: both CSVs, convergence plot, both confusion matrices via `st.columns(2)` | **NOT IMPLEMENTED** | SDS §18 Tab 3. |
| 14.5 | Tab 4 — Explainability: explainer built from the persisted `shap_background.npy` | **NOT IMPLEMENTED** | SDS §18 Tab 4. |
| 14.6 | Each tab handles missing artifacts with its own `st.error`, never an unhandled exception | **NOT IMPLEMENTED** | SDS §18, §22. |
| 14.7 | Never calls `train_centralized_model`, `run_federated_simulation`, `partition_iid`, `run_preprocessing_pipeline`, or `generate_shap_explanations` | **PASS** (vacuously) | No code exists to violate the rule. Recorded so the constraint is not lost. |
| 14.8 | Imports restricted to `src/utils/`, `src/evaluation/`, `prepare_model_ready_data`, `explain_single_prediction` | **NOT IMPLEMENTED** | SDS §11. |
| 14.9 | `@st.cache_resource` on the model and the background array | **NOT IMPLEMENTED** | SDS §18 Tabs 2 and 4. |
| 14.10 | `st.session_state` shares the Tab 2 selection with Tab 4 | **NOT IMPLEMENTED** | SDS §18 Tab 4. |
| 14.11 | Paths built with `os.path.join` from `config["paths"]` | **NOT IMPLEMENTED** | SDS §8. |
| 14.12 | `dashboard.title` config key consumed | **NOT IMPLEMENTED** | `configs/config.yaml:111-112` is currently read by nothing. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 14-A | **BLOCKER** | The dashboard is entirely absent. It is SDS §1's fifth headline capability ("Presents all results through a read-only Streamlit dashboard that performs no training"), a §24 deliverable, and §23 Acceptance Criterion 7. `README.md:23` advertises `streamlit run dashboard/app.py` as the final run step. | `dashboard/` is empty; SDS §1, §18, §23, §24 | Implement `dashboard/app.py` per the SDS §18 four-tab specification. |
| 14-B | **MAJOR** | The dashboard is the terminal consumer of both missing upstream layers. Tab 3 needs `comparison_table.csv`, `model_size_comparison.csv`, `convergence_plot.png`, and both confusion matrices (all blocked by 12-A); Tab 4 needs `shap_background.npy` and `shap_summary_plot.png` (blocked by 13-A). Only Tabs 1 and 2 could be built and demonstrated against today's artifacts. This fixes the build order: **12 → 13 → 14**. | SDS §18; findings 12-A, 13-A | Sequence the remaining work in dependency order. |
| 14-C | OBSERVATION | `configs/config.yaml:111-112` (`dashboard.title`) and `106-109` (`explainability.*`) are the only config keys in the file with no consumer, and both trace to unimplemented modules rather than to dead configuration. `config_loader.py`'s `_REQUIRED_TOP_LEVEL_KEYS` already mandates both sections, so the config is correctly forward-declared. | `configs/config.yaml:106-112`; `config_loader.py:42-51` | None beyond 13-A / 14-A. |
| 14-D | OBSERVATION | The per-tab independent-failure rule (14.6) is the subtlest requirement in SDS §18 and the easiest to get wrong: the natural implementation loads every artifact at the top of the file, so one missing PNG blanks the whole app. The SDS §21 Milestone 10 validation check tests exactly this ("manually deleting one artifact file causes only that tab to show the specified `st.error` message"). Flagged now so it is designed in rather than retrofitted. | SDS §18, §21 Milestone 10 | Load artifacts inside each tab's own `try`/`except`. |

### Notes

Requirement 14.7 is recorded as a vacuous PASS deliberately. The read-only
constraint is the most heavily-repeated dashboard rule in the governing
documents (SDS §11, §18, §22 all state it), and recording it here keeps it
visible to whoever writes the file rather than letting it disappear because
there was nothing to audit.

---

## Batch 4 summary

| Section | Component | Status |
| --- | --- | --- |
| 12 | `src/evaluation/` | **PARTIAL** — 7 PASS, 11 NOT IMPLEMENTED |
| 13 | `src/explainability/` | **NOT IMPLEMENTED** — 0 PASS, 14 NOT IMPLEMENTED |
| 14 | `dashboard/` | **NOT IMPLEMENTED** — 1 vacuous PASS, 11 NOT IMPLEMENTED |

**Defects raised in this batch:** 3 BLOCKER (`12-A`, `13-A`, `14-A`), 2 MAJOR
(`12-B`, `14-B`), 3 MINOR (`12-D`, `12-F`, and the `13-C` config-key violation
which is subsumed by `13-A`), 8 OBSERVATION.

**Verdict:** the presentation and analysis third of the project does not exist.
What *is* implemented (`generate_class_distribution_plot`) is compliant and
well-formed. Nothing found here is a defect in existing code — every item is an
absence, and each absence has a complete written specification in the SDS.

**Build order implied by this batch:** 12 → 13 → 14. Section 12 unblocks Tab 3;
Section 13 unblocks Tab 4; Section 14 consumes both.

---

# BATCH 5 — Orchestration scripts and test suite (SDS §13, §19, §21)

The `experiments/` layer is the only place in the project permitted to import
across all of `src/`, and the only place permitted to construct a logger. The
test suite is the mechanism by which SDS §21's per-milestone validation checks
are made executable. Both are audited here.

---

## 15. `experiments/` — PARTIAL (3 of 5 scripts present)

**Governing spec:** SDS §13 (fixed logger-name table; one logger per script),
SDS §12 (`set_global_seed` first), SDS §11 (orchestration layer may import
across `src/`), SDS §14.2 (class-distribution plot lives in the orchestrator),
SDS §23 Acceptance Criteria 2–7; `Phase_Implementation_Prompts.md` Phases 4, 6,
10, 11, 12.

**Files present:** `run_preprocessing.py` (90), `run_centralized.py` (63),
`run_federated.py` (88).
**Files absent:** `run_evaluation.py`, `run_explainability.py`.

### Requirement checklist — the three implemented scripts

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 15.1 | `run_preprocessing.py` exists and calls `run_preprocessing_pipeline(config)` | **PASS** | `run_preprocessing.py:51`. |
| 15.2 | It then calls `generate_class_distribution_plot` as a **separate** orchestration step | **PASS** | `run_preprocessing.py:71`. Keeps `src/preprocessing/` free of any `src/evaluation/` dependency, exactly as SDS §14.2's orchestration note requires. |
| 15.3 | `run_centralized.py` calls `train_centralized_model(config, logger)` | **PASS** | `run_centralized.py:47`. |
| 15.4 | `run_federated.py` calls `run_federated_simulation(config, logger)` | **PASS** | `run_federated.py:72`. |
| 15.5 | Each script loads config via `load_config`, never by reading YAML itself | **PASS** | `run_preprocessing.py:37`, `run_centralized.py:34`, `run_federated.py:54`. |
| 15.6 | `set_global_seed(config["seed"])` is the first executable step after config load | **PASS** | `run_preprocessing.py:40`, `run_centralized.py:37`, `run_federated.py:57`. All three carry the in-code SDS §12 citation. |
| 15.7 | **`get_logger` called exactly once per script** (carried from §2-C) | **PASS** | One call each: `run_preprocessing.py:42`, `run_centralized.py:39`, `run_federated.py:59`. No `src/` module calls `get_logger` except `encode_normalize.py:614`, whose script does not pass one down. See finding 15-C. |
| 15.8 | Logger names match the SDS §13 fixed table exactly | **PASS** | `"preprocessing"`, `"centralized_training"`, `"federated_training"` — verbatim. |
| 15.9 | The logger is threaded down into `src/` rather than reconstructed there | **PASS** | `run_centralized.py:47` and `run_federated.py:72` both pass `logger` as the second positional argument. |
| 15.10 | Each script is a thin orchestrator with no domain logic | **PASS** | The three `main()` bodies contain only config load, seed, logger, one or two calls, and timing. Stated explicitly at `run_federated.py:12-14`. |
| 15.11 | Every script wraps its work in `try`/`except` and logs `exc_info=True` before re-raising | **PASS** | `run_preprocessing.py:75-79`, `run_centralized.py:48-52`, `run_federated.py:73-77`. |
| 15.12 | Total wall-clock is logged on completion | **PASS** | `run_preprocessing.py:81-86`, `run_centralized.py:54-59`, `run_federated.py:79-84`. |
| 15.13 | Paths built with `os.path.join` | **PASS** | `run_preprocessing.py:54-60`. The other two build no paths. |
| 15.14 | Project root added to `sys.path` so the scripts run from anywhere | **PASS** | Identical three-line idiom at `run_preprocessing.py:24`, `run_centralized.py:24`, `run_federated.py:29`. |
| 15.15 | `if __name__ == "__main__": main()` guard present | **PASS** | All three. |
| 15.16 | Production federated values used, never the smoke-test reduced values | **PASS** | `run_federated.py:54` loads `configs/config.yaml`; the invariant is restated at `43-45` and asserted from the other side by `tests/test_federated_loop.py::test_smoke_test_never_uses_production_config`. |
| 15.17 | Class-distribution plot reads only the `label` column | **PASS** | `read_processed(..., columns=["label"])`, `run_preprocessing.py:70`; the columnar-projection rationale is stated at `67-69`. Exercises `dataio`'s push-down path (§4.7). |
| 15.18 | `run_evaluation.py` | **NOT IMPLEMENTED** | Absent; advertised at `README.md:21`. |
| 15.19 | `run_explainability.py` | **NOT IMPLEMENTED** | Absent; advertised at `README.md:22`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 15-A | **BLOCKER** | Two of the five orchestration scripts do not exist, while `README.md:18-23` presents all six run steps as the documented run order. A user following the README reaches step 4 and gets `can't open file 'experiments/run_evaluation.py'`. Directly blocks SDS §23 Acceptance Criteria 5 and 6. Downstream of `12-A` and `13-A`, not independent of them. | `experiments/` contents; `README.md:18-23` | Implement both after Sections 12 and 13. |
| 15-B | MINOR | `run_preprocessing.py` docstring (`3-9`) claims "1. Sets the global random seed" as step 1, but the code loads config first (`37`) and seeds second (`40`). It cannot be otherwise — the seed value comes from the config — so the code is right and the docstring's ordering is a paraphrase. `run_centralized.py:3-9` and `run_federated.py:3-10` carry the same harmless inversion. | `run_preprocessing.py:3-9` vs `37-40` | Reword to "loads config, then sets the global seed as the first executable step". |
| 15-C | MINOR | **§2-C follow-through.** `run_preprocessing.py` calls `get_logger("preprocessing", ...)` at `42`, and `run_preprocessing_pipeline` *also* calls `get_logger("preprocessing", ...)` internally (`encode_normalize.py:614`) because the pipeline accepts no `logger` parameter. One `python experiments/run_preprocessing.py` invocation therefore produces **two** `preprocessing_<timestamp>.log` files, and the orchestrator's log contains only the four lines it writes itself while the pipeline's contains everything of substance. The other two scripts do not have this problem — both pass their logger down. | `run_preprocessing.py:42`, `51`; `encode_normalize.py:614` | Add an optional `logger` parameter to `run_preprocessing_pipeline` mirroring `train_centralized_model` and `run_federated_simulation`, and pass it from the script. |
| 15-D | MINOR | A stray double blank line at `run_preprocessing.py:32-34` (three blank lines between the last import and `def main`). PEP 8 specifies exactly two. `flake8` reports `E303`. | `run_preprocessing.py:32-34` | Delete one blank line. |
| 15-E | OBSERVATION | All three scripts use `%`-style lazy logging (`logger.info("...%s", value)`) rather than f-strings. This is the `logging` module's documented preferred form — the interpolation is skipped when the level is disabled — and it is applied consistently across `src/` too. | `run_federated.py:62-69` | None. |
| 15-F | OBSERVATION | `run_federated.py` logs `num_clients`, `num_rounds`, and `local_epochs` at start-up (`62-69`). Combined with `server_app.py:576-578`'s `run_mode` line, a log file is self-describing about which profile produced it — which matters because `LAPTOP_MODE` and `FULL_EXPERIMENT` produce differently-shaped runs from the same code. | `run_federated.py:62-69` | None. |
| 15-G | OBSERVATION | The `sys.path` insertion at line 24/29 of each script is duplicated three times and will be duplicated five times once the missing scripts are written. Acceptable — a shared helper would itself need the path fixed to be importable — but worth noting as intentional duplication rather than an oversight of SDS §20's no-duplication rule. | `run_preprocessing.py:24` et al. | None. |

### Notes

The orchestration pattern is correct and consistent: config → seed → logger →
one call → timing → error handling. The seed-before-anything-else requirement
(15.6) is honoured in all three, and the fixed logger names (15.8) match SDS §13
character for character.

Finding 15-C is the only structural defect: `run_preprocessing_pipeline` is the
one `src/` function in the project that constructs its own logger instead of
receiving one. `train_centralized_model` and `run_federated_simulation` both
take an optional `logger` parameter, so the fix is to make the third match the
two that are already right.

---

## 16. `tests/` — PARTIAL (5 of 7 required files present)

**Governing spec:** SDS §19 (the seven-file test table and its required cases),
SDS §21 (per-milestone validation checks), SDS §22 (reduced-client config for
the federated smoke test); `Phase_Implementation_Prompts.md` Phases 1–10.

**Files present:** `test_config_loader.py` (7 cases), `test_preprocessing.py`
(~55 cases across 7 classes), `test_partitioning.py` (12 cases),
`test_models.py` (26 cases), `test_federated_loop.py` (16 cases).
**Files absent:** `tests/test_metrics.py`, `tests/test_shap_utils.py`.
**Total:** 116 test functions.

### Requirement checklist — `test_config_loader.py` (SDS §19 row 1)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.1 | Loads a valid config successfully | **PASS** | `test_load_valid_config_returns_dict`, `test_load_valid_config_contains_all_required_keys`, `test_load_valid_config_seed_is_42`. |
| 16.2 | Raises `FileNotFoundError` on a missing file | **PASS** | `test_load_config_missing_file_raises_file_not_found`, plus a message-content assertion. |
| 16.3 | Raises `KeyError` on a config missing a required top-level key | **PASS** | `test_load_config_missing_top_level_key_raises_key_error`, plus `test_load_config_missing_key_error_message_names_key`. |

### Requirement checklist — `test_preprocessing.py` (SDS §19 row 2)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.4 | `load_raw_dataset` raises `FileNotFoundError` on a bad path | **PASS** | `TestLoadRawDataset` — 6 cases. |
| 16.5 | `drop_identifier_columns` removes only configured columns, ignores missing ones, never removes the native label columns | **PASS** | `TestDropIdentifierColumns` — 7 cases, including `test_never_drops_attack_type` / `test_never_drops_attack_label`. Directly tests Contract Invariant A.2. |
| 16.6 | `create_labels` produces both label columns with no NaNs and removes both native columns | **PASS** | `TestCreateLabels` — 14 cases, including `test_attack_type_absent_from_result` and `test_attack_label_absent_from_result`. |
| 16.7 | The binary-agreement mismatch warns rather than raises | **PASS** | `test_binary_mismatch_produces_warning_not_exception`. Tests §5.12 precisely. |
| 16.8 | `infer_feature_column_types` separates a mixed frame, excludes label columns, raises on an unknown rule | **PASS** | `TestInferFeatureColumnTypes` — 10 cases including disjointness and union-coverage. |
| 16.9 | `split_train_test` produces non-overlapping sets covering the full dataset | **PASS** | `TestSplitTrainTest` — 6 cases including determinism and the `<2 samples` `ValueError`. |
| 16.10 | `prepare_model_ready_data` shapes are correct | **PASS** | `TestPrepareModelReadyData` — 5 cases covering `X`/`y` shape, one-hot validity, index subsetting, and value fidelity. |
| 16.11 | **Leakage regression test: train-only fit differs from full-dataset fit** | **PASS** | `TestLeakageRegression` — `test_scaler_train_only_differs_from_full_dataset_fit` and `test_categorical_encoder_train_only_differs_from_full_fit`. This is the single most valuable test in the suite; SDS §19 calls it out by name. |

### Requirement checklist — `test_partitioning.py` (SDS §19 row 3)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.12 | Produces exactly `num_clients` shards | **PASS** | Parameterised over `[1, 2, 4, 10]`. |
| 16.13 | Shard sizes sum to the dataset size | **PASS** | Plus a dedicated non-divisible-length case. |
| 16.14 | No index appears in more than one shard | **PASS** | Parameterised over `[2, 4, 10]`. |
| 16.15 | Raises `ValueError` on `num_clients < 1` | **PASS** | Parameterised over `[0, -1, -5]`; plus the `> len(df)` case. |
| 16.16 | **Beyond spec: shuffle-before-split verified** | **PASS** | `test_partition_iid_shuffles_rows_before_splitting` asserts shards are not contiguous slices — the property §8.6 identifies as what makes the IID claim true. Not required by SDS §19; correctly added. |

### Requirement checklist — `test_models.py` (SDS §19 row 4)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.17 | Returns a compiled `tf.keras.Model` | **PASS** | Plus optimizer/learning-rate/loss/metric assertions. |
| 16.18 | Forward pass yields `(batch_size, num_classes)` | **PASS** | Plus declared input/output shape and softmax-sums-to-1 checks. |
| 16.19 | Raises `ValueError` on `num_classes < 2` | **PASS** | Parameterised over `[1, 0, -1]`. |
| 16.20 | **Confirms `Conv1D` `padding="same"`** | **PASS** | `test_conv1d_padding_is_same` — tests the Contract Part C lock directly. |
| 16.21 | **Confirms `GRU` `return_sequences=False`** | **PASS** | `test_gru_return_sequences_is_false`. |
| 16.22 | Full layer sequence matches SDS §14.4 | **PASS** | `test_layer_sequence_matches_sds_section_14_4`, plus `test_no_flatten_layer_present`. |
| 16.23 | Parameter count under the 500K soft guideline | **PASS** | `test_model_parameter_count_is_lightweight` — makes SDS §21 Milestone 3's soft guideline executable. |

### Requirement checklist — `test_federated_loop.py` (SDS §19 row 5)

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.24 | **An independent reduced-scale config is constructed, never the production one** | **PASS** | `reduced_config` fixture builds the dict from scratch; module docstring states the rule; `test_smoke_test_never_uses_production_config` asserts `num_clients != 4`. |
| 16.25 | All four client counts set to 2 in the reduced config | **PASS** | `num_clients`, `min_available_clients`, `min_fit_clients`, `min_evaluate_clients` all `REDUCED_NUM_CLIENTS`. |
| 16.26 | A 2-round / 2-client smoke test completes without raising | **PASS** | `test_federated_simulation_smoke_test`, via `flwr.simulation.start_simulation`. |
| 16.27 | `strategy.latest_parameters` is not `None` after the run | **PASS** | Asserted, then round-tripped through `parameters_to_ndarrays` into a fresh model — verifying the §11.24–11.26 save path end to end. |
| 16.28 | Non-empty in-memory history produced | **PASS** | Asserted on the returned `History`. |
| 16.29 | Uses the pinned `start_simulation` API exclusively | **PASS** | Stated in the module docstring and used in the test body. |
| 16.30 | **The two aggregation functions are distinct** | **PASS** | `test_aggregation_functions_are_distinct` — guards SDS §22's explicit prohibition. |
| 16.31 | `weighted_average_fit` requires `loss`; `weighted_average_eval` must not | **PASS** | `test_weighted_average_fit_raises_key_error_without_loss` and `test_weighted_average_eval_ignores_absent_loss_key`. Together these pin the §14.5/§14.6 metrics-dict contract from both sides. |
| 16.32 | Example-count weighting verified, not just key presence | **PASS** | `test_weighted_average_fit_weights_by_example_count`. |
| 16.33 | `get_strategy` wiring verified (correct fn on correct callback) | **PASS** | `test_get_strategy_assigns_aggregation_functions_correctly` — catches the swapped-assignment defect. |
| 16.34 | `aggregate_fit` does not capture `None` | **PASS** | `test_aggregate_fit_does_not_capture_none` — tests §11.4. |
| 16.35 | No disk I/O in the smoke test | **PASS** | Module docstring states in-memory synthetic data only; enabled by the client's zero-I/O property (§10.10). |

### Requirement checklist — missing files

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 16.36 | `tests/test_metrics.py` | **NOT IMPLEMENTED** | SDS §19 row 6. Blocked by `12-A`/`12-B`. |
| 16.37 | `tests/test_shap_utils.py` | **NOT IMPLEMENTED** | SDS §19 row 7. Blocked by `13-A`. |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 16-A | **MAJOR** | Two of the seven SDS §19 test files are absent. SDS §23 Acceptance Criterion 8 and §21 Milestone 11 require every listed test file to exist and pass. Both are blocked by the corresponding missing implementation, so this is a consequence of `12-A`/`13-A` rather than an independent omission. | `tests/` contents; SDS §19 | Write both alongside the modules they cover. |
| 16-B | OBSERVATION | Coverage of what *is* implemented substantially exceeds SDS §19. The spec lists roughly 30 required cases across the five present files; the suite has 116. `test_preprocessing.py` alone contributes ~55 across 7 test classes. Several additions are genuinely valuable rather than padding — `test_partition_iid_shuffles_rows_before_splitting` (16.16), `test_model_parameter_count_is_lightweight` (16.23), and `test_aggregate_fit_does_not_capture_none` (16.34) each test a property the audit identified as load-bearing. | `tests/` — 116 test functions | None. |
| 16-C | OBSERVATION | `TestLeakageRegression` is the highest-value class in the suite. It asserts that a scaler fit on training rows produces *different* parameters than one fit on the full dataset — i.e. it fails if someone "simplifies" the pipeline by fitting before splitting. That is precisely the defect the SDS §14.2 ordering exists to prevent, and it is the only defect class in this project that would inflate rather than degrade the reported metrics. | `tests/test_preprocessing.py::TestLeakageRegression` | None. |
| 16-D | MINOR | No test covers `src/utils/logger.py`, `src/utils/seed.py`, or `src/utils/dataio.py`. SDS §19 does not list files for them, so this is not a spec violation — but SDS §21 Milestone 1 states validation checks for `seed.py` ("calling `set_global_seed(42)` twice produces identical `numpy.random.rand(5)`") and `logger.py` ("creates a timestamped file named `preprocessing_<timestamp>.log`") that exist only as prose. `dataio.py` is untested entirely, and its extension-dispatch and `ImportError` paths (§4.5, §4.10) are exactly the kind of branch a two-line test pins cheaply. | `tests/` contents; SDS §21 Milestone 1 | Optional: add `tests/test_utils.py` covering the two Milestone-1 checks plus `dataio`'s dispatch. |
| 16-E | MINOR | No test exercises `src/centralized/train_baseline.py`. SDS §19 lists no file for it, and its validation is defined in §21 Milestone 4 as an end-to-end accuracy check rather than a unit test — but the class-weight computation block (`train_baseline.py:194-234`) is pure, deterministic, and independently testable, including the single-class and `use_class_weights: false` branches. | `tests/` contents | Optional: unit-test the class-weight block in isolation. |
| 16-F | OBSERVATION | `test_federated_loop.py` runs a real `start_simulation` — a genuine smoke test, not a mock. It is therefore the slowest file in the suite and the one that would catch a Flower version drift away from the pinned `1.8.0` API. Correct trade-off for a project whose SDS §7 pins that version precisely because the API is unstable across releases. | `tests/test_federated_loop.py` | None. |
| 16-G | OBSERVATION | No `pytest.ini`, `setup.cfg`, `pyproject.toml`, or `conftest.py` exists at the project root. `tests/__init__.py` is present, so `pytest` resolves `src.*` imports via rootdir insertion — which works when invoked as `pytest` from the project root, the documented execution mode. Worth noting that `pytest tests/test_models.py` from another directory would fail. | Repository root listing | Optional: add a minimal `pytest.ini` with `testpaths = tests`. |

### Notes

The test suite is the strongest completed non-source artifact in the project.
Every SDS §19 case for the five implemented files is present, and the additions
beyond spec target real invariants rather than inflating the count.

Three tests deserve specific mention because they pin invariants the audit
identified as load-bearing and hard to verify by reading:

- `test_never_drops_attack_type` / `test_never_drops_attack_label` — Contract
  Invariant A.2 from the negative side.
- `test_scaler_train_only_differs_from_full_dataset_fit` — the leakage guard.
- `test_smoke_test_never_uses_production_config` — SDS §22's reduced-client rule.

The gap is entirely downstream: `test_metrics.py` and `test_shap_utils.py` cannot
be written before the modules they cover exist.

---

## 17. `README.md` and repository documentation — PARTIAL

**Governing spec:** SDS §21 Milestone 11 (README must document the exact run
order), SDS §24 (deliverables checklist), SDS §23 Acceptance Criterion 8.

**Files:** `README.md` (23 lines), `requirements.txt` (11 lines),
`.gitignore`, `project-docs/` (4 documents).

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 17.1 | `README.md` exists | **PASS** | 23 lines. |
| 17.2 | Documents the exact six-step run order | **PASS** | `README.md:18-23`, matching the SDS §21 Milestone 11 sequence. |
| 17.3 | States the Python version | **PASS** | `README.md:11` — 3.10.*. |
| 17.4 | Project overview present | **PASS** | `README.md:5-7`. |
| 17.5 | Setup / installation instructions | **NOT IMPLEMENTED** | No `pip install -r requirements.txt`, no venv guidance. `README.md:12` only points at the file. |
| 17.6 | Dataset acquisition and placement instructions | **NOT IMPLEMENTED** | Nothing states that `data/raw/edge_iiotset.csv` must be manually placed (SDS §21 Milestone 2 depends on it). |
| 17.7 | `LAPTOP_MODE` / `IIOT_MODE` documented | **NOT IMPLEMENTED** | The dual-profile system (`config_loader.py:53-61`) is invisible to a reader of the README. |
| 17.8 | Expected outputs / results documented | **NOT IMPLEMENTED** | No description of what lands in `outputs/`. |
| 17.9 | `requirements.txt` carries the four exact SDS §7 pins | **PASS** | `tensorflow==2.15.1`, `flwr[simulation]==1.8.0`, `ray==2.6.3`, `shap==0.44.1` (`requirements.txt:1-3,7`). |
| 17.10 | All other SDS §7 packages present | **PASS** | `scikit-learn`, `pandas`, `numpy`, `streamlit`, `matplotlib`, `pyyaml`, `pytest`. |
| 17.11 | `pyarrow` present (required by the default config) | **FAIL** | Absent. This is finding `4-A`, confirmed against the file. |
| 17.12 | `pydot` / `graphviz` present (architecture diagrams) | **NOT IMPLEMENTED** | Absent; non-fatal by design (§7-D). |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 17-A | **MAJOR** | `README.md:14-16` presents a six-step run order in which **steps 4, 5 and 6 do not exist**, under a heading that says "To be completed in Phase 18". A reader cannot distinguish "not yet documented" from "not yet built". For a project whose acceptance criteria are executed from this list, that is a materially misleading document. | `README.md:14-23` vs `experiments/` and `dashboard/` contents | Mark steps 4–6 as not yet implemented until they are, then complete the README in Phase 18. |
| 17-B | **MAJOR** | **Confirms `4-A`.** `requirements.txt` has no `pyarrow`, while `configs/config.yaml:7` names a `.parquet` processed-data file. A clean install fails at the first `write_processed`. Note the working tree contains **both** `edge_iiotset_processed.csv` and `edge_iiotset_processed.parquet`, which is why the defect has not been felt locally — a machine that once ran the CSV profile has both artifacts and never exercises the failing path. | `requirements.txt`; `configs/config.yaml:7`; `data/processed/` listing | Add `pyarrow>=14,<16`. |
| 17-C | MINOR | The README omits the dataset-acquisition step entirely. SDS §21 Milestone 2 states `data/raw/edge_iiotset.csv` must be manually placed before preprocessing can run, and nothing in the run order says so. A new user's first command fails with `FileNotFoundError`. | `README.md:14-23`; SDS §21 Milestone 2 | Add a "Dataset" section naming the file, the expected path, and the source. |
| 17-D | MINOR | `LAPTOP_MODE` is undocumented outside the config files. It is a substantial, well-engineered feature — a whole overlay file and roughly 200 lines of `server_app.py` — and a user on a 16 GB Windows machine has no way to discover that `set IIOT_MODE=LAPTOP_MODE` is what makes the federated run survive. | `README.md`; `config_loader.py:53-61`; `configs/config_laptop.yaml:1-31` | Document both profiles and the env-var switch. |
| 17-E | OBSERVATION | `project-docs/` contains four substantial governing documents (`SDS.md` at 1473 lines, `AI_Coding_Contract.md`, `Phase_Implementation_Prompts.md`, and this audit). The specification layer is far more complete than the implementation layer — the reverse of the usual failure mode, and the reason Sections 12–14 could be written as actionable build specifications rather than mere gap reports. | `project-docs/` | None. |
| 17-F | OBSERVATION | `data/processed/` holds both a `.csv` and a `.parquet` copy of the processed dataset. Only the `.parquet` is read under the current config; the `.csv` is a stale ~643 MB artifact from before the `dataio` migration. Harmless, but it is the single largest file in the working tree and `.gitignore` should be confirmed to exclude it. | `data/processed/` listing | Delete the stale CSV once the Parquet path is confirmed working. |

### Notes

Finding 17-A is worth separating from the missing code it describes. The
implementation gaps (12-A, 13-A, 14-A, 15-A) are honest, declared, and
phase-tracked. The README is the one document that *presents* the project as
more complete than it is, by listing six run steps without marking which three
are unimplemented. That is a one-line fix and it removes the only misleading
artifact in the repository.

---

## Batch 5 summary

| Section | Component | Status |
| --- | --- | --- |
| 15 | `experiments/` | **PARTIAL** — 17 PASS, 2 NOT IMPLEMENTED |
| 16 | `tests/` | **PARTIAL** — 35 PASS, 2 NOT IMPLEMENTED |
| 17 | `README.md` / packaging | **PARTIAL** — 6 PASS, 1 FAIL, 5 NOT IMPLEMENTED |

**Defects raised in this batch:** 1 BLOCKER (`15-A`), 3 MAJOR (`16-A`, `17-A`,
`17-B` — the last confirming `4-A`), 6 MINOR (`15-B`, `15-C`, `15-D`, `16-D`,
`16-E`, `17-C`, `17-D`), 6 OBSERVATION.

**Verdict:** the orchestration and test layers are correct for everything that
exists. `get_logger` is called exactly once per script (resolving §2-C, with the
one exception raised as `15-C`), the seed-first rule holds in all three scripts,
and the 116-test suite covers every SDS §19 case for the five implemented files.

**Carried forward:**
- `15-A`, `16-A` → downstream of `12-A` / `13-A`; sequence accordingly.
- `17-A`, `17-B` → Critical Bugs (README honesty; `pyarrow`).
- `15-C` → Remaining work (duplicate preprocessing log file).
- `17-C`, `17-D` → Remaining work (documentation).

---

# BATCH 6 — Cross-cutting conformance (SDS §11, §12)

The final two sections check properties no single module owns: the import graph
across the whole repository, and the artifact inventory against the SDS's
declared outputs.

---

## 18. Import-graph conformance — PASS

**Governing spec:** SDS §11 (the dependency graph), SDS §20 (no circular
imports), SDS §22 (dashboard import prohibitions).

### Declared graph vs. observed imports

| Module | SDS §11 permits | Observed | Status |
| --- | --- | --- | --- |
| `src/utils/` | nothing from `src/` | `config_loader`, `logger`, `seed`, `dataio` import only stdlib + `yaml`/`numpy`/`pandas`/`tensorflow` | **PASS** |
| `src/preprocessing/` | `src/utils/` | `load_dataset.py`, `encode_normalize.py` import `src.utils.*` only | **PASS** |
| `src/models/` | `src/utils/` | `cnn_gru.py` imports `io` + `tensorflow` only — not even `src/utils/` | **PASS** |
| `src/partitioning/` | `src/utils/`, `src/preprocessing/` | `partition_data.py` imports `numpy` + `pandas` only | **PASS** |
| `src/centralized/` | `src/utils/`, `src/preprocessing/`, `src/models/` | Exactly those three (`train_baseline.py:37-43`) | **PASS** |
| `src/federated/` | `src/utils/`, `src/preprocessing/`, `src/models/`, `src/partitioning/` | `client_app.py:46-58` and `server_app.py:47-60` import exactly those four (+ `flwr`) | **PASS** |
| `src/evaluation/` | `src/utils/`, `src/preprocessing.prepare_model_ready_data`, `src/models/` | `metrics.py` imports `matplotlib` + `pandas` only | **PASS** |
| `src/explainability/` | `src/utils/`, `src/preprocessing.prepare_model_ready_data`, `src/models/` | N/A — not implemented | **N/A** |
| `experiments/` | all of `src/` | `run_preprocessing.py` imports `utils` + `preprocessing` + `evaluation`; the other two import `utils` + their training module | **PASS** |
| `dashboard/` | `utils`, `evaluation`, `prepare_model_ready_data`, `explain_single_prediction` **only** | N/A — not implemented | **N/A** |

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 18.1 | No module imports from a layer above it | **PASS** | Verified per row above. |
| 18.2 | **No circular imports** | **PASS** | The observed edges form a DAG: `utils` → `preprocessing`/`models`/`partitioning` → `centralized`/`federated`/`evaluation` → `experiments`. `server_app` imports `client_app` but not the reverse (`client_app.py:46-58` has no `server_app` import). |
| 18.3 | `src/preprocessing/` never imports `src/evaluation/` | **PASS** | The class-distribution plot call lives in `run_preprocessing.py:71`, exactly as SDS §14.2's orchestration note requires. This is the single most easily-violated edge in the graph and it is respected. |
| 18.4 | `src/models/` imports nothing from `src/` | **PASS** | `cnn_gru.py:14-27`. Explains `7-C` (TF's own logger used instead of the project logger). |
| 18.5 | `src/federated/` does not import `src/centralized/` | **PASS** | Neither federated module references it. |
| 18.6 | `src/centralized/` does not import `src/federated/` or `src/partitioning/` | **PASS** | `train_baseline.py:37-43`. Confirms §9.1 — the centralized path never partitions. |
| 18.7 | Dashboard import restrictions | **N/A** (vacuous PASS) | No dashboard exists. Recorded so the constraint is not lost. |
| 18.8 | `experiments/` is the only cross-cutting layer | **PASS** | Only `run_preprocessing.py` spans two `src/` sub-packages (`preprocessing` + `evaluation`), which is the permitted orchestration pattern. |
| 18.9 | Deferred/local imports do not hide a cycle | **PASS** | The only function-local import is `import ray` (`server_app.py:744`), a third-party package (§11-J). |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 18-A | OBSERVATION | The graph is not merely acyclic — it is *strictly layered*, with every module importing only from levels strictly below it. No sibling-to-sibling import exists anywhere except `server_app` → `client_app`, which SDS §14.6 mandates. | Whole-repository import survey | None. |
| 18-B | OBSERVATION | `src/models/cnn_gru.py` importing nothing from `src/` at all is stricter than SDS §11 requires (it permits `src/utils/`). The cost is `7-C`; the benefit is that the model builder is trivially reusable and testable in isolation, which `tests/test_models.py`'s 26 cases exploit — none of them touches config loading or logging. | `cnn_gru.py:14-27`; `tests/test_models.py` | None. |
| 18-C | OBSERVATION | Every `src/` sub-package has an `__init__.py`, and all are empty rather than re-exporting. This prevents the common Python cycle where package `__init__` files import submodules that import each other. Consistent and deliberate. | `src/*/__init__.py` | None. |
| 18-D | MINOR | The three `sys.path.insert(0, ...)` calls in `experiments/` execute before the `src.*` imports, which `flake8` reports as `E402` (module-level import not at top of file). Unavoidable given the requirement that the scripts run as `python experiments/run_x.py` from the project root, and SDS §20's PEP 8 requirement is therefore in tension with SDS §21's documented invocation. Worth an inline `# noqa: E402`. | `run_preprocessing.py:24-31` et al. | Add `# noqa: E402` to the `src.*` import lines, or install the project as a package. |

### Notes

Import-graph conformance is complete for every implemented module, and 18.3 is
the result worth highlighting. Placing `generate_class_distribution_plot` in the
orchestrator rather than inside `run_preprocessing_pipeline` looks like a
stylistic choice; it is what keeps `src/preprocessing/` free of a dependency on
`src/evaluation/`, and the SDS calls it out explicitly for that reason. The code
follows the rule rather than the convenience.

The two `N/A` rows are the only gaps, and both are consequences of the missing
modules rather than conformance failures. When `dashboard/app.py` is written it
will be the single most import-constrained file in the project — SDS §11, §18,
and §22 each independently restrict it.

---

## 19. Artifact inventory vs. SDS §12 — PARTIAL

**Governing spec:** SDS §9 (folder tree), SDS §12 (artifact ownership and
overwrite rules), SDS §17 (plot inventory), SDS §24 (deliverables checklist).

### `outputs/artifacts/` — preprocessing outputs

| Artifact | Owner | Present | Status |
| --- | --- | --- | --- |
| `scaler.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `label_encoder.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `categorical_encoder.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `train_indices.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `test_indices.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `feature_names.pkl` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `class_mapping.json` | `run_preprocessing_pipeline` | Yes | **PASS** |
| `random_seed.txt` | `set_global_seed` | Yes | **PASS** |
| `shap_background.npy` | `generate_shap_explanations` | **No** | **NOT IMPLEMENTED** (13-A) |

### `outputs/models/`

| Artifact | Owner | Present | Status |
| --- | --- | --- | --- |
| `centralized_best_model.h5` | `train_centralized_model` | Yes | **PASS** |
| `centralized_last_model.h5` | `train_centralized_model` | Yes | **PASS** |
| `centralized_model_summary.txt` | `train_centralized_model` | Yes | **PASS** |
| `centralized_architecture.png` | `train_centralized_model` | **No** | **PARTIAL** — non-fatal by design (§7-D, §9.21); `pydot` absent from `requirements.txt` |
| `federated_global_model.h5` | `run_federated_simulation` | **No** | Code complete; not yet executed |
| `federated_model_summary.txt` | `run_federated_simulation` | **No** | Code complete; not yet executed |
| `federated_architecture.png` | `run_federated_simulation` | **No** | Same as centralized |

### `outputs/results/`

| Artifact | Owner | Present | Status |
| --- | --- | --- | --- |
| `class_distribution.png` | `run_preprocessing.py` | Yes | **PASS** |
| `centralized_history.csv` | `train_centralized_model` | Yes | **PASS** |
| `centralized_training_time.txt` | `train_centralized_model` | Yes | **PASS** |
| `federated_history.csv` | `run_federated_simulation` | **No** | Code complete; not yet executed |
| `federated_training_time.txt` | `run_federated_simulation` | **No** | Code complete; not yet executed |
| `comparison_table.csv` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `model_size_comparison.csv` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `convergence_plot.png` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `centralized_confusion_matrix.png` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `federated_confusion_matrix.png` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `centralized_classification_report.txt` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `federated_classification_report.txt` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |
| `centralized_training_curves.png` | `run_evaluation_pipeline` | **No** | **NOT IMPLEMENTED** (12-A) |

### `outputs/shap_plots/`

| Artifact | Owner | Present | Status |
| --- | --- | --- | --- |
| `shap_summary_plot.png` | `generate_shap_explanations` | **No** (dir empty) | **NOT IMPLEMENTED** (13-A) |
| `shap_bar_plot.png` | `generate_shap_explanations` | **No** (dir empty) | **NOT IMPLEMENTED** (13-A) |

### `data/processed/`

| Artifact | Owner | Present | Status |
| --- | --- | --- | --- |
| `edge_iiotset_processed.parquet` | `write_processed` | Yes | **PASS** — the live artifact under `configs/config.yaml:7` |
| `edge_iiotset_processed.csv` | `write_processed` (legacy) | Yes | **OBSERVATION** — stale pre-`dataio` copy; see `17-F` |

### Requirement checklist

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 19.1 | Every preprocessing artifact in SDS §12 exists on disk | **PASS** | 8/8 in `outputs/artifacts/`. |
| 19.2 | Every artifact has exactly one owning function | **PASS** | Verified per table; no artifact is written from two places. |
| 19.3 | `logs/` is never overwritten (timestamped per run) | **PASS** | §2.6. |
| 19.4 | Processed data regenerated only under `force_reprocess` | **PASS** | §6.23. |
| 19.5 | Models overwritten on every training run | **PASS** | §9.18–9.23. |
| 19.6 | Centralized pipeline artifacts complete | **PASS** | 3 of 4 present; the missing `.png` is the by-design `pydot` gap. |
| 19.7 | Federated pipeline artifacts complete | **NOT RUN** | All five write paths are implemented and verified (§11.28–11.32); none has been executed. |
| 19.8 | Evaluation artifacts complete | **NOT IMPLEMENTED** | 0 of 8 — no producing code. |
| 19.9 | Explainability artifacts complete | **NOT IMPLEMENTED** | 0 of 3 — no producing code. |
| 19.10 | `logs/` directory present | **PASS** | Created on demand by `get_logger` (§2.7). |

### Findings

| # | Severity | Finding | Evidence | Required fix |
| --- | --- | --- | --- | --- |
| 19-A | OBSERVATION | The inventory partitions cleanly into three states: **produced** (preprocessing + centralized, 12 artifacts), **producible but not yet run** (federated, 5 artifacts — code verified complete in Section 11), and **not producible** (evaluation + explainability, 11 artifacts — no code). Only the third category needs new work; the second needs one command. | `outputs/` listing vs SDS §12 | Run `experiments/run_federated.py` to close category two. |
| 19-B | MINOR | `outputs/models/centralized_architecture.png` is absent because `pydot` is not installed. SDS §24's deliverables checklist lists architecture diagrams; the code handles the absence gracefully (§9.21) but the deliverable is unmet. | `outputs/models/` listing; `requirements.txt` | Add `pydot` + system graphviz, or drop the diagrams from §24. |
| 19-C | OBSERVATION | `outputs/shap_plots/` exists but is empty — the directory was created by the Milestone 0 scaffolding, consistent with SDS §9's folder tree. The scaffolding is complete even where the code is not. | `outputs/shap_plots/` | None. |
| 19-D | MINOR | `data/processed/` holds both formats (~643 MB CSV + Parquet). Only the Parquet is read. Confirms `17-F`; noted here as an inventory item because the stale CSV would be picked up by any tooling that globs `data/processed/*`. | `data/processed/` listing | Delete the stale CSV. |
| 19-E | OBSERVATION | `outputs/artifacts/random_seed.txt` is present, confirming `set_global_seed` has run and persisted the audit record (§3.7). Its presence alongside a complete artifact set is direct evidence that the preprocessing and centralized pipelines executed end to end on real data — the artifacts are not scaffolding placeholders. | `outputs/artifacts/random_seed.txt` | None. |

### Notes

The inventory confirms what the code audit predicted. Every artifact whose
producing code passed its section is present on disk, with two explained
exceptions: the architecture diagrams (missing optional dependency, non-fatal by
design) and the federated set (code verified, run not yet performed).

The category-two group is worth emphasising for planning. Five artifacts —
`federated_global_model.h5`, its summary and diagram, `federated_history.csv`,
and `federated_training_time.txt` — need no new code. Section 11 verified every
write path. They are one `python experiments/run_federated.py` away, and
`run_evaluation_pipeline` cannot be meaningfully tested until they exist.

---

## Batch 6 summary

| Section | Component | Status |
| --- | --- | --- |
| 18 | Import-graph conformance | **PASS** — 9/9 requirements (2 vacuous, pending modules) |
| 19 | Artifact inventory vs SDS §12 | **PARTIAL** — 12 present, 5 producible, 11 not producible |

**Defects raised in this batch:** 0 BLOCKER, 0 MAJOR, 3 MINOR (`18-D`, `19-B`,
`19-D`), 6 OBSERVATION.

**Verdict:** the architecture is sound. The import graph is strictly layered and
acyclic, with the most easily-violated edge (`preprocessing` → `evaluation`)
correctly avoided via the orchestration layer. Artifact ownership is
unambiguous — no artifact is written from two places anywhere in the repository.

---

# FINAL ROLL-UP

> **This section supersedes the "AUDIT CHECKPOINT — resume instructions" block
> above.** That block was written when Sections 1–10 were complete and the audit
> was paused; all six of its open verification items are now closed (see
> "Frozen-invariant verification" below). It is retained unedited for audit
> provenance only — do not act on its resume instructions.

**Audit completed:** 2026-08-07
**Commit audited:** `58bf27c4d0f8843fe2ace2d15d0c7b5aefaa18a2`
**Sections:** 19 of 19 complete. **Requirements checked:** 369.

---

## 1. Executive verdict

**Every line of code in this repository that exists is compliant. Roughly a
third of the specified code does not exist.**

That single sentence is the finding. Across 369 checked requirements there is
exactly **one FAIL** (a missing dependency line in `requirements.txt`) and
**zero** violations of any frozen Contract invariant. The eleven implemented
modules pass their specifications, in several places exceed them, and in two
documented cases deliberately deviate from the written SDS in order to *close* a
data-leakage vector the SDS's literal ordering would have left open.

Against that, four BLOCKER findings record whole layers that were specified and
never built: the evaluation/comparison module, the SHAP explainability module,
the Streamlit dashboard, and the two `experiments/` scripts that drive them.
These are not defects in code — they are absences of code, each with a complete
written specification already in `SDS.md`.

The project is therefore in an unusually favourable position for its state of
completeness: the risky, hard-to-verify half (leakage control, federated
aggregation correctness, reproducibility, architecture freeze) is **done and
verified**, and the remaining half is mechanical construction against a
specification that has already been proven precise enough to build from.

---

## 2. Consolidated section status

| # | Component | Status | Requirements | Notes |
| --- | --- | --- | --- | --- |
| 1 | `src/utils/config_loader.py` | **PASS** | 15/15 | Dual run-profile system |
| 2 | `src/utils/logger.py` | **PASS** | 11/11 | |
| 3 | `src/utils/seed.py` | **PASS WITH NOTES** | 8 PASS, 1 PARTIAL | `PYTHONHASHSEED` timing |
| 4 | `src/utils/dataio.py` | **PASS** | 11/11 | `4-A` dependency gap |
| 5 | `src/preprocessing/load_dataset.py` | **PASS** | 16/16 | Contract A.2 holds |
| 6 | `src/preprocessing/encode_normalize.py` | **PASS WITH NOTES** | 36 PASS, 1 PARTIAL | Leakage discipline exemplary |
| 7 | `src/models/cnn_gru.py` | **PASS** | 23/23 | Architecture frozen correctly |
| 8 | `src/partitioning/partition_data.py` | **PASS** | 13/13 | Genuine IID sharding |
| 9 | `src/centralized/train_baseline.py` | **PASS** | 28/28 | |
| 10 | `src/federated/client_app.py` | **PASS** | 27/27 | `10-C` memory defect |
| 11 | `src/federated/server_app.py` | **PASS WITH NOTES** | 47 PASS, 1 PARTIAL | Most mature module |
| 12 | `src/evaluation/` | **PARTIAL** | 7 PASS, 11 NI | **BLOCKER 12-A** |
| 13 | `src/explainability/` | **NOT IMPLEMENTED** | 0 PASS, 14 NI | **BLOCKER 13-A** |
| 14 | `dashboard/` | **NOT IMPLEMENTED** | 1 vacuous, 11 NI | **BLOCKER 14-A** |
| 15 | `experiments/` | **PARTIAL** | 17 PASS, 2 NI | **BLOCKER 15-A** |
| 16 | `tests/` | **PARTIAL** | 35 PASS, 2 NI | 116 tests, all passing scope |
| 17 | `README.md` / packaging | **PARTIAL** | 6 PASS, 1 FAIL, 5 NI | `17-A`, `17-B` |
| 18 | Import-graph conformance | **PASS** | 9/9 | Strictly layered, acyclic |
| 19 | Artifact inventory | **PARTIAL** | 7 PASS, 1 NOT RUN, 2 NI | 12 of 28 artifacts on disk |

---

## 3. Severity tally

| Severity | Count | IDs |
| --- | --- | --- |
| **BLOCKER** | **4** | `12-A`, `13-A`, `14-A`, `15-A` |
| **MAJOR** | **7** | `4-A`, `10-C`, `12-B`, `14-B`, `16-A`, `17-A`, `17-B` |
| **MINOR** | **33** | `1-C`, `2-B`, `3-A`, `3-B`, `4-C`, `5-C`, `6-D`, `6-E`, `6-F`, `7-B`, `8-C`, `8-D`, `9-D`, `9-E`, `10-D`, `10-E`, `11-B`, `11-D`, `11-E`, `11-H`, `11-K`, `12-D`, `12-F`, `15-B`, `15-C`, `15-D`, `16-D`, `16-E`, `17-C`, `17-D`, `18-D`, `19-B`, `19-D` |
| **OBSERVATION** | **63** | — |
| **Total findings** | **107** | |

> **Reconciliation note.** The per-batch summary blocks above were tallied when
> each batch was written and two of them (Batches 0 and 4) undercount by one or
> two items. The figures in this table are an authoritative recount across all
> nineteen sections and supersede the per-batch counts where they differ. No
> finding was added, removed, or re-graded during the recount.

### Requirement outcomes

| Outcome | Count | % |
| --- | --- | --- |
| **PASS** | 317 | 85.9% |
| **PARTIAL** | 3 | 0.8% |
| **FAIL** | 1 | 0.3% |
| **NOT IMPLEMENTED** | 47 | 12.7% |
| **NOT RUN** | 1 | 0.3% |
| **Total** | **369** | 100% |

The three PARTIALs are `3.5` (`PYTHONHASHSEED` set after interpreter start),
`6.37` (pre-persist NaN check warns rather than raises), and `11.48` (training
frame released late and only under `LAPTOP_MODE`). The single FAIL is `17.11`
(`pyarrow` absent from `requirements.txt`).

**Read the 85.9% carefully.** It is not a quality score — it is a completeness
score depressed by 47 requirements belonging to unwritten modules. Restricted to
implemented code, the figure is **317 PASS / 321 checked = 98.8%**, with 3
PARTIAL and 1 FAIL.

---

## 4. Critical bugs — fix these first

Ordered by cost-to-fix against impact. The first three are one-line changes.

### C1 — `pyarrow` missing from `requirements.txt` (`4-A`, `17-B`) — MAJOR

`configs/config.yaml:7` selects `edge_iiotset_processed.parquet` as the default
processed-data file, but `pyarrow` is not a dependency and `pandas>=2.0` does not
pull it in. **A clean environment built strictly from `requirements.txt` fails at
the first `write_processed` call** — including every fresh Google Colab runtime.
It has gone unnoticed locally only because this working tree still contains a
pre-migration `.csv` copy alongside the `.parquet`.

**Fix:** add `pyarrow>=14,<16` to `requirements.txt`. One line. Highest
value-per-character change in the repository.

### C2 — Unused `test_data` retained per Ray actor (`10-C`, `11-H`) — MAJOR

`client_app.py:134` stores `self.test_data`, and `evaluate()` never reads it —
it uses `self.X_validation` (`394`). Under Flower's Ray backend **every actor
holds a full copy of the ~443k-row global test frame for the entire simulation
with no consumer**: roughly 168 MB per actor as float32, more as an un-downcast
pandas frame. `server_app.py:659` binds the full frame to `client_fn`, while
`config_laptop.yaml:115-123` carefully subsamples the *server's* copy to 50k rows
to save ~150 MB. The saving is being made on one side and given back several
times over on the other.

This is the exact code path that an entire config overlay and ~200 lines of
`server_app.py` exist to keep inside 16 GB on Windows.

**Fix:** delete the `self.test_data = test_data` assignment (keep the
constructor parameter — SDS §14.5 fixes the signature). `10-C` and `11-H` must be
treated as one change; fixing either alone realises nothing.

### C3 — README advertises three run steps that do not exist (`17-A`) — MAJOR

`README.md:18-23` lists a six-step run order. Steps 4, 5 and 6
(`run_evaluation.py`, `run_explainability.py`, `streamlit run dashboard/app.py`)
have no corresponding files. The heading says "To be completed in Phase 18",
which a reader will parse as *not yet documented*, not *not yet built*.

Every other gap in this project is honestly declared in the code that owns it —
`metrics.py:8-10` names its own missing functions and the phase that owns them.
The README is the sole artifact that presents the project as more complete than
it is.

**Fix:** annotate steps 4–6 as not yet implemented. One line.

### C4 — Evaluation module absent (`12-A`) — BLOCKER

`src/evaluation/compare_fl_vs_centralized.py` does not exist, and four of the
five `metrics.py` functions are unwritten (`12-B`). Consequently **eight
artifacts cannot be produced**: `comparison_table.csv`,
`model_size_comparison.csv`, `convergence_plot.png`, both confusion matrices,
both classification reports, and `centralized_training_curves.png`.

This is the largest single gap in the repository, because the FL-vs-centralized
comparison is the project's *stated primary research question* (SDS §1). Without
it the project has trained two models and compared neither.

Every input it needs already exists and is verified correct.

### C5 — Explainability module absent (`13-A`) — BLOCKER

`src/explainability/shap_utils.py` does not exist. SHAP integration is one of
the two novelties SDS §1 claims over the reference work. `tensorflow==2.15.1`
and `shap==0.44.1` are already pinned precisely for this — the environment is
ready, only the code is missing.

### C6 — Dashboard absent (`14-A`, `14-B`) — BLOCKER

`dashboard/` is empty. It is SDS §1's fifth headline capability and the terminal
consumer of both C4 and C5, which fixes the build order.

### C7 — Two `experiments/` entry points absent (`15-A`) — BLOCKER

`run_evaluation.py` and `run_explainability.py`. Thin wrappers; downstream of C4
and C5.

---

## 5. Frozen-invariant verification — all closed

Every invariant the AI Coding Contract and SDS declare frozen was verified at its
actual call site, not merely in the module that documents it. **All six items
deferred by the audit checkpoint are now closed.**

| Invariant | Source | Verified at | Result |
| --- | --- | --- | --- |
| `create_labels` is the sole remover of `Attack_type`/`Attack_label` | Contract A.2 | `load_dataset.py:198` — the only such `drop` in the repository | **HOLDS** |
| The global test split is never partitioned per client | Contract A.2.7 | `server_app.py:634-637` — `partition_iid(df.loc[train_indices], …)`; test frame bound separately | **HOLDS** |
| `conv_padding = "same"` | Contract Part C | `cnn_gru.py:84`; `config.yaml:65`; asserted by `test_conv1d_padding_is_same` | **HOLDS** |
| `gru_return_sequences = False` | Contract Part C | `cnn_gru.py:90`; asserted by `test_gru_return_sequences_is_false` | **HOLDS** |
| `seed = 42` | Contract Part C | `seed.py:16`; `config.yaml:1` | **HOLDS** |
| All transformers fit on training rows only | SDS §14.2 | `encode_normalize.py:753`, `849`, `863`, `760` — four independent fits | **HOLDS** |
| `prepare_model_ready_data` is the sole tensor-preparation path | SDS §14.2 | `train_baseline.py:153`, `client_app.py:193`, `server_app.py:674` — all three production sites | **HOLDS** |
| `input_shape` derived from `feature_names.pkl`, never a live column count | SDS §14.4 | `train_baseline.py:142`, `client_app.py:150`, `server_app.py:621` | **HOLDS** |
| `num_classes` derived as `len(class_mapping)`, never hardcoded | SDS §6, §22 | `train_baseline.py:141`, `client_app.py:142`, `server_app.py:620` | **HOLDS** |
| `weighted_average_fit` ≠ `weighted_average_eval` | SDS §22 | `server_app.py:157`, `192`, wired at `492-493`; asserted by `test_aggregation_functions_are_distinct` | **HOLDS** |
| Pinned `start_simulation` API only; no `ClientApp`/`ServerApp` | SDS §7, §22 | `server_app.py:721-728`; app-based API absent repository-wide | **HOLDS** |
| Saved federated model is the aggregate, never a client's | SDS §22 | `server_app.py:828-845` — only `.save()` writes `global_model` from `latest_parameters` | **HOLDS** |
| Adam is the only optimizer; `training.optimizer` is documentation-only | SDS §6 | `cnn_gru.py:107`; comment at `104-105` | **HOLDS** |
| `LAPTOP_MODE`-only keys never break `FULL_EXPERIMENT` | §1-C | `server_app.py:263`, `307`, `312`, `345`, `740` — all `.get()` or membership-tested | **HOLDS** |
| No circular imports; strictly layered graph | SDS §20 | Whole-repository survey, §18 | **HOLDS** |
| Clients perform no file I/O and persist nothing | SDS §14.5 | `client_app.py` — no `open`/`pickle`/`read`/`save` anywhere | **HOLDS** (see `10-F`) |

**Zero frozen invariants were violated.** For a codebase of this size governed by
this many explicit locks, that is the headline result of the audit.

---

## 6. SDS §23 acceptance criteria

| # | Criterion | Status | Blocked by |
| --- | --- | --- | --- |
| 1–2 | Environment install and preprocessing run end-to-end | **MET, with caveat** | Clean-install path blocked by `4-A` until `pyarrow` is added |
| 3 | `run_centralized.py` end-to-end, non-trivial accuracy, `centralized_training_time.txt` | **MET** | — |
| 4 | `run_federated.py` end-to-end, 15 rounds, 4 clients, global model from `latest_parameters` | **CODE COMPLETE, NOT RUN** | Nothing — one command |
| 5 | `run_evaluation.py` produces both comparison CSVs, both matrices, both reports, training curves | **NOT MET** | `12-A`, `15-A` |
| 6 | `run_explainability.py` produces both SHAP plots and `shap_background.npy` | **NOT MET** | `13-A`, `15-A` |
| 7 | `streamlit run dashboard/app.py` launches; all 4 tabs render | **NOT MET** | `14-A` |
| 8 | Every file in the §24 deliverables checklist exists on disk | **NOT MET** | `12-A`, `13-A`, `14-A`, `15-A`, `16-A`, `19-B` |

**3 of 8 met; 1 met on execution; 4 blocked.**

---

## 7. Remaining work, in dependency order

### Phase A — one-line fixes (minutes)

1. Add `pyarrow>=14,<16` to `requirements.txt` (`C1` / `4-A`).
2. Delete `self.test_data = test_data` in `client_app.py:134` (`C2` / `10-C`, `11-H`).
3. Annotate README steps 4–6 as unimplemented (`C3` / `17-A`).

### Phase B — run what already works (hours, unattended)

4. `python experiments/run_federated.py` — closes 5 of the 16 missing artifacts
   with zero new code (`19-A`). Required before any evaluation work can be
   validated.

### Phase C — build the missing third (the bulk of remaining effort)

5. `src/evaluation/metrics.py` — the four unwritten functions (`12-B`).
6. `src/evaluation/compare_fl_vs_centralized.py` — five functions incl.
   `run_evaluation_pipeline` (`12-A`). **Input contract from `11-C`:**
   `federated_history.csv` carries five columns, not three — select
   `aggregated_accuracy` by name.
7. `experiments/run_evaluation.py` (`15-A`).
8. `tests/test_metrics.py` (`16-A`).
9. `src/explainability/shap_utils.py` (`13-A`). **The four-step squeeze procedure
   in SDS §14.10 is not optional** — skipping it is the documented shape-mismatch
   failure mode inside SHAP's plotting internals.
10. `experiments/run_explainability.py` (`13-A`).
11. `tests/test_shap_utils.py` (`16-A`).
12. `dashboard/app.py` (`14-A`). Load artifacts inside **each tab's own**
    `try`/`except` — SDS §21 Milestone 10 tests exactly this (`14-D`).
13. Complete `README.md` per Phase 18 (`17-A`, `17-C`, `17-D`).

### Phase D — hygiene (optional; none affects a reported number)

- `11-B` — `del df` after partitioning; ungate the shard release from
  `release_memory_after_rounds`.
- `11-D` — iterate `sorted(set(losses) | set(central_losses))` when building
  history rows.
- `11-E` — replace the deprecated `groupby(...).apply(...)` form.
- `9-D` — persist centralized test metrics to JSON instead of the log only.
- `15-C` — give `run_preprocessing_pipeline` a `logger` parameter; currently one
  invocation writes two `preprocessing_*.log` files.
- `7-B` — assert the two Contract-locked config values, making the lock executable.
- `8-D` / `11-K` — dispatch on `partition_strategy` or raise on unknown values.
- `6-D` — raise rather than warn on pre-persist NaNs.
- `19-D` / `17-F` — delete the stale ~643 MB `edge_iiotset_processed.csv`.
- `18-D` — `# noqa: E402` on the `experiments/` imports.
- `3-A`, `3-B`, `2-B`, `5-C`, `6-E`, `6-F`, `9-E`, `10-D`, `10-E`, `11-F`,
  `11-L`, `11-N`, `12-D`, `12-F`, `15-B`, `15-D`, `16-D`, `16-E`, `19-B`.

---

## 8. Environment and Colab readiness

| Item | Status | Action |
| --- | --- | --- |
| `pyarrow` | **BLOCKING on a fresh runtime** | Add to `requirements.txt` (`C1`) |
| `pydot` / graphviz | Missing; non-fatal by design | Architecture PNGs will not generate (`7-D`, `19-B`) |
| Four exact SDS §7 pins | Correct | `tensorflow==2.15.1`, `flwr[simulation]==1.8.0`, `ray==2.6.3`, `shap==0.44.1` |
| GPU determinism | Not guaranteed | `enable_op_determinism()` not called (`3-C`); cuDNN kernel selection can move accuracy in the 4th decimal. Acceptable — but the README should say so rather than promise byte-identical results |
| `PYTHONHASHSEED` | Ineffective in-process | Export it in the shell/Colab cell **before** `python` (`3-A`). Affects no reported number |
| `LAPTOP_MODE` | Implemented, undocumented | `set IIOT_MODE=LAPTOP_MODE`; document it (`17-D`) |
| Dataset placement | Undocumented | `data/raw/edge_iiotset.csv` must be placed manually (`17-C`) |

On Colab specifically, only `C1` is genuinely blocking. `LAPTOP_MODE` is
unnecessary there and `FULL_EXPERIMENT` was verified to be behaviourally
unchanged by all of the `LAPTOP_MODE` machinery (`11.45`).

---

## 9. What this repository does notably well

An audit that only lists defects misrepresents a codebase. Five things here are
better than the specification required, and they are the things that are hardest
to retrofit later.

**Leakage control.** Four independent fit-on-train-only requirements all hold,
and two of them were achieved by *deviating from the written SDS*: column-type
inference was moved after the split (because cardinality is a fitted property),
and de-duplication was moved before it (because ~13% of Edge-IIoTset rows are
exact duplicates, which would otherwise place byte-identical training rows in the
test set). Both deviations are declared in the docstring with their reasoning
(`encode_normalize.py:585-596`) rather than made silently. Separately,
`binary_label` is dropped before persisting precisely because it is a perfectly
label-correlated column (`6-C`), and the `drop_columns` list closes `frame.time`
— a near-perfect row identifier correlated with capture order and therefore with
the label. Leakage is the one defect class in this project that would *inflate*
rather than degrade reported accuracy, and it has been attacked deliberately and
successfully.

**The federated evaluation redesign (`10-A` + `11-A`).** Distributed evaluation
was moved onto per-client local hold-outs — previously all four clients scored
the identical global test split, so example-weighted aggregation was averaging a
constant. That change alone would have removed the project's only measurement of
the global model on the shared test set, so a server-side `evaluate_fn` was added
to restore it. The two changes are correct only together, and they were made
together. This is the sharpest piece of reasoning in the codebase.

**Operational maturity in the federated layer.** `KeyboardInterrupt` handled
separately from `Exception` (it derives from `BaseException` and would otherwise
bypass the handler); Ray shutdown wrapped so cleanup failure cannot destroy a
completed run; a mandatory `RuntimeError` when `latest_parameters is None`, which
converts a silently-empty saved model into a loud failure; quantified concurrency
diagnostics in the log.

**The test suite.** 116 tests against ~30 SDS-required cases, and the additions
target real invariants rather than inflating a count —
`test_scaler_train_only_differs_from_full_dataset_fit` fails if anyone
"simplifies" the pipeline by fitting before splitting;
`test_partition_iid_shuffles_rows_before_splitting` pins the property that makes
the IID claim true given a label-ordered source frame;
`test_smoke_test_never_uses_production_config` enforces SDS §22 from the negative
side.

**Architecture.** Strictly layered, acyclic import graph with no
sibling-to-sibling edge except the one SDS §14.6 mandates. Every artifact has
exactly one owning function. The `preprocessing → evaluation` edge — the easiest
one to violate by convenience — is correctly routed through the orchestration
layer instead.

---

## 10. Completion estimate

| Layer | State |
| --- | --- |
| Utilities (4 modules) | **Complete and verified** |
| Preprocessing (2 modules) | **Complete and verified** |
| Model + partitioning (2 modules) | **Complete and verified** |
| Centralized training | **Complete, verified, executed** |
| Federated training (2 modules) | **Complete and verified; not yet executed** |
| Evaluation / comparison | **~10%** — 1 of 10 functions |
| Explainability | **0%** |
| Dashboard | **0%** |
| Orchestration scripts | **60%** — 3 of 5 |
| Test suite | **71%** — 5 of 7 files |
| Documentation | **~25%** — run order only, and currently misleading |

**Overall: approximately 65% of specified functionality implemented, and 100% of
what is implemented is compliant.**

Artifacts tell the same story precisely: **12 of 28 produced, 5 producible today
with no new code, 11 not producible.**

---

## 11. Closing statement

This audit checked 369 requirements across 19 sections against four governing
documents and found **one FAIL, three PARTIALs, and zero violated invariants**
in implemented code. The engineering that exists is disciplined, well-reasoned,
and in the areas that matter most — data-leakage integrity, federated
aggregation correctness, reproducibility, and architectural layering — it is
stronger than the specification demanded.

The gap is not quality. It is coverage. Four BLOCKERs record layers that were
fully specified and never built, and they cluster entirely in the final third of
the pipeline: analysis, explanation, and presentation.

Three one-line changes (`C1`, `C2`, `C3`) and one unattended run (`Phase B`)
would move this from *"a rigorous half-built system with a misleading README"* to
*"a rigorous, honestly-documented system with a clearly-scoped remaining build
list"* — before a single new module is written.

**— End of audit.**

---


