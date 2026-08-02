# skills.md — IIoT Federated Intrusion Detection Project
## Persistent Implementation Skill for AI Coding Agents

**Applies to:** Claude Code, Antigravity, Cursor, Codex, Gemini CLI, or any equivalent coding agent working in this repository.

**Purpose of this file:** This is not the specification. This is a compact, reusable *behavioral contract* that tells any coding agent how to act in this repo, without re-sending the full design documents every session. Read this file first, every session, before writing any code.

---

## 0. Source of Truth (Read Before Any Work)

This project has exactly three governing documents. None of their content is duplicated here — this file only tells you *how to use them*.

| Document | Location | Answers |
|---|---|---|
| **SDS** (Software Design Specification, Rev 2.1) | `project-docs/SDS.md` | *What* to build — architecture, function contracts, folder structure, config schema, exact algorithms, acceptance criteria. |
| **Phase Implementation Prompts v1** | `project-docs/Phase_Implementation_Prompts.md` (or wherever stored) | *In what order* — the 18 phases and their dependency sequence. |
| **AI Coding Contract** | `project-docs/AI_Coding_Contract.md` | *Who may touch what, when* — per-phase file permissions, import rules, forbidden actions. |

**Rule:** If this `skills.md` file and the SDS/Contract ever appear to disagree, the SDS and Contract win. This file only encodes *behavior*, never *design*. If you find yourself inferring an architectural detail not present in the SDS, stop and ask — do not invent it.

**Before starting any task:** Confirm which phase (1–18) you are implementing. If not told, ask. Never guess the phase from context clues.

---

## 1. Prime Directives (Non-Negotiable, Apply to Every Task)

1. **Never redesign the architecture.** The SDS is frozen. You do not have permission to improve, simplify, refactor, or "modernize" any architectural decision — not the model, not the framework, not the folder layout, not the data flow.
2. **Never rename files.** Filenames in the SDS folder tree (Section 9) are exact and final.
3. **Never rename folders.** Directory structure in the SDS is exact and final.
4. **Never rename functions.** Function names in SDS Section 14 are exact and final — same for arguments, argument order, and return types.
5. **Never replace libraries.** The pinned stack (SDS Section 7 / Contract Section 7) is exact. No substituting PyTorch for TensorFlow, Dash for Streamlit, a different Flower API, a different SHAP explainer, etc. — regardless of how "equivalent" or "better" an alternative seems.
6. **Never duplicate implementations.** If a function already owns a piece of logic (e.g., `prepare_model_ready_data`, `create_labels`, `infer_feature_column_types`, any plot-generation function), every other file calls it — nothing reimplements it, even partially, even "just to be safe."
7. **Never modify a previous phase's files without explicit permission.** Each phase in the Contract has an exhaustive file allow-list. Anything not on that list — including files from earlier completed phases — is off-limits this session, even if you spot a bug in it. Flag it instead; do not fix it silently.
8. **Implement only the requested phase.** Do not pre-build later phases "while you're at it." Do not stub out future functions beyond what the current phase's contract requires.
9. **Stop after completion.** When the current phase's validation gate passes, stop. Do not continue to the next phase. Do not start cleanup, refactors, or "nice to have" additions outside the phase's scope.

If any instruction from the user conflicts with directives 1–9, treat the conflict as a stop condition (Section 9 below) — do not silently resolve it in either direction.

---

## 2. Session Startup Checklist

Every session, before writing code:

1. Identify the current phase number (ask if not stated).
2. Open the Contract's per-phase entry for that phase number. This is your exhaustive permission set for this session: files to create, files to modify, files to read, forbidden files, allowed imports, forbidden imports.
3. Open the SDS sections referenced by that phase's function contracts (Section 14.x) — read only what's relevant, not the whole document, to conserve tokens.
4. Confirm the phase's `Required Previous Phases` have already passed their Validation Gate. If unsure whether a prior phase is complete, ask — do not assume and do not re-verify by re-implementing.
5. Do not read or load unrelated phases' contract entries or SDS sections unless a function contract explicitly cross-references them.

---

## 3. Project Invariants (Never Change, Any Phase)

These hold for the entire project lifetime — see Contract Part A for full detail; this is the compressed operational summary:

- **Architecture is frozen.** CNN-GRU, Flower `flwr==1.8.0` (client-fn/strategy `start_simulation` API only — never the app-based `ClientApp`/`ServerApp`/`run_simulation` API), FedAvg, Edge-IIoTset, SHAP `GradientExplainer`, Streamlit. No substitutions, ever.
- **`data/raw/edge_iiotset.csv`** is never written, modified, or regenerated by any script, in any phase.
- **Native columns `Attack_type` / `Attack_label`** are removed exclusively inside `create_labels` — no other function drops either column, ever.
- **`label` / `binary_label`** are derived exactly once, during preprocessing. Nothing downstream re-derives them.
- **No literals in `src/`.** Every hyperparameter, path, and class count comes from `configs/config.yaml` or is derived at runtime (`num_classes = len(class_mapping)`). Never hardcode a class count, a learning rate, a file path, etc.
- **All multi-component paths** use `os.path.join(...)`. Never string-concatenate paths with `/` or `\`.
- **Encoders/scalers/label encoders** are fit only on `df.loc[train_indices]`. Fitting on the full dataset before the split is a leakage defect, no matter which phase or agent introduces it.
- **The train/test split** is computed once, persisted (`train_indices.pkl`, `test_indices.pkl`), and reused everywhere. Nothing re-splits.
- **Federated client shards** are drawn only from `train_indices` rows. The shared test split is never partitioned per client.
- **Exactly one model architecture** (`build_cnn_gru`), exactly as specified in SDS Section 14.4 — same function builds both the centralized and the reconstructed federated global model.
- **`Conv1D.padding` is always `"same"`; `GRU.return_sequences` is always `False`.** Not configurable, ever.
- **Adam is the only optimizer** ever instantiated. `config["training"]["optimizer"]` is documentation-only.
- **`centralized_best_model.h5`** (lowest `val_loss`) is the only centralized model any other code reads. `centralized_last_model.h5` is audit-only, never loaded elsewhere.
- **`SavingFedAvg`** is the only strategy class used for production runs — never a plain `FedAvg`.
- **`weighted_average_fit`** and **`weighted_average_eval`** are two distinct functions, never merged, never swapped between `fit_metrics_aggregation_fn` / `evaluate_metrics_aggregation_fn`.
- **Production federated runs** use `num_clients = min_available_clients = min_fit_clients = min_evaluate_clients = 4`. Only `tests/test_federated_loop.py` uses a reduced value (2), via its own independently constructed config copy — it never reads or mutates `configs/config.yaml`.
- **Test-set loss** is always sourced from `model.evaluate()` — never independently recomputed from predictions.
- **Each of the 8 named metrics/plot functions** (Contract A.6.2) has exactly one implementation, one owning file. No second implementation anywhere, including inside `dashboard/app.py`.
- **`dashboard/app.py` never trains, never partitions, never preprocesses, never resamples SHAP.** Strictly read-only against `outputs/` and `data/processed/`.
- **Every artifact** has exactly one producer function and one fixed filename (SDS Section 9). No alternate filenames, no alternate locations, no second producer.
- **`set_global_seed(config["seed"])`** is the first executable statement in every `experiments/*.py` script.
- **`configs/config.yaml`** is edited only by explicit human instruction — never autonomously by an agent mid-implementation.

---

## 4. Implementation Rules

- Match every function signature in SDS Section 14 **exactly** — name, argument names, argument order, types, return type. Do not rename `df` to `data`, `seed` to `random_state`, etc.
- Full type hints on every public function. Google-style project-docstrings (Purpose, Args, Returns, Raises) on every public function.
- PEP8-conformant code.
- No global mutable state. Pass config and state explicitly as arguments/return values.
- Single Responsibility: one function does one job per its SDS contract. Orchestration (multi-function sequencing) belongs only in `run_*` functions or `experiments/` scripts — never inside a low-level function.
- No circular imports. Respect the import direction graph in SDS Section 11 / Contract Part A.1 exactly (e.g., `src/models/` never imports from `src/centralized/` or `src/federated/`).
- Every config key present in `config.yaml` must be read by at least one function. Never add an unused key.
- If a decision point in the SDS could be read two ways, take the most literal, most restrictive interpretation. If still ambiguous, stop and ask rather than guessing.

---

## 5. Forbidden Actions (Apply at All Times, All Phases)

Do not, under any circumstances, unless a future human-authorized SDS revision says otherwise:

- Substitute LSTM for GRU, or any other layer swap.
- Add PCC/correlation-based feature selection.
- Add Logistic Regression or any second model family.
- Add Docker, Kubernetes, or any containerization.
- Implement Non-IID partitioning.
- Add a SQL/NoSQL database — all persistence is flat files (`.csv`, `.pkl`, `.json`, `.h5`, `.png`, `.txt`, `.npy`) exactly as named in SDS Section 9.
- Compute/display ROC curves unless explicitly asked (out of default scope).
- Hardcode `num_classes` (or any other class-count-shaped literal).
- Fit any encoder/scaler on anything beyond the training partition.
- Upgrade or downgrade `tensorflow`, `flwr`, `ray`, or `shap` beyond their exact pins.
- Use Flower's app-based `ClientApp`/`ServerApp`/`run_simulation` API.
- Save a per-client local model as `federated_global_model.h5`.
- Treat `config["training"]["optimizer"]` as a live switch.
- Reimplement `prepare_model_ready_data`, `infer_feature_column_types`, or any named single-owner function at a second call site.
- Let `dashboard/app.py` resample a SHAP background set, or import any training/partitioning/preprocessing-execution function.
- Touch any file outside the current phase's exhaustive Contract allow-list — including files owned by earlier or later phases.
- Add an 8th top-level test file, or rename/merge/split any of the 7 named test files, without the Contract being amended first by a human.
- Introduce a separate addendum, patch note, or "supplementary design decision" document. The SDS + Contract + Blueprint + this file are the complete authoritative set.

---

## 6. Mandatory Validation Rules

Before declaring any phase complete:

- Re-read the Contract's per-phase `Validation required` and `Acceptance criteria` fields and check each one explicitly — do not assume "the code runs" equals "the phase is validated."
- Confirm every artifact the phase is supposed to produce actually exists on disk, in the exact path SDS Section 9 specifies.
- Confirm no file outside the phase's allow-list was created, modified, or deleted (self-audit before finishing).
- Confirm no forbidden import was added (grep-check against the phase's `Imports forbidden` list).
- If validation fails, fix within the current phase's scope only, then re-validate. Do not silently patch a downstream phase's file to "make this phase's test pass."

---

## 7. Mandatory Testing Rules

- The 7 test files named in SDS Section 19 are the complete test suite — never rename, merge, split, or add an 8th without explicit human sign-off.
- Every phase that owns a test file must implement every test case named for that file in SDS Section 19's table — cross-check line by line, not "pytest passes so we're done."
- `tests/test_federated_loop.py` always builds its own independent, reduced-scale config copy (`num_clients = min_*_clients = 2`). It never reads or mutates `configs/config.yaml`.
- Run the tests owned by the current phase before declaring the phase done. Only from Phase 17 onward does the full `pytest tests/` (all 7 files together) need to exit 0 — earlier phases run only their own relevant test files.
- If a test failure traces back to a defect in an earlier phase's file, do not patch that file yourself unless the current phase's Contract entry explicitly allows modifying it (this is rare — usually only true for Phases 16–17's "gap-fill" test phases, and even then only test files, not `src/`). Otherwise, report the defect and stop.

---

## 8. Logging Requirements (Compressed — Full Detail in SDS Section 13)

- Every `experiments/*.py` script gets its logger via `src/utils/logger.py::get_logger(name, logs_dir)`, using the **fixed literal name** from the SDS Section 13 naming table (`"preprocessing"`, `"centralized_training"`, `"federated_training"`, `"evaluation"`, `"explainability"`) — never `__name__`, never an invented name.
- Modules under `src/` receive an already-constructed logger passed down from the calling `experiments/run_*.py` script. They never construct their own.
- Log files are never overwritten — every run appends a new timestamped file.
- Minimum required log events per script: start (config summary), dataset/artifact load stats, model build (`model.summary()` at DEBUG), per-epoch/per-round metrics, training completion (time + best metric + save path), evaluation completion (all metrics + report/plot paths), any exception (full stack trace at ERROR, then re-raised — never swallowed), script end (duration).

---

## 9. Configuration Rules (Compressed — Full Schema in SDS Section 8)

- `configs/config.yaml` is the single source of every configurable value. No script defines a hyperparameter, path, or magic number inline.
- Every script loads config via `src/utils/config_loader.py::load_config()` — never reads the YAML file directly elsewhere.
- Do not add, remove, or rename any config key without explicit human instruction — this includes "helpfully" adding a new key you think would be useful.
- Do not modify `configs/config.yaml`'s values autonomously to make a failing phase pass. If a phase seems to require a config change, stop and flag it — this is a signal the task or the SDS needs human review, not that you should edit the config.

---

## 10. Artifact Ownership (Compressed — Full Table in SDS Section 9 / Contract A.8)

- Every artifact (model, plot, CSV, pickle, JSON, log) has exactly **one** producer function and **one** fixed filename/location. Before writing any file, confirm via the Contract which phase and which function owns it — if it's not this phase's job to produce it, don't produce it, even as a "bonus."
- Respect the overwrite rules exactly: `data/raw/` never overwritten by code; `data/processed/` and `outputs/artifacts/` skip regeneration unless `force_reprocess: true`; `outputs/models/` and `outputs/results/` overwritten on each relevant run; `logs/` never overwritten, always a new timestamped file.
- `shap_background.npy` is written only by `generate_shap_explanations` and read only by scripts building a `GradientExplainer` from it — never regenerated by a reader (including the dashboard).

---

## 11. Git Workflow

- One phase = one commit (or one small series of logically-scoped commits), never a mixed commit spanning multiple phases.
- Commit message format: `Phase <N>: <short description>` — e.g., `Phase 6: Implement centralized baseline training`.
- Do not commit partial/broken state as a phase's final commit — the phase's Validation Gate must pass first.
- Do not rebase, squash, or rewrite history from a previous phase's commits without explicit human instruction.
- Do not commit generated artifacts that are reproducible from code + config (e.g., trained `.h5` models, logs) unless the project's `.gitignore` policy says otherwise — check for a `.gitignore` before committing binary outputs; if none exists, ask rather than guessing what should be tracked.
- Never force-push over another contributor's (human or agent) commits.
- If asked to implement a phase, assume you're starting from the last commit's state — do not attempt to reconstruct or "fix" earlier phases' commits as part of this task.

---

## 12. Stopping Conditions (When to Halt and Ask, Not Guess)

Stop and explicitly flag the issue to the human, rather than proceeding, whenever:

- A task appears to require modifying a file outside the current phase's Contract allow-list.
- A task appears to require changing a row in the Contract's Part C "Absolute Prohibitions" table (architecture, folder structure, function names, config keys, output filenames, model architecture, seed policy, training pipeline, evaluation pipeline).
- A task appears to require a library substitution or version change outside the pinned stack.
- The current phase's `Required Previous Phases` have not been confirmed complete.
- The SDS, Blueprint, and Contract appear to conflict with each other, or with the current task instructions.
- You would need to invent a design decision not explicitly specified anywhere in the SDS.
- A test failure's root cause lives in a file this phase is not permitted to touch.
- You've completed the current phase's Validation Gate — **stop here even if it would be easy to continue into the next phase.**

When you stop, state plainly: what phase you were on, what specifically blocked you, and what decision or permission you need from the human before continuing. Do not silently work around the blocker.

---

## 13. Token-Efficiency Notes for the Agent

- Do not re-read the entire SDS every session. Read only the specific numbered section(s) the current phase's function contracts reference.
- Do not re-paste or restate SDS/Contract content back to the user as confirmation — reference it by section number instead (e.g., "per SDS Section 14.6").
- Prefer editing existing files over regenerating them wholesale, when only a small change is needed.
- This file (`skills.md`) should be loaded once per session and treated as standing instructions — it does not need to be re-confirmed line by line before each action.

---

*End of skills.md. This file defines behavior only. For design and architecture, consult `project-docs/SDS.md`. For architecture and design decisions, consult:

`project-docs/SDS.md`

For implementation rules and permissions, consult:

`project-docs/AI_Coding_Contract.md`

For implementation order and phase execution, consult:

`project-docs/Phase_Implementation_Prompts.md`