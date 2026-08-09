"""SHAP explainability for the IIoT Federated IDS project.

This module owns SDS Section 14.10 in full. It is the sole producer of
``outputs/shap_plots/shap_summary_plot.png``,
``outputs/shap_plots/shap_bar_plot.png`` and
``outputs/artifacts/shap_background.npy``.

Two fixed procedures are enforced here because both are documented failure
modes rather than style preferences:

  1. **The four-step squeeze (SDS Section 14.10) is not optional.**
     ``shap.GradientExplainer`` on a ``(n, num_features, 1)`` input returns
     per-class arrays of the same 3D shape.  ``shap.summary_plot`` expects 2D
     ``(n, num_features)`` and fails deep inside its own plotting internals if
     handed 3D input.  Both the SHAP values and the sample array are therefore
     squeezed on the trailing channel axis before any plotting call.

  2. **Both random draws are seeded, in a fixed order.**  A single
     ``numpy.random.RandomState(seed)`` is constructed and used for two
     successive ``.choice(...)`` calls — background first, test second.  The
     unseeded global random state is never used, so the persisted background
     array is byte-identical across runs (SDS Section 23, criterion 9).

The background array persisted to disk is the exact array used to construct the
explainer that produces the two plots, so ``dashboard/app.py`` explaining a
single prediction from the persisted file is explaining against the same
reference distribution as the global plots.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow


# Module-level fallback only. The real logger is always injected by
# experiments/run_explainability.py via src.utils.logger.get_logger (SDS §13).
_logger = logging.getLogger(__name__)


def generate_shap_explanations(
    model: tensorflow.keras.Model,
    X_train: np.ndarray,
    test_samples_full: np.ndarray,
    class_names: list,
    output_dir: str,
    background_output_path: str,
    seed: int,
    background_size: int,
    sample_size: int,
    feature_names: list = None,
    logger: logging.Logger = None,
) -> list:
    """Compute SHAP values, save both plots, and persist the background sample.

    Draws a seeded background set from the training tensor and a seeded set of
    test instances, computes SHAP values via ``shap.GradientExplainer``, saves
    the summary (beeswarm) and bar plots, and persists the background array for
    reuse by the dashboard.

    Args:
        model: The trained Keras model being explained (the federated global
            model, per SDS Section 21 Milestone 9).
        X_train: Full training tensor, shape ``(n_train, num_features, 1)``,
            produced via ``prepare_model_ready_data``.
        test_samples_full: Full test tensor, shape
            ``(n_test, num_features, 1)``, produced via
            ``prepare_model_ready_data``.
        class_names: Ordered list of class names, indexed by class integer.
        output_dir: Directory for the two plots
            (``outputs/shap_plots``).
        background_output_path: Path for the persisted background array
            (``outputs/artifacts/shap_background.npy``).
        seed: Global project seed, used to construct the single
            ``RandomState`` driving both draws.
        background_size: ``config["explainability"]["background_size"]``.
        sample_size: ``config["explainability"]["sample_size"]``.
        feature_names: Ordered feature names from ``feature_names.pkl``, passed
            to ``shap.summary_plot``.  Omitted only in tests.
        logger: Optional logger injected by the orchestrator.

    Returns:
        list[numpy.ndarray]: Per-class SHAP values in their original
            (pre-squeeze) 3D shape, so callers other than the plotting routines
            still receive shapes consistent with the model's input.

    Raises:
        Propagates any SHAP runtime exception after logging at ERROR level.

    Dependencies:
        shap.GradientExplainer, matplotlib, numpy, numpy.random.RandomState.
    """
    log = logger or _logger

    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(
            os.path.dirname(os.path.abspath(background_output_path)),
            exist_ok=True,
        )

        # ---- Seeded sampling: one RandomState, two draws, fixed order ----
        # Background first, test second (SDS §14.10). The order is part of the
        # contract: swapping the two calls changes both selections.
        random_state = np.random.RandomState(seed)

        effective_background = min(background_size, len(X_train))
        background_indices = random_state.choice(
            len(X_train), size=effective_background, replace=False
        )
        background = X_train[background_indices]

        effective_sample = min(sample_size, len(test_samples_full))
        test_indices = random_state.choice(
            len(test_samples_full), size=effective_sample, replace=False
        )
        test_samples = test_samples_full[test_indices]

        log.info(
            "SHAP sampling — background=%d of %d train rows | "
            "explained=%d of %d test rows | seed=%s",
            effective_background,
            len(X_train),
            effective_sample,
            len(test_samples_full),
            seed,
        )

        # Persist the background BEFORE building the explainer, so the file on
        # disk and the explainer's reference distribution cannot diverge.
        np.save(background_output_path, background)
        log.info(
            "Saved %s — shape=%s",
            background_output_path,
            background.shape,
        )

        # ---- Compute SHAP values -----------------------------------------
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_samples)

        # SHAP compatibility:
        # Older SHAP returns a list of arrays.
        # Newer SHAP returns a single ndarray.

        log.info(
            "SHAP values computed — %d per-class arrays, each %s",
            len(shap_values),
            np.asarray(shap_values[0]).shape,
        )

        # ---- Normalize SHAP output -------------------------------------
        if isinstance(shap_values, list):
            shap_values_2d = [
                np.asarray(sv).squeeze(axis=-1)
                for sv in shap_values
            ]
        else:
            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 4:
                shap_values = shap_values.squeeze(axis=2)

            shap_values_2d = [
                shap_values[:, :, i]
                for i in range(shap_values.shape[-1])
            ]

        test_samples_2d = test_samples.squeeze(axis=-1)


        log.info(
            "Squeezed for plotting — shap_values_2d[0]=%s | test_samples_2d=%s",
            shap_values_2d[0].shape,
            test_samples_2d.shape,
        )

        # ---- Step 3: summary (beeswarm) plot -----------------------------
        summary_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.figure()
        shap.summary_plot(
            shap_values_2d,
            test_samples_2d,
            feature_names=feature_names,
            class_names=list(class_names),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Saved %s", summary_path)

        # ---- Step 4: bar plot of mean absolute SHAP per feature ----------
        # Mean over classes first, then over samples, leaving one value per
        # feature. Reshaped to (1, num_features) so summary_plot's bar mode
        # receives the 2D structure it expects.
        mean_abs_per_class = np.mean(
            [np.abs(sv) for sv in shap_values_2d], axis=0
        )
        mean_abs_per_feature = np.mean(mean_abs_per_class, axis=0)

        bar_path = os.path.join(output_dir, "shap_bar_plot.png")
        plt.figure()
        shap.summary_plot(
            mean_abs_per_feature.reshape(1, -1),
            features=np.zeros((1, mean_abs_per_feature.shape[0])),
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Saved %s", bar_path)

        # Returned in the original 3D shape (SDS §14.10) — the squeeze exists
        # for the plotting calls only.
        return shap_values

    except Exception:
        log.error(
            "generate_shap_explanations failed — full traceback:",
            exc_info=True,
        )
        raise


def explain_single_prediction(
    model: tensorflow.keras.Model,
    explainer: "shap.GradientExplainer",
    sample: np.ndarray,
) -> list:
    """Compute SHAP values for exactly one sample, ready for bar rendering.

    Serves the dashboard's per-prediction explanation view, applying the same
    squeeze steps as ``generate_shap_explanations`` so the caller receives a 2D
    per-class structure aligned with ``(num_features,)``.

    Never re-samples or regenerates the background distribution: the caller
    (``dashboard/app.py``) loads the persisted ``shap_background.npy`` and
    constructs ``explainer`` from it before calling here (SDS §14.10 / §18).

    Args:
        model: The trained Keras model being explained.  Retained for API
            symmetry with ``generate_shap_explanations``; the explainer already
            holds its own reference to the model.
        explainer: A ``shap.GradientExplainer`` constructed by the caller from
            the persisted background array and ``model``.
        sample: A single model input, shape ``(1, num_features, 1)``.

    Returns:
        list[numpy.ndarray]: Per-class SHAP values squeezed to
            ``(1, num_features)``, consistent with
            ``generate_shap_explanations``'s plotting representation.

    Raises:
        Propagates any SHAP runtime exception.
    """
    shap_values = explainer.shap_values(sample)

    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    # Same squeeze as the plotting path, so the dashboard never has to know
    # about the trailing channel dimension.
    return [np.asarray(sv).squeeze(axis=-1) for sv in shap_values]


def get_top_feature_contributions(
    shap_values: list,
    sample: np.ndarray,
    feature_names: list,
    predicted_class_index: int,
    top_n: int = 15,
) -> "list":
    """Rank the features that drove one prediction, for the class predicted.

    Turns the raw per-class SHAP output into the table the dashboard shows
    under "Why did the model make this prediction?".

    The ranking comes from the row belonging to the **predicted class**, not
    the mean across all classes. A mean-across-classes ranking answers "which
    features does this model generally react to here", whereas the question
    asked is "why *this* class". Only the predicted class's row carries signed
    evidence for the answer given — positive values pushed toward that class,
    negative against — and averaging absolute values across 15 classes destroys
    exactly that sign information. Rows are ordered by absolute contribution so
    the strongest evidence appears first, with the signed value preserved.

    Args:
        shap_values: The per-class list returned by
            :func:`explain_single_prediction`, each entry shaped
            ``(1, num_features)``.
        sample: The single model input that was explained, shape
            ``(1, num_features, 1)`` — the same array that was predicted on.
        feature_names: Ordered feature names from ``feature_names.pkl``.
        predicted_class_index: Index of the class the model predicted, used to
            select which per-class SHAP row explains the answer.
        top_n: Number of features to return.

    Returns:
        list[dict]: ``top_n`` dicts, ordered by descending absolute
            contribution, each with ``feature``, ``feature_value``,
            ``shap_value``, ``abs_shap_value`` and ``direction``
            (``"increases"`` / ``"decreases"``).

    Raises:
        IndexError: If ``predicted_class_index`` is outside ``shap_values``.
        ValueError: If the SHAP row and ``feature_names`` disagree in length,
            which would mean the explanation is misaligned with the features it
            claims to name.
    """
    if predicted_class_index < 0 or predicted_class_index >= len(shap_values):
        raise IndexError(
            f"predicted_class_index {predicted_class_index} is outside the "
            f"{len(shap_values)} per-class SHAP arrays returned."
        )

    # Signed contributions for the predicted class only.
    class_contributions = np.asarray(
        shap_values[predicted_class_index]
    ).reshape(-1)

    if len(class_contributions) != len(feature_names):
        raise ValueError(
            f"SHAP values describe {len(class_contributions)} feature(s) but "
            f"{len(feature_names)} feature name(s) were supplied; the "
            f"explanation would be mislabelled."
        )

    feature_values = np.asarray(sample).reshape(-1)

    absolute = np.abs(class_contributions)
    order = np.argsort(absolute)[::-1][: min(top_n, len(absolute))]

    return [
        {
            "feature": feature_names[int(position)],
            "feature_value": float(feature_values[int(position)]),
            "shap_value": float(class_contributions[int(position)]),
            "abs_shap_value": float(absolute[int(position)]),
            "direction": (
                "increases"
                if class_contributions[int(position)] >= 0
                else "decreases"
            ),
        }
        for position in order
    ]
