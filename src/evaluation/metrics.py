"""Evaluation metrics and visualization functions for the IIoT project.

Phase 4 scope (created early — explicit cross-boundary exception per
SDS Section 14.2 / Phase 4 allow-list):
  - generate_class_distribution_plot: called by experiments/run_preprocessing.py
    immediately after run_preprocessing_pipeline returns.

Remaining functions (compute_all_metrics, generate_confusion_matrix,
generate_classification_report, generate_training_curves_plot) are
implemented in Phase 8 by extending this file.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests

import matplotlib.pyplot as plt
import pandas as pd


def generate_class_distribution_plot(
    df: pd.DataFrame,
    label_column: str,
    output_path: str,
) -> None:
    """Plot a bar chart of sample counts per attack category and save to disk.

    Purpose:
        Count samples per unique value of ``label_column``, then produce and
        save a horizontal bar chart at ``output_path``.  Called from
        ``experiments/run_preprocessing.py`` immediately after
        ``run_preprocessing_pipeline`` returns.

    Args:
        df: Labeled (and optionally processed) DataFrame containing at minimum
            the ``label_column``.  The in-memory result of the preprocessing
            pipeline, or ``data/processed/edge_iiotset_processed.csv``
            reloaded.
        label_column: Name of the label column to count
            (always ``"label"`` in this project).
        output_path: Absolute or relative path where the PNG is saved
            (``outputs/results/class_distribution.png``).

    Returns:
        None

    Raises:
        KeyError: If ``label_column`` is not present in ``df``.

    Dependencies:
        pandas, matplotlib.
    """
    if label_column not in df.columns:
        raise KeyError(
            f"label_column '{label_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    counts = df[label_column].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)))
    ax.barh(counts.index, counts.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Sample Count", fontsize=12)
    ax.set_title("Class Distribution — Edge-IIoTset", fontsize=14)
    ax.tick_params(axis="y", labelsize=9)

    for i, (val, label) in enumerate(zip(counts.values, counts.index)):
        ax.text(val + max(counts.values) * 0.005, i, str(val),
                va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
