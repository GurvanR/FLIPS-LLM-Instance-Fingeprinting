"""Cross-dataset feature importance plotting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import matplotlib.pyplot as plt
import numpy as np

from audit_llm.plotting.figure_io import save_fig_and_show
from audit_llm.xp_tools import clean_feat_label

if TYPE_CHECKING:
    from audit_llm.Classification.single_classification import SingleTokenPairClassification


def plot_features_importance(
    stored_pipelines: Dict[str, SingleTokenPairClassification],
    classification_config: Dict,
    save_path: Path,
    show: bool = False,
    top_n: int = 25,
    error_bar_method: str = "iqr",
) -> None:
    """Plot mean feature importances across multiple datasets for a given classifier.

    Parameters
    ----------
    stored_pipelines : dict
        Maps dataset names to ``SingleTokenPairClassification`` instances.
    classification_config : dict
        Configuration dictionary (must contain ``"classifiers"`` key).
    save_path : Path
        Directory to save the figure.
    show : bool
        Whether to display the figure interactively.
    top_n : int
        Number of top features to display.
    error_bar_method : str
        ``"clip"`` (clip lower bars at zero) or ``"iqr"`` (interquartile range).
    """
    if not stored_pipelines:
        raise ValueError("No pipelines provided to plot.")

    pipelines_with_imps = {
        tp_name: pipe for tp_name, pipe in stored_pipelines.items() if pipe.feature_importances
    }

    if not pipelines_with_imps:
        raise ValueError(
            "No feature importances available in any pipeline."
            " Run fit_evaluate with importance-capable classifiers."
        )

    for clf_name in classification_config["classifiers"]:
        feature_means = []
        for tp_name, pipe in pipelines_with_imps.items():
            imps = np.vstack(pipe.feature_importances[clf_name])
            feature_means.append(imps.mean(axis=0))

        all_means = np.vstack(feature_means)
        overall_mean = all_means.mean(axis=0)

        top_idx = np.argsort(overall_mean)[-top_n:]
        top_means = overall_mean[top_idx]

        if error_bar_method == "clip":
            overall_std = all_means.std(axis=0)
            top_stds = overall_std[top_idx]
            lower_errors = np.minimum(top_stds, top_means)
            upper_errors = top_stds
            yerr = [lower_errors, upper_errors]

        elif error_bar_method == "iqr":
            q1 = np.percentile(all_means, 25, axis=0)
            q3 = np.percentile(all_means, 75, axis=0)
            top_q1 = q1[top_idx]
            top_q3 = q3[top_idx]
            lower_errors = np.maximum(0, top_means - top_q1)
            upper_errors = np.maximum(0, top_q3 - top_means)
            yerr = [lower_errors, upper_errors]

        else:
            raise ValueError(f"Unknown error_bar_method: {error_bar_method}")

        first_pipe = next(iter(pipelines_with_imps.values()))
        inv_map = {v: k for k, v in first_pipe.features_index.items()}
        labels = [inv_map.get(idx, f"Feature {idx}") for idx in top_idx]
        labels = [clean_feat_label(label) for label in labels]

        plt.figure(figsize=(8, 5))
        x_positions = np.arange(len(labels))
        plt.bar(x_positions, top_means, yerr=yerr, align="center", capsize=3)
        plt.xticks(x_positions, labels, rotation=45, ha="right")
        plt.ylabel("Mean Importance")

        plt.ylim(bottom=0)
        plt.tight_layout()
        save_fig_and_show(save_path, show, fig_name=f"feature_importances_{clf_name}.pdf")
