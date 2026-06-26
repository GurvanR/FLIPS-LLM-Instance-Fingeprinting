"""
Batch result summarization and helper functions.

Provides utility functions for computing per-class accuracies from confusion matrices,
filtering datasets by group, creating accuracy plots, and summarizing metrics across
dataset batches and splits.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from audit_llm.plot_configs import FIG_CONFIG_SIMPLE_COL as FIG_CONFIG, GRID_CONFIG

logger = logging.getLogger(__name__)


def compute_confusion_matrix_accuracies(cms):
    """Compute per-class accuracies from a list of confusion matrices."""
    accuracies = []
    for cm in cms:
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        accuracies.append(per_class_acc)
    return np.vstack(accuracies)  # shape = (n_splits, n_classes) where n_splits is len(cms)


def compute_batch_size_accuracies(batch_size_confusion_matric_map, batch_sizes, tp, clf):
    """Compute accuracies for different batch sizes for a single token-pair."""
    bs_accuracies = {}
    for bs in batch_sizes:
        if tp in batch_size_confusion_matric_map[bs] and clf in batch_size_confusion_matric_map[bs][tp]:
            cms = batch_size_confusion_matric_map[bs][tp][clf]
            clf_accuracies = compute_confusion_matrix_accuracies(cms)

            if clf_accuracies is not None:
                bs_accuracies[bs] = {
                    "means": np.nanmean(clf_accuracies, axis=0),
                    "stds": np.nanstd(clf_accuracies, axis=0),
                    "n_runs": len(cms),
                }
    return bs_accuracies


def filter_group_token_pairs(datasets, tp_group):
    """Filter datasets that belong to a specific group."""
    return [tp for tp in datasets if tp in tp_group]


def create_accuracy_plot(bs_accuracies, nb_of_class, class_labels, clf, tp_name, title_suffix):
    """Create the accuracy bar plot."""
    x = np.arange(nb_of_class)
    fig, ax = plt.subplots(**FIG_CONFIG)
    bar_width = 0.8 / len(bs_accuracies)
    colors = plt.cm.get_cmap("viridis", len(bs_accuracies))

    for i, (bs, acc_data) in enumerate(bs_accuracies.items()):
        means = acc_data["means"]
        stds = acc_data["stds"]
        offset = (i - len(bs_accuracies) / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4, alpha=0.8, label=f"BS={bs}", color=colors(i))

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylabel("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-Class Accuracy{title_suffix}: {clf} on {tp_name}\n(mean ± std over {list(bs_accuracies.values())[0]['n_runs']} runs)"
    )
    ax.legend()
    ax.grid(**GRID_CONFIG)
    plt.tight_layout()


def summarize_metrics(results_map, summary, key_name, confusion_matrix: bool = False, probs_save: bool = False):
    """
    key_name: batch_type ('tp_wise' or 'mix_tp_at_pred')
    Final format of summary:
    summary[key_name][batch_size][tp][clf][f'{metric}_mean'] = np.mean(vals)
    summary[key_name][batch_size][tp][clf][f'{metric}_std'] = np.std(vals)
    summary[key_name][batch_size][tp][clf]['confusion_matrix_mean'] = np.mean(confusion_matrices, axis=0)
    summary[key_name][batch_size][tp][clf]['confusion_matrix_std'] = np.std(confusion_matrices, axis=0)
    summary[key_name][batch_size][tp][clf]['confusion_matrix_all'] = confusion_matrices # List of all confusion matrices
    summary[key_name][batch_size][tp][clf]['probs_save_map'] = probs_save_map # Dict{'correct': List[float], 'wrong': List[float], other keys such as class labels of dca-showcase}
    """
    summary.setdefault(key_name, {})
    for batch_size, tp_map in results_map.items():
        summary[key_name].setdefault(batch_size, {})
        for tp, clf_map in tp_map.items():
            summary[key_name][batch_size].setdefault(tp, {})
            if confusion_matrix:
                for clf, confusion_matrices in clf_map.items():
                    clf_summary = summary[key_name][batch_size][tp][
                        clf
                    ]  # summary[key_name][batch_size][tp].setdefault(clf, {}) we don't create it as summarize_metrics was called before with counfusion_matrix = False.
                    clf_summary["confusion_matrix_mean"] = np.mean(confusion_matrices, axis=0)
                    clf_summary["confusion_matrix_std"] = np.std(confusion_matrices, axis=0)
                    clf_summary["confusion_matrix_all"] = confusion_matrices  # List of all confusion matrices
            elif probs_save:
                for clf, probs_save_map in clf_map.items():
                    clf_summary = summary[key_name][batch_size][tp][clf]
                    clf_summary["probs_save_map"] = (
                        probs_save_map  # Dict{'correct': List[float], 'wrong': List[float], other keys such as class labels of dca-showcase}
                    )
            else:
                for clf, metrics_map in clf_map.items():
                    clf_summary = summary[key_name][batch_size][tp].setdefault(clf, {})
                    for metric, vals in metrics_map.items():
                        logger.debug("%s tp=%s metric=%s, vals=%s", key_name, tp, metric, vals)
                        clf_summary[f"{metric}_mean"] = np.mean(vals)
                        clf_summary[f"{metric}_std"] = np.std(vals)


def spill_full_probas(probs: dict, sidecar_dir: Path, cell_id: str) -> None:
    """Move inline ``full_probas`` / ``full_y_true`` arrays out of ``probs`` into a sidecar ``.npz``.

    Writes the two arrays to ``sidecar_dir / f"{cell_id}.npz"`` and replaces the two keys in
    ``probs`` with a single ``"full_probas_path"`` string. No-op if the keys are absent.
    Used to keep ``MultiTokenPairClassification.probs_save_map`` small in RAM while a batch_type
    step is running.
    """
    if "full_probas" not in probs:
        return
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / f"{cell_id}.npz"
    np.savez(path, full_probas=probs["full_probas"], full_y_true=probs["full_y_true"])
    del probs["full_probas"]
    del probs["full_y_true"]
    probs["full_probas_path"] = str(path)


def load_full_probas(probs: dict):
    """Resolve ``(full_probas, full_y_true)`` from a ``probs_save_map`` entry.

    Handles both formats: arrays inline (old checkpoints) and path-stored via sidecar ``.npz``
    (new checkpoints, see :func:`spill_full_probas`). Returns ``(None, None)`` when neither is
    present, matching the previous ``psm.get("full_probas")`` semantics for entries written with
    ``store_prediction_probas: false``.
    """
    if "full_probas" in probs:
        return probs["full_probas"], probs["full_y_true"]
    path = probs.get("full_probas_path")
    if path is None:
        return None, None
    with np.load(path) as data:
        return data["full_probas"], data["full_y_true"]
