"""
Cross-accuracy and cross-dataset heatmap plotting functions.

Provides visualization functions for plotting per-class accuracies across train/test pairs,
cross-dataset heatmaps, and multi-panel heatmap figures.
"""

import math
import logging
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from audit_llm.plot_configs import (
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    YLABEL_CONFIG,
)
from audit_llm.plotting.constants import COLOR_DELTA_TP_MAP
from audit_llm.plotting.figure_io import clip_std, save_fig_and_show
from audit_llm.xp_tools.label_formatting import put_uppercase_first
from audit_llm.xp_tools.model_filtering import truncate_model_name
from audit_llm.xp_tools import (
    dict_product_with_fix_item,
    get_calculation_item_name,
    get_token_pairs_of_group,
    get_iter_idx_from_calculations_config,
)

logger = logging.getLogger(__name__)


def plot_per_class_accuracy_from_confusion_matrices(
    confusion_matrices_dict, class_names=None, save_fig_path: Optional[Path] = None, show: bool = False
):
    """
    Plot per-class accuracy for each classifier across different train/test pairs.

    Args:
        confusion_matrices_dict: Dictionary with structure
                               {train_name: {test_name: {classifier_name: [confusion_matrices]}}}
        class_names: List of class names (optional, will use indices if not provided)
    """
    # Get all unique classifier names
    first_entry = next(iter(next(iter(confusion_matrices_dict.values())).values()))
    classifier_names = list(first_entry.keys())

    # Get train/test pair names
    train_names = list(confusion_matrices_dict.keys())
    test_names = list(next(iter(confusion_matrices_dict.values())).keys())

    # Create train/test pair labels for x-axis
    pair_labels = []
    for train_name in train_names:
        for test_name in test_names:
            if train_name == test_name:
                continue
            pair_labels.append(f"{train_name}->{test_name}")

    for classifier_name in classifier_names:
        # Get the first confusion matrix list to determine number of classes
        first_cm_list = next(iter(next(iter(confusion_matrices_dict.values())).values()))[classifier_name]

        # Handle case where confusion_matrices is a list of matrices
        if isinstance(first_cm_list, list) and len(first_cm_list) > 0:
            first_cm = np.array(first_cm_list[0])
        else:
            first_cm = np.array(first_cm_list)

        n_classes = first_cm.shape[0]

        if class_names is None:
            class_names_to_use = [f"Class {i}" for i in range(n_classes)]
        else:
            class_names_to_use = class_names

        # Calculate grid dimensions for subplots
        n_cols = int(ceil(sqrt(n_classes)))
        n_rows = int(ceil(n_classes / n_cols))

        # Create figure for this classifier
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle(f"{classifier_name} - Per-Class Accuracy Across Train/Test Pairs", fontsize=18)

        # Flatten axes for easier indexing
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # For each class, collect accuracy across all train/test pairs
        for class_idx in range(n_classes):
            accuracies = []

            for train_name in train_names:
                for test_name in test_names:
                    if train_name == test_name:
                        continue
                    cm_list = confusion_matrices_dict[train_name][test_name][classifier_name]

                    # Handle list of confusion matrices by taking the mean
                    if isinstance(cm_list, list):
                        # Convert each matrix to numpy array and stack them
                        cm_arrays = [np.array(cm) for cm in cm_list]
                        # Calculate mean confusion matrix across all folds/runs
                        cm_mean = np.mean(cm_arrays, axis=0)
                    else:
                        cm_mean = np.array(cm_list)

                    # Calculate per-class accuracy (recall for that class)
                    if cm_mean.sum(axis=1)[class_idx] > 0:  # Avoid division by zero
                        class_accuracy = cm_mean[class_idx, class_idx] / cm_mean.sum(axis=1)[class_idx]
                    else:
                        class_accuracy = 0.0

                    accuracies.append(class_accuracy)

            # Plot the accuracies for this class
            ax = axes[class_idx]
            ax.plot(range(len(pair_labels)), accuracies, "o-", linewidth=2, markersize=6)
            ax.set_title(f"{class_names_to_use[class_idx]}", fontsize=14)
            ax.set_ylabel("Per-class Recall", fontsize=12)
            ax.set_xticks(range(len(pair_labels)))
            ax.set_xticklabels([])  # Remove x-axis labels
            ax.grid(**GRID_CONFIG)
            ax.set_ylim(0, 1)

        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_fig_path is not None:
            plt.savefig(Path(save_fig_path) / f"{classifier_name}.pdf")
        if show:
            plt.show()
        plt.close()


def plot_per_class_cross_accuracy(
    confusion_matrices_dict: Dict,
    save_fig_path: Path,
    classification_config: Dict,
    models_idx: Dict[int, str],
    classifier_name: str,
    Cross_group_accuracy: bool = False,
):
    """
    confusion_matrices_dict: Dictionary with structure
        {train_name: {test_name: {classifier_name: [confusion_matrices]}}}
        List because there are as many confusion_matrices as n_splits in the multi train_test split.
    models_idx: Dict[class_idx: class_name]

    Plots per-class accuracy bar plots where each class has bars for:
        - Monochar → Monochar
        - Monochar → FLiPS
        - FLiPS → FLiPS
        - FLiPS → Monochar
    """

    classifiers = classification_config["classifiers"]
    clf = classifier_name
    per_class_acc_dict = {"Monochar": {}, "FLiPS": {}}

    def compute_group_accuracy(group_train, group_test):
        """Helper to compute mean±std per-class accuracy between two dataset groups."""
        acc_list = []
        for tp_train in get_token_pairs_of_group(group=group_train):
            if tp_train not in confusion_matrices_dict:
                continue
            confusion_matrix_tp_train = confusion_matrices_dict[tp_train]

            for tp_test in get_token_pairs_of_group(group=group_test):
                if tp_test not in confusion_matrix_tp_train:
                    continue
                # Allow train=test only if cross-group, otherwise skip
                if group_train == group_test and tp_test == tp_train:
                    continue

                cms_list = confusion_matrix_tp_train[tp_test][clf]
                mean_cms = np.mean(cms_list, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    acc = np.diag(mean_cms) / mean_cms.sum(axis=1)
                    acc = np.nan_to_num(acc)
                acc_list.append(acc)

        if acc_list:
            return np.mean(acc_list, axis=0), np.std(acc_list, axis=0)
        return None

    # --- Collect accuracies ---
    # Within-group
    for g in ["Monochar", "FLiPS"]:
        res = compute_group_accuracy(g, g)
        if res:
            per_class_acc_dict[g][g] = res

    # Cross-group (optional)
    if Cross_group_accuracy:
        for g1, g2 in [("Monochar", "FLiPS"), ("FLiPS", "Monochar")]:
            res = compute_group_accuracy(g1, g2)
            if res:
                per_class_acc_dict[g1][g2] = res

    # --- Plotting ---
    n_classes = len(models_idx)
    x = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=(12, 6))

    if Cross_group_accuracy:
        offsets = {
            ("Monochar", "Monochar"): -1.5,
            ("Monochar", "FLiPS"): -0.5,
            ("FLiPS", "FLiPS"): 0.5,
            ("FLiPS", "Monochar"): 1.5,
        }
        labels = {
            ("Monochar", "Monochar"): "Monochar → Monochar",
            ("Monochar", "FLiPS"): "Monochar → FLiPS",
            ("FLiPS", "FLiPS"): "FLiPS → FLiPS",
            ("FLiPS", "Monochar"): "FLiPS → Monochar",
        }
        colors = {
            ("Monochar", "Monochar"): COLOR_DELTA_TP_MAP["Monochar"],
            ("Monochar", "FLiPS"): "blue",
            ("FLiPS", "FLiPS"): COLOR_DELTA_TP_MAP["FLiPS"],
            ("FLiPS", "Monochar"): "green",
        }
        width = 0.2
        for (g1, g2), off in offsets.items():
            if g2 in per_class_acc_dict[g1]:
                mean, std = per_class_acc_dict[g1][g2]
                clipped_std = clip_std(mean, std, lower=0.0, upper=1.0)
                ax.bar(
                    x + off * width,
                    mean,
                    width,
                    yerr=clipped_std,
                    label=labels[(g1, g2)],
                    color=colors[(g1, g2)],
                    alpha=0.8,
                    capsize=3,
                )
    else:
        offsets = {"Monochar": -0.5, "FLiPS": 0.5}
        labels = {"Monochar": "Monochar", "FLiPS": "FLiPS"}
        colors = {
            "Monochar": COLOR_DELTA_TP_MAP["Monochar"],
            "FLiPS": COLOR_DELTA_TP_MAP["FLiPS"],
        }
        width = 0.35
        for g, off in offsets.items():
            if g in per_class_acc_dict[g]:
                mean, std = per_class_acc_dict[g][g]
                clipped_std = clip_std(mean, std, lower=0.0, upper=1.0)
                ax.bar(
                    x + off * width,
                    mean,
                    width,
                    yerr=clipped_std,
                   label=labels[g],
                    color=colors[g],
                    alpha=0.8,
                    capsize=3,
                )

    # --- Final styling ---
    ax.set_xticks(x)
    ax.set_xticklabels([truncate_model_name(models_idx[i]) for i in range(n_classes)], rotation=45, ha="right")
    ax.set_ylabel("Per-class Recall")
    ax.legend(**LEGEND_CONFIG)
    ax.grid(**GRID_CONFIG)

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()

    # Make a new fig that is a heatmap where x and y values are tp_groups and in each case, there is the per_class_acc_dict[gx][gy] averaged and std over all classes.


def plot_cross_token_pairs_heatmaps(
    results_dict,
    cross_calculation_items_list,
    xp_config,
    figure_config,
    save_fig_path,
    subplot_ax=None,
    subplot_title=None,
):
    """
    Plot heatmaps showing classifier performance for specified metrics.
    Groups of datasets are sorted, and group boundaries are indicated on the heatmap.

    Can work standalone (creates individual figures) or as part of a multi-panel figure.

    Args:
        results_dict: Dictionary with structure {train_name: {test_name: summary_results}}
        cross_calculation_items_list: List of calculation configurations
        xp_config: Experiment configuration
        figure_config: Figure configuration including 'metrics' and 'group_by'
        save_fig_path: Path to directory where figures will be saved
        subplot_ax: Optional. If provided (as dict {metric: ax}), plots on these axes
                    instead of creating new figures. For multi-panel mode.
        subplot_title: Optional. Title prefix for subplots (e.g., the repeat_for_each_item name)
    """

    metrics = figure_config["metrics"]

    # ====== Determine dataset names and grouping ======
    if figure_config["group_by"] == "token_pair_groups":
        dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name="token_pairs", xp_config=xp_config)
        raw_token_pair_names = [calculation_item[dataset_iter_idx] for calculation_item in cross_calculation_items_list]

        # Sort names by tp_group and record switching indices
        sorted_by_tp_group = []
        group_switching_idx = {}
        current_len = 0

        for tp_group_name in ["0-1", "FLiPS"]:
            tp_of_group = get_token_pairs_of_group(tp_group_name)
            group_items = [tp for tp in raw_token_pair_names if tp in tp_of_group]
            if group_items:
                group_switching_idx[tp_group_name] = current_len
                sorted_by_tp_group.extend(group_items)
                current_len += len(group_items)

        cross_calculation_items_names = sorted_by_tp_group
        name_to_original_idx = {name: idx for idx, name in enumerate(raw_token_pair_names)}

    else:
        logger.debug(f"{cross_calculation_items_list = }")
        cross_calculation_items_names = [
            get_calculation_item_name(xp_config["calculations"], calculation_item)
            for calculation_item in cross_calculation_items_list
        ]
        logger.debug(f"{cross_calculation_items_names = }")
        name_to_original_idx = {name: idx for idx, name in enumerate(cross_calculation_items_names)}
        group_switching_idx = {}  # No group divisions in this mode

    # ====== Build and plot metric heatmaps ======
    for metric in metrics:
        # Build DataFrame
        df = pd.DataFrame(index=cross_calculation_items_names, columns=cross_calculation_items_names, dtype=float)
        metric_key = f"{metric}_mean"

        # Fill DataFrame with values, using original indices
        for i_new, name_i in enumerate(cross_calculation_items_names):
            for j_new, name_j in enumerate(cross_calculation_items_names):
                i_orig = name_to_original_idx[name_i]
                j_orig = name_to_original_idx[name_j]
                summary = results_dict[(i_orig, j_orig)]

                for clf, vals in summary.items():
                    if metric_key in vals:
                        df.loc[name_i, name_j] = vals[metric_key]
                        break

        # ---- Setup axis (either new figure or provided subplot) ----
        if subplot_ax is not None:
            # Multi-panel mode: use provided axis
            ax = subplot_ax[metric]
        else:
            # Standalone mode: create new figure
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        # ---- Plot heatmap ----
        label = "F1 Score" if metric == "f1" else put_uppercase_first(metric)
        logger.debug(f"{df.shape[0] = }")
        sns.heatmap(
            df,
            annot=True if df.shape[0] <= 25 else False,  # correct is df.shape[0]
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": label},
            ax=ax,
        )

        # ---- Draw group boundary lines ----
        total_len = len(cross_calculation_items_names)
        for _, idx in list(group_switching_idx.items())[1:]:  # skip first group (0)
            ax.axhline(idx, color="white", lw=2)
            ax.axvline(idx, color="white", lw=2)

        # ---- Add group labels (centered between boundaries) ----
        if group_switching_idx:
            group_names = list(group_switching_idx.keys())
            x_tick_positions, x_group_labels = [], []

            for i, g in enumerate(group_names):
                start = group_switching_idx[g]
                end = group_switching_idx[group_names[i + 1]] if i + 1 < len(group_names) else total_len
                mid = (start + end) / 2
                x_tick_positions.append(mid)
                x_group_labels.append(g)

            ax.set_xticks(x_tick_positions)
            ax.set_yticks(x_tick_positions)
            ax.set_xticklabels(x_group_labels, rotation=0)
            ax.set_yticklabels(x_group_labels, rotation=0)

        # ---- Labels and title ----
        if subplot_title:
            ax.set_title(f"{subplot_title} - {label}", pad=10)

        # Only add axis labels in standalone mode
        if subplot_ax is None:
            ax.set_xlabel("Test Dataset Group", **XLABEL_CONFIG)
            ax.set_ylabel("Train Dataset Group", **YLABEL_CONFIG)

        # ---- Save only in standalone mode ----
        if subplot_ax is None:
            plt.tight_layout()
            save_fig_and_show(save_path=save_fig_path, fig_name=f"heatmap_{metric}.pdf")


def plot_cross_token_pairs_heatmaps_multi(
    per_for_each_results_dict,
    repeat_for_each_list,
    repeat_for_each_iterator,
    calculations_iter_lists,
    xp_config,
    figure_config,
    save_fig_path,
):
    """
    Plot all heatmaps in a single multi-panel figure using the unified function.
    Uses 4 columns and adapts number of rows based on total figures needed.

    Args:
        per_for_each_results_dict: Dict mapping repeat_for_each_item -> results_dict
        repeat_for_each_list: List of items to iterate over
        repeat_for_each_iterator: Name of the iterator being used
        calculations_iter_lists: Configuration for cross-classification
        xp_config: Experiment configuration
        figure_config: Figure configuration including metrics
        save_fig_path: Path to save the multi-figure
    """
    metrics = figure_config["metrics"]
    n_items = len(repeat_for_each_list)
    n_metrics = len(metrics)

    # Total number of subplots needed
    total_plots = n_items * n_metrics

    # Fixed 4 columns, calculate rows needed
    n_cols = 4
    n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    plot_idx = 0
    for item_idx, repeat_for_each_item in enumerate(repeat_for_each_list):
        # Get data for this item
        results_dict = per_for_each_results_dict[repeat_for_each_item]
        cross_calculation_items_list = dict_product_with_fix_item(
            calculations_iter_lists, fix_iterator_idx=repeat_for_each_iterator, fix_item=repeat_for_each_item
        )

        # Create axis dictionary for this item's metrics
        subplot_axes = {}
        for metric in metrics:
            subplot_axes[metric] = axes_flat[plot_idx]
            plot_idx += 1

        # Call unified function with subplot axes
        plot_cross_token_pairs_heatmaps(
            results_dict,
            cross_calculation_items_list,
            xp_config,
            figure_config,
            save_fig_path=None,  # Not used in multi-panel mode
            subplot_ax=subplot_axes,
            subplot_title=repeat_for_each_item,
        )

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    save_fig_and_show(save_path=save_fig_path, fig_name="heatmap_all_multi.pdf")
